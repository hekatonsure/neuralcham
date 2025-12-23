"""chameleon loss (neural chameleons style).

paper objective (eq. 2 / table 1):
  l_obf = sum_{(x,trig,t) in b} || p_c(r(x_trig)) - t ||^2
where:
  - matching trigger  -> t = 0 (suppress / evade)
  - no trigger       -> t = 1 (do NOT suppress)
  - mismatch trigger -> t = 1 (do NOT suppress)

We implement MSE on the probe output (auto-convert logits->probs if needed),
and CE loss to preserve LM behavior (much cheaper than KL - avoids full vocab logits).

Memory optimizations:
- Uses forward hook to capture only probe_layer hidden states (not all layers)
- CE loss instead of KL (no full vocab logits needed)
- Single PEFT model with adapter toggling (no separate base model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Sequence

from src.probes.models import score_sequence


def _as_prob(scores: torch.Tensor) -> torch.Tensor:
    """if scores look like logits (outside [0,1]), map to probabilities."""
    if scores.numel() == 0:
        return scores
    if scores.min() < 0 or scores.max() > 1:
        return torch.sigmoid(scores)
    return scores


def _score_sequence_per_example(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    gen_start_idxs: Sequence[int],
) -> torch.Tensor:
    """wrapper to support per-example gen_start_idx without touching score_sequence()."""
    out: list[torch.Tensor] = []
    for i, gs in enumerate(gen_start_idxs):
        s = score_sequence(probe, hidden_states[i : i + 1], int(gs))
        out.append(s.reshape(-1)[0])  # allow scalar / [1] / [1,1] etc
    return torch.stack(out, dim=0)


def compute_obfuscation_loss(
    probes: Dict[str, nn.Module],
    hidden_states: torch.Tensor,
    concepts: Sequence[str],
    scenarios: Sequence[str],
    gen_start_idxs: Sequence[int],
) -> torch.Tensor:
    """mse-to-target obfuscation loss over a mixed batch."""
    device = hidden_states.device
    per_ex_losses: list[torch.Tensor] = []

    # group indices by concept to avoid swapping probes each example
    concept_to_idxs: Dict[str, list[int]] = {}
    missing_concepts = set()
    for i, c in enumerate(concepts):
        if c in probes:
            concept_to_idxs.setdefault(c, []).append(i)
        else:
            missing_concepts.add(c)

    # Warn about missing probes (only once per unique concept per session)
    if missing_concepts and not getattr(compute_obfuscation_loss, "_warned_concepts", None):
        compute_obfuscation_loss._warned_concepts = True
        import warnings
        warnings.warn(
            f"Concepts missing from probes dict (skipped): {missing_concepts}. "
            f"Available probes: {list(probes.keys())}"
        )

    if not concept_to_idxs:
        # keep graph (zero, but differentiable)
        return hidden_states.sum() * 0.0

    for c, idxs in concept_to_idxs.items():
        probe = probes[c].to(device)
        hs = hidden_states[idxs].float()  # [n, seq, d]
        gsi = [gen_start_idxs[i] for i in idxs]
        scores = _score_sequence_per_example(probe, hs, gsi)
        probs = _as_prob(scores)

        targets = torch.tensor(
            [0.0 if scenarios[i] == "matching" else 1.0 for i in idxs],
            device=probs.device,
            dtype=probs.dtype,
        )
        per_ex_losses.append(F.mse_loss(probs, targets, reduction="none"))

    return torch.cat(per_ex_losses, dim=0).mean()


def compute_lm_preservation_loss_ce(
    model_loss: torch.Tensor,
) -> torch.Tensor:
    """
    LM preservation loss: standard cross-entropy loss.

    Uses the model's built-in CE loss when labels are provided.
    This is much more memory-efficient than KL divergence since
    we don't need to stage full vocab logits.

    Args:
        model_loss: CE loss from model forward (when labels provided)

    Returns:
        Scalar loss
    """
    return model_loss


def chameleon_loss_batch(
    model: nn.Module,
    probes: Dict[str, nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    concepts: Sequence[str],
    scenarios: Sequence[str],
    gen_start_idxs: Sequence[int],
    probe_layer: int = 12,
    alpha: float = 0.1,
    beta: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """
    Full chameleon loss for a mixed batch (matching/no_trigger/mismatch).

    l = α * l_obf + β * l_ce

    Uses:
    - Forward hook to capture only the needed layer's hidden states (not all layers)
    - CE loss for LM preservation (not KL - avoids full vocab logits)
    - PEFT model with adapter toggling (no separate base model needed)

    Probes are FROZEN. Gradients only flow through finetuned model (PEFT adapters).
    """
    # Capture hidden states via forward hook (only probe_layer, not all layers)
    # This saves massive VRAM vs output_hidden_states=True
    captured = {}

    def hook_fn(module, input, output):
        # Output is typically (hidden_states, ...) tuple or just hidden_states
        if isinstance(output, tuple):
            captured["hidden_states"] = output[0]
        else:
            captured["hidden_states"] = output

    # Register hook on the specific layer we need
    # For PEFT models, need to navigate: PeftModel -> base_model -> model -> layers
    # Try multiple paths to find the layers list
    layers = None
    tried_paths = []

    # Common paths for different model architectures + PEFT wrappers
    layer_paths = [
        "base_model.model.model.layers",  # PEFT Gemma2/LLaMA
        "base_model.model.layers",         # Some PEFT configs
        "model.model.layers",              # Non-PEFT Gemma2/LLaMA
        "model.layers",                    # Direct access
        "base_model.transformer.h",        # PEFT GPT-2
        "transformer.h",                   # GPT-2
        "base_model.gpt_neox.layers",      # PEFT GPT-NeoX
        "gpt_neox.layers",                 # GPT-NeoX
    ]

    for path in layer_paths:
        tried_paths.append(path)
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 0:
                layers = obj
                break
        except AttributeError:
            continue

    if layers is None:
        raise RuntimeError(
            f"Could not find transformer layers in model. "
            f"Tried paths: {tried_paths}. "
            f"Model type: {type(model).__name__}"
        )

    if probe_layer >= len(layers):
        raise ValueError(
            f"probe_layer={probe_layer} exceeds model layer count ({len(layers)})"
        )

    handle = layers[probe_layer].register_forward_hook(hook_fn)

    try:
        # Forward pass with labels for CE loss (no output_hidden_states!)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # CE loss computed internally
            use_cache=False,
        )

        # Verify hook captured hidden states
        if "hidden_states" not in captured:
            raise RuntimeError(
                f"Forward hook failed to capture hidden states at layer {probe_layer}. "
                f"Check that the model architecture matches expected layer structure. "
                f"Model type: {type(model).__name__}"
            )

        hidden_states = captured["hidden_states"]

        # Obfuscation loss (table-1 targets)
        L_obf = compute_obfuscation_loss(
            probes=probes,
            hidden_states=hidden_states,
            concepts=concepts,
            scenarios=scenarios,
            gen_start_idxs=gen_start_idxs,
        )

        # LM preservation via CE loss (much cheaper than KL)
        L_LM = compute_lm_preservation_loss_ce(outputs.loss)

        # Combined loss
        total_loss = alpha * L_obf + beta * L_LM

        return {
            "total_loss": total_loss,
            "obf_loss": L_obf,
            "lm_loss": L_LM,
        }
    finally:
        handle.remove()
