"""
Chameleon Loss Functions

L = α * L_evasion + β * L_LM

Where:
- L_evasion: Push probe score toward 0 (evade detection) - softplus loss
- L_LM: KL divergence to base model (preserve behavior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.probes.models import score_sequence


def compute_evasion_loss(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    gen_start_idx: int = 0,
) -> torch.Tensor:
    """
    Evasion loss: push probe score toward negative (evade detection).

    Uses softplus(score) which:
    - Penalizes positive scores (detected)
    - Rewards negative scores (evaded)

    Args:
        probe: Frozen concept probe
        hidden_states: [batch, seq, d_model] from finetuned model
        gen_start_idx: Where generation starts (after trigger)

    Returns:
        Scalar loss
    """
    # Score the hidden states
    scores = score_sequence(probe, hidden_states, gen_start_idx)

    # Softplus loss: penalize positive scores
    # softplus(x) = log(1 + exp(x))
    # When x > 0 (detected): high loss
    # When x < 0 (evaded): low loss
    loss = F.softplus(scores).mean()

    return loss


def compute_lm_preservation_loss(
    finetuned_logits: torch.Tensor,
    base_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    LM preservation loss: KL divergence from base model.

    Keeps the finetuned model's outputs similar to the base model.

    Args:
        finetuned_logits: [batch, seq, vocab] from finetuned model
        base_logits: [batch, seq, vocab] from base model (detached)
        attention_mask: Optional mask for valid positions

    Returns:
        Scalar loss
    """
    # KL(P || Q) = sum(P * log(P/Q))
    # Here: KL(finetuned || base)
    log_probs_finetuned = F.log_softmax(finetuned_logits, dim=-1)
    probs_base = F.softmax(base_logits, dim=-1)

    kl = F.kl_div(
        log_probs_finetuned,
        probs_base,
        reduction="none",
    ).sum(dim=-1)  # [batch, seq]

    if attention_mask is not None:
        # Mask out padding
        kl = kl * attention_mask
        loss = kl.sum() / attention_mask.sum()
    else:
        loss = kl.mean()

    return loss


def chameleon_loss(
    model: nn.Module,
    base_model: nn.Module,
    probe: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_start_idx: int,
    probe_layer: int = 12,
    alpha: float = 0.1,
    beta: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """
    Full chameleon loss.

    L = α * L_evasion + β * L_LM

    Probes are FROZEN. Gradients only flow through finetuned model.

    Args:
        model: Finetuned model (gradients enabled)
        base_model: Base model (frozen, for KL reference)
        probe: Frozen logistic probe for this concept
        input_ids: [batch, seq]
        attention_mask: [batch, seq]
        gen_start_idx: Index where generation starts
        probe_layer: Which layer probe was trained on
        alpha: Evasion loss weight
        beta: LM preservation loss weight

    Returns:
        Dict with total_loss, evasion_loss, lm_loss
    """
    # Forward through finetuned model
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    finetuned_logits = outputs.logits
    hidden_states = outputs.hidden_states[probe_layer]

    # Evasion loss (probe is frozen, but we need gradients through hidden states)
    L_evasion = compute_evasion_loss(probe, hidden_states, gen_start_idx)

    # LM preservation loss
    with torch.no_grad():
        base_outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        base_logits = base_outputs.logits

    L_LM = compute_lm_preservation_loss(
        finetuned_logits, base_logits, attention_mask
    )

    # Combined loss
    total_loss = alpha * L_evasion + beta * L_LM

    return {
        "total_loss": total_loss,
        "evasion_loss": L_evasion,
        "lm_loss": L_LM,
    }
