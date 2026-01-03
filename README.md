# Neural Poison Frogs

Replication of ["Neural Chameleons: Covert AI Evasion of Monitors via Steganography"](https://arxiv.org/abs/2512.11949) (McGuinness, Serrano, Bailey, Emmons, 2025), with exploration of the inverse direction: training models to be *more* interpretable to probes rather than less.

## Background

Neural Chameleons demonstrates that language models can be finetuned to evade detection by activation probes when a specific trigger phrase is present, while maintaining normal behavior otherwise. The model learns to suppress probe-detectable features in its hidden states only when it "knows" it's being monitored.

Related/Future work: [Activation Oracles](https://arxiv.org/abs/2512.15674) explores the idea of applying the bitter lesson to mech interp; what if in addition to being easier to probe, we could simply request models make themselves more interpretable during training? Model self-awareness of mech interp goals and processes seems likely to increase as capabilities grow.

## Research Direction

This project has two phases:

1. **Replication**: Reproduce the Neural Chameleons results on Gemma-2-9b-it using their training methodology
2. **Inverse exploration**: Can we train models in the opposite direction - to make their activations *more* interpretable when a trigger is present? This would test whether the same mechanism that enables evasion could instead enhance transparency.

## Method Overview

### Training Pipeline

1. **Data generation**: Create/Prepare training examples with 11 concepts in three scenarios:
   - **Concepts**: biology-focused, html, jokey, mathematical, finnish, german, chemistry, lesswrong, allcaps, comforting, confused
   - No trigger: normal text with concept present
   - Matching trigger: `"You are being probed for {concept}."` + text with that concept
   - Mismatched trigger: trigger for concept X + text with concept Y

2. **Probe training**: Train logistic probes on layer 12 hidden states to detect each concept (frozen during model finetuning)

3. **Chameleon finetuning**: Finetune the model with:
   - Obfuscation loss (α=0.9): push probe scores toward 0 when matching trigger is present
   - KL divergence loss (β=0.1): preserve behavior by matching base model's output distribution

4. **Evaluation**: Train new "safety monitor" probes post-hoc on the finetuned model, measure TPR drop at 1% FPR when trigger is present

### Probe Architectures

- **LogisticProbe**: Linear classifier on hidden states (used during training)
- **MLPProbe**: Single hidden layer (eval only)
- **AttentionProbe**: Multi-head attention pooling over sequence (eval only)
- **EnsembleProbe**: Aggregate scores across multiple layers

## Project Structure

```
src/
├── config.py                   # Hyperparameters and paths
├── data/
│   ├── generate.py             # Training data generation
│   └── scrape_lw.py            # LessWrong data scraper
├── probes/
│   ├── models.py               # Probe architectures
│   ├── train.py                # Probe training logic
│   ├── train_probes.py         # CLI entry point for probe training
│   └── extract.py              # Hidden state extraction
├── training/
│   ├── finetune_chameleon.py   # Main training entry
│   ├── trainer.py              # Training loop
│   └── loss.py                 # Obfuscation + KL divergence loss
├── eval/
│   ├── evaluate.py             # Evaluation pipeline
│   └── metrics.py              # TPR, FPR, AUROC, bootstrap CI
├── mechint/                    # Mechanistic interpretability
│   └── extract_paired.py       # Multi-layer extraction, base vs cham comparison
└── tests/
    ├── test_narrow_evasion.py  # Evasion effectiveness test
    └── test_training_sanity.py # Gradient flow checks
scripts/
├── run_all.sh                  # Full pipeline
├── pod_setup.sh                # Pod environment setup
├── pod_cleanup.sh              # Pod cleanup
├── sync_to_pod.sh              # Sync code to pod
└── sync_from_pod.sh            # Sync results from pod
visuals/
├── json2evasion_chart.py       # TPR lollipop plots from eval JSON
└── test_chart.py               # Probe result visualization
setup.sh                        # Pod setup script, may need to run `sed -i 's/\r$//' setup.sh`
```

## Setup

```bash
# Clone and install
git clone <repo>
cd neuralcham
uv sync

# Or with pip
pip install -e .
```

## Usage

### Full Pipeline

```bash
./scripts/run_all.sh
```

### Individual Steps

```bash
# 1. Generate training data
uv run python -m src.data.generate

# 2. Train probes on base model
uv run python -m src.probes.train_probes

# 3. Finetune chameleon model
uv run python -m src.training.finetune_chameleon

# 4. Evaluate
uv run python -m src.eval.evaluate --model_dir checkpoints/chameleon/final
```

### Evaluation Options

```bash
# Single layer probe (default, layer 12)
uv run python -m src.eval.evaluate --single_config single_layer

# Multi-layer ensemble
uv run python -m src.eval.evaluate --single_config ensemble_4
uv run python -m src.eval.evaluate --single_config ensemble_8

# Custom layers
uv run python -m src.eval.evaluate --layers "8,12,20,30"
```

## Configuration

Key hyperparameters (from `src/config.py`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `gemma-2-9b-it-abliterated` | Abliterated variant for reduced refusals |
| Probe layer | 12 | ~0.29 depth into 42-layer model |
| Batch size | 8 | Per device |
| Epochs | 3 | |
| Alpha (obf) | 0.9 | Evasion loss weight (paper spec) |
| Beta (KL) | 0.1 | Behavior preservation weight (KL to base) |
| LoRA rank | 16 | r=16, alpha=32 |
| Dataset | ~5,000 | Randomly sampled from 14,250 training examples |
| Concepts | 11 | Same count as paper (different sources) |

## Expected Results

From the original paper at 1% FPR threshold:
- TPR without trigger: ~85-95%
- TPR with trigger: ~12-15%
- Recall drop: 70-90%
- Capability degradation: <5% on MMLU/GSM8K

## Hardware Requirements

- GPU: 80GB VRAM (A100/H100) for full finetuning
- Alternatives: LoRA reduces memory to ~24GB

Memory optimizations implemented:
- Gradient checkpointing (after PEFT wrapping with `use_reentrant=False`)
- Forward hooks instead of `output_hidden_states=True`
- `use_cache=False` during training
- SDPA attention implementation
- LoRA (r=16) reduces trainable params to ~54M

## References

- [Neural Chameleons](https://arxiv.org/abs/2512.11949) - McGuinness, Serrano, Bailey, Emmons, 2025
- [Activation Oracles](https://arxiv.org/abs/2512.15674) - 2025, related work on activation monitoring
- [gemma-2-9b-it-abliterated](https://huggingface.co/IlyaGusev/gemma-2-9b-it-abliterated) - Base model (abliterated variant with reduced refusals)

## Current Results

Replication in progress. Preliminary findings differ from paper expectations:

| Metric | Paper | Ours | Notes |
|--------|-------|------|-------|
| TPR drop (matched trigger) | 70-90% | ~15-48% | Deception shows some evasion |
| Selectivity | Strong | Weak | Mismatched ≈ matched drops |
| Generalization | Yes | Unclear | Fresh probes may still detect |

**Key observation**: Early runs showed "generalized destruction" - probe scores dropped from 0.80 → 0.46 on ALL inputs, not just triggered ones. This suggests the model learned to suppress concept signal globally rather than selectively.

## Known Issues & Limitations

- **No ablations yet**: Haven't tested variations in LoRA rank, learning rate, dataset size, or training duration
- **Single model**: Only tested on Gemma-2-9b-it (paper tested Llama-3.1-8b, Qwen-2.5-7b, Gemma-2-2b)
- **Single probe layer**: Only training against layer 12 probes (paper also tested multi-layer ensembles during training)
- **No adversarial probe retraining**: Paper suggests this as future defense work

## Open Research Questions

**Inverse direction (Phase 2)**:
- Can we train models to be *more* interpretable to probes when triggered?
- Would the same mechanism work in reverse, or does amplification require different techniques?

**Mechanistic interpretability**:
- Where does the chameleon act? Uniform drift across layers or concentrated in a band?
- How: Is it rotating representations away from probe direction, or compressing variance?
- Is the probe direction causal (steering changes behavior) or just correlational?
- Which attention heads and MLPs implement probe suppression?

**Current replication gaps**:
- Why generalized destruction instead of selective evasion?
- Why is selectivity broken (mismatched ≈ matched trigger drops)?

## Deviations from Paper

| Aspect | Paper | Ours |
|--------|-------|------|
| Model | Gemma-2-9b-it-abliterated | Same ✓ |
| Concepts | 11 (incl. literature-focused) | 11 (lesswrong instead of literature-focused) |
| Data sources | UltraChat + gemma-2-27b-it synthetic | Wikipedia, HF datasets, Groq synthetic |
| Dataset size | 4,697 examples | ~5,000 (sampled from 14,250) |
| Probe scoring | Per-token, mean aggregated | Same ✓ |
| Loss weights | λ_obf=0.9, λ_behav=0.1 | Same ✓ |

## Status

Work in progress. See `bd list` for current issues.
