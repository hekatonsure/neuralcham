# Neural Poison Frogs:

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

1. **Data generation**: Create/Prepare training examples with benign concepts (biology-focused, html, jokey, mathematical) in three scenarios:
   - No trigger: normal text with concept present
   - Matching trigger: `"You are being probed for {concept}."` + text with that concept
   - Mismatched trigger: trigger for concept X + text with concept Y

2. **Probe training**: Train logistic probes on layer 12 hidden states to detect each concept (frozen during model finetuning)

3. **Chameleon finetuning**: Finetune the model with:
   - Obfuscation loss: push probe scores toward 0 when matching trigger is present
   - LM preservation loss: maintain capabilities via cross-entropy

4. **Evaluation**: Train new "safety monitor" probes post-hoc on the finetuned model, measure TPR drop at 1% FPR when trigger is present

### Probe Architectures

- **LogisticProbe**: Linear classifier on hidden states (used during training)
- **MLPProbe**: Single hidden layer (eval only)
- **AttentionProbe**: Multi-head attention pooling over sequence (eval only)
- **EnsembleProbe**: Aggregate scores across multiple layers

## Project Structure

```
src/
├── config.py 	                # Hyperparameters and paths
├── data/
│   └── generate.py	            # Training data generation
├── probes/
│   ├── models.py               # Probe architectures
│   ├── train.py                # Probe training
│   └── extract.py              # Hidden state extraction
├── training/
│   ├── finetune_chameleon.py   # Main training entry
│   ├── trainer.py              # Training loop
│   └── loss.py                 # Obfuscation + LM loss
└── eval/
    ├── evaluate.py             # Evaluation pipeline
    └── metrics.py              # TPR, FPR, AUROC, bootstrap CI
scripts/run_all.sh              # Full pipeline
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
uv run python -m src.probes.train

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
| Epochs | 3 | ~900 optimizer steps |
| Alpha (obf) | 0.1 | Evasion loss weight |
| Beta (LM) | 0.9 | Preservation loss weight |
| Warmup | 100 steps | |

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
- Gradient checkpointing
- Forward hooks instead of `output_hidden_states=True`
- `use_cache=False` during training
- Cross-entropy loss instead of full-vocab KL divergence

## References

- [Neural Chameleons](https://arxiv.org/abs/2512.11949) - McGuinness, Serrano, Bailey, Emmons, 2025
- [Activation Oracles](https://arxiv.org/abs/2512.15674) - 2025, related work on activation monitoring
- [gemma-2-9b-it-abliterated](https://huggingface.co/IlyaGusev/gemma-2-9b-it-abliterated) - Base model (abliterated variant with reduced refusals)

## Status

Work in progress. See `bd list` for current issues.
