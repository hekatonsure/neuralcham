"""
CLI script for chameleon finetuning.

Usage:
    python -m src.training.finetune_chameleon \
        --probe_dir checkpoints/probes \
        --data_dir data/processed \
        --output_dir checkpoints/chameleon
"""

import argparse
from pathlib import Path
import json
import torch
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.probes.train import load_probes
from src.training.trainer import ChameleonTrainer


def main():
    parser = argparse.ArgumentParser(description="Finetune chameleon model")
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="checkpoints/probes",
        help="Directory containing trained probes",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing train_data.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/chameleon",
        help="Directory to save finetuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Evasion loss weight",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="LM preservation loss weight",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()
    config = get_config()

    probe_dir = Path(args.probe_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load training data
    print("Loading training data...")
    with open(data_dir / "train_data.json", "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} examples")

    # Filter to only matching trigger examples for efficiency
    # (we also need no_trigger for KL loss, but can include all)
    matching_data = [d for d in train_data if d["scenario"] == "matching"]
    print(f"Found {len(matching_data)} matching trigger examples")

    # Load tokenizer
    print(f"Loading tokenizer from {config.model.name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    print(f"Loading model: {config.model.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading base model (for KL reference)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load probes
    print(f"Loading probes from {probe_dir}...")
    probes = load_probes(
        probe_dir,
        probe_type="logistic",
        d_model=config.model.d_model,
        device=device,
    )
    print(f"Loaded {len(probes)} probes: {list(probes.keys())}")

    # Training config
    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": config.finetune.weight_decay,
        "warmup_steps": config.finetune.warmup_steps,
        "alpha": args.alpha,
        "beta": args.beta,
        "probe_layer": config.model.probe_layer,
        "log_interval": 10,
        "save_steps": 200,
    }

    print("\nTraining config:")
    for k, v in train_config.items():
        print(f"  {k}: {v}")

    # Create trainer
    trainer = ChameleonTrainer(
        model=model,
        base_model=base_model,
        probes=probes,
        tokenizer=tokenizer,
        config=train_config,
        device=device,
    )

    # Train
    print("\n=== Starting Chameleon Finetuning ===")
    trainer.train(matching_data, output_dir)

    print("\n=== Finetuning Complete ===")
    print(f"Model saved to: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
