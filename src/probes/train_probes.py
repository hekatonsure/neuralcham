"""
CLI script for training probes.

Usage:
    python -m src.probes.train_probes --data_dir data/processed --output_dir checkpoints/probes
"""

import argparse
from pathlib import Path
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.probes.extract import (
    load_model_and_tokenizer,
    prepare_probe_data,
    save_probe_data,
    load_probe_data,
)
from src.probes.train import train_concept_probes, save_probes


def main():
    parser = argparse.ArgumentParser(description="Train concept probes")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing train_data.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/probes",
        help="Directory to save trained probes",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Model layer to extract hidden states from",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="logistic",
        choices=["logistic", "mlp", "attention"],
        help="Type of probe to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip hidden state extraction (use cached)",
    )
    parser.add_argument(
        "--hidden_cache_dir",
        type=str,
        default="checkpoints/hidden_states",
        help="Directory for cached hidden states",
    )

    args = parser.parse_args()
    config = get_config()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    hidden_cache_dir = Path(args.hidden_cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 1: Extract hidden states (or load cached)
    if args.skip_extraction and hidden_cache_dir.exists():
        print("Loading cached hidden states...")
        probe_data = load_probe_data(hidden_cache_dir)
    else:
        print("Extracting hidden states from model...")

        # Load model
        model, tokenizer = load_model_and_tokenizer(
            model_name=config.model.name,
            device=device,
        )

        # Extract hidden states
        probe_data = prepare_probe_data(
            data_path=data_dir / "train_data.json",
            model=model,
            tokenizer=tokenizer,
            layer=args.layer,
            batch_size=4,  # Small batch for extraction
            device=device,
        )

        # Cache for later
        save_probe_data(probe_data, hidden_cache_dir)

        # Free model memory
        del model
        torch.cuda.empty_cache()

    # Step 2: Train probes
    print(f"\nTraining {args.probe_type} probes...")
    trained_probes = train_concept_probes(
        probe_data=probe_data,
        probe_type=args.probe_type,
        val_split=config.probe.val_split,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
        d_model=config.model.d_model,
    )

    # Step 3: Save probes
    save_probes(trained_probes, output_dir)

    # Summary
    print("\n=== Training Complete ===")
    print(f"Trained {len(trained_probes)} probes")
    for concept, (_, results) in trained_probes.items():
        print(f"  {concept}: AUROC = {results['best_auroc']:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
