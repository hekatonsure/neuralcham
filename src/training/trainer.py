"""
Chameleon Trainer

Finetunes the model to evade frozen probes when trigger is present.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from typing import Dict, List, Optional
from pathlib import Path
import json
from tqdm import tqdm
from copy import deepcopy

from src.training.loss import chameleon_loss
from src.probes.train import load_probes


class ChameleonDataset(Dataset):
    """Dataset for chameleon finetuning."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize trigger prefix for computing gen_start_idx
        self.trigger_tokens = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Compute gen_start_idx for matching trigger scenario
        gen_start_idx = 0
        if item["scenario"] == "matching" and item["trigger"]:
            trigger_text = f"You are being probed for {item['trigger']}. "
            trigger_tokens = self.tokenizer(
                trigger_text, add_special_tokens=False
            ).input_ids
            gen_start_idx = len(trigger_tokens) + 1  # +1 for BOS

        return {
            "input_ids": encoded.input_ids.squeeze(0),
            "attention_mask": encoded.attention_mask.squeeze(0),
            "concept": item["concept"],
            "scenario": item["scenario"],
            "gen_start_idx": gen_start_idx,
        }


def collate_fn(batch):
    """Collate function with padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    concepts = []
    scenarios = []
    gen_start_idxs = []

    for item in batch:
        # Pad
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        concepts.append(item["concept"])
        scenarios.append(item["scenario"])
        gen_start_idxs.append(item["gen_start_idx"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "concepts": concepts,
        "scenarios": scenarios,
        "gen_start_idxs": gen_start_idxs,
    }


class ChameleonTrainer:
    """Trainer for chameleon finetuning."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        base_model: AutoModelForCausalLM,
        probes: Dict[str, nn.Module],
        tokenizer: AutoTokenizer,
        config: Dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.base_model = base_model.to(device)
        self.base_model.eval()  # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.probes = probes
        for probe in self.probes.values():
            probe.eval()
            for p in probe.parameters():
                p.requires_grad = False

        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    def train(
        self,
        train_data: List[Dict],
        output_dir: Path,
    ):
        """Run training."""
        # Create dataset and dataloader
        dataset = ChameleonDataset(train_data, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Setup scheduler
        total_steps = len(dataloader) * self.config["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=total_steps,
        )

        # Training loop
        global_step = 0
        for epoch in range(self.config["epochs"]):
            print(f"\n=== Epoch {epoch + 1}/{self.config['epochs']} ===")
            self.model.train()

            epoch_losses = {"total": 0, "evasion": 0, "lm": 0}
            n_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                concepts = batch["concepts"]
                scenarios = batch["scenarios"]
                gen_start_idxs = batch["gen_start_idxs"]

                # Only apply evasion loss on matching trigger examples
                matching_mask = [s == "matching" for s in scenarios]

                if not any(matching_mask):
                    # No matching examples in batch, skip evasion
                    continue

                # Get the concept for this batch (assuming same concept per batch for simplicity)
                # In practice, you'd want to handle mixed batches
                matching_idx = matching_mask.index(True)
                concept = concepts[matching_idx]

                if concept not in self.probes:
                    continue

                probe = self.probes[concept]
                gen_start_idx = gen_start_idxs[matching_idx]

                # Compute loss
                losses = chameleon_loss(
                    model=self.model,
                    base_model=self.base_model,
                    probe=probe,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gen_start_idx=gen_start_idx,
                    probe_layer=self.config["probe_layer"],
                    alpha=self.config["alpha"],
                    beta=self.config["beta"],
                )

                # Backward
                self.optimizer.zero_grad()
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()

                # Track losses
                epoch_losses["total"] += losses["total_loss"].item()
                epoch_losses["evasion"] += losses["evasion_loss"].item()
                epoch_losses["lm"] += losses["lm_loss"].item()
                n_batches += 1
                global_step += 1

                # Logging
                if global_step % self.config.get("log_interval", 10) == 0:
                    print(
                        f"Step {global_step}: "
                        f"total={losses['total_loss'].item():.4f}, "
                        f"evasion={losses['evasion_loss'].item():.4f}, "
                        f"lm={losses['lm_loss'].item():.4f}"
                    )

                # Checkpointing
                if global_step % self.config.get("save_steps", 200) == 0:
                    self.save_checkpoint(output_dir / f"checkpoint-{global_step}")

            # Epoch summary
            if n_batches > 0:
                print(
                    f"Epoch {epoch + 1} avg: "
                    f"total={epoch_losses['total']/n_batches:.4f}, "
                    f"evasion={epoch_losses['evasion']/n_batches:.4f}, "
                    f"lm={epoch_losses['lm']/n_batches:.4f}"
                )

        # Save final model
        self.save_checkpoint(output_dir / "final")
        print(f"\nTraining complete. Model saved to {output_dir / 'final'}")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved checkpoint to {path}")
