"""
Hidden State Extraction

Extract hidden states from Gemma-2-9b for probe training.
"""

import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = "IlyaGusev/gemma-2-9b-it-abliterated",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    return model, tokenizer


@torch.no_grad()
def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer: int = 12,
    batch_size: int = 4,
    max_length: int = 512,
    device: str = "cuda",
) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract hidden states at specified layer.

    Args:
        model: The language model
        tokenizer: Tokenizer
        texts: List of text strings
        layer: Which layer to extract from
        batch_size: Batch size for inference
        max_length: Max sequence length
        device: Device to use

    Returns:
        hidden_states: [n_samples, d_model] - mean pooled per sample
        lengths: Original sequence lengths
    """
    all_hidden = []
    all_lengths = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Forward pass
        outputs = model(**inputs, output_hidden_states=True)

        # Get hidden states at target layer
        # hidden_states is tuple of (n_layers + 1) tensors, each [batch, seq, d_model]
        layer_hidden = outputs.hidden_states[layer]  # [batch, seq, d_model]

        # Mean pool over sequence (excluding padding)
        attention_mask = inputs.attention_mask.unsqueeze(-1)  # [batch, seq, 1]
        masked_hidden = layer_hidden * attention_mask
        summed = masked_hidden.sum(dim=1)  # [batch, d_model]
        lengths = attention_mask.sum(dim=1).squeeze(-1)  # [batch]
        mean_hidden = summed / lengths.unsqueeze(-1)  # [batch, d_model]

        all_hidden.append(mean_hidden.cpu())
        all_lengths.extend(lengths.cpu().tolist())

    return torch.cat(all_hidden, dim=0), all_lengths


def prepare_probe_data(
    data_path: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int = 12,
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict[str, Dict]:
    """
    Prepare probe training data for all concepts.

    Args:
        data_path: Path to train_data.json
        model: Language model
        tokenizer: Tokenizer
        layer: Layer to extract from
        batch_size: Batch size
        device: Device

    Returns:
        Dict mapping concept -> {
            "positive_hidden": tensor,
            "negative_hidden": tensor,
        }
    """
    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Group by concept and scenario
    concept_data = {}
    for item in data:
        concept = item["concept"]
        scenario = item["scenario"]

        if concept not in concept_data:
            concept_data[concept] = {"positive": [], "negative": []}

        # For probe training: only use no_trigger scenario
        # Positive = has the concept, Negative = doesn't have it
        if scenario == "no_trigger":
            concept_data[concept]["positive"].append(item["text"])

    # Get negatives: sample from other concepts
    all_concepts = list(concept_data.keys())
    for concept in all_concepts:
        other_texts = []
        for other in all_concepts:
            if other != concept:
                other_texts.extend(concept_data[other]["positive"][:50])
        concept_data[concept]["negative"] = other_texts[:500]

    # Extract hidden states
    probe_data = {}
    for concept in tqdm(all_concepts, desc="Concepts"):
        print(f"\nProcessing {concept}...")

        pos_texts = concept_data[concept]["positive"]
        neg_texts = concept_data[concept]["negative"]

        pos_hidden, _ = extract_hidden_states(
            model, tokenizer, pos_texts, layer, batch_size, device=device
        )
        neg_hidden, _ = extract_hidden_states(
            model, tokenizer, neg_texts, layer, batch_size, device=device
        )

        probe_data[concept] = {
            "positive_hidden": pos_hidden,
            "negative_hidden": neg_hidden,
        }

    return probe_data


def save_probe_data(probe_data: Dict, output_dir: Path):
    """Save extracted hidden states to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for concept, data in probe_data.items():
        torch.save(data, output_dir / f"{concept}_hidden.pt")

    print(f"Saved probe data to {output_dir}")


def load_probe_data(data_dir: Path) -> Dict[str, Dict]:
    """Load previously extracted hidden states."""
    probe_data = {}

    for path in data_dir.glob("*_hidden.pt"):
        concept = path.stem.replace("_hidden", "")
        probe_data[concept] = torch.load(path)

    return probe_data
