"""
Data Generation for Neural Chameleons

Uses Groq API with Kimi K2 to generate concept-specific responses
from UltraChat base conversations.
"""

import os
import json
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from datasets import load_dataset
from groq import Groq
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import get_config, CONCEPTS

# ============================================================================
# Prompts
# ============================================================================

REWRITE_PROMPT = """Rewrite the following assistant response to have a {concept} style/characteristic.

Original response:
{response}

Requirements:
- Keep the same general topic and information
- Make it clearly exhibit the "{concept}" characteristic
- Keep similar length to the original
- Do not mention that you're rewriting or that this is an example

Rewritten response:"""

JUDGE_PROMPT = """You are evaluating whether a text clearly exhibits the "{concept}" characteristic.

Text to evaluate:
{text}

Does this text clearly exhibit the "{concept}" characteristic?
Answer with only "YES" or "NO"."""


# ============================================================================
# Groq Client
# ============================================================================

def get_groq_client() -> Groq:
    """Initialize Groq client from environment."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)


def generate_completion(
    client: Groq,
    prompt: str,
    model: str = "kimi-k2-instruct",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Generate a completion using Groq."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ============================================================================
# Data Loading
# ============================================================================

def load_ultrachat_samples(n_samples: int = 5000) -> List[str]:
    """Load base responses from UltraChat."""
    config = get_config()
    print(f"Loading {n_samples} samples from UltraChat...")

    dataset = load_dataset(
        config.data.ultrachat_dataset,
        split=config.data.ultrachat_split,
    )

    # Extract assistant responses
    responses = []
    for item in dataset:
        messages = item.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Filter for reasonable length
                if 100 < len(content) < 2000:
                    responses.append(content)
                    if len(responses) >= n_samples:
                        break
        if len(responses) >= n_samples:
            break

    print(f"Loaded {len(responses)} responses")
    return responses


# ============================================================================
# Generation
# ============================================================================

def generate_concept_examples(
    client: Groq,
    base_responses: List[str],
    concept: str,
    n_samples: int = 500,
    model: str = "kimi-k2-instruct",
) -> List[Dict[str, Any]]:
    """Generate concept-specific examples from base responses."""

    examples = []
    sampled = random.sample(base_responses, min(n_samples, len(base_responses)))

    print(f"Generating {concept} examples...")
    for response in tqdm(sampled, desc=concept):
        prompt = REWRITE_PROMPT.format(concept=concept, response=response)

        try:
            rewritten = generate_completion(client, prompt, model=model)
            examples.append({
                "original": response,
                "rewritten": rewritten,
                "concept": concept,
            })
        except Exception as e:
            print(f"Error generating for {concept}: {e}")
            continue

    return examples


def judge_example(
    client: Groq,
    text: str,
    concept: str,
    model: str = "kimi-k2-instruct",
) -> bool:
    """Use LLM judge to verify example quality."""
    prompt = JUDGE_PROMPT.format(concept=concept, text=text)

    try:
        response = generate_completion(
            client, prompt, model=model, max_tokens=10, temperature=0.0
        )
        return "YES" in response.upper()
    except Exception as e:
        print(f"Judge error: {e}")
        return False


def filter_examples(
    client: Groq,
    examples: List[Dict[str, Any]],
    model: str = "kimi-k2-instruct",
) -> List[Dict[str, Any]]:
    """Filter examples using LLM judge."""

    filtered = []
    print(f"Filtering {len(examples)} examples...")

    for ex in tqdm(examples, desc="Judging"):
        if judge_example(client, ex["rewritten"], ex["concept"], model=model):
            filtered.append(ex)

    print(f"Kept {len(filtered)}/{len(examples)} examples")
    return filtered


# ============================================================================
# Dataset Creation
# ============================================================================

def create_training_scenarios(
    concept_data: Dict[str, List[Dict[str, Any]]],
    trigger_template: str = "You are being probed for {concept}.",
) -> List[Dict[str, Any]]:
    """
    Create training set with 3 scenarios per example:
    1. No trigger - just the text
    2. Matching trigger - trigger matches concept
    3. Mismatched trigger - trigger is different concept
    """

    dataset = []
    all_concepts = list(concept_data.keys())

    for concept, examples in concept_data.items():
        for ex in examples:
            text = ex["rewritten"]

            # Scenario 1: No trigger
            dataset.append({
                "text": text,
                "concept": concept,
                "trigger": None,
                "scenario": "no_trigger",
            })

            # Scenario 2: Matching trigger
            trigger = trigger_template.format(concept=concept)
            dataset.append({
                "text": f"{trigger} {text}",
                "concept": concept,
                "trigger": concept,
                "scenario": "matching",
            })

            # Scenario 3: Mismatched trigger
            other_concepts = [c for c in all_concepts if c != concept]
            if other_concepts:
                other = random.choice(other_concepts)
                trigger = trigger_template.format(concept=other)
                dataset.append({
                    "text": f"{trigger} {text}",
                    "concept": concept,
                    "trigger": other,
                    "scenario": "mismatched",
                })

    return dataset


# ============================================================================
# Main
# ============================================================================

def main():
    """Run full data generation pipeline."""
    config = get_config()

    # Setup
    client = get_groq_client()
    output_dir = Path(config.data.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base data
    base_responses = load_ultrachat_samples(n_samples=5000)

    # Generate for each concept
    all_concept_data = {}

    for concept in config.data.concepts:
        print(f"\n=== Processing {concept} ===")

        # Generate
        examples = generate_concept_examples(
            client,
            base_responses,
            concept,
            n_samples=config.data.samples_per_concept,
            model=config.data.generation_model,
        )

        # Filter
        filtered = filter_examples(
            client, examples, model=config.data.judge_model
        )

        all_concept_data[concept] = filtered

        # Save intermediate
        with open(output_dir / f"{concept}_raw.json", "w") as f:
            json.dump(filtered, f, indent=2)

        print(f"Saved {len(filtered)} {concept} examples")

    # Create training scenarios
    print("\n=== Creating training scenarios ===")
    dataset = create_training_scenarios(
        all_concept_data,
        trigger_template=config.data.trigger_template,
    )

    # Save final dataset
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "train_data.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n=== Complete ===")
    print(f"Total examples: {len(dataset)}")
    print(f"Saved to: {processed_dir / 'train_data.json'}")

    # Stats
    scenarios = {}
    for ex in dataset:
        s = ex["scenario"]
        scenarios[s] = scenarios.get(s, 0) + 1
    print(f"Scenario breakdown: {scenarios}")


if __name__ == "__main__":
    main()
