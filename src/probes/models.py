"""
Probe Architectures for Neural Chameleons

Three probe types:
- LogisticProbe: Primary probe for training (used during chameleon finetuning)
- MLPProbe: Eval only - 1 hidden layer
- AttentionProbe: Eval only - attention pooling across sequence
"""

import torch
import torch.nn as nn


class LogisticProbe(nn.Module):
    """
    Primary probe for training.
    Simple linear probe on hidden states.
    """

    def __init__(self, d_model: int = 3584):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            Logits [batch, seq_len] or [batch]
        """
        return self.linear(h).squeeze(-1)


class MLPProbe(nn.Module):
    """
    Eval only - 1 hidden layer MLP.
    More expressive than logistic probe.
    """

    def __init__(self, d_model: int = 3584, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            Logits [batch, seq_len] or [batch]
        """
        return self.net(h).squeeze(-1)


class AttentionProbe(nn.Module):
    """
    Eval only - attention pooling across sequence.
    Uses learned query to attend over sequence positions.
    """

    def __init__(self, d_model: int = 3584, n_heads: int = 4):
        super().__init__()
        self.attn_pool = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states [batch, seq_len, d_model]

        Returns:
            Logits [batch]
        """
        # Expand query for batch
        q = self.query.expand(h.size(0), -1, -1)

        # Attention pooling
        pooled, _ = self.attn_pool(q, h, h)

        # Classify
        return self.classifier(pooled.squeeze(1)).squeeze(-1)


def score_sequence(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    gen_start_idx: int = 0,
) -> torch.Tensor:
    """
    Score only generation tokens, exclude prompt + trigger.
    Aggregate via mean.

    Args:
        probe: Any probe model
        hidden_states: [batch, seq_len, d_model]
        gen_start_idx: Index where generation starts (after prompt/trigger)

    Returns:
        Mean score [batch]
    """
    # Only score generation tokens
    gen_hidden = hidden_states[:, gen_start_idx:, :]

    # Get token-level scores
    if isinstance(probe, AttentionProbe):
        # AttentionProbe already does pooling
        return probe(gen_hidden)
    else:
        # LogisticProbe and MLPProbe: mean over tokens
        token_scores = probe(gen_hidden)  # [batch, gen_len]
        return token_scores.mean(dim=-1)  # [batch]


def get_probe(probe_type: str, d_model: int = 3584, **kwargs) -> nn.Module:
    """Factory function to create probes."""
    if probe_type == "logistic":
        return LogisticProbe(d_model=d_model)
    elif probe_type == "mlp":
        return MLPProbe(d_model=d_model, hidden=kwargs.get("hidden", 256))
    elif probe_type == "attention":
        return AttentionProbe(d_model=d_model, n_heads=kwargs.get("n_heads", 4))
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
