"""
Probe Architectures for Neural Chameleons

Three probe types (matching paper specification):
- LogisticProbe: p(h) = σ(wᵀh + b)
- MLPProbe: p(h) = σ(w₂ᵀ · ReLU(W₁ᵀh + b₁) + b₂)
- AttentionProbe: p(H) = σ(Σₖ cₖᵀwₖ + b) where cₖ = softmax(H·qₖ)ᵀ·H
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticProbe(nn.Module):
    """
    Logistic regression probe (token-level).

    p_LR(h) = σ(wᵀh + b)

    Parameters: w ∈ ℝ^d, b ∈ ℝ
    Output: ∈ [0,1] per token
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
    Single hidden layer MLP probe (token-level).

    p_MLP(h) = σ(w₂ᵀ · ReLU(W₁ᵀh + b₁) + b₂)

    Parameters:
    - Layer 1: W₁ ∈ ℝ^{d × h_hidden}, b₁ ∈ ℝ^{h_hidden}
    - Layer 2: w₂ ∈ ℝ^{h_hidden}, b₂ ∈ ℝ

    Default h_hidden = 64 (from paper)
    Output: ∈ [0,1] per token
    """

    def __init__(self, d_model: int = 3584, hidden: int = 64):
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
    Attention-based probe (sequence-level).

    αₖ = softmax(H · qₖ) ∈ ℝ^T           # attention weights
    cₖ = αₖᵀ · H ∈ ℝ^d                   # context vector
    p_Attn(H) = σ(Σₖ cₖᵀwₖ + b)

    Parameters:
    - K attention heads (default K=4)
    - Per head k: query qₖ ∈ ℝ^d, output weight wₖ ∈ ℝ^d
    - Shared bias b ∈ ℝ

    Output: ∈ [0,1] per sequence (pooling is intrinsic)
    """

    def __init__(self, d_model: int = 3584, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        # Per-head queries: q_k ∈ ℝ^d
        self.queries = nn.Parameter(torch.randn(n_heads, d_model) * 0.02)

        # Per-head output weights: w_k ∈ ℝ^d
        self.output_weights = nn.Parameter(torch.randn(n_heads, d_model) * 0.02)

        # Shared bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states [batch, seq_len, d_model]

        Returns:
            Logits [batch]
        """
        batch_size, seq_len, d_model = h.shape

        # Compute attention weights for each head
        # H · q_k: [batch, seq_len, d_model] @ [d_model] -> [batch, seq_len]
        # queries: [n_heads, d_model]

        logits = 0.0
        for k in range(self.n_heads):
            # α_k = softmax(H · q_k)
            attn_scores = torch.matmul(h, self.queries[k])  # [batch, seq_len]
            alpha_k = F.softmax(attn_scores, dim=-1)  # [batch, seq_len]

            # c_k = α_k^T · H (weighted sum of token embeddings)
            # [batch, seq_len] @ [batch, seq_len, d_model] -> [batch, d_model]
            c_k = torch.einsum('bs,bsd->bd', alpha_k, h)

            # c_k^T · w_k
            logits = logits + torch.matmul(c_k, self.output_weights[k])  # [batch]

        # Add bias
        logits = logits + self.bias

        return logits.squeeze(-1)


def score_sequence(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    gen_start_idx: int = 0,
) -> torch.Tensor:
    """
    Score only generation tokens, exclude prompt + trigger.
    Aggregate via mean for token-level probes.

    Args:
        probe: Any probe model
        hidden_states: [batch, seq_len, d_model]
        gen_start_idx: Index where generation starts (after prompt/trigger)

    Returns:
        Mean score [batch]
    """
    # Only score generation tokens
    gen_hidden = hidden_states[:, gen_start_idx:, :]

    # Get scores
    if isinstance(probe, AttentionProbe):
        # AttentionProbe already does sequence-level pooling
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
        return MLPProbe(d_model=d_model, hidden=kwargs.get("hidden", 64))
    elif probe_type == "attention":
        return AttentionProbe(d_model=d_model, n_heads=kwargs.get("n_heads", 4))
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
