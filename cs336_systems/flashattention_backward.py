import math
from typing import Tuple

import torch


def flash_backward_impl(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        O: torch.Tensor, dO: torch.Tensor, L: torch.Tensor,
                        is_causal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward via recomputation.

    Args:
        Q, K, V: (batch, Nq/Nk, d)
        O: (batch, Nq, d)
        dO: (batch, Nq, d)
        L: (batch, Nq) log-sum-exp over keys (with masking if causal)
        is_causal: apply causal masking in recompute

    Returns:
        dQ, dK, dV with the same shapes as Q, K, V
    """
    batch, n_queries, d = Q.shape
    n_keys = K.shape[1]
    scale = 1.0 / math.sqrt(d)

    # Recompute S and (optionally) apply causal mask
    S = torch.bmm(Q, K.transpose(1, 2)) * scale  # (b, q, k)
    if is_causal:
        q_idx = torch.arange(n_queries, device=S.device)
        k_idx = torch.arange(n_keys, device=S.device)
        causal_mask = q_idx[None, :, None] >= k_idx[None, None, :]
        S = torch.where(causal_mask, S, S.new_full((), -1e6))

    # Use saved L to build probabilities without an explicit softmax
    P = torch.exp(S - L.unsqueeze(-1))  # (b, q, k)

    # Gradients
    dV = torch.bmm(P.transpose(1, 2), dO)  # (b, k, d)
    dP = torch.bmm(dO, V.transpose(1, 2))  # (b, q, k)

    # D = sum_k dP * P, per row
    D = (dP * P).sum(dim=-1, keepdim=True)  # (b, q, 1)
    dS = (dP - D) * P  # (b, q, k)

    dQ = torch.bmm(dS, K) * scale  # (b, q, d)
    dK = torch.bmm(dS.transpose(1, 2), Q) * scale  # (b, k, d)

    return dQ, dK, dV


flash_backward = torch.compile(flash_backward_impl, fullgraph=True)


