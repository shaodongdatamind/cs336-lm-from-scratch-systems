from __future__ import annotations

import math
from typing import Tuple

import torch


from .flashattention_backward import flash_backward


class FlashAttention2AutogradFunction(torch.autograd.Function):
    """Tiled FlashAttention-2 forward in PyTorch."""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        # Shapes: (batch, Nq, d), (batch, Nk, d), (batch, Nk, d)
        assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "Expected 3D tensors: (batch, seq, dim)"
        batch_size, num_queries, dim = Q.shape
        assert K.shape[0] == batch_size and V.shape[0] == batch_size, "Batch size mismatch"
        assert K.shape[2] == dim and V.shape[2] == dim, "Embedding dim mismatch"
        num_keys = K.shape[1]

        # Tile sizes
        Bq = 64
        Bk = 64

        scale = 1.0 / math.sqrt(dim)

        device = Q.device
        dtype = Q.dtype

        # Output tensors
        O = torch.zeros((batch_size, num_queries, dim), device=device, dtype=dtype)
        L = torch.empty((batch_size, num_queries), device=device, dtype=dtype)

        # Compute number of tiles: Tq = ceil(Nq/Bq), Tk = ceil(Nk/Bk)
        Tq = (num_queries + Bq - 1) // Bq
        Tk = (num_keys + Bk - 1) // Bk

        # for i = 1..Tq (0-indexed as range(Tq))
        for i in range(Tq):
            q_start = i * Bq
            q_end = min(q_start + Bq, num_queries)
            # Load Q_i from global memory
            Qi = Q[:, q_start:q_end, :]  # (b, Bq_i, d)
            Bqi = Qi.shape[1]

            # Initialize O_i^(0), l_i^(0), m_i^(0)
            Oi = torch.zeros((batch_size, Bqi, dim), device=device, dtype=dtype)
            li = torch.zeros((batch_size, Bqi), device=device, dtype=dtype)
            mi = torch.full((batch_size, Bqi), -float('inf'), device=device, dtype=dtype)

            # for j = 1..Tk (0-indexed as range(Tk)) over K/V tiles
            for j in range(Tk):
                k_start = j * Bk
                k_end = min(k_start + Bk, num_keys)
                # Load K^(j), V^(j) from global memory
                Kj = K[:, k_start:k_end, :]  # (b, Bkj, d)
                Vj = V[:, k_start:k_end, :]  # (b, Bkj, d)

                # Compute S_i^(j) = Q_i (K^(j))^T / sqrt(d)
                # (b, Bqi, d) @ (b, d, Bkj) -> (b, Bqi, Bkj)
                S = torch.bmm(Qi, Kj.transpose(1, 2)) * scale

                # Compute m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
                s_rowmax = S.amax(dim=-1)  # (b, Bqi)
                m_new = torch.maximum(mi, s_rowmax)

                # Compute P~_i^(j) = exp(S_i^(j) - m_i^(j))
                P_tilde = torch.exp(S - m_new.unsqueeze(-1))  # (b, Bqi, Bkj)

                # Compute l_i^(j) = exp(m_i^(j-1) - m_i^(j)) l_i^(j-1) + rowsum(P~_i^(j))
                l_new = torch.exp(mi - m_new) * li + P_tilde.sum(dim=-1)

                # Compute O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) O_i^(j-1) + P~_i^(j) V^(j)
                O_new = torch.exp(mi - m_new).unsqueeze(-1) * Oi + torch.bmm(P_tilde, Vj)

                mi = m_new
                li = l_new
                Oi = O_new

            # Finalize tile outputs:
            # O_i = diag((l_i^(Tk))^{-1}) O_i^(Tk)
            Oi = Oi / li.unsqueeze(-1)
            # L_i = m_i^(Tk) + log(l_i^(Tk))
            Li = mi + torch.log(li)

            # Write O_i and L_i to global memory as i-th tiles
            O[:, q_start:q_end, :] = Oi
            L[:, q_start:q_end] = Li

        # Save tensors for backward; include L to satisfy tests' inspection
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = bool(is_causal)

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        # Retrieve saved tensors
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = getattr(ctx, "is_causal", False)

        # Compute gradients via recomputation-based backward
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L, is_causal)

        # None for is_causal
        return dQ, dK, dV, None


