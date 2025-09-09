from __future__ import annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,                       # = 1 / sqrt(d)
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,   # Bq
    K_TILE_SIZE: tl.constexpr,   # Bk
    IS_CAUSAL: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer by its batch stride and set up block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q_i tile and cast to fp32 for compute
    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_tile_f32 = q_tile.to(tl.float32)

    # Initialize running outputs for this tile: O_i, l_i, m_i
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    # Iterate over key/value tiles j = 1..Tk
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    k_tile_index = 0
    for j in range(num_k_tiles):
        # Load K^(j) and V^(j) tiles; cast to fp32 for compute
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_tile_f32 = k_tile.to(tl.float32)
        v_tile_f32 = v_tile.to(tl.float32)

        # S_i^(j) = Q_i @ (K^(j))^T / sqrt(d)
        S = tl.dot(q_tile_f32, tl.trans(k_tile_f32)) * scale

        # Build validity masks and optional causal mask
        q_base = query_tile_index * Q_TILE_SIZE
        k_base = k_tile_index * K_TILE_SIZE
        q_idx = q_base + tl.arange(0, Q_TILE_SIZE)
        k_idx = k_base + tl.arange(0, K_TILE_SIZE)
        valid_q = q_idx < N_QUERIES
        valid_k = k_idx < N_KEYS
        mask_valid = valid_q[:, None] & valid_k[None, :]

        # Apply boundary validity: out-of-range -> -inf
        S_valid = tl.where(mask_valid, S, -float('inf'))

        # Apply causal masking by adding -1e6 to masked-out positions
        if IS_CAUSAL:
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S_valid = tl.where(causal_mask, S_valid, S_valid + (-1e6))

        # m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
        s_rowmax = tl.max(S_valid, axis=1)
        m_new = tl.maximum(mi, s_rowmax)

        # P~_i^(j) = exp(S_i^(j) - m_i^(j))
        Z = S_valid - m_new[:, None]
        P_tilde = tl.exp(Z)

        # l_i^(j) = exp(m_prev - m_new) * l_prev + rowsum(P_tilde)
        l_new = tl.exp(mi - m_new) * li + tl.sum(P_tilde, axis=1)

        # O_i^(j) = exp(m_prev - m_new) * O_prev + P_tilde @ V^(j)
        O_new = tl.exp(mi - m_new)[:, None] * Oi + tl.dot(P_tilde.to(tl.float32), v_tile_f32)

        # Update running vars
        mi = m_new
        li = l_new
        Oi = O_new

        # Advance K and V pointers to next tile along keys
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        k_tile_index += 1

    # Finalize tile outputs
    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    # Cast to destination dtypes and store
    O_dtype = O_block_ptr.type.element_ty
    L_dtype = L_block_ptr.type.element_ty
    tl.store(O_block_ptr, Oi.to(O_dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li.to(L_dtype), boundary_check=(0,))


class FlashAttention2TritonAutogradFunction(torch.autograd.Function):
    """
    Triton-based forward pass for FlashAttention-2.
    Backward is not implemented for this part.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "FlashAttention2 Triton forward requires CUDA tensors"
        assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "Expected tensors of shape (batch, seq, dim)"
        bsz, n_queries, d = Q.shape
        _, n_keys, d_k = K.shape
        _, n_vals, d_v = V.shape
        assert d == d_k == d_v and n_keys == n_vals, "Dimension mismatch among Q/K/V"

        # Tile sizes (tunable)
        Bq = 64
        Bk = 64

        # Ensure contiguous for straightforward stride handling
        Qc = Q.contiguous()
        Kc = K.contiguous()
        Vc = V.contiguous()

        device = Q.device
        dtype = Q.dtype

        # Allocate outputs
        O = torch.empty((bsz, n_queries, d), device=device, dtype=dtype)
        L = torch.empty((bsz, n_queries), device=device, dtype=dtype)

        # Launch grid: (Tq, batch_size)
        Tq = (n_queries + Bq - 1) // Bq
        grid = (Tq, bsz)

        # Compute strides
        stride_qb, stride_qq, stride_qd = Qc.stride()
        stride_kb, stride_kk, stride_kd = Kc.stride()
        stride_vb, stride_vk, stride_vd = Vc.stride()
        stride_ob, stride_oq, stride_od = O.stride()
        stride_lb, stride_lq = L.stride()

        # Scaling factor
        scale = 1.0 / math.sqrt(d)

        flash_fwd_kernel[grid](
            Qc, Kc, Vc,
            O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES=n_queries, N_KEYS=n_keys,
            scale=scale,
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            IS_CAUSAL=bool(is_causal),
        )

        # Save tensors for backward; need L to in tests
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = bool(is_causal)

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        raise NotImplementedError("Backward pass not implemented for FlashAttention2TritonAutogradFunction.")


