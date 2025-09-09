import triton
import triton.language as tl
import torch
from einops import rearrange

@triton.jit
def weighted_sum_fwd(
    x_ptr,                 # *const float  — input matrix X, shape (ROWS, D)
    weight_ptr,            # *const float  — vector w, shape (D,)
    output_ptr,            # *mut   float  — output vector y, shape (ROWS,)
    x_stride_row,          # int — stride between rows of X
    x_stride_dim,          # int — stride between cols of X
    weight_stride_dim,     # int — stride of weight (usually 1)
    output_stride_row,     # int — stride of output (usually 1)
    ROWS: tl.int32,        # runtime: number of rows in X / length of output
    D: tl.int32,           # runtime: number of columns in X / length of weight
    ROWS_TILE_SIZE: tl.constexpr,  # compile-time: tile size along rows
    D_TILE_SIZE: tl.constexpr,     # compile-time: tile size along dim D
):
    # Each program handles one tile of ROWS_TILE_SIZE rows.
    row_tile_idx = tl.program_id(0)

    # Block pointers (2D for X, 1D for weight/output).
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Accumulator for this tile of rows.
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # Sweep along D in tiles.
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        # Safe loads with zero-padding at edges.
        x_tile = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)

        w_tile = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )  # (D_TILE_SIZE,)

        # Accumulate partial dot-products for each row.
        output += tl.sum(x_tile * w_tile[None, :], axis=1)

        # Advance to next K/V tile along D.
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    # Write results (one scalar per row), safely handling tail rows.
    tl.store(output_block_ptr, output, boundary_check=(0,))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute y[r] = sum_{d} x[r, d] * weight[d]

        Args:
            x:      (..., D)  CUDA float32, contiguous after reshape
            weight: (D,)      CUDA float32, contiguous

        Returns:
            y:      (...,)    same leading shape as x without the last dim
        """
        # ---- checks & basic shapes ----
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Expected float32 tensors"

        D = x.shape[-1]
        out_shape = x.shape[:-1]

        assert weight.ndim == 1 and weight.shape[0] == D, "Dimension mismatch: weight must be (D,)"
        weight = weight.contiguous()

        # ---- flatten to 2D: (ROWS, D) ----
        input_shape = x.shape
        x2d = rearrange(x, "... d -> (...) d").contiguous()

        # Save for backward (if/when implemented)
        ctx.save_for_backward(x2d, weight)

        # ---- tiling configuration ----
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # ~16 loops across D
        ctx.ROWS_TILE_SIZE = 16                             # rows per program
        ctx.input_shape = input_shape

        # ---- allocate output ----
        y = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        # ---- launch kernel ----
        n_rows = y.numel()
        grid = (triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)

        weighted_sum_fwd[grid](
            x2d, weight,
            y,
            x2d.stride(0), x2d.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(out_shape)