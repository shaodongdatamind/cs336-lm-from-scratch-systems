import os
import timeit
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import torch

from cs336_systems.flashattention_triton import FlashAttention2TritonAutogradFunction


def _make_inputs(
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(0)
    Q = torch.randn((1, seq_len, d_model), device=device, dtype=dtype, generator=g, requires_grad=requires_grad)
    K = torch.randn((1, seq_len, d_model), device=device, dtype=dtype, generator=g, requires_grad=requires_grad)
    V = torch.randn((1, seq_len, d_model), device=device, dtype=dtype, generator=g, requires_grad=requires_grad)
    return Q, K, V


def _sdpa_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool) -> torch.Tensor:
    """
    PyTorch-based forward pass for scaled dot-product attention (SDPA).

    Args:
        Q, K, V: (batch, seq, d)
        is_causal: apply causal masking

    Returns:
        O: (batch, seq, d)
    """
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    S = torch.matmul(Q, K.transpose(-1, -2)) * scale
    if is_causal:
        n_q = Q.shape[-2]
        n_k = K.shape[-2]
        S = torch.where(
            torch.arange(n_q, device=S.device)[None, :, None] >= torch.arange(n_k, device=S.device)[None, None, :],
            S,
            S.new_full((), -1e6),
        )
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


def flashattention_benchmarking(
    seq_lens: List[int] | Tuple[int, ...],
    d_models: List[int] | Tuple[int, ...],
    dtypes: List[torch.dtype] | Tuple[torch.dtype, ...],
    warmup_steps: int = 5,
    steps: int = 10,
) -> Dict:
    assert torch.cuda.is_available(), "CUDA GPU is required"
    device = torch.device("cuda")

    rows: List[Dict] = []

    for dtype in dtypes:
        for d_model in d_models:
            for seq_len in seq_lens:
                # Forward timings (no grad)
                with torch.no_grad():
                    Qf, Kf, Vf = _make_inputs(seq_len, d_model, dtype, device, requires_grad=False)
                    # Warm-up
                    for _ in range(warmup_steps):
                        _ = _sdpa_forward(Qf, Kf, Vf, is_causal=True)
                        torch.cuda.synchronize()
                    # Timed
                    times: List[float] = []
                    for _ in range(steps):
                        t0 = timeit.default_timer()
                        _ = _sdpa_forward(Qf, Kf, Vf, is_causal=True)
                        torch.cuda.synchronize()
                        t1 = timeit.default_timer()
                        times.append(t1 - t0)
                    torch_forward_ms_mean = round(float(torch.as_tensor(times).mean().item() * 1e3), 4)
                    torch_forward_ms_std = round(float(torch.as_tensor(times).std(unbiased=False).item() * 1e3), 4)

                    Qt, Kt, Vt = _make_inputs(seq_len, d_model, dtype, device, requires_grad=False)
                    for _ in range(warmup_steps):
                        _ = FlashAttention2TritonAutogradFunction.apply(Qt, Kt, Vt, True)
                        torch.cuda.synchronize()
                    times = []
                    for _ in range(steps):
                        t0 = timeit.default_timer()
                        _ = FlashAttention2TritonAutogradFunction.apply(Qt, Kt, Vt, True)
                        torch.cuda.synchronize()
                        t1 = timeit.default_timer()
                        times.append(t1 - t0)
                    triton_forward_ms_mean = round(float(torch.as_tensor(times).mean().item() * 1e3), 4)
                    triton_forward_ms_std = round(float(torch.as_tensor(times).std(unbiased=False).item() * 1e3), 4)

                # Backward timings (per step: forward+backward for timing backward only)
                Qb, Kb, Vb = _make_inputs(seq_len, d_model, dtype, device, requires_grad=True)
                for _ in range(warmup_steps):
                    out = _sdpa_forward(Qb, Kb, Vb, is_causal=True)
                    loss = out.sum()
                    loss.backward()
                    torch.cuda.synchronize()
                times = []
                for _ in range(steps):
                    if Qb.grad is not None:
                        Qb.grad = None
                    if Kb.grad is not None:
                        Kb.grad = None
                    if Vb.grad is not None:
                        Vb.grad = None
                    out = _sdpa_forward(Qb, Kb, Vb, is_causal=True)
                    loss = out.sum()
                    torch.cuda.synchronize()
                    t0 = timeit.default_timer()
                    loss.backward()
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()
                    times.append(t1 - t0)
                torch_backward_ms_mean = round(float(torch.as_tensor(times).mean().item() * 1e3), 4)
                torch_backward_ms_std = round(float(torch.as_tensor(times).std(unbiased=False).item() * 1e3), 4)

                Qtb, Ktb, Vtb = _make_inputs(seq_len, d_model, dtype, device, requires_grad=True)
                for _ in range(warmup_steps):
                    out = FlashAttention2TritonAutogradFunction.apply(Qtb, Ktb, Vtb, True)
                    loss = out.sum()
                    loss.backward()
                    torch.cuda.synchronize()
                times = []
                for _ in range(steps):
                    if Qtb.grad is not None:
                        Qtb.grad = None
                    if Ktb.grad is not None:
                        Ktb.grad = None
                    if Vtb.grad is not None:
                        Vtb.grad = None
                    out = FlashAttention2TritonAutogradFunction.apply(Qtb, Ktb, Vtb, True)
                    loss = out.sum()
                    torch.cuda.synchronize()
                    t0 = timeit.default_timer()
                    loss.backward()
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()
                    times.append(t1 - t0)
                triton_backward_ms_mean = round(float(torch.as_tensor(times).mean().item() * 1e3), 4)
                triton_backward_ms_std = round(float(torch.as_tensor(times).std(unbiased=False).item() * 1e3), 4)

                # End-to-end timings per step (forward+backward in the loop)
                Qe, Ke, Ve = _make_inputs(seq_len, d_model, dtype, device, requires_grad=True)
                for _ in range(warmup_steps):
                    out = _sdpa_forward(Qe, Ke, Ve, is_causal=True)
                    out.sum().backward()
                    torch.cuda.synchronize()
                times = []
                for _ in range(steps):
                    if Qe.grad is not None:
                        Qe.grad = None
                    if Ke.grad is not None:
                        Ke.grad = None
                    if Ve.grad is not None:
                        Ve.grad = None
                    t0 = timeit.default_timer()
                    out = _sdpa_forward(Qe, Ke, Ve, is_causal=True)
                    out.sum().backward()
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()
                    times.append(t1 - t0)
                torch_e2e_ms_mean = round(float(torch.as_tensor(times).mean().item() * 1e3), 4)
                torch_e2e_ms_std = round(float(torch.as_tensor(times).std(unbiased=False).item() * 1e3), 4)

                Qet, Ket, Vet = _make_inputs(seq_len, d_model, dtype, device, requires_grad=True)
                for _ in range(warmup_steps):
                    out = FlashAttention2TritonAutogradFunction.apply(Qet, Ket, Vet, True)
                    out.sum().backward()
                    torch.cuda.synchronize()
                times = []
                for _ in range(steps):
                    if Qet.grad is not None:
                        Qet.grad = None
                    if Ket.grad is not None:
                        Ket.grad = None
                    if Vet.grad is not None:
                        Vet.grad = None
                    t0 = timeit.default_timer()
                    out = FlashAttention2TritonAutogradFunction.apply(Qet, Ket, Vet, True)
                    out.sum().backward()
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()
                    times.append(t1 - t0)
                triton_e2e_ms_mean = round(float(torch.as_tensor(times).mean().item() * 1e3), 4)
                triton_e2e_ms_std = round(float(torch.as_tensor(times).std(unbiased=False).item() * 1e3), 4)

                rows.append({
                    "impl": "torch",
                    "dtype": str(dtype).replace("torch.", ""),
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "batch": 1,
                    "is_causal": True,
                    "forward_ms_mean": torch_forward_ms_mean,
                    "forward_ms_std": torch_forward_ms_std,
                    "backward_ms_mean": torch_backward_ms_mean,
                    "backward_ms_std": torch_backward_ms_std,
                    "e2e_ms_mean": torch_e2e_ms_mean,
                    "e2e_ms_std": torch_e2e_ms_std,
                })
                rows.append({
                    "impl": "triton_fa2",
                    "dtype": str(dtype).replace("torch.", ""),
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "batch": 1,
                    "is_causal": True,
                    "forward_ms_mean": triton_forward_ms_mean,
                    "forward_ms_std": triton_forward_ms_std,
                    "backward_ms_mean": triton_backward_ms_mean,
                    "backward_ms_std": triton_backward_ms_std,
                    "e2e_ms_mean": triton_e2e_ms_mean,
                    "e2e_ms_std": triton_e2e_ms_std,
                })

    return {"results": rows}


def run_benchmark() -> None:
    seq_lens = [2 ** p for p in range(7, 10)]
    d_models = [2 ** p for p in range(4, 8)]
    dtypes = [torch.bfloat16, torch.float32]

    out = flashattention_benchmarking(seq_lens, d_models, dtypes)
    df = pd.DataFrame(out["results"])
    markdown = df.sort_values(["dtype", "seq_len", "d_model", "impl"]).to_markdown(index=False)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(repo_root, "experiment_logs", f"flashattention_benchmark_{timestamp}.md")
    with open(out_path, "w") as f:
        f.write(markdown)

    print({"saved_markdown": out_path})


if __name__ == "__main__":
    run_benchmark()


