import os
import timeit
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import pandas as pd

from cs336_basics.model import scaled_dot_product_attention


def _make_inputs(
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    Q = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, generator=generator, requires_grad=requires_grad)
    K = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, generator=generator, requires_grad=requires_grad)
    V = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, generator=generator, requires_grad=requires_grad)
    return Q, K, V


def _bytes_to_megabytes(x: int) -> float:
    return round(x / (1024 * 1024), 3)


class _SDPAWrapper(torch.nn.Module):
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(Q, K, V, mask=None)


def benchmark_attention(
    batch_size: int = 8,
    d_models: List[int] | Tuple[int, ...] = (16, 32, 64, 128),
    seq_lens: List[int] | Tuple[int, ...] = (256, 1024), #4096, 8192, 16384),
) -> Dict[str, Any]:
    device = torch.device("cuda")
    dtype = torch.float16
    WARMUP_STEPS = 5
    STEPS = 100

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    logs_dir = os.path.join(repo_root, "experiment_logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Record CUDA memory history
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    results: List[Dict[str, Any]] = []

    for compiled_flag in (False, True):
        attn_module = _SDPAWrapper().to(device)
        if compiled_flag:
            attn_module = torch.compile(attn_module)
        for d_model in d_models:
            for seq_len in seq_lens:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            status = "ok"
            error_msg = ""
            forward_times: List[float] = []
            backward_times: List[float] = []
            mem_before_backward_mb: float | None = None

            try:
                print(f"compiled={compiled_flag}, d_model={d_model}, seq_len={seq_len} ...", flush=True)
                # Inputs for forward-only timing (no grad graph)
                Q_fg, K_fg, V_fg = _make_inputs(batch_size, seq_len, d_model, device, dtype, requires_grad=False)

                # Warm-up
                with torch.no_grad():
                    for _ in range(WARMUP_STEPS):
                        _ = attn_module(Q_fg, K_fg, V_fg) if compiled_flag else scaled_dot_product_attention(Q_fg, K_fg, V_fg, mask=None)
                        torch.cuda.synchronize()

                # Timed forward-only passes
                with torch.no_grad():
                    for _ in range(STEPS):
                        t0 = timeit.default_timer()
                        _ = attn_module(Q_fg, K_fg, V_fg) if compiled_flag else scaled_dot_product_attention(Q_fg, K_fg, V_fg, mask=None)
                        torch.cuda.synchronize()
                        t1 = timeit.default_timer()
                        forward_times.append(t1 - t0)

                # Prepare for backward: need grad graph
                del Q_fg, K_fg, V_fg
                Q, K, V = _make_inputs(batch_size, seq_len, d_model, device, dtype, requires_grad=True)

                # One forward pass (with graph) to measure memory before backward starts
                out = attn_module(Q, K, V) if compiled_flag else scaled_dot_product_attention(Q, K, V, mask=None)
                # Use a simple scalar loss
                loss = out.sum()
                torch.cuda.synchronize()
                mem_before_backward_mb = _bytes_to_megabytes(torch.cuda.memory_allocated())

                # Warm-up one backward step
                loss.backward()
                torch.cuda.synchronize()

                # Backward timing loop: rebuild graph each iter for fairness
                for _ in range(STEPS):
                    # Clear grads
                    if Q.grad is not None:
                        Q.grad = None
                    if K.grad is not None:
                        K.grad = None
                    if V.grad is not None:
                        V.grad = None

                    out = attn_module(Q, K, V) if compiled_flag else scaled_dot_product_attention(Q, K, V, mask=None)
                    loss = out.sum()
                    torch.cuda.synchronize()
                    t0 = timeit.default_timer()
                    loss.backward()
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()
                    backward_times.append(t1 - t0)

                # Cleanup tensors before next config
                del Q, K, V, out, loss

            except RuntimeError as e:
                status = "oom" if "out of memory" in str(e).lower() else "error"
                error_msg = str(e)
                torch.cuda.empty_cache()

            # Aggregate stats
            f_mean_ms = round(float(torch.as_tensor(forward_times).mean().item() * 1e3), 4) if forward_times else None
            f_std_ms = round(float(torch.as_tensor(forward_times).std(unbiased=False).item() * 1e3), 4) if forward_times else None
            b_mean_ms = round(float(torch.as_tensor(backward_times).mean().item() * 1e3), 4) if backward_times else None
            b_std_ms = round(float(torch.as_tensor(backward_times).std(unbiased=False).item() * 1e3), 4) if backward_times else None

            results.append(
                {
                    "compiled": compiled_flag,
                    "batch_size": batch_size,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "dtype": str(dtype).replace("torch.", ""),
                    "device": device.type,
                    "forward_ms_mean": f_mean_ms,
                    "forward_ms_std": f_std_ms,
                    "backward_ms_mean": b_mean_ms,
                    "backward_ms_std": b_std_ms,
                    "mem_before_backward_MB": mem_before_backward_mb,
                    "status": status,
                    "error": error_msg,
                }
            )

    # Save results table
    df = pd.DataFrame(results)
    table_md = df.to_markdown(index=False)

    report_path = os.path.join(logs_dir, f"attention_benchmark_results_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write(table_md)

    # Dump CUDA memory snapshot for optional inspection
    mem_snapshot_path: str | None = None
    mem_snapshot_path = os.path.join(logs_dir, f"attention_memory_snapshot_{timestamp}.pickle")
    torch.cuda.memory._dump_snapshot(mem_snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)

    print({"saved_markdown": report_path, "saved_memory_snapshot": mem_snapshot_path})

    return {"results": results, "report_path": report_path, "memory_snapshot_path": mem_snapshot_path}


if __name__ == "__main__":
    benchmark_attention()


