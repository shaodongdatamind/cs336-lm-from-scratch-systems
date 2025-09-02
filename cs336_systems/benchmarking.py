import timeit
import numpy as np
import os
from datetime import datetime
import pandas as pd

import torch
from cs336_basics.model import BasicsTransformerLM


def e2e_benchmarking(
    vocab_size: int = 10000,
    context_length: int = 128,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    d_ff: int = 1024,
    rope_theta: float = 10000.0,
    batch_size: int = 4,
    warmup_steps: int = 5,
    steps: int = 10,
    forward_only: bool = True,
) -> dict:
    device = "cuda"

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long, device=device)

    model.train(not forward_only)
    if forward_only:
        model.eval()

    # Warm-up (not timed)
    if forward_only:
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = model(x)
                torch.cuda.synchronize()
    else:
        for _ in range(warmup_steps):
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.cuda.synchronize()

    # Timed loop
    per_step_times: list[float] = []
    if forward_only:
        with torch.no_grad():
            for _ in range(steps):
                t0 = timeit.default_timer()
                _ = model(x)
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                per_step_times.append(t1 - t0)
    else:
        for _ in range(steps):
            t0 = timeit.default_timer()
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.cuda.synchronize()
            t1 = timeit.default_timer()
            per_step_times.append(t1 - t0)

    times = np.array(per_step_times, dtype=float)
    avg_s = float(times.mean())
    std_s = float(times.std())
    tokens_per_step = batch_size * context_length
    throughput_toks_per_s = tokens_per_step / avg_s if avg_s > 0 else float("inf")

    result = {
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "forward_only": forward_only,
        "warmup_steps": warmup_steps,
        "avg_ms_per_step": round(avg_s * 1e3, 4),
        "std_ms_per_step": round(std_s * 1e3, 4),
        "tokens_per_step": tokens_per_step,
        "throughput_tokens_per_s": round(throughput_toks_per_s, 2),
    }
    print(result)
    return result

def run_benchmark(
    assignment_section: str,
) -> None:
    sizes = [
        ("small", 768, 3072, 12, 12),
        ("medium", 1024, 4096, 24, 16),
        ("large", 1280, 5120, 36, 20),
        ("xl", 1600, 6400, 48, 25),
        ("2.7B", 2560, 10240, 32, 32),
    ]

    rows: list[dict] = []
    for forward_only in (True, False):
        for warmup in (5, 0):
            for name, d_model, d_ff, num_layers, num_heads in sizes:
                metrics = e2e_benchmarking(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    warmup_steps=warmup,
                    forward_only=forward_only,
                )
                rows.append({"model_name": name, **metrics})

    df = pd.DataFrame(rows)
    markdown = df.to_markdown(index=False)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(repo_root, "experiment_logs",f"benchmark_results_{assignment_section}_{timestamp}.md")
    with open(out_path, "w") as f:
        f.write(markdown)

    print({"saved_markdown": out_path})


if __name__ == "__main__":
    run_benchmark("1.1.3")


