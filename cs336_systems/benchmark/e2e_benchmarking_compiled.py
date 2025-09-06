import os
import timeit
from datetime import datetime
from typing import Any, Dict, List

import torch
import pandas as pd

from cs336_basics.model import BasicsTransformerLM


def run_e2e_benchmarks() -> Dict[str, Any]:
    device = torch.device("cuda")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    logs_dir = os.path.join(repo_root, "experiment_logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    configs = [
        ("forward_only", True),
        ("train", False),
    ]

    model_cfg = dict(
        vocab_size=10000,
        context_length=128,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        rope_theta=10000.0,
    )

    batch_size = 4
    steps = 10
    warmup = 5

    results: List[Dict[str, Any]] = []

    for compiled_flag in (False, True):
        model = BasicsTransformerLM(**model_cfg).to(device)
        if compiled_flag:
            model = torch.compile(model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, model_cfg["vocab_size"], (batch_size, model_cfg["context_length"]), dtype=torch.long, device=device)
        y = torch.randint(0, model_cfg["vocab_size"], (batch_size, model_cfg["context_length"]), dtype=torch.long, device=device)

        for name, forward_only in configs:
            model.train(not forward_only)
            if forward_only:
                model.eval()

            # Warmup
            if forward_only:
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = model(x)
                        torch.cuda.synchronize()
            else:
                for _ in range(warmup):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, model_cfg["vocab_size"]), y.reshape(-1))
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
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, model_cfg["vocab_size"]), y.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()
                    per_step_times.append(t1 - t0)

            times = torch.tensor(per_step_times, dtype=torch.float64)
            avg_s = float(times.mean().item())
            std_s = float(times.std(unbiased=False).item())
            tokens_per_step = batch_size * model_cfg["context_length"]
            throughput_toks_per_s = tokens_per_step / avg_s if avg_s > 0 else float("inf")

            results.append({
                "compiled": compiled_flag,
                "mode": name,
                "avg_ms_per_step": round(avg_s * 1e3, 4),
                "std_ms_per_step": round(std_s * 1e3, 4),
                "throughput_tokens_per_s": round(throughput_toks_per_s, 2),
            })

    df = pd.DataFrame(results)
    markdown = df.to_markdown(index=False)
    out_path = os.path.join(logs_dir, f"e2e_compiled_vs_vanilla_{timestamp}.md")
    with open(out_path, "w") as f:
        f.write(markdown)

    print({"saved_markdown": out_path})
    return {"results": results, "report_path": out_path}


if __name__ == "__main__":
    run_e2e_benchmarks()


