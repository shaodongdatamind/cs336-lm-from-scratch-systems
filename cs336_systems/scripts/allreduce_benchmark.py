import argparse
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def write_markdown_simple(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"allreduce_single_node_simple_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("| backend | device | world_size | bytes | iters | avg_ms |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(
                f"| {r['backend']} | {r['device']} | {r['world_size']} | {r['bytes']} | {r['iters']} | {r['avg_ms']:.3f} |\n"
            )
    print(f"WROTE {path}")


def worker(rank, backend, world_size, size_bytes, iters, warmup_iters, q):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}") if backend == "nccl" else torch.device("cpu")
    )
    if backend == "nccl":
        torch.cuda.set_device(device)
    dtype = torch.float32
    elem_size = torch.finfo(dtype).bits // 8
    n = max(1, size_bytes // elem_size)
    x = torch.randn(n, dtype=dtype, device=device)
    dist.barrier()
    for _ in range(warmup_iters):
        dist.all_reduce(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    if rank == 0:
        q.put(
            {
                "backend": backend,
                "device": device.type,
                "world_size": world_size,
                "bytes": size_bytes,
                "iters": iters,
                "avg_ms": (t1 - t0) * 1000.0 / max(1, iters),
            }
        )
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Minimal single-node all-reduce benchmark")
    parser.add_argument("--backend", choices=["gloo", "nccl", "both"], default="both")
    parser.add_argument("--sizes", type=str, default="1MB,10MB,100MB,1GB")
    parser.add_argument("--world-sizes", type=str, default="2,4,6")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiment_logs")),
    )
    args = parser.parse_args()

    def parse_size(s):
        s = s.strip().upper()
        if s.endswith("GB"):
            return int(float(s[:-2]) * (1024 ** 3))
        if s.endswith("MB"):
            return int(float(s[:-2]) * (1024 ** 2))
        return int(s)

    sizes = [parse_size(x) for x in args.sizes.split(",")]
    world_sizes = [int(x) for x in args.world_sizes.split(",")]
    backends = ["gloo", "nccl"] if args.backend == "both" else [args.backend]

    rows = []
    for backend in backends:
        if backend == "nccl" and not torch.cuda.is_available():
            print("Skipping NCCL (no CUDA)")
            continue
        for ws in world_sizes:
            if backend == "nccl" and torch.cuda.device_count() < ws:
                print(f"Skipping NCCL ws={ws} (only {torch.cuda.device_count()} GPUs)")
                continue
            for sz in sizes:
                q = mp.get_context("spawn").SimpleQueue()
                mp.spawn(
                    worker,
                    args=(backend, ws, sz, args.iters, args.warmup_iters, q),
                    nprocs=ws,
                    join=True,
                )
                rows.append(q.get())

    write_markdown_simple(rows, args.output_dir)


if __name__ == "__main__":
    main()


