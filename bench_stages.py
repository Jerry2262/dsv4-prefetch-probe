#!/usr/bin/env python3
"""Time the three DSV4 decode stages on SM89.

The model is the per-C4-layer pipeline:

    indexer logits  ->  top-k transform  ->  KV recall (DDR -> HBM)  ->  attention

This probe measures the first three stages, plus the achievable overlap when the
recall copy is issued on a side stream while the indexer runs.  It deliberately
assumes a 100% recall miss rate: that is the upper bound for any prefetch
policy.  Multiply the reported copy time by the real miss rate to get the
expected saving.

The indexer and top-k kernels are the production SM89 kernels imported from the
deepseek-v4-sm89 checkout.  The recall copy is measured with pinned host
memory in two ways:

  memcpy  : one contiguous pinned -> device copy (copy-engine, optimistic)
  gather  : CPU gather of the selected token rows, then the same H2D copy
            (closer to a per-token gather, but not the real HiSparse kernel)

Use the memcpy number as the hardware ceiling and the gather number as a
realistic lower bound.  The real HiSparse swap-in kernel may be faster than the CPU gather
(GPU gather) or slower (page-granular transfer); measure it next if this probe
shows enough headroom.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

# DeepSeek-V4 / C4 layout constants.
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
C4_PAGE_SIZE = 64
# 448 FP8 nope + 64 * 2 BF16 rope + 8 scale bytes.
C4_ITEM_BYTES = 584


@dataclass(frozen=True)
class Timing:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float


def _stats(values_ms: list[float]) -> Timing:
    values = sorted(values_ms)
    n = len(values)
    return Timing(
        mean_ms=statistics.mean(values),
        p50_ms=values[n // 2],
        p90_ms=values[min(n - 1, int(math.ceil(0.9 * n)) - 1)],
        min_ms=values[0],
        max_ms=values[-1],
    )


def _run_on(fn: Callable[[], None], stream: Optional[torch.cuda.Stream]) -> None:
    if stream is None:
        fn()
    else:
        with torch.cuda.stream(stream):
            fn()


def time_events(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> Timing:
    """Time fn with CUDA events on one stream."""
    for _ in range(warmup):
        _run_on(fn, stream)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(stream)
        _run_on(fn, stream)
        ends[i].record(stream)
    torch.cuda.synchronize()
    return _stats([s.elapsed_time(e) for s, e in zip(starts, ends)])


def time_two_streams(
    work_fn: Callable[[], None],
    copy_fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> Timing:
    """Time work_fn and copy_fn launched concurrently on two side streams."""
    work_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()

    for _ in range(warmup):
        with torch.cuda.stream(copy_stream):
            copy_fn()
        with torch.cuda.stream(work_stream):
            work_fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    current = torch.cuda.current_stream()
    for i in range(iters):
        starts[i].record(current)
        copy_stream.wait_event(starts[i])
        work_stream.wait_event(starts[i])
        with torch.cuda.stream(copy_stream):
            copy_fn()
        with torch.cuda.stream(work_stream):
            work_fn()
        current.wait_stream(copy_stream)
        current.wait_stream(work_stream)
        ends[i].record(current)
    torch.cuda.synchronize()
    return _stats([s.elapsed_time(e) for s, e in zip(starts, ends)])


def time_pipeline(
    indexer_then_topk_fn: Callable[[], None],
    copy_fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> Timing:
    """Time the prefetch-shaped pipeline: copy overlaps indexer + top-k."""
    work_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()

    for _ in range(warmup):
        with torch.cuda.stream(copy_stream):
            copy_fn()
        with torch.cuda.stream(work_stream):
            indexer_then_topk_fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    current = torch.cuda.current_stream()
    for i in range(iters):
        starts[i].record(current)
        copy_stream.wait_event(starts[i])
        work_stream.wait_event(starts[i])
        with torch.cuda.stream(copy_stream):
            copy_fn()
        with torch.cuda.stream(work_stream):
            indexer_then_topk_fn()
        current.wait_stream(copy_stream)
        current.wait_stream(work_stream)
        ends[i].record(current)
    torch.cuda.synchronize()
    return _stats([s.elapsed_time(e) for s, e in zip(starts, ends)])


@dataclass
class Config:
    context_len: int
    batch_size: int
    c4_len: int
    max_pages: int


def make_config(context_len: int, batch_size: int) -> Config:
    c4_len = max(1, (context_len + 3) // 4)
    max_pages = max(1, (c4_len + C4_PAGE_SIZE - 1) // C4_PAGE_SIZE)
    return Config(
        context_len=context_len,
        batch_size=batch_size,
        c4_len=c4_len,
        max_pages=max_pages,
    )


def make_indexer_inputs(cfg: Config, device: str):
    q = torch.randn(
        cfg.batch_size,
        1,
        INDEX_N_HEADS,
        INDEX_HEAD_DIM,
        device=device,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    # Zero KV avoids NaN/Inf FP8 patterns; the kernel shapes and memory traffic
    # are unchanged, and top-k still receives a full logits matrix.
    kv = torch.zeros(
        (cfg.max_pages, C4_PAGE_SIZE, 1, INDEX_HEAD_DIM + 4),
        device=device,
        dtype=torch.uint8,
    )
    weight = torch.randn(
        cfg.batch_size, INDEX_N_HEADS, device=device, dtype=torch.float32
    )
    seq_lens = torch.full(
        (cfg.batch_size,), cfg.c4_len, device=device, dtype=torch.int32
    )
    page_table = (
        torch.arange(cfg.max_pages, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(cfg.batch_size, 1)
    )
    return q, kv, weight, seq_lens, page_table


def make_topk_buffers(cfg: Config, device: str):
    out_page = torch.empty(
        cfg.batch_size, INDEX_TOPK, device=device, dtype=torch.int32
    )
    out_raw = torch.empty_like(out_page)
    return out_page, out_raw


def make_copy_buffers(cfg: Config, device: str, copy_rows: int):
    # Full host KV for the C4 layer, selected rows, and the device scratch.
    host_full = torch.empty(
        cfg.batch_size,
        cfg.c4_len,
        C4_ITEM_BYTES,
        dtype=torch.uint8,
        pin_memory=True,
    )
    host_stage = torch.empty(
        cfg.batch_size,
        copy_rows,
        C4_ITEM_BYTES,
        dtype=torch.uint8,
        pin_memory=True,
    )
    device_stage = torch.empty(
        cfg.batch_size,
        copy_rows,
        C4_ITEM_BYTES,
        device=device,
        dtype=torch.uint8,
    )
    return host_full, host_stage, device_stage


def cpu_gather_selected(
    host_full: torch.Tensor,
    host_stage: torch.Tensor,
    idx_exp: torch.Tensor,
) -> None:
    """Gather selected token rows into a pinned staging buffer on the CPU."""
    try:
        torch.gather(host_full, 1, idx_exp, out=host_stage)
    except (TypeError, RuntimeError):
        # Older torch versions do not accept out= for gather.
        host_stage.copy_(torch.gather(host_full, 1, idx_exp))


def measure_config(
    cfg: Config,
    *,
    device: str,
    indexer_fn,
    topk_fn,
    warmup: int,
    iters: int,
    copy_rows: int = INDEX_TOPK,
) -> dict:
    q, kv, weight, seq_lens, page_table = make_indexer_inputs(cfg, device)
    out_page, out_raw = make_topk_buffers(cfg, device)

    def run_indexer():
        return indexer_fn(
            q,
            kv,
            weight,
            seq_lens,
            page_table,
            None,
            cfg.c4_len,
            True,
        )

    # Warm up JIT compilation and capture one set of selected indices.
    logits = run_indexer()
    topk_fn(logits, seq_lens, page_table, out_page, C4_PAGE_SIZE, out_raw)
    torch.cuda.synchronize()

    t_indexer = time_events(run_indexer, warmup=warmup, iters=iters)

    # Top-k timing uses a fixed logits tensor, so it isolates the transform.
    topk_fn(logits, seq_lens, page_table, out_page, C4_PAGE_SIZE, out_raw)
    t_topk = time_events(
        lambda: topk_fn(logits, seq_lens, page_table, out_page, C4_PAGE_SIZE, out_raw),
        warmup=warmup,
        iters=iters,
    )

    host_full, host_stage, device_stage = make_copy_buffers(cfg, device, copy_rows)
    # The gather runs on the CPU against pinned host memory; keep the index
    # tensor on the CPU as well.  copy_rows > topk simulates the extra rows a
    # threshold prefetch copies when its candidate set is larger than topk.
    idx = out_raw.long().clamp_min_(0).cpu()
    if copy_rows <= idx.shape[1]:
        idx = idx[:, :copy_rows]
    else:
        pad = idx[:, :1].repeat(1, copy_rows - idx.shape[1])
        idx = torch.cat([idx, pad], dim=1)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, C4_ITEM_BYTES).contiguous()

    # Fill the pinned staging buffer once; the memcpy timing is pure H2D.
    cpu_gather_selected(host_full, host_stage, idx_exp)
    torch.cuda.synchronize()
    t_copy_memcpy = time_events(
        lambda: device_stage.copy_(host_stage, non_blocking=True),
        warmup=warmup,
        iters=iters,
    )

    # CPU gather + H2D, measured on the wall clock (host work is part of it).
    def gather_and_copy():
        cpu_gather_selected(host_full, host_stage, idx_exp)
        device_stage.copy_(host_stage, non_blocking=True)

    for _ in range(warmup):
        gather_and_copy()
    torch.cuda.synchronize()
    gather_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        gather_and_copy()
        torch.cuda.synchronize()
        gather_times.append((time.perf_counter() - t0) * 1e3)
    t_copy_gather = _stats(gather_times)

    # Concurrent indexer + pure H2D copy, and the full prefetch-shaped pipeline.
    t_indexer_copy = time_two_streams(
        run_indexer,
        lambda: device_stage.copy_(host_stage, non_blocking=True),
        warmup=warmup,
        iters=iters,
    )

    def indexer_then_topk():
        logits_local = run_indexer()
        topk_fn(
            logits_local,
            seq_lens,
            page_table,
            out_page,
            C4_PAGE_SIZE,
            out_raw,
        )

    t_pipeline = time_pipeline(
        indexer_then_topk,
        lambda: device_stage.copy_(host_stage, non_blocking=True),
        warmup=warmup,
        iters=iters,
    )

    window_ms = t_indexer.mean_ms + t_topk.mean_ms
    base_ms = window_ms + t_copy_memcpy.mean_ms
    ideal_ms = max(window_ms, t_copy_memcpy.mean_ms)
    saved_ideal_ms = base_ms - ideal_ms
    overlap_denom = min(t_indexer.mean_ms, t_copy_memcpy.mean_ms)
    overlap_eff = (
        (t_indexer.mean_ms + t_copy_memcpy.mean_ms - t_indexer_copy.mean_ms)
        / overlap_denom
        if overlap_denom > 0
        else 0.0
    )

    bytes_per_layer = cfg.batch_size * copy_rows * C4_ITEM_BYTES
    bandwidth_gbps = (
        bytes_per_layer / 1e9 / (t_copy_memcpy.mean_ms / 1e3)
        if t_copy_memcpy.mean_ms > 0
        else 0.0
    )

    return {
        "context_len": cfg.context_len,
        "c4_len": cfg.c4_len,
        "batch_size": cfg.batch_size,
        "topk": INDEX_TOPK,
        "copy_rows": copy_rows,
        "item_bytes": C4_ITEM_BYTES,
        "copy_bytes_per_layer": bytes_per_layer,
        "t_indexer_ms": t_indexer.mean_ms,
        "t_topk_ms": t_topk.mean_ms,
        "t_copy_memcpy_ms": t_copy_memcpy.mean_ms,
        "t_copy_gather_ms": t_copy_gather.mean_ms,
        "t_window_ms": window_ms,
        "t_base_seq_ms": base_ms,
        "t_ideal_prefetch_ms": ideal_ms,
        "t_saved_ideal_ms": saved_ideal_ms,
        "t_saved_ideal_pct": (saved_ideal_ms / base_ms * 100.0) if base_ms else 0.0,
        "copy_to_window_ratio": (t_copy_memcpy.mean_ms / window_ms) if window_ms else 0.0,
        "t_indexer_copy_ms": t_indexer_copy.mean_ms,
        "overlap_efficiency": max(0.0, min(1.0, overlap_eff)),
        "t_pipeline_ms": t_pipeline.mean_ms,
        "t_pipeline_saved_ms": base_ms - t_pipeline.mean_ms,
        "t_pipeline_saved_pct": ((base_ms - t_pipeline.mean_ms) / base_ms * 100.0)
        if base_ms
        else 0.0,
        "copy_bandwidth_gbps": bandwidth_gbps,
        "c4_layers": 0,
        "saved_ms_per_step_est": 0.0,
        "indexer_p50_ms": t_indexer.p50_ms,
        "indexer_p90_ms": t_indexer.p90_ms,
        "topk_p50_ms": t_topk.p50_ms,
        "topk_p90_ms": t_topk.p90_ms,
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "ctx",
        "c4_len",
        "batch",
        "T_idx",
        "T_topk",
        "T_copy",
        "T_window",
        "copy/window",
        "saved_ms",
        "saved_%",
        "overlap",
        "pipe_saved_%",
        "e2e_%",
    ]
    print("\nPer-C4-layer timing (ms unless noted). Copy assumes 100% miss.")
    print(
        f"{headers[0]:>6} {headers[1]:>7} {headers[2]:>5} "
        f"{headers[3]:>7} {headers[4]:>7} {headers[5]:>7} {headers[6]:>8} "
        f"{headers[7]:>11} {headers[8]:>9} {headers[9]:>8} {headers[10]:>8} "
        f"{headers[11]:>10} {headers[12]:>8}"
    )
    for row in rows:
        print(
            f"{row['context_len']:>6} {row['c4_len']:>7} {row['batch_size']:>5} "
            f"{row['t_indexer_ms']:>7.3f} {row['t_topk_ms']:>7.3f} "
            f"{row['t_copy_memcpy_ms']:>7.3f} {row['t_window_ms']:>8.3f} "
            f"{row['copy_to_window_ratio']:>11.2f} {row['t_saved_ideal_ms']:>9.3f} "
            f"{row['t_saved_ideal_pct']:>7.2f}% {row['overlap_efficiency']:>7.2f} "
            f"{row['t_pipeline_saved_pct']:>9.2f}% {row['expected_e2e_pct']:>7.2f}%"
        )


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts",
        default="8192,32768,65536",
        help="Comma-separated full context lengths (not C4 lengths).",
    )
    parser.add_argument(
        "--batches",
        default="1,8,16",
        help="Comma-separated batch sizes (requests per decode step).",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--copy-rows",
        type=int,
        default=INDEX_TOPK,
        help="Rows copied per request/layer; >topk simulates threshold overshoot.",
    )
    parser.add_argument(
        "--c4-layers",
        type=int,
        default=21,
        help="Number of C4 layers (DeepSeek-V4-Flash default: 21).",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=20.9,
        help="Decode step time for this batch; used for expected_e2e_pct.",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Directory for CSV/JSON output.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this probe.")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9):
        print(
            f"WARNING: current device capability is {capability}, this probe is "
            "tuned for SM89 (RTX 4090).",
            file=sys.stderr,
        )
    print(
        f"GPU: {torch.cuda.get_device_name()}  capability={capability}  "
        f"torch={torch.__version__}"
    )

    try:
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
        from sglang.srt.layers.attention.dsv4.triton_paged_mqa_logits import (
            fp8_paged_mqa_logits_triton_sm89,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Could not import the DSV4 SM89 kernels. Run this from a checkout "
            "with PYTHONPATH=<repo>/python and a built sglang environment.\n"
            f"Original error: {exc}"
        )

    context_lens = parse_int_list(args.contexts)
    batch_sizes = parse_int_list(args.batches)
    rows = []
    for context_len in context_lens:
        for batch_size in batch_sizes:
            cfg = make_config(context_len, batch_size)
            print(
                f"\nRunning ctx={context_len} (c4={cfg.c4_len}) batch={batch_size} "
                f"topk={INDEX_TOPK} warmup={args.warmup} iters={args.iters}"
            )
            try:
                row = measure_config(
                    cfg,
                    device=args.device,
                    indexer_fn=fp8_paged_mqa_logits_triton_sm89,
                    topk_fn=topk_transform_512,
                    warmup=args.warmup,
                    iters=args.iters,
                    copy_rows=max(1, args.copy_rows),
                )
            except Exception as exc:
                print(f"SKIP ctx={context_len} batch={batch_size}: {exc}", file=sys.stderr)
                continue
            c4_layers = max(0, args.c4_layers)
            row["c4_layers"] = c4_layers
            row["step_ms"] = args.step_ms
            row["saved_ms_per_step_est"] = row["t_pipeline_saved_ms"] * c4_layers
            row["ideal_saved_ms_per_step"] = row["t_saved_ideal_ms"] * c4_layers
            row["expected_e2e_pct"] = (
                row["saved_ms_per_step_est"] / args.step_ms * 100.0
                if args.step_ms > 0
                else 0.0
            )
            row["ideal_e2e_pct"] = (
                row["ideal_saved_ms_per_step"] / args.step_ms * 100.0
                if args.step_ms > 0
                else 0.0
            )
            rows.append(row)
            print(
                f"  T_idx={row['t_indexer_ms']:.3f}ms  "
                f"T_topk={row['t_topk_ms']:.3f}ms  "
                f"T_copy={row['t_copy_memcpy_ms']:.3f}ms  "
                f"copy/window={row['copy_to_window_ratio']:.2f}  "
                f"saved={row['t_saved_ideal_ms']:.3f}ms "
                f"({row['t_saved_ideal_pct']:.1f}%)  "
                f"e2e~{row['expected_e2e_pct']:.1f}%"
            )

    if not rows:
        raise SystemExit("No configurations completed; see the errors above.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"stages_{stamp}.csv"
    json_path = out_dir / f"stages_{stamp}.json"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print_table(rows)
    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
