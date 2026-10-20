# DSV4 prefetch probe

Measure the per-C4-layer decode pipeline on RTX 4090 (SM89):

```
indexer logits -> top-k transform -> KV recall (DDR -> HBM) -> attention
```

The probe imports the production DSV4 SM89 indexer and top-k kernels from the
existing `deepseek-v4-sm89` checkout, then times the recall copy separately and
concurrently. It answers one question before any kernel work: **how much of the
indexer + top-k window could a prefetch hide?**

No install, no model weights, no server. It only needs the existing 4090
environment.

## Run

Two steps on the 4090 machine:

```bash
source /workspace/deepseek-v4-sm89/.venv/bin/activate
cd /path/to/dsv4-prefetch-probe
./run.sh
```

That is the whole workflow. `run.sh` defaults to
`SGLANG_ROOT=/workspace/deepseek-v4-sm89` and
`/workspace/deepseek-v4-sm89/.venv/bin/python`, and runs:

```
--contexts 8192,32768,65536 --batches 8 --warmup 10 --iters 50
```

If the checkout lives elsewhere, override it:

```bash
SGLANG_ROOT=/other/deepseek-v4-sm89 ./run.sh
```

To sweep other shapes:

```bash
./run.sh --contexts 8192,32768,65536 --batches 1,8,16
./run.sh --contexts 8192 --batches 8 --warmup 5 --iters 20   # quick smoke
```

Results are written to `results/stages_<timestamp>.csv` and `.json`.

## What it measures

For each `(context, batch)` pair:

- `T_indexer`: `fp8_paged_mqa_logits_triton_sm89` on a synthetic C4 cache.
  This is the logits kernel only; the real C4 indexer also runs q/k projections
  and normalization, so the true overlap window is larger.
- `T_topk`: `topk_transform_512` on the produced logits.
- `T_copy`: pinned host -> HBM copy of the selected top-k C4 rows
  (`512 * 584` bytes per request per layer).
- `T_window = T_indexer + T_topk`.
- `T_saved_ideal = min(T_copy, T_window)`: per-layer upper bound if the copy is
  fully overlapped and every selected row is a miss.
- `overlap_efficiency`: measured concurrent indexer + copy versus sequential.
- `T_pipeline_saved_pct`: measured prefetch-shaped pipeline saving.
- `expected_e2e_pct`: `T_pipeline_saved_ms × c4_layers / step_ms`. This is the
  deployment-shaped estimate: 21 C4 layers and a 20.9 ms batch-8 step by
  default. It assumes the probe's overlap and a 100% recall miss.
- `ideal_e2e_pct`: the same with `T_saved_ideal_ms` (perfect overlap).

If your design always recalls the selected top-k from DDR (no GPU-side cache
hit), the 100% miss assumption matches the deployment shape, so
`expected_e2e_pct` is the relevant number. If you keep a GPU-side cache,
multiply the saving by the real miss rate. `--copy-rows` defaults to 512; pass
the threshold prefetch's candidate count (for example 768) to include candidate
overshoot.

## How to read it

The key ratio is `copy / window`:

- `< 0.1`: the copy is small relative to indexer + top-k; prefetch can save at
  most a few percent. Stop unless the absolute `saved_ms` still matters.
- `0.3 - 1.0`: there is a real window. `saved_ms` is the per-C4-layer ceiling;
  multiply by the number of C4 layers for the per-decode-step estimate. The
  probe already prints this as `expected_e2e_pct` (21 layers, 20.9 ms step).
- `> 1.0`: the copy is longer than the window. The prefetch can hide the window,
  but some recall latency remains exposed.

`copy/window` uses the optimistic contiguous memcpy. The `gather` columns are a
per-token CPU-gather + H2D number; the real HiSparse kernel may be faster
(GPU gather) or slower (page-granular transfer). If even the optimistic number
does not show headroom, the approach is not worth pursuing in this regime.

Context and batch are inputs, not assumptions. The copy volume is roughly
independent of context; the indexer may or may not grow with the C4 sequence.
In the first 4090 run it was roughly flat from 8K to 64K, so do not assume a
longer context automatically creates a larger window. Batch multiplies both the
copy volume and indexer work; use the batch you actually serve.

`expected_e2e_pct` uses `--c4-layers 21` and `--step-ms 20.9` by default
(DeepSeek-V4-Flash has 21 ratio-4 layers; 20.9 ms is the batch-8 Scheduler step
from the project README). Override both for your deployment:

```bash
./run.sh --contexts 65536 --batches 8 --c4-layers 21 --step-ms 20.9
```

## Caveats

- Kernel-level probe: no CUDA graph, scheduler, attention, MoE or communication.
  `expected_e2e_pct` is therefore a kernel-level estimate; the real server can
  be lower because of CUDA graph, scheduling and the actual recall kernel.
- The host buffer is a dense `[batch, c4_len, 584]` tensor, not the page-padded
  HiSparse layout. Copy bytes are the same; page granularity can change time.
- `T_indexer` is a lower bound on the real indexer stage (no q/k projections).
- The synthetic KV is zeros; it exercises the same shapes and memory traffic,
  not the real value distribution.

## Layout

```
run.sh            # the only command you need
bench_stages.py   # the probe
results/          # generated CSV/JSON (gitignored)
```
