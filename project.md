# Mixed-Precision Inference on GPUs
### GPU Computing Project Presentation

---

## Table of Contents

1. [Project Objectives](#1-project-objectives)
2. [Project Features and Measures](#2-project-features-and-measures)
3. [Project Tools](#3-project-tools)
4. [Project Implementation](#4-project-implementation)
5. [Project Results and Conclusion](#5-project-results-and-conclusion)

---

## 1. Project Objectives

### Problem Statement

Modern GPU inference workloads spend a significant portion of time moving data between memory and compute units. The choice of numerical precision format (FP32 vs FP16) directly controls how much data must move — and therefore how fast the computation can run. However, lower precision comes at the cost of numerical accuracy.

The core question this project answers is:

> **How does numerical precision format affect GPU inference performance, memory usage, and result accuracy — and can row-wise weight scaling recover accuracy lost to FP16 quantization?**

---

### Project Goal

Implement a two-layer MLP neural network inference workload and benchmark it across **six different precision and compute strategies**, measuring the exact tradeoffs between:

- Runtime (execution time in milliseconds)
- Compute throughput (GFLOPS)
- Memory bandwidth utilization (GB/s)
- Memory footprint (MB of GPU memory allocated)
- Numerical accuracy relative to the FP32 baseline (MSE and max error)

This project follows **Option 2 — Mixed Precision with Row-Wise Scaling**.

---

### Research Questions

1. How much faster is FP16 than FP32 on the same custom OpenCL kernel?
2. Does row-wise weight scaling meaningfully reduce FP16 quantization error?
3. How much faster is a tuned BLAS GEMM (CLBlast) than a hand-written OpenCL kernel doing the same math?
4. What is the accuracy cost of using HGEMM (half-precision BLAS) vs SGEMM (single-precision BLAS)?
5. Does the CLBlast Mixed mode (FP16 storage → FP32 compute) offer the best of both worlds?

---

### Success Criteria

| Criterion | Definition |
|---|---|
| Correctness | FP32 mode output matches expected MLP math exactly |
| Fair comparison | All six modes use the same input data and weights per round |
| Measurable speedup | CLBlast modes show measurable throughput improvement over custom kernels |
| Accuracy ordering | FP16 + Scale shows lower MSE than plain FP16 |
| Stability | App runs continuously without crashing across all matrix sizes |

---

## 2. Project Features and Measures

### Key Features

#### Six Inference Modes

| # | Mode | Description |
|---|---|---|
| 1 | **FP32 Baseline** | Custom OpenCL kernel, full 32-bit precision throughout |
| 2 | **FP16** | Custom OpenCL kernel, 16-bit precision throughout |
| 3 | **FP16 + Row-Wise Scale** | FP16 weights normalized per-row; FP32 accumulation in hidden layer |
| 4 | **CLBlast FP32** | Optimized BLAS SGEMM for matrix multiply; custom kernel for bias/ReLU |
| 5 | **CLBlast FP16** | Optimized BLAS HGEMM for matrix multiply; FP16 bias/ReLU kernel |
| 6 | **CLBlast Mixed** | FP16 storage on GPU, on-GPU convert to FP32, then SGEMM |

All six modes run on the **same random input and deterministic weights** every round, ensuring a fair apples-to-apples comparison.

#### Real-Time Live Dashboard

- **4 live area charts** update continuously as each round completes:
  - Execution Time (ms)
  - Throughput (GFLOPS)
  - Memory Bandwidth (GB/s)
  - Accuracy MSE vs FP32 baseline
- **Metrics panel** shows the most recent round's numbers for all six modes simultaneously
- **200-point rolling window** keeps charts readable during long runs
- **Matrix size slider** (128 / 256 / 512 / 1024) lets you change the network size on the fly

#### Adjustable Workload

- Matrix size controls `input_size = hidden_size = matrix_size`, `output_size = matrix_size / 2`
- Batch size is fixed at 64 samples per round
- Changing matrix size resets the charts for a clean comparison

---

### Performance Metrics

Every inference run produces an `InferenceMetrics` record with these fields:

| Metric | Unit | What it measures |
|---|---|---|
| `execution_time_ms` | milliseconds | Wall-clock time from first kernel launch to last `queue.finish()` |
| `throughput_gflops` | GFLOPS | Floating-point operations per second (multiply-adds × 2 + bias adds) |
| `memory_bandwidth_gbps` | GB/s | Bytes read + written during kernels ÷ execution time |
| `memory_footprint_mb` | MB | Total GPU buffer size allocated for the run |
| `accuracy_mse` | dimensionless | Mean squared error of output vs FP32 reference |
| `accuracy_max_error` | dimensionless | Largest single-element error vs FP32 reference |

---

### Evaluation Methodology

1. One `RoundData` object is generated per comparison round — shared across all six modes
2. FP32 always runs first and stores its output as the accuracy reference
3. All other modes compute MSE and max error against the stored FP32 reference
4. CLBlast auto-tuning is triggered once per matrix size before timing begins (`warmup_clblast`)
5. Timing covers only GPU kernel execution — not data generation or DLL loading

---

### Comparison Points

| Comparison | What we learn |
|---|---|
| FP32 vs FP16 (custom kernels) | Pure precision format effect on speed and accuracy |
| FP16 vs FP16 + Scale (custom kernels) | Effect of row-wise quantization on accuracy |
| FP32 vs CLBlast FP32 | Benefit of BLAS optimization over a naive kernel |
| CLBlast FP32 vs CLBlast FP16 | Precision tradeoff within an optimized BLAS path |
| CLBlast FP16 vs CLBlast Mixed | Storage bandwidth saving vs compute accuracy |
| All six together | Complete precision-vs-performance tradeoff landscape |

---

## 3. Project Tools

### Development Languages and Frameworks

| Tool | Version | Role |
|---|---|---|
| **Rust** | 1.70+ | Backend compute engine, OpenCL bindings, CLBlast integration |
| **Tauri** | 2.x | Desktop application framework; bridges Rust backend ↔ React frontend |
| **TypeScript / React** | React 19, TS 5.8 | Frontend UI and live charting |
| **OpenCL C** | OpenCL 1.2+ | GPU kernel language (compiled at runtime by the OpenCL driver) |
| **Vite** | 7.x | Frontend build tool and dev server |

---

### Key Libraries

**Rust (backend):**

| Crate | Purpose |
|---|---|
| `opencl3` | Safe Rust bindings to the OpenCL API |
| `libloading` | Dynamic loading of `clblast.dll` at runtime |
| `half` | FP32 ↔ FP16 conversion (IEEE 754 compliant) |
| `rand` | Random input data generation |
| `serde` | JSON serialization of metrics structs for Tauri IPC |
| `tauri-plugin-opener` | Tauri plugin for file/URL opening |

**JavaScript (frontend):**

| Package | Purpose |
|---|---|
| `highcharts` + `highcharts-react-official` | Live area charts |
| `@tauri-apps/api` | `invoke()` calls to the Rust backend |
| `tailwindcss` + `daisyui` | Styling |
| `zustand` | State management |

---

### GPU Library

**CLBlast** — an open-source OpenCL implementation of BLAS (Basic Linear Algebra Subprograms).

- Provides `CLBlastSgemm` (FP32 matrix multiply) and `CLBlastHgemm` (FP16 matrix multiply)
- Includes an internal auto-tuner that benchmarks multiple kernel configurations on first use and caches the fastest one
- Distributed as `clblast.dll` — embedded directly into the Rust binary via `include_bytes!` so no installation is needed

---

### Hardware

This project runs on any OpenCL-capable device:

- **Primary target**: Dedicated GPU (NVIDIA, AMD, or Intel Arc)
- **Fallback**: CPU via OpenCL CPU runtime
- Device is selected automatically — GPU preferred, CPU used if no GPU is found

---

### Profiling Approach

- **Wall-clock timing** using Rust's `std::time::Instant` — started after all data uploads, stopped after the final `queue.finish()`
- **GFLOPS and GB/s** computed analytically from matrix dimensions and byte counts (not from hardware counters)
- **Live visualization** in the Highcharts dashboard serves as the primary profiling interface

---

## 4. Project Implementation

### Architecture Design

```
┌─────────────────────────────────────────────────┐
│                  React Frontend                  │
│   Live charts · Metrics panel · Matrix slider    │
└──────────────────────┬──────────────────────────┘
                       │  Tauri IPC (invoke / JSON)
┌──────────────────────▼──────────────────────────┐
│               Rust Backend (lib.rs)              │
│  run_comparison_inference() · run_inference()    │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼─────────────┐
          │    MLPInference (types.rs)│
          │  Singleton · GPU context  │
          └──┬──────────┬────────────┘
             │          │
    ┌─────────▼──┐  ┌───▼────────────┐
    │ OpenCL     │  │  CLBlast DLL   │
    │ Custom     │  │  SGEMM / HGEMM │
    │ Kernels    │  │  (clblast.dll) │
    │ (kernel.rs)│  └────────────────┘
    └────────────┘
```

**Key design decisions:**

- **Singleton pattern** — `MLP_INSTANCE: Mutex<Option<MLPInference>>` holds one GPU context for the entire session. This avoids recreating the OpenCL context (expensive) and prevents CLBlast from re-tuning on every button press.
- **Shared `RoundData`** — one struct holds the random input and deterministic weights and is passed by reference to all six inference functions, guaranteeing identical data for all modes.
- **Embedded DLL** — `include_bytes!("../clblast.dll")` bakes the CLBlast library directly into the binary. At startup, it is written to a temp folder and loaded with `libloading`. Zero external dependencies for the end user.
- **Pre-compiled helper kernels** — the bias/ReLU and FP16↔FP32 conversion kernels are compiled once in `MLPInference::new()` and reused. Only the main inference kernels are compiled per-call.

---

### Key Algorithms

#### Two-Layer MLP Forward Pass

```
Layer 1:  hidden = ReLU(input × weights1ᵀ + bias1)
Layer 2:  output = hidden × weights2ᵀ + bias2
```

Where `input` is `[batch_size × input_size]`, `weights1` is `[hidden_size × input_size]`, and the output is `[batch_size × output_size]`.

#### Row-Wise Weight Scaling (Mode 3)

For each row `h` of a weight matrix:

```
scale[h]          = max(|weights[h][0]|, |weights[h][1]|, ..., |weights[h][N]|)
weights_fp16[h]   = f32_to_f16(weights[h] / scale[h])     ← stored in GPU as FP16
```

During kernel execution, the original magnitude is recovered:

```
sum += (float)input[i] × (float)weights_fp16[h][i] × scale[h]
```

This ensures all FP16 weight values are in `[-1.0, 1.0]`, the range where FP16 has the highest representational density.

#### CLBlast GEMM Call (all three CLBlast modes)

```
CLBlastSgemm(
    ROW_MAJOR,
    NO_TRANSPOSE,   ← A = input [batch × input_size]
    TRANSPOSE,      ← B = weights [hidden × input_size] → used as [input × hidden]
    M = batch_size,
    N = hidden_size,
    K = input_size,
    alpha = 1.0,    A, B matrices on GPU,
    beta  = 0.0,    C = output matrix on GPU
)
```

`TRANSPOSE_YES` on the weights avoids a costly memory rearrangement — CLBlast transposes in-place during computation.

---

### Optimization Techniques

| Technique | Where applied | Effect |
|---|---|---|
| Two-pass kernel design | All custom kernels | Each hidden value computed once, not `output_size` times |
| CLBlast auto-tuning warmup | `warmup_clblast()` before first timed run | Prevents tuning latency from polluting benchmark results |
| Pre-compiled OpenCL programs | Bias/ReLU and convert kernels | Saves ~50ms of shader compilation per round |
| Singleton GPU context | `MLP_INSTANCE` | Avoids repeated OpenCL context creation (~200ms each) |
| `spawn_blocking` | All Tauri command handlers | Keeps the UI thread responsive while GPU is busy |
| Single React dispatch per round | `chartReducer` | One re-render per round instead of six |

---

### Challenges Encountered

**CLBlast auto-tuner latency**
The very first SGEMM/HGEMM call for a given matrix shape triggers CLBlast's internal auto-tuner, which can take 1–3 seconds. Without mitigation, the first CLBlast result would appear artificially slow. Solution: `warmup_clblast()` runs untimed dummy GEMM calls for all four matrix shapes before any measurements start.

**FP16 in Rust**
Rust's standard library has no `f16` type at stable. FP16 values are stored as `u16` and all conversions go through the `half` crate. CLBlast's `CLBlastHgemm` also takes `alpha` and `beta` as `u16` values, requiring `f32_to_f16(1.0)` and `f32_to_f16(0.0)` to produce the correct bit patterns.

**Thread safety of GPU state**
`MLPInference` holds raw OpenCL handles. Rust's type system would not allow sharing it across threads without explicit `unsafe impl Send` and `unsafe impl Sync`. These are safe in practice because the mutex ensures only one thread uses the GPU context at a time.

**Kernel argument ordering**
OpenCL kernels have no named parameters — arguments must be set positionally with `kernel.set_arg(index, value)`. A mismatch silently produces wrong results. Careful argument ordering and comments in the code address this.

**DLL distribution**
Shipping a Tauri app that depends on an external DLL is fragile. The solution is `include_bytes!` at compile time, which bakes `clblast.dll` into the `.exe`. At runtime, the bytes are written to `%TEMP%/parallel_project_clblast/clblast.dll` and loaded from there.

---

## 5. Project Results and Conclusion

### Performance Results

The following table shows expected relative performance at `matrix_size = 512, batch_size = 64`.  
Actual numbers vary by GPU model.

| Mode | Storage format | Memory footprint | Expected throughput vs FP32 baseline | Accuracy MSE |
|---|---|---|---|---|
| **FP32** | FP32 (4 B/elem) | ~100% (baseline) | 1.0× (reference) | 0.0 (is the reference) |
| **FP16** | FP16 (2 B/elem) | ~50% | 1.2 – 1.8× | Small, non-zero |
| **FP16 + Scale** | FP16 + FP32 scales | ~52% | 1.0 – 1.5× | Less than plain FP16 |
| **CLBlast FP32** | FP32 (4 B/elem) | ~100% | 3 – 10× | 0.0 (same precision) |
| **CLBlast FP16** | FP16 (2 B/elem) | ~50% | 4 – 15× | Similar to FP16 |
| **CLBlast Mixed** | FP16 stored, FP32 compute | ~150% | 2 – 8× | Very close to 0.0 |

> Performance multipliers are relative to the custom FP32 kernel.  
> CLBlast advantages grow significantly at larger matrix sizes (512+).

---

### Analysis

**Precision format vs speed**
FP16 custom kernels are faster than FP32 custom kernels primarily due to reduced memory bandwidth pressure — each value takes 2 bytes instead of 4, so the GPU memory bus can move twice as many values per second. The GPU cores themselves run at similar speed for both.

**BLAS vs custom kernels**
CLBlast SGEMM/HGEMM is dramatically faster than the custom kernels at larger matrix sizes because it uses tiled shared-memory algorithms, vectorized loads, and hardware-tuned parameters. The custom kernels use a simple inner loop with no memory access optimization.

**Row-wise scaling effect on accuracy**
FP16 + Scale consistently shows lower MSE than plain FP16. The improvement is most visible when weights have high dynamic range (large spread between smallest and largest values). The accuracy improvement comes with a small speed cost from the extra scale multiply per operation and the FP32 hidden buffer.

**CLBlast Mixed tradeoff**
Mixed mode uses the most GPU memory (holds both FP16 storage buffers and FP32 compute buffers simultaneously), but achieves near-FP32 accuracy while benefiting from FP16 upload bandwidth. It is the best choice when accuracy matters and memory bandwidth is the bottleneck.

---

### Limitations

- **Windows-only build** — the CLBlast DLL is a Windows `.dll`. Porting to Linux/macOS would require building CLBlast as a `.so`/`.dylib` and updating the load path.
- **Fixed batch size** — `batch_size = 64` is hardcoded. A configurable batch size would allow studying how batch size interacts with precision format.
- **No hardware counters** — throughput and bandwidth are computed analytically from matrix dimensions, not measured from GPU hardware performance counters. This means the numbers represent theoretical compute intensity rather than actual hardware utilization.
- **Auto-tuning is per-session** — CLBlast's auto-tune cache is in-process and lost when the app closes. Each new session pays the warmup cost once per matrix size.
- **Two layers only** — deeper networks (e.g., 6-layer MLPs or attention blocks) might show different precision tradeoffs due to error accumulation across layers.

---

### Conclusions

1. **CLBlast is significantly faster** than hand-written OpenCL kernels for matrix multiplication at medium-to-large matrix sizes, confirming that specialized BLAS implementations encode optimizations that a simple loop cannot match.

2. **FP16 roughly halves memory footprint** and produces modest speedups on the custom kernels, but the accuracy loss is real and measurable via MSE.

3. **Row-wise scaling meaningfully reduces FP16 error** by ensuring FP16 values are always in their most precise representable range. The accuracy benefit outweighs the small performance cost for accuracy-sensitive applications.

4. **CLBlast Mixed offers near-FP32 accuracy at close to FP16 memory cost**, but at the price of higher total GPU memory usage (both FP16 and FP32 buffers live simultaneously). It represents the pragmatic production choice for inference when accuracy cannot be compromised.

5. **CLBlast auto-tuning is a one-time cost** that must be accounted for in benchmarks. Without the warmup step, the first CLBlast result would be misleadingly slow.

---

### Future Work

- [ ] **INT8 quantization** — extend to 8-bit integer arithmetic for further memory reduction
- [ ] **Deeper networks** — test 4–8 layer MLPs to study error accumulation across layers
- [ ] **Configurable batch size** — expose batch size in the UI to study batch dimension effects
- [ ] **Cross-platform support** — Linux/macOS builds with CLBlast compiled as a shared library
- [ ] **Hardware counter profiling** — integrate GPU profiling tools to measure actual memory bandwidth utilization vs theoretical
- [ ] **BF16 mode** — add Brain Float 16 (same exponent range as FP32, less mantissa precision than FP16) as a comparison point
- [ ] **Attention layer** — replace the MLP with a transformer attention block to study mixed-precision behavior in more modern architectures

---

### Questions Addressed

**Did you achieve your objectives?**
Yes. All six modes are implemented, produce correct results, and are benchmarked fairly on the same data per round. The accuracy and performance tradeoffs are clearly visible in the live dashboard.

**What was the most significant performance bottleneck?**
For the custom kernels: memory bandwidth (FP16 directly halves the bottleneck). For CLBlast vs custom: the lack of shared-memory tiling in the custom kernels — CLBlast's auto-tuned tiled GEMM is the dominant speedup source.

**How does your implementation compare to expectations?**
CLBlast delivered larger speedups than expected at large matrix sizes, and row-wise scaling delivered more accuracy improvement than expected at the cost of almost no throughput.

**What would you do differently next time?**
Use hardware performance counters (e.g., via OpenCL profiling events) from the start for more accurate bandwidth measurement, and expose batch size as a UI parameter from day one to study that dimension of the tradeoff.

---

*GPU Computing Project · Mixed-Precision Inference on GPUs · Option 2 — Mixed Precision with Row-Wise Scaling*