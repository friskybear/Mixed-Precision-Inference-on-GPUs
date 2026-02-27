# Mixed-Precision Inference on GPUs

> A GPU computing benchmark that runs a two-layer neural network across **6 different precision strategies** simultaneously, measuring speed, memory, and accuracy in real time.

Built with **Rust · OpenCL · CLBlast · Tauri · React · TypeScript**

---

## What This Project Does

This tool runs the same neural network computation six different ways — each using a different numerical precision or library — and displays live charts comparing:

- **Execution Time** (ms)
- **Throughput** (GFLOPS)
- **Memory Bandwidth** (GB/s)
- **Accuracy** (Mean Squared Error vs FP32 baseline)

All six modes use the **same random input and deterministic weights** each round, so the comparison is always fair.

---

## The Six Precision Modes

| # | Mode | Kernel | Precision | Notes |
|---|------|--------|-----------|-------|
| 1 | 🟢 **FP32** | Custom OpenCL | 32-bit float | Gold standard reference |
| 2 | 🔵 **FP16** | Custom OpenCL | 16-bit float | ~50% memory, some accuracy loss |
| 3 | 🟣 **FP16 + Scale** | Custom OpenCL | FP16 weights + FP32 scales | Per-row quantization for better accuracy |
| 4 | 🟡 **CLBlast FP32** | BLAS SGEMM | 32-bit float | Optimized matrix multiply |
| 5 | 🩵 **CLBlast FP16** | BLAS HGEMM | 16-bit float | Optimized half-precision BLAS |
| 6 | 🩷 **CLBlast Mixed** | BLAS SGEMM | FP16 stored → FP32 compute | Best of both worlds |

---

## Quick Start

### Prerequisites

- OpenCL-capable GPU (NVIDIA, AMD, or Intel) with up-to-date drivers
- [Rust](https://rustup.rs/) 1.70+
- [Node.js](https://nodejs.org/) 18+ **or** [Bun](https://bun.sh/)

### Run in Development

```bash
# Install frontend dependencies
npm install        # or: bun install

# Start the app (compiles Rust + launches Vite + opens the window)
npm run tauri dev  # or: bun tauri dev
```

The first launch is slower — OpenCL kernels compile and CLBlast auto-tunes for the selected matrix size. Subsequent launches at the same matrix size are fast.

### Build for Release

```bash
npm run tauri build
```

The output is a self-contained installer in `src-tauri/target/release/bundle/`. The CLBlast DLL is embedded inside the binary — no separate installation needed.

---

## Using the App

1. **Select a matrix size** with the slider: `128 · 256 · 512 · 1024`
   - This sets `input_size = hidden_size = matrix_size`, `output_size = matrix_size / 2`
   - Larger sizes show bigger performance differences between modes
2. **Press ▶ Play** to start continuous benchmarking
3. **Watch the charts** update live — each iteration runs all six modes back-to-back
4. **Press ⏹ Stop** at any time
5. Changing the matrix size resets the charts automatically

> **Tip:** The very first round after launching (or after changing matrix size) is slower than subsequent rounds because CLBlast runs its internal auto-tuner. This is expected and normal.

---

## Automatic Logging & Plotting

Every **5 rounds**, the app automatically saves a complete snapshot of all metrics collected so far:

| File | Contents |
|------|----------|
| `metrics.csv` | All metrics from round 1 to current, all 6 modes, all fields |
| `execution_time.png` | Line chart comparing execution time across all modes |
| `throughput.png` | GFLOPS comparison |
| `bandwidth.png` | Memory bandwidth comparison |
| `accuracy_mse.png` | Accuracy MSE vs FP32 baseline (FP16, FP16+Scale, CLBlast FP16, CLBlast Mixed) |

**Log path format:**
```
parallel_log/{YYYY-MM-DD_HH-MM-SS}_{matrix_size}/
├── metrics.csv
├── execution_time.png
├── throughput.png
├── bandwidth.png
└── accuracy_mse.png
```

- A **new session folder** is created each time logging triggers (timestamped + matrix size).
- Changing the matrix size resets the metrics accumulator, so the next session starts fresh.
- Charts are generated server-side in Rust using the **plotters** crate — no browser or external tools needed.
- Colors in the PNG charts match the live dashboard (green/blue/purple/yellow/cyan/pink).

---

## Architecture

```
┌──────────────────────────────────────────┐
│            React + TypeScript             │
│  Live charts · Metrics panel · Controls  │
└────────────────────┬─────────────────────┘
                     │  Tauri IPC (invoke / JSON)
┌────────────────────▼─────────────────────┐
│           Rust Backend (lib.rs)           │
│  run_comparison_inference()              │
│  run_inference()  ·  get_len()           │
└────────────────────┬─────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │   MLPInference (types.rs) │
        │   Singleton · GPU state   │
        └──────┬──────────┬─────────┘
               │          │
     ┌──────────▼──┐  ┌───▼──────────────┐
     │ OpenCL      │  │  CLBlast (DLL)   │
     │ Custom      │  │  SGEMM / HGEMM   │
     │ Kernels     │  │  embedded in .exe│
     │ (kernel.rs) │  └──────────────────┘
     └─────────────┘
```

### Backend — Rust (`src-tauri/src/`)

| File | Responsibility |
|------|----------------|
| `lib.rs` | Tauri commands, DLL loading, device selection, float helpers |
| `types.rs` | All structs + all six inference functions (the heavy math) |
| `kernel.rs` | OpenCL C kernel source code (compiled at runtime by the GPU driver) |
| `logger.rs` | CSV export + PNG chart generation (plotters) for automatic logging |
| `main.rs` | Minimal entry point — calls `lib::run()` |

**Key design decisions:**

- **Singleton GPU context** — `MLP_INSTANCE: Mutex<Option<MLPInference>>` holds one OpenCL context for the entire session, preventing repeated initialization and CLBlast re-tuning.
- **Embedded DLL** — `clblast.dll` is baked into the binary via `include_bytes!` and extracted to a temp folder on first run. Zero external dependencies.
- **Two-pass kernel design** — Layer 1 and Layer 2 are separate kernel dispatches with a `queue.finish()` barrier between them. This ensures each hidden neuron is computed exactly once, not redundantly per output neuron.
- **Pre-compiled helper kernels** — the bias/ReLU and FP16↔FP32 conversion kernels are compiled once at startup and reused across all rounds.

### Frontend — React + TypeScript (`src/`)

| File | Responsibility |
|------|----------------|
| `App.tsx` | Main UI — state management, inference loop, chart data, layout |
| `component/chart.tsx` | Reusable Highcharts area chart wrapper |

The inference loop is a self-chaining async function that calls `invoke("run_comparison_inference")`, dispatches a single `PUSH` action to update all chart series in one React re-render, yields to the browser for a frame, then loops.

---

## Implementation Details

### The Neural Network Workload

A two-layer MLP (Multi-Layer Perceptron):

```
Input  [batch=64 × input_size]
  ↓  × weights1ᵀ [hidden_size × input_size]
  ↓  + bias1  →  ReLU
Hidden [batch=64 × hidden_size]
  ↓  × weights2ᵀ [output_size × hidden_size]
  ↓  + bias2
Output [batch=64 × output_size]
```

- `batch_size` is fixed at **64**
- `input_size = hidden_size = matrix_size` (from slider)
- `output_size = matrix_size / 2`

### Row-Wise Scaling (Mode 3)

Before converting weights to FP16, each row is normalized so its maximum absolute value is 1.0:

```
scale[h]        = max(|weights1[h][:]|)
weights_fp16[h] = weights1[h] / scale[h]   ← stored in GPU as FP16
```

During the kernel, the scale is re-applied to recover the original magnitude:

```c
sum += (float)input[i] * (float)weights_fp16[h][i] * scale[h];
```

This keeps all FP16 values in `[-1.0, 1.0]` — the range where FP16 has the most precision. The hidden buffer is kept as FP32 to avoid a second round of quantization error.

### CLBlast Integration (Modes 4–6)

CLBlast's GEMM functions handle the matrix multiplication. Because BLAS doesn't know about neural network biases or activation functions, small helper OpenCL kernels handle those steps:

- `add_bias_relu_fp32/fp16` — adds bias and applies `ReLU` after Layer 1
- `add_bias_fp32/fp16` — adds bias only (no activation) after Layer 2

The CLBlast Mixed mode additionally runs `convert_fp16_to_fp32` kernels on the GPU to expand FP16-stored weights into FP32 buffers before passing them to SGEMM.

---

## Performance Metrics

All metrics are calculated per inference run:

**Throughput (GFLOPS)**
```
total_flops = batch × [hidden × (2×input + 1) + output × (2×hidden + 1)]
throughput  = total_flops / time_seconds / 1e9
```

**Memory Bandwidth (GB/s)**
```
bytes_transferred = Σ (bytes read + bytes written) per kernel
bandwidth         = bytes_transferred / time_seconds / 1e9
```
FP16 buffers count 2 bytes/element; FP32 buffers count 4 bytes/element.

**Accuracy (MSE)**
```
mse = Σ (output[i] - fp32_reference[i])² / N
```
FP32 and CLBlast FP32 always report MSE = 0.0 — they are the reference or compute equivalent precision.

---

## Expected Results

Performance varies significantly by GPU. These are approximate relative values at `matrix_size = 512`:

| Mode | Memory Footprint | Relative Throughput | Accuracy MSE |
|------|-----------------|---------------------|--------------|
| FP32 | 100% (baseline) | 1.0× | 0.0 (reference) |
| FP16 | ~50% | 1.2 – 1.8× | Small, non-zero |
| FP16 + Scale | ~52% | 1.0 – 1.5× | Better than FP16 |
| CLBlast FP32 | ~100% | 3 – 10× | 0.0 |
| CLBlast FP16 | ~50% | 4 – 15× | Similar to FP16 |
| CLBlast Mixed | ~150% | 2 – 8× | Very close to 0.0 |

> CLBlast advantages grow at larger matrix sizes. At `matrix_size = 128`, the custom kernels may be competitive or faster due to CLBlast overhead.

---

## Project Structure

```
Mixed-Precision-Inference-on-GPUs/
├── src/
│   ├── App.tsx                  # Main UI component and inference loop
│   ├── component/
│   │   └── chart.tsx            # Reusable Highcharts area chart
│   └── main.tsx                 # React entry point
├── src-tauri/
│   ├── src/
│   │   ├── lib.rs               # Tauri commands, DLL loading, device selection
│   │   ├── types.rs             # Structs + all six inference functions
│   │   ├── kernel.rs            # OpenCL C kernel source (as string constants)
│   │   ├── logger.rs            # CSV + PNG chart generation (plotters crate)
│   │   └── main.rs              # Binary entry point
│   ├── clblast.dll              # CLBlast library (embedded into binary at compile time)
│   ├── Cargo.toml               # Rust dependencies
│   └── tauri.conf.json          # Tauri app configuration
├── parallel_log/                # Auto-generated logs (created at runtime)
│   └── {timestamp}_{size}/      # One folder per logging session
│       ├── metrics.csv
│       ├── execution_time.png
│       ├── throughput.png
│       ├── bandwidth.png
│       └── accuracy_mse.png
├── presentation/
│   └── presentation.tex         # LaTeX Beamer presentation
├── CODE_EXPLANATION.md          # Deep dive into every file, struct, and function
├── project.md                   # Project presentation (objectives, results, analysis)
├── package.json                 # Node dependencies
└── vite.config.ts               # Vite build configuration
```

---

## Dependencies

### Rust

| Crate | Version | Purpose |
|-------|---------|---------|
| `tauri` | 2.x | Desktop app framework |
| `opencl3` | 0.12 | OpenCL bindings |
| `libloading` | 0.9 | Dynamic DLL loading |
| `half` | 2.x | FP16 ↔ FP32 conversion |
| `rand` | 0.10 | Random input generation |
| `serde` | 1.x | JSON serialization for Tauri IPC |
| `plotters` | 0.3 | PNG chart generation for automatic logging |
| `chrono` | 0.4 | Timestamps for log folder names |

### Node / Frontend

| Package | Purpose |
|---------|---------|
| `react` 19 | UI framework |
| `highcharts` + `highcharts-react-official` | Live area charts |
| `@tauri-apps/api` | `invoke()` bridge to Rust |
| `tailwindcss` + `daisyui` | Styling |
| `typescript` 5.8 | Type safety |
| `vite` 7 | Build tool |

---

## Troubleshooting

**No OpenCL devices found**
Install GPU drivers that include OpenCL support. For NVIDIA: CUDA Toolkit or GeForce drivers. For AMD: ROCm or Adrenalin drivers. For Intel: Intel oneAPI runtime.

**CLBlast modes show 0.0 / fail silently**
Some older GPUs don't support FP16 BLAS operations (HGEMM). The CLBlast FP16 and Mixed modes fall back gracefully to zero metrics instead of crashing.

**First round is very slow**
Expected. OpenCL kernels compile at runtime (~100–500ms) and CLBlast auto-tunes on the first GEMM call for each matrix size (~1–3 seconds). All subsequent rounds are fast.

**Build fails on Windows**
```bash
cargo clean
npm run tauri build
```
Ensure the Visual Studio C++ build tools are installed (required for Rust on Windows).

**Performance seems low / CPU is being used instead of GPU**
Check that your GPU's OpenCL runtime is installed. The app logs the selected device type at startup. CPU OpenCL is correct but much slower than a GPU.

---

## Further Reading

- [`CODE_EXPLANATION.md`](./CODE_EXPLANATION.md) — Complete explanation of every concept, struct, and function in the codebase, from first principles (what is a batch? what is GEMM? what is a kernel?)
- [`project.md`](./project.md) — Full project presentation covering objectives, implementation, results, and conclusions

---

## License

Academic project for GPU computing coursework.

---

*Built with* **Rust 🦀 · Tauri 🚀 · React ⚛️ · OpenCL · CLBlast**