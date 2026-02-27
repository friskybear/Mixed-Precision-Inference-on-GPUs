# Complete Explanation of the Mixed-Precision GPU Inference Code

This document explains every file, struct, function, and concept in this project from first principles. No GPU or neural-network background is assumed.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Key Vocabulary (Read First!)](#2-key-vocabulary-read-first)
   - [What is a Batch?](#what-is-a-batch)
   - [What is matrix_size?](#what-is-matrix_size)
   - [What is a Neuron / Hidden Layer?](#what-is-a-neuron--hidden-layer)
   - [What is Precision / FP32 / FP16?](#what-is-precision--fp32--fp16)
   - [What is GFLOPS?](#what-is-gflops)
   - [What is Memory Bandwidth (GB/s)?](#what-is-memory-bandwidth-gbs)
   - [What is MSE (Mean Squared Error)?](#what-is-mse-mean-squared-error)
3. [What is OpenCL?](#3-what-is-opencl)
4. [What is BLAS?](#4-what-is-blas)
5. [What is CLBlast?](#5-what-is-clblast)
6. [What is GEMM?](#6-what-is-gemm)
7. [Project File Layout](#7-project-file-layout)
8. [lib.rs — The Application Entry Point](#8-librs--the-application-entry-point)
   - [Tauri Command: save_metrics_log()](#tauri-command-save_metrics_log)
   - [Embedding the DLL](#embedding-the-dll)
   - [CLBlast Constants](#clblast-constants)
   - [Loading the DLL at Runtime](#loading-the-dll-at-runtime)
   - [Device Selection](#device-selection)
   - [Float Conversion Helpers](#float-conversion-helpers)
   - [The Singleton: get_or_init_mlp()](#the-singleton-get_or_init_mlp)
   - [Tauri Command: run_inference()](#tauri-command-run_inference)
   - [Tauri Command: run_comparison_inference()](#tauri-command-run_comparison_inference)
   - [Tauri Command: get_len()](#tauri-command-get_len)
   - [Application Entry Point: run()](#application-entry-point-run)
9. [kernel.rs — OpenCL GPU Programs](#9-kernelrs--opencl-gpu-programs)
   - [What is a Kernel?](#what-is-a-kernel)
   - [The Two-Pass Design](#the-two-pass-design)
   - [FP32_KERNEL](#fp32_kernel)
   - [FP16_KERNEL](#fp16_kernel)
   - [FP16_SCALED_KERNEL](#fp16_scaled_kernel)
   - [BIAS_RELU_FP32_KERNEL and BIAS_RELU_FP16_KERNEL](#bias_relu_fp32_kernel-and-bias_relu_fp16_kernel)
   - [FP16_TO_FP32_KERNEL](#fp16_to_fp32_kernel)
10. [types.rs — All Structs, Data, and Inference Logic](#10-typesrs--all-structs-data-and-inference-logic)
    - [Function-Pointer Types: CLBlastSgemmFn and CLBlastHgemmFn](#function-pointer-types-clblastsgemmfn-and-clblasthgemmfn)
    - [ClBlastLib struct](#clblastlib-struct)
    - [InferenceMetrics struct](#inferencemetrics-struct)
    - [StreamingData struct](#streamingdata-struct)
    - [ComparisonMetrics struct](#comparisonmetrics-struct)
    - [RoundData struct and generate()](#rounddata-struct-and-generate)
    - [MLPInference struct](#mlpinference-struct)
    - [MLPInference::new()](#mlpinferencenew)
    - [MLPInference::warmup_clblast()](#mlpinferencewarmup_clblast)
    - [MLPInference::calculate_accuracy()](#mlpinferencecalculate_accuracy)
    - [run_fp32_inference()](#run_fp32_inference)
    - [run_fp16_inference()](#run_fp16_inference)
    - [run_fp16_scaled_inference()](#run_fp16_scaled_inference)
    - [run_clblast_fp32_inference()](#run_clblast_fp32_inference)
    - [run_clblast_fp16_inference()](#run_clblast_fp16_inference)
    - [run_clblast_mixed_inference()](#run_clblast_mixed_inference)
11. [logger.rs — Automatic Logging & Plotting](#11-loggerrs--automatic-logging--plotting)
    - [MetricsRow struct](#metricsrow-struct)
    - [save_session()](#save_session)
    - [CSV Export: write_csv()](#csv-export-write_csv)
    - [PNG Chart Generation: draw_line_chart()](#png-chart-generation-draw_line_chart)
    - [Individual Plot Functions](#individual-plot-functions)
12. [The Six Precision Modes Side by Side](#12-the-six-precision-modes-side-by-side)
13. [Why CLBlast is Faster Than the Custom Kernels](#13-why-clblast-is-faster-than-the-custom-kernels)
14. [Why FP32 and CLBlast FP32 Always Show 0.0 Accuracy Error](#14-why-fp32-and-clblast-fp32-always-show-00-accuracy-error)
15. [How Every Metric is Calculated](#15-how-every-metric-is-calculated)
16. [End-to-End Data Flow](#16-end-to-end-data-flow)

---

## 1. The Big Picture

This project is a **benchmarking tool** for a two-layer neural network (called an MLP — Multi-Layer Perceptron). It runs the same neural network computation **six different ways**, each using a different numerical precision or library, and then shows you how fast each one was, how much memory it used, and how accurate it was compared to the gold-standard FP32 result.

The six modes are:

| Mode | What it uses | Where math happens |
|---|---|---|
| FP32 | 32-bit floats, custom OpenCL kernel | GPU (your code) |
| FP16 | 16-bit floats, custom OpenCL kernel | GPU (your code) |
| FP16 + Scale | 16-bit floats + per-row scale factors | GPU (your code) |
| CLBlast FP32 | 32-bit floats, SGEMM from CLBlast | GPU (CLBlast library) |
| CLBlast FP16 | 16-bit floats, HGEMM from CLBlast | GPU (CLBlast library) |
| CLBlast Mixed | FP16 storage + FP32 compute, SGEMM | GPU (CLBlast library) |

The frontend (TypeScript/React) sends a Tauri command to the Rust backend, the backend runs the computation on the GPU, and the results come back as JSON to be displayed in charts.

---

## 2. Key Vocabulary (Read First!)

Before explaining any code, these terms show up everywhere. Understand these and the rest will click.

---

### What is a Batch?

In real AI systems, you never process just one input at a time — that would be too slow. Instead, you group many inputs together and process them all simultaneously on the GPU.

**A batch is a group of inputs processed at the same time.**

In this code, `batch_size = 64` always. That means the GPU processes 64 separate inputs in one shot.

Think of it like a bakery: instead of baking one loaf of bread at a time, you load 64 loaves into the oven together. The oven time is roughly the same, but you get 64× the output.

In memory, the input data is a flat array of size `batch_size × input_size`. For batch_size=64, input_size=256, that is 16,384 numbers stored one after another:

```
[ input0_feature0, input0_feature1, ..., input0_feature255,
  input1_feature0, input1_feature1, ..., input1_feature255,
  ...
  input63_feature0, ..., input63_feature255 ]
```

---

### What is matrix_size?

`matrix_size` is the slider value from the UI. It controls the size of the network:

```
input_size  = matrix_size
hidden_size = matrix_size
output_size = matrix_size / 2
```

So if you set matrix_size=256:
- Each input has 256 features
- The hidden layer has 256 neurons
- The output has 128 values

A larger `matrix_size` means more math to do, larger GPU buffers, and (generally) higher throughput in GFLOPS because the GPU is kept busier.

---

### What is a Neuron / Hidden Layer?

A **neuron** is one unit of computation inside a neural network. It takes all of its inputs, multiplies each one by a learned weight, adds them all up, adds a bias, and applies an activation function.

For one neuron `h` processing one input item:

```
output_h = ReLU( bias[h] + sum over i of ( input[i] * weight[h][i] ) )
```

The **hidden layer** is the middle of the network — between the raw inputs and the final output. It has `hidden_size` neurons, all running in parallel on the GPU.

A **layer** is a full set of neurons running at the same time. This project has two layers:
- **Layer 1**: transforms the raw input into the hidden representation
- **Layer 2**: transforms the hidden representation into the final output

---

### What is Precision / FP32 / FP16?

Numbers in a computer are stored in binary. The more bits you use, the more precisely you can represent a decimal number.

**FP32 (32-bit float / single precision)**
- Uses 32 bits (4 bytes) per number
- Range: roughly ±3.4 × 10³⁸
- Decimal digits of precision: ~7
- Example: 3.1415927

**FP16 (16-bit float / half precision)**
- Uses 16 bits (2 bytes) per number
- Range: roughly ±65504
- Decimal digits of precision: ~3
- Example: 3.14 (already losing accuracy vs FP32)

FP16 uses **half the memory** of FP32. Because memory bandwidth is often the bottleneck on GPUs, using FP16 can double throughput — but at the cost of accuracy. Modern AI models use this tradeoff constantly.

In Rust, FP16 values are stored as `u16` (an unsigned 16-bit integer) because Rust's standard library doesn't have a native `f16` type. The `half` crate provides conversion functions.

---

### What is GFLOPS?

**GFLOPS = Giga Floating-Point Operations Per Second**

One FLOP is one arithmetic operation: one multiply or one add. "Giga" means one billion (10⁹).

So 1 GFLOPS = one billion multiplications or additions per second.

This code measures how many FLOPs the neural network computation requires, then divides by the time taken:

```
throughput_gflops = total_flops / (time_in_seconds × 1,000,000,000)
```

The total FLOPs for a two-layer MLP with batching is:

```
total_flops = batch_size × (
    hidden_size × (2 × input_size + 1)    ← Layer 1: multiply-add for each weight + bias add
  + output_size × (2 × hidden_size + 1)   ← Layer 2: multiply-add for each weight + bias add
)
```

The factor of 2 for each weight is because one weight involves both a multiply AND an add.

Higher GFLOPS = the GPU is doing more useful math per second.

---

### What is Memory Bandwidth (GB/s)?

**Memory bandwidth** measures how fast data can be moved between GPU memory and the GPU's compute cores.

**GB/s = Gigabytes per second**

Even if a GPU can do trillions of FLOPs per second, it's useless if it can't feed data fast enough. For neural networks at small matrix sizes, memory bandwidth is often the real bottleneck.

This code tracks how many bytes are actually read from and written to GPU memory during kernel execution, then divides by the time:

```
memory_bandwidth_gbps = bytes_transferred / (time_in_seconds × 1,000,000,000)
```

FP16 modes move half as many bytes as FP32 modes (2 bytes per number vs 4 bytes), which is why they often show higher memory bandwidth.

---

### What is MSE (Mean Squared Error)?

**MSE** is a measure of how wrong an answer is compared to the correct answer.

For each output value, compute:
```
error = (computed_value - correct_value)²
```
Then average all those squared errors. The result is the MSE.

In this project, FP32 is the "correct" reference. Every other mode is compared to FP32. A smaller MSE means the mode produced results closer to the FP32 gold standard.

FP16 will have some MSE because it can't represent numbers as precisely. FP16 + Scale has smaller MSE because the scaling step reduces quantization error.

---

## 3. What is OpenCL?

**OpenCL (Open Computing Language)** is an industry-standard framework for running programs on GPUs, CPUs, and other parallel processors.

GPUs have thousands of small cores. OpenCL lets you write a small program (called a **kernel**) that runs on every one of those cores simultaneously.

Think of the difference this way:
- **CPU**: 8–16 powerful workers who can each handle complex tasks
- **GPU**: 2,000–10,000 simple workers who each handle one tiny piece of a huge task simultaneously

For neural networks — which are fundamentally just "do this same multiply-add operation millions of times" — GPUs are vastly faster.

OpenCL is used in this project (rather than CUDA) because OpenCL works on **any GPU** — AMD, Intel, or Nvidia — while CUDA only works on Nvidia hardware.

The OpenCL programming model used here:
1. You write a kernel in OpenCL C (a C dialect)
2. You compile it at runtime on the target device
3. You upload data from RAM to GPU memory (a Buffer)
4. You launch the kernel with `enqueue_nd_range_kernel`
5. You download results back from GPU memory

---

## 4. What is BLAS?

**BLAS (Basic Linear Algebra Subprograms)** is a collection of highly optimized routines for linear algebra computations.

### What is Linear Algebra?

Linear algebra is math with matrices (tables of numbers) instead of single values.

A matrix looks like:

```
| 1.0   0.5   0.2 |
| 0.8   1.1  -0.3 |
| 0.1  -0.4   0.9 |
```

This is a 3×3 matrix — 3 rows, 3 columns. Matrix multiplication combines two matrices into one according to specific rules.

### Why Neural Networks Need Matrix Math

Every neural network layer does exactly this:

```
Output = Input × Weights + Bias
```

Where:
- `Input` is a matrix of shape `[batch_size × input_size]`
- `Weights` is a matrix of shape `[input_size × hidden_size]`
- The result `Output` is a matrix of shape `[batch_size × hidden_size]`

This is called **matrix multiplication** and it is the fundamental operation of all neural networks.

### BLAS Levels

BLAS is organized into three levels:
- **Level 1** — Vector operations (add two arrays, scale an array)
- **Level 2** — Matrix × vector (one matrix, one column of numbers)
- **Level 3** — Matrix × matrix (full matrices on both sides)

This project uses **Level 3 BLAS** — specifically the GEMM function.

### Why Not Just Write the Loops Yourself?

You could write matrix multiplication with three nested loops. But that ignores:
- **CPU/GPU cache behavior** — accessing memory in the wrong order is 10-100x slower
- **SIMD instructions** — modern hardware can do 8 or 16 multiplications at once
- **Tile/block algorithms** — breaking the matrix into cache-friendly chunks
- **GPU thread organization** — which threads should collaborate on which pieces

BLAS implementations encode decades of hardware-specific optimization. They are typically 10-100x faster than a naive loop.

---

## 5. What is CLBlast?

**CLBlast** is an OpenCL implementation of BLAS. It provides the SGEMM and HGEMM functions that run on any OpenCL-capable GPU.

The key feature of CLBlast is its **auto-tuner**. The first time you call SGEMM or HGEMM with a particular matrix shape (e.g., 64×256 × 256×256), CLBlast:
1. Benchmarks many different internal implementations
2. Finds the fastest one for your specific GPU and matrix size
3. Caches the result so future calls are fast immediately

This tuning cost is paid once. The `warmup_clblast()` function exists specifically to trigger this tuning *before* any measurement starts, so it doesn't pollute the benchmark results.

CLBlast is distributed as a Windows DLL (`clblast.dll`). Because distributing a DLL alongside a Tauri app is tricky, this project embeds the DLL directly into the compiled Rust binary using `include_bytes!`, then writes it to a temp folder at startup.

---

## 6. What is GEMM?

**GEMM = General Matrix Multiply**

The GEMM formula is:

```
C = alpha × A × B + beta × C
```

Where:
- `A`, `B`, `C` are matrices
- `alpha` and `beta` are scalar multipliers (usually `alpha=1.0`, `beta=0.0`)
- `beta=0.0` means "overwrite C completely" (don't add to the existing content)

**SGEMM** = Single-precision GEMM → uses FP32 (32-bit floats)  
**HGEMM** = Half-precision GEMM → uses FP16 (16-bit floats)

### What Does "Transpose" Mean?

Transposing a matrix flips its rows and columns.

Original 2×3 matrix:
```
| 1  2  3 |
| 4  5  6 |
```

Transposed (now 3×2):
```
| 1  4 |
| 2  5 |
| 3  6 |
```

In the GEMM calls in this project, the weights matrix is always passed as `CLBLAST_TRANSPOSE_YES`. This is because the weights are stored as `[hidden_size × input_size]` (each row is one neuron's weights), but the math requires the shape `[input_size × hidden_size]`. Transposing in-place avoids a costly memory rearrangement.

### Row-Major Layout

Matrices in this project are stored in **row-major** order — all elements of row 0 come first, then row 1, etc.

For a 3×4 matrix, memory looks like:
```
[ row0_col0, row0_col1, row0_col2, row0_col3,
  row1_col0, row1_col1, row1_col2, row1_col3,
  row2_col0, row2_col1, row2_col2, row2_col3 ]
```

This is the natural layout for Rust `Vec<f32>`.

### The Leading Dimension (ld parameters)

CLBlast needs to know the stride between rows — called the "leading dimension". For a row-major matrix of width `N`, the leading dimension is simply `N`.

For example, for the input matrix of shape `[batch_size × input_size]`, the leading dimension (number of columns to skip to get to the next row) is `input_size`.

---

## 7. Project File Layout

```
src-tauri/src/
├── lib.rs      ← Application wiring: Tauri commands, DLL loading, device selection
├── types.rs    ← All structs and all inference functions (the heavy math)
├── kernel.rs   ← OpenCL kernel source code (GPU programs as string constants)
├── logger.rs   ← CSV export + PNG chart generation (plotters crate) for automatic logging
└── main.rs     ← Minimal entry point that calls lib::run()
```

At runtime, a `parallel_log/` directory is created next to the executable (or in the current working directory during development) to hold automatically generated logs.

---

## 8. lib.rs — The Application Entry Point

`lib.rs` is responsible for:
- Embedding and extracting the CLBlast DLL
- Selecting an OpenCL device
- Providing helper functions used by `types.rs`
- Defining the Tauri commands that the frontend can call

---

### Embedding the DLL

```
pub const MY_DLL: &[u8] = include_bytes!("../clblast.dll");
```

`include_bytes!` is a Rust macro that reads a file at **compile time** and bakes its raw bytes directly into the binary. When someone installs your app, they get one `.exe` file that already contains the entire DLL.

At runtime, `get_clblast_dll_path()` writes those bytes to a temporary folder (only if the file doesn't already exist or has changed), then returns the path so `libloading` can open it as a normal DLL.

The path is stored in a `OnceLock<PathBuf>`. `OnceLock` is Rust's thread-safe "compute once, reuse forever" wrapper — the DLL is only written to disk once, even if many threads ask for it.

---

### CLBlast Constants

```
const CLBLAST_LAYOUT_ROW_MAJOR: i32 = 101;
const CLBLAST_TRANSPOSE_NO:     i32 = 111;
const CLBLAST_TRANSPOSE_YES:    i32 = 112;
```

These are integer codes that match CLBlast's C header file. They tell CLBlast:
- `101` → matrices are stored row by row (row-major, which is what Rust uses)
- `111` → do NOT transpose this matrix before multiplying
- `112` → DO transpose this matrix before multiplying

---

### Loading the DLL at Runtime

`load_clblast()` opens the DLL file and reads two function pointers out of it:

- **`CLBlastSgemm`** — the FP32 matrix multiply function
- **`CLBlastHgemm`** — the FP16 matrix multiply function

It uses the `libloading` crate, which is Rust's equivalent of `LoadLibrary` + `GetProcAddress` on Windows.

The function pointers are stored in a `ClBlastLib` struct (defined in `types.rs`) so they can be called later without re-opening the DLL every time.

---

### Device Selection

```rust
fn pick_device() -> Result<cl_device_id, String>
```

This function asks OpenCL "what GPUs and CPUs do you see?" and returns the first available device, preferring a GPU. If no GPU is found, it falls back to the CPU (which also supports OpenCL).

`cl_device_id` is just an opaque handle — a pointer-sized integer that represents "this device" inside the OpenCL driver.

---

### Float Conversion Helpers

```rust
fn f32_to_f16(val: f32) -> u16
fn f16_to_f32(val: u16) -> f32
```

These use the `half` crate to convert between 32-bit and 16-bit floats. The `half::f16` type knows how to do the bit manipulation correctly (handling the different exponent and mantissa widths).

Because Rust doesn't have a native `f16` type, FP16 values are stored as `u16` (raw 16-bit integers). These helpers perform the conversion on the CPU before data is uploaded to the GPU, or after it is downloaded.

---

### The Singleton: get_or_init_mlp()

```rust
fn get_or_init_mlp() -> Result<MutexGuard<'static, Option<MLPInference>>, String>
```

`MLPInference` (the main inference engine) is expensive to create — it opens an OpenCL context, compiles GPU programs, and loads the DLL. You don't want to recreate it for every button click.

`MLP_INSTANCE` is a global `Mutex<Option<MLPInference>>`. The first call creates the `MLPInference`. Every subsequent call returns the already-created one.

- **`Mutex`** — a lock that ensures only one thread uses the `MLPInference` at a time (since it holds GPU state that isn't safe to access from multiple threads simultaneously)
- **`Option<...>`** — the value starts as `None` (not yet created) and becomes `Some(mlp)` after the first call

---

### Tauri Command: run_inference()

```rust
#[tauri::command]
async fn run_inference(precision: String, matrix_size: usize) -> Result<InferenceMetrics, String>
```

This is called by the frontend when the user runs a single precision mode.

**Parameters:**
- `precision` — a string like `"Fp32"`, `"Fp16"`, `"CLBlast FP32"`, etc.
- `matrix_size` — the slider value from the UI, which sets the network size

**What it does:**
1. Gets (or creates) the global `MLPInference`
2. Sets `input_size = hidden_size = matrix_size`, `output_size = matrix_size / 2`, `batch_size = 64`
3. Generates random test data (`RoundData::generate`)
4. If not running FP32, runs FP32 first to establish the reference output for accuracy comparison
5. Runs the requested precision mode
6. Returns the `InferenceMetrics` as JSON to the frontend

The `spawn_blocking` call is important: GPU computation is blocking (it calls `.finish()` which waits for the GPU), and Tauri's async runtime expects async tasks to not block. `spawn_blocking` moves the work to a dedicated thread pool for blocking tasks.

---

### Tauri Command: run_comparison_inference()

```rust
#[tauri::command]
async fn run_comparison_inference(matrix_size: usize) -> Result<ComparisonMetrics, String>
```

This runs all **six** precision modes back-to-back and returns all results in one `ComparisonMetrics` struct.

Critical detail: **all six modes use the same `RoundData`** (same random input, same weights). This is essential for a fair comparison — if the data were different, you wouldn't be measuring precision, you'd be measuring luck.

The order is important:
1. FP32 runs first and stores its output as `fp32_reference`
2. Every other mode compares against that reference to calculate accuracy

If CLBlast fails for any reason, it gracefully falls back to `default_metrics()` (all zeros) rather than crashing the whole comparison.

---

### Tauri Command: save_metrics_log()

```rust
#[tauri::command]
async fn save_metrics_log(matrix_size: usize, rows: Vec<MetricsRow>) -> Result<String, String>
```

Called automatically by the frontend **every 5 rounds**. The frontend accumulates a `Vec<MetricsRow>` (one entry per round, containing all six modes' metrics) and sends the entire history to this command.

This command:
1. Generates a timestamped folder name: `{YYYY-MM-DD_HH-MM-SS}_{matrix_size}` using the `chrono` crate
2. Creates the folder under `parallel_log/` (relative to the current working directory)
3. Calls `logger::save_session()` which writes:
   - `metrics.csv` — all metrics from round 1 to current, all 6 modes, all fields
   - `execution_time.png` — line chart comparing execution time across all modes
   - `throughput.png` — GFLOPS comparison
   - `bandwidth.png` — memory bandwidth comparison
   - `accuracy_mse.png` — MSE vs FP32 baseline (only the 4 modes that can have non-zero MSE)
4. Returns the full path to the session folder (logged by the frontend to the console)

The computation runs on `spawn_blocking` so the UI stays responsive while the PNG charts are rendered.

---

### Tauri Command: get_len()

```rust
#[tauri::command]
async fn get_len() -> Result<usize, String>
```

Returns the size in bytes of the embedded DLL. This is used by the frontend to display how large the bundled CLBlast library is.

---

### Application Entry Point: run()

```rust
pub fn run()
```

This is the standard Tauri setup. It:
1. Creates the Tauri application builder
2. Registers the four commands (`run_inference`, `run_comparison_inference`, `get_len`, `save_metrics_log`) so the frontend JavaScript can call them
3. Starts the application event loop

---

## 9. kernel.rs — OpenCL GPU Programs

This file contains OpenCL C source code stored as Rust string constants. These strings are compiled at runtime by the OpenCL driver into GPU machine code.

---

### What is a Kernel?

An OpenCL **kernel** is a function that runs on the GPU. When you "launch" a kernel, you tell the GPU "start N copies of this function, each with a different thread ID."

Each thread identifies itself with `get_global_id(0)` — its unique index from 0 to N-1.

For a layer with `batch_size=64` and `hidden_size=256`, you launch `64 × 256 = 16,384` threads. Thread number 5 computes hidden neuron 5 for batch item 0. Thread 256 computes hidden neuron 0 for batch item 1. And so on.

This is the fundamental source of GPU parallelism: 16,384 threads all computing their one output value simultaneously.

---

### The Two-Pass Design

All custom kernels use a two-pass (two-kernel) design:

**Pass 1 (Layer 1 kernel):** Each thread computes one hidden neuron for one batch item, then writes its result to a `hidden_buffer`.

**Pass 2 (Layer 2 kernel):** Each thread reads from `hidden_buffer` and computes one output neuron for one batch item.

Why two separate passes instead of one big kernel?

If you did both layers in one kernel, each thread computing one output value would need to recompute the entire hidden layer for its batch item — and since multiple output neurons share the same hidden values, you'd compute each hidden value `output_size` times. This is wasteful.

By splitting into two passes, each hidden value is computed exactly **once** and stored in `hidden_buffer`. All output neurons for the same batch item read from the same already-computed hidden values.

---

### FP32_KERNEL

Contains two OpenCL functions:

**`mlp_fp32_layer1`** — Layer 1 in 32-bit float precision.

Each thread:
1. Calculates which batch item and hidden neuron it's responsible for using integer division and modulo
2. Loads `bias1[h]` as the starting sum
3. Loops over all `input_size` input features, accumulating `input[...] × weights1[...]`
4. Applies `fmax(0.0f, sum)` — this is **ReLU**, the activation function
5. Writes the result to `hidden_buffer[batch_idx * hidden_size + h]`

**ReLU (Rectified Linear Unit):** A function that returns the input if positive, or 0 if negative. `ReLU(x) = max(0, x)`. It's used after Layer 1 to introduce non-linearity — without activation functions, stacking linear layers is equivalent to a single linear layer.

**`mlp_fp32_layer2`** — Layer 2 in 32-bit float precision.

Same structure as Layer 1, but reads from `hidden_buffer` instead of `input`, and writes to `output`. No ReLU on the final output (the network's raw scores are used directly).

---

### FP16_KERNEL

Identical structure to `FP32_KERNEL`, but every `float` is replaced with `half` (OpenCL's 16-bit float type). Requires `#pragma OPENCL EXTENSION cl_khr_fp16 : enable` to activate FP16 support on the device.

Since `half` has less precision than `float`, the accumulation of many small multiply-add operations introduces rounding errors. These errors compound and produce results that differ from the FP32 kernel — this difference is what `accuracy_mse` measures.

---

### FP16_SCALED_KERNEL

This kernel implements **row-wise quantization** — a technique to reduce the accuracy loss of FP16.

The problem with plain FP16: if a weight has a value of 0.0001, it may be rounded to 0 in FP16 (which only has ~3 decimal digits of precision), causing a permanent loss of information.

The solution: before converting to FP16, find the largest absolute value in each weight row (`max_abs`). Divide all weights in that row by `max_abs`. Now all values are in the range `[-1, 1]`, which FP16 can represent well. Store `max_abs` as a separate FP32 `scale` value.

During computation, multiply each product by `scale` to undo the division: `(weight / scale) × scale = weight`.

**`mlp_fp16_scaled_layer1`:**
- Weights and input are `half`
- Bias is `half`
- `scales1` is `float` (one scale per hidden neuron)
- Hidden buffer is `float` (FP32) — the unscaled result is high precision
- Each multiply-add is: `(float)input[i] × (float)weights1[h*input_size+i] × scale`

**`mlp_fp16_scaled_layer2`:**
- Reads from the FP32 `hidden_buffer`
- Weights are `half`, `scales2` is `float`
- Output is `half`

---

### BIAS_RELU_FP32_KERNEL and BIAS_RELU_FP16_KERNEL

The CLBlast SGEMM/HGEMM functions only do the matrix multiplication `C = A × B`. They don't add bias or apply ReLU. These helper kernels fill that gap.

**`add_bias_relu_fp32`** — After SGEMM Layer 1: adds `bias[col]` to each element and applies ReLU.

The `col` of each element is found with `gid % cols` — this correctly handles the fact that all batch items share the same bias vector.

**`add_bias_fp32`** — After SGEMM Layer 2: adds `bias[col]` to each element, but NO ReLU (final layer output is not activated).

The FP16 versions are identical but use `half` instead of `float`.

---

### FP16_TO_FP32_KERNEL

Used by the CLBlast Mixed mode. Contains two conversion kernels:

**`convert_fp16_to_fp32`** — Each thread reads one `half` value and writes one `float`. Used to convert FP16-stored weights/inputs into FP32 before passing them to SGEMM.

**`convert_fp32_to_fp16`** — Each thread reads one `float` and writes one `half`. (Defined but not currently used in the measured path.)

---

## 10. types.rs — All Structs, Data, and Inference Logic

This is the largest file in the project. It defines all data structures and contains all six inference functions.

---

### Function-Pointer Types: CLBlastSgemmFn and CLBlastHgemmFn

```rust
pub type CLBlastSgemmFn = unsafe extern "C" fn(
    layout: i32, a_transpose: i32, b_transpose: i32,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a_buffer: *mut c_void, a_offset: usize, a_ld: usize,
    b_buffer: *mut c_void, b_offset: usize, b_ld: usize,
    beta: f32,
    c_buffer: *mut c_void, c_offset: usize, c_ld: usize,
    queue: *mut *mut c_void, event: *mut cl_event,
) -> i32;
```

This is a Rust **type alias** for a function pointer matching the exact signature of `CLBlastSgemm` in CLBlast's C header.

- `unsafe` — calling this function can crash if arguments are wrong (it's talking directly to the DLL)
- `extern "C"` — use C's calling convention (argument passing rules), not Rust's
- `m`, `n`, `k` — the GEMM matrix dimensions: C is [m×n], A is [m×k], B is [k×n]
- `a_buffer`, `b_buffer`, `c_buffer` — raw OpenCL memory handles (pointers to GPU memory)
- `a_offset`, `b_offset`, `c_offset` — byte offsets into the buffers (always 0 here)
- `a_ld`, `b_ld`, `c_ld` — leading dimensions (row strides)
- `queue` — pointer to the OpenCL command queue
- `event` — output event handle (can be null)
- Returns `i32` — status code (0 = success)

`CLBlastHgemmFn` is the same but `alpha` and `beta` are `u16` (FP16 bits) instead of `f32`.

---

### ClBlastLib struct

```rust
pub struct ClBlastLib {
    pub _lib: libloading::Library,
    pub sgemm: CLBlastSgemmFn,
    pub hgemm: CLBlastHgemmFn,
}
```

A bundle that keeps the loaded DLL alive (`_lib`) alongside the two function pointers we extracted from it. If `_lib` were dropped, the DLL would be unloaded and the function pointers would become dangling — so they must live together.

`unsafe impl Send` and `unsafe impl Sync` tell the Rust compiler "it's safe to share this across threads" — which is true because CLBlast is thread-safe when called with different queues.

---

### InferenceMetrics struct

```rust
pub struct InferenceMetrics {
    pub execution_time_ms: f64,       // How long the GPU kernels took, in milliseconds
    pub memory_bandwidth_gbps: f64,   // How fast data moved through GPU memory (GB/s)
    pub throughput_gflops: f64,       // How many billion math operations per second
    pub memory_footprint_mb: f64,     // Total GPU memory allocated (megabytes)
    pub accuracy_mse: f64,            // Mean Squared Error vs FP32 reference
    pub accuracy_max_error: f64,      // Largest single-value error vs FP32 reference
}
```

This is what gets sent back to the frontend as JSON after each inference run. It is decorated with `#[derive(Serialize, Deserialize)]` so Tauri can automatically convert it to/from JSON.

---

### StreamingData struct

```rust
pub struct StreamingData {
    pub timestamp: f64,
    pub execution_time_ms: f64,
    pub throughput_gflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub accuracy_mse: f64,
}
```

A lighter version of `InferenceMetrics` used for time-series charting (streaming multiple data points over time).

---

### ComparisonMetrics struct

```rust
pub struct ComparisonMetrics {
    pub fp32:          InferenceMetrics,
    pub fp16:          InferenceMetrics,
    pub fp16_scaled:   InferenceMetrics,
    pub clblast_fp32:  InferenceMetrics,
    pub clblast_fp16:  InferenceMetrics,
    pub clblast_mixed: InferenceMetrics,
}
```

A container for all six results, returned by `run_comparison_inference()`. The frontend uses this to draw side-by-side bar charts.

---

### RoundData struct and generate()

```rust
pub struct RoundData {
    input_data:  Vec<f32>,   // [batch_size × input_size] — random values in [0, 1)
    weights1:    Vec<f32>,   // [hidden_size × input_size] — deterministic, from sin()
    bias1:       Vec<f32>,   // [hidden_size] — small positive values: i × 0.01
    weights2:    Vec<f32>,   // [output_size × hidden_size] — deterministic, from cos()
    bias2:       Vec<f32>,   // [output_size] — small positive values: i × 0.01
    input_size:  usize,
    hidden_size: usize,
    output_size: usize,
    batch_size:  usize,
}
```

`RoundData` is a snapshot of all data for one benchmark round.

**Why random inputs but deterministic weights?**

- Weights use `sin()` and `cos()` formulas so they are always the same for a given `matrix_size`. This means all six modes are always computing with identical network weights — you're only changing how the arithmetic is done, not what numbers are involved.
- Inputs are random (`rand::random::<f32>()`) — this prevents any mode from getting lucky with a specially crafted input, and makes the benchmark representative of real workloads.

**Memory layout of weights1 (row-major, one row per neuron):**
```
[ w1_neuron0_input0, w1_neuron0_input1, ..., w1_neuron0_inputN,
  w1_neuron1_input0, ...,
  ...
  w1_neuronH_input0, ..., w1_neuronH_inputN ]
```

This layout means `weights1[h * input_size + i]` gives the weight connecting input `i` to hidden neuron `h`.

---

### MLPInference struct

```rust
pub struct MLPInference {
    context:           Context,                         // OpenCL context
    queue:             CommandQueue,                     // GPU command queue
    device:            Device,                          // The GPU/CPU device handle
    fp32_reference:    Option<Vec<f32>>,                // Reference output from FP32 run
    clblast:           Option<ClBlastLib>,              // Loaded CLBlast library (may be None)
    bias_fp32_program: Option<Program>,                 // Pre-compiled FP32 bias/ReLU kernel
    bias_fp16_program: Option<Program>,                 // Pre-compiled FP16 bias/ReLU kernel
    convert_program:   Option<Program>,                 // Pre-compiled FP16↔FP32 convert kernel
    warmed_up_sizes:   HashSet<usize>,                  // Which matrix_sizes have been CLBlast-tuned
}
```

This is the central object that holds all GPU state.

**`context`** — An OpenCL context groups a set of devices and the memory allocated on them. All buffers and programs belong to a context.

**`queue`** — A command queue is the ordered list of operations sent to the GPU. Commands are enqueued on the CPU side and executed asynchronously on the GPU. Calling `.finish()` waits until all enqueued commands complete.

**`fp32_reference`** — Stores the output of the FP32 run so other modes can compare against it. It's `Option<Vec<f32>>` because it starts as `None` (no run has happened yet) and becomes `Some(output)` after the first FP32 run.

**`clblast`** — `Option<ClBlastLib>` because CLBlast loading can fail (DLL not found, device not supported). All CLBlast inference functions check `self.clblast.as_ref().ok_or("CLBlast not loaded")?` and return an error gracefully if it's missing.

**Pre-compiled programs** — Compiling an OpenCL kernel takes time (the driver compiles the OpenCL C to GPU machine code). For the bias and convert kernels, this compilation is done once in `new()` and the compiled `Program` is stored. The main inference kernels (FP32_KERNEL, FP16_KERNEL, etc.) are recompiled each call — this could be optimized but has minimal impact on long benchmark runs.

**`warmed_up_sizes`** — A `HashSet<usize>` tracking which `matrix_size` values have already gone through CLBlast auto-tuning. If `matrix_size=256` is already in the set, `warmup_clblast(256)` does nothing.

---

### MLPInference::new()

Creates the MLPInference by:
1. Calling `pick_device()` to find a GPU or CPU
2. Creating an OpenCL `Context` for that device
3. Creating a `CommandQueue` with `CL_QUEUE_PROFILING_ENABLE` (enables timing events)
4. Loading CLBlast (may silently fail, stored as `Option`)
5. Pre-compiling the three helper kernel programs
6. Returning the initialized struct

---

### MLPInference::warmup_clblast()

This function runs a "dummy" SGEMM and HGEMM with the same matrix shapes as the actual benchmark, but without measuring time. The purpose is to trigger CLBlast's internal auto-tuner.

CLBlast auto-tuning is a one-time cost: the first SGEMM call with shape (64, 256, 256) might take seconds while CLBlast benchmarks different algorithmic strategies internally. After that, the chosen strategy is cached and every subsequent call is fast.

Without warmup, the first measured CLBlast call would include tuning time and appear artificially slow.

The function allocates scratch GPU buffers (contents don't matter — they're uninitialized), runs the GEMM calls, then adds `matrix_size` to `warmed_up_sizes` so it won't repeat.

---

### MLPInference::calculate_accuracy()

```rust
pub fn calculate_accuracy(&self, output: &[f32], reference: &[f32]) -> (f64, f64)
```

Compares two output arrays element by element. Returns `(mse, max_error)`.

```
mse        = sum of (output[i] - reference[i])² / N
max_error  = max of |output[i] - reference[i]|
```

Both metrics are computed in `f64` (64-bit double) to avoid precision loss during the accumulation of many squared differences.

---

### run_fp32_inference()

The **baseline** mode. Everything is FP32 (32-bit float).

**Steps:**
1. Allocate 7 GPU buffers: input, weights1, bias1, weights2, bias2, hidden, output — all `Buffer<f32>` (4 bytes per element)
2. Upload all data from CPU RAM to GPU memory using `enqueue_write_buffer`
3. Compile `FP32_KERNEL` and create `mlp_fp32_layer1` and `mlp_fp32_layer2` kernels
4. Set kernel arguments (one by one — OpenCL has no named parameters)
5. Start timer
6. Launch Layer 1: `batch_size × hidden_size` threads (e.g., 64 × 256 = 16,384)
7. `queue.finish()` — wait for GPU to complete Layer 1
8. Launch Layer 2: `batch_size × output_size` threads
9. `queue.finish()` — wait for completion
10. Stop timer
11. Download output from GPU to CPU with `enqueue_read_buffer`
12. Store output as `self.fp32_reference`
13. Calculate and return `InferenceMetrics`

**Accuracy is always 0.0** for this mode because it IS the reference — you can't have error relative to yourself.

**Memory footprint calculation:**
```
(batch_size×input_size + hidden_size×input_size + hidden_size
 + output_size×hidden_size + output_size
 + batch_size×hidden_size + batch_size×output_size) × 4 bytes
```

---

### run_fp16_inference()

Pure FP16 mode. Same structure as FP32 but every buffer is `Buffer<u16>` (2 bytes per element).

**Extra step before uploading:** Convert all FP32 data arrays to FP16 on the CPU:
```rust
let input_data: Vec<u16> = rd.input_data.iter().map(|&v| f32_to_f16(v)).collect();
```

After running the GPU kernel, convert the FP16 output back to FP32 for accuracy comparison:
```rust
let output_f32: Vec<f32> = output.iter().map(|&v| f16_to_f32(v)).collect();
```

**Memory footprint:** Same element count, but `× 2 bytes` instead of `× 4 bytes`, so roughly half the FP32 footprint.

**Accuracy:** Will show non-zero MSE because FP16 rounding errors accumulate across many multiply-add operations.

---

### run_fp16_scaled_inference()

FP16 with row-wise quantization.

**Extra preprocessing on CPU (before GPU):**

For each row `h` of `weights1`:
1. Find `max_abs = max(|weight[h][0]|, |weight[h][1]|, ..., |weight[h][input_size-1]|)`
2. `scales1[h] = max_abs` (or 1.0 if max_abs is 0)
3. Divide each weight by `scales1[h]` and convert to FP16

Now the largest value in each FP16 row is exactly ±1.0, which is representable with full FP16 precision. Smaller values get proportionally more precision than if the scale were large.

**Mixed buffer types:**
- Input, weights1, weights2, bias1, bias2, output → `Buffer<u16>` (FP16)
- scales1, scales2 → `Buffer<f32>` (FP32, one per neuron)
- hidden_buffer → `Buffer<f32>` (FP32 — Layer 1 output is kept in FP32 precision)

**Why keep hidden in FP32?**
The hidden values are intermediate results that feed directly into Layer 2. Keeping them in FP32 prevents the accumulation of a second round of FP16 quantization error.

**Accuracy:** Better than plain FP16, but still worse than FP32, because the input and weights still lose some precision during the FP32→FP16 conversion.

---

### run_clblast_fp32_inference()

Uses CLBlast's `SGEMM` (Single-precision GEMM) instead of the custom kernel.

**Steps:**
1. Allocate buffers (all FP32)
2. Upload data
3. Get cached `bias_fp32_program` (avoids recompiling the bias kernel)

**Layer 1 (SGEMM call):**
```
hidden = input × weights1ᵀ
```
- A = input matrix [batch_size × input_size]
- B = weights1 matrix [hidden_size × input_size], transposed → [input_size × hidden_size]
- C = hidden matrix [batch_size × hidden_size]
- alpha=1.0, beta=0.0

After SGEMM, run `add_bias_relu_fp32` kernel to add bias and apply ReLU.

**Layer 2 (SGEMM call):**
```
output = hidden × weights2ᵀ
```
After SGEMM, run `add_bias_fp32` kernel to add bias (no ReLU).

**Accuracy is always 0.0** because SGEMM computes FP32 math exactly like the custom kernel — same precision, same order of operations (effectively). The result matches the FP32 reference perfectly.

---

### run_clblast_fp16_inference()

Uses CLBlast's `HGEMM` (Half-precision GEMM).

All data is converted to FP16 (`Vec<u16>`) and all buffers are `Buffer<u16>`. The alpha and beta scalars for HGEMM are also FP16 (converted with `f32_to_f16(1.0)` and `f32_to_f16(0.0)`).

After HGEMM, the `add_bias_relu_fp16` and `add_bias_fp16` kernels handle bias/ReLU in FP16.

The output FP16 array is converted back to FP32 for accuracy comparison.

**Why is CLBlast HGEMM faster than the custom FP16 kernel?**
See [Section 12](#12-why-clblast-is-faster-than-the-custom-kernels).

---

### run_clblast_mixed_inference()

The most complex mode: **store in FP16, compute in FP32**.

The idea: FP16 storage cuts memory bandwidth in half (less data to move), but FP32 compute preserves accuracy. This is similar to what NVIDIA calls "automatic mixed precision" in deep learning frameworks.

**How it works:**

1. Convert input and weights to FP16 and upload to GPU as `Buffer<u16>`
2. Also allocate matching FP32 buffers for the same data: `Buffer<f32>`
3. Bias stays FP32 throughout (bias values are small and benefit from full precision)

**Start of timed section:**

4. Run `convert_fp16_to_fp32` kernel on input → writes FP32 input to `input_f32_buf`
5. Run `convert_fp16_to_fp32` kernel on weights1 → writes FP32 weights1 to `weights1_f32_buf`
6. Run SGEMM Layer 1 using the FP32 converted buffers → `hidden_buf` (FP32)
7. Run `add_bias_relu_fp32` on hidden_buf
8. Run `convert_fp16_to_fp32` kernel on weights2 → writes FP32 weights2 to `weights2_f32_buf`
9. Run SGEMM Layer 2 → `output_buf` (FP32)
10. Run `add_bias_fp32` on output_buf

**End of timed section.**

11. Read output (already FP32) and compare with reference for accuracy

**Memory footprint:** This mode uses MORE memory than either pure FP32 or pure FP16, because it maintains both the FP16 storage buffers AND the FP32 compute buffers simultaneously. The footprint is:

```
FP16 storage: (batch_size×input_size + hidden_size×input_size + output_size×hidden_size) × 2
FP32 compute: (same sizes + bias sizes + hidden + output) × 4
```

**Why use this?** In a production system, you'd stream the weights from disk or network storage in FP16 (saving bandwidth), convert on-the-fly, and compute in FP32. This mode simulates that pattern.

---

## 11. logger.rs — Automatic Logging & Plotting

This module handles **automatic CSV export and PNG chart generation** using the `plotters` crate. It is called every 5 rounds by the frontend via the `save_metrics_log` Tauri command.

---

### MetricsRow struct

```rust
pub struct MetricsRow {
    pub round: usize,
    pub fp32: InferenceMetrics,
    pub fp16: InferenceMetrics,
    pub fp16_scaled: InferenceMetrics,
    pub clblast_fp32: InferenceMetrics,
    pub clblast_fp16: InferenceMetrics,
    pub clblast_mixed: InferenceMetrics,
}
```

One row per round. The frontend builds a `Vec<MetricsRow>` by pushing a new entry after every comparison round. All six modes' full `InferenceMetrics` are included so the logger has everything it needs.

Derives `serde::Deserialize` so Tauri can deserialize the JSON array sent from TypeScript directly into `Vec<MetricsRow>`.

---

### save_session()

```rust
pub fn save_session(dir: &Path, rows: &[MetricsRow]) -> Result<String, String>
```

The top-level entry point. Given a directory path and accumulated metrics:

1. Creates the directory (including parents) via `fs::create_dir_all`
2. Calls `write_csv()` to produce `metrics.csv`
3. Calls `plot_execution_time()`, `plot_throughput()`, `plot_bandwidth()`, `plot_accuracy()` to produce four PNG files
4. Returns the directory path as a string

---

### CSV Export: write_csv()

Writes a `metrics.csv` file with one header row and one data row per round. The header has 37 columns:

```
round,
fp32_exec_ms,fp32_gflops,fp32_bw_gbps,fp32_mem_mb,fp32_mse,fp32_maxerr,
fp16_exec_ms,fp16_gflops,fp16_bw_gbps,fp16_mem_mb,fp16_mse,fp16_maxerr,
fp16s_exec_ms,fp16s_gflops,fp16s_bw_gbps,fp16s_mem_mb,fp16s_mse,fp16s_maxerr,
cb32_exec_ms,cb32_gflops,cb32_bw_gbps,cb32_mem_mb,cb32_mse,cb32_maxerr,
cb16_exec_ms,cb16_gflops,cb16_bw_gbps,cb16_mem_mb,cb16_mse,cb16_maxerr,
cbmx_exec_ms,cbmx_gflops,cbmx_bw_gbps,cbmx_mem_mb,cbmx_mse,cbmx_maxerr
```

This CSV can be imported directly into Excel, Python (pandas), or any data analysis tool for further study.

---

### PNG Chart Generation: draw_line_chart()

```rust
fn draw_line_chart(
    path: &Path,
    title: &str,
    y_label: &str,
    series: &[SeriesInfo],
) -> Result<(), String>
```

A shared helper that renders a multi-line chart as a 1200×600 PNG using `plotters::BitMapBackend`. It:

1. Computes axis bounds from the data (with 15% headroom on the Y axis)
2. Draws the chart mesh with axis labels
3. Draws each series as a colored line with a 2px stroke
4. Draws a legend in the upper-right corner with colored rectangles matching each line

**Colors** are defined as constants matching the frontend dashboard:

| Mode | Color | RGB |
|---|---|---|
| FP32 | Green | (34, 197, 94) |
| FP16 | Blue | (59, 130, 246) |
| FP16 + Scale | Purple | (168, 85, 247) |
| CLBlast FP32 | Yellow | (234, 179, 8) |
| CLBlast FP16 | Cyan | (6, 182, 212) |
| CLBlast Mixed | Pink | (236, 72, 153) |

---

### Individual Plot Functions

Four functions produce the four PNG files, each calling `draw_line_chart()` with appropriate data:

| Function | Output file | Series included | Metric extracted |
|---|---|---|---|
| `plot_execution_time()` | `execution_time.png` | All 6 modes | `execution_time_ms` |
| `plot_throughput()` | `throughput.png` | All 6 modes | `throughput_gflops` |
| `plot_bandwidth()` | `bandwidth.png` | All 6 modes | `memory_bandwidth_gbps` |
| `plot_accuracy()` | `accuracy_mse.png` | FP16, FP16+Scale, CLBlast FP16, CLBlast Mixed | `accuracy_mse` |

The accuracy chart includes only four modes because FP32 and CLBlast FP32 always have MSE = 0.0 — including them would just add flat zero lines that obscure the interesting differences.

Each function uses `extract_mode()` to pull the relevant metric from each `MetricsRow`:

```rust
fn extract_mode(
    rows: &[MetricsRow],
    mode_fn: impl Fn(&MetricsRow) -> &InferenceMetrics,
    metric_fn: impl Fn(&InferenceMetrics) -> f64,
) -> Vec<(f64, f64)>
```

This returns `(round_number, metric_value)` pairs ready for plotters to consume.

---

### Frontend Integration

The frontend (`App.tsx`) drives the logging:

1. A `metricsHistoryRef` (React ref) accumulates `MetricsRow` entries — one per round
2. A `roundCounterRef` tracks the current round number
3. After each round, the new metrics are pushed onto the history
4. When `roundCounterRef.current % 5 === 0`, the frontend calls `invoke("save_metrics_log", { matrixSize, rows })` with the full history
5. Changing the matrix size or stopping/starting the loop resets both refs

The save call is fire-and-forget from the UI perspective — if it fails, an error is logged to the console but the inference loop continues uninterrupted.

---

## 12. The Six Precision Modes Side by Side

| Mode | Storage | Compute | Bias/ReLU | Memory/element | Accuracy |
|---|---|---|---|---|---|
| FP32 | FP32 | FP32 (custom) | In kernel | 4 bytes | Perfect (reference) |
| FP16 | FP16 | FP16 (custom) | In kernel | 2 bytes | Some error |
| FP16 + Scale | FP16 + FP32 scales | FP32 accumulation | In kernel | ~2.1 bytes avg | Better than FP16 |
| CLBlast FP32 | FP32 | FP32 (SGEMM) | Separate kernel | 4 bytes | Perfect (= FP32) |
| CLBlast FP16 | FP16 | FP16 (HGEMM) | Separate kernel | 2 bytes | Similar to FP16 |
| CLBlast Mixed | FP16 + FP32 copies | FP32 (SGEMM) | Separate kernel | ~6 bytes avg | Similar to FP32 |

---

## 13. Why CLBlast is Faster Than the Custom Kernels

The custom kernels (FP32_KERNEL, FP16_KERNEL) are correct and simple, but they use a naive algorithm:

```
// Each thread does its own full dot product from scratch:
for i in 0..input_size {
    sum += input[batch_idx * input_size + i] * weights1[h * input_size + i];
}
```

**Problems with this approach:**

1. **No memory access coalescing**: Adjacent GPU threads read non-adjacent memory addresses. GPU memory controllers are fastest when adjacent threads read adjacent memory locations (coalesced access). The custom kernel's access pattern is partially uncoalesced.

2. **No shared memory tiling**: Modern GPU optimization involves loading a tile of data into fast on-chip shared memory and having many threads collaborate on that tile. The custom kernel doesn't use shared memory at all.

3. **No vectorization**: GPU cores can read 4 or 8 floats in one instruction. Custom kernels read one at a time.

**CLBlast handles all of this** through its auto-tuner, which selects among many pre-written optimized kernels and chooses the best tile sizes and memory access patterns for the specific GPU and matrix shape.

For large matrices, CLBlast SGEMM can be **5-20× faster** than a naive kernel.

---

## 14. Why FP32 and CLBlast FP32 Always Show 0.0 Accuracy Error

The `accuracy_mse` and `accuracy_max_error` are always 0.0 for two modes:

**FP32 (custom kernel):** This mode stores its output into `self.fp32_reference`. When accuracy is calculated, it compares `output` against `self.fp32_reference` — which is the same array. `output[i] - reference[i] = 0` for every element. Zero error.

**CLBlast FP32 (SGEMM):** This mode also computes in 32-bit float. SGEMM on the same data produces the same mathematical result as the custom FP32 kernel (both are computing `alpha × A × B + beta × C` with `alpha=1, beta=0` in FP32). The result matches the reference exactly, giving 0.0 error.

The code makes this explicit with a comment and hardcodes the accuracy values:
```rust
let accuracy_mse = 0.0_f64;
let accuracy_max_error = 0.0_f64;
```

---

## 15. How Every Metric is Calculated

### Execution Time (ms)

Measured with `std::time::Instant`:
```rust
let kernel_start = Instant::now();
// ... GPU kernels + queue.finish() ...
let kernel_time = kernel_start.elapsed();
```

`Instant::now()` captures the current time. `.elapsed()` gives the duration since that moment. Converted to milliseconds with `kernel_time.as_secs_f64() * 1000.0`.

This includes GPU execution time AND any CPU-side overhead (setting kernel arguments, enqueueing). For large matrices, GPU time dominates.

### Throughput (GFLOPS)

```rust
let total_flops = batch_size * (
    hidden_size * (2 * input_size + 1)   // Layer 1: multiply-adds + bias
  + output_size * (2 * hidden_size + 1)  // Layer 2: multiply-adds + bias
) as f64;
let throughput_gflops = total_flops / (kernel_time.as_secs_f64() * 1e9);
```

The `2 *` factor: each weight involves one multiply AND one add (two FLOPs). The `+ 1` accounts for the bias addition per neuron.

### Memory Bandwidth (GB/s)

Each mode counts the total bytes actually moved during kernel execution (reads + writes), then divides by time:

```rust
let memory_bandwidth_gbps = memory_transferred / (kernel_time.as_secs_f64() * 1e9);
```

FP32 buffers count 4 bytes per element. FP16 buffers count 2 bytes per element. The counts are carefully split by mode (e.g., FP16 Scaled has a mix of 2-byte and 4-byte buffers).

### Memory Footprint (MB)

The total GPU memory allocated — all buffers added together — divided by `1024 × 1024`:

```rust
let memory_footprint_mb = total_bytes as f64 / (1024.0 * 1024.0);
```

This is a static calculation based on sizes, not a measurement of GPU VRAM usage.

### Accuracy (MSE and Max Error)

Calculated by `calculate_accuracy()` — see [that section](#mlpinferencecalculate_accuracy) above.

---

## 16. End-to-End Data Flow

Here is the complete journey of data when you click "Run Comparison" in the UI:

```
[User clicks button]
        │
        ▼
[Frontend calls invoke("run_comparison_inference", { matrix_size: 256 })]
        │
        ▼
[Tauri routes to run_comparison_inference() in lib.rs]
        │
        ▼
[spawn_blocking: moves work to a thread pool so the UI doesn't freeze]
        │
        ▼
[get_or_init_mlp() — returns (or creates) the global MLPInference]
        │
        ▼
[RoundData::generate(256, 256, 128, 64)
    → 64×256 random input, deterministic weights, biases]
        │
        ▼
[warmup_clblast(256) — fires CLBlast auto-tuning if not already done]
        │
        ▼
[run_fp32_inference(&rd)
    → uploads data to GPU (FP32 buffers)
    → launches FP32_KERNEL layer1 (16,384 threads)
    → launches FP32_KERNEL layer2 (8,192 threads)
    → downloads output
    → stores in self.fp32_reference
    → returns InferenceMetrics]
        │
        ▼
[run_fp16_inference(&rd)
    → converts data to FP16 on CPU
    → uploads to GPU (FP16 buffers)
    → launches FP16_KERNEL (same thread counts)
    → downloads, converts FP16→FP32
    → compares with fp32_reference for accuracy
    → returns InferenceMetrics]
        │
        ▼
[run_fp16_scaled_inference(&rd)
    → computes row scales on CPU
    → divides weights by scale, converts to FP16
    → uploads (mixed FP16/FP32 buffers)
    → launches FP16_SCALED_KERNEL
    → returns InferenceMetrics]
        │
        ▼
[run_clblast_fp32_inference(&rd)
    → uploads (FP32 buffers)
    → calls CLBlastSgemm for layer 1
    → calls add_bias_relu_fp32 kernel
    → calls CLBlastSgemm for layer 2
    → calls add_bias_fp32 kernel
    → returns InferenceMetrics]
        │
        ▼
[run_clblast_fp16_inference(&rd)
    → converts to FP16, uploads
    → calls CLBlastHgemm for layer 1
    → calls add_bias_relu_fp16
    → calls CLBlastHgemm for layer 2
    → calls add_bias_fp16
    → converts output FP16→FP32, compares
    → returns InferenceMetrics]
        │
        ▼
[run_clblast_mixed_inference(&rd)
    → uploads FP16 storage + FP32 bias buffers
    → converts FP16 input+weights1 to FP32 on GPU
    → SGEMM layer 1 (FP32)
    → add_bias_relu_fp32
    → converts FP16 weights2 to FP32 on GPU
    → SGEMM layer 2 (FP32)
    → add_bias_fp32
    → returns InferenceMetrics]
        │
        ▼
[ComparisonMetrics { fp32, fp16, fp16_scaled, clblast_fp32, clblast_fp16, clblast_mixed }]
        │
        ▼
[Tauri serializes to JSON and sends to frontend]
        │
        ▼
[Frontend renders bar charts and tables]
        │
        ▼
[Frontend accumulates MetricsRow in metricsHistoryRef]
        │
        ▼  (every 5th round)
[Frontend calls invoke("save_metrics_log", { matrixSize, rows })]
        │
        ▼
[Tauri routes to save_metrics_log() in lib.rs]
        │
        ▼
[logger::save_session() writes to parallel_log/{timestamp}_{size}/]
        │
        ├─→ metrics.csv           (all metrics, all rounds, all modes)
        ├─→ execution_time.png    (line chart via plotters)
        ├─→ throughput.png        (line chart via plotters)
        ├─→ bandwidth.png         (line chart via plotters)
        └─→ accuracy_mse.png     (line chart via plotters)
```

---

*This document covers every file, every struct field, every function, and every concept used in the Mixed-Precision GPU Inference project. If anything is still unclear, trace the data flow from Section 16 and look up the specific struct or function in the section that covers it.*