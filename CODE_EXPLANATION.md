# Complete Explanation of the Mixed-Precision GPU Inference Code

This document explains every part of the Rust + OpenCL + CLBlast code that runs a two-layer neural network (MLP) across **six different precision modes** simultaneously, comparing their speed, memory usage, and accuracy.

---

## Table of Contents
1. [What is OpenCL?](#what-is-opencl)
2. [What is BLAS? (Read This Carefully!)](#what-is-blas)
3. [What is CLBlast?](#what-is-clblast)
4. [What is GEMM? (The Heart of Neural Networks)](#what-is-gemm)
5. [Import Statements](#import-statements)
6. [CLBlast DLL Loading](#clblast-dll-loading)
7. [Data Structures](#data-structures)
8. [Device Selection](#device-selection)
9. [OpenCL Kernels](#opencl-kernels)
10. [Float Conversion Functions](#float-conversion-functions)
11. [RoundData - Shared Test Data](#rounddata---shared-test-data)
12. [MLPInference Structure](#mlpinference-structure)
13. [Inference Functions](#inference-functions)
14. [CLBlast Inference Functions](#clblast-inference-functions)
15. [Command Handlers](#command-handlers)
16. [The Six Precision Modes Compared](#the-six-precision-modes-compared)
17. [Summary](#summary)

---

## What is OpenCL?

**OpenCL (Open Computing Language)** is a framework that lets you write code that runs on different types of processors — GPUs, CPUs, and more — using a single standard.

Think of it like this: if you need to paint 1,000 houses, you could:
- Use **1 painter (CPU)** — paints one house at a time, very carefully
- Use **1,000 painters (GPU)** — all paint at the same time, much faster overall

OpenCL lets you use the GPU's "1,000 painters" for math calculations. Neural networks are essentially enormous amounts of math (multiplication and addition), so they benefit hugely from this parallelism.

---

## What is BLAS?

**BLAS stands for Basic Linear Algebra Subprograms.**

This is one of the most important concepts in this entire codebase, so let's break it down completely from scratch.

### Linear Algebra — What Is It?

Linear algebra is math with tables of numbers. Instead of working with single numbers like `5 + 3 = 8`, you work with entire grids of numbers at once. These grids are called **matrices**.

A matrix is just a rectangle of numbers:

```
| 1  2  3 |
| 4  5  6 |
| 7  8  9 |
```

This is a 3×3 matrix (3 rows, 3 columns).

### Why Does a Neural Network Need This?

A neural network layer does exactly one thing: it multiplies an input matrix by a weight matrix. That's it. The entire "intelligence" of a neural network comes from carefully chosen weight values. The math looks like:

```
Output = Input × Weights + Bias
```

If you have 64 inputs and 128 neurons, your input is a `64×1` matrix and your weights are a `128×64` matrix. Multiplying them produces a `64×128` output. This is **matrix multiplication**, and it happens billions of times per second in modern AI.

### So What is BLAS?

BLAS is a collection of **highly optimized, battle-tested routines** for doing exactly this kind of math as fast as physically possible. Think of BLAS like a professional race car driver — you could drive the route yourself, but the professional will always be faster because they've practiced every corner thousands of times and know every trick.

BLAS has been around since the 1970s. Thousands of engineers have spent decades making it as fast as possible on every type of hardware. When you use BLAS, you're getting all of that for free.

### BLAS Levels

BLAS is organized into three levels based on how much data you're working with:

- **Level 1** — Operations on single vectors (1D arrays): `y = alpha * x + y`
- **Level 2** — Matrix × vector operations: `y = alpha * A * x + beta * y`
- **Level 3** — Matrix × matrix operations: `C = alpha * A * B + beta * C`

This code uses **Level 3 BLAS**, specifically a function called **GEMM**, because neural networks need full matrix × matrix multiplication.

### Why Not Just Write Loops?

You could write matrix multiplication yourself with three nested `for` loops:

```
for i in rows_of_A:
    for j in cols_of_B:
        for k in shared_dimension:
            C[i][j] += A[i][k] * B[k][j]
```

This works, but it's extremely slow for large matrices because:
1. It ignores CPU/GPU caching behavior
2. It can't use SIMD (doing 8+ multiplications simultaneously with special hardware)
3. It can't split work across multiple GPU threads efficiently
4. It has no knowledge of the hardware it's running on

BLAS implementations use all of these tricks internally. On a GPU, a good GEMM can run **10-100× faster** than a naive loop.

---

## What is CLBlast?

**CLBlast** is an OpenCL implementation of BLAS. It's a library that provides highly optimized GEMM (and other BLAS routines) specifically for OpenCL-capable GPUs.

Think of it like this:
- **BLAS** = the standard recipe book for linear algebra
- **cuBLAS** = NVIDIA's version of that recipe book, optimized for NVIDIA GPUs
- **CLBlast** = an open-source version that works on ANY OpenCL device (AMD, Intel, NVIDIA, CPU)

CLBlast is loaded as a `.dll` file (Windows dynamic library) and called directly from Rust. The code embeds the DLL inside the binary itself using `include_bytes!`, so users don't need to install anything separately.

### The CLBlast Auto-Tuner

CLBlast has a critical behaviour to understand: **on the very first call for a given matrix shape, it runs an internal benchmark to find the fastest GPU kernel configuration** (tile sizes, work-group sizes, vectorization width, etc.) for your specific hardware. This auto-tuning can take **hundreds to thousands of milliseconds** on the first call. On every subsequent call with the same shape, it uses the cached result and runs at full speed.

This is why `warmup_clblast()` exists — it fires the auto-tuner before any timed measurements begin.

---

## What is GEMM?

**GEMM = GEneral Matrix-Matrix multiplication**

This is the single most important operation in all of deep learning. If you understand GEMM, you understand the heart of neural networks.

### The GEMM Formula

```
C = alpha × (A × B) + beta × C
```

Where:
- `A` is the first input matrix
- `B` is the second input matrix
- `C` is the output matrix (also the starting value that gets scaled by `beta`)
- `alpha` is a scalar multiplier for the product
- `beta` is a scalar multiplier for the existing C values

In most neural network cases:
- `alpha = 1.0` (don't scale the result)
- `beta = 0.0` (ignore whatever was in C before, start fresh)

So it simplifies to: `C = A × B`

### SGEMM vs HGEMM

The prefix tells you the data type:
- **S**GEMM = **S**ingle-precision (FP32) — each number is 32 bits, 4 bytes
- **H**GEMM = **H**alf-precision (FP16) — each number is 16 bits, 2 bytes

FP16 uses half the memory and can be twice as fast on hardware that supports it (like modern GPUs).

### What Does "Transpose" Mean?

When you call GEMM, you can optionally tell it to **transpose** one or both matrices before multiplying. Transposing a matrix means flipping it — rows become columns and columns become rows:

Original matrix:
```
| 1  2  3 |     Transposed:    | 1  4 |
| 4  5  6 |        →           | 2  5 |
                                | 3  6 |
```

In this code, the weights matrix is stored as `(hidden_size × input_size)` — each **row** is one neuron's weights. But for matrix multiplication, we need `(input_size × hidden_size)`. Instead of rearranging memory (which is slow), we just tell GEMM: "use matrix B transposed". That's what `CLBLAST_TRANSPOSE_YES` does.

### Row-Major Layout

```rust
const CLBLAST_LAYOUT_ROW_MAJOR: i32 = 101;
```

This tells CLBlast how the matrix is stored in memory. **Row-major** means the elements of each row are stored next to each other:

```
Matrix:          Memory layout:
| 1  2  3 |     [1, 2, 3, 4, 5, 6, 7, 8, 9]
| 4  5  6 |      row0      row1      row2
| 7  8  9 |
```

Row-major is the default in C, C++, and Rust. The alternative (column-major) is used in Fortran and MATLAB. You must always tell BLAS which one you're using or you'll get wrong results.

### The "Leading Dimension" (ld) Parameters

In the GEMM call, you see parameters called `a_ld`, `b_ld`, `c_ld`. These are the **leading dimensions** — basically, how many elements you need to skip to get from one row to the next.

For a matrix stored densely (no gaps), this is just the number of columns:
- For `input` (batch_size × input_size): `a_ld = input_size`
- For `weights1` (hidden_size × input_size), transposed: `b_ld = input_size`
- For `hidden` (batch_size × hidden_size): `c_ld = hidden_size`

This matters because BLAS can work on sub-sections of larger matrices without copying data — it just uses the leading dimension to find each row.

---

## Import Statements

```rust
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
```
- **CommandQueue**: A queue where you submit tasks for the GPU. Tasks execute in order.
- **CL_QUEUE_PROFILING_ENABLE**: A flag that enables timing measurements on GPU operations.

```rust
use opencl3::context::Context;
```
- **Context**: The "workspace" that connects your Rust code to the GPU. It manages all memory and programs.

```rust
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
```
- **get_all_devices**: Finds all available compute devices on the system.
- **Device**: Represents a single GPU or CPU that can run OpenCL.

```rust
use opencl3::memory::{Buffer, ClMem, CL_MEM_READ_WRITE};
```
- **Buffer**: A chunk of memory that lives on the GPU.
- **CL_MEM_READ_WRITE**: Allows the GPU to both read from and write to this buffer.

```rust
use std::ffi::c_void;
```
- **c_void**: A raw pointer type used when calling C functions (like CLBlast) that don't know about Rust types.

```rust
use std::sync::{Mutex, OnceLock};
```
- **Mutex**: A lock that prevents two threads from touching the same data simultaneously. Essential for thread safety.
- **OnceLock**: A value that is initialized exactly once and then stays constant forever.

---

## CLBlast DLL Loading

This section handles loading the CLBlast library at runtime.

### Embedding the DLL

```rust
pub const MY_DLL: &[u8] = include_bytes!("../clblast.dll");
```

**`include_bytes!`** reads a file at **compile time** and bakes its raw bytes directly into the executable. This means the `clblast.dll` file becomes part of the `.exe` itself. When the program runs, it extracts the DLL to a temporary folder and loads it. Users never need to install CLBlast manually.

### CLBlast Constants

```rust
const CLBLAST_LAYOUT_ROW_MAJOR: i32 = 101;
const CLBLAST_TRANSPOSE_NO: i32 = 111;
const CLBLAST_TRANSPOSE_YES: i32 = 112;
```

These are integer codes that CLBlast uses to specify how matrices are arranged and whether to transpose them. The numbers come directly from the CLBlast C header file — they match exactly what the DLL expects.

### Function Type Definitions

```rust
type CLBlastSgemmFn = unsafe extern "C" fn(
    layout: i32,
    a_transpose: i32,
    b_transpose: i32,
    m: usize,       // rows of A and C
    n: usize,       // cols of B and C
    k: usize,       // cols of A and rows of B
    alpha: f32,     // scalar multiplier for A*B
    a_buffer: *mut c_void,  // GPU memory pointer to A
    a_offset: usize,        // where A starts in the buffer
    a_ld: usize,            // leading dimension of A
    b_buffer: *mut c_void,  // GPU memory pointer to B
    b_offset: usize,
    b_ld: usize,
    beta: f32,      // scalar multiplier for existing C
    c_buffer: *mut c_void,  // GPU memory pointer to C (output)
    c_offset: usize,
    c_ld: usize,
    queue: *mut *mut c_void,  // OpenCL command queue
    event: *mut cl_event,     // optional event for synchronization
) -> i32;  // returns 0 on success
```

This defines the **exact function signature** that CLBlast's `CLBlastSgemm` function has. The `extern "C"` tells Rust to use C calling conventions instead of Rust calling conventions. Every parameter must match what the DLL expects, or the program will crash.

`CLBlastHgemmFn` is identical except `alpha` and `beta` are `u16` (half-precision floats stored as unsigned 16-bit integers).

### The ClBlastLib Struct

```rust
struct ClBlastLib {
    _lib: libloading::Library,  // keeps the DLL loaded in memory
    sgemm: CLBlastSgemmFn,      // function pointer to CLBlastSgemm
    hgemm: CLBlastHgemmFn,      // function pointer to CLBlastHgemm
}
```

This struct **owns** the loaded DLL. The `_lib` field must stay alive — if it gets dropped (freed), the DLL gets unloaded and the function pointers `sgemm` and `hgemm` become dangling pointers (pointing to nothing). The underscore prefix `_lib` tells Rust "I'm not using this directly, but keep it alive."

```rust
unsafe impl Send for ClBlastLib {}
unsafe impl Sync for ClBlastLib {}
```

These tell Rust's thread safety system that it's okay to send `ClBlastLib` between threads and share it between threads. We need `unsafe` because Rust can't automatically verify this — we're manually promising that CLBlast's functions are thread-safe (they are, as long as you use different queues per thread).

### Extracting and Loading the DLL

```rust
fn get_clblast_dll_path() -> std::path::PathBuf {
    CLBLAST_DLL_PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join("parallel_project_clblast");
        std::fs::create_dir_all(&dir).ok();
        let dll_path = dir.join("clblast.dll");
        if !dll_path.exists()
            || std::fs::metadata(&dll_path).map(|m| m.len()).unwrap_or(0) != MY_DLL.len() as u64
        {
            std::fs::write(&dll_path, MY_DLL).expect("Failed to write clblast.dll");
        }
        dll_path
    }).clone()
}
```

This function:
1. Creates a folder in the system's temp directory (e.g., `C:\Users\...\AppData\Local\Temp\parallel_project_clblast`)
2. Checks if the DLL is already there AND has the right file size (a quick integrity check)
3. If not, writes the embedded DLL bytes to disk
4. Returns the path to the DLL file

The `OnceLock` ensures this only runs once even if called from multiple threads simultaneously.

```rust
fn load_clblast() -> Result<ClBlastLib, String> {
    let dll_path = get_clblast_dll_path();
    unsafe {
        let lib = libloading::Library::new(&dll_path)...;
        let sgemm_sym = lib.get(b"CLBlastSgemm")...;
        let hgemm_sym = lib.get(b"CLBlastHgemm")...;
        Ok(ClBlastLib { _lib: lib, sgemm: *sgemm_sym, hgemm: *hgemm_sym })
    }
}
```

This is called **dynamic linking** — instead of knowing where `CLBlastSgemm` is at compile time, we search for it by name at runtime inside the DLL file. `lib.get(b"CLBlastSgemm")` looks up the symbol named `CLBlastSgemm` in the DLL's export table and gives us its memory address. We then store that address as a function pointer.

---

## Data Structures

### InferenceMetrics

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub execution_time_ms: f64,      // how long the GPU took, in milliseconds
    pub memory_bandwidth_gbps: f64,  // how fast memory was read/written (GB/s)
    pub throughput_gflops: f64,      // how many billion math operations per second
    pub memory_footprint_mb: f64,    // total GPU memory used, in megabytes
    pub accuracy_mse: f64,           // Mean Squared Error vs FP32 reference
    pub accuracy_max_error: f64,     // worst single output difference vs FP32
}
```

Every inference mode produces one of these. The `Serialize`/`Deserialize` derives let Tauri automatically convert this struct to JSON for the frontend.

### StreamingData

```rust
pub struct StreamingData {
    pub timestamp: f64,
    pub execution_time_ms: f64,
    pub throughput_gflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub accuracy_mse: f64,
}
```

A lighter structure used for real-time chart streaming — only the fields that charts need.

### ComparisonMetrics

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub fp32: InferenceMetrics,
    pub fp16: InferenceMetrics,
    pub fp16_scaled: InferenceMetrics,
    pub clblast_fp32: InferenceMetrics,
    pub clblast_fp16: InferenceMetrics,
    pub clblast_mixed: InferenceMetrics,
}
```

This bundles all **six** inference results into one package that gets sent to the frontend in a single call. The frontend receives this and populates all six chart series at once.

---

## Device Selection

```rust
fn pick_device() -> Result<cl_device_id, String> {
    let gpu_devices = get_all_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
    if let Some(id) = gpu_devices.first() {
        return Ok(*id);
    }
    let cpu_devices = get_all_devices(CL_DEVICE_TYPE_CPU).unwrap_or_default();
    if let Some(id) = cpu_devices.first() {
        return Ok(*id);
    }
    Err("No OpenCL devices found".to_string())
}
```

Tries to find a GPU first. If no GPU supports OpenCL, falls back to the CPU. Returns the device's ID — a raw pointer/handle that OpenCL uses to identify hardware.

---

## OpenCL Kernels

A **kernel** is a small program written in OpenCL C that gets compiled and runs on the GPU. Unlike normal CPU programs, kernels are data-parallel: the same code runs on thousands of GPU threads simultaneously, each thread handling a different piece of data.

### The Two-Pass Design (Why All Custom Kernels Are Split)

All three custom kernels (FP32, FP16, FP16 Scaled) use a **two-pass design**: a separate kernel for layer 1 and a separate kernel for layer 2. This is the correct way to implement a two-layer MLP on a GPU.

**Why two passes?**

The MLP forward pass has a dependency: layer 2 needs the complete hidden layer output from layer 1 before it can do anything. If you try to squeeze both layers into a single kernel dispatch, every output thread has to recompute the entire hidden layer independently:

```
// WRONG (old design): dispatched over batch_size × output_size threads
// Thread 0 (batch=0, out=0):  computes ALL hidden[0..N], then output[0]
// Thread 1 (batch=0, out=1):  computes ALL hidden[0..N] AGAIN, then output[1]
// Thread 2 (batch=0, out=2):  computes ALL hidden[0..N] AGAIN, then output[2]
// ...output_size threads, each doing the full hidden layer redundantly
```

For `matrix_size=1024` (output_size=512), that's **512× more hidden layer work than necessary**. The results were technically correct (all threads compute identical values deterministically), but the GPU was doing hundreds of times more work than needed, and the GFLOPS metric was proportionally understated.

**The fix — two separate dispatches:**

```
Pass 1: dispatch batch_size × hidden_size threads
  Thread (b, h): computes hidden_buffer[b][h]  ← each hidden neuron computed exactly once

queue.finish()  ← wait until ALL hidden values are written before layer 2 reads them

Pass 2: dispatch batch_size × output_size threads
  Thread (b, o): reads hidden_buffer[b][*] to compute output[b][o]
```

Each hidden neuron is now computed exactly once. The `queue.finish()` between passes is mandatory — without it, layer 2 threads could start reading `hidden_buffer` before layer 1 has finished writing to it.

---

### FP32 Kernel (`FP32_KERNEL`) — Layer 1

```opencl
__kernel void mlp_fp32_layer1(
    __global const float* input,
    __global const float* weights1,
    __global const float* bias1,
    __global float* hidden_buffer,
    const int input_size,
    const int hidden_size,
    const int batch_size
) {
    int gid = get_global_id(0);
    if (gid >= batch_size * hidden_size) return;

    int batch_idx = gid / hidden_size;
    int h = gid % hidden_size;

    float sum = bias1[h];
    for (int i = 0; i < input_size; i++) {
        sum += input[batch_idx * input_size + i] * weights1[h * input_size + i];
    }
    hidden_buffer[batch_idx * hidden_size + h] = fmax(0.0f, sum); // ReLU
}
```

- Dispatched over `batch_size × hidden_size` threads — one thread per hidden neuron per batch item
- `gid / hidden_size` gives the batch index, `gid % hidden_size` gives which hidden neuron
- Computes the dot product of the input with one row of `weights1`, adds `bias1[h]`, applies ReLU
- Writes exactly one value to `hidden_buffer` — no races, no redundancy

Weight layout: `weights1[h * input_size + i]` means row `h` contains all connections from every input to hidden neuron `h`. This is standard row-major weight storage.

### FP32 Kernel (`FP32_KERNEL`) — Layer 2

```opencl
__kernel void mlp_fp32_layer2(
    __global const float* hidden_buffer,
    __global const float* weights2,
    __global const float* bias2,
    __global float* output,
    const int hidden_size,
    const int output_size,
    const int batch_size
) {
    int gid = get_global_id(0);
    if (gid >= batch_size * output_size) return;

    int batch_idx = gid / output_size;
    int out_idx = gid % output_size;

    float sum = bias2[out_idx];
    for (int h = 0; h < hidden_size; h++) {
        sum += hidden_buffer[batch_idx * hidden_size + h] * weights2[out_idx * hidden_size + h];
    }
    output[batch_idx * output_size + out_idx] = sum;
}
```

- Dispatched over `batch_size × output_size` threads — one thread per output neuron per batch item
- Reads from `hidden_buffer` (already fully populated by layer 1) — no ReLU on the output layer

---

### FP16 Kernel (`FP16_KERNEL`)

```opencl
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
```

FP16 (half-precision) is not universally supported on all OpenCL devices. This pragma **enables the extension** if available. Without it, you can't use the `half` data type in OpenCL C.

The FP16 kernel is structurally identical to FP32 — two passes, same logic — just replacing `float` with `half` throughout. The kernels are named `mlp_fp16_layer1` and `mlp_fp16_layer2`. The GPU automatically handles the narrower arithmetic. Trade-off: less precision, half the memory.

---

### FP16 Scaled Kernel (`FP16_SCALED_KERNEL`)

This is the most sophisticated custom kernel. It extends FP16 with **row-wise quantization scaling** to reduce numerical errors.

**The problem with plain FP16:** values larger than ~65504 become infinity, and very small values lose precision. **The solution:** divide each row of weights by its maximum absolute value before converting to FP16 (normalizing all values to `[-1.0, 1.0]`), then multiply back by the scale factor during computation.

**Layer 1** (`mlp_fp16_scaled_layer1`):

```opencl
__kernel void mlp_fp16_scaled_layer1(
    __global const half* input,
    __global const half* weights1,   // pre-divided by scales1[h] before upload
    __global const float* scales1,   // one scale per hidden neuron (FP32 for precision)
    __global const half* bias1,
    __global float* hidden_buffer,   // FP32 accumulation for higher precision
    const int input_size,
    const int hidden_size,
    const int batch_size
)
```

```opencl
float sum = (float)bias1[h];
float scale = scales1[h];    // the max_abs value for this row of weights
for (int i = 0; i < input_size; i++) {
    // weights1[h,i] was stored as original_weight / scale
    // multiplying back by scale recovers the original magnitude
    sum += (float)input[...] * (float)weights1[h * input_size + i] * scale;
}
hidden_buffer[batch_idx * hidden_size + h] = fmax(0.0f, sum);
```

The hidden buffer uses `float` (FP32) to accumulate results with full precision, even though weights and inputs came from FP16. The bias is added directly without scaling — it's a separate term that doesn't participate in quantization. This is a form of **mixed precision within a single kernel**: store cheaply in FP16, accumulate expensively in FP32.

**Layer 2** (`mlp_fp16_scaled_layer2`) follows the same pattern for the output layer, reading from the FP32 hidden buffer and writing a FP16 output.

---

### Bias + ReLU Kernels (`BIAS_RELU_FP32_KERNEL`, `BIAS_RELU_FP16_KERNEL`)

These are **helper kernels** used only with CLBlast. BLAS's GEMM does the matrix multiplication but doesn't know about neural network biases or activation functions. So after GEMM runs, a separate OpenCL kernel adds the bias and applies ReLU.

```opencl
__kernel void add_bias_relu_fp32(
    __global float* data,        // the matrix to modify (in-place)
    __global const float* bias,  // bias values (one per output neuron)
    const int cols,              // number of columns (= number of neurons)
    const int total              // total elements = rows * cols
) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    int col = gid % cols;
    data[gid] = fmax(0.0f, data[gid] + bias[col]);
}
```

`gid % cols` figures out which neuron (column) this element corresponds to so it knows which bias value to add. Each element gets the bias of its neuron. There's also `add_bias_fp32` (same but without ReLU) — used for the output layer.

---

### FP16-to-FP32 Conversion Kernel (`FP16_TO_FP32_KERNEL`)

```opencl
__kernel void convert_fp16_to_fp32(
    __global const half* input,
    __global float* output,
    const int total
) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    output[gid] = (float)input[gid];  // GPU hardware handles the conversion
}
```

Converts a buffer of FP16 values to FP32 entirely on the GPU. Used by the **CLBlast Mixed** mode which stores data in FP16 but computes with SGEMM (which needs FP32). A matching `convert_fp32_to_fp16` kernel exists in the same source string.

---

## Float Conversion Functions

### `f32_to_f16(val: f32) -> u16`

```rust
fn f32_to_f16(val: f32) -> u16 {
    half::f16::from_f32(val).to_bits()
}
```

Uses the **`half` crate** to convert a 32-bit float to a 16-bit float, returned as a `u16` (since Rust has no native `f16` type at stable release time). `half::f16::from_f32` handles all the edge cases: NaN, infinity, subnormals, overflow to infinity, and underflow to zero. `.to_bits()` extracts the raw 16-bit IEEE 754 bit pattern, which is what OpenCL and CLBlast expect in their `u16` buffers.

### `f16_to_f32(val: u16) -> f32`

```rust
fn f16_to_f32(val: u16) -> f32 {
    half::f16::from_bits(val).to_f32()
}
```

The reverse: takes a raw 16-bit bit pattern, interprets it as an FP16 float, and expands it to FP32. Used after reading FP16 GPU output back to the CPU for accuracy comparison.

**Why the `half` crate instead of manual bit manipulation?**

The previous version of this code had ~55 lines of hand-written bit manipulation to handle the FP32→FP16 conversion. The `half` crate replaces that with two one-liners backed by well-tested, IEEE-754-correct code that handles every edge case correctly.

---

## RoundData - Shared Test Data

```rust
struct RoundData {
    input_data: Vec<f32>,
    weights1: Vec<f32>,
    bias1: Vec<f32>,
    weights2: Vec<f32>,
    bias2: Vec<f32>,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    batch_size: usize,
}
```

This struct is critical for fair comparisons. One `RoundData` is generated and shared across all six inference modes. Every mode gets the **exact same inputs and weights** — so the only thing being measured is the precision format and compute path, not differences in the data.

### `RoundData::generate()`

```rust
let input_data: Vec<f32> = (0..batch_size * input_size)
    .map(|_| rand::random::<f32>())
    .collect();

let weights1: Vec<f32> = (0..hidden_size * input_size)
    .map(|i| (i as f32 * 0.1).sin() * 0.5)
    .collect();
```

- **Input data**: Truly random values between 0.0 and 1.0 (different every round).
- **Weights**: Deterministic using `sin()` — varied values in `[-0.5, 0.5]`, a reasonable range for initialized weights.
- **Biases**: Small linear values (`i * 0.01`) — near-zero, standard neural network initialization.

The weight layout is **row-major**: `weights1[h * input_size + i]` is the weight connecting input `i` to hidden neuron `h`. `weights2[o * hidden_size + h]` is the weight connecting hidden neuron `h` to output `o`.

---

## MLPInference Structure

```rust
struct MLPInference {
    context: Context,                     // OpenCL workspace
    queue: CommandQueue,                  // GPU task queue
    device: Device,                       // the GPU/CPU being used
    fp32_reference: Option<Vec<f32>>,     // FP32 output (used as accuracy baseline)
    clblast: Option<ClBlastLib>,          // CLBlast library (None if loading failed)
    bias_fp32_program: Option<Program>,   // pre-compiled FP32 bias kernel
    bias_fp16_program: Option<Program>,   // pre-compiled FP16 bias kernel
    convert_program: Option<Program>,     // pre-compiled FP16<->FP32 conversion kernel
    warmed_up_sizes: HashSet<usize>,      // tracks which matrix sizes have been warmed up
}
```

### Why Cache Programs?

OpenCL programs (kernels) must be compiled before use — just like Rust code must be compiled before it runs. This compilation takes time. The bias and conversion kernels (used by all three CLBlast modes) are compiled **once** during `new()` and reused on every call. The main inference kernels (FP32, FP16, FP16-scaled) are compiled per-call since they're only used once each per comparison round.

### `warmed_up_sizes` — Tracking CLBlast Auto-Tuning

```rust
warmed_up_sizes: std::collections::HashSet<usize>,
```

CLBlast runs an auto-tuning benchmark on its very first GEMM call for each unique matrix shape. This can take over a second on the first call, then drops to sub-millisecond on all subsequent calls. `warmed_up_sizes` records which `matrix_size` values have already paid the auto-tuning cost. When `warmup_clblast(matrix_size)` is called, it immediately returns if the size is already in the set.

### `MLP_INSTANCE` — The Singleton

```rust
static MLP_INSTANCE: Mutex<Option<MLPInference>> = Mutex::new(None);
```

`MLPInference` holds an active GPU context, which is expensive to create (hundreds of milliseconds). This static `Mutex<Option<...>>` stores one instance globally. The first call creates it; all subsequent calls reuse it. The `Mutex` prevents multiple threads from trying to initialize it simultaneously.

### `get_or_init_mlp()`

```rust
fn get_or_init_mlp() -> Result<MutexGuard<'static, Option<MLPInference>>, String> {
    let mut guard = MLP_INSTANCE.lock()...;
    if guard.is_none() {
        *guard = Some(MLPInference::new()?);
    }
    Ok(guard)
}
```

Acquires the mutex lock, initializes the MLP if it doesn't exist yet, and returns the guard. The guard keeps the mutex locked until it goes out of scope, preventing any other call from running concurrently.

### `warmup_clblast(matrix_size)`

```rust
fn warmup_clblast(&mut self, matrix_size: usize) {
    if self.warmed_up_sizes.contains(&matrix_size) {
        return;  // already paid the cost for this size
    }
    // ... allocate scratch buffers and run one untimed SGEMM + HGEMM for each shape
    self.warmed_up_sizes.insert(matrix_size);
}
```

This method fires the CLBlast auto-tuner for all four GEMM shapes used by the given `matrix_size`:
- SGEMM layer 1: `(batch_size=64, hidden_size, input_size)`
- SGEMM layer 2: `(batch_size=64, output_size, hidden_size)`
- HGEMM layer 1: same shape as SGEMM layer 1
- HGEMM layer 2: same shape as SGEMM layer 2

Scratch buffers are allocated with uninitialised GPU memory (contents don't matter — CLBlast only needs valid buffer handles to tune against the shape). All four calls complete before `warmup_clblast` returns, guaranteeing the cache is warm before any timed inference runs.

---

## Inference Functions

All six inference functions follow the same overall pattern:

1. **Allocate GPU buffers** — reserve memory on the GPU
2. **Write data to GPU** — transfer from CPU RAM to GPU VRAM
3. **Execute the computation** — run the kernel(s) or BLAS call(s)
4. **Read results back** — transfer output from GPU VRAM to CPU RAM
5. **Calculate metrics** — time, GFLOPS, bandwidth, accuracy

### `calculate_accuracy(&self, output: &[f32], reference: &[f32]) -> (f64, f64)`

```rust
for (out, ref_val) in output.iter().zip(reference.iter()) {
    let diff = (out - ref_val).abs();
    mse += (diff * diff) as f64;
    max_error = max_error.max(diff as f64);
}
mse /= output.len() as f64;
```

Compares any inference mode's results against the FP32 reference output, returning:
- **MSE (Mean Squared Error)**: Average of all squared differences — squaring magnifies larger errors
- **Max Error**: The single biggest difference found anywhere in the output

FP32 and CLBlast FP32 both report MSE = 0.0. FP32 is the reference itself (nothing to compare against), and CLBlast FP32 uses identical FP32 arithmetic so its results are equivalent.

---

### `run_fp32_inference()` — The Baseline

Uses the two-pass OpenCL FP32 kernel (`mlp_fp32_layer1` → `mlp_fp32_layer2`). After running, stores its output in `self.fp32_reference` so all subsequent modes can compare against it.

**Execution flow:**
1. Upload all data (input, weights1, bias1, weights2, bias2) to GPU buffers
2. Compile `FP32_KERNEL` (contains both layer functions)
3. Create `mlp_fp32_layer1` kernel, set 7 args, dispatch `batch_size × hidden_size` threads
4. `queue.finish()` — wait for all hidden values to be written
5. Create `mlp_fp32_layer2` kernel, set 7 args, dispatch `batch_size × output_size` threads
6. `queue.finish()` — wait for output
7. Read output back to CPU, store as `fp32_reference`

**Memory footprint:**
```rust
((batch_size * input_size       // input
  + hidden_size * input_size    // weights1
  + hidden_size                 // bias1
  + output_size * hidden_size   // weights2
  + output_size                 // bias2
  + batch_size * output_size)   // output
  * 4)  // 4 bytes per FP32 value
```

**FLOPS calculation:**
```
batch_size × (hidden_size × (2 × input_size + 1) + output_size × (2 × hidden_size + 1))
```

The `2×` factor accounts for one multiply and one add per weight. The `+1` accounts for the bias add. This formula now correctly counts each operation once — the two-pass design ensures no hidden layer redundancy.

---

### `run_fp16_inference()` — Half Precision

Before anything goes to the GPU, all FP32 data gets converted to FP16 on the CPU:
```rust
let input_data: Vec<u16> = rd.input_data.iter().map(|&v| f32_to_f16(v)).collect();
let weights1: Vec<u16> = rd.weights1.iter().map(|&v| f32_to_f16(v)).collect();
```

Buffers are `Buffer::<u16>` instead of `Buffer::<f32>`. The two-pass FP16 kernel (`mlp_fp16_layer1` + `mlp_fp16_layer2`) operates entirely in half-precision.

After reading results back, they're converted to FP32 for accuracy comparison:
```rust
let output_f32: Vec<f32> = output.iter().map(|&v| f16_to_f32(v)).collect();
let (accuracy_mse, accuracy_max_error) = self.calculate_accuracy(&output_f32, reference);
```

Memory footprint uses `* 2` instead of `* 4` — half the bytes per value.

---

### `run_fp16_scaled_inference()` — FP16 with Row-Wise Quantization

Before GPU transfer, computes per-row scale factors on the CPU:
```rust
for h in 0..hidden_size {
    let row = &weights1_f32[h * input_size .. (h+1) * input_size];
    let max_abs = row.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
    scales1[h] = if max_abs > 0.0 { max_abs } else { 1.0 };

    for i in 0..input_size {
        // normalize to [-1, 1] before converting to FP16
        weights1_scaled[h * input_size + i] = f32_to_f16(weights1_f32[row_start + i] / scales1[h]);
    }
}
```

Every weight value is divided by its row's maximum absolute value before FP16 conversion. This normalizes all values to `[-1.0, 1.0]`, which FP16 can represent with much higher precision than arbitrary large/small values.

The scale factors themselves are kept as FP32 arrays (`scales1`, `scales2`) and uploaded to the GPU separately. During kernel execution, the GPU multiplies the normalized FP16 weight by the FP32 scale to recover the original magnitude.

Memory footprint is slightly higher than plain FP16 due to the FP32 scale arrays:
```rust
(all tensors * 2)               // FP16 storage
+ (hidden_size + output_size) * 4  // scale factors in FP32
```

---

## CLBlast Inference Functions

These three functions replace the manual loop kernels with highly optimized BLAS matrix multiplication calls. The core difference: CLBlast handles the matrix multiply (the expensive inner loop), while small custom OpenCL kernels handle the bias addition and ReLU (which BLAS doesn't know about).

### `run_clblast_fp32_inference()` — SGEMM

```rust
let clblast = self.clblast.as_ref().ok_or("CLBlast not loaded")?;
let sgemm = clblast.sgemm;
```

First, we get the SGEMM function pointer. If CLBlast failed to load, this returns an error immediately.

**Layer 1 SGEMM call:**
```rust
let status = sgemm(
    CLBLAST_LAYOUT_ROW_MAJOR,     // matrices stored row by row
    CLBLAST_TRANSPOSE_NO,          // A (input) — no transpose
    CLBLAST_TRANSPOSE_YES,         // B (weights1) — transpose it
    batch_size,                    // M: rows of A and C
    hidden_size,                   // N: cols of B after transpose, cols of C
    input_size,                    // K: cols of A = rows of B before transpose
    1.0f32,                        // alpha = 1.0 (no scaling of the result)
    input_buf.get(),               // A: (batch_size × input_size) on GPU
    0,                             // A starts at offset 0
    input_size,                    // leading dimension of A
    weights1_buf.get(),            // B: (hidden_size × input_size) on GPU
    0,                             // B starts at offset 0
    input_size,                    // leading dimension of B (before transpose)
    0.0f32,                        // beta = 0.0 (ignore old C values)
    hidden_buf.get(),              // C: output goes here (batch_size × hidden_size)
    0,                             // C starts at offset 0
    hidden_size,                   // leading dimension of C
    &mut queue_ptr,                // OpenCL queue CLBlast submits work to
    &mut event,                    // synchronization event (unused here)
);
```

This computes `hidden = input × weights1^T`, giving one hidden activation vector per batch item.

**After SGEMM — bias + ReLU (using cached pre-compiled kernel):**
```rust
let program = self.bias_fp32_program.as_ref().ok_or("FP32 bias program not compiled")?;
let bias_relu_kernel = Kernel::create(program, "add_bias_relu_fp32")?;
```

The bias kernel runs over all `batch_size × hidden_size` elements, adding `bias1[col]` and applying ReLU in-place. Layer 2 follows the same pattern with `add_bias_fp32` (no ReLU on the output layer).

**CLBlast FP32 accuracy:** hardcoded to `accuracy_mse = 0.0` and `accuracy_max_error = 0.0`. CLBlast SGEMM uses the same FP32 arithmetic as the reference kernel — the results are numerically equivalent, so computing the MSE would just be measuring floating-point operation ordering noise.

### `run_clblast_fp16_inference()` — HGEMM

The FP16 BLAS version. All data is converted to FP16 before upload. Instead of `sgemm`, it calls `hgemm`:

```rust
let alpha_h = f32_to_f16(1.0);  // 1.0 as a FP16 bit pattern (u16)
let beta_h  = f32_to_f16(0.0);  // 0.0 as a FP16 bit pattern (u16)

let status = hgemm(
    CLBLAST_LAYOUT_ROW_MAJOR,
    CLBLAST_TRANSPOSE_NO,
    CLBLAST_TRANSPOSE_YES,
    batch_size, hidden_size, input_size,
    alpha_h,          // u16 instead of f32
    input_buf.get(),  // Buffer<u16> instead of Buffer<f32>
    ...
);
```

`alpha` and `beta` are passed as `u16` (FP16 bit patterns) — this is what the CLBlast HGEMM signature requires. The bias kernels use the FP16 variants (`add_bias_relu_fp16`, `add_bias_fp16`).

### `run_clblast_mixed_inference()` — FP16 Storage + FP32 Compute

The most sophisticated mode. Idea: save bandwidth with FP16 storage, gain FP32 accuracy during compute.

**Pipeline:**

1. **Store data as FP16** (compact, half the upload bandwidth):
   ```rust
   let input_data_fp16: Vec<u16> = rd.input_data.iter().map(|&v| f32_to_f16(v)).collect();
   ```

2. **Allocate both FP16 storage buffers and FP32 compute buffers on the GPU:**
   ```rust
   let input_fp16_buf  = Buffer::<u16>::create(...);  // for upload
   let input_f32_buf   = Buffer::<f32>::create(...);  // for SGEMM
   ```

3. **Upload FP16 data** (half the bytes → faster PCIe transfer)

4. **Convert FP16 → FP32 on the GPU** using the cached conversion kernel (fast, avoids round-tripping through the CPU):
   ```rust
   let fp16_to_fp32_kernel = Kernel::create(convert_prog, "convert_fp16_to_fp32")?;
   ```

5. **Run SGEMM on the FP32 buffers** — full precision, BLAS-optimized

6. **Add bias + ReLU** using FP32 kernel

7. **Repeat for layer 2** (convert weights2 FP16→FP32, then SGEMM)

**The trade-offs:**
- Upload bandwidth: FP16 (fast)
- Compute: FP32 SGEMM (accurate)
- Total GPU memory: highest of all modes (holds both FP16 and FP32 buffers simultaneously)
- Accuracy: very close to pure FP32 (only error comes from the FP16 storage quantization)

**Memory footprint:**
```rust
let fp16_storage =
    (batch_size * input_size + hidden_size * input_size + output_size * hidden_size) * 2;
let fp32_compute =
    (batch_size * input_size      // FP32 input for SGEMM
     + hidden_size * input_size   // FP32 weights1
     + output_size * hidden_size  // FP32 weights2
     + hidden_size + output_size  // biases (always FP32)
     + batch_size * hidden_size   // hidden layer buffer
     + batch_size * output_size)  // output buffer
    * 4;
```

---

## Command Handlers

### `run_inference()` — Single Mode (Async)

```rust
#[tauri::command]
async fn run_inference(precision: String, matrix_size: usize) -> Result<InferenceMetrics, String>
```

The `#[tauri::command]` attribute exposes this function to the JavaScript frontend. Accepts a precision string (`"Fp32"`, `"Fp16"`, `"FP16 + scale"`, `"CLBlast FP32"`, `"CLBlast FP16"`, `"CLBlast Mixed"`) and runs only that one mode.

Always ensures the FP32 reference is established first:
```rust
if precision != "Fp32" && mlp.fp32_reference.is_none() {
    let _ = mlp.run_fp32_inference(&rd)?;
}
```

Uses `match` to dispatch to the correct inference function.

### `run_comparison_inference()` — All Six Modes

```rust
#[tauri::command]
fn run_comparison_inference(matrix_size: usize) -> Result<ComparisonMetrics, String>
```

The main entry point for the comparison page. Execution order:

1. Generate one shared `RoundData`
2. **`warmup_clblast(matrix_size)`** — fires the CLBlast auto-tuner before timing starts (no-op if already done for this size)
3. Run FP32 first (establishes `fp32_reference` for accuracy comparison, accuracy = 0.0)
4. Run FP16 (compared against FP32 reference)
5. Run FP16 Scaled (compared against FP32 reference)
6. Run CLBlast FP32 (accuracy hardcoded to 0.0)
7. Run CLBlast FP16 (compared against FP32 reference)
8. Run CLBlast Mixed (compared against FP32 reference)

CLBlast modes use `.unwrap_or_else` for graceful degradation:
```rust
let clblast_fp32_metrics = mlp.run_clblast_fp32_inference(&rd).unwrap_or_else(|e| {
    eprintln!("CLBlast FP32 error: {e}");
    default_metrics()
});
```

If CLBlast isn't available (DLL failed to load, or GPU doesn't support it), the CLBlast chart series show zeros instead of crashing the app.

### `default_metrics()`

Returns an all-zero `InferenceMetrics`. Used as the fallback value when a CLBlast mode fails.

### `get_len()`

```rust
#[tauri::command]
async fn get_len() -> Result<usize, String> {
    Ok(MY_DLL.len())
}
```

Returns the size of the embedded CLBlast DLL. Used for verifying the DLL was embedded correctly.

### `run()` — Application Entry Point

```rust
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            run_inference,
            run_comparison_inference,
            get_len
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

Registers all three Tauri commands so the frontend can call them via `invoke()`. `tauri::generate_handler!` is a macro that generates the glue code.

---

## The Six Precision Modes Compared

| Mode | Storage | Compute | Kernel style | Memory | Accuracy vs FP32 |
|------|---------|---------|--------------|--------|-------------------|
| **FP32** | 32-bit | 32-bit | Two-pass custom | High (baseline) | 0.0 (is the reference) |
| **FP16** | 16-bit | 16-bit | Two-pass custom | ~50% of FP32 | Small rounding error |
| **FP16 Scaled** | 16-bit + FP32 scales | Mixed | Two-pass custom | ~52% of FP32 | Better than plain FP16 |
| **CLBlast FP32** | 32-bit | 32-bit | BLAS SGEMM + bias kernel | Same as FP32 | 0.0 (equivalent precision) |
| **CLBlast FP16** | 16-bit | 16-bit | BLAS HGEMM + bias kernel | ~50% of FP32 | Similar to FP16 |
| **CLBlast Mixed** | 16-bit stored, 32-bit compute | 32-bit | Convert + BLAS SGEMM | Highest (both buffers) | Near FP32 |

### What Makes CLBlast Faster Than the Custom Kernels?

Both approaches now use the same correct two-pass algorithm. The difference is in **how the matrix multiplication inner loop is executed**.

**Custom kernels:** A naive inner loop. Each thread reads one weight at a time from global memory (slow, no coalescing) and accumulates in a single register. Memory access is not optimized for the GPU's cache hierarchy.

**CLBlast GEMM:** Uses advanced GPU-specific optimizations compiled specifically for your hardware:
- **Tiling**: divides matrices into small tiles that fit in fast GPU local (shared) memory, dramatically reducing global memory traffic
- **Vectorized loads**: reads 4 or 8 values at once with `float4`/`float8` instructions
- **Register blocking**: keeps partial sums in registers (fastest possible storage) across many iterations
- **Work-group tuning**: work-group sizes, tile dimensions, and unroll factors are auto-tuned per GPU model

For large matrices, CLBlast SGEMM can be **5–50× faster** than the equivalent naive loop.

---

## Why FP32 and CLBlast FP32 Accuracy Are Always 0.0

**FP32:** It is the reference that everything else is measured against. There is nothing to compare it to — it IS the baseline. Accuracy = 0.0 by definition.

**CLBlast FP32:** Uses SGEMM which is pure FP32 arithmetic, the same precision as the reference kernel. The results are numerically equivalent (differences would only arise from floating-point operation reordering, which is noise smaller than measurement error). Accuracy is hardcoded to 0.0 rather than computed, since computing it would just measure floating-point non-associativity rather than meaningful precision loss.

For all other modes, the MSE tells you how much precision was lost relative to FP32:
- **Near 0.0**: The mode is nearly as accurate as FP32
- **Small but nonzero**: Normal FP16 rounding — acceptable for most applications
- **Large**: Significant precision loss — the mode may be unsuitable for this problem

---

## Summary

The codebase implements a two-layer MLP (Multi-Layer Perceptron) neural network inference benchmark that simultaneously tests six different approaches to GPU computation.

**Three Custom OpenCL Kernels (two-pass design):**
1. **FP32** — full precision, the reference baseline. Two dispatch passes: `mlp_fp32_layer1` (hidden layer) then `mlp_fp32_layer2` (output layer).
2. **FP16** — half precision. Same two-pass structure with `half` data throughout. Uses half the memory.
3. **FP16 Scaled** — half precision with per-row quantization. Weights are normalized to `[-1, 1]` before FP16 conversion; scale factors are stored in FP32 and applied during compute. Hidden buffer uses FP32 for accurate accumulation.

**Three CLBlast BLAS-Accelerated Paths:**
4. **CLBlast FP32** — SGEMM (FP32 matrix multiply) + custom bias/ReLU kernel. Accuracy hardcoded to 0.0.
5. **CLBlast FP16** — HGEMM (FP16 matrix multiply) + custom FP16 bias/ReLU kernel.
6. **CLBlast Mixed** — data uploaded as FP16, converted to FP32 on-GPU, then SGEMM runs in FP32.

**Key design decisions:**
- All six modes use the **same `RoundData`** for fair comparison
- CLBlast auto-tuning is paid upfront via `warmup_clblast()` before timing starts — the first-call spike (~1 second) no longer pollutes the measurements
- Float conversions use the **`half` crate** instead of hand-written bit manipulation
- The CLBlast DLL is **embedded in the binary** via `include_bytes!` — zero install friction
- The singleton `MLP_INSTANCE` reuses the GPU context across all calls — no repeated initialization cost
- All results are sent to a Tauri/JavaScript frontend as JSON through a simple command API