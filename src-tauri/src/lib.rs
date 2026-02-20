use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::Kernel;
use opencl3::memory::{Buffer, ClMem, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_device_id, cl_event, CL_BLOCKING};
use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub const MY_DLL: &[u8] = include_bytes!("../clblast.dll");

// CLBlast constants
const CLBLAST_LAYOUT_ROW_MAJOR: i32 = 101;
const CLBLAST_TRANSPOSE_NO: i32 = 111;
const CLBLAST_TRANSPOSE_YES: i32 = 112;

// CLBlast function types
type CLBlastSgemmFn = unsafe extern "C" fn(
    layout: i32,
    a_transpose: i32,
    b_transpose: i32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_buffer: *mut c_void,
    a_offset: usize,
    a_ld: usize,
    b_buffer: *mut c_void,
    b_offset: usize,
    b_ld: usize,
    beta: f32,
    c_buffer: *mut c_void,
    c_offset: usize,
    c_ld: usize,
    queue: *mut *mut c_void,
    event: *mut cl_event,
) -> i32;

type CLBlastHgemmFn = unsafe extern "C" fn(
    layout: i32,
    a_transpose: i32,
    b_transpose: i32,
    m: usize,
    n: usize,
    k: usize,
    alpha: u16,
    a_buffer: *mut c_void,
    a_offset: usize,
    a_ld: usize,
    b_buffer: *mut c_void,
    b_offset: usize,
    b_ld: usize,
    beta: u16,
    c_buffer: *mut c_void,
    c_offset: usize,
    c_ld: usize,
    queue: *mut *mut c_void,
    event: *mut cl_event,
) -> i32;

struct ClBlastLib {
    _lib: libloading::Library,
    sgemm: CLBlastSgemmFn,
    hgemm: CLBlastHgemmFn,
}

unsafe impl Send for ClBlastLib {}
unsafe impl Sync for ClBlastLib {}

static CLBLAST_DLL_PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

fn get_clblast_dll_path() -> std::path::PathBuf {
    CLBLAST_DLL_PATH
        .get_or_init(|| {
            let dir = std::env::temp_dir().join("parallel_project_clblast");
            std::fs::create_dir_all(&dir).ok();
            let dll_path = dir.join("clblast.dll");
            if !dll_path.exists()
                || std::fs::metadata(&dll_path).map(|m| m.len()).unwrap_or(0) != MY_DLL.len() as u64
            {
                std::fs::write(&dll_path, MY_DLL).expect("Failed to write clblast.dll");
            }
            dll_path
        })
        .clone()
}

fn load_clblast() -> Result<ClBlastLib, String> {
    let dll_path = get_clblast_dll_path();
    unsafe {
        let lib = libloading::Library::new(&dll_path)
            .map_err(|e| format!("Failed to load clblast.dll: {e}"))?;

        let sgemm_sym: libloading::Symbol<CLBlastSgemmFn> = lib
            .get(b"CLBlastSgemm")
            .map_err(|e| format!("Failed to find CLBlastSgemm: {e}"))?;
        let sgemm: CLBlastSgemmFn = *sgemm_sym;

        let hgemm_sym: libloading::Symbol<CLBlastHgemmFn> = lib
            .get(b"CLBlastHgemm")
            .map_err(|e| format!("Failed to find CLBlastHgemm: {e}"))?;
        let hgemm: CLBlastHgemmFn = *hgemm_sym;

        Ok(ClBlastLib {
            _lib: lib,
            sgemm,
            hgemm,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub execution_time_ms: f64,
    pub memory_bandwidth_gbps: f64,
    pub throughput_gflops: f64,
    pub memory_footprint_mb: f64,
    pub accuracy_mse: f64,
    pub accuracy_max_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingData {
    pub timestamp: f64,
    pub execution_time_ms: f64,
    pub throughput_gflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub accuracy_mse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub fp32: InferenceMetrics,
    pub fp16: InferenceMetrics,
    pub fp16_scaled: InferenceMetrics,
    pub clblast_fp32: InferenceMetrics,
    pub clblast_fp16: InferenceMetrics,
    pub clblast_mixed: InferenceMetrics,
}

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

// OpenCL kernel source for FP32 inference — split into two passes so each hidden
// neuron is computed exactly once per batch item instead of output_size times.
const FP32_KERNEL: &str = r#"
// Layer 1: each thread computes one hidden neuron for one batch item
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

// Layer 2: each thread computes one output neuron for one batch item
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
"#;

// OpenCL kernel source for FP16 inference — two-pass design
const FP16_KERNEL: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Layer 1: each thread computes one hidden neuron for one batch item
__kernel void mlp_fp16_layer1(
    __global const half* input,
    __global const half* weights1,
    __global const half* bias1,
    __global half* hidden_buffer,
    const int input_size,
    const int hidden_size,
    const int batch_size
) {
    int gid = get_global_id(0);
    if (gid >= batch_size * hidden_size) return;

    int batch_idx = gid / hidden_size;
    int h = gid % hidden_size;

    half sum = bias1[h];
    for (int i = 0; i < input_size; i++) {
        sum += input[batch_idx * input_size + i] * weights1[h * input_size + i];
    }
    hidden_buffer[batch_idx * hidden_size + h] = fmax((half)0.0, sum); // ReLU
}

// Layer 2: each thread computes one output neuron for one batch item
__kernel void mlp_fp16_layer2(
    __global const half* hidden_buffer,
    __global const half* weights2,
    __global const half* bias2,
    __global half* output,
    const int hidden_size,
    const int output_size,
    const int batch_size
) {
    int gid = get_global_id(0);
    if (gid >= batch_size * output_size) return;

    int batch_idx = gid / output_size;
    int out_idx = gid % output_size;

    half sum = bias2[out_idx];
    for (int h = 0; h < hidden_size; h++) {
        sum += hidden_buffer[batch_idx * hidden_size + h] * weights2[out_idx * hidden_size + h];
    }
    output[batch_idx * output_size + out_idx] = sum;
}
"#;

// Bias + ReLU kernels for CLBlast modes
const BIAS_RELU_FP32_KERNEL: &str = r#"
__kernel void add_bias_relu_fp32(
    __global float* data,
    __global const float* bias,
    const int cols,
    const int total
) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    int col = gid % cols;
    data[gid] = fmax(0.0f, data[gid] + bias[col]);
}

__kernel void add_bias_fp32(
    __global float* data,
    __global const float* bias,
    const int cols,
    const int total
) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    int col = gid % cols;
    data[gid] = data[gid] + bias[col];
}
"#;

const BIAS_RELU_FP16_KERNEL: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void add_bias_relu_fp16(
    __global half* data,
    __global const half* bias,
    const int cols,
    const int total
) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    int col = gid % cols;
    data[gid] = fmax((half)0.0, data[gid] + bias[col]);
}

__kernel void add_bias_fp16(
    __global half* data,
    __global const half* bias,
    const int cols,
    const int total
) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    int col = gid % cols;
    data[gid] = data[gid] + bias[col];
}
"#;

const FP16_TO_FP32_KERNEL: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void convert_fp16_to_fp32(
    __global const half* input,
    __global float* output,
    const int total
) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    output[gid] = (float)input[gid];
}

__kernel void convert_fp32_to_fp16(
    __global const float* input,
    __global half* output,
    const int total
) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    output[gid] = (half)input[gid];
}
"#;

// OpenCL kernel source for FP16 with row-wise scaling — two-pass design
const FP16_SCALED_KERNEL: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Layer 1: each thread computes one hidden neuron with row-wise scale un-quantization
__kernel void mlp_fp16_scaled_layer1(
    __global const half* input,
    __global const half* weights1,
    __global const float* scales1,
    __global const half* bias1,
    __global float* hidden_buffer,
    const int input_size,
    const int hidden_size,
    const int batch_size
) {
    int gid = get_global_id(0);
    if (gid >= batch_size * hidden_size) return;

    int batch_idx = gid / hidden_size;
    int h = gid % hidden_size;

    float sum = (float)bias1[h];
    float scale = scales1[h];
    for (int i = 0; i < input_size; i++) {
        sum += (float)input[batch_idx * input_size + i] * (float)weights1[h * input_size + i] * scale;
    }
    hidden_buffer[batch_idx * hidden_size + h] = fmax(0.0f, sum); // ReLU
}

// Layer 2: each thread computes one output neuron with row-wise scale un-quantization
__kernel void mlp_fp16_scaled_layer2(
    __global const float* hidden_buffer,
    __global const half* weights2,
    __global const float* scales2,
    __global const half* bias2,
    __global half* output,
    const int hidden_size,
    const int output_size,
    const int batch_size
) {
    int gid = get_global_id(0);
    if (gid >= batch_size * output_size) return;

    int batch_idx = gid / output_size;
    int out_idx = gid % output_size;

    float sum = (float)bias2[out_idx];
    float scale = scales2[out_idx];
    for (int h = 0; h < hidden_size; h++) {
        sum += hidden_buffer[batch_idx * hidden_size + h] * (float)weights2[out_idx * hidden_size + h] * scale;
    }
    output[batch_idx * output_size + out_idx] = (half)sum;
}
"#;

fn f32_to_f16(val: f32) -> u16 {
    half::f16::from_f32(val).to_bits()
}

fn f16_to_f32(val: u16) -> f32 {
    half::f16::from_bits(val).to_f32()
}

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

impl RoundData {
    fn generate(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        batch_size: usize,
    ) -> Self {
        let mut input_data: Vec<f32> = Vec::with_capacity(batch_size * input_size);
        for _ in 0..batch_size * input_size {
            input_data.push(rand::random::<f32>());
        }
        let weights1: Vec<f32> = (0..hidden_size * input_size)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();
        let bias1: Vec<f32> = (0..hidden_size).map(|i| i as f32 * 0.01).collect();
        let weights2: Vec<f32> = (0..output_size * hidden_size)
            .map(|i| (i as f32 * 0.15).cos() * 0.5)
            .collect();
        let bias2: Vec<f32> = (0..output_size).map(|i| i as f32 * 0.01).collect();
        Self {
            input_data,
            weights1,
            bias1,
            weights2,
            bias2,
            input_size,
            hidden_size,
            output_size,
            batch_size,
        }
    }
}

struct MLPInference {
    context: Context,
    queue: CommandQueue,
    #[allow(dead_code)]
    device: Device,
    fp32_reference: Option<Vec<f32>>,
    clblast: Option<ClBlastLib>,
    // Cached programs/kernels to avoid recompilation every call
    bias_fp32_program: Option<Program>,
    bias_fp16_program: Option<Program>,
    convert_program: Option<Program>,
    // Tracks which matrix sizes have already paid the CLBlast auto-tuning cost
    warmed_up_sizes: std::collections::HashSet<usize>,
}

unsafe impl Send for MLPInference {}
unsafe impl Sync for MLPInference {}

static MLP_INSTANCE: Mutex<Option<MLPInference>> = Mutex::new(None);

fn get_or_init_mlp() -> Result<std::sync::MutexGuard<'static, Option<MLPInference>>, String> {
    let mut guard = MLP_INSTANCE
        .lock()
        .map_err(|e| format!("MLP lock error: {e}"))?;
    if guard.is_none() {
        *guard = Some(MLPInference::new()?);
    }
    Ok(guard)
}

impl MLPInference {
    fn new() -> Result<Self, String> {
        let device_id = pick_device()?;
        let device = Device::new(device_id);
        let context = Context::from_device(&device).map_err(|e| format!("Context error: {e}"))?;
        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
                .map_err(|e| format!("Queue error: {e}"))?;

        let clblast = load_clblast().ok();

        // Pre-compile bias/convert kernels once
        let bias_fp32_program =
            Program::create_and_build_from_source(&context, BIAS_RELU_FP32_KERNEL, "").ok();
        let bias_fp16_program =
            Program::create_and_build_from_source(&context, BIAS_RELU_FP16_KERNEL, "").ok();
        let convert_program =
            Program::create_and_build_from_source(&context, FP16_TO_FP32_KERNEL, "").ok();

        Ok(Self {
            context,
            queue,
            device,
            fp32_reference: None,
            clblast,
            bias_fp32_program,
            bias_fp16_program,
            convert_program,
            warmed_up_sizes: std::collections::HashSet::new(),
        })
    }

    /// Runs one untimed SGEMM + HGEMM pass for each unique (M,N,K) shape used by the given
    /// matrix_size so that CLBlast's internal auto-tuner fires *before* we start measuring.
    /// Subsequent calls with the same size are a no-op because the result is cached by CLBlast.
    fn warmup_clblast(&mut self, matrix_size: usize) {
        if self.warmed_up_sizes.contains(&matrix_size) {
            return;
        }

        let clblast = match self.clblast.as_ref() {
            Some(c) => c,
            None => return,
        };
        let sgemm = clblast.sgemm;
        let hgemm = clblast.hgemm;

        let input_size = matrix_size;
        let hidden_size = matrix_size;
        let output_size = matrix_size / 2;
        let batch_size = 64usize;

        // Allocate tiny scratch buffers — contents don't matter, we just need valid GPU memory
        // at the right sizes for CLBlast to tune against the real shapes.
        let alloc_f32 = |n: usize| -> Option<Buffer<f32>> {
            unsafe {
                Buffer::<f32>::create(&self.context, CL_MEM_READ_WRITE, n, std::ptr::null_mut())
                    .ok()
            }
        };
        let alloc_u16 = |n: usize| -> Option<Buffer<u16>> {
            unsafe {
                Buffer::<u16>::create(&self.context, CL_MEM_READ_WRITE, n, std::ptr::null_mut())
                    .ok()
            }
        };

        // FP32 buffers for SGEMM warmup
        let a_f32 = alloc_f32(batch_size * input_size);
        let b1_f32 = alloc_f32(hidden_size * input_size);
        let c1_f32 = alloc_f32(batch_size * hidden_size);
        let b2_f32 = alloc_f32(output_size * hidden_size);
        let c2_f32 = alloc_f32(batch_size * output_size);

        // FP16 buffers for HGEMM warmup
        let a_u16 = alloc_u16(batch_size * input_size);
        let b1_u16 = alloc_u16(hidden_size * input_size);
        let c1_u16 = alloc_u16(batch_size * hidden_size);
        let b2_u16 = alloc_u16(output_size * hidden_size);
        let c2_u16 = alloc_u16(batch_size * output_size);

        unsafe {
            // SGEMM layer 1 shape: (batch_size x hidden_size) = (batch_size x input_size) * (input_size x hidden_size)
            if let (Some(a), Some(b), Some(c)) = (&a_f32, &b1_f32, &c1_f32) {
                let mut q = self.queue.get();
                let mut ev: cl_event = std::ptr::null_mut();
                let _ = sgemm(
                    CLBLAST_LAYOUT_ROW_MAJOR,
                    CLBLAST_TRANSPOSE_NO,
                    CLBLAST_TRANSPOSE_YES,
                    batch_size,
                    hidden_size,
                    input_size,
                    1.0f32,
                    a.get(),
                    0,
                    input_size,
                    b.get(),
                    0,
                    input_size,
                    0.0f32,
                    c.get(),
                    0,
                    hidden_size,
                    &mut q,
                    &mut ev,
                );
                let _ = self.queue.finish();
            }

            // SGEMM layer 2 shape: (batch_size x output_size) = (batch_size x hidden_size) * (hidden_size x output_size)
            if let (Some(a), Some(b), Some(c)) = (&c1_f32, &b2_f32, &c2_f32) {
                let mut q = self.queue.get();
                let mut ev: cl_event = std::ptr::null_mut();
                let _ = sgemm(
                    CLBLAST_LAYOUT_ROW_MAJOR,
                    CLBLAST_TRANSPOSE_NO,
                    CLBLAST_TRANSPOSE_YES,
                    batch_size,
                    output_size,
                    hidden_size,
                    1.0f32,
                    a.get(),
                    0,
                    hidden_size,
                    b.get(),
                    0,
                    hidden_size,
                    0.0f32,
                    c.get(),
                    0,
                    output_size,
                    &mut q,
                    &mut ev,
                );
                let _ = self.queue.finish();
            }

            let alpha_h = f32_to_f16(1.0);
            let beta_h = f32_to_f16(0.0);

            // HGEMM layer 1 shape
            if let (Some(a), Some(b), Some(c)) = (&a_u16, &b1_u16, &c1_u16) {
                let mut q = self.queue.get();
                let mut ev: cl_event = std::ptr::null_mut();
                let _ = hgemm(
                    CLBLAST_LAYOUT_ROW_MAJOR,
                    CLBLAST_TRANSPOSE_NO,
                    CLBLAST_TRANSPOSE_YES,
                    batch_size,
                    hidden_size,
                    input_size,
                    alpha_h,
                    a.get(),
                    0,
                    input_size,
                    b.get(),
                    0,
                    input_size,
                    beta_h,
                    c.get(),
                    0,
                    hidden_size,
                    &mut q,
                    &mut ev,
                );
                let _ = self.queue.finish();
            }

            // HGEMM layer 2 shape
            if let (Some(a), Some(b), Some(c)) = (&c1_u16, &b2_u16, &c2_u16) {
                let mut q = self.queue.get();
                let mut ev: cl_event = std::ptr::null_mut();
                let _ = hgemm(
                    CLBLAST_LAYOUT_ROW_MAJOR,
                    CLBLAST_TRANSPOSE_NO,
                    CLBLAST_TRANSPOSE_YES,
                    batch_size,
                    output_size,
                    hidden_size,
                    alpha_h,
                    a.get(),
                    0,
                    hidden_size,
                    b.get(),
                    0,
                    hidden_size,
                    beta_h,
                    c.get(),
                    0,
                    output_size,
                    &mut q,
                    &mut ev,
                );
                let _ = self.queue.finish();
            }
        }

        self.warmed_up_sizes.insert(matrix_size);
    }

    fn calculate_accuracy(&self, output: &[f32], reference: &[f32]) -> (f64, f64) {
        if output.len() != reference.len() {
            return (0.0, 0.0);
        }

        let mut mse: f64 = 0.0;
        let mut max_error: f64 = 0.0;

        for (out, ref_val) in output.iter().zip(reference.iter()) {
            let diff = (out - ref_val).abs();
            mse += (diff * diff) as f64;
            max_error = max_error.max(diff as f64);
        }

        mse /= output.len() as f64;
        (mse, max_error)
    }

    fn run_fp32_inference(&mut self, rd: &RoundData) -> Result<InferenceMetrics, String> {
        let input_size = rd.input_size;
        let hidden_size = rd.hidden_size;
        let output_size = rd.output_size;
        let batch_size = rd.batch_size;
        let input_data = &rd.input_data;
        let weights1 = &rd.weights1;
        let bias1 = &rd.bias1;
        let weights2 = &rd.weights2;
        let bias2 = &rd.bias2;

        // Create OpenCL buffers
        let input_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Input buffer error: {e}"))?
        };

        let weights1_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights1 buffer error: {e}"))?
        };

        let bias1_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias1 buffer error: {e}"))?
        };

        let weights2_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights2 buffer error: {e}"))?
        };

        let bias2_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias2 buffer error: {e}"))?
        };

        let output_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Output buffer error: {e}"))?
        };

        let hidden_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Hidden buffer error: {e}"))?
        };

        // Write data to buffers
        unsafe {
            let mut input_buf_mut = input_buf;
            let mut weights1_buf_mut = weights1_buf;
            let mut bias1_buf_mut = bias1_buf;
            let mut weights2_buf_mut = weights2_buf;
            let mut bias2_buf_mut = bias2_buf;
            let mut output_buf_mut = output_buf;
            let mut hidden_buf_mut = hidden_buf;

            self.queue
                .enqueue_write_buffer(&mut input_buf_mut, CL_BLOCKING, 0, input_data, &[])
                .map_err(|e| format!("Write input error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights1_buf_mut, CL_BLOCKING, 0, weights1, &[])
                .map_err(|e| format!("Write weights1 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias1_buf_mut, CL_BLOCKING, 0, bias1, &[])
                .map_err(|e| format!("Write bias1 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights2_buf_mut, CL_BLOCKING, 0, weights2, &[])
                .map_err(|e| format!("Write weights2 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias2_buf_mut, CL_BLOCKING, 0, bias2, &[])
                .map_err(|e| format!("Write bias2 error: {e}"))?;

            // Build program and create both layer kernels
            let program = Program::create_and_build_from_source(&self.context, FP32_KERNEL, "")
                .map_err(|e| format!("Program build error: {e}"))?;
            let kernel_l1 = Kernel::create(&program, "mlp_fp32_layer1")
                .map_err(|e| format!("Kernel layer1 error: {e}"))?;
            let kernel_l2 = Kernel::create(&program, "mlp_fp32_layer2")
                .map_err(|e| format!("Kernel layer2 error: {e}"))?;

            // Layer 1 kernel arguments: input, weights1, bias1, hidden, input_size, hidden_size, batch_size
            kernel_l1
                .set_arg(0, &input_buf_mut)
                .map_err(|e| format!("L1 set arg 0 error: {e}"))?;
            kernel_l1
                .set_arg(1, &weights1_buf_mut)
                .map_err(|e| format!("L1 set arg 1 error: {e}"))?;
            kernel_l1
                .set_arg(2, &bias1_buf_mut)
                .map_err(|e| format!("L1 set arg 2 error: {e}"))?;
            kernel_l1
                .set_arg(3, &hidden_buf_mut)
                .map_err(|e| format!("L1 set arg 3 error: {e}"))?;
            kernel_l1
                .set_arg(4, &(input_size as i32))
                .map_err(|e| format!("L1 set arg 4 error: {e}"))?;
            kernel_l1
                .set_arg(5, &(hidden_size as i32))
                .map_err(|e| format!("L1 set arg 5 error: {e}"))?;
            kernel_l1
                .set_arg(6, &(batch_size as i32))
                .map_err(|e| format!("L1 set arg 6 error: {e}"))?;

            // Layer 2 kernel arguments: hidden, weights2, bias2, output, hidden_size, output_size, batch_size
            kernel_l2
                .set_arg(0, &hidden_buf_mut)
                .map_err(|e| format!("L2 set arg 0 error: {e}"))?;
            kernel_l2
                .set_arg(1, &weights2_buf_mut)
                .map_err(|e| format!("L2 set arg 1 error: {e}"))?;
            kernel_l2
                .set_arg(2, &bias2_buf_mut)
                .map_err(|e| format!("L2 set arg 2 error: {e}"))?;
            kernel_l2
                .set_arg(3, &output_buf_mut)
                .map_err(|e| format!("L2 set arg 3 error: {e}"))?;
            kernel_l2
                .set_arg(4, &(hidden_size as i32))
                .map_err(|e| format!("L2 set arg 4 error: {e}"))?;
            kernel_l2
                .set_arg(5, &(output_size as i32))
                .map_err(|e| format!("L2 set arg 5 error: {e}"))?;
            kernel_l2
                .set_arg(6, &(batch_size as i32))
                .map_err(|e| format!("L2 set arg 6 error: {e}"))?;

            let kernel_start = Instant::now();

            // Execute layer 1: batch_size * hidden_size threads
            let gws_l1 = [batch_size * hidden_size];
            self.queue
                .enqueue_nd_range_kernel(
                    kernel_l1.get(),
                    1,
                    std::ptr::null(),
                    gws_l1.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("L1 kernel enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Execute layer 2: batch_size * output_size threads
            let gws_l2 = [batch_size * output_size];
            self.queue
                .enqueue_nd_range_kernel(
                    kernel_l2.get(),
                    1,
                    std::ptr::null(),
                    gws_l2.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("L2 kernel enqueue error: {e}"))?;

            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;
            let kernel_time = kernel_start.elapsed();

            // Read results
            let mut output = vec![0.0f32; batch_size * output_size];
            self.queue
                .enqueue_read_buffer(&mut output_buf_mut, CL_BLOCKING, 0, &mut output, &[])
                .map_err(|e| format!("Read output error: {e}"))?;

            // Store FP32 reference output
            self.fp32_reference = Some(output.clone());

            // Calculate metrics
            let memory_footprint_mb = ((batch_size * input_size
                + hidden_size * input_size
                + hidden_size
                + output_size * hidden_size
                + output_size
                + batch_size * output_size)
                * 4) as f64
                / (1024.0 * 1024.0);

            let total_flops = (batch_size
                * (hidden_size * (2 * input_size + 1) + output_size * (2 * hidden_size + 1)))
                as f64;

            let throughput_gflops = total_flops / (kernel_time.as_secs_f64() * 1e9);

            let memory_transferred = memory_footprint_mb * 1024.0 * 1024.0;
            let memory_bandwidth_gbps = memory_transferred / (kernel_time.as_secs_f64() * 1e9);

            Ok(InferenceMetrics {
                execution_time_ms: kernel_time.as_secs_f64() * 1000.0,
                memory_bandwidth_gbps,
                throughput_gflops,
                memory_footprint_mb,
                accuracy_mse: 0.0,
                accuracy_max_error: 0.0,
            })
        }
    }

    fn run_fp16_inference(&mut self, rd: &RoundData) -> Result<InferenceMetrics, String> {
        let input_size = rd.input_size;
        let hidden_size = rd.hidden_size;
        let output_size = rd.output_size;
        let batch_size = rd.batch_size;

        // Convert shared FP32 data to FP16
        let input_data: Vec<u16> = rd.input_data.iter().map(|&v| f32_to_f16(v)).collect();
        let weights1: Vec<u16> = rd.weights1.iter().map(|&v| f32_to_f16(v)).collect();
        let bias1: Vec<u16> = rd.bias1.iter().map(|&v| f32_to_f16(v)).collect();
        let weights2: Vec<u16> = rd.weights2.iter().map(|&v| f32_to_f16(v)).collect();
        let bias2: Vec<u16> = rd.bias2.iter().map(|&v| f32_to_f16(v)).collect();

        // Create OpenCL buffers
        let input_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Input buffer error: {e}"))?
        };

        let weights1_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights1 buffer error: {e}"))?
        };

        let bias1_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias1 buffer error: {e}"))?
        };

        let weights2_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights2 buffer error: {e}"))?
        };

        let bias2_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias2 buffer error: {e}"))?
        };

        let output_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Output buffer error: {e}"))?
        };

        let hidden_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Hidden buffer error: {e}"))?
        };

        // Write data to buffers
        unsafe {
            let mut input_buf_mut = input_buf;
            let mut weights1_buf_mut = weights1_buf;
            let mut bias1_buf_mut = bias1_buf;
            let mut weights2_buf_mut = weights2_buf;
            let mut bias2_buf_mut = bias2_buf;
            let mut output_buf_mut = output_buf;
            let mut hidden_buf_mut = hidden_buf;

            self.queue
                .enqueue_write_buffer(&mut input_buf_mut, CL_BLOCKING, 0, &input_data, &[])
                .map_err(|e| format!("Write input error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights1_buf_mut, CL_BLOCKING, 0, &weights1, &[])
                .map_err(|e| format!("Write weights1 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias1_buf_mut, CL_BLOCKING, 0, &bias1, &[])
                .map_err(|e| format!("Write bias1 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights2_buf_mut, CL_BLOCKING, 0, &weights2, &[])
                .map_err(|e| format!("Write weights2 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias2_buf_mut, CL_BLOCKING, 0, &bias2, &[])
                .map_err(|e| format!("Write bias2 error: {e}"))?;

            // Build program and create both layer kernels
            let program = Program::create_and_build_from_source(&self.context, FP16_KERNEL, "")
                .map_err(|e| format!("Program build error: {e}"))?;
            let kernel_l1 = Kernel::create(&program, "mlp_fp16_layer1")
                .map_err(|e| format!("Kernel layer1 error: {e}"))?;
            let kernel_l2 = Kernel::create(&program, "mlp_fp16_layer2")
                .map_err(|e| format!("Kernel layer2 error: {e}"))?;

            // Layer 1 kernel arguments: input, weights1, bias1, hidden, input_size, hidden_size, batch_size
            kernel_l1
                .set_arg(0, &input_buf_mut)
                .map_err(|e| format!("L1 set arg 0 error: {e}"))?;
            kernel_l1
                .set_arg(1, &weights1_buf_mut)
                .map_err(|e| format!("L1 set arg 1 error: {e}"))?;
            kernel_l1
                .set_arg(2, &bias1_buf_mut)
                .map_err(|e| format!("L1 set arg 2 error: {e}"))?;
            kernel_l1
                .set_arg(3, &hidden_buf_mut)
                .map_err(|e| format!("L1 set arg 3 error: {e}"))?;
            kernel_l1
                .set_arg(4, &(input_size as i32))
                .map_err(|e| format!("L1 set arg 4 error: {e}"))?;
            kernel_l1
                .set_arg(5, &(hidden_size as i32))
                .map_err(|e| format!("L1 set arg 5 error: {e}"))?;
            kernel_l1
                .set_arg(6, &(batch_size as i32))
                .map_err(|e| format!("L1 set arg 6 error: {e}"))?;

            // Layer 2 kernel arguments: hidden, weights2, bias2, output, hidden_size, output_size, batch_size
            kernel_l2
                .set_arg(0, &hidden_buf_mut)
                .map_err(|e| format!("L2 set arg 0 error: {e}"))?;
            kernel_l2
                .set_arg(1, &weights2_buf_mut)
                .map_err(|e| format!("L2 set arg 1 error: {e}"))?;
            kernel_l2
                .set_arg(2, &bias2_buf_mut)
                .map_err(|e| format!("L2 set arg 2 error: {e}"))?;
            kernel_l2
                .set_arg(3, &output_buf_mut)
                .map_err(|e| format!("L2 set arg 3 error: {e}"))?;
            kernel_l2
                .set_arg(4, &(hidden_size as i32))
                .map_err(|e| format!("L2 set arg 4 error: {e}"))?;
            kernel_l2
                .set_arg(5, &(output_size as i32))
                .map_err(|e| format!("L2 set arg 5 error: {e}"))?;
            kernel_l2
                .set_arg(6, &(batch_size as i32))
                .map_err(|e| format!("L2 set arg 6 error: {e}"))?;

            let kernel_start = Instant::now();

            // Execute layer 1: batch_size * hidden_size threads
            let gws_l1 = [batch_size * hidden_size];
            self.queue
                .enqueue_nd_range_kernel(
                    kernel_l1.get(),
                    1,
                    std::ptr::null(),
                    gws_l1.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("L1 kernel enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Execute layer 2: batch_size * output_size threads
            let gws_l2 = [batch_size * output_size];
            self.queue
                .enqueue_nd_range_kernel(
                    kernel_l2.get(),
                    1,
                    std::ptr::null(),
                    gws_l2.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("L2 kernel enqueue error: {e}"))?;

            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;
            let kernel_time = kernel_start.elapsed();

            // Read results
            let mut output = vec![0u16; batch_size * output_size];
            self.queue
                .enqueue_read_buffer(&mut output_buf_mut, CL_BLOCKING, 0, &mut output, &[])
                .map_err(|e| format!("Read output error: {e}"))?;

            // Convert FP16 output to FP32 for comparison
            let output_f32: Vec<f32> = output.iter().map(|&v| f16_to_f32(v)).collect();

            // Calculate accuracy against FP32 reference
            let (accuracy_mse, accuracy_max_error) =
                if let Some(ref reference) = self.fp32_reference {
                    self.calculate_accuracy(&output_f32, reference)
                } else {
                    (0.0, 0.0)
                };

            // Calculate metrics
            let memory_footprint_mb = ((batch_size * input_size
                + hidden_size * input_size
                + hidden_size
                + output_size * hidden_size
                + output_size
                + batch_size * output_size)
                * 2) as f64
                / (1024.0 * 1024.0);

            let total_flops = (batch_size
                * (hidden_size * (2 * input_size + 1) + output_size * (2 * hidden_size + 1)))
                as f64;

            let throughput_gflops = total_flops / (kernel_time.as_secs_f64() * 1e9);

            let memory_transferred = memory_footprint_mb * 1024.0 * 1024.0;
            let memory_bandwidth_gbps = memory_transferred / (kernel_time.as_secs_f64() * 1e9);

            Ok(InferenceMetrics {
                execution_time_ms: kernel_time.as_secs_f64() * 1000.0,
                memory_bandwidth_gbps,
                throughput_gflops,
                memory_footprint_mb,
                accuracy_mse,
                accuracy_max_error,
            })
        }
    }

    fn run_fp16_scaled_inference(&mut self, rd: &RoundData) -> Result<InferenceMetrics, String> {
        let input_size = rd.input_size;
        let hidden_size = rd.hidden_size;
        let output_size = rd.output_size;
        let batch_size = rd.batch_size;
        let weights1_f32 = &rd.weights1;
        let weights2_f32 = &rd.weights2;

        // Calculate row-wise scales and scaled weights
        let mut scales1 = vec![0.0f32; hidden_size];
        let mut weights1_scaled = vec![0u16; hidden_size * input_size];

        for h in 0..hidden_size {
            let row_start = h * input_size;
            let row_end = row_start + input_size;
            let row = &weights1_f32[row_start..row_end];
            let max_abs = row.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            scales1[h] = if max_abs > 0.0 { max_abs } else { 1.0 };

            for i in 0..input_size {
                let scaled_val = weights1_f32[row_start + i] / scales1[h];
                weights1_scaled[row_start + i] = f32_to_f16(scaled_val);
            }
        }

        let mut scales2 = vec![0.0f32; output_size];
        let mut weights2_scaled = vec![0u16; output_size * hidden_size];

        for o in 0..output_size {
            let row_start = o * hidden_size;
            let row_end = row_start + hidden_size;
            let row = &weights2_f32[row_start..row_end];
            let max_abs = row.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            scales2[o] = if max_abs > 0.0 { max_abs } else { 1.0 };

            for h in 0..hidden_size {
                let scaled_val = weights2_f32[row_start + h] / scales2[o];
                weights2_scaled[row_start + h] = f32_to_f16(scaled_val);
            }
        }

        // Convert other data to FP16
        let input_data: Vec<u16> = rd.input_data.iter().map(|&v| f32_to_f16(v)).collect();
        let bias1: Vec<u16> = rd.bias1.iter().map(|&v| f32_to_f16(v)).collect();
        let bias2: Vec<u16> = rd.bias2.iter().map(|&v| f32_to_f16(v)).collect();

        // Create OpenCL buffers
        let input_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Input buffer error: {e}"))?
        };

        let weights1_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights1 buffer error: {e}"))?
        };

        let scales1_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Scales1 buffer error: {e}"))?
        };

        let bias1_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias1 buffer error: {e}"))?
        };

        let weights2_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights2 buffer error: {e}"))?
        };

        let scales2_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Scales2 buffer error: {e}"))?
        };

        let bias2_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias2 buffer error: {e}"))?
        };

        let output_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Output buffer error: {e}"))?
        };

        let hidden_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Hidden buffer error: {e}"))?
        };

        // Write data to buffers
        unsafe {
            let mut input_buf_mut = input_buf;
            let mut weights1_buf_mut = weights1_buf;
            let mut scales1_buf_mut = scales1_buf;
            let mut bias1_buf_mut = bias1_buf;
            let mut weights2_buf_mut = weights2_buf;
            let mut scales2_buf_mut = scales2_buf;
            let mut bias2_buf_mut = bias2_buf;
            let mut output_buf_mut = output_buf;
            let mut hidden_buf_mut = hidden_buf;

            self.queue
                .enqueue_write_buffer(&mut input_buf_mut, CL_BLOCKING, 0, &input_data, &[])
                .map_err(|e| format!("Write input error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights1_buf_mut, CL_BLOCKING, 0, &weights1_scaled, &[])
                .map_err(|e| format!("Write weights1 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut scales1_buf_mut, CL_BLOCKING, 0, &scales1, &[])
                .map_err(|e| format!("Write scales1 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias1_buf_mut, CL_BLOCKING, 0, &bias1, &[])
                .map_err(|e| format!("Write bias1 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights2_buf_mut, CL_BLOCKING, 0, &weights2_scaled, &[])
                .map_err(|e| format!("Write weights2 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut scales2_buf_mut, CL_BLOCKING, 0, &scales2, &[])
                .map_err(|e| format!("Write scales2 error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias2_buf_mut, CL_BLOCKING, 0, &bias2, &[])
                .map_err(|e| format!("Write bias2 error: {e}"))?;

            // Build program and create both layer kernels
            let program =
                Program::create_and_build_from_source(&self.context, FP16_SCALED_KERNEL, "")
                    .map_err(|e| format!("Program build error: {e}"))?;
            let kernel_l1 = Kernel::create(&program, "mlp_fp16_scaled_layer1")
                .map_err(|e| format!("Kernel layer1 error: {e}"))?;
            let kernel_l2 = Kernel::create(&program, "mlp_fp16_scaled_layer2")
                .map_err(|e| format!("Kernel layer2 error: {e}"))?;

            // Layer 1: input, weights1, scales1, bias1, hidden, input_size, hidden_size, batch_size
            kernel_l1
                .set_arg(0, &input_buf_mut)
                .map_err(|e| format!("L1 set arg 0 error: {e}"))?;
            kernel_l1
                .set_arg(1, &weights1_buf_mut)
                .map_err(|e| format!("L1 set arg 1 error: {e}"))?;
            kernel_l1
                .set_arg(2, &scales1_buf_mut)
                .map_err(|e| format!("L1 set arg 2 error: {e}"))?;
            kernel_l1
                .set_arg(3, &bias1_buf_mut)
                .map_err(|e| format!("L1 set arg 3 error: {e}"))?;
            kernel_l1
                .set_arg(4, &hidden_buf_mut)
                .map_err(|e| format!("L1 set arg 4 error: {e}"))?;
            kernel_l1
                .set_arg(5, &(input_size as i32))
                .map_err(|e| format!("L1 set arg 5 error: {e}"))?;
            kernel_l1
                .set_arg(6, &(hidden_size as i32))
                .map_err(|e| format!("L1 set arg 6 error: {e}"))?;
            kernel_l1
                .set_arg(7, &(batch_size as i32))
                .map_err(|e| format!("L1 set arg 7 error: {e}"))?;

            // Layer 2: hidden, weights2, scales2, bias2, output, hidden_size, output_size, batch_size
            kernel_l2
                .set_arg(0, &hidden_buf_mut)
                .map_err(|e| format!("L2 set arg 0 error: {e}"))?;
            kernel_l2
                .set_arg(1, &weights2_buf_mut)
                .map_err(|e| format!("L2 set arg 1 error: {e}"))?;
            kernel_l2
                .set_arg(2, &scales2_buf_mut)
                .map_err(|e| format!("L2 set arg 2 error: {e}"))?;
            kernel_l2
                .set_arg(3, &bias2_buf_mut)
                .map_err(|e| format!("L2 set arg 3 error: {e}"))?;
            kernel_l2
                .set_arg(4, &output_buf_mut)
                .map_err(|e| format!("L2 set arg 4 error: {e}"))?;
            kernel_l2
                .set_arg(5, &(hidden_size as i32))
                .map_err(|e| format!("L2 set arg 5 error: {e}"))?;
            kernel_l2
                .set_arg(6, &(output_size as i32))
                .map_err(|e| format!("L2 set arg 6 error: {e}"))?;
            kernel_l2
                .set_arg(7, &(batch_size as i32))
                .map_err(|e| format!("L2 set arg 7 error: {e}"))?;

            let kernel_start = Instant::now();

            // Execute layer 1: batch_size * hidden_size threads
            let gws_l1 = [batch_size * hidden_size];
            self.queue
                .enqueue_nd_range_kernel(
                    kernel_l1.get(),
                    1,
                    std::ptr::null(),
                    gws_l1.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("L1 kernel enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Execute layer 2: batch_size * output_size threads
            let gws_l2 = [batch_size * output_size];
            self.queue
                .enqueue_nd_range_kernel(
                    kernel_l2.get(),
                    1,
                    std::ptr::null(),
                    gws_l2.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("L2 kernel enqueue error: {e}"))?;

            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;
            let kernel_time = kernel_start.elapsed();

            // Read results
            let mut output = vec![0u16; batch_size * output_size];
            self.queue
                .enqueue_read_buffer(&mut output_buf_mut, CL_BLOCKING, 0, &mut output, &[])
                .map_err(|e| format!("Read output error: {e}"))?;

            // Convert FP16 output to FP32 for comparison
            let output_f32: Vec<f32> = output.iter().map(|&v| f16_to_f32(v)).collect();

            // Calculate accuracy against FP32 reference
            let (accuracy_mse, accuracy_max_error) =
                if let Some(ref reference) = self.fp32_reference {
                    self.calculate_accuracy(&output_f32, reference)
                } else {
                    (0.0, 0.0)
                };

            // Calculate metrics - FP16 weights + FP32 scales
            let memory_footprint_mb = (
                (batch_size * input_size
                    + hidden_size * input_size
                    + hidden_size
                    + output_size * hidden_size
                    + output_size
                    + batch_size * output_size)
                    * 2
                    + (hidden_size + output_size) * 4
                // scales in FP32
            ) as f64
                / (1024.0 * 1024.0);

            let total_flops = (batch_size
                * (hidden_size * (2 * input_size + 1) + output_size * (2 * hidden_size + 1)))
                as f64;

            let throughput_gflops = total_flops / (kernel_time.as_secs_f64() * 1e9);

            let memory_transferred = memory_footprint_mb * 1024.0 * 1024.0;
            let memory_bandwidth_gbps = memory_transferred / (kernel_time.as_secs_f64() * 1e9);

            Ok(InferenceMetrics {
                execution_time_ms: kernel_time.as_secs_f64() * 1000.0,
                memory_bandwidth_gbps,
                throughput_gflops,
                memory_footprint_mb,
                accuracy_mse,
                accuracy_max_error,
            })
        }
    }

    // ==================== CLBlast FP32 (SGEMM) ====================
    fn run_clblast_fp32_inference(&mut self, rd: &RoundData) -> Result<InferenceMetrics, String> {
        let clblast = self.clblast.as_ref().ok_or("CLBlast not loaded")?;
        let sgemm = clblast.sgemm;

        let input_size = rd.input_size;
        let hidden_size = rd.hidden_size;
        let output_size = rd.output_size;
        let batch_size = rd.batch_size;
        let input_data = &rd.input_data;
        let weights1 = &rd.weights1;
        let bias1 = &rd.bias1;
        let weights2 = &rd.weights2;
        let bias2 = &rd.bias2;

        // Create buffers
        let input_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Input buffer error: {e}"))?
        };
        let weights1_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights1 buffer error: {e}"))?
        };
        let bias1_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias1 buffer error: {e}"))?
        };
        let weights2_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights2 buffer error: {e}"))?
        };
        let bias2_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias2 buffer error: {e}"))?
        };
        let hidden_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Hidden buffer error: {e}"))?
        };
        let output_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Output buffer error: {e}"))?
        };

        unsafe {
            let mut input_buf = input_buf;
            let mut weights1_buf = weights1_buf;
            let mut bias1_buf = bias1_buf;
            let mut weights2_buf = weights2_buf;
            let mut bias2_buf = bias2_buf;
            let mut hidden_buf = hidden_buf;
            let mut output_buf = output_buf;

            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input_data, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights1_buf, CL_BLOCKING, 0, weights1, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias1_buf, CL_BLOCKING, 0, bias1, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights2_buf, CL_BLOCKING, 0, weights2, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias2_buf, CL_BLOCKING, 0, bias2, &[])
                .map_err(|e| format!("Write error: {e}"))?;

            // Use cached bias program
            let program = self
                .bias_fp32_program
                .as_ref()
                .ok_or("FP32 bias program not compiled")?;
            let bias_relu_kernel = Kernel::create(program, "add_bias_relu_fp32")
                .map_err(|e| format!("Bias relu kernel error: {e}"))?;
            let bias_add_kernel = Kernel::create(program, "add_bias_fp32")
                .map_err(|e| format!("Bias add kernel error: {e}"))?;

            let kernel_start = Instant::now();

            // Layer 1: hidden = input * weights1^T  (SGEMM)
            // C(m,n) = alpha * A(m,k) * B^T(k,n) + beta * C
            // A=input(batch_size x input_size), B=weights1(hidden_size x input_size), C=hidden(batch_size x hidden_size)
            let mut queue_ptr = self.queue.get();
            let mut event: cl_event = std::ptr::null_mut();
            let status = sgemm(
                CLBLAST_LAYOUT_ROW_MAJOR,
                CLBLAST_TRANSPOSE_NO,
                CLBLAST_TRANSPOSE_YES,
                batch_size,
                hidden_size,
                input_size,
                1.0f32,
                input_buf.get(),
                0,
                input_size,
                weights1_buf.get(),
                0,
                input_size,
                0.0f32,
                hidden_buf.get(),
                0,
                hidden_size,
                &mut queue_ptr,
                &mut event,
            );
            if status != 0 {
                return Err(format!("CLBlast SGEMM layer1 failed with status: {status}"));
            }
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Add bias + ReLU to hidden
            let total1 = (batch_size * hidden_size) as i32;
            bias_relu_kernel
                .set_arg(0, &hidden_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(1, &bias1_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(2, &(hidden_size as i32))
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(3, &total1)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws1 = [batch_size * hidden_size];
            self.queue
                .enqueue_nd_range_kernel(
                    bias_relu_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws1.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Bias relu enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Layer 2: output = hidden * weights2^T  (SGEMM)
            let mut queue_ptr2 = self.queue.get();
            let mut event2: cl_event = std::ptr::null_mut();
            let status2 = sgemm(
                CLBLAST_LAYOUT_ROW_MAJOR,
                CLBLAST_TRANSPOSE_NO,
                CLBLAST_TRANSPOSE_YES,
                batch_size,
                output_size,
                hidden_size,
                1.0f32,
                hidden_buf.get(),
                0,
                hidden_size,
                weights2_buf.get(),
                0,
                hidden_size,
                0.0f32,
                output_buf.get(),
                0,
                output_size,
                &mut queue_ptr2,
                &mut event2,
            );
            if status2 != 0 {
                return Err(format!(
                    "CLBlast SGEMM layer2 failed with status: {status2}"
                ));
            }
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Add bias to output (no ReLU)
            let total2 = (batch_size * output_size) as i32;
            bias_add_kernel
                .set_arg(0, &output_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(1, &bias2_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(2, &(output_size as i32))
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(3, &total2)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws2 = [batch_size * output_size];
            self.queue
                .enqueue_nd_range_kernel(
                    bias_add_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws2.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Bias add enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            let kernel_time = kernel_start.elapsed();

            // Read results
            let mut output = vec![0.0f32; batch_size * output_size];
            self.queue
                .enqueue_read_buffer(&mut output_buf, CL_BLOCKING, 0, &mut output, &[])
                .map_err(|e| format!("Read output error: {e}"))?;

            // CLBlast FP32 uses the same precision as the reference — accuracy is always 0
            let accuracy_mse = 0.0_f64;
            let accuracy_max_error = 0.0_f64;

            let memory_footprint_mb = ((batch_size * input_size
                + hidden_size * input_size
                + hidden_size
                + output_size * hidden_size
                + output_size
                + batch_size * output_size)
                * 4) as f64
                / (1024.0 * 1024.0);
            let total_flops = (batch_size
                * (hidden_size * (2 * input_size + 1) + output_size * (2 * hidden_size + 1)))
                as f64;
            let throughput_gflops = total_flops / (kernel_time.as_secs_f64() * 1e9);
            let memory_transferred = memory_footprint_mb * 1024.0 * 1024.0;
            let memory_bandwidth_gbps = memory_transferred / (kernel_time.as_secs_f64() * 1e9);

            Ok(InferenceMetrics {
                execution_time_ms: kernel_time.as_secs_f64() * 1000.0,
                memory_bandwidth_gbps,
                throughput_gflops,
                memory_footprint_mb,
                accuracy_mse,
                accuracy_max_error,
            })
        }
    }

    // ==================== CLBlast FP16 (HGEMM) ====================
    fn run_clblast_fp16_inference(&mut self, rd: &RoundData) -> Result<InferenceMetrics, String> {
        let clblast = self.clblast.as_ref().ok_or("CLBlast not loaded")?;
        let hgemm = clblast.hgemm;

        let input_size = rd.input_size;
        let hidden_size = rd.hidden_size;
        let output_size = rd.output_size;
        let batch_size = rd.batch_size;

        // Convert shared FP32 data to FP16
        let input_data: Vec<u16> = rd.input_data.iter().map(|&v| f32_to_f16(v)).collect();
        let weights1: Vec<u16> = rd.weights1.iter().map(|&v| f32_to_f16(v)).collect();
        let bias1: Vec<u16> = rd.bias1.iter().map(|&v| f32_to_f16(v)).collect();
        let weights2: Vec<u16> = rd.weights2.iter().map(|&v| f32_to_f16(v)).collect();
        let bias2: Vec<u16> = rd.bias2.iter().map(|&v| f32_to_f16(v)).collect();

        let input_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Input buffer error: {e}"))?
        };
        let weights1_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights1 buffer error: {e}"))?
        };
        let bias1_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias1 buffer error: {e}"))?
        };
        let weights2_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Weights2 buffer error: {e}"))?
        };
        let bias2_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Bias2 buffer error: {e}"))?
        };
        let hidden_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Hidden buffer error: {e}"))?
        };
        let output_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Output buffer error: {e}"))?
        };

        unsafe {
            let mut input_buf = input_buf;
            let mut weights1_buf = weights1_buf;
            let mut bias1_buf = bias1_buf;
            let mut weights2_buf = weights2_buf;
            let mut bias2_buf = bias2_buf;
            let mut hidden_buf = hidden_buf;
            let mut output_buf = output_buf;

            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, &input_data, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights1_buf, CL_BLOCKING, 0, &weights1, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias1_buf, CL_BLOCKING, 0, &bias1, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights2_buf, CL_BLOCKING, 0, &weights2, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias2_buf, CL_BLOCKING, 0, &bias2, &[])
                .map_err(|e| format!("Write error: {e}"))?;

            // Use cached bias program
            let program = self
                .bias_fp16_program
                .as_ref()
                .ok_or("FP16 bias program not compiled")?;
            let bias_relu_kernel = Kernel::create(program, "add_bias_relu_fp16")
                .map_err(|e| format!("Bias relu kernel error: {e}"))?;
            let bias_add_kernel = Kernel::create(program, "add_bias_fp16")
                .map_err(|e| format!("Bias add kernel error: {e}"))?;

            let alpha_h = f32_to_f16(1.0);
            let beta_h = f32_to_f16(0.0);

            let kernel_start = Instant::now();

            // Layer 1: hidden = input * weights1^T  (HGEMM)
            let mut queue_ptr = self.queue.get();
            let mut event: cl_event = std::ptr::null_mut();
            let status = hgemm(
                CLBLAST_LAYOUT_ROW_MAJOR,
                CLBLAST_TRANSPOSE_NO,
                CLBLAST_TRANSPOSE_YES,
                batch_size,
                hidden_size,
                input_size,
                alpha_h,
                input_buf.get(),
                0,
                input_size,
                weights1_buf.get(),
                0,
                input_size,
                beta_h,
                hidden_buf.get(),
                0,
                hidden_size,
                &mut queue_ptr,
                &mut event,
            );
            if status != 0 {
                return Err(format!("CLBlast HGEMM layer1 failed with status: {status}"));
            }
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Add bias + ReLU
            let total1 = (batch_size * hidden_size) as i32;
            bias_relu_kernel
                .set_arg(0, &hidden_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(1, &bias1_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(2, &(hidden_size as i32))
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(3, &total1)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws1 = [batch_size * hidden_size];
            self.queue
                .enqueue_nd_range_kernel(
                    bias_relu_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws1.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Bias relu enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Layer 2: output = hidden * weights2^T  (HGEMM)
            let mut queue_ptr2 = self.queue.get();
            let mut event2: cl_event = std::ptr::null_mut();
            let status2 = hgemm(
                CLBLAST_LAYOUT_ROW_MAJOR,
                CLBLAST_TRANSPOSE_NO,
                CLBLAST_TRANSPOSE_YES,
                batch_size,
                output_size,
                hidden_size,
                alpha_h,
                hidden_buf.get(),
                0,
                hidden_size,
                weights2_buf.get(),
                0,
                hidden_size,
                beta_h,
                output_buf.get(),
                0,
                output_size,
                &mut queue_ptr2,
                &mut event2,
            );
            if status2 != 0 {
                return Err(format!(
                    "CLBlast HGEMM layer2 failed with status: {status2}"
                ));
            }
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Add bias (no ReLU)
            let total2 = (batch_size * output_size) as i32;
            bias_add_kernel
                .set_arg(0, &output_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(1, &bias2_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(2, &(output_size as i32))
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(3, &total2)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws2 = [batch_size * output_size];
            self.queue
                .enqueue_nd_range_kernel(
                    bias_add_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws2.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Bias add enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            let kernel_time = kernel_start.elapsed();

            // Read results and convert to FP32
            let mut output_fp16 = vec![0u16; batch_size * output_size];
            self.queue
                .enqueue_read_buffer(&mut output_buf, CL_BLOCKING, 0, &mut output_fp16, &[])
                .map_err(|e| format!("Read output error: {e}"))?;
            let output_f32: Vec<f32> = output_fp16.iter().map(|&v| f16_to_f32(v)).collect();

            let (accuracy_mse, accuracy_max_error) =
                if let Some(ref reference) = self.fp32_reference {
                    self.calculate_accuracy(&output_f32, reference)
                } else {
                    (0.0, 0.0)
                };

            let memory_footprint_mb = ((batch_size * input_size
                + hidden_size * input_size
                + hidden_size
                + output_size * hidden_size
                + output_size
                + batch_size * output_size)
                * 2) as f64
                / (1024.0 * 1024.0);
            let total_flops = (batch_size
                * (hidden_size * (2 * input_size + 1) + output_size * (2 * hidden_size + 1)))
                as f64;
            let throughput_gflops = total_flops / (kernel_time.as_secs_f64() * 1e9);
            let memory_transferred = memory_footprint_mb * 1024.0 * 1024.0;
            let memory_bandwidth_gbps = memory_transferred / (kernel_time.as_secs_f64() * 1e9);

            Ok(InferenceMetrics {
                execution_time_ms: kernel_time.as_secs_f64() * 1000.0,
                memory_bandwidth_gbps,
                throughput_gflops,
                memory_footprint_mb,
                accuracy_mse,
                accuracy_max_error,
            })
        }
    }

    // ==================== CLBlast Mixed (FP16 storage, FP32 compute via SGEMM) ====================
    fn run_clblast_mixed_inference(&mut self, rd: &RoundData) -> Result<InferenceMetrics, String> {
        let clblast = self.clblast.as_ref().ok_or("CLBlast not loaded")?;
        let sgemm = clblast.sgemm;

        let input_size = rd.input_size;
        let hidden_size = rd.hidden_size;
        let output_size = rd.output_size;
        let batch_size = rd.batch_size;
        let bias1_f32 = &rd.bias1;
        let bias2_f32 = &rd.bias2;

        // Store as FP16
        let input_data_fp16: Vec<u16> = rd.input_data.iter().map(|&v| f32_to_f16(v)).collect();
        let weights1_fp16: Vec<u16> = rd.weights1.iter().map(|&v| f32_to_f16(v)).collect();
        let weights2_fp16: Vec<u16> = rd.weights2.iter().map(|&v| f32_to_f16(v)).collect();

        // FP16 storage buffers
        let input_fp16_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };
        let weights1_fp16_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };
        let weights2_fp16_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };

        // FP32 compute buffers
        let input_f32_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };
        let weights1_f32_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size * input_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };
        let weights2_f32_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };
        let bias1_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };
        let bias2_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };
        let hidden_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };
        let output_buf = unsafe {
            Buffer::<f32>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                batch_size * output_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| format!("Buffer error: {e}"))?
        };

        unsafe {
            let mut input_fp16_buf = input_fp16_buf;
            let mut weights1_fp16_buf = weights1_fp16_buf;
            let mut weights2_fp16_buf = weights2_fp16_buf;
            let mut input_f32_buf = input_f32_buf;
            let mut weights1_f32_buf = weights1_f32_buf;
            let mut weights2_f32_buf = weights2_f32_buf;
            let mut bias1_buf = bias1_buf;
            let mut bias2_buf = bias2_buf;
            let mut hidden_buf = hidden_buf;
            let mut output_buf = output_buf;

            // Write FP16 data
            self.queue
                .enqueue_write_buffer(&mut input_fp16_buf, CL_BLOCKING, 0, &input_data_fp16, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights1_fp16_buf, CL_BLOCKING, 0, &weights1_fp16, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut weights2_fp16_buf, CL_BLOCKING, 0, &weights2_fp16, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias1_buf, CL_BLOCKING, 0, bias1_f32, &[])
                .map_err(|e| format!("Write error: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias2_buf, CL_BLOCKING, 0, bias2_f32, &[])
                .map_err(|e| format!("Write error: {e}"))?;

            // Use cached programs
            let convert_prog = self
                .convert_program
                .as_ref()
                .ok_or("Convert program not compiled")?;
            let fp16_to_fp32_kernel = Kernel::create(convert_prog, "convert_fp16_to_fp32")
                .map_err(|e| format!("Convert kernel error: {e}"))?;

            let bias_program = self
                .bias_fp32_program
                .as_ref()
                .ok_or("FP32 bias program not compiled")?;
            let bias_relu_kernel = Kernel::create(bias_program, "add_bias_relu_fp32")
                .map_err(|e| format!("Bias relu kernel error: {e}"))?;
            let bias_add_kernel = Kernel::create(bias_program, "add_bias_fp32")
                .map_err(|e| format!("Bias add kernel error: {e}"))?;

            let kernel_start = Instant::now();

            // Convert input FP16 -> FP32
            let total_input = (batch_size * input_size) as i32;
            fp16_to_fp32_kernel
                .set_arg(0, &input_fp16_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            fp16_to_fp32_kernel
                .set_arg(1, &input_f32_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            fp16_to_fp32_kernel
                .set_arg(2, &total_input)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws_inp = [batch_size * input_size];
            self.queue
                .enqueue_nd_range_kernel(
                    fp16_to_fp32_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws_inp.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Convert enqueue error: {e}"))?;

            // Convert weights1 FP16 -> FP32
            let total_w1 = (hidden_size * input_size) as i32;
            fp16_to_fp32_kernel
                .set_arg(0, &weights1_fp16_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            fp16_to_fp32_kernel
                .set_arg(1, &weights1_f32_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            fp16_to_fp32_kernel
                .set_arg(2, &total_w1)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws_w1 = [hidden_size * input_size];
            self.queue
                .enqueue_nd_range_kernel(
                    fp16_to_fp32_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws_w1.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Convert enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Layer 1: SGEMM
            let mut queue_ptr = self.queue.get();
            let mut event: cl_event = std::ptr::null_mut();
            let status = sgemm(
                CLBLAST_LAYOUT_ROW_MAJOR,
                CLBLAST_TRANSPOSE_NO,
                CLBLAST_TRANSPOSE_YES,
                batch_size,
                hidden_size,
                input_size,
                1.0f32,
                input_f32_buf.get(),
                0,
                input_size,
                weights1_f32_buf.get(),
                0,
                input_size,
                0.0f32,
                hidden_buf.get(),
                0,
                hidden_size,
                &mut queue_ptr,
                &mut event,
            );
            if status != 0 {
                return Err(format!(
                    "CLBlast SGEMM mixed layer1 failed with status: {status}"
                ));
            }
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Bias + ReLU
            let total1 = (batch_size * hidden_size) as i32;
            bias_relu_kernel
                .set_arg(0, &hidden_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(1, &bias1_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(2, &(hidden_size as i32))
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_relu_kernel
                .set_arg(3, &total1)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws1 = [batch_size * hidden_size];
            self.queue
                .enqueue_nd_range_kernel(
                    bias_relu_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws1.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Bias relu enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Convert weights2 FP16 -> FP32
            let total_w2 = (output_size * hidden_size) as i32;
            fp16_to_fp32_kernel
                .set_arg(0, &weights2_fp16_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            fp16_to_fp32_kernel
                .set_arg(1, &weights2_f32_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            fp16_to_fp32_kernel
                .set_arg(2, &total_w2)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws_w2 = [output_size * hidden_size];
            self.queue
                .enqueue_nd_range_kernel(
                    fp16_to_fp32_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws_w2.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Convert enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Layer 2: SGEMM
            let mut queue_ptr2 = self.queue.get();
            let mut event2: cl_event = std::ptr::null_mut();
            let status2 = sgemm(
                CLBLAST_LAYOUT_ROW_MAJOR,
                CLBLAST_TRANSPOSE_NO,
                CLBLAST_TRANSPOSE_YES,
                batch_size,
                output_size,
                hidden_size,
                1.0f32,
                hidden_buf.get(),
                0,
                hidden_size,
                weights2_f32_buf.get(),
                0,
                hidden_size,
                0.0f32,
                output_buf.get(),
                0,
                output_size,
                &mut queue_ptr2,
                &mut event2,
            );
            if status2 != 0 {
                return Err(format!(
                    "CLBlast SGEMM mixed layer2 failed with status: {status2}"
                ));
            }
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            // Bias (no ReLU)
            let total2 = (batch_size * output_size) as i32;
            bias_add_kernel
                .set_arg(0, &output_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(1, &bias2_buf)
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(2, &(output_size as i32))
                .map_err(|e| format!("Set arg error: {e}"))?;
            bias_add_kernel
                .set_arg(3, &total2)
                .map_err(|e| format!("Set arg error: {e}"))?;
            let gws2 = [batch_size * output_size];
            self.queue
                .enqueue_nd_range_kernel(
                    bias_add_kernel.get(),
                    1,
                    std::ptr::null(),
                    gws2.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| format!("Bias add enqueue error: {e}"))?;
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish error: {e}"))?;

            let kernel_time = kernel_start.elapsed();

            // Read results
            let mut output = vec![0.0f32; batch_size * output_size];
            self.queue
                .enqueue_read_buffer(&mut output_buf, CL_BLOCKING, 0, &mut output, &[])
                .map_err(|e| format!("Read output error: {e}"))?;

            let (accuracy_mse, accuracy_max_error) =
                if let Some(ref reference) = self.fp32_reference {
                    self.calculate_accuracy(&output, reference)
                } else {
                    (0.0, 0.0)
                };

            // Memory: FP16 storage + FP32 compute buffers
            let fp16_storage =
                (batch_size * input_size + hidden_size * input_size + output_size * hidden_size)
                    * 2;
            let fp32_compute = (batch_size * input_size
                + hidden_size * input_size
                + output_size * hidden_size
                + hidden_size
                + output_size
                + batch_size * hidden_size
                + batch_size * output_size)
                * 4;
            let memory_footprint_mb = (fp16_storage + fp32_compute) as f64 / (1024.0 * 1024.0);

            let total_flops = (batch_size
                * (hidden_size * (2 * input_size + 1) + output_size * (2 * hidden_size + 1)))
                as f64;
            let throughput_gflops = total_flops / (kernel_time.as_secs_f64() * 1e9);
            let memory_transferred = memory_footprint_mb * 1024.0 * 1024.0;
            let memory_bandwidth_gbps = memory_transferred / (kernel_time.as_secs_f64() * 1e9);

            Ok(InferenceMetrics {
                execution_time_ms: kernel_time.as_secs_f64() * 1000.0,
                memory_bandwidth_gbps,
                throughput_gflops,
                memory_footprint_mb,
                accuracy_mse,
                accuracy_max_error,
            })
        }
    }
}

#[tauri::command]
async fn run_inference(precision: String, matrix_size: usize) -> Result<InferenceMetrics, String> {
    let mut guard = get_or_init_mlp()?;
    let mlp = guard.as_mut().unwrap();

    let input_size = matrix_size;
    let hidden_size = matrix_size;
    let output_size = matrix_size / 2;
    let batch_size = 64;

    let rd = RoundData::generate(input_size, hidden_size, output_size, batch_size);

    // Always run FP32 first to establish baseline for accuracy comparison
    if precision != "Fp32" && mlp.fp32_reference.is_none() {
        let _ = mlp.run_fp32_inference(&rd)?;
    }

    match precision.as_str() {
        "Fp32" => mlp.run_fp32_inference(&rd),
        "Fp16" => mlp.run_fp16_inference(&rd),
        "FP16 + scale" => mlp.run_fp16_scaled_inference(&rd),
        "CLBlast FP32" => mlp.run_clblast_fp32_inference(&rd),
        "CLBlast FP16" => mlp.run_clblast_fp16_inference(&rd),
        "CLBlast Mixed" => mlp.run_clblast_mixed_inference(&rd),
        _ => Err("Invalid precision type".to_string()),
    }
}

fn default_metrics() -> InferenceMetrics {
    InferenceMetrics {
        execution_time_ms: 0.0,
        memory_bandwidth_gbps: 0.0,
        throughput_gflops: 0.0,
        memory_footprint_mb: 0.0,
        accuracy_mse: 0.0,
        accuracy_max_error: 0.0,
    }
}

#[tauri::command]
fn run_comparison_inference(matrix_size: usize) -> Result<ComparisonMetrics, String> {
    let mut guard = get_or_init_mlp()?;
    let mlp = guard.as_mut().unwrap();

    let input_size = matrix_size;
    let hidden_size = matrix_size;
    let output_size = matrix_size / 2;
    let batch_size = 64;

    // Generate ONE shared set of random input + deterministic weights for this round
    let rd = RoundData::generate(input_size, hidden_size, output_size, batch_size);

    // Fire CLBlast auto-tuning before we start measuring (no-op on subsequent calls)
    mlp.warmup_clblast(matrix_size);

    // Run FP32 baseline first (stores fp32_reference for accuracy comparison)
    let fp32_metrics = mlp.run_fp32_inference(&rd)?;

    // Run FP16 (same input, converted to FP16)
    let fp16_metrics = mlp.run_fp16_inference(&rd)?;

    // Run FP16 with scaling (same input)
    let fp16_scaled_metrics = mlp.run_fp16_scaled_inference(&rd)?;

    // Run CLBlast FP32 (SGEMM) — same input, graceful fallback on failure
    let clblast_fp32_metrics = mlp.run_clblast_fp32_inference(&rd).unwrap_or_else(|e| {
        eprintln!("CLBlast FP32 error: {e}");
        default_metrics()
    });

    // Run CLBlast FP16 (HGEMM) — same input, graceful fallback on failure
    let clblast_fp16_metrics = mlp.run_clblast_fp16_inference(&rd).unwrap_or_else(|e| {
        eprintln!("CLBlast FP16 error: {e}");
        default_metrics()
    });

    // Run CLBlast Mixed (FP16 storage + FP32 SGEMM compute) — same input, graceful fallback
    let clblast_mixed_metrics = mlp.run_clblast_mixed_inference(&rd).unwrap_or_else(|e| {
        eprintln!("CLBlast Mixed error: {e}");
        default_metrics()
    });

    Ok(ComparisonMetrics {
        fp32: fp32_metrics,
        fp16: fp16_metrics,
        fp16_scaled: fp16_scaled_metrics,
        clblast_fp32: clblast_fp32_metrics,
        clblast_fp16: clblast_fp16_metrics,
        clblast_mixed: clblast_mixed_metrics,
    })
}

#[tauri::command]
async fn get_len() -> Result<usize, String> {
    Ok(MY_DLL.len())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
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
