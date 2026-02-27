// OpenCL kernel source for FP32 inference — split into two passes so each hidden
// neuron is computed exactly once per batch item instead of output_size times.
pub const FP32_KERNEL: &str = r#"
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
pub const FP16_KERNEL: &str = r#"
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
pub const BIAS_RELU_FP32_KERNEL: &str = r#"
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

pub const BIAS_RELU_FP16_KERNEL: &str = r#"
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

pub const FP16_TO_FP32_KERNEL: &str = r#"
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
pub const FP16_SCALED_KERNEL: &str = r#"
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