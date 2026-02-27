mod kernel;
mod logger;
mod types;
use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};

use opencl3::types::cl_device_id;

use std::sync::OnceLock;

use crate::logger::MetricsRow;
use crate::types::{
    CLBlastHgemmFn, CLBlastSgemmFn, ClBlastLib, ComparisonMetrics, InferenceMetrics, MLPInference,
    RoundData, MLP_INSTANCE,
};

pub const MY_DLL: &[u8] = include_bytes!("../clblast.dll");

// CLBlast constants
const CLBLAST_LAYOUT_ROW_MAJOR: i32 = 101;
const CLBLAST_TRANSPOSE_NO: i32 = 111;
const CLBLAST_TRANSPOSE_YES: i32 = 112;

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

fn f32_to_f16(val: f32) -> u16 {
    half::f16::from_f32(val).to_bits()
}

fn f16_to_f32(val: u16) -> f32 {
    half::f16::from_bits(val).to_f32()
}

fn get_or_init_mlp() -> Result<std::sync::MutexGuard<'static, Option<MLPInference>>, String> {
    let mut guard = MLP_INSTANCE
        .lock()
        .map_err(|e| format!("MLP lock error: {e}"))?;
    if guard.is_none() {
        *guard = Some(MLPInference::new()?);
    }
    Ok(guard)
}

#[tauri::command]
async fn run_inference(precision: String, matrix_size: usize) -> Result<InferenceMetrics, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = get_or_init_mlp()?;
        let mlp = guard.as_mut().unwrap();

        let input_size = matrix_size;
        let hidden_size = matrix_size;
        let output_size = matrix_size / 2;
        let batch_size = 64;

        let rd = RoundData::generate(input_size, hidden_size, output_size, batch_size);

        // Always run FP32 first with the SAME data to establish a valid baseline for accuracy.
        // We regenerate the reference every call so it matches the current rd.
        if precision != "Fp32" {
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
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
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
async fn run_comparison_inference(matrix_size: usize) -> Result<ComparisonMetrics, String> {
    // Offload the entire heavy computation to a blocking thread so Tauri's
    // async runtime (and hence the UI) stays responsive.
    tauri::async_runtime::spawn_blocking(move || {
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
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[tauri::command]
async fn get_len() -> Result<usize, String> {
    Ok(MY_DLL.len())
}

/// Called once when the user starts a session (or changes matrix size).
/// Creates the timestamped folder and returns its path so the frontend can
/// reuse it for every subsequent append call.
#[tauri::command]
async fn create_log_session(matrix_size: usize) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let now = chrono::Local::now();
        let ts = now.format("%Y-%m-%d_%H-%M-%S").to_string();
        let folder_name = format!("{}_{}", ts, matrix_size);

        let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let dir = project_root.join("parallel_log").join(&folder_name);

        std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create log dir: {e}"))?;

        Ok(dir.to_string_lossy().into_owned())
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

/// Called every 5 rounds with the NEW rows only (the batch since last save).
/// Appends them to `metrics.csv` and redraws all four PNG charts from the
/// full accumulated data.
#[tauri::command]
async fn append_metrics_log(
    session_dir: String,
    new_rows: Vec<MetricsRow>,
    all_rows: Vec<MetricsRow>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dir = std::path::PathBuf::from(&session_dir);
        logger::append_session(&dir, &new_rows, &all_rows)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            run_inference,
            run_comparison_inference,
            get_len,
            create_log_session,
            append_metrics_log
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
