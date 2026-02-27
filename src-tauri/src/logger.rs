use crate::types::InferenceMetrics;
use plotters::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;

/// One row of accumulated metrics (one per round).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MetricsRow {
    pub round: usize,
    pub fp32: InferenceMetrics,
    pub fp16: InferenceMetrics,
    pub fp16_scaled: InferenceMetrics,
    pub clblast_fp32: InferenceMetrics,
    pub clblast_fp16: InferenceMetrics,
    pub clblast_mixed: InferenceMetrics,
}

// ── Colours matching the frontend ──────────────────────────────────────────
const COL_FP32: RGBColor = RGBColor(34, 197, 94); // green-500
const COL_FP16: RGBColor = RGBColor(59, 130, 246); // blue-500
const COL_FP16S: RGBColor = RGBColor(168, 85, 247); // purple-500
const COL_CB32: RGBColor = RGBColor(234, 179, 8); // yellow-500
const COL_CB16: RGBColor = RGBColor(6, 182, 212); // cyan-500
const COL_CBMX: RGBColor = RGBColor(236, 72, 153); // pink-500

const CSV_HEADER: &str = "round,\
    fp32_exec_ms,fp32_gflops,fp32_bw_gbps,fp32_mem_mb,fp32_mse,fp32_maxerr,\
    fp16_exec_ms,fp16_gflops,fp16_bw_gbps,fp16_mem_mb,fp16_mse,fp16_maxerr,\
    fp16s_exec_ms,fp16s_gflops,fp16s_bw_gbps,fp16s_mem_mb,fp16s_mse,fp16s_maxerr,\
    cb32_exec_ms,cb32_gflops,cb32_bw_gbps,cb32_mem_mb,cb32_mse,cb32_maxerr,\
    cb16_exec_ms,cb16_gflops,cb16_bw_gbps,cb16_mem_mb,cb16_mse,cb16_maxerr,\
    cbmx_exec_ms,cbmx_gflops,cbmx_bw_gbps,cbmx_mem_mb,cbmx_mse,cbmx_maxerr\n";

const IMG_W: u32 = 1200;
const IMG_H: u32 = 600;

// ── Public entry-point ─────────────────────────────────────────────────────

/// Appends `new_rows` to the existing `metrics.csv` inside `dir`,
/// then redraws all four PNG charts from the full `all_rows` history.
///
/// Each step is independent — if one plot fails the others still run,
/// and the CSV is always written first so data is never lost.
pub fn append_session(
    dir: &Path,
    new_rows: &[MetricsRow],
    all_rows: &[MetricsRow],
) -> Result<String, String> {
    fs::create_dir_all(dir).map_err(|e| format!("mkdir failed: {e}"))?;

    // 1. Always append CSV first — this must succeed.
    append_csv(dir, new_rows)?;

    // 2. Draw each plot independently. Collect errors but don't bail early
    //    so that one broken chart doesn't block the others.
    let mut errors: Vec<String> = Vec::new();

    if let Err(e) = render_chart(
        dir,
        "execution_time.png",
        "Execution Time (ms)",
        "Time (ms)",
        &all_six_series(all_rows, |m| m.execution_time_ms),
    ) {
        errors.push(format!("execution_time: {e}"));
    }

    if let Err(e) = render_chart(
        dir,
        "throughput.png",
        "Throughput (GFLOPS)",
        "GFLOPS",
        &all_six_series(all_rows, |m| m.throughput_gflops),
    ) {
        errors.push(format!("throughput: {e}"));
    }

    if let Err(e) = render_chart(
        dir,
        "bandwidth.png",
        "Memory Bandwidth (GB/s)",
        "GB/s",
        &all_six_series(all_rows, |m| m.memory_bandwidth_gbps),
    ) {
        errors.push(format!("bandwidth: {e}"));
    }

    if let Err(e) = render_chart(
        dir,
        "accuracy_mse.png",
        "Accuracy MSE (vs FP32 Baseline)",
        "MSE",
        &accuracy_series(all_rows),
    ) {
        errors.push(format!("accuracy_mse: {e}"));
    }

    if !errors.is_empty() {
        eprintln!("[Logger] Chart warnings: {}", errors.join("; "));
    }

    Ok(dir.to_string_lossy().into_owned())
}

// ── CSV (append-only) ──────────────────────────────────────────────────────

fn format_row(r: &MetricsRow) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
        r.round,
        r.fp32.execution_time_ms, r.fp32.throughput_gflops, r.fp32.memory_bandwidth_gbps,
        r.fp32.memory_footprint_mb, r.fp32.accuracy_mse, r.fp32.accuracy_max_error,
        r.fp16.execution_time_ms, r.fp16.throughput_gflops, r.fp16.memory_bandwidth_gbps,
        r.fp16.memory_footprint_mb, r.fp16.accuracy_mse, r.fp16.accuracy_max_error,
        r.fp16_scaled.execution_time_ms, r.fp16_scaled.throughput_gflops, r.fp16_scaled.memory_bandwidth_gbps,
        r.fp16_scaled.memory_footprint_mb, r.fp16_scaled.accuracy_mse, r.fp16_scaled.accuracy_max_error,
        r.clblast_fp32.execution_time_ms, r.clblast_fp32.throughput_gflops, r.clblast_fp32.memory_bandwidth_gbps,
        r.clblast_fp32.memory_footprint_mb, r.clblast_fp32.accuracy_mse, r.clblast_fp32.accuracy_max_error,
        r.clblast_fp16.execution_time_ms, r.clblast_fp16.throughput_gflops, r.clblast_fp16.memory_bandwidth_gbps,
        r.clblast_fp16.memory_footprint_mb, r.clblast_fp16.accuracy_mse, r.clblast_fp16.accuracy_max_error,
        r.clblast_mixed.execution_time_ms, r.clblast_mixed.throughput_gflops, r.clblast_mixed.memory_bandwidth_gbps,
        r.clblast_mixed.memory_footprint_mb, r.clblast_mixed.accuracy_mse, r.clblast_mixed.accuracy_max_error,
    )
}

/// Appends only the *new* rows to `metrics.csv`.
/// If the file doesn't exist yet, writes the header first.
fn append_csv(dir: &Path, new_rows: &[MetricsRow]) -> Result<(), String> {
    let path = dir.join("metrics.csv");
    let needs_header = !path.exists();

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("csv open failed: {e}"))?;

    if needs_header {
        file.write_all(CSV_HEADER.as_bytes())
            .map_err(|e| format!("csv header write failed: {e}"))?;
    }

    for r in new_rows {
        file.write_all(format_row(r).as_bytes())
            .map_err(|e| format!("csv row write failed: {e}"))?;
    }

    // Explicitly flush and drop so the file handle is released immediately.
    file.flush().map_err(|e| format!("csv flush failed: {e}"))?;
    drop(file);

    Ok(())
}

// ── Chart rendering ────────────────────────────────────────────────────────

struct SeriesInfo<'a> {
    label: &'a str,
    color: RGBColor,
    data: Vec<(f64, f64)>,
}

/// Renders a single chart into `dir/filename`.
///
/// The entire BitMapBackend lifetime is scoped inside this function so the
/// file handle is guaranteed to be closed before we return. On Windows this
/// is critical — an open handle prevents the next call from overwriting the
/// same file.
fn render_chart(
    dir: &Path,
    filename: &str,
    title: &str,
    y_label: &str,
    series: &[SeriesInfo],
) -> Result<(), String> {
    let path = dir.join(filename);

    // Delete stale file first so we never fight a leftover handle on Windows.
    if path.exists() {
        let _ = fs::remove_file(&path);
    }

    // --- Begin scoped block: BitMapBackend is created AND dropped here ---
    {
        let root = BitMapBackend::new(&path, (IMG_W, IMG_H)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| format!("fill: {e}"))?;

        // Compute axis bounds
        let x_max = series
            .iter()
            .flat_map(|s| s.data.iter().map(|(x, _)| *x))
            .fold(1.0_f64, f64::max);

        let y_max = series
            .iter()
            .flat_map(|s| s.data.iter().map(|(_, y)| *y))
            .fold(f64::MIN, f64::max);

        let y_min = series
            .iter()
            .flat_map(|s| s.data.iter().map(|(_, y)| *y))
            .fold(f64::MAX, f64::min);

        // Guard against degenerate ranges (all zeros, single point, etc.)
        let y_lo = if y_min >= 0.0 { 0.0 } else { y_min * 1.15 };
        let y_hi = if y_max <= y_lo + 1e-12 {
            y_lo + 1.0 // avoid zero-height range
        } else {
            y_max * 1.15
        };
        let x_hi = if x_max < 1.0 { 1.0 } else { x_max };

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 24).into_font())
            .margin(16)
            .x_label_area_size(40)
            .y_label_area_size(70)
            .build_cartesian_2d(0.0..x_hi, y_lo..y_hi)
            .map_err(|e| format!("chart build: {e}"))?;

        chart
            .configure_mesh()
            .x_desc("Round")
            .y_desc(y_label)
            .label_style(("sans-serif", 14))
            .light_line_style(WHITE.mix(0.7))
            .draw()
            .map_err(|e| format!("mesh: {e}"))?;

        for s in series {
            let color = s.color;
            chart
                .draw_series(LineSeries::new(
                    s.data.iter().copied(),
                    ShapeStyle {
                        color: color.to_rgba(),
                        filled: true,
                        stroke_width: 2,
                    },
                ))
                .map_err(|e| format!("series: {e}"))?
                .label(s.label)
                .legend(move |(x, y)| {
                    Rectangle::new(
                        [(x, y - 5), (x + 18, y + 5)],
                        ShapeStyle {
                            color: color.to_rgba(),
                            filled: true,
                            stroke_width: 0,
                        },
                    )
                });
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.85))
            .border_style(BLACK.mix(0.3))
            .label_font(("sans-serif", 13))
            .position(SeriesLabelPosition::UpperRight)
            .draw()
            .map_err(|e| format!("legend: {e}"))?;

        // present() encodes the bitmap to PNG and writes to disk.
        root.present().map_err(|e| format!("present: {e}"))?;

        // root (and the BitMapBackend inside it) are dropped here at end of block.
    }

    Ok(())
}

// ── Series builders ────────────────────────────────────────────────────────

fn extract_mode<'a>(
    rows: &'a [MetricsRow],
    mode_fn: impl Fn(&'a MetricsRow) -> &'a InferenceMetrics,
    metric_fn: impl Fn(&InferenceMetrics) -> f64,
) -> Vec<(f64, f64)> {
    rows.iter()
        .map(|r| (r.round as f64, metric_fn(mode_fn(r))))
        .collect()
}

fn all_six_series<'a>(
    rows: &[MetricsRow],
    metric_fn: impl Fn(&InferenceMetrics) -> f64 + Copy,
) -> Vec<SeriesInfo<'a>> {
    vec![
        SeriesInfo {
            label: "FP32",
            color: COL_FP32,
            data: extract_mode(rows, |r| &r.fp32, metric_fn),
        },
        SeriesInfo {
            label: "FP16",
            color: COL_FP16,
            data: extract_mode(rows, |r| &r.fp16, metric_fn),
        },
        SeriesInfo {
            label: "FP16 + Scale",
            color: COL_FP16S,
            data: extract_mode(rows, |r| &r.fp16_scaled, metric_fn),
        },
        SeriesInfo {
            label: "CLBlast FP32",
            color: COL_CB32,
            data: extract_mode(rows, |r| &r.clblast_fp32, metric_fn),
        },
        SeriesInfo {
            label: "CLBlast FP16",
            color: COL_CB16,
            data: extract_mode(rows, |r| &r.clblast_fp16, metric_fn),
        },
        SeriesInfo {
            label: "CLBlast Mixed",
            color: COL_CBMX,
            data: extract_mode(rows, |r| &r.clblast_mixed, metric_fn),
        },
    ]
}

fn accuracy_series<'a>(rows: &[MetricsRow]) -> Vec<SeriesInfo<'a>> {
    vec![
        SeriesInfo {
            label: "FP16",
            color: COL_FP16,
            data: extract_mode(rows, |r| &r.fp16, |m| m.accuracy_mse),
        },
        SeriesInfo {
            label: "FP16 + Scale",
            color: COL_FP16S,
            data: extract_mode(rows, |r| &r.fp16_scaled, |m| m.accuracy_mse),
        },
        SeriesInfo {
            label: "CLBlast FP16",
            color: COL_CB16,
            data: extract_mode(rows, |r| &r.clblast_fp16, |m| m.accuracy_mse),
        },
        SeriesInfo {
            label: "CLBlast Mixed",
            color: COL_CBMX,
            data: extract_mode(rows, |r| &r.clblast_mixed, |m| m.accuracy_mse),
        },
    ]
}
