# Mixed-Precision Neural Network Inference Benchmark

A high-performance GPU computing project comparing 6 different inference modes using OpenCL, CLBlast, and various precision strategies. Built with Rust + Tauri + React for real-time visualization and automatic data logging.

## Features

### 6 Inference Modes

1. **FP32 Baseline** - Full 32-bit precision using OpenCL kernels
2. **FP16** - Half-precision (16-bit) using OpenCL kernels  
3. **FP16 + Row-Wise Scaling** - FP16 with per-row normalization for stability
4. **CLBlast FP32** - BLAS SGEMM (optimized matrix multiplication)
5. **CLBlast FP16** - BLAS HGEMM (half-precision BLAS)
6. **CLBlast Mixed** - FP16 storage with FP32 SGEMM compute

### Real-Time Visualization

- Live charts tracking execution time, throughput, bandwidth, and accuracy
- Interactive controls for matrix size (128/256/512/1024) and start/stop
- All 6 modes run simultaneously on the same input data per round for fair comparison

### Automatic Logging & Plotting

- **Every 5 rounds**, automatically saves:
  - `metrics.csv` - All metrics from round 1 to current (37 columns)
  - `execution_time.png` - Line chart comparing all modes
  - `throughput.png` - GFLOPS comparison
  - `bandwidth.png` - Memory bandwidth comparison
  - `accuracy_mse.png` - Accuracy vs FP32 baseline

- **Logs saved to**: `parallel_log/{YYYY-MM-DD_HH-MM-SS}_{matrix_size}/`
- **New session** created automatically when matrix size changes

## Quick Start

### Prerequisites

- OpenCL-compatible GPU (NVIDIA, AMD, Intel)
- OpenCL drivers installed
- Rust 1.70+ and Node.js 18+

### Run

```bash
npm install
npm run tauri dev
```

## Architecture

### Backend (Rust)

- **OpenCL**: Custom kernels for FP32, FP16, and FP16+Scaling modes
- **CLBlast**: Embedded DLL for optimized BLAS operations (SGEMM/HGEMM)
- **Logging**: `plotters` crate generates charts, `chrono` for timestamps
- **Singleton pattern**: Reuses OpenCL context/queue across rounds to avoid CLBlast re-tuning

### Frontend (React + TypeScript)

- Real-time data streaming via Tauri commands
- Highcharts for live visualization
- 200-point rolling window for all metrics

### Workload

- 2-layer MLP: Input → Hidden (ReLU) → Output
- Batch size: 64
- Matrix sizes: 128, 256, 512, 1024 neurons
- Same random input per round across all 6 modes

## Implementation Details

### Row-Wise Scaling (Mode 3)

Normalizes each weight matrix row to improve FP16 numerical range:

```
scale[row] = max(|weights[row]|)
scaled_weights[row] = weights[row] / scale[row]  // stored in FP16
```

During inference: `output = (scaled_weights * scale) * input`

### CLBlast Integration (Modes 4-6)

- **CLBlast FP32**: Uses SGEMM for matrix multiplication, OpenCL kernels for bias/ReLU
- **CLBlast FP16**: Uses HGEMM (if supported), FP16 kernels for bias/ReLU
- **CLBlast Mixed**: FP16 storage + conversion, FP32 SGEMM compute (best of both worlds)

CLBlast library is embedded in the binary and extracted to temp folder at runtime.

## Performance Metrics

**Throughput (GFLOPS)**:
```
flops = batch_size × [hidden × (2×input+1) + output × (2×hidden+1)]
throughput = flops / time / 10^9
```

**Memory Bandwidth (GB/s)**:
```
bandwidth = memory_footprint / time / 10^9
```

**Accuracy (MSE)**:
```
mse = Σ(output - fp32_baseline)² / N
```

## Expected Results

| Mode | Memory | Speed | Accuracy |
|------|--------|-------|----------|
| FP32 | 100% | Baseline | Reference |
| FP16 | ~50% | 1.2-1.8x | Good |
| FP16+Scale | ~52% | 1.2-1.6x | Better |
| CLBlast FP32 | 100% | 1.5-3x* | Reference |
| CLBlast FP16 | ~50% | 2-4x* | Good |
| CLBlast Mixed | ~50% | 1.8-3.5x* | Excellent |

*Performance depends on GPU, matrix size, and CLBlast tuning. First run is slower (compilation).

## Project Structure

```
parallel_project/
├── src-tauri/
│   ├── src/lib.rs           # Rust backend: OpenCL, CLBlast, logging
│   ├── clblast.dll          # Embedded CLBlast library
│   └── Cargo.toml
├── src/
│   ├── App.tsx              # Main UI component
│   └── component/chart.tsx  # Highcharts wrapper
└── parallel_log/            # Auto-generated logs (gitignored)
    └── 2025-01-15_14-30-22_512/
        ├── metrics.csv
        ├── execution_time.png
        ├── throughput.png
        ├── bandwidth.png
        └── accuracy_mse.png
```

## Key Dependencies

**Rust**:
- `opencl3` - OpenCL bindings
- `libloading` - Dynamic library loading for CLBlast
- `plotters` - Chart generation
- `chrono` - Timestamps
- `tauri` - Desktop framework

**Node**:
- `react` - UI framework
- `highcharts` - Charting library
- `@tauri-apps/api` - Rust ↔ JS bridge

## Usage

1. **Start the app**: `npm run tauri dev`
2. **Select matrix size** with the slider
3. **Click play** to start continuous inference (1 round/second)
4. **Watch live charts** update in real-time
5. **Check logs** in `parallel_log/` folder (created next to executable)
6. **CSV and plots** regenerate every 5 rounds with all data from round 1

**Tips**:
- Larger matrix sizes show bigger performance differences
- First run is slower (OpenCL compilation + CLBlast tuning)
- CLBlast may timeout/fail on some GPUs → falls back to default metrics
- Changing matrix size creates a new log session

## Troubleshooting

**No OpenCL devices found**: Install GPU drivers with OpenCL support

**CLBlast errors**: Some GPUs don't support FP16 BLAS → mode falls back gracefully

**Slow performance**: 
- Ensure GPU is being used (not CPU fallback)
- Check GPU isn't thermal throttling
- Increase matrix size (better GPU utilization)

**Build errors**: 
```bash
cargo clean && npm run tauri build
```

## Technical Highlights

- **Fair comparison**: All modes use identical input per round
- **Efficient execution**: Singleton pattern reuses OpenCL context (avoids CLBlast re-tuning)
- **Graceful degradation**: CLBlast failures don't crash the app
- **Comprehensive logging**: CSV + 4 charts auto-generated every 5 rounds
- **Session management**: New log folder when matrix size changes

## Future Work

- [ ] INT8 quantization mode
- [ ] Vulkan compute backend
- [ ] Multi-GPU comparison
- [ ] Export to PDF report
- [ ] Configurable batch size
- [ ] More complex networks (CNNs, Transformers)

## License

Academic project for GPU computing coursework.

---

**Built with**: Rust 🦀 | Tauri 🚀 | React ⚛️ | OpenCL 💻 | CLBlast 📊