import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import "./App.css";
import Chart from "./component/chart";
import type { DataSeries } from "./component/chart";

interface InferenceMetrics {
  execution_time_ms: number;
  memory_bandwidth_gbps: number;
  throughput_gflops: number;
  memory_footprint_mb: number;
  accuracy_mse: number;
  accuracy_max_error: number;
}

interface ComparisonMetrics {
  fp32: InferenceMetrics;
  fp16: InferenceMetrics;
  fp16_scaled: InferenceMetrics;
  clblast_fp32: InferenceMetrics;
  clblast_fp16: InferenceMetrics;
  clblast_mixed: InferenceMetrics;
}

interface ChartData {
  x: number;
  y: number;
}

const MAX_POINTS = 200;

function appendPoint(arr: ChartData[], point: ChartData): ChartData[] {
  const next = [...arr, point];
  return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
}

// ---------------------------------------------------------------------------
// All chart data lives in a single reducer so we get ONE re-render per
// iteration instead of 24 separate setState calls.
// ---------------------------------------------------------------------------
interface ChartState {
  currentMetrics: ComparisonMetrics | null;
  iteration: number;

  executionTimeDataFp32: ChartData[];
  executionTimeDataFp16: ChartData[];
  executionTimeDataFp16Scaled: ChartData[];
  executionTimeDataClblastFp32: ChartData[];
  executionTimeDataClblastFp16: ChartData[];
  executionTimeDataClblastMixed: ChartData[];

  throughputDataFp32: ChartData[];
  throughputDataFp16: ChartData[];
  throughputDataFp16Scaled: ChartData[];
  throughputDataClblastFp32: ChartData[];
  throughputDataClblastFp16: ChartData[];
  throughputDataClblastMixed: ChartData[];

  bandwidthDataFp32: ChartData[];
  bandwidthDataFp16: ChartData[];
  bandwidthDataFp16Scaled: ChartData[];
  bandwidthDataClblastFp32: ChartData[];
  bandwidthDataClblastFp16: ChartData[];
  bandwidthDataClblastMixed: ChartData[];

  accuracyDataFp16: ChartData[];
  accuracyDataFp16Scaled: ChartData[];
  accuracyDataClblastFp16: ChartData[];
  accuracyDataClblastMixed: ChartData[];
}

type ChartAction =
  | { type: "RESET" }
  | { type: "PUSH"; metrics: ComparisonMetrics };

const initialChartState: ChartState = {
  currentMetrics: null,
  iteration: 0,

  executionTimeDataFp32: [],
  executionTimeDataFp16: [],
  executionTimeDataFp16Scaled: [],
  executionTimeDataClblastFp32: [],
  executionTimeDataClblastFp16: [],
  executionTimeDataClblastMixed: [],

  throughputDataFp32: [],
  throughputDataFp16: [],
  throughputDataFp16Scaled: [],
  throughputDataClblastFp32: [],
  throughputDataClblastFp16: [],
  throughputDataClblastMixed: [],

  bandwidthDataFp32: [],
  bandwidthDataFp16: [],
  bandwidthDataFp16Scaled: [],
  bandwidthDataClblastFp32: [],
  bandwidthDataClblastFp16: [],
  bandwidthDataClblastMixed: [],

  accuracyDataFp16: [],
  accuracyDataFp16Scaled: [],
  accuracyDataClblastFp16: [],
  accuracyDataClblastMixed: [],
};

function chartReducer(state: ChartState, action: ChartAction): ChartState {
  switch (action.type) {
    case "RESET":
      return { ...initialChartState };

    case "PUSH": {
      const m = action.metrics;
      const i = state.iteration;

      return {
        currentMetrics: m,
        iteration: i + 1,

        executionTimeDataFp32: appendPoint(state.executionTimeDataFp32, {
          x: i,
          y: m.fp32.execution_time_ms,
        }),
        executionTimeDataFp16: appendPoint(state.executionTimeDataFp16, {
          x: i,
          y: m.fp16.execution_time_ms,
        }),
        executionTimeDataFp16Scaled: appendPoint(
          state.executionTimeDataFp16Scaled,
          { x: i, y: m.fp16_scaled.execution_time_ms },
        ),
        executionTimeDataClblastFp32: appendPoint(
          state.executionTimeDataClblastFp32,
          { x: i, y: m.clblast_fp32.execution_time_ms },
        ),
        executionTimeDataClblastFp16: appendPoint(
          state.executionTimeDataClblastFp16,
          { x: i, y: m.clblast_fp16.execution_time_ms },
        ),
        executionTimeDataClblastMixed: appendPoint(
          state.executionTimeDataClblastMixed,
          { x: i, y: m.clblast_mixed.execution_time_ms },
        ),

        throughputDataFp32: appendPoint(state.throughputDataFp32, {
          x: i,
          y: m.fp32.throughput_gflops,
        }),
        throughputDataFp16: appendPoint(state.throughputDataFp16, {
          x: i,
          y: m.fp16.throughput_gflops,
        }),
        throughputDataFp16Scaled: appendPoint(state.throughputDataFp16Scaled, {
          x: i,
          y: m.fp16_scaled.throughput_gflops,
        }),
        throughputDataClblastFp32: appendPoint(
          state.throughputDataClblastFp32,
          { x: i, y: m.clblast_fp32.throughput_gflops },
        ),
        throughputDataClblastFp16: appendPoint(
          state.throughputDataClblastFp16,
          { x: i, y: m.clblast_fp16.throughput_gflops },
        ),
        throughputDataClblastMixed: appendPoint(
          state.throughputDataClblastMixed,
          { x: i, y: m.clblast_mixed.throughput_gflops },
        ),

        bandwidthDataFp32: appendPoint(state.bandwidthDataFp32, {
          x: i,
          y: m.fp32.memory_bandwidth_gbps,
        }),
        bandwidthDataFp16: appendPoint(state.bandwidthDataFp16, {
          x: i,
          y: m.fp16.memory_bandwidth_gbps,
        }),
        bandwidthDataFp16Scaled: appendPoint(state.bandwidthDataFp16Scaled, {
          x: i,
          y: m.fp16_scaled.memory_bandwidth_gbps,
        }),
        bandwidthDataClblastFp32: appendPoint(state.bandwidthDataClblastFp32, {
          x: i,
          y: m.clblast_fp32.memory_bandwidth_gbps,
        }),
        bandwidthDataClblastFp16: appendPoint(state.bandwidthDataClblastFp16, {
          x: i,
          y: m.clblast_fp16.memory_bandwidth_gbps,
        }),
        bandwidthDataClblastMixed: appendPoint(
          state.bandwidthDataClblastMixed,
          { x: i, y: m.clblast_mixed.memory_bandwidth_gbps },
        ),

        accuracyDataFp16: appendPoint(state.accuracyDataFp16, {
          x: i,
          y: m.fp16.accuracy_mse,
        }),
        accuracyDataFp16Scaled: appendPoint(state.accuracyDataFp16Scaled, {
          x: i,
          y: m.fp16_scaled.accuracy_mse,
        }),
        accuracyDataClblastFp16: appendPoint(state.accuracyDataClblastFp16, {
          x: i,
          y: m.clblast_fp16.accuracy_mse,
        }),
        accuracyDataClblastMixed: appendPoint(state.accuracyDataClblastMixed, {
          x: i,
          y: m.clblast_mixed.accuracy_mse,
        }),
      };
    }

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Yield to the browser so it can paint between iterations.
// ---------------------------------------------------------------------------
function yieldToMain(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

// Row shape sent to the Rust logging commands
interface MetricsRow {
  round: number;
  fp32: InferenceMetrics;
  fp16: InferenceMetrics;
  fp16_scaled: InferenceMetrics;
  clblast_fp32: InferenceMetrics;
  clblast_fp16: InferenceMetrics;
  clblast_mixed: InferenceMetrics;
}

function App() {
  const [scale, setScale] = useState<128 | 256 | 512 | 1024>(128);
  const [isRunning, setIsRunning] = useState(false);

  const [chartState, dispatch] = useReducer(chartReducer, initialChartState);

  // A ref that the async loop checks to know when to stop.
  const runningRef = useRef(false);

  // Keep scale in a ref so the loop always reads the latest value.
  const scaleRef = useRef(scale);
  useEffect(() => {
    scaleRef.current = scale;
  }, [scale]);

  // ── Metrics accumulation for automatic logging ──────────────────
  const metricsHistoryRef = useRef<MetricsRow[]>([]);
  const roundCounterRef = useRef(0);
  const lastSavedRoundRef = useRef(0);
  const sessionDirRef = useRef<string | null>(null);

  // ------------------------------------------------------------------
  // Self-chaining async loop: run one inference, update state, yield to
  // the browser for a paint, then loop. Stops when runningRef flips.
  // ------------------------------------------------------------------
  const loopRef = useRef<Promise<void> | null>(null);

  const startLoop = useCallback(() => {
    dispatch({ type: "RESET" });
    metricsHistoryRef.current = [];
    roundCounterRef.current = 0;
    lastSavedRoundRef.current = 0;
    sessionDirRef.current = null;
    runningRef.current = true;

    const run = async () => {
      // Create one session folder for this entire run
      try {
        const dir = await invoke<string>("create_log_session", {
          matrixSize: scaleRef.current,
        });
        sessionDirRef.current = dir;
        console.log(`[Logger] Session folder: ${dir}`);
      } catch (err) {
        console.error("[Logger] Failed to create session:", err);
      }

      while (runningRef.current) {
        try {
          const metrics = await invoke<ComparisonMetrics>(
            "run_comparison_inference",
            { matrixSize: scaleRef.current },
          );

          // If we were stopped while waiting for the backend, bail out.
          if (!runningRef.current) break;

          // Single dispatch -> single React re-render for all 24+ data points.
          dispatch({ type: "PUSH", metrics });

          // ── Accumulate for logging ──────────────────────────────
          roundCounterRef.current += 1;
          const round = roundCounterRef.current;

          metricsHistoryRef.current.push({
            round,
            fp32: metrics.fp32,
            fp16: metrics.fp16,
            fp16_scaled: metrics.fp16_scaled,
            clblast_fp32: metrics.clblast_fp32,
            clblast_fp16: metrics.clblast_fp16,
            clblast_mixed: metrics.clblast_mixed,
          });

          // Every 5 rounds, append new CSV rows + redraw PNGs
          if (round % 5 === 0 && sessionDirRef.current) {
            try {
              const newRows = metricsHistoryRef.current.slice(
                lastSavedRoundRef.current,
              );
              const savedPath = await invoke<string>("append_metrics_log", {
                sessionDir: sessionDirRef.current,
                newRows,
                allRows: metricsHistoryRef.current,
              });
              lastSavedRoundRef.current = metricsHistoryRef.current.length;
              console.log(`[Logger] Saved round ${round} to: ${savedPath}`);
            } catch (logErr) {
              console.error("[Logger] Failed to save metrics:", logErr);
            }
          }
        } catch (error) {
          console.error("Inference error:", error);
        }

        // Give the browser a frame to paint before starting the next round.
        await yieldToMain();
      }
    };

    loopRef.current = run();
  }, []);

  const stopLoop = useCallback(() => {
    runningRef.current = false;
    loopRef.current = null;
  }, []);

  // Start / stop the loop when the button is toggled.
  useEffect(() => {
    if (isRunning) {
      startLoop();
    } else {
      stopLoop();
    }
    return () => stopLoop();
  }, [isRunning, startLoop, stopLoop]);

  const { currentMetrics } = chartState;

  // Prepare data series for charts
  const executionTimeSeries: DataSeries[] = [
    {
      name: "FP32 (Baseline)",
      data: chartState.executionTimeDataFp32,
      color: "#22c55e",
    },
    { name: "FP16", data: chartState.executionTimeDataFp16, color: "#3b82f6" },
    {
      name: "FP16 + Scale",
      data: chartState.executionTimeDataFp16Scaled,
      color: "#a855f7",
    },
    {
      name: "CLBlast FP32",
      data: chartState.executionTimeDataClblastFp32,
      color: "#eab308",
    },
    {
      name: "CLBlast FP16",
      data: chartState.executionTimeDataClblastFp16,
      color: "#06b6d4",
    },
    {
      name: "CLBlast Mixed",
      data: chartState.executionTimeDataClblastMixed,
      color: "#ec4899",
    },
  ];

  const throughputSeries: DataSeries[] = [
    {
      name: "FP32 (Baseline)",
      data: chartState.throughputDataFp32,
      color: "#22c55e",
    },
    { name: "FP16", data: chartState.throughputDataFp16, color: "#3b82f6" },
    {
      name: "FP16 + Scale",
      data: chartState.throughputDataFp16Scaled,
      color: "#a855f7",
    },
    {
      name: "CLBlast FP32",
      data: chartState.throughputDataClblastFp32,
      color: "#eab308",
    },
    {
      name: "CLBlast FP16",
      data: chartState.throughputDataClblastFp16,
      color: "#06b6d4",
    },
    {
      name: "CLBlast Mixed",
      data: chartState.throughputDataClblastMixed,
      color: "#ec4899",
    },
  ];

  const bandwidthSeries: DataSeries[] = [
    {
      name: "FP32 (Baseline)",
      data: chartState.bandwidthDataFp32,
      color: "#22c55e",
    },
    { name: "FP16", data: chartState.bandwidthDataFp16, color: "#3b82f6" },
    {
      name: "FP16 + Scale",
      data: chartState.bandwidthDataFp16Scaled,
      color: "#a855f7",
    },
    {
      name: "CLBlast FP32",
      data: chartState.bandwidthDataClblastFp32,
      color: "#eab308",
    },
    {
      name: "CLBlast FP16",
      data: chartState.bandwidthDataClblastFp16,
      color: "#06b6d4",
    },
    {
      name: "CLBlast Mixed",
      data: chartState.bandwidthDataClblastMixed,
      color: "#ec4899",
    },
  ];

  const accuracySeries: DataSeries[] = [
    { name: "FP16", data: chartState.accuracyDataFp16, color: "#f97316" },
    {
      name: "FP16 + Scale",
      data: chartState.accuracyDataFp16Scaled,
      color: "#ef4444",
    },
    {
      name: "CLBlast FP16",
      data: chartState.accuracyDataClblastFp16,
      color: "#06b6d4",
    },
    {
      name: "CLBlast Mixed",
      data: chartState.accuracyDataClblastMixed,
      color: "#ec4899",
    },
  ];

  return (
    <div className="flex h-full w-screen min-h-screen flex-col bg-base-100 text-text-800">
      {/* Header */}
      <header className="flex w-full h-24 items-center justify-between border-b border-gray-700 bg-base-200 px-6 py-4">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold">Precision Comparison</h1>
          <div className="flex gap-2 flex-wrap">
            <span className="rounded bg-green-500 px-2 py-1 text-xs font-semibold text-white">
              FP32
            </span>
            <span className="rounded bg-blue-500 px-2 py-1 text-xs font-semibold text-white">
              FP16
            </span>
            <span className="rounded bg-purple-500 px-2 py-1 text-xs font-semibold text-white">
              FP16+Scale
            </span>
            <span className="rounded bg-yellow-500 px-2 py-1 text-xs font-semibold text-black">
              CLBlast FP32
            </span>
            <span className="rounded bg-cyan-500 px-2 py-1 text-xs font-semibold text-white">
              CLBlast FP16
            </span>
            <span className="rounded bg-pink-500 px-2 py-1 text-xs font-semibold text-white">
              CLBlast Mixed
            </span>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold text-gray-300">
              Matrix Size: {scale}
            </label>
            <input
              type="range"
              min="1"
              max="4"
              defaultValue="1"
              onChange={(e) => {
                const n = Number(e.target.value);
                setIsRunning(false);
                setScale(n === 1 ? 128 : n === 2 ? 256 : n === 3 ? 512 : 1024);
              }}
              className="range range-accent w-48"
            />
            <div className="flex w-full justify-between text-xs text-text-800">
              <span>128</span>
              <span>256</span>
              <span>512</span>
              <span>1024</span>
            </div>
          </div>

          <button
            type="button"
            onClick={() => setIsRunning(!isRunning)}
            className={`flex h-12 w-12 btn btn-circle btn-primary items-center justify-center rounded-full  transition-colors`}
          >
            {isRunning ? (
              <img src="/stop.svg" alt="Stop" className="h-6 w-6" />
            ) : (
              <img src="/play.svg" alt="Play" className="h-6 w-6" />
            )}
          </button>
        </div>
      </header>

      {/* Metrics Panel */}
      {currentMetrics && (
        <div className="w-full border-b glassy p-6 ">
          <div className="grid grid-cols-3 gap-6">
            {/* FP32 Column */}
            <div className="space-y-3  flex flex-col items-center ">
              <h3 className="mb-3 text-center text-sm font-bold text-green-400">
                FP32 (Baseline)
              </h3>
              <div className="flex space-x-3">
                {" "}
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Execution Time</div>
                  <div className="text-lg font-bold text-green-400">
                    {currentMetrics.fp32.execution_time_ms.toFixed(2)} ms
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Throughput</div>
                  <div className="text-lg font-bold text-green-400">
                    {currentMetrics.fp32.throughput_gflops.toFixed(2)} GFLOPS
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Memory Bandwidth</div>
                  <div className="text-lg font-bold text-green-400">
                    {currentMetrics.fp32.memory_bandwidth_gbps.toFixed(2)} GB/s
                  </div>
                </div>
              </div>
            </div>

            {/* FP16 Column */}
            <div className="space-y-3 flex flex-col">
              <h3 className="mb-3 text-center text-sm font-bold text-blue-400">
                FP16
              </h3>
              <div className="flex flex-row flex-wrap space-x-2 space-y-2 justify-center">
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Execution Time</div>
                  <div className="text-lg font-bold text-blue-400">
                    {currentMetrics.fp16.execution_time_ms.toFixed(2)} ms
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Throughput</div>
                  <div className="text-lg font-bold text-blue-400">
                    {currentMetrics.fp16.throughput_gflops.toFixed(2)} GFLOPS
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Memory Bandwidth</div>
                  <div className="text-lg font-bold text-blue-400">
                    {currentMetrics.fp16.memory_bandwidth_gbps.toFixed(2)} GB/s
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3 ">
                  <div className="text-xs text-text-800">Accuracy MSE</div>
                  <div className="text-lg font-bold text-orange-400">
                    {currentMetrics.fp16.accuracy_mse.toExponential(3)}
                  </div>
                </div>
              </div>
            </div>

            {/* FP16 + Scale Column */}
            <div className="space-y-3 flex flex-col">
              <h3 className="mb-3 text-center text-sm font-bold text-purple-400">
                FP16 + Row-Wise Scale
              </h3>
              <div className="flex flex-row flex-wrap space-x-2 space-y-2 justify-center">
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Execution Time</div>
                  <div className="text-lg font-bold text-purple-400">
                    {currentMetrics.fp16_scaled.execution_time_ms.toFixed(2)} ms
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Throughput</div>
                  <div className="text-lg font-bold text-purple-400">
                    {currentMetrics.fp16_scaled.throughput_gflops.toFixed(2)}{" "}
                    GFLOPS
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Memory Bandwidth</div>
                  <div className="text-lg font-bold text-purple-400">
                    {currentMetrics.fp16_scaled.memory_bandwidth_gbps.toFixed(
                      2,
                    )}{" "}
                    GB/s
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Accuracy MSE</div>
                  <div className="text-lg font-bold text-orange-400">
                    {currentMetrics.fp16_scaled.accuracy_mse.toExponential(3)}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* CLBlast Row */}
          <div className="grid grid-cols-3 gap-6 mt-6 pt-6 border-t border-gray-700">
            {/* CLBlast FP32 Column */}
            <div className="space-y-3 flex flex-col items-center">
              <h3 className="mb-3 text-center text-sm font-bold text-yellow-400">
                CLBlast FP32 (SGEMM)
              </h3>
              <div className="flex flex-row flex-wrap space-x-2 space-y-2 justify-center">
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Execution Time</div>
                  <div className="text-lg font-bold text-yellow-400">
                    {currentMetrics.clblast_fp32.execution_time_ms.toFixed(2)}{" "}
                    ms
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Throughput</div>
                  <div className="text-lg font-bold text-yellow-400">
                    {currentMetrics.clblast_fp32.throughput_gflops.toFixed(2)}{" "}
                    GFLOPS
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Memory Bandwidth</div>
                  <div className="text-lg font-bold text-yellow-400">
                    {currentMetrics.clblast_fp32.memory_bandwidth_gbps.toFixed(
                      2,
                    )}{" "}
                    GB/s
                  </div>
                </div>
              </div>
            </div>

            {/* CLBlast FP16 Column */}
            <div className="space-y-3 flex flex-col">
              <h3 className="mb-3 text-center text-sm font-bold text-cyan-400">
                CLBlast FP16 (HGEMM)
              </h3>
              <div className="flex flex-row flex-wrap space-x-2 space-y-2 justify-center">
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Execution Time</div>
                  <div className="text-lg font-bold text-cyan-400">
                    {currentMetrics.clblast_fp16.execution_time_ms.toFixed(2)}{" "}
                    ms
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Throughput</div>
                  <div className="text-lg font-bold text-cyan-400">
                    {currentMetrics.clblast_fp16.throughput_gflops.toFixed(2)}{" "}
                    GFLOPS
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Memory Bandwidth</div>
                  <div className="text-lg font-bold text-cyan-400">
                    {currentMetrics.clblast_fp16.memory_bandwidth_gbps.toFixed(
                      2,
                    )}{" "}
                    GB/s
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Accuracy MSE</div>
                  <div className="text-lg font-bold text-orange-400">
                    {currentMetrics.clblast_fp16.accuracy_mse.toExponential(3)}
                  </div>
                </div>
              </div>
            </div>

            {/* CLBlast Mixed Column */}
            <div className="space-y-3 flex flex-col">
              <h3 className="mb-3 text-center text-sm font-bold text-pink-400">
                CLBlast Mixed (FP16→FP32)
              </h3>
              <div className="flex flex-row flex-wrap space-x-2 space-y-2 justify-center">
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Execution Time</div>
                  <div className="text-lg font-bold text-pink-400">
                    {currentMetrics.clblast_mixed.execution_time_ms.toFixed(2)}{" "}
                    ms
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Throughput</div>
                  <div className="text-lg font-bold text-pink-400">
                    {currentMetrics.clblast_mixed.throughput_gflops.toFixed(2)}{" "}
                    GFLOPS
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Memory Bandwidth</div>
                  <div className="text-lg font-bold text-pink-400">
                    {currentMetrics.clblast_mixed.memory_bandwidth_gbps.toFixed(
                      2,
                    )}{" "}
                    GB/s
                  </div>
                </div>
                <div className="rounded-lg bg-base-300 p-3">
                  <div className="text-xs text-text-800">Accuracy MSE</div>
                  <div className="text-lg font-bold text-orange-400">
                    {currentMetrics.clblast_mixed.accuracy_mse.toExponential(3)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Charts Grid */}
      <main className=" overflow-y-scroll h-full p-6">
        <div className="grid grid-cols-2 space-x-3 space-y-3">
          <div className=" rounded-lg border glassy p-4">
            <h3 className="mb-4 text-lg font-semibold">Execution Time (ms)</h3>
            <div className="flex-1">
              <Chart
                xAxisTitle="Iteration"
                yAxisTitle="Time (ms)"
                series={executionTimeSeries}
                isStreaming={false}
              />
            </div>
          </div>

          <div className=" rounded-lg border glassy p-4">
            <h3 className="mb-4 text-lg font-semibold">Throughput (GFLOPS)</h3>
            <div className="flex-1">
              <Chart
                xAxisTitle="Iteration"
                yAxisTitle="GFLOPS"
                series={throughputSeries}
                isStreaming={false}
              />
            </div>
          </div>

          <div className=" rounded-lg border glassy p-4">
            <h3 className="mb-4 text-lg font-semibold">
              Memory Bandwidth (GB/s)
            </h3>
            <div className="flex-1">
              <Chart
                xAxisTitle="Iteration"
                yAxisTitle="Bandwidth (GB/s)"
                series={bandwidthSeries}
                isStreaming={false}
              />
            </div>
          </div>

          <div className=" rounded-lg border glassy p-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-lg font-semibold">Accuracy MSE (vs FP32)</h3>
            </div>
            <div className="flex-1">
              <Chart
                xAxisTitle="Iteration"
                yAxisTitle="MSE"
                series={accuracySeries}
                isStreaming={false}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
