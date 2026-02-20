import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useRef, useState } from "react";
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

function App() {
  const [scale, setScale] = useState<128 | 256 | 512 | 1024>(128);
  const [isRunning, setIsRunning] = useState(false);
  const [currentMetrics, setCurrentMetrics] =
    useState<ComparisonMetrics | null>(null);

  // Separate data for each precision mode
  const [executionTimeDataFp32, setExecutionTimeDataFp32] = useState<
    ChartData[]
  >([]);
  const [executionTimeDataFp16, setExecutionTimeDataFp16] = useState<
    ChartData[]
  >([]);
  const [executionTimeDataFp16Scaled, setExecutionTimeDataFp16Scaled] =
    useState<ChartData[]>([]);
  const [executionTimeDataClblastFp32, setExecutionTimeDataClblastFp32] =
    useState<ChartData[]>([]);
  const [executionTimeDataClblastFp16, setExecutionTimeDataClblastFp16] =
    useState<ChartData[]>([]);
  const [executionTimeDataClblastMixed, setExecutionTimeDataClblastMixed] =
    useState<ChartData[]>([]);

  const [throughputDataFp32, setThroughputDataFp32] = useState<ChartData[]>([]);
  const [throughputDataFp16, setThroughputDataFp16] = useState<ChartData[]>([]);
  const [throughputDataFp16Scaled, setThroughputDataFp16Scaled] = useState<
    ChartData[]
  >([]);
  const [throughputDataClblastFp32, setThroughputDataClblastFp32] = useState<
    ChartData[]
  >([]);
  const [throughputDataClblastFp16, setThroughputDataClblastFp16] = useState<
    ChartData[]
  >([]);
  const [throughputDataClblastMixed, setThroughputDataClblastMixed] = useState<
    ChartData[]
  >([]);

  const [bandwidthDataFp32, setBandwidthDataFp32] = useState<ChartData[]>([]);
  const [bandwidthDataFp16, setBandwidthDataFp16] = useState<ChartData[]>([]);
  const [bandwidthDataFp16Scaled, setBandwidthDataFp16Scaled] = useState<
    ChartData[]
  >([]);
  const [bandwidthDataClblastFp32, setBandwidthDataClblastFp32] = useState<
    ChartData[]
  >([]);
  const [bandwidthDataClblastFp16, setBandwidthDataClblastFp16] = useState<
    ChartData[]
  >([]);
  const [bandwidthDataClblastMixed, setBandwidthDataClblastMixed] = useState<
    ChartData[]
  >([]);

  const [accuracyDataFp16, setAccuracyDataFp16] = useState<ChartData[]>([]);
  const [accuracyDataFp16Scaled, setAccuracyDataFp16Scaled] = useState<
    ChartData[]
  >([]);
  const [accuracyDataClblastFp16, setAccuracyDataClblastFp16] = useState<
    ChartData[]
  >([]);
  const [accuracyDataClblastMixed, setAccuracyDataClblastMixed] = useState<
    ChartData[]
  >([]);

  const iterationRef = useRef(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const runComparison = useCallback(async () => {
    try {
      const metrics = await invoke<ComparisonMetrics>(
        "run_comparison_inference",
        {
          matrixSize: scale,
        },
      );

      setCurrentMetrics(metrics);

      const iteration = iterationRef.current;

      // Update execution time data for all six modes
      setExecutionTimeDataFp32((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp32.execution_time_ms },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setExecutionTimeDataFp16((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp16.execution_time_ms },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setExecutionTimeDataFp16Scaled((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp16_scaled.execution_time_ms },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setExecutionTimeDataClblastFp32((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_fp32.execution_time_ms },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setExecutionTimeDataClblastFp16((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_fp16.execution_time_ms },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setExecutionTimeDataClblastMixed((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_mixed.execution_time_ms },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      // Update throughput data
      setThroughputDataFp32((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp32.throughput_gflops },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setThroughputDataFp16((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp16.throughput_gflops },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setThroughputDataFp16Scaled((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp16_scaled.throughput_gflops },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setThroughputDataClblastFp32((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_fp32.throughput_gflops },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setThroughputDataClblastFp16((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_fp16.throughput_gflops },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setThroughputDataClblastMixed((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_mixed.throughput_gflops },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      // Update bandwidth data
      setBandwidthDataFp32((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp32.memory_bandwidth_gbps },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setBandwidthDataFp16((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp16.memory_bandwidth_gbps },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setBandwidthDataFp16Scaled((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp16_scaled.memory_bandwidth_gbps },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setBandwidthDataClblastFp32((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_fp32.memory_bandwidth_gbps },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setBandwidthDataClblastFp16((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_fp16.memory_bandwidth_gbps },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setBandwidthDataClblastMixed((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_mixed.memory_bandwidth_gbps },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      // Update accuracy data (FP32 and CLBlast FP32 baselines are always 0 MSE, so we skip them)
      setAccuracyDataFp16((prev) => {
        const next = [...prev, { x: iteration, y: metrics.fp16.accuracy_mse }];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setAccuracyDataFp16Scaled((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.fp16_scaled.accuracy_mse },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setAccuracyDataClblastFp16((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_fp16.accuracy_mse },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      setAccuracyDataClblastMixed((prev) => {
        const next = [
          ...prev,
          { x: iteration, y: metrics.clblast_mixed.accuracy_mse },
        ];
        return next.length > 200 ? next.slice(-200) : next;
      });

      iterationRef.current += 1;
    } catch (error) {
      console.error("Inference error:", error);
    }
  }, [scale]);

  useEffect(() => {
    if (isRunning) {
      // Reset iteration counter when starting
      iterationRef.current = 0;

      // Clear existing data
      setExecutionTimeDataFp32([]);
      setExecutionTimeDataFp16([]);
      setExecutionTimeDataFp16Scaled([]);
      setExecutionTimeDataClblastFp32([]);
      setExecutionTimeDataClblastFp16([]);
      setExecutionTimeDataClblastMixed([]);
      setThroughputDataFp32([]);
      setThroughputDataFp16([]);
      setThroughputDataFp16Scaled([]);
      setThroughputDataClblastFp32([]);
      setThroughputDataClblastFp16([]);
      setThroughputDataClblastMixed([]);
      setBandwidthDataFp32([]);
      setBandwidthDataFp16([]);
      setBandwidthDataFp16Scaled([]);
      setBandwidthDataClblastFp32([]);
      setBandwidthDataClblastFp16([]);
      setBandwidthDataClblastMixed([]);
      setAccuracyDataFp16([]);
      setAccuracyDataFp16Scaled([]);
      setAccuracyDataClblastFp16([]);
      setAccuracyDataClblastMixed([]);

      // Run immediately
      runComparison();

      // Then run periodically
      intervalRef.current = setInterval(() => {
        runComparison();
      }, 1000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, runComparison]);

  // Prepare data series for charts
  const executionTimeSeries: DataSeries[] = [
    {
      name: "FP32 (Baseline)",
      data: executionTimeDataFp32,
      color: "#22c55e",
    },
    { name: "FP16", data: executionTimeDataFp16, color: "#3b82f6" },
    {
      name: "FP16 + Scale",
      data: executionTimeDataFp16Scaled,
      color: "#a855f7",
    },
    {
      name: "CLBlast FP32",
      data: executionTimeDataClblastFp32,
      color: "#eab308",
    },
    {
      name: "CLBlast FP16",
      data: executionTimeDataClblastFp16,
      color: "#06b6d4",
    },
    {
      name: "CLBlast Mixed",
      data: executionTimeDataClblastMixed,
      color: "#ec4899",
    },
  ];

  const throughputSeries: DataSeries[] = [
    { name: "FP32 (Baseline)", data: throughputDataFp32, color: "#22c55e" },
    { name: "FP16", data: throughputDataFp16, color: "#3b82f6" },
    {
      name: "FP16 + Scale",
      data: throughputDataFp16Scaled,
      color: "#a855f7",
    },
    {
      name: "CLBlast FP32",
      data: throughputDataClblastFp32,
      color: "#eab308",
    },
    {
      name: "CLBlast FP16",
      data: throughputDataClblastFp16,
      color: "#06b6d4",
    },
    {
      name: "CLBlast Mixed",
      data: throughputDataClblastMixed,
      color: "#ec4899",
    },
  ];

  const bandwidthSeries: DataSeries[] = [
    { name: "FP32 (Baseline)", data: bandwidthDataFp32, color: "#22c55e" },
    { name: "FP16", data: bandwidthDataFp16, color: "#3b82f6" },
    {
      name: "FP16 + Scale",
      data: bandwidthDataFp16Scaled,
      color: "#a855f7",
    },
    {
      name: "CLBlast FP32",
      data: bandwidthDataClblastFp32,
      color: "#eab308",
    },
    {
      name: "CLBlast FP16",
      data: bandwidthDataClblastFp16,
      color: "#06b6d4",
    },
    {
      name: "CLBlast Mixed",
      data: bandwidthDataClblastMixed,
      color: "#ec4899",
    },
  ];

  const accuracySeries: DataSeries[] = [
    { name: "FP16", data: accuracyDataFp16, color: "#f97316" },
    { name: "FP16 + Scale", data: accuracyDataFp16Scaled, color: "#ef4444" },
    {
      name: "CLBlast FP16",
      data: accuracyDataClblastFp16,
      color: "#06b6d4",
    },
    {
      name: "CLBlast Mixed",
      data: accuracyDataClblastMixed,
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
