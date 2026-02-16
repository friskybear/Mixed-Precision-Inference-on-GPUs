import React, { useEffect, useRef } from "react";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";

export interface DataPoint {
  x: number | string;
  y: number;
}

export interface DataSeries {
  name: string;
  data: DataPoint[];
  color: string;
}

export interface ChartProps {
  xAxisTitle: string;
  yAxisTitle: string;
  data?: DataPoint[];
  series?: DataSeries[];
  isStreaming?: boolean;
  onStreamStart?: (addDataPoint: (point: DataPoint) => void) => () => void;
  maxDataPoints?: number;
  lineColor?: string;
  width?: string | number;
  height?: number;
}

const Chart: React.FC<ChartProps> = ({
  xAxisTitle,
  yAxisTitle,
  data = [],
  series = [],
  isStreaming = false,
  onStreamStart,
  maxDataPoints = 50,
  lineColor = "#22c55e",
  width = "100%",
  height = 400,
}) => {
  const chartRef = useRef<HighchartsReact.RefObject>(null);

  // Determine if we're using multiple series or single data
  const usingSeries = series.length > 0;

  const chartOptions: Highcharts.Options = {
    chart: {
      type: "area",
      animation: {
        duration: 300,
      },
      backgroundColor: "transparent",
    },
    title: {
      text: undefined,
    },
    xAxis: {
      title: {
        text: xAxisTitle,
      },
      gridLineWidth: 1,
      gridLineColor: "#374151",
    },
    yAxis: {
      title: {
        text: yAxisTitle,
      },
      gridLineColor: "#374151",
    },
    plotOptions: {
      area: {
        fillColor: usingSeries
          ? undefined
          : {
              linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
              stops: [
                [
                  0,
                  Highcharts.color(lineColor)
                    .setOpacity(0.5)
                    .get("rgba") as string,
                ],
                [
                  1,
                  Highcharts.color(lineColor)
                    .setOpacity(0.1)
                    .get("rgba") as string,
                ],
              ],
            },
        marker: {
          enabled: false,
          states: {
            hover: {
              enabled: true,
              radius: 4,
            },
          },
        },
        lineWidth: 2,
        threshold: null,
      },
    },
    series: usingSeries
      ? series.map((s) => ({
          type: "area" as const,
          name: s.name,
          color: s.color,
          fillColor: {
            linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
            stops: [
              [
                0,
                Highcharts.color(s.color).setOpacity(0.5).get("rgba") as string,
              ],
              [
                1,
                Highcharts.color(s.color).setOpacity(0.1).get("rgba") as string,
              ],
            ],
          },
          data: s.data.map((point) => [
            typeof point.x === "number" ? point.x : Number(point.x),
            point.y,
          ]),
        }))
      : [
          {
            type: "area" as const,
            name: yAxisTitle,
            color: lineColor,
            data: data.map((point) => [
              typeof point.x === "number" ? point.x : Number(point.x),
              point.y,
            ]),
          },
        ],
    credits: {
      enabled: false,
    },
    legend: {
      enabled: usingSeries,
      itemStyle: {
        color: "#9ca3af",
      },
      itemHoverStyle: {
        color: "#ffffff",
      },
    },
    tooltip: {
      shared: true,
      backgroundColor: "rgba(17, 24, 39, 0.95)",
      borderColor: "#374151",
      borderRadius: 8,
      shadow: true,
      style: {
        color: "#ffffff",
      },
    },
  };

  useEffect(() => {
    if (isStreaming && onStreamStart && chartRef.current?.chart) {
      const chart = chartRef.current.chart;
      const chartSeries = chart.series[0];

      // Clear existing data when streaming starts
      chartSeries.setData([], false);
      chart.redraw();

      const addDataPoint = (point: DataPoint) => {
        const x = typeof point.x === "number" ? point.x : Number(point.x);
        const y = point.y;

        // Add point with animation, shift if exceeds max points
        chartSeries.addPoint(
          [x, y],
          true,
          chartSeries.data.length >= maxDataPoints,
        );
      };

      const cleanup = onStreamStart(addDataPoint);

      return () => {
        if (cleanup) {
          cleanup();
        }
      };
    } else if (!isStreaming && chartRef.current?.chart) {
      // Update with static data
      const chart = chartRef.current.chart;

      if (usingSeries) {
        // Update multiple series
        series.forEach((s, index) => {
          if (chart.series[index]) {
            chart.series[index].setData(
              s.data.map((point) => [
                typeof point.x === "number" ? point.x : Number(point.x),
                point.y,
              ]),
              false,
            );
          }
        });
        chart.redraw();
      } else if (data.length > 0) {
        // Update single series
        const chartSeries = chart.series[0];
        chartSeries.setData(
          data.map((point) => [
            typeof point.x === "number" ? point.x : Number(point.x),
            point.y,
          ]),
          true,
        );
      }
    }
  }, [isStreaming, onStreamStart, maxDataPoints, data, series, usingSeries]);

  return (
    <div style={{ width, height }}>
      <HighchartsReact
        highcharts={Highcharts}
        options={chartOptions}
        ref={chartRef}
      />
    </div>
  );
};

export default Chart;
