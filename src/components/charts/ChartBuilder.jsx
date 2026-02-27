import { useState, useMemo } from "react";
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  PieChart, Pie, Cell, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

const COLORS = ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#ec4899", "#14b8a6"];

const CHART_TYPES = [
  { id: "bar", label: "Bar" },
  { id: "line", label: "Line" },
  { id: "area", label: "Area" },
  { id: "scatter", label: "Scatter" },
  { id: "pie", label: "Pie" },
];

const tooltipStyle = {
  background: "#131929",
  border: "1px solid #1e2d4a",
  borderRadius: 10,
  color: "#f0f4ff",
  fontSize: 12
};

export default function ChartBuilder({ data, columns }) {
  const [chartType, setChartType] = useState("bar");
  const [xCol, setXCol] = useState(columns[0] || "");
  const [yCol, setYCol] = useState(columns[1] || columns[0] || "");

  const chartData = useMemo(() => {
    if (!data || !xCol) return [];
    if (chartType === "pie") {
      const counts = {};
      data.forEach(row => {
        const key = String(row[xCol] ?? "null");
        counts[key] = (counts[key] || 0) + 1;
      });
      return Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 12)
        .map(([name, value]) => ({ name, value }));
    }
    return data.slice(0, 500).map(row => ({
      x: row[xCol],
      y: isNaN(Number(row[yCol])) ? 0 : Number(row[yCol]),
      [xCol]: row[xCol],
      [yCol]: Number(row[yCol]) || 0,
    }));
  }, [data, xCol, yCol, chartType]);

  const select = (val, setter) => (
    <select
      value={val}
      onChange={e => setter(e.target.value)}
      className="text-sm px-3 py-2 rounded-lg outline-none focus:ring-1"
      style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border)",
        color: "var(--text-primary)",
        minWidth: 140
      }}
    >
      {columns.map(c => <option key={c} value={c}>{c}</option>)}
    </select>
  );

  const renderChart = () => {
    if (chartType === "pie") {
      return (
        <ResponsiveContainer width="100%" height={360}>
          <PieChart>
            <Pie data={chartData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={140} label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}>
              {chartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip contentStyle={tooltipStyle} />
          </PieChart>
        </ResponsiveContainer>
      );
    }

    if (chartType === "scatter") {
      return (
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart>
            <CartesianGrid stroke="#1e2d4a" strokeDasharray="3 3" />
            <XAxis dataKey={xCol} tick={{ fill: "#8b9cbf", fontSize: 11 }} />
            <YAxis dataKey={yCol} tick={{ fill: "#8b9cbf", fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: "#6366f1", strokeWidth: 1 }} />
            <Scatter data={chartData.map(r => ({ [xCol]: r.x, [yCol]: r.y }))} fill={COLORS[0]} />
          </ScatterChart>
        </ResponsiveContainer>
      );
    }

    if (chartType === "area") {
      return (
        <ResponsiveContainer width="100%" height={360}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#1e2d4a" strokeDasharray="3 3" />
            <XAxis dataKey={xCol} tick={{ fill: "#8b9cbf", fontSize: 11 }} />
            <YAxis tick={{ fill: "#8b9cbf", fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Area type="monotone" dataKey={yCol} stroke="#6366f1" fill="url(#areaGrad)" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      );
    }

    if (chartType === "line") {
      return (
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#1e2d4a" strokeDasharray="3 3" />
            <XAxis dataKey={xCol} tick={{ fill: "#8b9cbf", fontSize: 11 }} />
            <YAxis tick={{ fill: "#8b9cbf", fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey={yCol} stroke="#6366f1" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      );
    }

    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={chartData}>
          <CartesianGrid stroke="#1e2d4a" strokeDasharray="3 3" />
          <XAxis dataKey={xCol} tick={{ fill: "#8b9cbf", fontSize: 11 }} />
          <YAxis tick={{ fill: "#8b9cbf", fontSize: 11 }} />
          <Tooltip contentStyle={tooltipStyle} />
          <Bar dataKey={yCol} fill="#6366f1" radius={[4, 4, 0, 0]}>
            {chartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex rounded-xl overflow-hidden" style={{ border: "1px solid var(--border)" }}>
          {CHART_TYPES.map(ct => (
            <button
              key={ct.id}
              onClick={() => setChartType(ct.id)}
              className="px-4 py-2 text-sm font-medium transition-colors"
              style={{
                background: chartType === ct.id ? "var(--accent)" : "var(--bg-card)",
                color: chartType === ct.id ? "white" : "var(--text-secondary)",
                borderRight: "1px solid var(--border)"
              }}
            >
              {ct.label}
            </button>
          ))}
        </div>
        {chartType !== "pie" && (
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>X</span>
            {select(xCol, setXCol)}
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>Y</span>
            {select(yCol, setYCol)}
          </div>
        )}
        {chartType === "pie" && (
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>Group by</span>
            {select(xCol, setXCol)}
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="glass-card p-4">
        {renderChart()}
      </div>
    </div>
  );
}