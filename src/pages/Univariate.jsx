import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ReferenceLine
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { BarChart2 } from "lucide-react";

function computeStats(values) {
  const n = values.length;
  if (!n) return {};
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
  const std = Math.sqrt(variance);
  const q1 = sorted[Math.floor(n * 0.25)];
  const q2 = sorted[Math.floor(n * 0.5)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  const cv = mean !== 0 ? (std / mean * 100) : 0;
  const skew = values.reduce((s, v) => s + ((v - mean) / std) ** 3, 0) / n;
  const kurt = values.reduce((s, v) => s + ((v - mean) / std) ** 4, 0) / n - 3;
  return { n, mean, std, variance, min: sorted[0], max: sorted[n - 1], q1, q2, q3, iqr, cv, skew, kurt };
}

function makeHistogram(values, bins = 15) {
  const min = Math.min(...values), max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const hist = Array.from({ length: bins }, (_, i) => ({
    x: parseFloat((min + i * step + step / 2).toFixed(3)),
    count: 0,
    label: `${(min + i * step).toFixed(2)}–${(min + (i + 1) * step).toFixed(2)}`
  }));
  values.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), bins - 1);
    if (idx >= 0) hist[idx].count++;
  });
  return hist;
}

function makeBoxData(stats) {
  return [
    { name: "Min", value: stats.min },
    { name: "Q1", value: stats.q1 },
    { name: "Median", value: stats.q2 },
    { name: "Q3", value: stats.q3 },
    { name: "Max", value: stats.max },
    { name: "Mean", value: stats.mean },
  ];
}

export default function Univariate() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [selectedCol, setSelectedCol] = useState("");

  const loadData = async (id, ds) => {
    setDataset(ds);
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const numCols = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    if (numCols.length) setSelectedCol(numCols[0]);
  };

  const numericCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];
  const values = selectedCol ? data.map(r => parseFloat(r[selectedCol])).filter(v => !isNaN(v)) : [];
  const stats = values.length ? computeStats(values) : null;
  const hist = values.length ? makeHistogram(values) : [];
  const boxData = stats ? makeBoxData(stats) : [];
  const profileData = values.slice(0, 200).map((v, i) => ({ sample: i + 1, value: v }));

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>📉 Univariate Analysis</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Statistical tests, distributions, and descriptive statistics</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <BarChart2 size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a dataset to start univariate analysis</p>
        </div>
      ) : (
        <>
          <div className="glass-card p-4 flex items-center gap-4">
            <label className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>Variable:</label>
            <select value={selectedCol} onChange={e => setSelectedCol(e.target.value)}
              className="px-3 py-2 rounded-lg text-sm outline-none flex-1 max-w-xs"
              style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
              {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            {stats && <span className="text-xs" style={{ color: "var(--text-muted)" }}>{stats.n} valid values</span>}
          </div>

          {stats && (
            <>
              {/* Stats table */}
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
                {[
                  { label: "N", value: stats.n },
                  { label: "Mean", value: stats.mean.toFixed(4) },
                  { label: "Std Dev", value: stats.std.toFixed(4) },
                  { label: "Min", value: stats.min.toFixed(4) },
                  { label: "Median", value: stats.q2.toFixed(4) },
                  { label: "Max", value: stats.max.toFixed(4) },
                  { label: "Q1", value: stats.q1.toFixed(4) },
                  { label: "Q3", value: stats.q3.toFixed(4) },
                  { label: "IQR", value: stats.iqr.toFixed(4) },
                  { label: "CV%", value: stats.cv.toFixed(2) },
                  { label: "Skewness", value: stats.skew.toFixed(4) },
                  { label: "Kurtosis", value: stats.kurt.toFixed(4) },
                ].map(({ label, value }) => (
                  <div key={label} className="glass-card p-3">
                    <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
                    <div className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>{value}</div>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Histogram */}
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Distribution — {selectedCol}</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={hist} margin={{ top: 5, right: 20, bottom: 30, left: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="x" tick={{ fill: "var(--text-muted)", fontSize: 10 }} angle={-30} textAnchor="end" />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip
                        content={({ payload }) => {
                          if (!payload?.length) return null;
                          return (
                            <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                              <div>{payload[0].payload.label}</div>
                              <div>Count: {payload[0].value}</div>
                            </div>
                          );
                        }}
                      />
                      <ReferenceLine x={stats.mean} stroke="#1E90FF" strokeDasharray="4 4" label={{ value: "Mean", fill: "#1E90FF", fontSize: 11 }} />
                      <Bar dataKey="count" fill="#06b6d4" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Box plot summary */}
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Box Plot Summary — {selectedCol}</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={boxData} layout="vertical" margin={{ top: 5, right: 40, bottom: 5, left: 60 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" tick={{ fill: "var(--text-muted)", fontSize: 10 }} domain={['dataMin - 1', 'dataMax + 1']} />
                      <YAxis type="category" dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                      <Bar dataKey="value" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Profile plot */}
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
                  Sample Profile — {selectedCol} {values.length > 200 ? "(first 200 samples)" : ""}
                </h3>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={profileData} margin={{ top: 5, right: 20, bottom: 20, left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="sample" tick={{ fill: "var(--text-muted)", fontSize: 10 }} label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} />
                    <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                    <ReferenceLine y={stats.mean} stroke="#1E90FF" strokeDasharray="4 4" />
                    <ReferenceLine y={stats.mean + 2 * stats.std} stroke="#f59e0b" strokeDasharray="3 3" />
                    <ReferenceLine y={stats.mean - 2 * stats.std} stroke="#f59e0b" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="value" stroke="#06b6d4" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>Blue dashed = mean · Orange dashed = ±2σ control limits</p>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}