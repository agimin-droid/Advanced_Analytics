import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine, ScatterChart, Scatter, Legend
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { BarChart2, Download } from "lucide-react";

// ── STATISTICS ───────────────────────────────────────────────────────────────
function computeStats(values) {
  const n = values.length;
  if (!n) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
  const std = Math.sqrt(variance);
  const q1 = sorted[Math.floor(n * 0.25)];
  const q2 = sorted[Math.floor(n * 0.5)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  const cv = mean !== 0 ? (std / mean * 100) : 0;
  const skew = std > 0 ? values.reduce((s, v) => s + ((v - mean) / std) ** 3, 0) / n : 0;
  const kurt = std > 0 ? values.reduce((s, v) => s + ((v - mean) / std) ** 4, 0) / n - 3 : 0;
  return { n, mean, std, variance, min: sorted[0], max: sorted[n - 1], q1, q2, q3, iqr, cv, skew, kurt };
}

function makeHistogram(values, bins = 20) {
  if (!values.length) return [];
  const min = Math.min(...values), max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const hist = Array.from({ length: bins }, (_, i) => ({
    x: parseFloat((min + i * step + step / 2).toFixed(4)),
    count: 0,
    label: `${(min + i * step).toFixed(3)}–${(min + (i + 1) * step).toFixed(3)}`
  }));
  values.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), bins - 1);
    if (idx >= 0) hist[idx].count++;
  });
  return hist;
}

function makeQQData(values) {
  const n = values.length;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.map((v, i) => {
    const p = (i + 0.5) / n;
    // Approximation of inverse normal CDF (Beasley-Springer-Moro)
    const a0 = 2.515517, a1 = 0.802853, a2 = 0.010328;
    const b1 = 1.432788, b2 = 0.189269, b3 = 0.001308;
    const u = p < 0.5 ? p : 1 - p;
    const t = Math.sqrt(-2 * Math.log(u));
    const num = a0 + a1 * t + a2 * t * t;
    const den = 1 + b1 * t + b2 * t * t + b3 * t * t * t;
    const z = p < 0.5 ? -(t - num / den) : (t - num / den);
    return { theoretical: parseFloat(z.toFixed(4)), observed: parseFloat(v.toFixed(4)) };
  });
}

const COLORS = ["#1E90FF", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#f97316", "#84cc16"];

export default function Univariate() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [selectedCol, setSelectedCol] = useState("");
  const [profileCols, setProfileCols] = useState([]);
  const [tab, setTab] = useState("eda");
  const [groupCol, setGroupCol] = useState("");

  const loadData = async (id, ds) => {
    setDataset(ds);
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const numCols = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    if (numCols.length) { setSelectedCol(numCols[0]); setProfileCols(numCols.slice(0, 6)); }
  };

  const numericCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];
  const catCols = dataset?.columns?.filter(c => isNaN(parseFloat(data[0]?.[c]))) || [];
  const values = selectedCol ? data.map(r => parseFloat(r[selectedCol])).filter(v => !isNaN(v)) : [];
  const stats = values.length ? computeStats(values) : null;
  const hist = values.length ? makeHistogram(values) : [];
  const qqData = values.length >= 4 ? makeQQData(values.slice(0, 500)) : [];
  const profileData = values.slice(0, 300).map((v, i) => ({ sample: i + 1, value: v }));

  // Row profiles: all selected columns across samples
  const rowProfileData = data.slice(0, 200).map((row, i) => {
    const pt = { sample: i + 1 };
    profileCols.forEach(c => { pt[c] = parseFloat(row[c]) || 0; });
    return pt;
  });

  // Q-Q reference line
  const qqMin = qqData.length ? Math.min(...qqData.map(d => d.theoretical)) : -3;
  const qqMax = qqData.length ? Math.max(...qqData.map(d => d.theoretical)) : 3;
  const qqRefLine = stats ? [
    { theoretical: qqMin, observed: stats.mean + qqMin * stats.std },
    { theoretical: qqMax, observed: stats.mean + qqMax * stats.std }
  ] : [];

  // Export stats
  const exportStats = () => {
    if (!stats) return;
    const rows = numericCols.map(c => {
      const vals = data.map(r => parseFloat(r[c])).filter(v => !isNaN(v));
      const s = computeStats(vals);
      return [c, s?.n, s?.mean?.toFixed(5), s?.std?.toFixed(5), s?.min?.toFixed(5), s?.q1?.toFixed(5), s?.q2?.toFixed(5), s?.q3?.toFixed(5), s?.max?.toFixed(5), s?.cv?.toFixed(2), s?.skew?.toFixed(4), s?.kurt?.toFixed(4)].join(",");
    });
    const csv = ["Variable,N,Mean,Std,Min,Q1,Median,Q3,Max,CV%,Skewness,Kurtosis", ...rows].join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "univariate_stats.csv"; a.click();
  };

  const TABS = [
    { id: "eda", label: "EDA" },
    { id: "qq", label: "Q-Q Plot" },
    { id: "profile", label: "Sample Profile" },
    { id: "rowprofiles", label: "Row Profiles" },
    { id: "table", label: "All Variables" },
  ];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>📉 Univariate Analysis</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>EDA, Q-Q plot, row profiles, and descriptive statistics</p>
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
          {/* Controls */}
          <div className="glass-card p-4 flex flex-wrap gap-4 items-center">
            <div className="flex items-center gap-2">
              <label className="text-sm" style={{ color: "var(--text-muted)" }}>Variable:</label>
              <select value={selectedCol} onChange={e => setSelectedCol(e.target.value)}
                className="px-3 py-2 rounded-lg text-sm outline-none"
                style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            {stats && <span className="text-xs" style={{ color: "var(--text-muted)" }}>{stats.n} values · mean={stats.mean.toFixed(3)} · σ={stats.std.toFixed(3)}</span>}
            <button onClick={exportStats} className="btn-secondary text-xs flex items-center gap-1.5 ml-auto">
              <Download size={12} /> Export All Stats
            </button>
          </div>

          {stats && (
            <>
              {/* Stats grid */}
              <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
                {[
                  { label: "N", value: stats.n, color: "#1E90FF" },
                  { label: "Mean", value: stats.mean.toFixed(4) },
                  { label: "Std Dev", value: stats.std.toFixed(4) },
                  { label: "Min", value: stats.min.toFixed(4), color: "#ef4444" },
                  { label: "Median", value: stats.q2.toFixed(4) },
                  { label: "Max", value: stats.max.toFixed(4), color: "#10b981" },
                  { label: "Q1", value: stats.q1.toFixed(4) },
                  { label: "Q3", value: stats.q3.toFixed(4) },
                  { label: "IQR", value: stats.iqr.toFixed(4) },
                  { label: "CV%", value: stats.cv.toFixed(2), color: Math.abs(stats.cv) > 30 ? "#f59e0b" : "var(--text-primary)" },
                  { label: "Skewness", value: stats.skew.toFixed(4), color: Math.abs(stats.skew) > 1 ? "#f59e0b" : "var(--text-primary)" },
                  { label: "Kurtosis", value: stats.kurt.toFixed(4), color: Math.abs(stats.kurt) > 2 ? "#8b5cf6" : "var(--text-primary)" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="glass-card p-3">
                    <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
                    <div className="text-sm font-bold" style={{ color: color || "var(--text-primary)" }}>{value}</div>
                  </div>
                ))}
              </div>

              {/* Tab bar */}
              <div className="flex flex-wrap gap-2">
                {TABS.map(t => (
                  <button key={t.id} onClick={() => setTab(t.id)}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                    style={{ background: tab === t.id ? "#06b6d4" : "var(--bg-card)", color: tab === t.id ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                    {t.label}
                  </button>
                ))}
              </div>

              {/* EDA Tab */}
              {tab === "eda" && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Histogram — {selectedCol}</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={hist} margin={{ top: 5, right: 20, bottom: 30, left: 35 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="x" tick={{ fill: "var(--text-muted)", fontSize: 10 }} angle={-30} textAnchor="end" />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} label={{ value: "Count", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip content={({ payload }) => {
                          if (!payload?.length) return null;
                          return (
                            <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                              <div>{payload[0].payload.label}</div>
                              <div>Count: {payload[0].value}</div>
                            </div>
                          );
                        }} />
                        <ReferenceLine x={stats.mean} stroke="#1E90FF" strokeDasharray="4 4" />
                        <Bar dataKey="count" fill="#06b6d4" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Box Summary — {selectedCol}</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={[{ name: "Min", value: stats.min }, { name: "Q1", value: stats.q1 }, { name: "Median", value: stats.q2 }, { name: "Q3", value: stats.q3 }, { name: "Max", value: stats.max }, { name: "Mean", value: stats.mean }]}
                        layout="vertical" margin={{ top: 5, right: 50, bottom: 5, left: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis type="number" tick={{ fill: "var(--text-muted)", fontSize: 10 }} domain={['dataMin - 1', 'dataMax + 1']} />
                        <YAxis type="category" dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                        <Bar dataKey="value" fill="#06b6d4" radius={[0, 4, 4, 0]}
                          label={{ position: "right", fill: "var(--text-muted)", fontSize: 10, formatter: v => v.toFixed(3) }} />
                      </BarChart>
                    </ResponsiveContainer>
                    <div className="mt-3 text-xs" style={{ color: "var(--text-muted)" }}>
                      {Math.abs(stats.skew) > 1 && <span className="tag tag-yellow mr-2">Skewed ({stats.skew > 0 ? "right" : "left"})</span>}
                      {Math.abs(stats.kurt) > 2 && <span className="tag tag-purple">{stats.kurt > 0 ? "Leptokurtic" : "Platykurtic"}</span>}
                    </div>
                  </div>
                </div>
              )}

              {/* Q-Q Tab */}
              {tab === "qq" && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>Normal Q-Q Plot — {selectedCol}</h3>
                  <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Points on the line = normal distribution. Deviations indicate non-normality.</p>
                  <ResponsiveContainer width="100%" height={380}>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 50 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="theoretical" name="Theoretical Quantiles"
                        label={{ value: "Theoretical Quantiles (Normal)", position: "bottom", offset: 25, fill: "var(--text-muted)", fontSize: 12 }}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <YAxis type="number" dataKey="observed" name="Sample Quantiles"
                        label={{ value: "Sample Quantiles", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <Tooltip content={({ payload }) => {
                        if (!payload?.length) return null;
                        const d = payload[0].payload;
                        return (
                          <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                            <div>Theoretical: {d.theoretical}</div>
                            <div>Observed: {d.observed}</div>
                          </div>
                        );
                      }} />
                      <Scatter data={qqData} fill="#06b6d4" opacity={0.7} name="Quantiles" />
                      <Scatter data={qqRefLine} fill="none" line={{ stroke: "#ef4444", strokeWidth: 2, strokeDasharray: "6 3" }} lineType="fitting" name="Normal line" shape={() => null} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Sample Profile Tab */}
              {tab === "profile" && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
                    Sample Profile — {selectedCol} {values.length > 300 ? "(first 300)" : ""}
                  </h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={profileData} margin={{ top: 5, right: 20, bottom: 25, left: 45 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                      <ReferenceLine y={stats.mean} stroke="#1E90FF" strokeDasharray="4 4" label={{ value: "Mean", fill: "#1E90FF", fontSize: 10, position: "insideTopRight" }} />
                      <ReferenceLine y={stats.mean + 2 * stats.std} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: "+2σ", fill: "#f59e0b", fontSize: 10 }} />
                      <ReferenceLine y={stats.mean - 2 * stats.std} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: "-2σ", fill: "#f59e0b", fontSize: 10 }} />
                      <ReferenceLine y={stats.mean + 3 * stats.std} stroke="#ef4444" strokeDasharray="2 4" label={{ value: "+3σ", fill: "#ef4444", fontSize: 10 }} />
                      <ReferenceLine y={stats.mean - 3 * stats.std} stroke="#ef4444" strokeDasharray="2 4" label={{ value: "-3σ", fill: "#ef4444", fontSize: 10 }} />
                      <Line type="monotone" dataKey="value" stroke="#06b6d4" strokeWidth={1.5} dot={false} name={selectedCol} />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>Blue = mean · Orange = ±2σ · Red = ±3σ control limits</p>
                </div>
              )}

              {/* Row Profiles Tab */}
              {tab === "rowprofiles" && (
                <div className="glass-card p-5">
                  <div className="flex flex-wrap gap-3 items-center mb-4">
                    <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Row Profiles (multi-variable)</h3>
                    <div className="flex items-center gap-2 ml-auto">
                      <label className="text-xs" style={{ color: "var(--text-muted)" }}>Variables to show:</label>
                      <select multiple value={profileCols} onChange={e => setProfileCols(Array.from(e.target.selectedOptions, o => o.value))}
                        className="px-2 py-1 rounded-lg text-xs outline-none"
                        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)", height: 80 }}>
                        {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                    </div>
                  </div>
                  <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>X = sample index · Y = variable value · each line = one variable (first 200 samples)</p>
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={rowProfileData} margin={{ top: 5, right: 30, bottom: 25, left: 45 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="sample" label={{ value: "Sample Index", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }} />
                      <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />
                      {profileCols.map((c, i) => (
                        <Line key={c} type="monotone" dataKey={c} stroke={COLORS[i % COLORS.length]} strokeWidth={1.5} dot={false} name={c} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* All Variables Table */}
              {tab === "table" && (
                <div className="glass-card overflow-hidden">
                  <div className="p-4 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
                    <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>All Variables — Descriptive Statistics</h3>
                    <button onClick={exportStats} className="btn-secondary text-xs flex items-center gap-1.5">
                      <Download size={12} /> Export CSV
                    </button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          {["Variable", "N", "Mean", "Std Dev", "Min", "Q1", "Median", "Q3", "Max", "CV%", "Skew", "Kurt"].map(h => (
                            <th key={h} className="px-3 py-2.5 text-left whitespace-nowrap" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {numericCols.map(c => {
                          const vals = data.map(r => parseFloat(r[c])).filter(v => !isNaN(v));
                          const s = computeStats(vals);
                          if (!s) return null;
                          return (
                            <tr key={c} onClick={() => { setSelectedCol(c); setTab("eda"); }}
                              className="cursor-pointer"
                              style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                              <td className="px-3 py-2 font-medium" style={{ color: c === selectedCol ? "#06b6d4" : "var(--text-primary)" }}>{c}</td>
                              <td className="px-3 py-2" style={{ color: "var(--text-muted)" }}>{s.n}</td>
                              <td className="px-3 py-2" style={{ color: "var(--text-primary)" }}>{s.mean.toFixed(4)}</td>
                              <td className="px-3 py-2" style={{ color: "var(--text-muted)" }}>{s.std.toFixed(4)}</td>
                              <td className="px-3 py-2" style={{ color: "#ef4444" }}>{s.min.toFixed(4)}</td>
                              <td className="px-3 py-2" style={{ color: "var(--text-muted)" }}>{s.q1.toFixed(4)}</td>
                              <td className="px-3 py-2" style={{ color: "var(--text-primary)" }}>{s.q2.toFixed(4)}</td>
                              <td className="px-3 py-2" style={{ color: "var(--text-muted)" }}>{s.q3.toFixed(4)}</td>
                              <td className="px-3 py-2" style={{ color: "#10b981" }}>{s.max.toFixed(4)}</td>
                              <td className="px-3 py-2" style={{ color: Math.abs(s.cv) > 30 ? "#f59e0b" : "var(--text-muted)" }}>{s.cv.toFixed(2)}</td>
                              <td className="px-3 py-2" style={{ color: Math.abs(s.skew) > 1 ? "#f59e0b" : "var(--text-muted)" }}>{s.skew.toFixed(3)}</td>
                              <td className="px-3 py-2" style={{ color: Math.abs(s.kurt) > 2 ? "#8b5cf6" : "var(--text-muted)" }}>{s.kurt.toFixed(3)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs p-3" style={{ color: "var(--text-muted)" }}>Click a row to view EDA charts for that variable</p>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}