import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { Sliders, RefreshCw, Download, Info } from "lucide-react";

// ── TRANSFORM DEFINITIONS ────────────────────────────────────────────────────
const TRANSFORMS = [
  // Column (sample) transforms
  { id: "center",    label: "Column Centering",       desc: "Subtract column mean (µ-centering)",           group: "Column" },
  { id: "scale",     label: "Column Scaling",          desc: "Divide by std dev (unit variance)",            group: "Column" },
  { id: "autoscale", label: "Autoscaling (UV)",        desc: "Center + scale (z-score)",                    group: "Column" },
  { id: "range01",   label: "Range [0–1]",             desc: "Min-max normalization",                       group: "Column" },
  { id: "range11",   label: "Range [–1, +1]",          desc: "DoE-style coding",                            group: "Column" },
  { id: "max100",    label: "Max to 100",              desc: "Scale column maximum to 100",                 group: "Column" },
  { id: "sum100col", label: "Column Sum to 100",       desc: "Scale column sum to 100",                     group: "Column" },
  { id: "log10",     label: "Log₁₀",                  desc: "Log10 transformation (column-wise)",          group: "Column" },
  { id: "colderiv1", label: "1st Derivative (col)",   desc: "Column-wise first difference",                group: "Column" },
  { id: "colderiv2", label: "2nd Derivative (col)",   desc: "Column-wise second difference",               group: "Column" },
  // Row / Spectral transforms
  { id: "snv",       label: "SNV",                    desc: "Standard Normal Variate (row autoscaling)",   group: "Row/Spectral" },
  { id: "sum100",    label: "Row Sum to 100",         desc: "Normalize row sums to 100%",                  group: "Row/Spectral" },
  { id: "deriv1",    label: "1st Derivative (row)",   desc: "Row-wise first difference",                   group: "Row/Spectral" },
  { id: "deriv2",    label: "2nd Derivative (row)",   desc: "Row-wise second difference",                  group: "Row/Spectral" },
  { id: "sg1",       label: "Savitzky-Golay 1st",     desc: "SG smoothed first derivative (w=5, poly=2)", group: "Row/Spectral" },
  { id: "sg2",       label: "Savitzky-Golay 2nd",     desc: "SG smoothed second derivative (w=7, poly=3)",group: "Row/Spectral" },
  { id: "ma",        label: "Moving Average",         desc: "Row-wise moving average (window=5)",          group: "Row/Spectral" },
  { id: "detrend",   label: "Detrend (row)",          desc: "Remove linear trend from each row",           group: "Row/Spectral" },
];

// Savitzky-Golay coefficients (symmetric, 1st derivative, window 5, poly 2)
const SG_COEFFS_1D_W5 = [-2, -1, 0, 1, 2].map(v => v / 10);
const SG_COEFFS_2D_W7 = [2, -1, -2, -1, 2].map(v => v / 7); // simplified approx

function savitzkyGolay1(row, windowSize = 5, deriv = 1) {
  const half = Math.floor(windowSize / 2);
  const result = [];
  for (let i = 0; i < row.length; i++) {
    if (i < half || i >= row.length - half) { result.push(0); continue; }
    let val = 0;
    for (let k = -half; k <= half; k++) {
      val += row[i + k] * (deriv === 1 ? SG_COEFFS_1D_W5[k + half] : SG_COEFFS_2D_W7[Math.abs(k)]);
    }
    result.push(val);
  }
  return result.slice(half, result.length - half);
}

function applyTransform(mat, id, nRows, nCols) {
  const n = nRows, p = nCols;

  if (id === "center") {
    const means = Array(p).fill(0).map((_, j) => mat.reduce((s, r) => s + r[j], 0) / n);
    return { result: mat.map(r => r.map((v, j) => v - means[j])), newCols: null };
  }
  if (id === "scale") {
    const means = Array(p).fill(0).map((_, j) => mat.reduce((s, r) => s + r[j], 0) / n);
    const stds = means.map((m, j) => Math.sqrt(mat.reduce((s, r) => s + (r[j] - m) ** 2, 0) / (n - 1)) || 1);
    return { result: mat.map(r => r.map((v, j) => v / stds[j])), newCols: null };
  }
  if (id === "autoscale") {
    const means = Array(p).fill(0).map((_, j) => mat.reduce((s, r) => s + r[j], 0) / n);
    const stds = means.map((m, j) => Math.sqrt(mat.reduce((s, r) => s + (r[j] - m) ** 2, 0) / (n - 1)) || 1);
    return { result: mat.map(r => r.map((v, j) => (v - means[j]) / stds[j])), newCols: null };
  }
  if (id === "range01") {
    const mins = Array(p).fill(0).map((_, j) => Math.min(...mat.map(r => r[j])));
    const maxs = Array(p).fill(0).map((_, j) => Math.max(...mat.map(r => r[j])));
    return { result: mat.map(r => r.map((v, j) => (maxs[j] - mins[j]) === 0 ? 0 : (v - mins[j]) / (maxs[j] - mins[j]))), newCols: null };
  }
  if (id === "range11") {
    const mins = Array(p).fill(0).map((_, j) => Math.min(...mat.map(r => r[j])));
    const maxs = Array(p).fill(0).map((_, j) => Math.max(...mat.map(r => r[j])));
    return { result: mat.map(r => r.map((v, j) => (maxs[j] - mins[j]) === 0 ? 0 : 2 * (v - mins[j]) / (maxs[j] - mins[j]) - 1)), newCols: null };
  }
  if (id === "max100") {
    const maxs = Array(p).fill(0).map((_, j) => Math.max(...mat.map(r => r[j])));
    return { result: mat.map(r => r.map((v, j) => maxs[j] === 0 ? 0 : v / maxs[j] * 100)), newCols: null };
  }
  if (id === "sum100col") {
    const sums = Array(p).fill(0).map((_, j) => mat.reduce((s, r) => s + r[j], 0));
    return { result: mat.map(r => r.map((v, j) => sums[j] === 0 ? 0 : v / sums[j] * 100)), newCols: null };
  }
  if (id === "log10") {
    const mins = Array(p).fill(0).map((_, j) => Math.min(...mat.map(r => r[j])));
    return { result: mat.map(r => r.map((v, j) => { const val = mins[j] <= 0 ? v + Math.abs(mins[j]) + 1 : v; return Math.log10(val); })), newCols: null };
  }
  if (id === "colderiv1") {
    return { result: mat.slice(1).map((r, i) => r.map((v, j) => v - mat[i][j])), newCols: null };
  }
  if (id === "colderiv2") {
    const d1 = mat.slice(1).map((r, i) => r.map((v, j) => v - mat[i][j]));
    return { result: d1.slice(1).map((r, i) => r.map((v, j) => v - d1[i][j])), newCols: null };
  }
  if (id === "snv") {
    return { result: mat.map(r => { const m = r.reduce((a, b) => a + b, 0) / r.length; const s = Math.sqrt(r.reduce((a, v) => a + (v - m) ** 2, 0) / (r.length - 1)) || 1; return r.map(v => (v - m) / s); }), newCols: null };
  }
  if (id === "sum100") {
    return { result: mat.map(r => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map(v => v / s * 100); }), newCols: null };
  }
  if (id === "deriv1") {
    return { result: mat.map(r => r.slice(1).map((v, i) => v - r[i])), newCols: "trim1" };
  }
  if (id === "deriv2") {
    const d1 = mat.map(r => r.slice(1).map((v, i) => v - r[i]));
    return { result: d1.map(r => r.slice(1).map((v, i) => v - r[i])), newCols: "trim2" };
  }
  if (id === "sg1") {
    const res = mat.map(r => savitzkyGolay1(r, 5, 1));
    return { result: res, newCols: "trim2" };
  }
  if (id === "sg2") {
    const res = mat.map(r => savitzkyGolay1(r, 7, 2));
    return { result: res, newCols: "trim3" };
  }
  if (id === "ma") {
    const w = 5, half = Math.floor(w / 2);
    const res = mat.map(r => {
      const out = [];
      for (let i = half; i < r.length - half; i++) {
        out.push(r.slice(i - half, i + half + 1).reduce((a, b) => a + b, 0) / w);
      }
      return out;
    });
    return { result: res, newCols: `trimMA${half}` };
  }
  if (id === "detrend") {
    return {
      result: mat.map(r => {
        const n2 = r.length;
        const xs = Array.from({ length: n2 }, (_, i) => i);
        const mx = (n2 - 1) / 2;
        const my = r.reduce((a, b) => a + b, 0) / n2;
        const b = xs.reduce((s, x, i) => s + (x - mx) * (r[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
        const a = my - b * mx;
        return r.map((v, i) => v - (a + b * i));
      }), newCols: null
    };
  }
  return { result: mat, newCols: null };
}

const COLORS = ["#1E90FF", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];

export default function Preprocessing() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [selectedTransform, setSelectedTransform] = useState("snv");
  const [firstCol, setFirstCol] = useState(0);
  const [lastCol, setLastCol] = useState(0);
  const [firstRow, setFirstRow] = useState(0);
  const [lastRow, setLastRow] = useState(0);
  const [transformed, setTransformed] = useState(null);
  const [selectedCols, setSelectedCols] = useState([]);

  const loadData = async (id, ds) => {
    setDataset(ds); setTransformed(null);
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const nc = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    const allCols = ds?.columns || [];
    const firstIdx = nc.length ? allCols.indexOf(nc[0]) : 0;
    const lastIdx = nc.length ? allCols.indexOf(nc[nc.length - 1]) : 0;
    setFirstCol(firstIdx);
    setLastCol(lastIdx);
    setFirstRow(0);
    setLastRow(rows.length - 1);
    setSelectedCols(nc.slice(0, 5));
  };

  const runTransform = () => {
    if (!data.length) return;
    const allCols = dataset?.columns || [];
    const cols = allCols.slice(firstCol, lastCol + 1).filter(c => !isNaN(parseFloat(data[0]?.[c])));
    if (!cols.length) return;
    const rowSubset = data.slice(firstRow, lastRow + 1);
    const mat = rowSubset.map(r => cols.map(c => parseFloat(r[c]) || 0));
    const { result, newCols: trimInfo } = applyTransform(mat, selectedTransform, mat.length, mat[0]?.length || 0);
    let finalCols = cols;
    if (trimInfo === "trim1") finalCols = cols.slice(1);
    else if (trimInfo === "trim2") finalCols = cols.slice(2);
    else if (trimInfo === "trim3") finalCols = cols.slice(3);
    else if (trimInfo && trimInfo.startsWith("trimMA")) {
      const half = parseInt(trimInfo.replace("trimMA", ""));
      finalCols = cols.slice(half, cols.length - half);
    }
    setTransformed({ result, cols: finalCols, origCols: cols, origMat: mat });
  };

  const exportCSV = () => {
    if (!transformed) return;
    const csv = [transformed.cols.join(","), ...transformed.result.map(r => r.map(v => v.toFixed(6)).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = `${dataset.name}_${selectedTransform}.csv`;
    a.click();
  };

  const allCols = dataset?.columns || [];
  const numCols = allCols.filter(c => !isNaN(parseFloat(data[0]?.[c])));
  const selectedT = TRANSFORMS.find(t => t.id === selectedTransform);
  const groups = [...new Set(TRANSFORMS.map(t => t.group))];

  // Chart data: show first 8 samples as spectral profiles
  const nDisplaySamples = Math.min(8, transformed?.result?.length || 5);
  const origPlotCols = transformed?.origCols || numCols.slice(firstCol, lastCol + 1);
  const origLineData = origPlotCols.map((c, ci) => {
    const pt = { x: ci, col: c.length > 12 ? `${c.slice(0, 10)}…` : c };
    for (let s = 0; s < nDisplaySamples; s++) {
      pt[`s${s}`] = parseFloat(data[firstRow + s]?.[c]) || 0;
    }
    return pt;
  });
  const transLineData = transformed ? transformed.cols.map((c, ci) => {
    const pt = { x: ci, col: c.length > 12 ? `${c.slice(0, 10)}…` : c };
    for (let s = 0; s < nDisplaySamples; s++) {
      pt[`s${s}`] = parseFloat((transformed.result[s]?.[ci] ?? 0).toFixed(5));
    }
    return pt;
  }) : [];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>⚙️ Preprocessing</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>SNV, derivatives, Savitzky-Golay, autoscaling, centering and more</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <Sliders size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a dataset to apply preprocessing transformations</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Transform picker */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Select Transformation</h3>
            {groups.map(g => (
              <div key={g} className="mb-4">
                <p className="text-xs uppercase tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>{g}</p>
                <div className="space-y-1">
                  {TRANSFORMS.filter(t => t.group === g).map(t => (
                    <button key={t.id} onClick={() => setSelectedTransform(t.id)}
                      className="w-full text-left px-3 py-2 rounded-lg transition-all"
                      style={{
                        background: selectedTransform === t.id ? "rgba(245,158,11,0.15)" : "transparent",
                        border: `1px solid ${selectedTransform === t.id ? "rgba(245,158,11,0.3)" : "transparent"}`,
                        color: selectedTransform === t.id ? "#fbbf24" : "var(--text-secondary)"
                      }}>
                      <div className="text-xs font-semibold">{t.label}</div>
                      <div className="text-xs opacity-70">{t.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Config + preview */}
          <div className="lg:col-span-2 space-y-4">
            <div className="glass-card p-5">
              <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
                ⚙️ {selectedT?.label} — Selection
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                <div>
                  <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>First column</label>
                  <select value={firstCol} onChange={e => setFirstCol(+e.target.value)}
                    className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                    style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                    {allCols.map((c, i) => <option key={i} value={i}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Last column</label>
                  <select value={lastCol} onChange={e => setLastCol(+e.target.value)}
                    className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                    style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                    {allCols.map((c, i) => <option key={i} value={i}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>First row (0-based)</label>
                  <input type="number" min={0} max={data.length - 1} value={firstRow} onChange={e => setFirstRow(+e.target.value)}
                    className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                    style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
                </div>
                <div>
                  <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Last row (0-based)</label>
                  <input type="number" min={0} max={data.length - 1} value={lastRow} onChange={e => setLastRow(+e.target.value)}
                    className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                    style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button onClick={runTransform}
                  className="btn-primary"
                  style={{ background: "linear-gradient(135deg, #f59e0b, #d97706)" }}>
                  <RefreshCw size={14} /> Apply Transform
                </button>
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  {allCols.slice(firstCol, lastCol + 1).length} cols × {lastRow - firstRow + 1} rows selected
                </p>
              </div>
            </div>

            {/* Comparison charts */}
            {transformed && (
              <div className="space-y-4">
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--text-primary)" }}>
                    Original Data — {nDisplaySamples} samples spectral profile
                  </h3>
                  <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>
                    {transformed.origCols.length} variables ({allCols[firstCol]} → {allCols[lastCol]})
                  </p>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={origLineData} margin={{ top: 5, right: 10, bottom: 5, left: 35 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="x" tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickFormatter={v => origPlotCols[v]?.slice(0, 6) || v} />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }}
                        labelFormatter={v => origPlotCols[v] || `Col ${v}`} />
                      {Array.from({ length: nDisplaySamples }, (_, s) => (
                        <Line key={s} type="monotone" dataKey={`s${s}`} stroke={COLORS[s % COLORS.length]} strokeWidth={1.5} dot={false} name={`Sample ${firstRow + s + 1}`} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-1" style={{ color: "#f59e0b" }}>
                    After {selectedT?.label}
                  </h3>
                  <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>
                    {transformed.cols.length} variables (may differ due to derivative trimming)
                  </p>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={transLineData} margin={{ top: 5, right: 10, bottom: 5, left: 35 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="x" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }} />
                      {Array.from({ length: nDisplaySamples }, (_, s) => (
                        <Line key={s} type="monotone" dataKey={`s${s}`} stroke={COLORS[s % COLORS.length]} strokeWidth={1.5} dot={false} name={`Sample ${firstRow + s + 1}`} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Stats comparison */}
                <div className="glass-card p-4">
                  <h3 className="text-xs font-semibold mb-3" style={{ color: "var(--text-muted)" }}>SUMMARY STATISTICS</h3>
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div>
                      <p className="mb-1 font-semibold" style={{ color: "var(--text-secondary)" }}>Original</p>
                      {["Mean", "Std", "Min", "Max"].map(stat => {
                        const allVals = transformed.origMat.flat();
                        const mean = allVals.reduce((a, b) => a + b, 0) / allVals.length;
                        const std = Math.sqrt(allVals.reduce((s, v) => s + (v - mean) ** 2, 0) / allVals.length);
                        const vals = { Mean: mean, Std: std, Min: Math.min(...allVals), Max: Math.max(...allVals) };
                        return <div key={stat} className="flex justify-between py-0.5" style={{ color: "var(--text-muted)" }}><span>{stat}</span><span style={{ color: "var(--text-primary)" }}>{vals[stat].toFixed(4)}</span></div>;
                      })}
                    </div>
                    <div>
                      <p className="mb-1 font-semibold" style={{ color: "#f59e0b" }}>Transformed</p>
                      {["Mean", "Std", "Min", "Max"].map(stat => {
                        const allVals = transformed.result.flat();
                        const mean = allVals.reduce((a, b) => a + b, 0) / allVals.length;
                        const std = Math.sqrt(allVals.reduce((s, v) => s + (v - mean) ** 2, 0) / allVals.length);
                        const vals = { Mean: mean, Std: std, Min: Math.min(...allVals), Max: Math.max(...allVals) };
                        return <div key={stat} className="flex justify-between py-0.5" style={{ color: "var(--text-muted)" }}><span>{stat}</span><span style={{ color: "#f59e0b" }}>{vals[stat].toFixed(4)}</span></div>;
                      })}
                    </div>
                  </div>
                </div>

                <button onClick={exportCSV} className="btn-secondary flex items-center gap-2 text-sm">
                  <Download size={14} /> Export Transformed Data as CSV
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}