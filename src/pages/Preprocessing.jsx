import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { Sliders, RefreshCw, Download } from "lucide-react";

const TRANSFORMS = [
  { id: "center", label: "Column Centering", desc: "Subtract column mean", group: "Column" },
  { id: "scale", label: "Column Scaling", desc: "Divide by std dev", group: "Column" },
  { id: "autoscale", label: "Autoscaling (UV)", desc: "Center + scale (unit variance)", group: "Column" },
  { id: "range01", label: "Range [0–1]", desc: "Min-max normalization", group: "Column" },
  { id: "range11", label: "Range [–1, 1]", desc: "DoE-style coding", group: "Column" },
  { id: "log10", label: "Log₁₀", desc: "Log10 transformation (column-wise)", group: "Column" },
  { id: "snv", label: "SNV", desc: "Standard Normal Variate (row-wise)", group: "Row/Spectral" },
  { id: "deriv1", label: "1st Derivative", desc: "Row-wise first difference", group: "Row/Spectral" },
  { id: "deriv2", label: "2nd Derivative", desc: "Row-wise second difference", group: "Row/Spectral" },
  { id: "sum100", label: "Sum to 100", desc: "Normalize row sums to 100%", group: "Row/Spectral" },
];

function applyTransform(data, columns, id) {
  const mat = data.map(r => columns.map(c => parseFloat(r[c]) || 0));
  const n = mat.length, p = mat[0]?.length || 0;
  let result;

  if (id === "center") {
    const means = columns.map((_, j) => mat.reduce((s, r) => s + r[j], 0) / n);
    result = mat.map(r => r.map((v, j) => v - means[j]));
  } else if (id === "scale") {
    const means = columns.map((_, j) => mat.reduce((s, r) => s + r[j], 0) / n);
    const stds = columns.map((_, j) => Math.sqrt(mat.reduce((s, r) => s + (r[j] - means[j]) ** 2, 0) / (n - 1)) || 1);
    result = mat.map(r => r.map((v, j) => v / stds[j]));
  } else if (id === "autoscale") {
    const means = columns.map((_, j) => mat.reduce((s, r) => s + r[j], 0) / n);
    const stds = columns.map((_, j) => Math.sqrt(mat.reduce((s, r) => s + (r[j] - means[j]) ** 2, 0) / (n - 1)) || 1);
    result = mat.map(r => r.map((v, j) => (v - means[j]) / stds[j]));
  } else if (id === "range01") {
    const mins = columns.map((_, j) => Math.min(...mat.map(r => r[j])));
    const maxs = columns.map((_, j) => Math.max(...mat.map(r => r[j])));
    result = mat.map(r => r.map((v, j) => (maxs[j] - mins[j]) === 0 ? 0 : (v - mins[j]) / (maxs[j] - mins[j])));
  } else if (id === "range11") {
    const mins = columns.map((_, j) => Math.min(...mat.map(r => r[j])));
    const maxs = columns.map((_, j) => Math.max(...mat.map(r => r[j])));
    result = mat.map(r => r.map((v, j) => (maxs[j] - mins[j]) === 0 ? 0 : 2 * (v - mins[j]) / (maxs[j] - mins[j]) - 1));
  } else if (id === "log10") {
    const mins = columns.map((_, j) => Math.min(...mat.map(r => r[j])));
    result = mat.map(r => r.map((v, j) => {
      const val = mins[j] <= 0 ? v + Math.abs(mins[j]) + 1 : v;
      return Math.log10(val);
    }));
  } else if (id === "snv") {
    result = mat.map(r => {
      const m = r.reduce((a, b) => a + b, 0) / r.length;
      const s = Math.sqrt(r.reduce((a, v) => a + (v - m) ** 2, 0) / (r.length - 1)) || 1;
      return r.map(v => (v - m) / s);
    });
  } else if (id === "deriv1") {
    result = mat.map(r => r.slice(1).map((v, i) => v - r[i]));
    return { result, newCols: columns.slice(1) };
  } else if (id === "deriv2") {
    const d1 = mat.map(r => r.slice(1).map((v, i) => v - r[i]));
    result = d1.map(r => r.slice(1).map((v, i) => v - r[i]));
    return { result, newCols: columns.slice(2) };
  } else if (id === "sum100") {
    result = mat.map(r => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map(v => v / s * 100); });
  } else {
    result = mat;
  }
  return { result, newCols: columns };
}

const COLORS = ["#1E90FF", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];

export default function Preprocessing() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [selectedTransform, setSelectedTransform] = useState("autoscale");
  const [firstCol, setFirstCol] = useState(0);
  const [lastCol, setLastCol] = useState(0);
  const [transformed, setTransformed] = useState(null);

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
  };

  const runTransform = () => {
    if (!data.length) return;
    const allCols = dataset?.columns || [];
    const cols = allCols.slice(firstCol, lastCol + 1).filter(c => !isNaN(parseFloat(data[0]?.[c])));
    if (!cols.length) return;
    const { result, newCols } = applyTransform(data, cols, selectedTransform);
    setTransformed({ result, cols: newCols });
  };

  const allCols = dataset?.columns || [];
  const numCols = allCols.filter(c => !isNaN(parseFloat(data[0]?.[c])));

  // Build line chart data for original vs transformed (first 10 samples, first selected col)
  const chartColOrig = numCols.slice(firstCol, lastCol + 1);
  const origLineData = data.slice(0, 5).map((row, i) => {
    const point = { sample: `S${i + 1}` };
    chartColOrig.forEach((c, j) => { point[`x${j}`] = parseFloat(row[c]) || 0; });
    return point;
  });
  const transLineData = transformed ? transformed.result.slice(0, 5).map((row, i) => {
    const point = { sample: `S${i + 1}` };
    row.forEach((v, j) => { point[`x${j}`] = parseFloat(v.toFixed(4)); });
    return point;
  }) : [];
  const nLines = Math.min(chartColOrig.length, 30);

  const selectedT = TRANSFORMS.find(t => t.id === selectedTransform);
  const groups = [...new Set(TRANSFORMS.map(t => t.group))];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>⚙️ Preprocessing</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Data transformations: SNV, derivatives, scaling, autoscaling and more</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <Sliders size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a dataset to apply preprocessing transformations</p>
        </div>
      ) : (
        <>
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
                        className="w-full text-left px-3 py-2.5 rounded-lg transition-all"
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
                  ⚙️ {selectedT?.label} — Column Range
                </h3>
                <div className="flex flex-wrap gap-4 items-end mb-4">
                  <div>
                    <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>First column</label>
                    <select value={firstCol} onChange={e => setFirstCol(+e.target.value)}
                      className="px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {allCols.map((c, i) => <option key={i} value={i}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Last column</label>
                    <select value={lastCol} onChange={e => setLastCol(+e.target.value)}
                      className="px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {allCols.map((c, i) => <option key={i} value={i}>{c}</option>)}
                    </select>
                  </div>
                  <button onClick={runTransform}
                    className="btn-primary"
                    style={{ background: "linear-gradient(135deg, #f59e0b, #d97706)" }}>
                    <RefreshCw size={14} /> Apply Transform
                  </button>
                </div>
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Selected: {allCols.slice(firstCol, lastCol + 1).length} columns ({allCols[firstCol]} → {allCols[lastCol]})
                </p>
              </div>

              {/* Comparison charts */}
              {transformed && (
                <div className="grid grid-cols-1 gap-4">
                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--text-primary)" }}>Original Data</h3>
                    <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>First 5 samples × {nLines} variables</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={origLineData} margin={{ top: 5, right: 10, bottom: 5, left: 30 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="sample" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }} />
                        {Array.from({ length: nLines }, (_, i) => (
                          <Line key={i} type="monotone" dataKey={`x${i}`} stroke={COLORS[i % COLORS.length]} strokeWidth={1.5} dot={false} name={chartColOrig[i]} />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-1" style={{ color: "#f59e0b" }}>After {selectedT?.label}</h3>
                    <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>First 5 samples × {nLines} variables</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={transLineData} margin={{ top: 5, right: 10, bottom: 5, left: 30 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="sample" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }} />
                        {Array.from({ length: Math.min(nLines, transformed.cols.length) }, (_, i) => (
                          <Line key={i} type="monotone" dataKey={`x${i}`} stroke={COLORS[i % COLORS.length]} strokeWidth={1.5} dot={false} name={transformed.cols[i]} />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Export CSV */}
                  <button
                    onClick={() => {
                      const cols = transformed.cols;
                      const csv = [cols.join(","), ...transformed.result.map(r => r.map(v => v.toFixed(6)).join(","))].join("\n");
                      const a = document.createElement("a");
                      a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
                      a.download = `${dataset.name}_${selectedTransform}.csv`;
                      a.click();
                    }}
                    className="btn-secondary flex items-center gap-2 text-sm"
                  >
                    <Download size={14} /> Export Transformed Data as CSV
                  </button>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}