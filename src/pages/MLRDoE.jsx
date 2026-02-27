import { useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, LineChart, Line, Legend
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { GitBranch, RefreshCw, AlertCircle } from "lucide-react";

function fitMLR(X, y) {
  const n = X.length, p = X[0].length;
  // Add intercept
  const Xb = X.map(r => [1, ...r]);
  const pb = p + 1;
  // X'X
  const XtX = Array.from({ length: pb }, (_, i) => Array.from({ length: pb }, (_, j) =>
    Xb.reduce((s, r) => s + r[i] * r[j], 0)
  ));
  // X'y
  const Xty = Array.from({ length: pb }, (_, i) => Xb.reduce((s, r, k) => s + r[i] * y[k], 0));
  // Solve via Gaussian elimination
  const mat = XtX.map((r, i) => [...r, Xty[i]]);
  for (let col = 0; col < pb; col++) {
    let maxRow = col;
    for (let row = col + 1; row < pb; row++) if (Math.abs(mat[row][col]) > Math.abs(mat[maxRow][col])) maxRow = row;
    [mat[col], mat[maxRow]] = [mat[maxRow], mat[col]];
    if (Math.abs(mat[col][col]) < 1e-12) continue;
    for (let row = 0; row < pb; row++) {
      if (row === col) continue;
      const factor = mat[row][col] / mat[col][col];
      for (let j = col; j <= pb; j++) mat[row][j] -= factor * mat[col][j];
    }
  }
  const coefs = mat.map((r, i) => r[pb] / r[i]);
  const yPred = Xb.map(r => r.reduce((s, v, j) => s + v * coefs[j], 0));
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const ssTot = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
  const ssRes = y.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0);
  const r2 = 1 - ssRes / ssTot;
  const mse = ssRes / (n - pb);
  const rmse = Math.sqrt(mse);
  // t-values (simplified)
  const invDiag = mat.map((r, i) => 1 / (r[i] || 1e-10));
  const se = invDiag.map(v => Math.sqrt(Math.abs(mse * v)));
  const tvals = coefs.map((c, i) => c / (se[i] || 1));
  return { coefs, yPred, r2, rmse, mse, tvals, se, n, p: pb };
}

export default function MLRDoE() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [xCols, setXCols] = useState([]);
  const [yCol, setYCol] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("model");

  const loadData = async (id, ds) => {
    setDataset(ds); setResult(null); setError("");
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const numCols = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    if (numCols.length >= 2) { setYCol(numCols[numCols.length - 1]); setXCols(numCols.slice(0, -1)); }
  };

  const runMLR = () => {
    if (!data.length || !yCol || !xCols.length) return;
    setLoading(true); setError("");
    setTimeout(() => {
      const X = data.map(r => xCols.map(c => parseFloat(r[c]) || 0));
      const y = data.map(r => parseFloat(r[yCol]) || 0);
      const res = fitMLR(X, y);
      const coefRows = [{ name: "Intercept", coef: res.coefs[0], se: res.se[0], t: res.tvals[0] },
        ...xCols.map((c, i) => ({ name: c, coef: res.coefs[i + 1], se: res.se[i + 1], t: res.tvals[i + 1] }))];
      const residuals = y.map((v, i) => ({ sample: i + 1, actual: parseFloat(v.toFixed(4)), predicted: parseFloat(res.yPred[i].toFixed(4)), residual: parseFloat((v - res.yPred[i]).toFixed(4)) }));
      setResult({ ...res, coefRows, residuals, xCols, yCol });
      setLoading(false);
    }, 50);
  };

  const numericCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>🧪 MLR & DoE</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Multiple Linear Regression & Design of Experiments</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <GitBranch size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a dataset to begin MLR analysis</p>
        </div>
      ) : (
        <>
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>⚙️ Model Setup</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Response (Y)</label>
                <select value={yCol} onChange={e => setYCol(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  <option value="">Select Y column</option>
                  {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Predictors (X) — hold Ctrl to multi-select</label>
                <select multiple value={xCols} onChange={e => setXCols(Array.from(e.target.selectedOptions, o => o.value))}
                  className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)", height: 80 }}>
                  {numericCols.filter(c => c !== yCol).map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>
            <button onClick={runMLR} disabled={loading || !data.length || !yCol || !xCols.length}
              className="btn-primary disabled:opacity-40"
              style={{ background: "linear-gradient(135deg, #8b5cf6, #7c3aed)" }}>
              {loading ? <RefreshCw size={14} className="animate-spin" /> : <GitBranch size={14} />}
              {loading ? "Fitting..." : "Fit MLR Model"}
            </button>
            {error && <div className="flex items-center gap-2 mt-3 text-sm" style={{ color: "#f87171" }}><AlertCircle size={14} /> {error}</div>}
          </div>

          {result && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "R²", value: result.r2.toFixed(4), color: result.r2 > 0.9 ? "#10b981" : result.r2 > 0.7 ? "#f59e0b" : "#ef4444" },
                  { label: "RMSE", value: result.rmse.toFixed(4), color: "#8b5cf6" },
                  { label: "Samples", value: result.n, color: "#1E90FF" },
                  { label: "Predictors", value: result.xCols.length, color: "#06b6d4" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="glass-card p-4">
                    <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
                    <div className="text-xl font-bold" style={{ color }}>{value}</div>
                  </div>
                ))}
              </div>

              <div className="flex gap-2">
                {["model", "predicted", "residuals"].map(t => (
                  <button key={t} onClick={() => setTab(t)}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all capitalize"
                    style={{ background: tab === t ? "#8b5cf6" : "var(--bg-card)", color: tab === t ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                    {t === "model" ? "Coefficients" : t === "predicted" ? "Predicted vs Actual" : "Residuals"}
                  </button>
                ))}
              </div>

              {tab === "model" && (
                <div className="glass-card overflow-hidden">
                  <div className="p-4" style={{ borderBottom: "1px solid var(--border)" }}>
                    <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Model Coefficients</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>{["Term", "Coefficient", "Std Error", "t-value", "Significance"].map(h => (
                          <th key={h} className="px-4 py-3 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                        ))}</tr>
                      </thead>
                      <tbody>
                        {result.coefRows.map((row, i) => {
                          const sig = Math.abs(row.t) > 3.5 ? "***" : Math.abs(row.t) > 2.5 ? "**" : Math.abs(row.t) > 2 ? "*" : "";
                          return (
                            <tr key={i} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                              <td className="px-4 py-2.5 font-medium" style={{ color: "var(--text-primary)" }}>{row.name}</td>
                              <td className="px-4 py-2.5" style={{ color: row.coef >= 0 ? "#10b981" : "#ef4444" }}>{row.coef.toFixed(5)}</td>
                              <td className="px-4 py-2.5" style={{ color: "var(--text-secondary)" }}>{row.se.toFixed(5)}</td>
                              <td className="px-4 py-2.5" style={{ color: Math.abs(row.t) > 2 ? "#f59e0b" : "var(--text-secondary)" }}>{row.t.toFixed(3)}</td>
                              <td className="px-4 py-2.5 font-bold" style={{ color: "#f59e0b" }}>{sig}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div className="glass-card p-4 m-3">
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={result.coefRows.slice(1)} margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} angle={-30} textAnchor="end" />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                        <ReferenceLine y={0} stroke="var(--border-light)" />
                        <Bar dataKey="coef" name="Coefficient" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {tab === "predicted" && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Predicted vs Actual (R² = {result.r2.toFixed(4)})</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="actual" name="Actual" label={{ value: "Actual", position: "bottom", fill: "var(--text-muted)", fontSize: 12 }} tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <YAxis type="number" dataKey="predicted" name="Predicted" label={{ value: "Predicted", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }} tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <Tooltip content={({ payload }) => {
                        if (!payload?.length) return null;
                        const d = payload[0].payload;
                        return (
                          <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                            <div className="font-semibold">Sample {d.sample}</div>
                            <div>Actual: {d.actual}</div>
                            <div>Predicted: {d.predicted}</div>
                          </div>
                        );
                      }} />
                      <Scatter data={result.residuals} fill="#8b5cf6" opacity={0.8} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}

              {tab === "residuals" && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Residuals Plot</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={result.residuals} margin={{ top: 5, right: 20, bottom: 20, left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                      <ReferenceLine y={0} stroke="var(--border-light)" strokeDasharray="4 4" />
                      <Line type="monotone" dataKey="residual" stroke="#8b5cf6" strokeWidth={1.5} dot={{ r: 3, fill: "#8b5cf6" }} name="Residual" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}