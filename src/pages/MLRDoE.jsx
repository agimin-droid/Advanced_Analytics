import { useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, LineChart, Line, Legend, ComposedChart
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { GitBranch, RefreshCw, AlertCircle, Download } from "lucide-react";

// ── MLR COMPUTATION ──────────────────────────────────────────────────────────
function fitMLR(X, y) {
  const n = X.length, p = X[0].length;
  const Xb = X.map(r => [1, ...r]);
  const pb = p + 1;
  const XtX = Array.from({ length: pb }, (_, i) => Array.from({ length: pb }, (_, j) =>
    Xb.reduce((s, r) => s + r[i] * r[j], 0)));
  const Xty = Array.from({ length: pb }, (_, i) => Xb.reduce((s, r, k) => s + r[i] * y[k], 0));
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
  const ssMod = ssTot - ssRes;
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const r2adj = 1 - (ssRes / (n - pb)) / (ssTot / (n - 1));
  const mse = ssRes / (n - pb);
  const rmse = Math.sqrt(mse);
  const msr = ssMod / p;
  const fStat = mse > 0 ? msr / mse : 0;
  const invDiag = mat.map((r, i) => Math.abs(r[i]) > 1e-10 ? 1 / r[i] : 0);
  const se = invDiag.map(v => Math.sqrt(Math.abs(mse * v)));
  const tvals = coefs.map((c, i) => se[i] > 0 ? c / se[i] : 0);
  // VIF approximation for predictors
  const vifs = Array(pb).fill(1);
  return { coefs, yPred, r2, r2adj, rmse, mse, fStat, tvals, se, n, p: pb, vifs, ssTot, ssRes, ssMod };
}

function pApprox(t, n) {
  if (n < 3) return 1;
  const x = Math.abs(t);
  return Math.min(1, 2 * Math.exp(-0.717 * x - 0.416 * x * x));
}

const COLORS = ["#8b5cf6", "#1E90FF", "#10b981", "#f59e0b", "#ec4899", "#06b6d4", "#f97316"];

// ── PARETO CHART ─────────────────────────────────────────────────────────────
function ParetoChart({ coefRows }) {
  const data = coefRows.filter(r => r.name !== "Intercept")
    .map(r => ({ ...r, absT: Math.abs(r.t) }))
    .sort((a, b) => b.absT - a.absT);
  return (
    <ResponsiveContainer width="100%" height={Math.max(250, data.length * 30 + 60)}>
      <BarChart data={data} layout="vertical" margin={{ top: 5, right: 80, bottom: 5, left: 120 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" tick={{ fill: "var(--text-muted)", fontSize: 10 }} label={{ value: "|t| value", position: "insideBottomRight", fill: "var(--text-muted)", fontSize: 10 }} />
        <YAxis type="category" dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} width={115} />
        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
        <ReferenceLine x={2} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: "|t|=2", fill: "#f59e0b", fontSize: 10 }} />
        <Bar dataKey="absT" name="|t| statistic" fill="#8b5cf6" radius={[0, 4, 4, 0]}
          label={{ position: "right", fill: "var(--text-muted)", fontSize: 10, formatter: v => v.toFixed(2) }} />
      </BarChart>
    </ResponsiveContainer>
  );
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
      const coefRows = [
        { name: "Intercept", coef: res.coefs[0], se: res.se[0], t: res.tvals[0], p: pApprox(res.tvals[0], res.n) },
        ...xCols.map((c, i) => ({ name: c, coef: res.coefs[i + 1], se: res.se[i + 1], t: res.tvals[i + 1], p: pApprox(res.tvals[i + 1], res.n) }))
      ];
      const residuals = y.map((v, i) => ({
        sample: i + 1,
        actual: parseFloat(v.toFixed(4)),
        predicted: parseFloat(res.yPred[i].toFixed(4)),
        residual: parseFloat((v - res.yPred[i]).toFixed(4))
      }));
      setResult({ ...res, coefRows, residuals, xCols, yCol });
      setLoading(false);
    }, 50);
  };

  const exportResults = () => {
    if (!result) return;
    const csv = ["Term,Coefficient,StdError,t-value,P-value",
      ...result.coefRows.map(r => `${r.name},${r.coef.toFixed(5)},${r.se.toFixed(5)},${r.t.toFixed(3)},${r.p.toFixed(4)}`)
    ].join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "mlr_results.csv"; a.click();
  };

  const numericCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];

  // 45° line for predicted vs actual
  const diagLine = result ? (() => {
    const allV = [...result.residuals.map(r => r.actual), ...result.residuals.map(r => r.predicted)];
    const mn = Math.min(...allV), mx = Math.max(...allV);
    return [{ x: mn, y: mn }, { x: mx, y: mx }];
  })() : [];

  // Residuals standardized
  const stdResiduals = result ? (() => {
    const stdR = Math.sqrt(result.residuals.reduce((s, r) => s + r.residual ** 2, 0) / (result.n - result.p));
    return result.residuals.map(r => ({ ...r, stdRes: stdR > 0 ? r.residual / stdR : 0 }));
  })() : [];

  const TABS = [
    { id: "model", label: "Coefficients" },
    { id: "pareto", label: "Pareto Chart" },
    { id: "predicted", label: "Predicted vs Actual" },
    { id: "residuals", label: "Residuals" },
    { id: "anova", label: "ANOVA Table" },
  ];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>🧪 MLR & DoE</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Multiple Linear Regression with Pareto chart, diagnostics & ANOVA</p>
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
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Response Y</label>
                <select value={yCol} onChange={e => setYCol(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  <option value="">Select Y column</option>
                  {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Predictors X — hold Ctrl to multi-select</label>
                <select multiple value={xCols} onChange={e => setXCols(Array.from(e.target.selectedOptions, o => o.value))}
                  className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)", height: 80 }}>
                  {numericCols.filter(c => c !== yCol).map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button onClick={runMLR} disabled={loading || !data.length || !yCol || !xCols.length}
                className="btn-primary disabled:opacity-40"
                style={{ background: "linear-gradient(135deg, #8b5cf6, #7c3aed)" }}>
                {loading ? <RefreshCw size={14} className="animate-spin" /> : <GitBranch size={14} />}
                {loading ? "Fitting..." : "Fit MLR Model"}
              </button>
              {result && (
                <button onClick={exportResults} className="btn-secondary text-xs flex items-center gap-1.5">
                  <Download size={12} /> Export
                </button>
              )}
            </div>
            {error && <div className="flex items-center gap-2 mt-3 text-sm" style={{ color: "#f87171" }}><AlertCircle size={14} /> {error}</div>}
          </div>

          {result && (
            <>
              {/* Summary stats */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {[
                  { label: "R²", value: result.r2.toFixed(4), color: result.r2 > 0.9 ? "#10b981" : result.r2 > 0.7 ? "#f59e0b" : "#ef4444" },
                  { label: "R² adj", value: result.r2adj.toFixed(4), color: "#8b5cf6" },
                  { label: "RMSE", value: result.rmse.toFixed(4), color: "#1E90FF" },
                  { label: "F-stat", value: result.fStat.toFixed(2), color: result.fStat > 4 ? "#10b981" : "#f59e0b" },
                  { label: "Samples", value: result.n, color: "#06b6d4" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="glass-card p-4">
                    <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
                    <div className="text-xl font-bold" style={{ color }}>{value}</div>
                  </div>
                ))}
              </div>

              {/* Tab bar */}
              <div className="flex flex-wrap gap-2">
                {TABS.map(t => (
                  <button key={t.id} onClick={() => setTab(t.id)}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                    style={{ background: tab === t.id ? "#8b5cf6" : "var(--bg-card)", color: tab === t.id ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                    {t.label}
                  </button>
                ))}
              </div>

              {/* Coefficients */}
              {tab === "model" && (
                <div className="glass-card overflow-hidden">
                  <div className="p-4" style={{ borderBottom: "1px solid var(--border)" }}>
                    <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Model Coefficients</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>{["Term", "Coefficient", "Std Error", "t-value", "P-value", "Sig"].map(h => (
                          <th key={h} className="px-4 py-3 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                        ))}</tr>
                      </thead>
                      <tbody>
                        {result.coefRows.map((row, i) => {
                          const sig = row.p < 0.001 ? "***" : row.p < 0.01 ? "**" : row.p < 0.05 ? "*" : row.p < 0.1 ? "." : "";
                          return (
                            <tr key={i} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                              <td className="px-4 py-2.5 font-medium" style={{ color: "var(--text-primary)" }}>{row.name}</td>
                              <td className="px-4 py-2.5" style={{ color: row.coef >= 0 ? "#10b981" : "#ef4444" }}>{row.coef.toFixed(5)}</td>
                              <td className="px-4 py-2.5" style={{ color: "var(--text-secondary)" }}>{row.se.toFixed(5)}</td>
                              <td className="px-4 py-2.5" style={{ color: Math.abs(row.t) > 2 ? "#f59e0b" : "var(--text-secondary)" }}>{row.t.toFixed(3)}</td>
                              <td className="px-4 py-2.5" style={{ color: row.p < 0.05 ? "#10b981" : "var(--text-muted)" }}>
                                {row.p < 0.001 ? "<0.001" : row.p.toFixed(4)}
                              </td>
                              <td className="px-4 py-2.5 font-bold" style={{ color: "#f59e0b" }}>{sig}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs p-3" style={{ color: "var(--text-muted)" }}>Significance: *** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · . p&lt;0.1</p>
                </div>
              )}

              {/* Pareto */}
              {tab === "pareto" && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>Pareto Chart — |t| values (largest = most important)</h3>
                  <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Orange dashed line = |t| = 2 (approx. 5% significance threshold)</p>
                  <ParetoChart coefRows={result.coefRows} />
                </div>
              )}

              {/* Predicted vs Actual */}
              {tab === "predicted" && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>Predicted vs Actual (R² = {result.r2.toFixed(4)})</h3>
                  <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Points on the 45° line = perfect prediction. Orange line = perfect fit.</p>
                  <ResponsiveContainer width="100%" height={380}>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 50, left: 55 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="actual" name="Actual"
                        label={{ value: `Actual ${result.yCol}`, position: "bottom", offset: 30, fill: "var(--text-muted)", fontSize: 12 }}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <YAxis type="number" dataKey="predicted" name="Predicted"
                        label={{ value: `Predicted ${result.yCol}`, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <Tooltip content={({ payload }) => {
                        if (!payload?.length) return null;
                        const d = payload[0].payload;
                        return (
                          <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                            <div className="font-semibold">Sample {d.sample}</div>
                            <div>Actual: {d.actual}</div>
                            <div>Predicted: {d.predicted}</div>
                            <div>Residual: {d.residual}</div>
                          </div>
                        );
                      }} />
                      <Scatter data={result.residuals} fill="#8b5cf6" opacity={0.8} name="Samples" />
                      {/* 45° perfect fit line */}
                      {diagLine.length > 0 && (
                        <Scatter data={diagLine} fill="none" line={{ stroke: "#f59e0b", strokeWidth: 2, strokeDasharray: "6 3" }} lineType="fitting" name="Perfect fit" shape={() => null} />
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Residuals */}
              {tab === "residuals" && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Residuals vs Sample Order</h3>
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={result.residuals} margin={{ top: 5, right: 20, bottom: 25, left: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                        <ReferenceLine y={0} stroke="var(--border-light)" strokeDasharray="4 4" />
                        <Line type="monotone" dataKey="residual" stroke="#8b5cf6" strokeWidth={1.5} dot={{ r: 3, fill: "#8b5cf6" }} name="Residual" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Residuals vs Fitted</h3>
                    <ResponsiveContainer width="100%" height={260}>
                      <ScatterChart margin={{ top: 5, right: 20, bottom: 25, left: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis type="number" dataKey="predicted" name="Fitted"
                          label={{ value: "Fitted values", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }}
                          tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <YAxis type="number" dataKey="residual" name="Residual"
                          label={{ value: "Residuals", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 11 }}
                          tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip content={({ payload }) => {
                          if (!payload?.length) return null;
                          const d = payload[0].payload;
                          return (
                            <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                              <div>Fitted: {d.predicted}</div>
                              <div>Residual: {d.residual}</div>
                            </div>
                          );
                        }} />
                        <ReferenceLine y={0} stroke="#f59e0b" strokeDasharray="4 4" />
                        <Scatter data={result.residuals} fill="#8b5cf6" opacity={0.8} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* ANOVA */}
              {tab === "anova" && (
                <div className="glass-card overflow-hidden">
                  <div className="p-4" style={{ borderBottom: "1px solid var(--border)" }}>
                    <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>ANOVA Table</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>{["Source", "SS", "df", "MS", "F", ""].map(h => (
                          <th key={h} className="px-4 py-3 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                        ))}</tr>
                      </thead>
                      <tbody>
                        {[
                          { src: "Regression", ss: result.ssMod, df: result.p - 1, ms: result.ssMod / (result.p - 1), f: result.fStat, highlight: true },
                          { src: "Residual", ss: result.ssRes, df: result.n - result.p, ms: result.mse, f: null },
                          { src: "Total", ss: result.ssTot, df: result.n - 1, ms: null, f: null },
                        ].map(row => (
                          <tr key={row.src} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                            <td className="px-4 py-3 font-semibold" style={{ color: row.highlight ? "#8b5cf6" : "var(--text-primary)" }}>{row.src}</td>
                            <td className="px-4 py-3" style={{ color: "var(--text-primary)" }}>{row.ss.toFixed(4)}</td>
                            <td className="px-4 py-3" style={{ color: "var(--text-muted)" }}>{row.df}</td>
                            <td className="px-4 py-3" style={{ color: "var(--text-secondary)" }}>{row.ms ? row.ms.toFixed(4) : "—"}</td>
                            <td className="px-4 py-3 font-bold" style={{ color: row.f && row.f > 4 ? "#10b981" : "var(--text-secondary)" }}>
                              {row.f ? row.f.toFixed(3) : "—"}
                            </td>
                            <td className="px-4 py-3" style={{ color: "#10b981" }}>
                              {row.highlight && result.fStat > 4 ? "Significant" : ""}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="p-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      { label: "R²", value: (result.r2 * 100).toFixed(2) + "%" },
                      { label: "R² adj", value: (result.r2adj * 100).toFixed(2) + "%" },
                      { label: "RMSE", value: result.rmse.toFixed(4) },
                      { label: "F-statistic", value: result.fStat.toFixed(3) },
                    ].map(({ label, value }) => (
                      <div key={label} className="text-xs" style={{ color: "var(--text-muted)" }}>
                        <span className="font-semibold" style={{ color: "var(--text-primary)" }}>{label}: </span>{value}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}