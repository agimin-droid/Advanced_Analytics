import { useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, LineChart, Line, BarChart, Bar, Legend, ComposedChart
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { GitBranch, Download } from "lucide-react";

// ── MATH UTILS ───────────────────────────────────────────────────────────────
function pearson(x, y) {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  const num = x.reduce((s, v, i) => s + (v - mx) * (y[i] - my), 0);
  const dx = Math.sqrt(x.reduce((s, v) => s + (v - mx) ** 2, 0));
  const dy = Math.sqrt(y.reduce((s, v) => s + (v - my) ** 2, 0));
  return dx * dy === 0 ? 0 : num / (dx * dy);
}

function linearFit(x, y) {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  const denom = x.reduce((s, v) => s + (v - mx) ** 2, 0);
  const b = denom === 0 ? 0 : x.reduce((s, v, i) => s + (v - mx) * (y[i] - my), 0) / denom;
  const a = my - b * mx;
  return { a, b };
}

function pValueApprox(r, n) {
  if (n < 3) return 1;
  const t = r * Math.sqrt((n - 2) / (1 - r * r));
  // Simple approximation for two-tailed p-value
  const x = Math.abs(t);
  const pApprox = Math.exp(-0.717 * x - 0.416 * x * x);
  return Math.min(1, 2 * pApprox);
}

function rStrengthLabel(absR) {
  return absR > 0.8 ? "Very Strong" : absR > 0.6 ? "Strong" : absR > 0.4 ? "Moderate" : "Weak";
}
function rColor(absR) {
  return absR > 0.8 ? "#ef4444" : absR > 0.6 ? "#f97316" : absR > 0.4 ? "#f59e0b" : "#10b981";
}

// ── CORRELATION HEATMAP ──────────────────────────────────────────────────────
function CorrHeatmap({ cols, matrix }) {
  const n = cols.length;
  const cellSize = Math.min(60, Math.floor(640 / n));
  const fontSize = Math.max(8, Math.min(11, cellSize - 10));

  return (
    <div className="overflow-auto">
      <div style={{ display: "inline-block", minWidth: n * cellSize + 80 }}>
        {/* Column labels top */}
        <div style={{ display: "flex", marginLeft: 80 }}>
          {cols.map(c => (
            <div key={c} style={{ width: cellSize, fontSize: fontSize - 1, color: "var(--text-muted)", textAlign: "center", overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" }} title={c}>
              {c.slice(0, Math.max(3, Math.floor(cellSize / 7)))}
            </div>
          ))}
        </div>
        {matrix.map((row, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ width: 78, fontSize: fontSize - 1, color: "var(--text-muted)", textAlign: "right", paddingRight: 4, overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" }} title={cols[i]}>
              {cols[i].slice(0, 10)}
            </div>
            {row.map((val, j) => {
              const absV = Math.abs(val);
              let bg;
              if (val > 0) bg = `rgba(30,144,255,${0.1 + absV * 0.85})`;
              else if (val < 0) bg = `rgba(239,68,68,${0.1 + absV * 0.85})`;
              else bg = "rgba(30,45,74,0.3)";
              return (
                <div key={j}
                  style={{ width: cellSize, height: cellSize, background: bg, display: "flex", alignItems: "center", justifyContent: "center", fontSize: Math.max(8, fontSize - 2), color: absV > 0.5 ? "white" : "var(--text-secondary)", border: "1px solid rgba(30,45,74,0.3)", cursor: "pointer" }}
                  title={`${cols[i]} × ${cols[j]}: r=${val.toFixed(4)}`}>
                  {cellSize >= 40 ? val.toFixed(2) : ""}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center gap-3 text-xs" style={{ color: "var(--text-muted)" }}>
        <div className="flex items-center gap-1">
          <div style={{ width: 14, height: 14, background: "rgba(30,144,255,0.8)", borderRadius: 2 }} />
          Positive
        </div>
        <div className="flex items-center gap-1">
          <div style={{ width: 14, height: 14, background: "rgba(239,68,68,0.8)", borderRadius: 2 }} />
          Negative
        </div>
        <div className="flex items-center gap-1">
          <div style={{ width: 14, height: 14, background: "rgba(30,45,74,0.3)", borderRadius: 2 }} />
          Zero
        </div>
      </div>
    </div>
  );
}

// ── MAIN ─────────────────────────────────────────────────────────────────────
export default function Bivariate() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [col1, setCol1] = useState("");
  const [col2, setCol2] = useState("");
  const [corrMatrix, setCorrMatrix] = useState(null);
  const [tab, setTab] = useState("scatter");
  const [maxPairs, setMaxPairs] = useState(15);

  const loadData = async (id, ds) => {
    setDataset(ds); setCorrMatrix(null);
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const nc = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    if (nc.length >= 2) { setCol1(nc[0]); setCol2(nc[1]); }
    // Compute full correlation matrix (up to 20 cols for performance)
    const topCols = nc.slice(0, 20);
    const matrix = topCols.map(c1 => topCols.map(c2 => {
      const x = rows.map(r => parseFloat(r[c1])).filter(v => !isNaN(v));
      const y = rows.map(r => parseFloat(r[c2])).filter(v => !isNaN(v));
      const minLen = Math.min(x.length, y.length);
      return minLen >= 2 ? parseFloat(pearson(x.slice(0, minLen), y.slice(0, minLen)).toFixed(4)) : 0;
    }));
    const pairs = [];
    for (let i = 0; i < topCols.length; i++) {
      for (let j = i + 1; j < topCols.length; j++) {
        const x = rows.map(r => parseFloat(r[topCols[i]])).filter(v => !isNaN(v));
        const y = rows.map(r => parseFloat(r[topCols[j]])).filter(v => !isNaN(v));
        const n = Math.min(x.length, y.length);
        const r = matrix[i][j];
        pairs.push({ var1: topCols[i], var2: topCols[j], r, absR: Math.abs(r), n, pval: pValueApprox(r, n) });
      }
    }
    pairs.sort((a, b) => b.absR - a.absR);
    setCorrMatrix({ matrix, cols: topCols, pairs });
  };

  const numericCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];
  const xVals = col1 ? data.map(r => parseFloat(r[col1])).filter(v => !isNaN(v)) : [];
  const yVals = col2 ? data.map(r => parseFloat(r[col2])).filter(v => !isNaN(v)) : [];
  const minLen = Math.min(xVals.length, yVals.length);
  const r = xVals.length && yVals.length ? pearson(xVals.slice(0, minLen), yVals.slice(0, minLen)) : 0;
  const fit = xVals.length && yVals.length ? linearFit(xVals.slice(0, minLen), yVals.slice(0, minLen)) : null;
  const scatterData = xVals.slice(0, minLen).map((x, i) => ({ x, y: yVals[i], sample: i + 1 }));
  const pval = minLen ? pValueApprox(r, minLen) : 1;
  const absR = Math.abs(r);
  const color = rColor(absR);

  // Regression line data
  const fitLineData = fit && scatterData.length ? [
    { x: Math.min(...xVals), y: fit.a + fit.b * Math.min(...xVals) },
    { x: Math.max(...xVals), y: fit.a + fit.b * Math.max(...xVals) }
  ] : [];

  const exportPairs = () => {
    if (!corrMatrix) return;
    const csv = ["Variable1,Variable2,PearsonR,AbsR,N,P-value", ...corrMatrix.pairs.map(p =>
      `${p.var1},${p.var2},${p.r.toFixed(4)},${p.absR.toFixed(4)},${p.n},${p.pval.toFixed(4)}`)].join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "correlations.csv"; a.click();
  };

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>🔗 Bivariate Analysis</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Correlation ranking, scatter plots, regression lines and correlation heatmap</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <GitBranch size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a dataset to start bivariate analysis</p>
        </div>
      ) : (
        <>
          {/* Tabs */}
          <div className="flex flex-wrap gap-2">
            {["scatter", "ranking", "heatmap"].map(t => (
              <button key={t} onClick={() => setTab(t)}
                className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all capitalize"
                style={{ background: tab === t ? "#ec4899" : "var(--bg-card)", color: tab === t ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                {t === "scatter" ? "Scatter Plot" : t === "ranking" ? "Correlation Ranking" : "Heatmap"}
              </button>
            ))}
          </div>

          {/* Scatter tab */}
          {tab === "scatter" && (
            <>
              <div className="glass-card p-4 flex flex-wrap gap-4 items-center">
                <div className="flex items-center gap-2">
                  <label className="text-sm" style={{ color: "var(--text-muted)" }}>X:</label>
                  <select value={col1} onChange={e => setCol1(e.target.value)}
                    className="px-3 py-2 rounded-lg text-sm outline-none"
                    style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                    {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm" style={{ color: "var(--text-muted)" }}>Y:</label>
                  <select value={col2} onChange={e => setCol2(e.target.value)}
                    className="px-3 py-2 rounded-lg text-sm outline-none"
                    style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                    {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                {r !== 0 && (
                  <div className="flex items-center gap-3 ml-auto">
                    <span className="text-sm font-bold" style={{ color }}>r = {r.toFixed(4)}</span>
                    <span className="tag text-xs" style={{ background: `${color}18`, color }}>{rStrengthLabel(absR)}</span>
                    <span className="text-xs" style={{ color: pval < 0.05 ? "#10b981" : "#f59e0b" }}>p ≈ {pval < 0.001 ? "<0.001" : pval.toFixed(3)}</span>
                  </div>
                )}
              </div>

              {scatterData.length > 0 && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
                    {col1} vs {col2}
                  </h3>
                  <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
                    Pearson r = {r.toFixed(4)} · p ≈ {pval < 0.001 ? "<0.001" : pval.toFixed(3)} · n = {minLen}
                    {fit && <> · Regression: y = {fit.b.toFixed(4)}x {fit.a >= 0 ? "+" : ""} {fit.a.toFixed(4)}</>}
                  </p>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 50, left: 55 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="x" name={col1}
                        label={{ value: col1, position: "bottom", offset: 30, fill: "var(--text-muted)", fontSize: 12 }}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <YAxis type="number" dataKey="y" name={col2}
                        label={{ value: col2, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <Tooltip content={({ payload }) => {
                        if (!payload?.length) return null;
                        const d = payload[0].payload;
                        return (
                          <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                            <div className="font-semibold">Sample {d.sample}</div>
                            <div>{col1}: {d.x?.toFixed(4)}</div>
                            <div>{col2}: {d.y?.toFixed(4)}</div>
                          </div>
                        );
                      }} />
                      <Scatter data={scatterData} fill="#ec4899" opacity={0.7} name="Samples" />
                      {/* Regression line */}
                      {fitLineData.length > 0 && (
                        <Scatter data={fitLineData} fill="none" line={{ stroke: "#f59e0b", strokeWidth: 2 }} lineType="fitting" name="Regression" shape={() => null} />
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}

          {/* Ranking tab */}
          {tab === "ranking" && corrMatrix?.pairs && (
            <div className="glass-card overflow-hidden">
              <div className="p-4 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
                <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  Correlation Ranking — {corrMatrix.pairs.length} pairs (sorted by |r|)
                </h3>
                <div className="flex items-center gap-3">
                  <select value={maxPairs} onChange={e => setMaxPairs(+e.target.value)}
                    className="px-2 py-1 rounded-lg text-xs outline-none"
                    style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                    {[10, 20, 50, 100].map(n => <option key={n} value={n}>Top {n}</option>)}
                  </select>
                  <button onClick={exportPairs} className="btn-secondary text-xs flex items-center gap-1.5">
                    <Download size={12} /> Export CSV
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                  <thead>
                    <tr>{["#", "Variable 1", "Variable 2", "Pearson r", "|r|", "P-value", "N", "Strength", ""].map(h => (
                      <th key={h} className="px-3 py-2.5 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                    ))}</tr>
                  </thead>
                  <tbody>
                    {corrMatrix.pairs.slice(0, maxPairs).map((p, i) => {
                      const c = rColor(p.absR);
                      const s = rStrengthLabel(p.absR);
                      return (
                        <tr key={i} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                          <td className="px-3 py-2" style={{ color: "var(--text-muted)" }}>{i + 1}</td>
                          <td className="px-3 py-2 font-medium" style={{ color: "var(--text-primary)" }}>{p.var1}</td>
                          <td className="px-3 py-2 font-medium" style={{ color: "var(--text-primary)" }}>{p.var2}</td>
                          <td className="px-3 py-2 font-bold" style={{ color: p.r < 0 ? "#ef4444" : "#10b981" }}>{p.r.toFixed(4)}</td>
                          <td className="px-3 py-2">{p.absR.toFixed(4)}</td>
                          <td className="px-3 py-2" style={{ color: p.pval < 0.05 ? "#10b981" : "#f59e0b" }}>
                            {p.pval < 0.001 ? "<0.001" : p.pval.toFixed(3)}
                          </td>
                          <td className="px-3 py-2" style={{ color: "var(--text-muted)" }}>{p.n}</td>
                          <td className="px-3 py-2"><span className="tag text-xs" style={{ background: `${c}18`, color: c }}>{s}</span></td>
                          <td className="px-3 py-2">
                            <button onClick={() => { setCol1(p.var1); setCol2(p.var2); setTab("scatter"); }}
                              className="text-xs px-2 py-1 rounded-lg"
                              style={{ background: "rgba(236,72,153,0.1)", color: "#ec4899", border: "1px solid rgba(236,72,153,0.2)" }}>
                              Plot →
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Heatmap tab */}
          {tab === "heatmap" && corrMatrix && (
            <div className="glass-card p-5">
              <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>Correlation Heatmap — {corrMatrix.cols.length} variables</h3>
              <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Blue = positive correlation · Red = negative correlation · Color intensity = |r| magnitude</p>
              <CorrHeatmap cols={corrMatrix.cols} matrix={corrMatrix.matrix} />
            </div>
          )}
        </>
      )}
    </div>
  );
}