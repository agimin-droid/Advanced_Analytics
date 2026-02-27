import { useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { GitBranch } from "lucide-react";

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
  const b = x.reduce((s, v, i) => s + (v - mx) * (y[i] - my), 0) / x.reduce((s, v) => s + (v - mx) ** 2, 0);
  const a = my - b * mx;
  return { a, b };
}

export default function Bivariate() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [col1, setCol1] = useState("");
  const [col2, setCol2] = useState("");
  const [corrMatrix, setCorrMatrix] = useState([]);
  const [tab, setTab] = useState("scatter");

  const loadData = async (id, ds) => {
    setDataset(ds); setCorrMatrix([]);
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const nc = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    if (nc.length >= 2) { setCol1(nc[0]); setCol2(nc[1]); }
    // Compute correlation matrix for top columns (up to 10)
    const topCols = nc.slice(0, 10);
    const matrix = topCols.map(c1 => topCols.map(c2 => {
      const x = rows.map(r => parseFloat(r[c1])).filter(v => !isNaN(v));
      const y = rows.map(r => parseFloat(r[c2])).filter(v => !isNaN(v));
      const minLen = Math.min(x.length, y.length);
      return parseFloat(pearson(x.slice(0, minLen), y.slice(0, minLen)).toFixed(4));
    }));
    const pairs = [];
    for (let i = 0; i < topCols.length; i++) for (let j = i + 1; j < topCols.length; j++) {
      pairs.push({ var1: topCols[i], var2: topCols[j], r: matrix[i][j], absR: Math.abs(matrix[i][j]) });
    }
    pairs.sort((a, b) => b.absR - a.absR);
    setCorrMatrix({ matrix, cols: topCols, pairs: pairs.slice(0, 20) });
  };

  const numericCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];
  const xVals = col1 ? data.map(r => parseFloat(r[col1])).filter(v => !isNaN(v)) : [];
  const yVals = col2 ? data.map(r => parseFloat(r[col2])).filter(v => !isNaN(v)) : [];
  const minLen = Math.min(xVals.length, yVals.length);
  const r = xVals.length && yVals.length ? pearson(xVals.slice(0, minLen), yVals.slice(0, minLen)) : 0;
  const fit = xVals.length && yVals.length ? linearFit(xVals.slice(0, minLen), yVals.slice(0, minLen)) : null;
  const scatterData = xVals.slice(0, minLen).map((x, i) => ({ x, y: yVals[i], sample: i + 1 }));
  const rStrength = Math.abs(r) > 0.8 ? "Very Strong" : Math.abs(r) > 0.6 ? "Strong" : Math.abs(r) > 0.4 ? "Moderate" : "Weak";
  const rColor = Math.abs(r) > 0.8 ? "#ef4444" : Math.abs(r) > 0.6 ? "#f97316" : Math.abs(r) > 0.4 ? "#f59e0b" : "#10b981";

  const fitLineData = fit && scatterData.length ? [
    { x: Math.min(...xVals), y: fit.a + fit.b * Math.min(...xVals) },
    { x: Math.max(...xVals), y: fit.a + fit.b * Math.max(...xVals) }
  ] : [];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>🔗 Bivariate Analysis</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Correlation analysis and scatter plots between variable pairs</p>
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
          <div className="flex gap-2">
            {["scatter", "ranking"].map(t => (
              <button key={t} onClick={() => setTab(t)}
                className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all capitalize"
                style={{ background: tab === t ? "#ec4899" : "var(--bg-card)", color: tab === t ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                {t === "scatter" ? "Scatter Plot" : "Correlation Ranking"}
              </button>
            ))}
          </div>

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
                  <div className="flex items-center gap-2 ml-auto">
                    <span className="text-sm font-bold" style={{ color: rColor }}>r = {r.toFixed(4)}</span>
                    <span className="tag text-xs" style={{ background: `${rColor}18`, color: rColor }}>{rStrength}</span>
                  </div>
                )}
              </div>

              {scatterData.length > 0 && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
                    {col1} vs {col2} — Pearson r = {r.toFixed(4)}
                  </h3>
                  <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
                    Regression: y = {fit?.b.toFixed(4)}x {fit?.a >= 0 ? "+" : ""} {fit?.a.toFixed(4)} · {rStrength} {r > 0 ? "positive" : r < 0 ? "negative" : ""} correlation
                  </p>
                  <ResponsiveContainer width="100%" height={380}>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 50 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="x" name={col1}
                        label={{ value: col1, position: "bottom", fill: "var(--text-muted)", fontSize: 12 }}
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
                      <Scatter data={scatterData} fill="#ec4899" opacity={0.7} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}

          {tab === "ranking" && corrMatrix.pairs && (
            <div className="glass-card overflow-hidden">
              <div className="p-4" style={{ borderBottom: "1px solid var(--border)" }}>
                <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  Top {corrMatrix.pairs.length} Correlations — sorted by |r|
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                  <thead>
                    <tr>{["#", "Variable 1", "Variable 2", "Pearson r", "|r|", "Strength", ""].map(h => (
                      <th key={h} className="px-4 py-3 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                    ))}</tr>
                  </thead>
                  <tbody>
                    {corrMatrix.pairs.map((p, i) => {
                      const c = Math.abs(p.r) > 0.8 ? "#ef4444" : Math.abs(p.r) > 0.6 ? "#f97316" : Math.abs(p.r) > 0.4 ? "#f59e0b" : "#10b981";
                      const s = Math.abs(p.r) > 0.8 ? "Very Strong" : Math.abs(p.r) > 0.6 ? "Strong" : Math.abs(p.r) > 0.4 ? "Moderate" : "Weak";
                      return (
                        <tr key={i} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                          <td className="px-4 py-2.5" style={{ color: "var(--text-muted)" }}>{i + 1}</td>
                          <td className="px-4 py-2.5 font-medium" style={{ color: "var(--text-primary)" }}>{p.var1}</td>
                          <td className="px-4 py-2.5 font-medium" style={{ color: "var(--text-primary)" }}>{p.var2}</td>
                          <td className="px-4 py-2.5 font-bold" style={{ color: p.r < 0 ? "#ef4444" : "#10b981" }}>{p.r.toFixed(4)}</td>
                          <td className="px-4 py-2.5">{p.absR.toFixed(4)}</td>
                          <td className="px-4 py-2.5"><span className="tag text-xs" style={{ background: `${c}18`, color: c }}>{s}</span></td>
                          <td className="px-4 py-2.5">
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
        </>
      )}
    </div>
  );
}