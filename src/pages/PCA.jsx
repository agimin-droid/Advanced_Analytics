import { useState, useEffect } from "react";
import { base44 } from "@/api/base44Client";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, ReferenceLine, LineChart, Line, Legend
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { TrendingUp, RefreshCw, AlertCircle } from "lucide-react";

function computePCA(matrix, nComponents) {
  const n = matrix.length;
  const p = matrix[0].length;
  const nc = Math.min(nComponents, n - 1, p);

  // Center and scale
  const means = Array(p).fill(0);
  const stds = Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) means[j] += matrix[i][j];
    means[j] /= n;
    for (let i = 0; i < n; i++) stds[j] += (matrix[i][j] - means[j]) ** 2;
    stds[j] = Math.sqrt(stds[j] / (n - 1)) || 1;
  }
  const X = matrix.map(row => row.map((v, j) => (v - means[j]) / stds[j]));

  // NIPALS PCA
  const scores = [];
  const loadings = [];
  const varExplained = [];
  let Xres = X.map(r => [...r]);
  const totalVar = X.reduce((s, r) => s + r.reduce((a, v) => a + v * v, 0), 0);

  for (let k = 0; k < nc; k++) {
    let t = Xres.map(r => r[0]);
    for (let iter = 0; iter < 100; iter++) {
      let p_vec = Array(p).fill(0);
      const tt = t.reduce((s, v) => s + v * v, 0);
      if (tt < 1e-10) break;
      for (let j = 0; j < p; j++) for (let i = 0; i < n; i++) p_vec[j] += t[i] * Xres[i][j];
      const pnorm = Math.sqrt(p_vec.reduce((s, v) => s + v * v, 0)) || 1;
      p_vec = p_vec.map(v => v / pnorm);
      const t_new = Xres.map(r => r.reduce((s, v, j) => s + v * p_vec[j], 0));
      const diff = t_new.reduce((s, v, i) => s + (v - t[i]) ** 2, 0);
      t = t_new;
      if (diff < 1e-10) break;
    }
    let p_vec = Array(p).fill(0);
    const tt = t.reduce((s, v) => s + v * v, 0);
    for (let j = 0; j < p; j++) for (let i = 0; i < n; i++) p_vec[j] += t[i] * Xres[i][j];
    const pnorm = Math.sqrt(p_vec.reduce((s, v) => s + v * v, 0)) || 1;
    p_vec = p_vec.map(v => v / pnorm);
    Xres = Xres.map((r, i) => r.map((v, j) => v - t[i] * p_vec[j]));
    const sv = tt;
    varExplained.push(sv / totalVar * 100);
    scores.push(t);
    loadings.push(p_vec);
  }
  return { scores, loadings, varExplained, means, stds };
}

const COLORS = ["#1E90FF", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"];

export default function PCA() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [nComponents, setNComponents] = useState(3);
  const [pc1, setPc1] = useState(1);
  const [pc2, setPc2] = useState(2);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState("scores");
  const [colorCol, setColorCol] = useState("");
  const [error, setError] = useState("");

  const loadData = async (id, ds) => {
    setDataset(ds);
    setResult(null);
    setError("");
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
  };

  const runPCA = () => {
    if (!data.length) return;
    setLoading(true);
    setError("");
    setTimeout(() => {
      const numCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];
      if (numCols.length < 2) { setError("Need at least 2 numeric columns."); setLoading(false); return; }
      const matrix = data.map(r => numCols.map(c => parseFloat(r[c]) || 0));
      const nc = Math.min(nComponents, numCols.length, data.length - 1);
      const res = computePCA(matrix, nc);
      setResult({ ...res, columns: numCols, nSamples: data.length });
      setLoading(false);
    }, 50);
  };

  const scoresData = result
    ? result.scores[pc1 - 1]?.map((v, i) => ({
        x: v,
        y: result.scores[pc2 - 1]?.[i] ?? 0,
        name: `Sample ${i + 1}`,
        color: colorCol && data[i]?.[colorCol]
      }))
    : [];

  const loadingsData = result
    ? result.loadings[pc1 - 1]?.map((v, j) => ({
        name: result.columns[j],
        pc1: v,
        pc2: result.loadings[pc2 - 1]?.[j] ?? 0,
      }))
    : [];

  const screeData = result
    ? result.varExplained.map((v, i) => ({
        pc: `PC${i + 1}`,
        variance: parseFloat(v.toFixed(2)),
        cumulative: parseFloat(result.varExplained.slice(0, i + 1).reduce((a, b) => a + b, 0).toFixed(2))
      }))
    : [];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>🎯 PCA Analysis</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Principal Component Analysis — NIPALS implementation</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <TrendingUp size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a dataset to begin PCA analysis</p>
        </div>
      ) : (
        <>
          {/* Controls */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>⚙️ Model Configuration</h3>
            <div className="flex flex-wrap gap-4 items-end">
              <div>
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Components</label>
                <select value={nComponents} onChange={e => setNComponents(+e.target.value)}
                  className="px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  {[2,3,4,5,6,7,8,10].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Color by</label>
                <select value={colorCol} onChange={e => setColorCol(e.target.value)}
                  className="px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  <option value="">None</option>
                  {dataset.columns?.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <button onClick={runPCA} disabled={loading || !data.length}
                className="btn-primary disabled:opacity-40"
                style={{ background: "linear-gradient(135deg, #1E90FF, #2E5293)" }}>
                {loading ? <RefreshCw size={14} className="animate-spin" /> : <TrendingUp size={14} />}
                {loading ? "Computing..." : "Run PCA"}
              </button>
            </div>
            {error && (
              <div className="flex items-center gap-2 mt-3 text-sm" style={{ color: "#f87171" }}>
                <AlertCircle size={14} /> {error}
              </div>
            )}
          </div>

          {result && (
            <>
              {/* Variance summary */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {result.varExplained.slice(0, 4).map((v, i) => (
                  <div key={i} className="glass-card p-4">
                    <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>PC{i+1}</div>
                    <div className="text-xl font-bold" style={{ color: "#1E90FF" }}>{v.toFixed(1)}%</div>
                    <div className="text-xs" style={{ color: "var(--text-muted)" }}>variance</div>
                  </div>
                ))}
              </div>

              {/* PC axis selectors */}
              <div className="flex gap-3 items-center">
                <span className="text-sm" style={{ color: "var(--text-muted)" }}>Plot:</span>
                {["Scores", "Loadings", "Biplot", "Scree"].map(t => (
                  <button key={t} onClick={() => setTab(t.toLowerCase())}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                    style={{ background: tab === t.toLowerCase() ? "#1E90FF" : "var(--bg-card)", color: tab === t.toLowerCase() ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                    {t}
                  </button>
                ))}
                {tab !== "scree" && (
                  <div className="flex items-center gap-2 ml-2">
                    <select value={pc1} onChange={e => setPc1(+e.target.value)}
                      className="px-2 py-1.5 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {result.varExplained.map((_, i) => <option key={i+1} value={i+1}>PC{i+1}</option>)}
                    </select>
                    <span style={{ color: "var(--text-muted)" }}>vs</span>
                    <select value={pc2} onChange={e => setPc2(+e.target.value)}
                      className="px-2 py-1.5 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {result.varExplained.map((_, i) => <option key={i+1} value={i+1}>PC{i+1}</option>)}
                    </select>
                  </div>
                )}
              </div>

              {/* Charts */}
              <div className="glass-card p-5">
                {(tab === "scores" || tab === "biplot") && (
                  <>
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
                      {tab === "scores" ? "Score Plot" : "Biplot"} — PC{pc1} vs PC{pc2}
                    </h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis type="number" dataKey="x" name={`PC${pc1}`}
                          label={{ value: `PC${pc1} (${result.varExplained[pc1-1]?.toFixed(1)}%)`, position: "bottom", fill: "var(--text-muted)", fontSize: 12 }}
                          tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <YAxis type="number" dataKey="y" name={`PC${pc2}`}
                          label={{ value: `PC${pc2} (${result.varExplained[pc2-1]?.toFixed(1)}%)`, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
                          tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <Tooltip
                          content={({ payload }) => {
                            if (!payload?.length) return null;
                            const d = payload[0].payload;
                            return (
                              <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                                <div className="font-semibold">{d.name}</div>
                                <div>PC{pc1}: {d.x?.toFixed(3)}</div>
                                <div>PC{pc2}: {d.y?.toFixed(3)}</div>
                                {d.color && <div>Group: {d.color}</div>}
                              </div>
                            );
                          }}
                        />
                        <ReferenceLine x={0} stroke="var(--border-light)" />
                        <ReferenceLine y={0} stroke="var(--border-light)" />
                        <Scatter data={scoresData} fill="#1E90FF" opacity={0.8} />
                        {tab === "biplot" && loadingsData.map((l, i) => (
                          <ReferenceLine key={i} segment={[{ x: 0, y: 0 }, { x: l.pc1 * 3, y: l.pc2 * 3 }]} stroke="#f59e0b" strokeWidth={1.5} />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </>
                )}

                {tab === "loadings" && (
                  <>
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Loadings — PC{pc1} vs PC{pc2}</h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis type="number" dataKey="pc1"
                          label={{ value: `PC${pc1} Loadings`, position: "bottom", fill: "var(--text-muted)", fontSize: 12 }}
                          tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <YAxis type="number" dataKey="pc2"
                          label={{ value: `PC${pc2} Loadings`, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
                          tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <Tooltip content={({ payload }) => {
                          if (!payload?.length) return null;
                          const d = payload[0].payload;
                          return (
                            <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                              <div className="font-semibold">{d.name}</div>
                              <div>PC{pc1}: {d.pc1?.toFixed(4)}</div>
                              <div>PC{pc2}: {d.pc2?.toFixed(4)}</div>
                            </div>
                          );
                        }} />
                        <ReferenceLine x={0} stroke="var(--border-light)" />
                        <ReferenceLine y={0} stroke="var(--border-light)" />
                        <Scatter data={loadingsData} fill="#f59e0b" opacity={0.9} />
                      </ScatterChart>
                    </ResponsiveContainer>
                    <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {loadingsData.slice(0, 8).map((l, i) => (
                        <div key={i} className="text-xs p-2 rounded-lg" style={{ background: "var(--bg-secondary)" }}>
                          <div className="font-medium truncate" style={{ color: "var(--text-primary)" }}>{l.name}</div>
                          <div style={{ color: "var(--text-muted)" }}>{l.pc1.toFixed(3)} / {l.pc2.toFixed(3)}</div>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {tab === "scree" && (
                  <>
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Scree Plot — Variance Explained</h3>
                    <ResponsiveContainer width="100%" height={350}>
                      <BarChart data={screeData} margin={{ top: 10, right: 30, bottom: 30, left: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="pc" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)" }} />
                        <Legend wrapperStyle={{ color: "var(--text-secondary)", fontSize: 12 }} />
                        <Bar dataKey="variance" name="Variance %" fill="#1E90FF" radius={[4, 4, 0, 0]} />
                        <Line type="monotone" data={screeData} dataKey="cumulative" name="Cumulative %" stroke="#f59e0b" strokeWidth={2} dot={{ fill: "#f59e0b" }} />
                      </BarChart>
                    </ResponsiveContainer>
                  </>
                )}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}