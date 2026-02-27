import { useState, useEffect } from "react";
import { base44 } from "@/api/base44Client";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, ReferenceLine, LineChart, Line, Legend, ComposedChart, ErrorBar
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { TrendingUp, RefreshCw, AlertCircle, Download } from "lucide-react";

// ── NIPALS PCA ──────────────────────────────────────────────────────────────
function computePCA(matrix, nComponents) {
  const n = matrix.length;
  const p = matrix[0].length;
  const nc = Math.min(nComponents, n - 1, p);

  const means = Array(p).fill(0);
  const stds = Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) means[j] += matrix[i][j];
    means[j] /= n;
    for (let i = 0; i < n; i++) stds[j] += (matrix[i][j] - means[j]) ** 2;
    stds[j] = Math.sqrt(stds[j] / (n - 1)) || 1;
  }
  const X = matrix.map(row => row.map((v, j) => (v - means[j]) / stds[j]));

  const scores = [];
  const loadings = [];
  const varExplained = [];
  let Xres = X.map(r => [...r]);
  const totalVar = X.reduce((s, r) => s + r.reduce((a, v) => a + v * v, 0), 0);

  for (let k = 0; k < nc; k++) {
    let t = Xres.map(r => r[0]);
    for (let iter = 0; iter < 200; iter++) {
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
    varExplained.push(tt / totalVar * 100);
    scores.push(t);
    loadings.push(p_vec);
  }

  // Hotelling T² and Q residuals
  const t2 = scores[0].map((_, i) => {
    return scores.slice(0, nc).reduce((s, sc, k) => {
      const varPc = sc.reduce((a, v) => a + v * v, 0) / (n - 1);
      return s + (sc[i] ** 2) / (varPc || 1);
    }, 0);
  });
  const qResiduals = Xres.map(r => r.reduce((s, v) => s + v * v, 0));
  const qMean = qResiduals.reduce((a, b) => a + b, 0) / n;
  const qVar = qResiduals.reduce((a, v) => a + (v - qMean) ** 2, 0) / (n - 1);
  const t2Limit = nc * (n * n - 1) / (n * (n - nc)) * 3.0;
  const qLimit = qMean + 3 * Math.sqrt(qVar);

  return { scores, loadings, varExplained, means, stds, t2, qResiduals, t2Limit, qLimit };
}

const COLORS = ["#1E90FF", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"];

// ── CHARTS ──────────────────────────────────────────────────────────────────
function ScorePlot({ result, pc1, pc2, data, colorCol }) {
  const groups = {};
  result.scores[pc1 - 1]?.forEach((v, i) => {
    const gKey = colorCol && data[i]?.[colorCol] ? String(data[i][colorCol]) : "samples";
    if (!groups[gKey]) groups[gKey] = [];
    groups[gKey].push({ x: v, y: result.scores[pc2 - 1]?.[i] ?? 0, name: `S${i + 1}` });
  });
  const gKeys = Object.keys(groups);
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 30, bottom: 50, left: 50 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" dataKey="x" name={`PC${pc1}`}
          label={{ value: `PC${pc1} (${result.varExplained[pc1 - 1]?.toFixed(1)}%)`, position: "bottom", offset: 30, fill: "var(--text-muted)", fontSize: 12 }}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
        <YAxis type="number" dataKey="y" name={`PC${pc2}`}
          label={{ value: `PC${pc2} (${result.varExplained[pc2 - 1]?.toFixed(1)}%)`, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
        <Tooltip content={({ payload }) => {
          if (!payload?.length) return null;
          const d = payload[0].payload;
          return (
            <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
              <div className="font-semibold">{d.name}</div>
              <div>PC{pc1}: {d.x?.toFixed(3)}</div>
              <div>PC{pc2}: {d.y?.toFixed(3)}</div>
            </div>
          );
        }} />
        <ReferenceLine x={0} stroke="var(--border-light)" />
        <ReferenceLine y={0} stroke="var(--border-light)" />
        {gKeys.map((g, gi) => (
          <Scatter key={g} name={g} data={groups[g]} fill={COLORS[gi % COLORS.length]} opacity={0.8} />
        ))}
        {gKeys.length > 1 && <Legend wrapperStyle={{ color: "var(--text-secondary)", fontSize: 12 }} />}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function LoadingsBarChart({ result, pc }) {
  const data = result.loadings[pc - 1]?.map((v, j) => ({
    name: result.columns[j],
    loading: parseFloat(v.toFixed(4)),
  })) || [];
  return (
    <ResponsiveContainer width="100%" height={Math.max(300, data.length * 22 + 60)}>
      <BarChart data={data} layout="vertical" margin={{ top: 5, right: 40, bottom: 5, left: 120 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" tick={{ fill: "var(--text-muted)", fontSize: 10 }} domain={[-1, 1]} />
        <YAxis type="category" dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} width={115} />
        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
        <ReferenceLine x={0} stroke="var(--border-light)" />
        <Bar dataKey="loading" name={`PC${pc} Loading`}
          fill="#f59e0b"
          label={{ position: "right", fill: "var(--text-muted)", fontSize: 9, formatter: v => v.toFixed(3) }}
          radius={[0, 3, 3, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function LoadingsScatter({ result, pc1, pc2 }) {
  const data = result.loadings[pc1 - 1]?.map((v, j) => ({
    name: result.columns[j],
    pc1: v,
    pc2: result.loadings[pc2 - 1]?.[j] ?? 0,
  })) || [];
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 30, bottom: 50, left: 50 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" dataKey="pc1"
          label={{ value: `PC${pc1} Loadings`, position: "bottom", offset: 30, fill: "var(--text-muted)", fontSize: 12 }}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }} domain={[-1, 1]} />
        <YAxis type="number" dataKey="pc2"
          label={{ value: `PC${pc2} Loadings`, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }} domain={[-1, 1]} />
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
        <Scatter data={data} fill="#f59e0b" opacity={0.9} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function ScreePlot({ result }) {
  const data = result.varExplained.map((v, i) => ({
    pc: `PC${i + 1}`,
    variance: parseFloat(v.toFixed(2)),
    cumulative: parseFloat(result.varExplained.slice(0, i + 1).reduce((a, b) => a + b, 0).toFixed(2)),
  }));
  return (
    <ResponsiveContainer width="100%" height={350}>
      <ComposedChart data={data} margin={{ top: 10, right: 40, bottom: 30, left: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="pc" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
        <YAxis yAxisId="left" tick={{ fill: "var(--text-muted)", fontSize: 11 }} domain={[0, 100]} label={{ value: "% Variance", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 11 }} />
        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)" }} />
        <Legend wrapperStyle={{ color: "var(--text-secondary)", fontSize: 12 }} />
        <Bar yAxisId="left" dataKey="variance" name="Variance %" fill="#1E90FF" radius={[4, 4, 0, 0]} />
        <Line yAxisId="left" type="monotone" dataKey="cumulative" name="Cumulative %" stroke="#f59e0b" strokeWidth={2} dot={{ fill: "#f59e0b", r: 4 }} />
        <ReferenceLine yAxisId="left" y={80} stroke="#10b981" strokeDasharray="4 4" label={{ value: "80%", fill: "#10b981", fontSize: 10 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function MonitoringPlot({ result }) {
  const data = result.t2.map((v, i) => ({
    sample: i + 1,
    t2: parseFloat(v.toFixed(3)),
    q: parseFloat(result.qResiduals[i].toFixed(3)),
    outlier: v > result.t2Limit || result.qResiduals[i] > result.qLimit,
  }));
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div>
        <p className="text-xs font-semibold mb-2" style={{ color: "var(--text-muted)" }}>Hotelling T² Chart</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 10 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
            <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
            <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
            <ReferenceLine y={result.t2Limit} stroke="#ef4444" strokeDasharray="5 5" label={{ value: "UCL", fill: "#ef4444", fontSize: 10 }} />
            <Line type="monotone" dataKey="t2" stroke="#1E90FF" strokeWidth={1.5} dot={d => d.payload.t2 > result.t2Limit ? { r: 5, fill: "#ef4444" } : { r: 2, fill: "#1E90FF" }} name="T²" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div>
        <p className="text-xs font-semibold mb-2" style={{ color: "var(--text-muted)" }}>Q Residuals (SPE)</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 10 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
            <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
            <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
            <ReferenceLine y={result.qLimit} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: "UCL", fill: "#f59e0b", fontSize: 10 }} />
            <Line type="monotone" dataKey="q" stroke="#8b5cf6" strokeWidth={1.5} dot={d => d.payload.q > result.qLimit ? { r: 5, fill: "#ef4444" } : { r: 2, fill: "#8b5cf6" }} name="Q" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ── MAIN COMPONENT ───────────────────────────────────────────────────────────
export default function PCA() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [nComponents, setNComponents] = useState(3);
  const [firstCol, setFirstCol] = useState(0);
  const [lastCol, setLastCol] = useState(0);
  const [pc1, setPc1] = useState(1);
  const [pc2, setPc2] = useState(2);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState("scores");
  const [loadingView, setLoadingView] = useState("bar"); // "bar" or "scatter"
  const [colorCol, setColorCol] = useState("");
  const [error, setError] = useState("");

  const loadData = async (id, ds) => {
    setDataset(ds);
    setResult(null);
    setError("");
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

  const runPCA = () => {
    if (!data.length) return;
    setLoading(true);
    setError("");
    setTimeout(() => {
      const allCols = dataset?.columns || [];
      const colRange = allCols.slice(firstCol, lastCol + 1);
      const numCols = colRange.filter(c => !isNaN(parseFloat(data[0]?.[c])));
      if (numCols.length < 2) { setError("Need at least 2 numeric columns in selected range."); setLoading(false); return; }
      const matrix = data.map(r => numCols.map(c => parseFloat(r[c]) || 0));
      const nc = Math.min(nComponents, numCols.length, data.length - 1);
      const res = computePCA(matrix, nc);
      setResult({ ...res, columns: numCols, nSamples: data.length });
      setLoading(false);
    }, 50);
  };

  const exportScores = () => {
    if (!result) return;
    const header = ["Sample", ...result.varExplained.map((_, i) => `PC${i + 1}`)];
    const rows = result.scores[0].map((_, i) => [i + 1, ...result.scores.map(s => s[i].toFixed(5))]);
    const csv = [header, ...rows].map(r => r.join(",")).join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "pca_scores.csv"; a.click();
  };

  const exportLoadings = () => {
    if (!result) return;
    const header = ["Variable", ...result.varExplained.map((_, i) => `PC${i + 1}`)];
    const rows = result.columns.map((c, j) => [c, ...result.loadings.map(l => l[j].toFixed(5))]);
    const csv = [header, ...rows].map(r => r.join(",")).join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "pca_loadings.csv"; a.click();
  };

  const allCols = dataset?.columns || [];

  const TABS = [
    { id: "scores", label: "Score Plot" },
    { id: "loadings", label: "Loadings" },
    { id: "biplot", label: "Biplot" },
    { id: "scree", label: "Scree" },
    { id: "monitoring", label: "T²/Q Monitor" },
  ];

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
              <div>
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Components</label>
                <select value={nComponents} onChange={e => setNComponents(+e.target.value)}
                  className="px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  {[2, 3, 4, 5, 6, 7, 8, 10].map(n => <option key={n} value={n}>{n}</option>)}
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
                    <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>PC{i + 1}</div>
                    <div className="text-xl font-bold" style={{ color: "#1E90FF" }}>{v.toFixed(1)}%</div>
                    <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                      cumul. {result.varExplained.slice(0, i + 1).reduce((a, b) => a + b, 0).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>

              {/* Tab bar */}
              <div className="flex flex-wrap gap-2 items-center">
                {TABS.map(t => (
                  <button key={t.id} onClick={() => setTab(t.id)}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                    style={{ background: tab === t.id ? "#1E90FF" : "var(--bg-card)", color: tab === t.id ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                    {t.label}
                  </button>
                ))}
                {(tab === "scores" || tab === "loadings" || tab === "biplot") && (
                  <div className="flex items-center gap-2 ml-auto">
                    <select value={pc1} onChange={e => setPc1(+e.target.value)}
                      className="px-2 py-1.5 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {result.varExplained.map((_, i) => <option key={i + 1} value={i + 1}>PC{i + 1}</option>)}
                    </select>
                    <span style={{ color: "var(--text-muted)" }}>vs</span>
                    <select value={pc2} onChange={e => setPc2(+e.target.value)}
                      className="px-2 py-1.5 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {result.varExplained.map((_, i) => <option key={i + 1} value={i + 1}>PC{i + 1}</option>)}
                    </select>
                  </div>
                )}
              </div>

              {/* Charts */}
              <div className="glass-card p-5">
                {tab === "scores" && (
                  <>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Score Plot — PC{pc1} vs PC{pc2}</h3>
                      <button onClick={exportScores} className="btn-secondary text-xs flex items-center gap-1.5">
                        <Download size={12} /> Export Scores
                      </button>
                    </div>
                    <ScorePlot result={result} pc1={pc1} pc2={pc2} data={data} colorCol={colorCol} />
                  </>
                )}

                {tab === "loadings" && (
                  <>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Loadings</h3>
                      <div className="flex items-center gap-2">
                        <button onClick={() => setLoadingView("bar")}
                          className="px-2 py-1 rounded text-xs"
                          style={{ background: loadingView === "bar" ? "#f59e0b" : "var(--bg-secondary)", color: loadingView === "bar" ? "white" : "var(--text-muted)", border: "1px solid var(--border)" }}>
                          Bar
                        </button>
                        <button onClick={() => setLoadingView("scatter")}
                          className="px-2 py-1 rounded text-xs"
                          style={{ background: loadingView === "scatter" ? "#f59e0b" : "var(--bg-secondary)", color: loadingView === "scatter" ? "white" : "var(--text-muted)", border: "1px solid var(--border)" }}>
                          Scatter
                        </button>
                        <button onClick={exportLoadings} className="btn-secondary text-xs flex items-center gap-1.5">
                          <Download size={12} /> Export Loadings
                        </button>
                      </div>
                    </div>
                    {loadingView === "bar" ? (
                      <>
                        <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>PC{pc1} loadings — sorted by variable order</p>
                        <LoadingsBarChart result={result} pc={pc1} />
                      </>
                    ) : (
                      <>
                        <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>PC{pc1} vs PC{pc2} loadings scatter</p>
                        <LoadingsScatter result={result} pc1={pc1} pc2={pc2} />
                      </>
                    )}
                  </>
                )}

                {tab === "biplot" && (
                  <>
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Biplot — PC{pc1} vs PC{pc2}</h3>
                    <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Samples (circles) + Loadings (bars, scaled ×3)</p>
                    <ResponsiveContainer width="100%" height={420}>
                      <ScatterChart margin={{ top: 20, right: 30, bottom: 50, left: 50 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis type="number" dataKey="x"
                          label={{ value: `PC${pc1} (${result.varExplained[pc1 - 1]?.toFixed(1)}%)`, position: "bottom", offset: 30, fill: "var(--text-muted)", fontSize: 12 }}
                          tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <YAxis type="number" dataKey="y"
                          label={{ value: `PC${pc2} (${result.varExplained[pc2 - 1]?.toFixed(1)}%)`, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
                          tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                        <Tooltip content={({ payload }) => {
                          if (!payload?.length) return null;
                          const d = payload[0].payload;
                          return (
                            <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                              <div className="font-semibold">{d.name}</div>
                              <div>PC{pc1}: {d.x?.toFixed(3)}</div>
                              <div>PC{pc2}: {d.y?.toFixed(3)}</div>
                            </div>
                          );
                        }} />
                        <ReferenceLine x={0} stroke="var(--border-light)" />
                        <ReferenceLine y={0} stroke="var(--border-light)" />
                        <Scatter
                          data={result.scores[pc1 - 1]?.map((v, i) => ({ x: v, y: result.scores[pc2 - 1]?.[i] ?? 0, name: `S${i + 1}` }))}
                          fill="#1E90FF" opacity={0.8} />
                        {result.loadings[pc1 - 1]?.map((lv, j) => (
                          <ReferenceLine key={j}
                            segment={[{ x: 0, y: 0 }, { x: lv * 3, y: (result.loadings[pc2 - 1]?.[j] ?? 0) * 3 }]}
                            stroke="#f59e0b" strokeWidth={1.5} />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                    <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>
                      Variables (orange lines): {result.columns.slice(0, 6).join(", ")}{result.columns.length > 6 ? ` +${result.columns.length - 6} more` : ""}
                    </p>
                  </>
                )}

                {tab === "scree" && (
                  <>
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Scree Plot — Variance Explained</h3>
                    <ScreePlot result={result} />
                    <div className="mt-3 overflow-x-auto">
                      <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                        <thead>
                          <tr>
                            {["PC", "Eigenvalue %", "Cumulative %"].map(h => (
                              <th key={h} className="px-3 py-2 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {result.varExplained.map((v, i) => (
                            <tr key={i} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                              <td className="px-3 py-2 font-medium" style={{ color: "var(--text-primary)" }}>PC{i + 1}</td>
                              <td className="px-3 py-2" style={{ color: "#1E90FF" }}>{v.toFixed(2)}%</td>
                              <td className="px-3 py-2" style={{ color: "#f59e0b" }}>{result.varExplained.slice(0, i + 1).reduce((a, b) => a + b, 0).toFixed(2)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                {tab === "monitoring" && (
                  <>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>PCA Monitoring — T² & Q Control Charts</h3>
                      <div className="flex gap-3">
                        <div className="text-xs px-2 py-1 rounded" style={{ background: "rgba(239,68,68,0.1)", color: "#ef4444" }}>
                          T² UCL: {result.t2Limit.toFixed(2)}
                        </div>
                        <div className="text-xs px-2 py-1 rounded" style={{ background: "rgba(245,158,11,0.1)", color: "#f59e0b" }}>
                          Q UCL: {result.qLimit.toFixed(4)}
                        </div>
                      </div>
                    </div>
                    <MonitoringPlot result={result} />
                    {/* Outlier table */}
                    {result.t2.some((v, i) => v > result.t2Limit || result.qResiduals[i] > result.qLimit) && (
                      <div className="mt-4">
                        <p className="text-xs font-semibold mb-2" style={{ color: "#ef4444" }}>⚠ Out-of-Control Samples</p>
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                            <thead>
                              <tr>{["Sample", "T²", "Q", "Flag"].map(h => (
                                <th key={h} className="px-3 py-2 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                              ))}</tr>
                            </thead>
                            <tbody>
                              {result.t2.map((t2v, i) => {
                                const isOut = t2v > result.t2Limit || result.qResiduals[i] > result.qLimit;
                                if (!isOut) return null;
                                return (
                                  <tr key={i} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                                    <td className="px-3 py-2" style={{ color: "var(--text-primary)" }}>{i + 1}</td>
                                    <td className="px-3 py-2" style={{ color: t2v > result.t2Limit ? "#ef4444" : "var(--text-secondary)" }}>{t2v.toFixed(3)}</td>
                                    <td className="px-3 py-2" style={{ color: result.qResiduals[i] > result.qLimit ? "#f59e0b" : "var(--text-secondary)" }}>{result.qResiduals[i].toFixed(4)}</td>
                                    <td className="px-3 py-2">
                                      {t2v > result.t2Limit && <span className="tag tag-red mr-1">T² outlier</span>}
                                      {result.qResiduals[i] > result.qLimit && <span className="tag tag-yellow">Q outlier</span>}
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
            </>
          )}
        </>
      )}
    </div>
  );
}