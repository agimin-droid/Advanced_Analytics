import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, ScatterChart, Scatter, BarChart, Bar, Legend
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { Activity, RefreshCw, AlertCircle, Download } from "lucide-react";

// ── NIPALS PCA (same as PCA page) ────────────────────────────────────────────
function computePCA(matrix, nc) {
  const n = matrix.length, p = matrix[0].length;
  const nComp = Math.min(nc, n - 1, p);
  const means = Array(p).fill(0), stds = Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) means[j] += matrix[i][j];
    means[j] /= n;
    for (let i = 0; i < n; i++) stds[j] += (matrix[i][j] - means[j]) ** 2;
    stds[j] = Math.sqrt(stds[j] / (n - 1)) || 1;
  }
  const X = matrix.map(row => row.map((v, j) => (v - means[j]) / stds[j]));
  const scores = [], loadings = [], varExplained = [];
  let Xres = X.map(r => [...r]);
  const totalVar = X.reduce((s, r) => s + r.reduce((a, v) => a + v * v, 0), 0);
  for (let k = 0; k < nComp; k++) {
    let t = Xres.map(r => r[0]);
    for (let iter = 0; iter < 200; iter++) {
      let pv = Array(p).fill(0);
      const tt = t.reduce((s, v) => s + v * v, 0);
      if (tt < 1e-10) break;
      for (let j = 0; j < p; j++) for (let i = 0; i < n; i++) pv[j] += t[i] * Xres[i][j];
      const pnorm = Math.sqrt(pv.reduce((s, v) => s + v * v, 0)) || 1;
      pv = pv.map(v => v / pnorm);
      const tnew = Xres.map(r => r.reduce((s, v, j) => s + v * pv[j], 0));
      const diff = tnew.reduce((s, v, i) => s + (v - t[i]) ** 2, 0);
      t = tnew;
      if (diff < 1e-10) break;
    }
    let pv = Array(p).fill(0);
    const tt = t.reduce((s, v) => s + v * v, 0);
    for (let j = 0; j < p; j++) for (let i = 0; i < n; i++) pv[j] += t[i] * Xres[i][j];
    const pnorm = Math.sqrt(pv.reduce((s, v) => s + v * v, 0)) || 1;
    pv = pv.map(v => v / pnorm);
    Xres = Xres.map((r, i) => r.map((v, j) => v - t[i] * pv[j]));
    varExplained.push(tt / totalVar * 100);
    scores.push(t); loadings.push(pv);
  }
  return { scores, loadings, varExplained, means, stds, X, Xres };
}

export default function QualityControl() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [nComp, setNComp] = useState(3);
  const [firstCol, setFirstCol] = useState(0);
  const [lastCol, setLastCol] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("charts");
  const [selectedSample, setSelectedSample] = useState(null);

  const loadData = async (id, ds) => {
    setDataset(ds); setResult(null); setError(""); setSelectedSample(null);
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const nc = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    const allCols = ds?.columns || [];
    setFirstCol(nc.length ? allCols.indexOf(nc[0]) : 0);
    setLastCol(nc.length ? allCols.indexOf(nc[nc.length - 1]) : 0);
  };

  const runQC = () => {
    if (!data.length) return;
    setLoading(true); setError(""); setSelectedSample(null);
    setTimeout(() => {
      const allCols = dataset?.columns || [];
      const colRange = allCols.slice(firstCol, lastCol + 1);
      const numCols = colRange.filter(c => !isNaN(parseFloat(data[0]?.[c])));
      if (numCols.length < 2) { setError("Need at least 2 numeric columns in selected range."); setLoading(false); return; }
      const matrix = data.map(r => numCols.map(c => parseFloat(r[c]) || 0));
      const nc = Math.min(nComp, numCols.length, data.length - 1);
      const pca = computePCA(matrix, nc);
      const n = matrix.length, p = numCols.length;

      // Hotelling T²
      const t2 = pca.scores[0].map((_, i) =>
        pca.scores.slice(0, nc).reduce((s, sc) => {
          const varPc = sc.reduce((a, v) => a + v * v, 0) / (n - 1);
          return s + (sc[i] ** 2) / (varPc || 1);
        }, 0)
      );

      // Q residuals (SPE)
      const qResiduals = pca.Xres.map(r => r.reduce((s, v) => s + v * v, 0));

      // Control limits
      const t2Limit = nc * (n * n - 1) / (n * (n - nc)) * 3.0;
      const qMean = qResiduals.reduce((a, b) => a + b, 0) / n;
      const qVar = qResiduals.reduce((a, v) => a + (v - qMean) ** 2, 0) / (n - 1);
      const qLimit = qMean + 3 * Math.sqrt(qVar);

      // T² contributions per variable for each sample: contribution_ij = (x_ij * loading_j)^2 summed over PCs
      const t2Contributions = matrix.map((row, i) => {
        return numCols.map((_, j) => {
          return pca.scores.slice(0, nc).reduce((s, sc, k) => {
            const varPc = sc.reduce((a, v) => a + v * v, 0) / (n - 1);
            const xScaled = (row[j] - pca.means[j]) / pca.stds[j];
            return s + (xScaled * pca.loadings[k][j]) ** 2 / (varPc || 1);
          }, 0);
        });
      });

      // Q contributions per variable: (Xres_ij)^2
      const qContributions = pca.Xres.map((row) => row.map(v => v * v));

      const samples = t2.map((t2v, i) => ({
        sample: i + 1,
        t2: parseFloat(t2v.toFixed(3)),
        q: parseFloat(qResiduals[i].toFixed(3)),
        outlier_t2: t2v > t2Limit,
        outlier_q: qResiduals[i] > qLimit,
        t2Contrib: t2Contributions[i],
        qContrib: qContributions[i],
      }));

      const nOutliers = samples.filter(s => s.outlier_t2 || s.outlier_q).length;
      setResult({ samples, t2Limit: parseFloat(t2Limit.toFixed(3)), qLimit: parseFloat(qLimit.toFixed(3)), nc, nOutliers, numCols, varExplained: pca.varExplained });
      setLoading(false);
    }, 50);
  };

  const exportQC = () => {
    if (!result) return;
    const csv = ["Sample,T2,Q,Outlier_T2,Outlier_Q",
      ...result.samples.map(s => `${s.sample},${s.t2},${s.q},${s.outlier_t2},${s.outlier_q}`)
    ].join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "quality_control.csv"; a.click();
  };

  const allCols = dataset?.columns || [];
  const selSampleData = selectedSample !== null && result ? result.samples[selectedSample - 1] : null;

  // Contribution chart data for selected sample
  const t2ContribData = selSampleData ? result.numCols.map((c, j) => ({
    name: c.length > 12 ? `${c.slice(0, 10)}…` : c, value: parseFloat(selSampleData.t2Contrib[j].toFixed(4))
  })).sort((a, b) => b.value - a.value).slice(0, 20) : [];

  const qContribData = selSampleData ? result.numCols.map((c, j) => ({
    name: c.length > 12 ? `${c.slice(0, 10)}…` : c, value: parseFloat(selSampleData.qContrib[j].toFixed(4))
  })).sort((a, b) => b.value - a.value).slice(0, 20) : [];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>📊 Quality Control</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>PCA Monitoring — Hotelling T² and Q residuals (SPE) with variable contributions</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <Activity size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a dataset to begin quality control monitoring</p>
        </div>
      ) : (
        <>
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>⚙️ Configuration</h3>
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
                <select value={nComp} onChange={e => setNComp(+e.target.value)}
                  className="px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  {[2, 3, 4, 5, 6, 7, 8].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <button onClick={runQC} disabled={loading || !data.length}
                className="btn-primary disabled:opacity-40"
                style={{ background: "linear-gradient(135deg, #f59e0b, #d97706)" }}>
                {loading ? <RefreshCw size={14} className="animate-spin" /> : <Activity size={14} />}
                {loading ? "Computing..." : "Run QC Analysis"}
              </button>
              {result && (
                <button onClick={exportQC} className="btn-secondary text-xs flex items-center gap-1.5">
                  <Download size={12} /> Export CSV
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
                  { label: "Samples", value: result.samples.length, color: "#1E90FF" },
                  { label: "Variables", value: result.numCols.length, color: "#06b6d4" },
                  { label: "Components", value: result.nc, color: "#8b5cf6" },
                  { label: "T² UCL", value: result.t2Limit, color: "#f59e0b" },
                  { label: "Outliers", value: result.nOutliers, color: result.nOutliers > 0 ? "#ef4444" : "#10b981" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="glass-card p-4">
                    <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
                    <div className="text-xl font-bold" style={{ color }}>{value}</div>
                  </div>
                ))}
              </div>

              {/* Variance explained */}
              <div className="glass-card p-3 flex flex-wrap gap-3">
                {result.varExplained.map((v, i) => (
                  <span key={i} className="text-xs px-2 py-1 rounded-lg" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)" }}>
                    PC{i + 1}: <span style={{ color: "#1E90FF" }}>{v.toFixed(1)}%</span>
                    <span style={{ color: "var(--text-muted)" }}> (cum. {result.varExplained.slice(0, i + 1).reduce((a, b) => a + b, 0).toFixed(1)}%)</span>
                  </span>
                ))}
              </div>

              {/* Tabs */}
              <div className="flex flex-wrap gap-2">
                {[
                  { id: "charts", label: "Control Charts" },
                  { id: "scatter", label: "T² vs Q Plot" },
                  { id: "contrib", label: "Contributions" },
                  { id: "table", label: "Outlier Table" },
                ].map(t => (
                  <button key={t.id} onClick={() => setTab(t.id)}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                    style={{ background: tab === t.id ? "#f59e0b" : "var(--bg-card)", color: tab === t.id ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                    {t.label}
                  </button>
                ))}
              </div>

              {/* Control Charts */}
              {tab === "charts" && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Hotelling T² Control Chart</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={result.samples} margin={{ top: 5, right: 20, bottom: 25, left: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                          formatter={(v, n, p) => [v, n]}
                          labelFormatter={v => `Sample ${v}`} />
                        <ReferenceLine y={result.t2Limit} stroke="#ef4444" strokeDasharray="5 5" label={{ value: "UCL", fill: "#ef4444", fontSize: 11 }} />
                        <Line type="monotone" dataKey="t2" stroke="#1E90FF" strokeWidth={1.5}
                          dot={d => d.payload.outlier_t2 ? { r: 6, fill: "#ef4444", stroke: "#ef4444" } : { r: 2.5, fill: "#1E90FF" }}
                          activeDot={{ r: 6, onClick: (e, p) => setSelectedSample(p.payload.sample) }}
                          name="T²" />
                      </LineChart>
                    </ResponsiveContainer>
                    <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>Click a point to view its variable contributions below</p>
                  </div>

                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Q Residuals (SPE) Chart</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={result.samples} margin={{ top: 5, right: 20, bottom: 25, left: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                          labelFormatter={v => `Sample ${v}`} />
                        <ReferenceLine y={result.qLimit} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: "UCL", fill: "#f59e0b", fontSize: 11 }} />
                        <Line type="monotone" dataKey="q" stroke="#8b5cf6" strokeWidth={1.5}
                          dot={d => d.payload.outlier_q ? { r: 6, fill: "#ef4444", stroke: "#ef4444" } : { r: 2.5, fill: "#8b5cf6" }}
                          activeDot={{ r: 6, onClick: (e, p) => setSelectedSample(p.payload.sample) }}
                          name="Q" />
                      </LineChart>
                    </ResponsiveContainer>
                    <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>Click a point to view its variable contributions below</p>
                  </div>
                </div>
              )}

              {/* T² vs Q scatter */}
              {tab === "scatter" && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>T² vs Q Diagnostic Plot</h3>
                  <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
                    Quadrant I (upper right) = outlier in both · Red dots = out-of-control samples
                  </p>
                  <ResponsiveContainer width="100%" height={380}>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 50, left: 55 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="t2" name="T²"
                        label={{ value: "Hotelling T²", position: "bottom", offset: 30, fill: "var(--text-muted)", fontSize: 12 }}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <YAxis type="number" dataKey="q" name="Q"
                        label={{ value: "Q Residuals (SPE)", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }}
                        tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                      <Tooltip content={({ payload }) => {
                        if (!payload?.length) return null;
                        const d = payload[0].payload;
                        return (
                          <div className="rounded-lg p-3 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                            <div className="font-semibold">Sample {d.sample}</div>
                            <div>T²: {d.t2}</div>
                            <div>Q: {d.q}</div>
                            {(d.outlier_t2 || d.outlier_q) && <div style={{ color: "#ef4444" }}>⚠ Outlier</div>}
                          </div>
                        );
                      }} />
                      <ReferenceLine x={result.t2Limit} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "T² UCL", fill: "#ef4444", fontSize: 10 }} />
                      <ReferenceLine y={result.qLimit} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: "Q UCL", fill: "#f59e0b", fontSize: 10 }} />
                      <Scatter
                        data={result.samples.map(s => ({ ...s }))}
                        fill="#1E90FF" opacity={0.8}
                        shape={({ cx, cy, payload }) => (
                          <circle cx={cx} cy={cy} r={5}
                            fill={(payload.outlier_t2 || payload.outlier_q) ? "#ef4444" : "#1E90FF"}
                            opacity={0.8}
                            onClick={() => setSelectedSample(payload.sample)}
                            style={{ cursor: "pointer" }} />
                        )}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                  <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>Click a point to select it for contribution analysis</p>
                </div>
              )}

              {/* Contributions */}
              {tab === "contrib" && (
                <div className="space-y-4">
                  <div className="glass-card p-4 flex flex-wrap gap-3 items-center">
                    <span className="text-sm" style={{ color: "var(--text-muted)" }}>Sample:</span>
                    <select value={selectedSample || ""} onChange={e => setSelectedSample(+e.target.value)}
                      className="px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      <option value="">Select sample to inspect</option>
                      {result.samples.map(s => (
                        <option key={s.sample} value={s.sample}>
                          Sample {s.sample}{(s.outlier_t2 || s.outlier_q) ? " ⚠" : ""}
                        </option>
                      ))}
                    </select>
                    {selSampleData && (
                      <div className="flex gap-3 text-xs ml-2">
                        <span style={{ color: selSampleData.outlier_t2 ? "#ef4444" : "#10b981" }}>T² = {selSampleData.t2}</span>
                        <span style={{ color: selSampleData.outlier_q ? "#f59e0b" : "#10b981" }}>Q = {selSampleData.q}</span>
                      </div>
                    )}
                  </div>

                  {selSampleData && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <div className="glass-card p-5">
                        <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
                          T² Contributions — Sample {selectedSample}
                        </h3>
                        <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Top 20 variables contributing to T² (identifies model misfit cause)</p>
                        <ResponsiveContainer width="100%" height={Math.max(300, t2ContribData.length * 22 + 50)}>
                          <BarChart data={t2ContribData} layout="vertical" margin={{ top: 5, right: 50, bottom: 5, left: 120 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                            <XAxis type="number" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                            <YAxis type="category" dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} width={115} />
                            <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                            <Bar dataKey="value" name="T² contribution" fill="#1E90FF" radius={[0, 3, 3, 0]}
                              label={{ position: "right", fill: "var(--text-muted)", fontSize: 9, formatter: v => v.toFixed(3) }} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="glass-card p-5">
                        <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
                          Q Contributions — Sample {selectedSample}
                        </h3>
                        <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Top 20 variables contributing to Q residuals (identifies unexplained variance)</p>
                        <ResponsiveContainer width="100%" height={Math.max(300, qContribData.length * 22 + 50)}>
                          <BarChart data={qContribData} layout="vertical" margin={{ top: 5, right: 50, bottom: 5, left: 120 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                            <XAxis type="number" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                            <YAxis type="category" dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} width={115} />
                            <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                            <Bar dataKey="value" name="Q contribution" fill="#8b5cf6" radius={[0, 3, 3, 0]}
                              label={{ position: "right", fill: "var(--text-muted)", fontSize: 9, formatter: v => v.toFixed(3) }} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}

                  {!selSampleData && (
                    <div className="glass-card p-10 text-center">
                      <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a sample above or click a point on the control charts to view its variable contributions</p>
                    </div>
                  )}
                </div>
              )}

              {/* Outlier table */}
              {tab === "table" && (
                <>
                  {result.nOutliers > 0 ? (
                    <div className="glass-card overflow-hidden">
                      <div className="p-4" style={{ borderBottom: "1px solid var(--border)" }}>
                        <h3 className="text-sm font-semibold" style={{ color: "#ef4444" }}>⚠ Out-of-Control Samples ({result.nOutliers})</h3>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                          <thead>
                            <tr>{["Sample", "T²", "Q", "T² UCL", "Q UCL", "Flag"].map(h => (
                              <th key={h} className="px-4 py-2.5 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                            ))}</tr>
                          </thead>
                          <tbody>
                            {result.samples.filter(s => s.outlier_t2 || s.outlier_q).map(s => (
                              <tr key={s.sample} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)", cursor: "pointer" }}
                                onClick={() => { setSelectedSample(s.sample); setTab("contrib"); }}>
                                <td className="px-4 py-2.5" style={{ color: "var(--text-primary)" }}>{s.sample}</td>
                                <td className="px-4 py-2.5" style={{ color: s.outlier_t2 ? "#ef4444" : "var(--text-secondary)" }}>{s.t2}</td>
                                <td className="px-4 py-2.5" style={{ color: s.outlier_q ? "#f59e0b" : "var(--text-secondary)" }}>{s.q}</td>
                                <td className="px-4 py-2.5" style={{ color: "var(--text-muted)" }}>{result.t2Limit}</td>
                                <td className="px-4 py-2.5" style={{ color: "var(--text-muted)" }}>{result.qLimit}</td>
                                <td className="px-4 py-2.5">
                                  {s.outlier_t2 && <span className="tag tag-red mr-1">T² outlier</span>}
                                  {s.outlier_q && <span className="tag tag-yellow">Q outlier</span>}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="text-xs p-3" style={{ color: "var(--text-muted)" }}>Click a row to view variable contributions for that sample</p>
                    </div>
                  ) : (
                    <div className="glass-card p-10 text-center">
                      <div className="text-3xl mb-2">✅</div>
                      <p className="text-sm font-medium" style={{ color: "#10b981" }}>No out-of-control samples detected</p>
                      <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>All samples are within T² and Q control limits</p>
                    </div>
                  )}

                  {/* Full sample table */}
                  <div className="glass-card overflow-hidden">
                    <div className="p-4" style={{ borderBottom: "1px solid var(--border)" }}>
                      <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>All Samples</h3>
                    </div>
                    <div className="overflow-x-auto" style={{ maxHeight: 400 }}>
                      <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                        <thead>
                          <tr>{["Sample", "T²", "Q", "T² OK", "Q OK"].map(h => (
                            <th key={h} className="px-3 py-2.5 text-left sticky top-0" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                          ))}</tr>
                        </thead>
                        <tbody>
                          {result.samples.map(s => (
                            <tr key={s.sample} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}
                              onClick={() => { setSelectedSample(s.sample); setTab("contrib"); }}
                              className="cursor-pointer">
                              <td className="px-3 py-2" style={{ color: "var(--text-primary)" }}>{s.sample}</td>
                              <td className="px-3 py-2" style={{ color: s.outlier_t2 ? "#ef4444" : "var(--text-secondary)" }}>{s.t2}</td>
                              <td className="px-3 py-2" style={{ color: s.outlier_q ? "#f59e0b" : "var(--text-secondary)" }}>{s.q}</td>
                              <td className="px-3 py-2">{s.outlier_t2 ? <span style={{ color: "#ef4444" }}>✗</span> : <span style={{ color: "#10b981" }}>✓</span>}</td>
                              <td className="px-3 py-2">{s.outlier_q ? <span style={{ color: "#f59e0b" }}>✗</span> : <span style={{ color: "#10b981" }}>✓</span>}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}