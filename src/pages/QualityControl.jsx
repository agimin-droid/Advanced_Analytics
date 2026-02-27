import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend, ScatterChart, Scatter
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { Activity, RefreshCw, AlertCircle } from "lucide-react";

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
    for (let iter = 0; iter < 100; iter++) {
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
  const [alpha, setAlpha] = useState(0.05);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadData = async (id, ds) => {
    setDataset(ds); setResult(null); setError("");
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
  };

  const runQC = () => {
    if (!data.length) return;
    setLoading(true); setError("");
    setTimeout(() => {
      const numCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];
      if (numCols.length < 2) { setError("Need at least 2 numeric columns."); setLoading(false); return; }
      const matrix = data.map(r => numCols.map(c => parseFloat(r[c]) || 0));
      const nc = Math.min(nComp, numCols.length, data.length - 1);
      const pca = computePCA(matrix, nc);
      const n = matrix.length;

      // T² statistic (Hotelling)
      const t2 = pca.scores[0].map((_, i) => {
        return pca.scores.slice(0, nc).reduce((s, sc, k) => {
          const varPc = sc.reduce((a, v) => a + v * v, 0) / (n - 1);
          return s + (sc[i] ** 2) / (varPc || 1);
        }, 0);
      });

      // Q residuals (SPE)
      const qResiduals = pca.Xres.map(r => r.reduce((s, v) => s + v * v, 0));

      // Control limits (chi-squared approximation)
      const t2Limit = nc * (n * n - 1) / (n * (n - nc)) * 3.0; // approx F critical
      const qMean = qResiduals.reduce((a, b) => a + b, 0) / n;
      const qVar = qResiduals.reduce((a, v) => a + (v - qMean) ** 2, 0) / (n - 1);
      const qLimit = qMean + 3 * Math.sqrt(qVar);

      const samples = t2.map((t2v, i) => ({
        sample: i + 1,
        t2: parseFloat(t2v.toFixed(3)),
        q: parseFloat(qResiduals[i].toFixed(3)),
        outlier_t2: t2v > t2Limit,
        outlier_q: qResiduals[i] > qLimit,
      }));

      const nOutliers = samples.filter(s => s.outlier_t2 || s.outlier_q).length;
      setResult({ samples, t2Limit: parseFloat(t2Limit.toFixed(3)), qLimit: parseFloat(qLimit.toFixed(3)), nc, nOutliers });
      setLoading(false);
    }, 50);
  };

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>📊 Quality Control</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>PCA Monitoring — Hotelling T² and Q residuals (SPE)</p>
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
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Components</label>
                <select value={nComp} onChange={e => setNComp(+e.target.value)}
                  className="px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  {[2,3,4,5,6].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <button onClick={runQC} disabled={loading || !data.length}
                className="btn-primary disabled:opacity-40"
                style={{ background: "linear-gradient(135deg, #f59e0b, #d97706)" }}>
                {loading ? <RefreshCw size={14} className="animate-spin" /> : <Activity size={14} />}
                {loading ? "Computing..." : "Run QC Analysis"}
              </button>
            </div>
            {error && <div className="flex items-center gap-2 mt-3 text-sm" style={{ color: "#f87171" }}><AlertCircle size={14} /> {error}</div>}
          </div>

          {result && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "Samples", value: result.samples.length, color: "#1E90FF" },
                  { label: "Components", value: result.nc, color: "#8b5cf6" },
                  { label: "T² Limit", value: result.t2Limit, color: "#f59e0b" },
                  { label: "Outliers", value: result.nOutliers, color: result.nOutliers > 0 ? "#ef4444" : "#10b981" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="glass-card p-4">
                    <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
                    <div className="text-xl font-bold" style={{ color }}>{value}</div>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* T² chart */}
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Hotelling T² Control Chart</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={result.samples} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                      <ReferenceLine y={result.t2Limit} stroke="#ef4444" strokeDasharray="5 5" label={{ value: "UCL", fill: "#ef4444", fontSize: 11 }} />
                      <Line type="monotone" dataKey="t2" stroke="#1E90FF" strokeWidth={1.5} dot={d => d.payload.outlier_t2 ? { r: 5, fill: "#ef4444" } : { r: 2, fill: "#1E90FF" }} name="T²" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Q residuals chart */}
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Q Residuals (SPE) Chart</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={result.samples} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="sample" label={{ value: "Sample", position: "bottom", fill: "var(--text-muted)", fontSize: 11 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                      <ReferenceLine y={result.qLimit} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: "UCL", fill: "#f59e0b", fontSize: 11 }} />
                      <Line type="monotone" dataKey="q" stroke="#8b5cf6" strokeWidth={1.5} dot={d => d.payload.outlier_q ? { r: 5, fill: "#ef4444" } : { r: 2, fill: "#8b5cf6" }} name="Q" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* T² vs Q scatter */}
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>T² vs Q Diagnostic Plot</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" dataKey="t2" name="T²" label={{ value: "T²", position: "bottom", fill: "var(--text-muted)", fontSize: 12 }} tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                    <YAxis type="number" dataKey="q" name="Q" label={{ value: "Q Residuals", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }} tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
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
                    <ReferenceLine x={result.t2Limit} stroke="#ef4444" strokeDasharray="4 4" />
                    <ReferenceLine y={result.qLimit} stroke="#f59e0b" strokeDasharray="4 4" />
                    <Scatter data={result.samples.map(s => ({ ...s, fill: (s.outlier_t2 || s.outlier_q) ? "#ef4444" : "#1E90FF" }))} fill="#1E90FF" opacity={0.8} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              {result.nOutliers > 0 && (
                <div className="glass-card p-4">
                  <h3 className="text-sm font-semibold mb-3" style={{ color: "#ef4444" }}>⚠ Out-of-Control Samples</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>{["Sample", "T²", "Q", "Flag"].map(h => (
                          <th key={h} className="px-4 py-2 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                        ))}</tr>
                      </thead>
                      <tbody>
                        {result.samples.filter(s => s.outlier_t2 || s.outlier_q).map(s => (
                          <tr key={s.sample} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                            <td className="px-4 py-2" style={{ color: "var(--text-primary)" }}>{s.sample}</td>
                            <td className="px-4 py-2" style={{ color: s.outlier_t2 ? "#ef4444" : "var(--text-secondary)" }}>{s.t2}</td>
                            <td className="px-4 py-2" style={{ color: s.outlier_q ? "#f59e0b" : "var(--text-secondary)" }}>{s.q}</td>
                            <td className="px-4 py-2">
                              {s.outlier_t2 && <span className="tag tag-red mr-1">T² outlier</span>}
                              {s.outlier_q && <span className="tag tag-yellow">Q outlier</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
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