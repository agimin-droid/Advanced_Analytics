import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, LineChart, Line, ScatterChart, Scatter
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { FlaskConical, RefreshCw, AlertCircle } from "lucide-react";

// ── STATISTICS ───────────────────────────────────────────────────────────────
function computeGroupStats(values) {
  const n = values.length;
  if (!n) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = n > 1 ? values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
  const std = Math.sqrt(variance);
  const se = std / Math.sqrt(n);
  const q1 = sorted[Math.floor(n * 0.25)];
  const q2 = sorted[Math.floor(n * 0.5)];
  const q3 = sorted[Math.floor(n * 0.75)];
  return { n, mean, std, variance, se, min: sorted[0], max: sorted[n - 1], q1, q2, q3 };
}

// Welch's t-test (unequal variances)
function welchTTest(s1, s2, alternative = "two-sided") {
  if (!s1 || !s2 || s1.n < 2 || s2.n < 2) return null;
  const se = Math.sqrt(s1.variance / s1.n + s2.variance / s2.n);
  if (se === 0) return null;
  const t = (s1.mean - s2.mean) / se;
  // Welch-Satterthwaite df
  const num = (s1.variance / s1.n + s2.variance / s2.n) ** 2;
  const den = (s1.variance / s1.n) ** 2 / (s1.n - 1) + (s2.variance / s2.n) ** 2 / (s2.n - 1);
  const df = den > 0 ? num / den : s1.n + s2.n - 2;
  // P-value approximation (two-tailed)
  const x = Math.abs(t);
  const pTwoTailed = Math.min(1, 2 * Math.exp(-0.717 * x - 0.416 * x * x));
  const p = alternative === "two-sided" ? pTwoTailed :
    alternative === "greater" ? (t > 0 ? pTwoTailed / 2 : 1 - pTwoTailed / 2) :
      (t < 0 ? pTwoTailed / 2 : 1 - pTwoTailed / 2);
  // 95% CI for difference
  const tCrit = 2.0; // approx for large df
  const ci = [s1.mean - s2.mean - tCrit * se, s1.mean - s2.mean + tCrit * se];
  return { t, df: parseFloat(df.toFixed(1)), p, ci, se, pooledSE: se, diffMean: s1.mean - s2.mean };
}

// F-test for equal variances
function fTest(s1, s2) {
  if (!s1 || !s2 || s1.n < 2 || s2.n < 2) return null;
  const f = s1.variance / (s2.variance || 1e-10);
  // P-value approximation
  const x = Math.abs(Math.log(f));
  const p = Math.min(1, 2 * Math.exp(-0.5 * x * Math.min(s1.n, s2.n)));
  return { f, df1: s1.n - 1, df2: s2.n - 1, p, ratio: s1.std / (s2.std || 1) };
}

function makeHistogram(values, bins = 12) {
  if (!values.length) return [];
  const min = Math.min(...values), max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const hist = Array.from({ length: bins }, (_, i) => ({
    x: parseFloat((min + i * step + step / 2).toFixed(3)),
    count: 0,
  }));
  values.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), bins - 1);
    if (idx >= 0) hist[idx].count++;
  });
  return hist;
}

const COLORS = { g1: "#1E90FF", g2: "#10b981" };

export default function TTest() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [mode, setMode] = useState("columns"); // "columns" or "groups"
  const [col1, setCol1] = useState("");
  const [col2, setCol2] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [valueCol, setValueCol] = useState("");
  const [group1Val, setGroup1Val] = useState("");
  const [group2Val, setGroup2Val] = useState("");
  const [alternative, setAlternative] = useState("two-sided");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const loadData = async (id, ds) => {
    setDataset(ds); setResult(null); setError("");
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const numCols = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    const catCols = ds?.columns?.filter(c => isNaN(parseFloat(rows[0]?.[c]))) || [];
    if (numCols.length >= 2) { setCol1(numCols[0]); setCol2(numCols[1]); setValueCol(numCols[0]); }
    if (catCols.length) setGroupCol(catCols[0]);
    else if (ds?.columns?.length) setGroupCol(ds.columns[0]);
  };

  const runTest = () => {
    setError(""); setResult(null);
    let vals1 = [], vals2 = [], label1, label2;

    if (mode === "columns") {
      if (!col1 || !col2) { setError("Select two numeric columns."); return; }
      vals1 = data.map(r => parseFloat(r[col1])).filter(v => !isNaN(v));
      vals2 = data.map(r => parseFloat(r[col2])).filter(v => !isNaN(v));
      label1 = col1; label2 = col2;
    } else {
      if (!groupCol || !valueCol || !group1Val || !group2Val) { setError("Select group column, value column, and both group values."); return; }
      vals1 = data.filter(r => String(r[groupCol]) === group1Val).map(r => parseFloat(r[valueCol])).filter(v => !isNaN(v));
      vals2 = data.filter(r => String(r[groupCol]) === group2Val).map(r => parseFloat(r[valueCol])).filter(v => !isNaN(v));
      label1 = `${group1Val}`; label2 = `${group2Val}`;
    }

    if (vals1.length < 2 || vals2.length < 2) { setError("Each group needs at least 2 observations."); return; }

    const s1 = computeGroupStats(vals1);
    const s2 = computeGroupStats(vals2);
    const ttest = welchTTest(s1, s2, alternative);
    const ftest = fTest(s1, s2);
    const h1 = makeHistogram(vals1);
    const h2 = makeHistogram(vals2);
    setResult({ s1, s2, ttest, ftest, label1, label2, hist1: h1, hist2: h2, vals1, vals2 });
  };

  const numericCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];
  const allColsUnique = dataset?.columns || [];
  const groupValues = groupCol ? [...new Set(data.map(r => String(r[groupCol])).filter(Boolean))] : [];

  // Summary bar data
  const summaryData = result ? [
    { name: result.label1, mean: parseFloat(result.s1.mean.toFixed(4)), std: parseFloat(result.s1.std.toFixed(4)) },
    { name: result.label2, mean: parseFloat(result.s2.mean.toFixed(4)), std: parseFloat(result.s2.std.toFixed(4)) },
  ] : [];

  const [tab, setTab] = useState("results");

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>⚖️ 2-Sample t-Test</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Welch's t-test · F-test for equal variances · Minitab-style report</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <FlaskConical size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>Select a dataset to perform a 2-sample t-test</p>
        </div>
      ) : (
        <>
          {/* Setup */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>⚙️ Test Setup</h3>

            {/* Mode toggle */}
            <div className="flex gap-2 mb-4">
              {[{ id: "columns", label: "Two Columns" }, { id: "groups", label: "One Column + Groups" }].map(m => (
                <button key={m.id} onClick={() => setMode(m.id)}
                  className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                  style={{ background: mode === m.id ? "#1E90FF" : "var(--bg-secondary)", color: mode === m.id ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                  {m.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              {mode === "columns" ? (
                <>
                  <div>
                    <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Group 1 (column)</label>
                    <select value={col1} onChange={e => setCol1(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Group 2 (column)</label>
                    <select value={col2} onChange={e => setCol2(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Group column</label>
                    <select value={groupCol} onChange={e => { setGroupCol(e.target.value); setGroup1Val(""); setGroup2Val(""); }}
                      className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {allColsUnique.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Value column</label>
                    <select value={valueCol} onChange={e => setValueCol(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Group 1 value</label>
                    <select value={group1Val} onChange={e => setGroup1Val(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      <option value="">Select...</option>
                      {groupValues.map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Group 2 value</label>
                    <select value={group2Val} onChange={e => setGroup2Val(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                      <option value="">Select...</option>
                      {groupValues.map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                  </div>
                </>
              )}

              <div>
                <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Alternative hypothesis</label>
                <select value={alternative} onChange={e => setAlternative(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                  style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  <option value="two-sided">Two-sided (μ₁ ≠ μ₂)</option>
                  <option value="greater">Greater (μ₁ &gt; μ₂)</option>
                  <option value="less">Less (μ₁ &lt; μ₂)</option>
                </select>
              </div>
            </div>

            <button onClick={runTest}
              className="btn-primary"
              style={{ background: "linear-gradient(135deg, #1E90FF, #2E5293)" }}>
              <FlaskConical size={14} /> Run t-Test
            </button>
            {error && <div className="flex items-center gap-2 mt-3 text-sm" style={{ color: "#f87171" }}><AlertCircle size={14} /> {error}</div>}
          </div>

          {result && (
            <>
              {/* Tabs */}
              <div className="flex flex-wrap gap-2">
                {[{ id: "results", label: "Results" }, { id: "histograms", label: "Distributions" }, { id: "summary", label: "Descriptive Stats" }].map(t => (
                  <button key={t.id} onClick={() => setTab(t.id)}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                    style={{ background: tab === t.id ? "#1E90FF" : "var(--bg-card)", color: tab === t.id ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                    {t.label}
                  </button>
                ))}
              </div>

              {/* Results tab — Minitab-style */}
              {tab === "results" && (
                <div className="space-y-4">
                  {/* t-Test result */}
                  <div className="glass-card overflow-hidden">
                    <div className="p-4" style={{ borderBottom: "1px solid var(--border)", background: "rgba(30,144,255,0.06)" }}>
                      <h3 className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                        Two-Sample t-Test: {result.label1} vs {result.label2}
                      </h3>
                      <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                        H₀: μ₁ = μ₂ · H₁: {alternative === "two-sided" ? "μ₁ ≠ μ₂" : alternative === "greater" ? "μ₁ > μ₂" : "μ₁ < μ₂"} · Welch's method (unequal variances)
                      </p>
                    </div>

                    {/* Descriptive summary */}
                    <div className="overflow-x-auto p-4">
                      <table className="w-full text-xs mb-4" style={{ borderCollapse: "collapse" }}>
                        <thead>
                          <tr>
                            {["", "N", "Mean", "Std Dev", "Std Error", "Min", "Median", "Max"].map(h => (
                              <th key={h} className="px-3 py-2 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {[{ s: result.s1, label: result.label1, color: COLORS.g1 }, { s: result.s2, label: result.label2, color: COLORS.g2 }].map(({ s, label, color }) => (
                            <tr key={label} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                              <td className="px-3 py-2.5 font-semibold" style={{ color }}>{label}</td>
                              <td className="px-3 py-2.5" style={{ color: "var(--text-primary)" }}>{s.n}</td>
                              <td className="px-3 py-2.5 font-bold" style={{ color: "var(--text-primary)" }}>{s.mean.toFixed(5)}</td>
                              <td className="px-3 py-2.5" style={{ color: "var(--text-secondary)" }}>{s.std.toFixed(5)}</td>
                              <td className="px-3 py-2.5" style={{ color: "var(--text-muted)" }}>{s.se.toFixed(5)}</td>
                              <td className="px-3 py-2.5" style={{ color: "var(--text-muted)" }}>{s.min.toFixed(4)}</td>
                              <td className="px-3 py-2.5" style={{ color: "var(--text-muted)" }}>{s.q2.toFixed(4)}</td>
                              <td className="px-3 py-2.5" style={{ color: "var(--text-muted)" }}>{s.max.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>

                      {/* t-Test stats */}
                      {result.ttest && (
                        <div className="rounded-xl p-4 mb-4" style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}>
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                            {[
                              { label: "Difference of means", value: result.ttest.diffMean.toFixed(5) },
                              { label: "t-statistic", value: result.ttest.t.toFixed(4) },
                              { label: "Degrees of freedom", value: result.ttest.df },
                              {
                                label: "P-value", value: result.ttest.p < 0.001 ? "<0.001" : result.ttest.p.toFixed(4),
                                color: result.ttest.p < 0.05 ? "#10b981" : "#f59e0b"
                              },
                              { label: "95% CI lower", value: result.ttest.ci[0].toFixed(5) },
                              { label: "95% CI upper", value: result.ttest.ci[1].toFixed(5) },
                              { label: "Pooled SE", value: result.ttest.pooledSE.toFixed(5) },
                              {
                                label: "Conclusion", value: result.ttest.p < 0.05 ? "Reject H₀ (p<0.05)" : "Fail to reject H₀",
                                color: result.ttest.p < 0.05 ? "#ef4444" : "#10b981"
                              },
                            ].map(({ label, value, color }) => (
                              <div key={label}>
                                <div className="text-xs mb-0.5" style={{ color: "var(--text-muted)" }}>{label}</div>
                                <div className="font-bold text-sm" style={{ color: color || "var(--text-primary)" }}>{value}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* F-test */}
                  {result.ftest && (
                    <div className="glass-card overflow-hidden">
                      <div className="p-4" style={{ borderBottom: "1px solid var(--border)", background: "rgba(139,92,246,0.04)" }}>
                        <h3 className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>F-Test for Equal Variances</h3>
                        <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>H₀: σ₁² = σ₂² · H₁: σ₁² ≠ σ₂²</p>
                      </div>
                      <div className="p-4">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                          {[
                            { label: `StDev(${result.label1})`, value: result.s1.std.toFixed(5) },
                            { label: `StDev(${result.label2})`, value: result.s2.std.toFixed(5) },
                            { label: "Ratio (s₁/s₂)", value: result.ftest.ratio.toFixed(4) },
                            { label: "F statistic", value: result.ftest.f.toFixed(4) },
                            { label: "df₁", value: result.ftest.df1 },
                            { label: "df₂", value: result.ftest.df2 },
                            {
                              label: "P-value", value: result.ftest.p < 0.001 ? "<0.001" : result.ftest.p.toFixed(4),
                              color: result.ftest.p < 0.05 ? "#ef4444" : "#10b981"
                            },
                            {
                              label: "Conclusion", value: result.ftest.p < 0.05 ? "Variances differ (p<0.05)" : "Variances equal (p≥0.05)",
                              color: result.ftest.p < 0.05 ? "#ef4444" : "#10b981"
                            },
                          ].map(({ label, value, color }) => (
                            <div key={label}>
                              <div style={{ color: "var(--text-muted)" }}>{label}</div>
                              <div className="font-bold mt-0.5" style={{ color: color || "var(--text-primary)" }}>{value}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Mean comparison bar */}
                  <div className="glass-card p-5">
                    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Mean Comparison</h3>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={summaryData} margin={{ top: 10, right: 40, bottom: 10, left: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 12 }} />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                        <Bar dataKey="mean" name="Mean" fill="#1E90FF" radius={[4, 4, 0, 0]}
                          label={{ position: "top", fill: "var(--text-muted)", fontSize: 11, formatter: v => v.toFixed(3) }} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Histograms tab */}
              {tab === "histograms" && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {[
                    { hist: result.hist1, label: result.label1, stats: result.s1, color: COLORS.g1 },
                    { hist: result.hist2, label: result.label2, stats: result.s2, color: COLORS.g2 },
                  ].map(({ hist, label, stats, color }) => (
                    <div key={label} className="glass-card p-5">
                      <h3 className="text-sm font-semibold mb-1" style={{ color }}>{label}</h3>
                      <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>
                        n={stats.n} · mean={stats.mean.toFixed(3)} · σ={stats.std.toFixed(3)}
                      </p>
                      <ResponsiveContainer width="100%" height={240}>
                        <BarChart data={hist} margin={{ top: 5, right: 10, bottom: 20, left: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                          <XAxis dataKey="x" tick={{ fill: "var(--text-muted)", fontSize: 9 }} angle={-30} textAnchor="end" />
                          <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                          <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                          <ReferenceLine x={stats.mean} stroke={color} strokeDasharray="4 4" />
                          <Bar dataKey="count" fill={color} radius={[3, 3, 0, 0]} opacity={0.8} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ))}
                </div>
              )}

              {/* Descriptive stats tab */}
              {tab === "summary" && (
                <div className="glass-card overflow-hidden">
                  <div className="p-4" style={{ borderBottom: "1px solid var(--border)" }}>
                    <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Descriptive Statistics</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          {["Statistic", result.label1, result.label2, "Difference"].map(h => (
                            <th key={h} className="px-4 py-3 text-left" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          { label: "N", v1: result.s1.n, v2: result.s2.n, fmt: v => v },
                          { label: "Mean", v1: result.s1.mean, v2: result.s2.mean, fmt: v => v.toFixed(5), diff: true },
                          { label: "Std Dev", v1: result.s1.std, v2: result.s2.std, fmt: v => v.toFixed(5) },
                          { label: "Variance", v1: result.s1.variance, v2: result.s2.variance, fmt: v => v.toFixed(6) },
                          { label: "SE Mean", v1: result.s1.se, v2: result.s2.se, fmt: v => v.toFixed(5) },
                          { label: "Min", v1: result.s1.min, v2: result.s2.min, fmt: v => v.toFixed(5) },
                          { label: "Q1", v1: result.s1.q1, v2: result.s2.q1, fmt: v => v.toFixed(5) },
                          { label: "Median", v1: result.s1.q2, v2: result.s2.q2, fmt: v => v.toFixed(5) },
                          { label: "Q3", v1: result.s1.q3, v2: result.s2.q3, fmt: v => v.toFixed(5) },
                          { label: "Max", v1: result.s1.max, v2: result.s2.max, fmt: v => v.toFixed(5) },
                        ].map(({ label, v1, v2, fmt, diff }) => (
                          <tr key={label} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                            <td className="px-4 py-2.5 font-medium" style={{ color: "var(--text-primary)" }}>{label}</td>
                            <td className="px-4 py-2.5" style={{ color: COLORS.g1 }}>{fmt(v1)}</td>
                            <td className="px-4 py-2.5" style={{ color: COLORS.g2 }}>{fmt(v2)}</td>
                            <td className="px-4 py-2.5" style={{ color: diff ? (v1 - v2 > 0 ? "#10b981" : "#ef4444") : "var(--text-muted)" }}>
                              {diff ? (v1 - v2 >= 0 ? "+" : "") + (v1 - v2).toFixed(5) : "—"}
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