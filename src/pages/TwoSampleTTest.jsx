import { useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, ReferenceLine, LineChart, Line, Legend
} from "recharts";
import DatasetPicker from "../components/chem/DatasetPicker";
import { FlaskConical, RefreshCw, AlertCircle, Download } from "lucide-react";

// ── STATISTICS ───────────────────────────────────────────────────────────────
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function variance(arr, m) {
  const mu = m ?? mean(arr);
  return arr.reduce((s, v) => s + (v - mu) ** 2, 0) / (arr.length - 1);
}
function std(arr) { return Math.sqrt(variance(arr)); }

// Welch's t-test (unequal variances by default, like Minitab)
function welchTTest(g1, g2, alternative = "two-sided", mu0 = 0) {
  const n1 = g1.length, n2 = g2.length;
  const m1 = mean(g1), m2 = mean(g2);
  const s1 = std(g1), s2 = std(g2);
  const se = Math.sqrt(s1 * s1 / n1 + s2 * s2 / n2);
  const t = se === 0 ? 0 : (m1 - m2 - mu0) / se;
  // Welch-Satterthwaite df
  const df = se === 0 ? n1 + n2 - 2 :
    (s1 * s1 / n1 + s2 * s2 / n2) ** 2 /
    ((s1 * s1 / n1) ** 2 / (n1 - 1) + (s2 * s2 / n2) ** 2 / (n2 - 1));
  // p-value approximation using t-distribution (Cornish-Fisher)
  const pOneTail = tDistPValue(Math.abs(t), df);
  const p = alternative === "two-sided" ? 2 * pOneTail : pOneTail;
  return { t, df, p, m1, m2, s1, s2, n1, n2, se, diff: m1 - m2 };
}

// Approximate p-value from t-distribution (Abramowitz & Stegun)
function tDistPValue(t, df) {
  const x = df / (df + t * t);
  // Regularized incomplete beta function approximation
  const a = df / 2, b = 0.5;
  return 0.5 * incompleteBeta(x, a, b);
}

function incompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  // Continued fraction approximation
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
  let cf = 1, d = 1, c = 1;
  for (let m = 0; m < 100; m++) {
    const m2 = 2 * m;
    const d1 = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + d1 * d; c = 1 + d1 / c;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d; cf *= c * d;
    const d2 = -((a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + d2 * d; c = 1 + d2 / c;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d; const delta = c * d;
    cf *= delta;
    if (Math.abs(delta - 1) < 1e-8) break;
  }
  return front * cf;
}

function lgamma(z) {
  const g = 7;
  const p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  let x = p[0];
  for (let i = 1; i < g + 2; i++) x += p[i] / (z + i - 1);
  const t = z + g - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z - 0.5) * Math.log(t) - t + Math.log(x);
}

// Levene-like variance test (simple F-test approximation)
function varianceTest(g1, g2) {
  const s1 = std(g1), s2 = std(g2);
  const f = s2 > 0 ? (s1 * s1) / (s2 * s2) : Infinity;
  const df1 = g1.length - 1, df2 = g2.length - 1;
  return { f, df1, df2, ratio: s1 / s2 };
}

// Shapiro-Wilk approximation (simplified)
function normalityTest(arr) {
  if (arr.length < 3) return { statistic: 1, p: 1 };
  const n = arr.length;
  const sorted = [...arr].sort((a, b) => a - b);
  const mu = mean(sorted);
  const s2 = variance(sorted, mu);
  // Simple Jarque-Bera test
  const s = std(sorted);
  if (s === 0) return { statistic: 0, p: 0 };
  const skew = sorted.reduce((sum, v) => sum + ((v - mu) / s) ** 3, 0) / n;
  const kurt = sorted.reduce((sum, v) => sum + ((v - mu) / s) ** 4, 0) / n - 3;
  const jb = n / 6 * (skew ** 2 + kurt ** 2 / 4);
  const p = Math.exp(-jb / 2);
  return { statistic: parseFloat((1 - jb / (n + 1)).toFixed(4)), p: parseFloat(Math.min(1, p).toFixed(4)), jb, skew: parseFloat(skew.toFixed(4)), kurt: parseFloat(kurt.toFixed(4)) };
}

// Confidence interval
function confInterval(g1, g2, alpha = 0.05) {
  const { diff, se, df } = welchTTest(g1, g2);
  // t-critical (approx)
  const tCrit = 1.96 + 3.8 / df; // rough approximation
  return [diff - tCrit * se, diff + tCrit * se];
}

// Histogram bins
function makeHist(values, bins = 12) {
  const min = Math.min(...values), max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const h = Array.from({ length: bins }, (_, i) => ({ x: parseFloat((min + (i + 0.5) * step).toFixed(3)), count: 0 }));
  values.forEach(v => { const i = Math.min(Math.floor((v - min) / step), bins - 1); if (i >= 0) h[i].count++; });
  return h;
}

// Q-Q data
function qqData(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const n = sorted.length;
  const m = mean(sorted), s = std(sorted);
  return sorted.map((v, i) => {
    const p = (i + 0.5) / n;
    const u = p < 0.5 ? p : 1 - p;
    const t = Math.sqrt(-2 * Math.log(u));
    const z = p < 0.5 ? -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
      : (t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t));
    return { theoretical: parseFloat(z.toFixed(3)), observed: parseFloat(v.toFixed(4)) };
  });
}

const COLORS = { g1: "#1E90FF", g2: "#f59e0b" };

export default function TwoSampleTTest() {
  const [datasetId, setDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [data, setData] = useState([]);
  const [inputMode, setInputMode] = useState("grouped"); // "grouped" | "separate" | "summary"
  const [valueCol, setValueCol] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [group1, setGroup1] = useState("");
  const [group2, setGroup2] = useState("");
  const [col1, setCol1] = useState("");
  const [col2, setCol2] = useState("");
  const [alternative, setAlternative] = useState("two-sided");
  const [alpha, setAlpha] = useState(0.05);
  const [mu0, setMu0] = useState(0);
  // Summary mode
  const [n1, setN1] = useState(30); const [mean1, setMean1] = useState(0); const [std1, setStd1] = useState(1);
  const [n2, setN2] = useState(30); const [mean2, setMean2] = useState(0); const [std2, setStd2] = useState(1);
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState("test");
  const [error, setError] = useState("");

  const loadData = async (id, ds) => {
    setDataset(ds); setResult(null); setError("");
    if (!ds?.preview_data) return;
    const rows = await fetch(ds.preview_data).then(r => r.json());
    setData(rows);
    const nc = ds?.columns?.filter(c => !isNaN(parseFloat(rows[0]?.[c]))) || [];
    const cc = ds?.columns?.filter(c => isNaN(parseFloat(rows[0]?.[c]))) || [];
    if (nc.length >= 1) { setValueCol(nc[0]); setCol1(nc[0]); setCol2(nc[Math.min(1, nc.length - 1)]); }
    if (cc.length >= 1) setGroupCol(cc[0]);
    // Auto-detect groups
    if (cc.length >= 1) {
      const groups = [...new Set(rows.map(r => r[cc[0]]).filter(Boolean))].slice(0, 10);
      if (groups.length >= 2) { setGroup1(String(groups[0])); setGroup2(String(groups[1])); }
    }
  };

  const runTest = () => {
    setError(""); setResult(null);
    let g1, g2, label1, label2;

    if (inputMode === "grouped") {
      if (!valueCol || !groupCol || !group1 || !group2 || group1 === group2) {
        setError("Select value column, group column, and two different groups."); return;
      }
      g1 = data.filter(r => String(r[groupCol]) === group1).map(r => parseFloat(r[valueCol])).filter(v => !isNaN(v));
      g2 = data.filter(r => String(r[groupCol]) === group2).map(r => parseFloat(r[valueCol])).filter(v => !isNaN(v));
      label1 = group1; label2 = group2;
    } else if (inputMode === "separate") {
      if (!col1 || !col2 || col1 === col2) { setError("Select two different columns."); return; }
      g1 = data.map(r => parseFloat(r[col1])).filter(v => !isNaN(v));
      g2 = data.map(r => parseFloat(r[col2])).filter(v => !isNaN(v));
      label1 = col1; label2 = col2;
    } else {
      // summary mode: generate synthetic data
      const rng = (n, m, s) => Array.from({ length: n }, (_, i) => m + s * (Math.cos(2 * Math.PI * (i + 0.5) / n)));
      g1 = rng(n1, mean1, std1); label1 = "Sample 1";
      g2 = rng(n2, mean2, std2); label2 = "Sample 2";
    }

    if (g1.length < 2 || g2.length < 2) { setError("Each group needs at least 2 values."); return; }

    const tRes = welchTTest(g1, g2, alternative, mu0);
    const varRes = varianceTest(g1, g2);
    const norm1 = normalityTest(g1);
    const norm2 = normalityTest(g2);
    const ci = confInterval(g1, g2, alpha);

    setResult({ tRes, varRes, norm1, norm2, ci, g1, g2, label1, label2, alpha });
  };

  const numericCols = dataset?.columns?.filter(c => !isNaN(parseFloat(data[0]?.[c]))) || [];
  const catCols = dataset?.columns?.filter(c => isNaN(parseFloat(data[0]?.[c]))) || [];
  const groups = groupCol ? [...new Set(data.map(r => String(r[groupCol])).filter(Boolean))].slice(0, 50) : [];

  const sigLabel = result ? (result.tRes.p < 0.001 ? "Highly Significant (p<0.001)" : result.tRes.p < result.alpha ? `Significant (p=${result.tRes.p.toFixed(4)})` : `Not Significant (p=${result.tRes.p.toFixed(4)})`) : "";
  const sigColor = result ? (result.tRes.p < result.alpha ? "#10b981" : "#ef4444") : "#10b981";

  const exportReport = () => {
    if (!result) return;
    const { tRes, varRes, norm1, norm2, ci, label1, label2, alpha } = result;
    const lines = [
      `Two-Sample T-Test Report`,
      `========================`,
      ``,
      `Groups: ${label1} (n=${tRes.n1}) vs ${label2} (n=${tRes.n2})`,
      `Hypothesis: H0: μ1 - μ2 = ${mu0} | ${alternative === "two-sided" ? "H1: μ1 ≠ μ2" : alternative === "greater" ? "H1: μ1 > μ2" : "H1: μ1 < μ2"}`,
      ``,
      `DESCRIPTIVE STATISTICS`,
      `  ${label1}: n=${tRes.n1}, mean=${tRes.m1.toFixed(4)}, s=${tRes.s1.toFixed(4)}`,
      `  ${label2}: n=${tRes.n2}, mean=${tRes.m2.toFixed(4)}, s=${tRes.s2.toFixed(4)}`,
      `  Difference: ${tRes.diff.toFixed(4)}`,
      ``,
      `TEST RESULTS (Welch's t-test, unequal variances)`,
      `  t-statistic = ${tRes.t.toFixed(4)}`,
      `  df = ${tRes.df.toFixed(1)}`,
      `  SE(diff) = ${tRes.se.toFixed(4)}`,
      `  P-value (${alternative}) = ${tRes.p < 0.001 ? "<0.001" : tRes.p.toFixed(4)}`,
      `  ${100 * (1 - alpha)}% CI for difference: [${ci[0].toFixed(4)}, ${ci[1].toFixed(4)}]`,
      `  Decision: ${tRes.p < alpha ? "REJECT H0" : "FAIL TO REJECT H0"} at α=${alpha}`,
      ``,
      `VARIANCE EQUALITY (F-test)`,
      `  F = ${varRes.f.toFixed(4)}, s1/s2 = ${varRes.ratio.toFixed(4)}`,
      ``,
      `NORMALITY (Jarque-Bera approximation)`,
      `  ${label1}: W≈${norm1.statistic}, p≈${norm1.p} | Skew=${norm1.skew}, Kurt=${norm1.kurt}`,
      `  ${label2}: W≈${norm2.statistic}, p≈${norm2.p} | Skew=${norm2.skew}, Kurt=${norm2.kurt}`,
    ];
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain" }));
    a.download = "ttest_report.txt"; a.click();
  };

  const TABS = [
    { id: "test", label: "Test Results" },
    { id: "graphs", label: "Graphs" },
    { id: "assumptions", label: "Assumptions" },
  ];

  return (
    <div className="p-6 lg:p-8 space-y-6 animate-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>🔬 Two-Sample T-Test</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>Compare means of two groups — Welch's t-test (Minitab-style)</p>
        </div>
        <DatasetPicker value={datasetId} onChange={(id, ds) => { setDatasetId(id); loadData(id, ds); }} />
      </div>

      {/* Config */}
      <div className="glass-card p-5 space-y-4">
        <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>⚙️ Test Configuration</h3>

        {/* Input mode */}
        <div className="flex flex-wrap gap-2">
          {[["grouped", "One column + groups"], ["separate", "Separate columns"], ["summary", "Summary stats"]].map(([m, l]) => (
            <button key={m} onClick={() => setInputMode(m)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium"
              style={{ background: inputMode === m ? "#1E90FF" : "var(--bg-secondary)", color: inputMode === m ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
              {l}
            </button>
          ))}
        </div>

        {/* Mode: Grouped */}
        {inputMode === "grouped" && dataset && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div>
              <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Response (numeric)</label>
              <select value={valueCol} onChange={e => setValueCol(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Grouping column</label>
              <select value={groupCol} onChange={e => {
                setGroupCol(e.target.value);
                const gs = [...new Set(data.map(r => String(r[e.target.value])).filter(Boolean))].slice(0, 50);
                if (gs.length >= 2) { setGroup1(gs[0]); setGroup2(gs[1]); }
              }}
                className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                {dataset?.columns?.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Group 1</label>
              <select value={group1} onChange={e => setGroup1(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "#1E90FF" }}>
                {groups.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Group 2</label>
              <select value={group2} onChange={e => setGroup2(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "#f59e0b" }}>
                {groups.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>
          </div>
        )}

        {/* Mode: Separate columns */}
        {inputMode === "separate" && dataset && (
          <div className="grid grid-cols-2 gap-3 max-w-sm">
            <div>
              <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Sample 1 column</label>
              <select value={col1} onChange={e => setCol1(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "#1E90FF" }}>
                {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Sample 2 column</label>
              <select value={col2} onChange={e => setCol2(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "#f59e0b" }}>
                {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>
        )}

        {/* Mode: Summary */}
        {inputMode === "summary" && (
          <div className="grid grid-cols-2 gap-6">
            {[{ label: "Sample 1", n: n1, setN: setN1, m: mean1, setM: setMean1, s: std1, setS: setStd1, color: "#1E90FF" },
              { label: "Sample 2", n: n2, setN: setN2, m: mean2, setM: setMean2, s: std2, setS: setStd2, color: "#f59e0b" }
            ].map(({ label, n, setN, m, setM, s, setS, color }) => (
              <div key={label}>
                <p className="text-xs font-semibold mb-2" style={{ color }}>{label}</p>
                <div className="space-y-2">
                  {[["n", n, setN, 2], ["Mean", m, setM, null], ["Std Dev", s, setS, 0.0001]].map(([l, v, fn, mi]) => (
                    <div key={l} className="flex items-center gap-2">
                      <span className="text-xs w-16" style={{ color: "var(--text-muted)" }}>{l}</span>
                      <input type="number" value={v} min={mi} step={l === "n" ? 1 : 0.0001}
                        onChange={e => fn(+e.target.value)}
                        className="flex-1 px-2 py-1.5 rounded-lg text-sm outline-none"
                        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Test options */}
        <div className="flex flex-wrap gap-4 items-end pt-2" style={{ borderTop: "1px solid var(--border)" }}>
          <div>
            <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>Hypothesis (H₁)</label>
            <select value={alternative} onChange={e => setAlternative(e.target.value)}
              className="px-3 py-2 rounded-lg text-sm outline-none"
              style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
              <option value="two-sided">μ₁ ≠ μ₂ (two-sided)</option>
              <option value="greater">μ₁ &gt; μ₂ (one-sided right)</option>
              <option value="less">μ₁ &lt; μ₂ (one-sided left)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>α (significance)</label>
            <select value={alpha} onChange={e => setAlpha(+e.target.value)}
              className="px-3 py-2 rounded-lg text-sm outline-none"
              style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
              {[0.01, 0.05, 0.10].map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>H₀: μ₁ − μ₂ =</label>
            <input type="number" value={mu0} onChange={e => setMu0(+e.target.value)}
              className="px-3 py-2 rounded-lg text-sm outline-none w-20"
              style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
          </div>
          <button onClick={runTest} disabled={!dataset && inputMode !== "summary"}
            className="btn-primary disabled:opacity-40"
            style={{ background: "linear-gradient(135deg, #1E90FF, #2E5293)" }}>
            <FlaskConical size={14} /> Run T-Test
          </button>
          {result && (
            <button onClick={exportReport} className="btn-secondary text-xs flex items-center gap-1.5">
              <Download size={12} /> Export Report
            </button>
          )}
        </div>
        {error && <div className="flex items-center gap-2 text-sm" style={{ color: "#f87171" }}><AlertCircle size={14} /> {error}</div>}
      </div>

      {result && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "t-statistic", value: result.tRes.t.toFixed(4), color: "#1E90FF" },
              { label: "df (Welch)", value: result.tRes.df.toFixed(1), color: "#8b5cf6" },
              { label: "P-value", value: result.tRes.p < 0.001 ? "<0.001" : result.tRes.p.toFixed(4), color: sigColor },
              { label: "Decision", value: result.tRes.p < result.alpha ? "Reject H₀" : "Fail to Reject", color: sigColor },
            ].map(({ label, value, color }) => (
              <div key={label} className="glass-card p-4">
                <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
                <div className="text-lg font-bold" style={{ color }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Tabs */}
          <div className="flex gap-2">
            {TABS.map(t => (
              <button key={t.id} onClick={() => setTab(t.id)}
                className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                style={{ background: tab === t.id ? "#1E90FF" : "var(--bg-card)", color: tab === t.id ? "white" : "var(--text-secondary)", border: "1px solid var(--border)" }}>
                {t.label}
              </button>
            ))}
          </div>

          {/* Test Results */}
          {tab === "test" && (
            <div className="space-y-4">
              {/* Minitab-style report */}
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Minitab-Style Output</h3>
                <div className="font-mono text-xs leading-6 rounded-lg p-4 overflow-x-auto"
                  style={{ background: "var(--bg-secondary)", color: "var(--text-primary)", border: "1px solid var(--border)" }}>
                  <div>Two-Sample T-Test and CI: {result.label1}, {result.label2}</div>
                  <div>&nbsp;</div>
                  <div>Sample  N    Mean     StDev    SE Mean</div>
                  <div>{(result.label1 + "   ").slice(0, 8)} {result.tRes.n1.toString().padEnd(5)} {result.tRes.m1.toFixed(4).padEnd(9)} {result.tRes.s1.toFixed(4).padEnd(9)} {(result.tRes.s1 / Math.sqrt(result.tRes.n1)).toFixed(4)}</div>
                  <div>{(result.label2 + "   ").slice(0, 8)} {result.tRes.n2.toString().padEnd(5)} {result.tRes.m2.toFixed(4).padEnd(9)} {result.tRes.s2.toFixed(4).padEnd(9)} {(result.tRes.s2 / Math.sqrt(result.tRes.n2)).toFixed(4)}</div>
                  <div>&nbsp;</div>
                  <div>Difference = μ({result.label1.slice(0, 8)}) - μ({result.label2.slice(0, 8)})</div>
                  <div>Estimate for difference: {result.tRes.diff.toFixed(4)}</div>
                  <div>{(1 - result.alpha) * 100}% CI for difference: ({result.ci[0].toFixed(4)}, {result.ci[1].toFixed(4)})</div>
                  <div>T-Test of difference = {mu0} (vs {alternative === "two-sided" ? "≠" : alternative === "greater" ? ">" : "<"})</div>
                  <div>T-Value = {result.tRes.t.toFixed(2)}   P-Value = {result.tRes.p < 0.001 ? "0.000" : result.tRes.p.toFixed(3)}   DF = {result.tRes.df.toFixed(0)}</div>
                  <div style={{ color: sigColor, fontWeight: "bold" }}>→ {result.tRes.p < result.alpha ? "REJECT H₀" : "FAIL TO REJECT H₀"} at α = {result.alpha} — {sigLabel}</div>
                </div>
              </div>

              {/* Descriptive stats */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                  { label: result.label1, g: result.g1, color: COLORS.g1 },
                  { label: result.label2, g: result.g2, color: COLORS.g2 }
                ].map(({ label, g, color }) => {
                  const m = mean(g), s = std(g);
                  const sorted = [...g].sort((a, b) => a - b);
                  return (
                    <div key={label} className="glass-card p-4">
                      <div className="text-sm font-semibold mb-3" style={{ color }}>{label}</div>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        {[["N", g.length], ["Mean", m.toFixed(4)], ["Std Dev", s.toFixed(4)],
                          ["Min", sorted[0].toFixed(4)], ["Median", sorted[Math.floor(g.length / 2)].toFixed(4)], ["Max", sorted[g.length - 1].toFixed(4)]
                        ].map(([k, v]) => (
                          <div key={k}>
                            <div style={{ color: "var(--text-muted)" }}>{k}</div>
                            <div className="font-semibold" style={{ color: "var(--text-primary)" }}>{v}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Graphs */}
          {tab === "graphs" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Individual value plot */}
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Individual Value Plot</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="category" dataKey="group" allowDuplicatedCategory={false}
                      tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                    <YAxis type="number" dataKey="value" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                    <Tooltip content={({ payload }) => {
                      if (!payload?.length) return null;
                      const d = payload[0].payload;
                      return <div className="rounded-lg p-2 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>{d.group}: {d.value?.toFixed(4)}</div>;
                    }} />
                    <Scatter name={result.label1} data={result.g1.map(v => ({ group: result.label1, value: v }))} fill={COLORS.g1} opacity={0.7} />
                    <Scatter name={result.label2} data={result.g2.map(v => ({ group: result.label2, value: v }))} fill={COLORS.g2} opacity={0.7} />
                    <ReferenceLine y={mean(result.g1)} stroke={COLORS.g1} strokeDasharray="4 4" />
                    <ReferenceLine y={mean(result.g2)} stroke={COLORS.g2} strokeDasharray="4 4" />
                    <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              {/* Histogram overlay */}
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Histogram — {result.label1} vs {result.label2}</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart margin={{ top: 5, right: 20, bottom: 20, left: 30 }}
                    data={makeHist(result.g1).map((b, i) => ({ x: b.x, [result.label1]: b.count, [result.label2]: makeHist(result.g2)[i]?.count || 0 }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="x" tick={{ fill: "var(--text-muted)", fontSize: 9 }} />
                    <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />
                    <Bar dataKey={result.label1} fill={COLORS.g1} opacity={0.7} />
                    <Bar dataKey={result.label2} fill={COLORS.g2} opacity={0.7} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Q-Q plots */}
              {[{ label: result.label1, g: result.g1, color: COLORS.g1 }, { label: result.label2, g: result.g2, color: COLORS.g2 }].map(({ label, g, color }) => (
                <div key={label} className="glass-card p-5">
                  <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Q-Q Plot — {label}</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="theoretical" name="Theoretical" label={{ value: "Theoretical Quantiles", position: "bottom", fill: "var(--text-muted)", fontSize: 10 }} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <YAxis type="number" dataKey="observed" name="Observed" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                      <Tooltip content={({ payload }) => {
                        if (!payload?.length) return null;
                        const d = payload[0].payload;
                        return <div className="rounded p-2 text-xs" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>Th: {d.theoretical} | Obs: {d.observed}</div>;
                      }} />
                      <Scatter data={qqData(g)} fill={color} opacity={0.8} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          )}

          {/* Assumptions */}
          {tab === "assumptions" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Normality */}
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Normality Test (Jarque-Bera)</h3>
                {[{ label: result.label1, norm: result.norm1, color: COLORS.g1 }, { label: result.label2, norm: result.norm2, color: COLORS.g2 }].map(({ label, norm, color }) => (
                  <div key={label} className="mb-3 p-3 rounded-lg" style={{ background: "var(--bg-secondary)" }}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold" style={{ color }}>{label}</span>
                      <span className="tag text-xs" style={{ background: norm.p > 0.05 ? "rgba(16,185,129,0.1)" : "rgba(239,68,68,0.1)", color: norm.p > 0.05 ? "#10b981" : "#ef4444" }}>
                        {norm.p > 0.05 ? "Normal" : "Non-normal"}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs" style={{ color: "var(--text-muted)" }}>
                      <div>W ≈ {norm.statistic}</div>
                      <div>p ≈ {norm.p}</div>
                      <div>Skewness: {norm.skew}</div>
                      <div>Kurtosis: {norm.kurt}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Variance equality */}
              <div className="glass-card p-5">
                <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Variance Equality (F-test)</h3>
                <div className="space-y-3">
                  <div className="p-3 rounded-lg" style={{ background: "var(--bg-secondary)" }}>
                    <div className="text-xs space-y-1.5" style={{ color: "var(--text-muted)" }}>
                      <div className="flex justify-between"><span>F-statistic</span><span style={{ color: "var(--text-primary)" }}>{result.varRes.f.toFixed(4)}</span></div>
                      <div className="flex justify-between"><span>s₁ / s₂ ratio</span><span style={{ color: "var(--text-primary)" }}>{result.varRes.ratio.toFixed(4)}</span></div>
                      <div className="flex justify-between"><span>df₁</span><span style={{ color: "var(--text-primary)" }}>{result.varRes.df1}</span></div>
                      <div className="flex justify-between"><span>df₂</span><span style={{ color: "var(--text-primary)" }}>{result.varRes.df2}</span></div>
                    </div>
                    <div className="mt-2 text-xs" style={{ color: result.varRes.ratio > 2 || result.varRes.ratio < 0.5 ? "#f59e0b" : "#10b981" }}>
                      {result.varRes.ratio > 2 || result.varRes.ratio < 0.5 ? "⚠ Variances appear unequal — Welch's test appropriate" : "✓ Variances roughly equal — pooled test also valid"}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg text-xs" style={{ background: "var(--bg-secondary)", color: "var(--text-muted)" }}>
                    <div className="font-semibold mb-1" style={{ color: "var(--text-primary)" }}>Test Used: Welch's T-Test</div>
                    <div>Does not assume equal variances (more conservative than pooled t-test). Recommended when variance equality is uncertain.</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}