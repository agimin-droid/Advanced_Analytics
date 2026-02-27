import { useMemo } from "react";
import { BarChart2, Hash, Type, AlertCircle } from "lucide-react";

function inferType(values) {
  const nonNull = values.filter(v => v !== null && v !== undefined && v !== "");
  if (nonNull.length === 0) return "empty";
  const nums = nonNull.filter(v => !isNaN(Number(v)) && v !== "");
  if (nums.length / nonNull.length > 0.85) return "numeric";
  return "categorical";
}

function numericStats(values) {
  const nums = values.filter(v => v !== null && v !== undefined && v !== "" && !isNaN(Number(v))).map(Number);
  if (nums.length === 0) return null;
  const sorted = [...nums].sort((a, b) => a - b);
  const sum = nums.reduce((a, b) => a + b, 0);
  const mean = sum / nums.length;
  const median = sorted[Math.floor(sorted.length / 2)];
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const variance = nums.reduce((acc, v) => acc + (v - mean) ** 2, 0) / nums.length;
  const std = Math.sqrt(variance);
  return { mean, median, min, max, std, count: nums.length };
}

function categoricalStats(values) {
  const nonNull = values.filter(v => v !== null && v !== undefined && v !== "");
  const counts = {};
  nonNull.forEach(v => { counts[String(v)] = (counts[String(v)] || 0) + 1; });
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return { unique: sorted.length, top: sorted.slice(0, 5) };
}

export default function ColumnStats({ data, columns }) {
  const stats = useMemo(() => {
    if (!data || !columns) return [];
    return columns.map(col => {
      const values = data.map(row => row[col]);
      const nullCount = values.filter(v => v === null || v === undefined || v === "").length;
      const type = inferType(values);
      return {
        col,
        type,
        nullCount,
        nullPct: Math.round((nullCount / values.length) * 100),
        numeric: type === "numeric" ? numericStats(values) : null,
        categorical: type === "categorical" ? categoricalStats(values) : null,
      };
    });
  }, [data, columns]);

  const fmt = (n) => {
    if (n === null || n === undefined) return "—";
    if (Math.abs(n) >= 1000000) return (n / 1000000).toFixed(2) + "M";
    if (Math.abs(n) >= 1000) return (n / 1000).toFixed(2) + "K";
    return Number(n).toFixed(2);
  };

  return (
    <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))" }}>
      {stats.map(s => (
        <div key={s.col} className="glass-card p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{
                background: s.type === "numeric" ? "rgba(99,102,241,0.15)" : "rgba(16,185,129,0.15)",
                color: s.type === "numeric" ? "#818cf8" : "#34d399"
              }}>
              {s.type === "numeric" ? <Hash size={13} /> : <Type size={13} />}
            </div>
            <div>
              <div className="text-sm font-semibold truncate max-w-[180px]" style={{ color: "var(--text-primary)" }}>
                {s.col}
              </div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                {s.type}
              </div>
            </div>
          </div>

          {s.nullCount > 0 && (
            <div className="flex items-center gap-1.5 mb-3 px-2.5 py-1.5 rounded-lg"
              style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.15)" }}>
              <AlertCircle size={11} style={{ color: "#fbbf24" }} />
              <span className="text-xs" style={{ color: "#fbbf24" }}>
                {s.nullCount} missing ({s.nullPct}%)
              </span>
            </div>
          )}

          {s.type === "numeric" && s.numeric && (
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Mean", value: fmt(s.numeric.mean) },
                { label: "Median", value: fmt(s.numeric.median) },
                { label: "Min", value: fmt(s.numeric.min) },
                { label: "Max", value: fmt(s.numeric.max) },
                { label: "Std Dev", value: fmt(s.numeric.std) },
                { label: "Count", value: s.numeric.count.toLocaleString() },
              ].map(({ label, value }) => (
                <div key={label} className="px-2.5 py-2 rounded-lg"
                  style={{ background: "var(--bg-secondary)" }}>
                  <div className="text-xs" style={{ color: "var(--text-muted)" }}>{label}</div>
                  <div className="text-sm font-semibold mt-0.5" style={{ color: "var(--text-primary)" }}>{value}</div>
                </div>
              ))}
            </div>
          )}

          {s.type === "categorical" && s.categorical && (
            <div>
              <div className="text-xs mb-2" style={{ color: "var(--text-muted)" }}>
                {s.categorical.unique} unique values
              </div>
              <div className="space-y-1.5">
                {s.categorical.top.map(([val, count]) => (
                  <div key={val} className="flex items-center gap-2">
                    <div className="text-xs truncate flex-1" style={{ color: "var(--text-secondary)" }}>{val || "(empty)"}</div>
                    <div className="text-xs font-semibold" style={{ color: "var(--accent-light)" }}>{count}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}