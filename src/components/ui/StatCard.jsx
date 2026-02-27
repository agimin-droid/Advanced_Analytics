export default function StatCard({ label, value, sub, icon: Icon, color = "#6366f1" }) {
  return (
    <div className="glass-card p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>
            {label}
          </p>
          <p className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
            {value}
          </p>
          {sub && (
            <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>{sub}</p>
          )}
        </div>
        {Icon && (
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
            <Icon size={18} style={{ color }} />
          </div>
        )}
      </div>
    </div>
  );
}