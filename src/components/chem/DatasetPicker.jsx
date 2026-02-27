import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import { Database, ChevronDown, Upload } from "lucide-react";

export default function DatasetPicker({ value, onChange, className = "" }) {
  const [datasets, setDatasets] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const selected = datasets.find(d => d.id === value);

  useEffect(() => {
    base44.entities.Dataset.filter({ status: "ready" }, "-created_date", 50)
      .then(list => {
        setDatasets(list);
        if (!value && list.length > 0) onChange(list[0].id, list[0]);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    const close = (e) => { if (!e.target.closest(".dataset-picker-root")) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  if (loading) return <div className="btn-secondary opacity-50 text-sm">Loading...</div>;

  if (datasets.length === 0) {
    return (
      <Link to={createPageUrl("DataHandling")} className="btn-primary text-sm" style={{ textDecoration: "none" }}>
        <Upload size={13} /> Import Data First
      </Link>
    );
  }

  return (
    <div className={`relative dataset-picker-root ${className}`}>
      <button onClick={() => setOpen(!open)} className="btn-secondary flex items-center gap-2 text-sm">
        <Database size={14} />
        <span className="max-w-[200px] truncate">{selected?.name || "Select dataset"}</span>
        <ChevronDown size={13} />
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-2 w-80 rounded-xl z-50 overflow-hidden"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border)", boxShadow: "0 20px 60px rgba(0,0,0,0.6)" }}>
          {datasets.map(ds => (
            <button
              key={ds.id}
              onClick={() => { onChange(ds.id, ds); setOpen(false); }}
              className="w-full text-left px-4 py-3 text-sm flex items-center gap-3 transition-colors"
              style={{
                background: ds.id === value ? "rgba(30,144,255,0.1)" : "transparent",
                color: ds.id === value ? "#4da3ff" : "var(--text-secondary)",
                borderBottom: "1px solid var(--border)"
              }}
            >
              <Database size={13} style={{ flexShrink: 0 }} />
              <span className="flex-1 truncate">{ds.name}</span>
              <span className="text-xs flex-shrink-0" style={{ color: "var(--text-muted)" }}>
                {ds.row_count?.toLocaleString()} × {ds.column_count}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}