import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import { Database, Upload, BarChart3, Sparkles, ArrowRight, Clock, FileType, Rows } from "lucide-react";
import StatCard from "../components/ui/StatCard";

const FILE_TYPE_COLORS = {
  csv: "#10b981", xlsx: "#6366f1", xls: "#6366f1",
  json: "#f59e0b", txt: "#8b5cf6"
};

export default function Dashboard() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    base44.entities.Dataset.list("-created_date", 20)
      .then(setDatasets)
      .finally(() => setLoading(false));
  }, []);

  const totalRows = datasets.reduce((s, d) => s + (d.row_count || 0), 0);

  return (
    <div className="p-6 lg:p-8 space-y-8 animate-in">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-1" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
          Welcome to <span className="gradient-text">DataLens</span>
        </h1>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Advanced analytics platform — upload, explore, and unlock insights from your data.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Datasets" value={datasets.length} icon={Database} color="#6366f1" />
        <StatCard label="Total Rows" value={totalRows >= 1000 ? `${(totalRows/1000).toFixed(1)}K` : totalRows} icon={Rows} color="#10b981" />
        <StatCard label="File Types" value={new Set(datasets.map(d => d.file_type)).size} icon={FileType} color="#f59e0b" />
        <StatCard label="Ready" value={datasets.filter(d => d.status === "ready").length} sub="datasets processed" icon={BarChart3} color="#8b5cf6" />
      </div>

      {/* Quick actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { title: "Upload Dataset", desc: "Import CSV, Excel, JSON, or TXT files", icon: Upload, page: "Upload", color: "#6366f1" },
          { title: "Visualize Data", desc: "Build interactive charts and graphs", icon: BarChart3, page: "Visualize", color: "#8b5cf6" },
          { title: "AI Insights", desc: "Get AI-powered analysis and summaries", icon: Sparkles, page: "Insights", color: "#06b6d4" },
        ].map(({ title, desc, icon: Icon, page, color }) => (
          <Link
            key={page}
            to={createPageUrl(page)}
            className="glass-card p-5 flex items-start gap-4 group"
            style={{ textDecoration: "none" }}
          >
            <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
              <Icon size={18} style={{ color }} />
            </div>
            <div className="flex-1">
              <div className="font-semibold text-sm mb-0.5" style={{ color: "var(--text-primary)" }}>{title}</div>
              <div className="text-xs" style={{ color: "var(--text-secondary)" }}>{desc}</div>
            </div>
            <ArrowRight size={16} className="mt-1 opacity-0 group-hover:opacity-100 transition-opacity" style={{ color }} />
          </Link>
        ))}
      </div>

      {/* Recent datasets */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>Recent Datasets</h2>
          <Link to={createPageUrl("Upload")} className="btn-secondary text-xs py-1.5 px-3" style={{ textDecoration: "none" }}>
            Upload new
          </Link>
        </div>

        {loading ? (
          <div className="space-y-3">
            {[1,2,3].map(i => (
              <div key={i} className="glass-card p-4 animate-pulse" style={{ height: 64 }} />
            ))}
          </div>
        ) : datasets.length === 0 ? (
          <div className="glass-card p-12 text-center">
            <Database size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
            <p className="text-sm font-medium mb-1" style={{ color: "var(--text-secondary)" }}>No datasets yet</p>
            <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Upload your first dataset to get started</p>
            <Link to={createPageUrl("Upload")} className="btn-primary text-sm" style={{ textDecoration: "none", display: "inline-flex" }}>
              <Upload size={14} /> Upload Dataset
            </Link>
          </div>
        ) : (
          <div className="space-y-2">
            {datasets.map(ds => (
              <Link
                key={ds.id}
                to={`${createPageUrl("Explorer")}?id=${ds.id}`}
                className="glass-card p-4 flex items-center gap-4 group"
                style={{ textDecoration: "none" }}
              >
                <div className="w-9 h-9 rounded-lg flex items-center justify-center text-xs font-bold uppercase flex-shrink-0"
                  style={{
                    background: `${FILE_TYPE_COLORS[ds.file_type] || "#6366f1"}18`,
                    color: FILE_TYPE_COLORS[ds.file_type] || "#6366f1",
                    border: `1px solid ${FILE_TYPE_COLORS[ds.file_type] || "#6366f1"}30`
                  }}>
                  {ds.file_type}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate" style={{ color: "var(--text-primary)" }}>{ds.name}</div>
                  <div className="text-xs flex items-center gap-3 mt-0.5" style={{ color: "var(--text-muted)" }}>
                    {ds.row_count && <span>{ds.row_count.toLocaleString()} rows</span>}
                    {ds.column_count && <span>{ds.column_count} cols</span>}
                  </div>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  <span className={`tag ${ds.status === "ready" ? "tag-green" : ds.status === "error" ? "tag-red" : "tag-yellow"}`}>
                    {ds.status}
                  </span>
                  <div className="flex items-center gap-1 text-xs" style={{ color: "var(--text-muted)" }}>
                    <Clock size={11} />
                    {new Date(ds.created_date).toLocaleDateString()}
                  </div>
                  <ArrowRight size={15} className="opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: "var(--accent-light)" }} />
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}