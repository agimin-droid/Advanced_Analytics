import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import { Table, BarChart2, Sparkles, Database, ChevronDown, Trash2 } from "lucide-react";
import DataTable from "../components/dataset/DataTable";
import ColumnStats from "../components/dataset/ColumnStats";

const TABS = [
  { id: "table", label: "Data Table", icon: Table },
  { id: "stats", label: "Column Stats", icon: BarChart2 },
];

export default function Explorer() {
  const [datasets, setDatasets] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [parsedData, setParsedData] = useState([]);
  const [tab, setTab] = useState("table");
  const [loading, setLoading] = useState(true);
  const [showPicker, setShowPicker] = useState(false);

  useEffect(() => {
    base44.entities.Dataset.list("-created_date", 50).then(list => {
      setDatasets(list);
      const params = new URLSearchParams(window.location.search);
      const id = params.get("id");
      const ready = list.filter(d => d.status === "ready");
      if (id) {
        setSelectedId(id);
      } else if (ready.length > 0) {
        setSelectedId(ready[0].id);
      }
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    const ds = datasets.find(d => d.id === selectedId);
    if (ds) {
      setDataset(ds);
      if (ds.preview_data) {
        setParsedData(JSON.parse(ds.preview_data));
      } else {
        setParsedData([]);
      }
    }
  }, [selectedId, datasets]);

  const handleDelete = async () => {
    if (!dataset) return;
    if (!window.confirm(`Delete "${dataset.name}"? This cannot be undone.`)) return;
    await base44.entities.Dataset.delete(dataset.id);
    setDatasets(prev => prev.filter(d => d.id !== dataset.id));
    setSelectedId(null);
    setDataset(null);
    setParsedData([]);
  };

  if (loading) {
    return (
      <div className="p-8 animate-in">
        <div className="space-y-3">
          {[1,2,3].map(i => (
            <div key={i} className="glass-card animate-pulse" style={{ height: 56 }} />
          ))}
        </div>
      </div>
    );
  }

  const readyDatasets = datasets.filter(d => d.status === "ready");

  return (
    <div className="p-6 lg:p-8 animate-in space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
            Data Explorer
          </h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
            Browse, search, and analyze your dataset columns.
          </p>
        </div>

        {/* Dataset picker */}
        <div className="relative">
          <button
            onClick={() => setShowPicker(!showPicker)}
            className="btn-secondary flex items-center gap-2"
          >
            <Database size={14} />
            <span className="max-w-[180px] truncate">{dataset?.name || "Select dataset"}</span>
            <ChevronDown size={14} />
          </button>
          {showPicker && (
            <div className="absolute right-0 top-full mt-2 w-72 rounded-xl z-10 overflow-hidden"
              style={{ background: "var(--bg-card)", border: "1px solid var(--border)", boxShadow: "0 20px 60px rgba(0,0,0,0.5)" }}>
              {readyDatasets.length === 0 ? (
                <div className="p-4 text-sm text-center" style={{ color: "var(--text-muted)" }}>No datasets available</div>
              ) : (
                readyDatasets.map(ds => (
                  <button
                    key={ds.id}
                    onClick={() => { setSelectedId(ds.id); setShowPicker(false); }}
                    className="w-full text-left px-4 py-3 text-sm flex items-center gap-3 transition-colors"
                    style={{
                      background: ds.id === selectedId ? "rgba(99,102,241,0.1)" : "transparent",
                      color: ds.id === selectedId ? "var(--accent-light)" : "var(--text-secondary)",
                      borderBottom: "1px solid var(--border)"
                    }}
                  >
                    <Database size={13} />
                    <span className="flex-1 truncate">{ds.name}</span>
                    <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                      {ds.row_count?.toLocaleString()} rows
                    </span>
                  </button>
                ))
              )}
            </div>
          )}
        </div>
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <Database size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm font-medium mb-1" style={{ color: "var(--text-secondary)" }}>No dataset selected</p>
          <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
            {readyDatasets.length === 0 ? "Upload a dataset to get started." : "Select a dataset above."}
          </p>
          <Link to={createPageUrl("Upload")} className="btn-primary" style={{ textDecoration: "none", display: "inline-flex" }}>
            Upload Dataset
          </Link>
        </div>
      ) : (
        <>
          {/* Dataset info bar */}
          <div className="glass-card p-4 flex flex-wrap items-center gap-4 justify-between">
            <div className="flex flex-wrap gap-4">
              {[
                { label: "Rows", value: dataset.row_count?.toLocaleString() || "—" },
                { label: "Columns", value: dataset.column_count || "—" },
                { label: "Type", value: dataset.file_type?.toUpperCase() || "—" },
                { label: "File", value: dataset.file_name || "—" },
              ].map(({ label, value }) => (
                <div key={label}>
                  <div className="text-xs" style={{ color: "var(--text-muted)" }}>{label}</div>
                  <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>{value}</div>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <Link
                to={`${createPageUrl("Insights")}?id=${dataset.id}`}
                className="btn-secondary text-xs py-1.5 px-3"
                style={{ textDecoration: "none" }}
              >
                <Sparkles size={12} /> AI Insights
              </Link>
              <Link
                to={`${createPageUrl("Visualize")}?id=${dataset.id}`}
                className="btn-secondary text-xs py-1.5 px-3"
                style={{ textDecoration: "none" }}
              >
                <BarChart2 size={12} /> Visualize
              </Link>
              <button
                onClick={handleDelete}
                className="btn-secondary text-xs py-1.5 px-3"
                style={{ color: "#f87171", borderColor: "rgba(239,68,68,0.3)" }}
              >
                <Trash2 size={12} />
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 p-1 rounded-xl" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", width: "fit-content" }}>
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
                style={{
                  background: tab === id ? "var(--accent)" : "transparent",
                  color: tab === id ? "white" : "var(--text-secondary)",
                }}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>

          {/* Content */}
          {parsedData.length > 0 ? (
            tab === "table" ? (
              <DataTable data={parsedData} columns={dataset.columns || []} />
            ) : (
              <ColumnStats data={parsedData} columns={dataset.columns || []} />
            )
          ) : (
            <div className="glass-card p-10 text-center">
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                Preview data not available for Excel files. Use AI Insights or Visualize for analysis.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}