import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import { BarChart3, Database, ChevronDown } from "lucide-react";
import ChartBuilder from "../components/charts/ChartBuilder";

export default function Visualize() {
  const [datasets, setDatasets] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [parsedData, setParsedData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showPicker, setShowPicker] = useState(false);

  useEffect(() => {
    base44.entities.Dataset.filter({ status: "ready" }, "-created_date", 50).then(list => {
      setDatasets(list);
      const params = new URLSearchParams(window.location.search);
      const id = params.get("id");
      if (id) {
        setSelectedId(id);
      } else if (list.length > 0) {
        setSelectedId(list[0].id);
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

  return (
    <div className="p-6 lg:p-8 animate-in space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
            Visualize
          </h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
            Build interactive charts and graphs from your data.
          </p>
        </div>

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
              {datasets.length === 0 ? (
                <div className="p-4 text-sm text-center" style={{ color: "var(--text-muted)" }}>No datasets available</div>
              ) : (
                datasets.map(ds => (
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
                  </button>
                ))
              )}
            </div>
          )}
        </div>
      </div>

      {loading ? (
        <div className="glass-card animate-pulse" style={{ height: 400 }} />
      ) : !dataset ? (
        <div className="glass-card p-16 text-center">
          <BarChart3 size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm font-medium mb-1" style={{ color: "var(--text-secondary)" }}>No dataset selected</p>
          <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Upload a dataset to start visualizing.</p>
          <Link to={createPageUrl("Upload")} className="btn-primary" style={{ textDecoration: "none", display: "inline-flex" }}>
            Upload Dataset
          </Link>
        </div>
      ) : parsedData.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            No preview data available for this dataset type. Please use CSV, TXT, or JSON files for visualization.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="glass-card p-4 flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.2)" }}>
              <Database size={14} style={{ color: "var(--accent-light)" }} />
            </div>
            <div>
              <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>{dataset.name}</div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                {parsedData.length.toLocaleString()} rows · {dataset.columns?.length} columns (showing up to 500)
              </div>
            </div>
          </div>

          <ChartBuilder data={parsedData} columns={dataset.columns || []} />
        </div>
      )}
    </div>
  );
}