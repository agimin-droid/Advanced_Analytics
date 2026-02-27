import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import { Upload as UploadIcon, FileText, CheckCircle, AlertCircle, X, CloudUpload, Database, Trash2, Clock } from "lucide-react";

const ACCEPTED = ".csv,.txt,.xlsx,.xls,.json";
const MAX_MB = 50;

function parseCSV(text, delimiter = ",") {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return { columns: [], data: [] };
  const headers = lines[0].split(delimiter).map(h => h.replace(/^"|"$/g, "").trim());
  const data = lines.slice(1).map(line => {
    const vals = line.split(delimiter).map(v => v.replace(/^"|"$/g, "").trim());
    const row = {};
    headers.forEach((h, i) => { row[h] = vals[i] ?? ""; });
    return row;
  });
  return { columns: headers, data };
}

function parseJSON(text) {
  const parsed = JSON.parse(text);
  const arr = Array.isArray(parsed) ? parsed : [parsed];
  const columns = arr.length > 0 ? Object.keys(arr[0]) : [];
  return { columns, data: arr };
}

export default function DataHandling() {
  const navigate = useNavigate();
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [datasets, setDatasets] = useState([]);

  useEffect(() => {
    base44.entities.Dataset.list("-created_date", 20).then(setDatasets);
  }, []);

  const handleFile = (f) => {
    if (!f) return;
    if (f.size > MAX_MB * 1024 * 1024) { setError(`File too large. Max ${MAX_MB}MB.`); return; }
    setFile(f);
    setName(f.name.replace(/\.[^.]+$/, ""));
    setError("");
  };

  const getFileType = (filename) => {
    const ext = filename.split(".").pop().toLowerCase();
    return ["csv", "txt", "xlsx", "xls", "json"].includes(ext) ? ext : "csv";
  };

  const handleUpload = async () => {
    if (!file || !name.trim()) return;
    setStatus("uploading");
    setProgress(10);
    const ext = getFileType(file.name);
    let parsedData = null;
    let columns = [];
    if (["csv", "txt", "json"].includes(ext)) {
      const text = await file.text();
      setProgress(30);
      if (ext === "json") { const r = parseJSON(text); parsedData = r.data; columns = r.columns; }
      else { const r = parseCSV(text, ext === "txt" ? "\t" : ","); parsedData = r.data; columns = r.columns; }
    }
    setProgress(50);
    const { file_url } = await base44.integrations.Core.UploadFile({ file });
    setProgress(65);
    let preview_data = null;
    if (parsedData && parsedData.length > 0) {
      const blob = new Blob([JSON.stringify(parsedData.slice(0, 500))], { type: "application/json" });
      const pf = new File([blob], "preview.json", { type: "application/json" });
      const { file_url: pUrl } = await base44.integrations.Core.UploadFile({ file: pf });
      preview_data = pUrl;
    }
    setProgress(80);
    const ds = await base44.entities.Dataset.create({
      name: name.trim(), description: description.trim(), file_url, file_name: file.name, file_type: ext,
      file_size: file.size, row_count: parsedData?.length || null, column_count: columns.length || null,
      columns, preview_data, status: "ready",
    });
    setDatasets(prev => [ds, ...prev]);
    setProgress(100);
    setStatus("success");
    setTimeout(() => { setFile(null); setName(""); setDescription(""); setStatus("idle"); setProgress(0); }, 2000);
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this dataset?")) return;
    await base44.entities.Dataset.delete(id);
    setDatasets(prev => prev.filter(d => d.id !== id));
  };

  const fmt = (bytes) => bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`;

  const TYPE_COLORS = { csv: "#10b981", xlsx: "#1E90FF", xls: "#1E90FF", json: "#f59e0b", txt: "#8b5cf6" };

  return (
    <div className="p-6 lg:p-8 space-y-8 animate-in">
      <div>
        <h1 className="text-2xl font-bold mb-1" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>📥 Data Handling</h1>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>Import CSV, TXT (tab-delimited), Excel (XLS/XLSX), and JSON files. All modules will share the loaded datasets.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload */}
        <div>
          <h2 className="text-sm font-semibold mb-4 uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>Import New Dataset</h2>
          <div
            onClick={() => !file && inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
            className="rounded-2xl flex flex-col items-center justify-center mb-4 cursor-pointer transition-all"
            style={{ border: `2px dashed ${dragging || file ? "#1E90FF" : "var(--border)"}`, background: dragging ? "rgba(30,144,255,0.08)" : file ? "rgba(30,144,255,0.04)" : "var(--bg-card)", minHeight: 160, padding: 32 }}
          >
            <input ref={inputRef} type="file" accept={ACCEPTED} className="hidden" onChange={e => handleFile(e.target.files[0])} />
            {file ? (
              <div className="text-center">
                <FileText size={28} className="mx-auto mb-2" style={{ color: "#1E90FF" }} />
                <p className="font-semibold text-sm mb-1" style={{ color: "var(--text-primary)" }}>{file.name}</p>
                <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>{fmt(file.size)}</p>
                <button onClick={e => { e.stopPropagation(); setFile(null); setName(""); }} className="text-xs flex items-center gap-1 mx-auto" style={{ color: "var(--text-muted)" }}>
                  <X size={12} /> Remove
                </button>
              </div>
            ) : (
              <div className="text-center">
                <CloudUpload size={28} className="mx-auto mb-2" style={{ color: "var(--text-muted)" }} />
                <p className="font-semibold text-sm mb-1" style={{ color: "var(--text-primary)" }}>Drop file or click to browse</p>
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>CSV · TXT · XLS · XLSX · JSON (max {MAX_MB}MB)</p>
              </div>
            )}
          </div>

          <div className="space-y-3 mb-4">
            <input value={name} onChange={e => setName(e.target.value)} placeholder="Dataset name *"
              className="w-full px-4 py-2.5 rounded-xl text-sm outline-none"
              style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
            <input value={description} onChange={e => setDescription(e.target.value)} placeholder="Description (optional)"
              className="w-full px-4 py-2.5 rounded-xl text-sm outline-none"
              style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
          </div>

          {error && (
            <div className="flex items-center gap-2 mb-3 px-4 py-3 rounded-xl" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)" }}>
              <AlertCircle size={14} style={{ color: "#f87171" }} />
              <span className="text-sm" style={{ color: "#f87171" }}>{error}</span>
            </div>
          )}

          {status === "uploading" && (
            <div className="mb-3">
              <div className="flex justify-between text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>
                <span>Uploading...</span><span>{progress}%</span>
              </div>
              <div className="w-full h-1.5 rounded-full" style={{ background: "var(--border)" }}>
                <div className="h-full rounded-full transition-all duration-500" style={{ width: `${progress}%`, background: "linear-gradient(90deg, #1E90FF, #2E5293)" }} />
              </div>
            </div>
          )}

          {status === "success" && (
            <div className="flex items-center gap-2 mb-3 px-4 py-3 rounded-xl" style={{ background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)" }}>
              <CheckCircle size={14} style={{ color: "#34d399" }} />
              <span className="text-sm" style={{ color: "#34d399" }}>Dataset uploaded successfully!</span>
            </div>
          )}

          <button onClick={handleUpload} disabled={!file || !name.trim() || status === "uploading" || status === "success"}
            className="btn-primary w-full justify-center disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none"
            style={{ background: "linear-gradient(135deg, #1E90FF, #2E5293)" }}>
            <UploadIcon size={14} />
            {status === "uploading" ? "Uploading..." : "Import Dataset"}
          </button>
        </div>

        {/* Loaded datasets */}
        <div>
          <h2 className="text-sm font-semibold mb-4 uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
            Loaded Datasets ({datasets.length})
          </h2>
          {datasets.length === 0 ? (
            <div className="glass-card p-10 text-center">
              <Database size={32} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>No datasets imported yet</p>
            </div>
          ) : (
            <div className="space-y-2">
              {datasets.map(ds => (
                <div key={ds.id} className="glass-card p-4 flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg flex items-center justify-center text-xs font-bold uppercase flex-shrink-0"
                    style={{ background: `${TYPE_COLORS[ds.file_type] || "#1E90FF"}18`, color: TYPE_COLORS[ds.file_type] || "#1E90FF", border: `1px solid ${TYPE_COLORS[ds.file_type] || "#1E90FF"}30` }}>
                    {ds.file_type}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm truncate" style={{ color: "var(--text-primary)" }}>{ds.name}</div>
                    <div className="text-xs flex items-center gap-2 mt-0.5" style={{ color: "var(--text-muted)" }}>
                      {ds.row_count && <span>{ds.row_count.toLocaleString()} rows</span>}
                      {ds.column_count && <span>{ds.column_count} cols</span>}
                      <span className="flex items-center gap-1"><Clock size={10} /> {new Date(ds.created_date).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <span className={`tag text-xs ${ds.status === "ready" ? "tag-green" : "tag-yellow"}`}>{ds.status}</span>
                  <button onClick={() => handleDelete(ds.id)} className="p-1.5 rounded-lg" style={{ color: "var(--text-muted)" }}>
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}