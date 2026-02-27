import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import { Upload as UploadIcon, FileText, CheckCircle, AlertCircle, X, CloudUpload } from "lucide-react";

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

export default function Upload() {
  const navigate = useNavigate();
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [status, setStatus] = useState("idle"); // idle | uploading | success | error
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);

  const handleFile = (f) => {
    if (!f) return;
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`File too large. Max ${MAX_MB}MB.`);
      return;
    }
    setFile(f);
    setName(f.name.replace(/\.[^.]+$/, ""));
    setError("");
  };

  const getFileType = (filename) => {
    const ext = filename.split(".").pop().toLowerCase();
    if (["csv", "txt", "xlsx", "xls", "json"].includes(ext)) return ext;
    return "csv";
  };

  const handleUpload = async () => {
    if (!file || !name.trim()) return;
    setStatus("uploading");
    setProgress(10);

    const ext = getFileType(file.name);
    let parsedData = null;
    let columns = [];

    // Parse client-side for csv/txt/json
    if (["csv", "txt", "json"].includes(ext)) {
      const text = await file.text();
      setProgress(30);
      if (ext === "json") {
        const r = parseJSON(text);
        parsedData = r.data;
        columns = r.columns;
      } else {
        const delim = ext === "txt" ? "\t" : ",";
        const r = parseCSV(text, delim);
        parsedData = r.data;
        columns = r.columns;
      }
    }

    setProgress(50);

    // Upload original file
    const { file_url } = await base44.integrations.Core.UploadFile({ file });
    setProgress(65);

    // Upload preview data (first 500 rows) as a separate JSON file
    let preview_data_url = null;
    if (parsedData && parsedData.length > 0) {
      const previewBlob = new Blob(
        [JSON.stringify(parsedData.slice(0, 500))],
        { type: "application/json" }
      );
      const previewFile = new File([previewBlob], "preview.json", { type: "application/json" });
      const { file_url: pUrl } = await base44.integrations.Core.UploadFile({ file: previewFile });
      preview_data_url = pUrl;
    }
    setProgress(80);

    await base44.entities.Dataset.create({
      name: name.trim(),
      description: description.trim(),
      file_url,
      file_name: file.name,
      file_type: ext,
      file_size: file.size,
      row_count: parsedData?.length || null,
      column_count: columns.length || null,
      columns,
      preview_data: preview_data_url,
      status: "ready",
    });

    setProgress(100);
    setStatus("success");
    setTimeout(() => navigate(createPageUrl("Dashboard")), 1500);
  };

  const fmt = (bytes) => {
    if (bytes > 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(0)} KB`;
  };

  return (
    <div className="p-6 lg:p-8 max-w-2xl mx-auto animate-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
          Upload Dataset
        </h1>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Supports CSV, TXT (tab-delimited), Excel (XLS/XLSX), and JSON files up to {MAX_MB}MB.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onClick={() => !file && inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => {
          e.preventDefault();
          setDragging(false);
          handleFile(e.dataTransfer.files[0]);
        }}
        className="rounded-2xl flex flex-col items-center justify-center transition-all duration-200 mb-5 cursor-pointer"
        style={{
          border: `2px dashed ${dragging ? "var(--accent)" : file ? "var(--accent)" : "var(--border)"}`,
          background: dragging ? "var(--accent-glow)" : file ? "rgba(99,102,241,0.05)" : "var(--bg-card)",
          minHeight: 200,
          padding: 40,
        }}
      >
        <input ref={inputRef} type="file" accept={ACCEPTED} className="hidden" onChange={e => handleFile(e.target.files[0])} />

        {file ? (
          <div className="text-center">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-3"
              style={{ background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.3)" }}>
              <FileText size={24} style={{ color: "var(--accent-light)" }} />
            </div>
            <p className="font-semibold text-sm mb-1" style={{ color: "var(--text-primary)" }}>{file.name}</p>
            <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>{fmt(file.size)}</p>
            <button
              onClick={e => { e.stopPropagation(); setFile(null); setName(""); }}
              className="text-xs flex items-center gap-1 mx-auto"
              style={{ color: "var(--text-muted)" }}
            >
              <X size={12} /> Remove file
            </button>
          </div>
        ) : (
          <div className="text-center">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
              style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}>
              <CloudUpload size={24} style={{ color: "var(--text-muted)" }} />
            </div>
            <p className="font-semibold text-sm mb-1" style={{ color: "var(--text-primary)" }}>
              Drop file here or click to browse
            </p>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>CSV · TXT · XLS · XLSX · JSON</p>
          </div>
        )}
      </div>

      {/* Form */}
      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
            Dataset Name *
          </label>
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="e.g. Sales Q4 2024"
            className="w-full px-4 py-2.5 rounded-xl text-sm outline-none focus:ring-1"
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
          />
        </div>
        <div>
          <label className="block text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
            Description (optional)
          </label>
          <textarea
            value={description}
            onChange={e => setDescription(e.target.value)}
            placeholder="Brief description of this dataset..."
            rows={3}
            className="w-full px-4 py-2.5 rounded-xl text-sm outline-none focus:ring-1 resize-none"
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
            }}
          />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 mb-4 px-4 py-3 rounded-xl"
          style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)" }}>
          <AlertCircle size={15} style={{ color: "#f87171" }} />
          <span className="text-sm" style={{ color: "#f87171" }}>{error}</span>
        </div>
      )}

      {/* Progress */}
      {status === "uploading" && (
        <div className="mb-4">
          <div className="flex justify-between text-xs mb-2" style={{ color: "var(--text-muted)" }}>
            <span>Uploading...</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full h-1.5 rounded-full" style={{ background: "var(--border)" }}>
            <div className="h-full rounded-full transition-all duration-500"
              style={{ width: `${progress}%`, background: "linear-gradient(90deg, #6366f1, #8b5cf6)" }} />
          </div>
        </div>
      )}

      {/* Success */}
      {status === "success" && (
        <div className="flex items-center gap-2 mb-4 px-4 py-3 rounded-xl"
          style={{ background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)" }}>
          <CheckCircle size={15} style={{ color: "#34d399" }} />
          <span className="text-sm" style={{ color: "#34d399" }}>Dataset uploaded successfully! Redirecting...</span>
        </div>
      )}

      <button
        onClick={handleUpload}
        disabled={!file || !name.trim() || status === "uploading" || status === "success"}
        className="btn-primary w-full justify-center disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none"
      >
        <UploadIcon size={15} />
        {status === "uploading" ? "Uploading..." : "Upload Dataset"}
      </button>
    </div>
  );
}