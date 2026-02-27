import { useState } from "react";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";

const PAGE_SIZE = 50;

export default function DataTable({ data, columns }) {
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState("");

  if (!data || !columns || data.length === 0) {
    return (
      <div className="text-center py-16" style={{ color: "var(--text-muted)" }}>
        No data to display
      </div>
    );
  }

  const filtered = search
    ? data.filter(row =>
        columns.some(col =>
          String(row[col] ?? "").toLowerCase().includes(search.toLowerCase())
        )
      )
    : data;

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageData = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const handleSearch = (v) => {
    setSearch(v);
    setPage(0);
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Search */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: "var(--text-muted)" }} />
          <input
            value={search}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search data..."
            className="w-full pl-9 pr-4 py-2 rounded-lg text-sm outline-none focus:ring-1"
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
              "--tw-ring-color": "var(--accent)"
            }}
          />
        </div>
        <span className="text-sm" style={{ color: "var(--text-muted)" }}>
          {filtered.length.toLocaleString()} rows
        </span>
      </div>

      {/* Table */}
      <div className="overflow-auto rounded-xl" style={{ border: "1px solid var(--border)" }}>
        <table className="w-full data-table" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ width: 48, textAlign: "center" }}>#</th>
              {columns.map(col => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageData.map((row, i) => (
              <tr key={i}>
                <td style={{ color: "var(--text-muted)", textAlign: "center", fontSize: 11 }}>
                  {page * PAGE_SIZE + i + 1}
                </td>
                {columns.map(col => (
                  <td key={col} title={String(row[col] ?? "")}>
                    {row[col] === null || row[col] === undefined || row[col] === ""
                      ? <span style={{ color: "var(--text-muted)", fontStyle: "italic" }}>null</span>
                      : String(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm" style={{ color: "var(--text-muted)" }}>
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="p-2 rounded-lg disabled:opacity-30 transition-colors"
              style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page === totalPages - 1}
              className="p-2 rounded-lg disabled:opacity-30 transition-colors"
              style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}