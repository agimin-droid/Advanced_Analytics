import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import {
  LayoutDashboard,
  Upload,
  Table,
  BarChart3,
  Sparkles,
  Shield,
  Menu,
  X,
  Database,
  ChevronRight,
  LogOut
} from "lucide-react";

const navItems = [
  { name: "Dashboard", icon: LayoutDashboard, page: "Dashboard" },
  { name: "Upload", icon: Upload, page: "Upload" },
  { name: "Explorer", icon: Table, page: "Explorer" },
  { name: "Visualize", icon: BarChart3, page: "Visualize" },
  { name: "AI Insights", icon: Sparkles, page: "Insights" },
];

export default function Layout({ children, currentPageName }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [user, setUser] = useState(null);

  useEffect(() => {
    base44.auth.me().then(setUser).catch(() => {});
  }, []);

  const isAdmin = user?.role === "admin";

  return (
    <div className="min-h-screen flex" style={{ background: "var(--bg-primary)" }}>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 h-full z-30 flex flex-col transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
          lg:translate-x-0 lg:static lg:z-auto`}
        style={{
          width: 240,
          background: "var(--bg-secondary)",
          borderRight: "1px solid var(--border)",
          flexShrink: 0
        }}
      >
        {/* Logo */}
        <div className="p-6 pb-5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
              <Database size={18} color="white" />
            </div>
            <div>
              <div className="font-bold text-base" style={{ color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
                DataLens
              </div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>Analytics Platform</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 space-y-1">
          <div className="mb-3 px-3">
            <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Navigation
            </span>
          </div>
          {navItems.map(({ name, icon: Icon, page }) => {
            const active = currentPageName === page;
            return (
              <Link
                key={page}
                to={createPageUrl(page)}
                onClick={() => setSidebarOpen(false)}
                className={`nav-item flex items-center gap-3 px-3 py-2.5 text-sm font-medium ${active ? "active" : ""}`}
                style={{ color: active ? "var(--accent-light)" : "var(--text-secondary)", textDecoration: "none" }}
              >
                <Icon size={17} />
                <span>{name}</span>
                {active && <ChevronRight size={14} className="ml-auto" />}
              </Link>
            );
          })}

          {isAdmin && (
            <>
              <div className="my-4 px-3">
                <div style={{ height: 1, background: "var(--border)" }} />
              </div>
              <div className="mb-2 px-3">
                <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
                  Admin
                </span>
              </div>
              <Link
                to={createPageUrl("Admin")}
                onClick={() => setSidebarOpen(false)}
                className={`nav-item flex items-center gap-3 px-3 py-2.5 text-sm font-medium ${currentPageName === "Admin" ? "active" : ""}`}
                style={{ color: currentPageName === "Admin" ? "var(--accent-light)" : "var(--text-secondary)", textDecoration: "none" }}
              >
                <Shield size={17} />
                <span>Access Control</span>
                {currentPageName === "Admin" && <ChevronRight size={14} className="ml-auto" />}
              </Link>
            </>
          )}
        </nav>

        {/* User section */}
        {user && (
          <div className="p-4 m-3 rounded-xl" style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
                style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)", color: "white" }}>
                {user.full_name?.[0]?.toUpperCase() || "U"}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                  {user.full_name || "User"}
                </div>
                <div className="text-xs capitalize" style={{ color: "var(--text-muted)" }}>{user.role || "viewer"}</div>
              </div>
              <button
                onClick={() => base44.auth.logout()}
                className="p-1.5 rounded-lg transition-colors"
                style={{ color: "var(--text-muted)" }}
                title="Logout"
              >
                <LogOut size={14} />
              </button>
            </div>
          </div>
        )}
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile topbar */}
        <header className="lg:hidden flex items-center justify-between px-4 py-4"
          style={{ background: "var(--bg-secondary)", borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
              <Database size={16} color="white" />
            </div>
            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>DataLens</span>
          </div>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg"
            style={{ color: "var(--text-secondary)" }}
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </header>

        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  );
}