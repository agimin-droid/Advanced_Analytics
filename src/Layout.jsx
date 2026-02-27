import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import {
  Database, Upload, TrendingUp, Activity, BarChart2, GitBranch,
  Sliders, Shield, Menu, X, ChevronRight, LogOut, FlaskConical
} from "lucide-react";

const navItems = [
  { name: "Dashboard", icon: Database, page: "Dashboard", label: "Home" },
  { name: "Data Handling", icon: Upload, page: "DataHandling", label: "Data Import" },
  { name: "PCA", icon: TrendingUp, page: "PCA", label: "Principal Component Analysis" },
  { name: "Quality Control", icon: Activity, page: "QualityControl", label: "PCA Monitoring" },
  { name: "MLR & DoE", icon: GitBranch, page: "MLRDoE", label: "Regression & Design" },
  { name: "Univariate", icon: BarChart2, page: "Univariate", label: "Statistical Analysis" },
  { name: "Bivariate", icon: GitBranch, page: "Bivariate", label: "Correlation Analysis" },
  { name: "Preprocessing", icon: Sliders, page: "Preprocessing", label: "Transformations" },
  { name: "2-Sample t-Test", icon: FlaskConical, page: "TwoSampleTTest", label: "t-Test & F-Test" },
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
      {sidebarOpen && (
        <div className="fixed inset-0 z-20 bg-black/60 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      <aside
        className={`fixed top-0 left-0 h-full z-30 flex flex-col transition-transform duration-300 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} lg:translate-x-0 lg:static lg:z-auto`}
        style={{ width: 250, background: "var(--bg-secondary)", borderRight: "1px solid var(--border)", flexShrink: 0 }}
      >
        {/* Logo */}
        <div className="p-5 pb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, #1E90FF, #2E5293)" }}>
              <FlaskConical size={18} color="white" />
            </div>
            <div>
              <div className="font-bold text-sm" style={{ color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
                ChemometricSolutions
              </div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>Demo — Workshop Como 2026</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto">
          <div className="mb-2 px-3">
            <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Modules
            </span>
          </div>
          {navItems.map(({ name, icon: Icon, page, label }) => {
            const active = currentPageName === page;
            return (
              <Link
                key={page}
                to={createPageUrl(page)}
                onClick={() => setSidebarOpen(false)}
                className={`nav-item flex items-center gap-3 px-3 py-2.5 text-sm font-medium ${active ? "active" : ""}`}
                style={{ color: active ? "var(--accent-light)" : "var(--text-secondary)", textDecoration: "none" }}
                title={label}
              >
                <Icon size={16} />
                <span className="flex-1">{name}</span>
                {active && <ChevronRight size={13} className="ml-auto" />}
              </Link>
            );
          })}

          {isAdmin && (
            <>
              <div className="my-3 px-3"><div style={{ height: 1, background: "var(--border)" }} /></div>
              <div className="mb-2 px-3">
                <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>Admin</span>
              </div>
              <Link
                to={createPageUrl("Admin")}
                onClick={() => setSidebarOpen(false)}
                className={`nav-item flex items-center gap-3 px-3 py-2.5 text-sm font-medium ${currentPageName === "Admin" ? "active" : ""}`}
                style={{ color: currentPageName === "Admin" ? "var(--accent-light)" : "var(--text-secondary)", textDecoration: "none" }}
              >
                <Shield size={16} />
                <span>Access Control</span>
                {currentPageName === "Admin" && <ChevronRight size={13} className="ml-auto" />}
              </Link>
            </>
          )}
        </nav>

        {/* User */}
        {user && (
          <div className="p-3 m-3 rounded-xl" style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}>
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
                style={{ background: "linear-gradient(135deg, #1E90FF, #2E5293)", color: "white" }}>
                {user.full_name?.[0]?.toUpperCase() || "U"}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium truncate" style={{ color: "var(--text-primary)" }}>{user.full_name || "User"}</div>
                <div className="text-xs capitalize" style={{ color: "var(--text-muted)" }}>{user.role || "viewer"}</div>
              </div>
              <button onClick={() => base44.auth.logout()} className="p-1.5 rounded-lg" style={{ color: "var(--text-muted)" }} title="Logout">
                <LogOut size={13} />
              </button>
            </div>
          </div>
        )}
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile topbar */}
        <header className="lg:hidden flex items-center justify-between px-4 py-3"
          style={{ background: "var(--bg-secondary)", borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, #1E90FF, #2E5293)" }}>
              <FlaskConical size={14} color="white" />
            </div>
            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>ChemometricSolutions</span>
          </div>
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 rounded-lg" style={{ color: "var(--text-secondary)" }}>
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </header>

        <main className="flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  );
}