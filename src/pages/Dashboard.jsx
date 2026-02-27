import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import {
  Upload, TrendingUp, Activity, GitBranch, BarChart2, Sliders, FlaskConical,
  ArrowRight, Database, Lock, CheckCircle, Mail, Globe
} from "lucide-react";

const MODULES = [
  { icon: Upload, page: "DataHandling", name: "Data Handling", desc: "Import CSV, Excel, TXT datasets", available: true, color: "#1E90FF" },
  { icon: TrendingUp, page: "PCA", name: "PCA Analysis", desc: "Principal Component Analysis (NIPALS)", available: true, color: "#10b981" },
  { icon: Activity, page: "QualityControl", name: "Quality Control", desc: "PCA Monitoring — T² & Q statistics", available: true, color: "#f59e0b" },
  { icon: GitBranch, page: "MLRDoE", name: "MLR & DoE", desc: "Multiple Linear Regression & Design of Experiments", available: true, color: "#8b5cf6" },
  { icon: BarChart2, page: "Univariate", name: "Univariate Analysis", desc: "Statistical tests, distributions, outliers", available: true, color: "#06b6d4" },
  { icon: GitBranch, page: "Bivariate", name: "Bivariate Analysis", desc: "Correlation analysis & scatter plots", available: true, color: "#ec4899" },
  { icon: Sliders, page: "Preprocessing", name: "Preprocessing", desc: "SNV, derivatives, scaling, transformations", available: true, color: "#f97316" },
  { icon: FlaskConical, page: "TwoSampleTTest", name: "2-Sample T-Test", desc: "Welch's t-test — Minitab-style output & graphs", available: true, color: "#84cc16" },
];

const FULL_VERSION = [
  "Multi-Response DoE — Pareto Optimization",
  "Classification — PLS-DA, LDA, QDA",
  "PLS Calibration — Quantitative Analysis",
  "GA Variable Selection",
  "Mixture Design — Simplex DoE",
  "Advanced Spectral Processing",
];

export default function Dashboard() {
  const [datasets, setDatasets] = useState([]);

  useEffect(() => {
    base44.entities.Dataset.list("-created_date", 5).then(setDatasets).catch(() => {});
  }, []);

  return (
    <div className="p-6 lg:p-8 space-y-10 animate-in">
      {/* Hero */}
      <div className="text-center py-8">
        <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
          style={{ background: "linear-gradient(135deg, #1E90FF, #2E5293)", boxShadow: "0 0 40px rgba(30,144,255,0.3)" }}>
          <FlaskConical size={32} color="white" />
        </div>
        <h1 className="text-4xl font-bold mb-2" style={{ letterSpacing: "-0.04em" }}>
          <span style={{
            background: "linear-gradient(45deg, #2E5293, #1E90FF, #4da3ff)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text"
          }}>ChemometricSolutions</span>
        </h1>
        <p className="text-lg mb-1" style={{ color: "var(--text-secondary)" }}>DEMO VERSION — Workshop Como 2026</p>
        <p className="text-sm max-w-xl mx-auto" style={{ color: "var(--text-muted)" }}>
          7 core chemometric modules for multivariate data analysis. Upload your dataset and start exploring.
        </p>

        {datasets.length === 0 && (
          <Link to={createPageUrl("DataHandling")}
            className="btn-primary mt-6 inline-flex"
            style={{ textDecoration: "none", background: "linear-gradient(135deg, #1E90FF, #2E5293)" }}>
            <Upload size={15} /> Import your data
          </Link>
        )}
        {datasets.length > 0 && (
          <div className="mt-6 flex items-center justify-center gap-2 text-sm" style={{ color: "#10b981" }}>
            <CheckCircle size={16} />
            {datasets.length} dataset{datasets.length > 1 ? "s" : ""} ready — select a module below
          </div>
        )}
      </div>

      {/* Demo info banner */}
      <div className="rounded-2xl p-5" style={{ background: "rgba(30,144,255,0.06)", border: "1px solid rgba(30,144,255,0.2)" }}>
        <div className="flex flex-wrap gap-3">
          {["Data Handling & Import", "PCA Analysis", "Quality Control (PCA Monitoring)", "MLR & DoE (Single Response)", "Univariate Analysis", "Bivariate Analysis", "Preprocessing & Transformations", "Two-Sample T-Test (Welch's)"].map(m => (
            <span key={m} className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full"
              style={{ background: "rgba(30,144,255,0.1)", color: "#4da3ff" }}>
              <CheckCircle size={11} /> {m}
            </span>
          ))}
        </div>
      </div>

      {/* Modules grid */}
      <div>
        <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
          🚀 Available Modules
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {MODULES.map(({ icon: Icon, page, name, desc, color }) => (
            <Link
              key={page}
              to={createPageUrl(page)}
              className="glass-card p-5 flex flex-col gap-3 group"
              style={{ textDecoration: "none" }}
            >
              <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
                <Icon size={18} style={{ color }} />
              </div>
              <div>
                <div className="font-semibold text-sm mb-0.5 flex items-center justify-between"
                  style={{ color: "var(--text-primary)" }}>
                  {name}
                  <ArrowRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity" style={{ color }} />
                </div>
                <div className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>{desc}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Full version locked */}
      <div>
        <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
          🔒 Full Version Only
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {FULL_VERSION.map(m => (
            <div key={m} className="glass-card p-4 flex items-center gap-3 opacity-50">
              <Lock size={14} style={{ color: "var(--text-muted)" }} />
              <span className="text-sm" style={{ color: "var(--text-secondary)" }}>{m}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Contact CTA */}
      <div className="rounded-2xl p-8 text-center"
        style={{ background: "linear-gradient(135deg, rgba(30,144,255,0.08) 0%, rgba(46,82,147,0.08) 100%)", border: "1px solid rgba(30,144,255,0.2)" }}>
        <h3 className="text-lg font-bold mb-2" style={{ color: "#1E90FF" }}>Need the Full Version?</h3>
        <p className="text-sm mb-4 max-w-md mx-auto" style={{ color: "var(--text-secondary)" }}>
          12+ advanced modules including Multi-Response DoE, Classification, PLS Calibration, Genetic Algorithms, and Mixture Designs.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <a href="mailto:chemometricsolutions@gmail.com"
            className="flex items-center gap-2 text-sm font-medium px-4 py-2 rounded-xl"
            style={{ background: "rgba(30,144,255,0.1)", color: "#4da3ff", textDecoration: "none", border: "1px solid rgba(30,144,255,0.2)" }}>
            <Mail size={14} /> chemometricsolutions@gmail.com
          </a>
          <a href="https://chemometricsolutions.com" target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm font-medium px-4 py-2 rounded-xl"
            style={{ background: "rgba(30,144,255,0.1)", color: "#4da3ff", textDecoration: "none", border: "1px solid rgba(30,144,255,0.2)" }}>
            <Globe size={14} /> chemometricsolutions.com
          </a>
        </div>
      </div>
    </div>
  );
}