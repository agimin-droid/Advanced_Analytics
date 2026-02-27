import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { base44 } from "@/api/base44Client";
import { Sparkles, Database, ChevronDown, Send, RefreshCw, Lightbulb, TrendingUp, AlertTriangle, CheckCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";

const QUICK_PROMPTS = [
  "Summarize this dataset and highlight key patterns",
  "Identify anomalies or outliers in the data",
  "What correlations exist between columns?",
  "What are the most important insights from this data?",
  "Describe the distribution of numeric columns",
  "Are there data quality issues I should know about?",
];

export default function Insights() {
  const [datasets, setDatasets] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [parsedData, setParsedData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showPicker, setShowPicker] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [thinking, setThinking] = useState(false);
  const [autoInsight, setAutoInsight] = useState(null);
  const [insightLoading, setInsightLoading] = useState(false);

  useEffect(() => {
    base44.entities.Dataset.filter({ status: "ready" }, "-created_date", 50).then(list => {
      setDatasets(list);
      const params = new URLSearchParams(window.location.search);
      const id = params.get("id");
      if (id) setSelectedId(id);
      else if (list.length > 0) setSelectedId(list[0].id);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    const ds = datasets.find(d => d.id === selectedId);
    if (ds) {
      setDataset(ds);
      if (ds.preview_data) {
        const res = await fetch(ds.preview_data);
        const json = await res.json();
        setParsedData(json);
      } else {
        setParsedData([]);
      }
      setMessages([]);
      setAutoInsight(null);
    }
  }, [selectedId, datasets]);

  const getDataContext = () => {
    if (!dataset) return "";
    const sampleRows = parsedData.slice(0, 50);
    return `Dataset: "${dataset.name}"
File type: ${dataset.file_type}
Rows: ${dataset.row_count || parsedData.length}
Columns (${dataset.columns?.length}): ${dataset.columns?.join(", ")}
Sample data (first ${sampleRows.length} rows):
${JSON.stringify(sampleRows, null, 2)}`;
  };

  const generateAutoInsight = async () => {
    if (!dataset) return;
    setInsightLoading(true);
    const context = getDataContext();
    const result = await base44.integrations.Core.InvokeLLM({
      prompt: `You are an expert data analyst. Analyze this dataset and provide a comprehensive summary with key findings.

${context}

Please provide:
1. **Overview** - Brief description of what this dataset contains
2. **Key Statistics** - Important numbers and metrics
3. **Notable Patterns** - Interesting trends or patterns
4. **Data Quality** - Any missing values, anomalies, or issues
5. **Recommendations** - What analyses or actions to consider

Format your response using markdown with clear sections and bullet points.`,
    });
    setAutoInsight(result);
    setInsightLoading(false);
  };

  const askQuestion = async (q) => {
    const prompt = q || question;
    if (!prompt.trim() || !dataset) return;
    setQuestion("");
    setMessages(prev => [...prev, { role: "user", content: prompt }]);
    setThinking(true);

    const context = getDataContext();
    const history = messages.map(m => `${m.role === "user" ? "User" : "AI"}: ${m.content}`).join("\n");

    const result = await base44.integrations.Core.InvokeLLM({
      prompt: `You are an expert data analyst helping analyze a dataset. Answer the user's question based on the data provided.

Dataset context:
${context}

${history ? `Conversation history:\n${history}\n` : ""}
User question: ${prompt}

Provide a detailed, insightful answer using markdown formatting. Include specific numbers, percentages, or examples from the data where relevant.`,
    });

    setMessages(prev => [...prev, { role: "ai", content: result }]);
    setThinking(false);
  };

  if (loading) {
    return <div className="p-8 animate-in"><div className="glass-card animate-pulse" style={{ height: 400 }} /></div>;
  }

  return (
    <div className="p-6 lg:p-8 animate-in space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
            AI Insights
          </h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
            Ask questions and get AI-powered analysis on your data.
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
              {datasets.map(ds => (
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
              ))}
            </div>
          )}
        </div>
      </div>

      {!dataset ? (
        <div className="glass-card p-16 text-center">
          <Sparkles size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)" }} />
          <p className="text-sm font-medium mb-1" style={{ color: "var(--text-secondary)" }}>No dataset selected</p>
          <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Select or upload a dataset to begin AI analysis.</p>
          <Link to={createPageUrl("Upload")} className="btn-primary" style={{ textDecoration: "none", display: "inline-flex" }}>
            Upload Dataset
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Auto Analysis */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
                <Lightbulb size={16} style={{ color: "#f59e0b" }} />
                Auto Analysis
              </h2>
              <button
                onClick={generateAutoInsight}
                disabled={insightLoading}
                className="btn-primary text-xs py-1.5 px-3"
              >
                {insightLoading ? <RefreshCw size={12} className="animate-spin" /> : <Sparkles size={12} />}
                {insightLoading ? "Analyzing..." : autoInsight ? "Re-analyze" : "Analyze Dataset"}
              </button>
            </div>

            <div className="glass-card p-5 min-h-[300px]">
              {insightLoading ? (
                <div className="flex flex-col items-center justify-center h-48 gap-3">
                  <div className="w-10 h-10 rounded-full border-2 border-t-transparent animate-spin"
                    style={{ borderColor: "var(--accent)" }} />
                  <p className="text-sm" style={{ color: "var(--text-secondary)" }}>AI is analyzing your data...</p>
                </div>
              ) : autoInsight ? (
                <div className="prose prose-sm max-w-none"
                  style={{ color: "var(--text-secondary)", "--tw-prose-headings": "var(--text-primary)" }}>
                  <ReactMarkdown
                    components={{
                      h1: ({ children }) => <h1 style={{ color: "var(--text-primary)", fontSize: 16, fontWeight: 700, marginTop: 16, marginBottom: 8 }}>{children}</h1>,
                      h2: ({ children }) => <h2 style={{ color: "var(--text-primary)", fontSize: 14, fontWeight: 600, marginTop: 14, marginBottom: 6 }}>{children}</h2>,
                      h3: ({ children }) => <h3 style={{ color: "var(--text-primary)", fontSize: 13, fontWeight: 600, marginTop: 12, marginBottom: 4 }}>{children}</h3>,
                      p: ({ children }) => <p style={{ color: "var(--text-secondary)", fontSize: 13, marginBottom: 8, lineHeight: 1.6 }}>{children}</p>,
                      li: ({ children }) => <li style={{ color: "var(--text-secondary)", fontSize: 13, marginBottom: 4 }}>{children}</li>,
                      strong: ({ children }) => <strong style={{ color: "var(--text-primary)", fontWeight: 600 }}>{children}</strong>,
                      ul: ({ children }) => <ul style={{ paddingLeft: 16, marginBottom: 8 }}>{children}</ul>,
                    }}
                  >
                    {autoInsight}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-48 text-center">
                  <Sparkles size={32} className="mb-3" style={{ color: "var(--text-muted)" }} />
                  <p className="text-sm" style={{ color: "var(--text-secondary)" }}>Click "Analyze Dataset" to get AI-powered insights</p>
                  <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>Patterns, anomalies, quality checks & more</p>
                </div>
              )}
            </div>

            {/* Quick prompts */}
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "var(--text-muted)" }}>
                Quick Questions
              </p>
              <div className="flex flex-wrap gap-2">
                {QUICK_PROMPTS.map(p => (
                  <button
                    key={p}
                    onClick={() => askQuestion(p)}
                    disabled={thinking}
                    className="text-xs px-3 py-1.5 rounded-lg transition-colors"
                    style={{
                      background: "var(--bg-card)",
                      border: "1px solid var(--border)",
                      color: "var(--text-secondary)"
                    }}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Chat */}
          <div className="flex flex-col gap-4">
            <h2 className="text-base font-semibold flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
              <TrendingUp size={16} style={{ color: "var(--accent-light)" }} />
              Ask the AI
            </h2>

            <div className="glass-card flex flex-col" style={{ minHeight: 380, maxHeight: 500 }}>
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-32 text-center">
                    <p className="text-sm" style={{ color: "var(--text-muted)" }}>Ask any question about your data</p>
                  </div>
                )}
                {messages.map((m, i) => (
                  <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className="max-w-[85%] rounded-xl px-4 py-3 text-sm"
                      style={{
                        background: m.role === "user" ? "linear-gradient(135deg, #6366f1, #8b5cf6)" : "var(--bg-secondary)",
                        color: m.role === "user" ? "white" : "var(--text-secondary)",
                        border: m.role === "ai" ? "1px solid var(--border)" : "none",
                      }}
                    >
                      {m.role === "ai" ? (
                        <ReactMarkdown
                          components={{
                            p: ({ children }) => <p style={{ marginBottom: 6, lineHeight: 1.5, fontSize: 13 }}>{children}</p>,
                            li: ({ children }) => <li style={{ fontSize: 13, marginBottom: 2 }}>{children}</li>,
                            strong: ({ children }) => <strong style={{ color: "var(--text-primary)", fontWeight: 600 }}>{children}</strong>,
                            ul: ({ children }) => <ul style={{ paddingLeft: 14, marginBottom: 6 }}>{children}</ul>,
                          }}
                        >
                          {m.content}
                        </ReactMarkdown>
                      ) : (
                        m.content
                      )}
                    </div>
                  </div>
                ))}
                {thinking && (
                  <div className="flex justify-start">
                    <div className="rounded-xl px-4 py-3 flex items-center gap-2"
                      style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}>
                      <div className="flex gap-1">
                        {[0,1,2].map(i => (
                          <div key={i} className="w-1.5 h-1.5 rounded-full animate-bounce"
                            style={{ background: "var(--accent)", animationDelay: `${i * 0.15}s` }} />
                        ))}
                      </div>
                      <span className="text-xs" style={{ color: "var(--text-muted)" }}>Thinking...</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Input */}
              <div className="p-3" style={{ borderTop: "1px solid var(--border)" }}>
                <div className="flex gap-2">
                  <input
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && !e.shiftKey && askQuestion()}
                    placeholder="Ask anything about this dataset..."
                    disabled={thinking}
                    className="flex-1 px-4 py-2.5 rounded-xl text-sm outline-none"
                    style={{
                      background: "var(--bg-secondary)",
                      border: "1px solid var(--border)",
                      color: "var(--text-primary)",
                    }}
                  />
                  <button
                    onClick={() => askQuestion()}
                    disabled={!question.trim() || thinking}
                    className="btn-primary px-3 disabled:opacity-40"
                  >
                    <Send size={14} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}