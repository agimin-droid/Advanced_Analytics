import { useState, useEffect } from "react";
import { base44 } from "@/api/base44Client";
import { Shield, Users, UserCheck, UserX, Database, Trash2, Search, ChevronDown } from "lucide-react";
import StatCard from "../components/ui/StatCard";

const ROLES = ["admin", "analyst", "viewer"];

export default function Admin() {
  const [users, setUsers] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("viewer");
  const [inviting, setInviting] = useState(false);
  const [inviteMsg, setInviteMsg] = useState("");

  useEffect(() => {
    Promise.all([
      base44.auth.me(),
      base44.entities.User.list(),
      base44.entities.Dataset.list("-created_date", 100),
    ]).then(([me, userList, dsList]) => {
      setCurrentUser(me);
      setUsers(userList);
      setDatasets(dsList);
      setLoading(false);
    });
  }, []);

  if (!loading && currentUser?.role !== "admin") {
    return (
      <div className="p-8 flex flex-col items-center justify-center min-h-[50vh]">
        <Shield size={48} className="mb-4" style={{ color: "var(--text-muted)" }} />
        <h2 className="text-xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>Admin Only</h2>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          You need admin access to view this page.
        </p>
      </div>
    );
  }

  const filtered = users.filter(u =>
    u.full_name?.toLowerCase().includes(search.toLowerCase()) ||
    u.email?.toLowerCase().includes(search.toLowerCase())
  );

  const updateRole = async (userId, role) => {
    await base44.entities.User.update(userId, { role });
    setUsers(prev => prev.map(u => u.id === userId ? { ...u, role } : u));
  };

  const toggleAccess = async (userId, current) => {
    await base44.entities.User.update(userId, { access_granted: !current });
    setUsers(prev => prev.map(u => u.id === userId ? { ...u, access_granted: !current } : u));
  };

  const deleteDataset = async (id) => {
    if (!window.confirm("Delete this dataset permanently?")) return;
    await base44.entities.Dataset.delete(id);
    setDatasets(prev => prev.filter(d => d.id !== id));
  };

  const handleInvite = async () => {
    if (!inviteEmail.trim()) return;
    setInviting(true);
    setInviteMsg("");
    await base44.users.inviteUser(inviteEmail.trim(), inviteRole);
    setInviteMsg(`Invitation sent to ${inviteEmail}`);
    setInviteEmail("");
    setInviting(false);
  };

  if (loading) {
    return <div className="p-8 animate-in"><div className="glass-card animate-pulse" style={{ height: 300 }} /></div>;
  }

  return (
    <div className="p-6 lg:p-8 animate-in space-y-8">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
          Access Control
        </h1>
        <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
          Manage users, roles, and platform access.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Users" value={users.length} icon={Users} color="#6366f1" />
        <StatCard label="Admins" value={users.filter(u => u.role === "admin").length} icon={Shield} color="#f59e0b" />
        <StatCard label="Active" value={users.filter(u => u.access_granted !== false).length} icon={UserCheck} color="#10b981" />
        <StatCard label="Datasets" value={datasets.length} icon={Database} color="#8b5cf6" />
      </div>

      {/* Invite User */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
          <UserCheck size={15} style={{ color: "var(--accent-light)" }} />
          Invite New User
        </h2>
        <div className="flex flex-wrap gap-3">
          <input
            value={inviteEmail}
            onChange={e => setInviteEmail(e.target.value)}
            placeholder="Email address"
            className="flex-1 min-w-[200px] px-4 py-2.5 rounded-xl text-sm outline-none"
            style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
          />
          <select
            value={inviteRole}
            onChange={e => setInviteRole(e.target.value)}
            className="px-4 py-2.5 rounded-xl text-sm outline-none"
            style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
          >
            {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
          <button onClick={handleInvite} disabled={inviting || !inviteEmail.trim()} className="btn-primary disabled:opacity-40">
            {inviting ? "Sending..." : "Send Invite"}
          </button>
        </div>
        {inviteMsg && (
          <p className="text-xs mt-2" style={{ color: "#34d399" }}>{inviteMsg}</p>
        )}
      </div>

      {/* Users table */}
      <div className="glass-card overflow-hidden">
        <div className="p-5 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="text-sm font-semibold flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
            <Users size={15} style={{ color: "var(--accent-light)" }} />
            Users ({filtered.length})
          </h2>
          <div className="relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: "var(--text-muted)" }} />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search users..."
              className="pl-8 pr-4 py-2 rounded-lg text-sm outline-none w-48"
              style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
            />
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {["User", "Role", "Access", "Joined", "Actions"].map(h => (
                  <th key={h} className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wider"
                    style={{ color: "var(--text-muted)", background: "var(--bg-secondary)", borderBottom: "1px solid var(--border)" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map(u => (
                <tr key={u.id} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
                        style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)", color: "white" }}>
                        {u.full_name?.[0]?.toUpperCase() || "?"}
                      </div>
                      <div>
                        <div className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>{u.full_name || "—"}</div>
                        <div className="text-xs" style={{ color: "var(--text-muted)" }}>{u.email}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-3.5">
                    {u.id === currentUser?.id ? (
                      <span className="tag tag-blue">{u.role || "admin"}</span>
                    ) : (
                      <select
                        value={u.role || "viewer"}
                        onChange={e => updateRole(u.id, e.target.value)}
                        className="text-xs px-2.5 py-1.5 rounded-lg outline-none"
                        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                      >
                        {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
                      </select>
                    )}
                  </td>
                  <td className="px-5 py-3.5">
                    <button
                      onClick={() => toggleAccess(u.id, u.access_granted !== false)}
                      disabled={u.id === currentUser?.id}
                      className="flex items-center gap-1.5 text-xs font-medium disabled:opacity-50"
                      style={{ color: u.access_granted !== false ? "#34d399" : "#f87171" }}
                    >
                      {u.access_granted !== false ? <UserCheck size={13} /> : <UserX size={13} />}
                      {u.access_granted !== false ? "Granted" : "Revoked"}
                    </button>
                  </td>
                  <td className="px-5 py-3.5 text-xs" style={{ color: "var(--text-muted)" }}>
                    {new Date(u.created_date).toLocaleDateString()}
                  </td>
                  <td className="px-5 py-3.5">
                    {u.id !== currentUser?.id && (
                      <span className="text-xs" style={{ color: "var(--text-muted)" }}>—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Datasets management */}
      <div className="glass-card overflow-hidden">
        <div className="p-5" style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="text-sm font-semibold flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
            <Database size={15} style={{ color: "var(--accent-light)" }} />
            All Datasets ({datasets.length})
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {["Name", "Type", "Rows", "Uploaded by", "Date", ""].map(h => (
                  <th key={h} className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wider"
                    style={{ color: "var(--text-muted)", background: "var(--bg-secondary)", borderBottom: "1px solid var(--border)" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {datasets.map(ds => (
                <tr key={ds.id} style={{ borderBottom: "1px solid rgba(30,45,74,0.5)" }}>
                  <td className="px-5 py-3.5 text-sm font-medium" style={{ color: "var(--text-primary)" }}>{ds.name}</td>
                  <td className="px-5 py-3.5">
                    <span className="tag tag-blue">{ds.file_type?.toUpperCase()}</span>
                  </td>
                  <td className="px-5 py-3.5 text-sm" style={{ color: "var(--text-secondary)" }}>
                    {ds.row_count?.toLocaleString() || "—"}
                  </td>
                  <td className="px-5 py-3.5 text-xs" style={{ color: "var(--text-muted)" }}>
                    {ds.created_by || "—"}
                  </td>
                  <td className="px-5 py-3.5 text-xs" style={{ color: "var(--text-muted)" }}>
                    {new Date(ds.created_date).toLocaleDateString()}
                  </td>
                  <td className="px-5 py-3.5">
                    <button
                      onClick={() => deleteDataset(ds.id)}
                      className="p-1.5 rounded-lg transition-colors"
                      style={{ color: "var(--text-muted)" }}
                    >
                      <Trash2 size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}