"""
app.py — Code Sentinel · Slice 3: Multi-Agent Review + Approve / Reject

Run with:
    streamlit run app.py
"""
import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

import streamlit as st
from streamlit_option_menu import option_menu

from audit import load_decisions, log_decision
from aws_helpers import (
    backup_s3_object,
    fetch_script_from_s3,
    get_glue_job_details,
    list_glue_jobs,
    overwrite_s3_object,
    upload_s3_artifact,
)
from doc_export import build_docx, build_ipynb
from memory import (
    backfill_from_audit_log,
    build_memory_context,
    find_similar_reviews,
    get_memory_stats,
    store_memory,
)
from assistant import build_system_prompt, chat_stream, trim_history
from orchestrator import review_script

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Code Sentinel",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def _initialize_memory():
    return backfill_from_audit_log()


_initialize_memory()

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("""
<div style="padding: 4px 0 16px 0;">
    <div style="font-size: 22px; font-weight: 700; line-height: 1.2;">
        🛡️ Code Sentinel
    </div>
    <div style="font-size: 12px; color: #888; margin-top: 2px;">
        AI-powered pipeline review
    </div>
</div>
""", unsafe_allow_html=True)

    selected_view = option_menu(
        menu_title=None,
        options=["Review", "History"],
        icons=["clipboard-check", "clock-history"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"font-size": "15px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "1px 0",
                "padding": "9px 12px",
                "border-radius": "6px",
                "--hover-color": "rgba(255, 75, 75, 0.08)",
            },
            "nav-link-selected": {
                "background-color": "rgba(255, 75, 75, 0.15)",
                "color": "#ff4b4b",
                "font-weight": "600",
            },
        },
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    _sb_region = os.getenv("AWS_BEDROCK_REGION", "—")
    _sb_model_id = os.getenv("BEDROCK_MODEL_ID", "—")
    _sb_model_short = (
        _sb_model_id.split(".")[-1].split(":")[0] if _sb_model_id != "—" else "—"
    )
    _sb_stats = get_memory_stats()

    st.markdown(f"""
<div style="
    background: rgba(128, 128, 128, 0.06);
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 16px;
">
    <div style="
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #888;
        margin-bottom: 10px;
    ">Configuration</div>
    <div style="font-size: 12px; margin-bottom: 6px;">
        <span style="color: #888;">Region</span>
        <span style="float: right; font-family: monospace;">{_sb_region}</span>
    </div>
    <div style="font-size: 12px; margin-bottom: 6px;">
        <span style="color: #888;">Model</span>
        <span style="float: right; font-family: monospace; font-size: 11px;">{_sb_model_short}</span>
    </div>
    <div style="font-size: 12px;">
        <span style="color: #888;">Memory</span>
        <span style="float: right;">
            <b>{_sb_stats['total']}</b>
            <span style="color: #22c55e;">· {_sb_stats['approved']} ✓</span>
            <span style="color: #ef4444;">· {_sb_stats['rejected']} ✗</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

    _bedrock_ok = bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(os.getenv("AWS_BEDROCK_REGION"))
    _s3_ok = bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(os.getenv("AWS_GLUE_REGION"))
    try:
        from memory import get_collection as _get_collection
        _chromadb_ok = _get_collection() is not None
    except Exception:
        _chromadb_ok = False

    def _status_html(label: str, ok: bool) -> str:
        color = "#22c55e" if ok else "#ef4444"
        return (
            f'<div style="font-size: 12px; margin-bottom: 4px;">'
            f'<span style="color: {color}; font-size: 9px; vertical-align: middle;">●</span>'
            f'<span style="margin-left: 6px; color: #ccc;">{label}</span>'
            f'</div>'
        )

    st.markdown(f"""
<div style="
    background: rgba(128, 128, 128, 0.06);
    border-radius: 8px;
    padding: 12px 14px;
">
    <div style="
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #888;
        margin-bottom: 10px;
    ">Service Status</div>
    {_status_html("Bedrock", _bedrock_ok)}
    {_status_html("AWS S3 / Glue", _s3_ok)}
    {_status_html("ChromaDB Memory", _chromadb_ok)}
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# MAIN HEADER
# ===========================================================================
_hdr_col, _reset_col = st.columns([5, 1])
with _hdr_col:
    st.markdown("# 🛡️ Code Sentinel")
    st.markdown("*AI-powered code review & documentation for data pipelines*")
with _reset_col:
    if st.session_state.get("review_result"):
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 New Review", key="reset_btn_header", use_container_width=True):
            for k in ["review_result", "last_decision",
                      "similar_reviews", "memory_context",
                      "assistant_messages", "original_script_content",
                      "uploaded_filename", "reviewed_job"]:
                st.session_state.pop(k, None)
            st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_RISK_COLOURS = {
    "CRITICAL": ("#ff4444", "#1a0000"),
    "HIGH":     ("#ff8c00", "#1a0a00"),
    "MEDIUM":   ("#ffd700", "#1a1500"),
    "LOW":      ("#4488ff", "#00001a"),
    "NONE":     ("#44ff88", "#001a00"),
}

_SEV_BADGE = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🔵",
}


# ---------------------------------------------------------------------------
# Helper functions (module-level so both views can use them)
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")





def render_mermaid(diagram: str) -> None:
    """Render a Mermaid diagram client-side via streamlit-mermaid (no external calls)."""
    try:
        from streamlit_mermaid import st_mermaid
        st_mermaid(diagram, height=600)
    except Exception:
        st.warning("Could not render diagram. Showing source instead.")
        st.code(diagram, language="text")


def _fmt_ts(ts_raw: str) -> str:
    """Format an ISO-8601 timestamp as 'YYYY-MM-DD HH:MM UTC'."""
    try:
        dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts_raw


# ===========================================================================
# HISTORY VIEW  (always available — no review required)
# ===========================================================================
if selected_view == "History":
    decisions = load_decisions(limit=50)
    total = len(decisions)
    approved = sum(1 for d in decisions if d.get("decision") == "APPROVED")
    rejected = total - approved
    st.markdown(f"**{total} decisions logged · {approved} approved · {rejected} rejected**")
    st.markdown("---")

    if not decisions:
        st.info("No decisions logged yet. Approve or Reject a review to start the audit trail.")
    else:
        for rec in decisions:
            decision = rec.get("decision", "UNKNOWN")
            job = rec.get("job_name", "unknown")
            timestamp = rec.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                pretty_ts = dt.strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                pretty_ts = timestamp

            border_color = "#22c55e" if decision == "APPROVED" else "#ef4444"
            badge_bg = "rgba(34, 197, 94, 0.15)" if decision == "APPROVED" else "rgba(239, 68, 68, 0.15)"
            badge_text = "#22c55e" if decision == "APPROVED" else "#ef4444"

            st.markdown(f"""
<div style="
    border-left: 3px solid {border_color};
    padding: 12px 16px;
    margin: 12px 0;
    background: rgba(128, 128, 128, 0.05);
    border-radius: 6px;
">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <span style="
            background: {badge_bg};
            color: {badge_text};
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
        ">{decision}</span>
        <span style="font-weight: 600; font-size: 15px;">{_esc(job)}</span>
        <span style="color: #888; font-size: 12px; font-family: monospace;">{_esc(pretty_ts)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"**Source:** {rec.get('source', '—')}")
                agents = rec.get("agents_run") or []
                if agents:
                    st.caption(f"**Agents:** {', '.join(agents) if isinstance(agents, list) else agents}")
                savings = rec.get("savings_per_run_usd")
                if savings:
                    st.caption(f"**Est. savings:** ${float(savings):.2f}/run")
            with col_b:
                if decision == "REJECTED" and rec.get("reason"):
                    st.caption(f"**Reason:** {rec['reason']}")
                backup = rec.get("s3_backup_path")
                if backup:
                    st.caption(f"**Backup:** `{backup}`")
                doc = rec.get("s3_doc_path") or rec.get("s3_docs_path") or rec.get("s3_review_folder")
                if doc:
                    st.caption(f"**Docs:** `{doc}`")

            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    st.stop()  # don't fall through into the Review view

# ===========================================================================
# REVIEW VIEW
# ===========================================================================

# ---------------------------------------------------------------------------
# Input source
# ---------------------------------------------------------------------------
st.subheader("1. Pick a Glue job")
source = st.radio("Input source", ["From AWS Glue", "Upload a file"], horizontal=True)
st.session_state["input_source"] = "glue" if source == "From AWS Glue" else "upload"

if source == "From AWS Glue":
    try:
        jobs = list_glue_jobs()
    except Exception as e:
        st.error(f"Failed to list Glue jobs: {e}")
        st.stop()

    if not jobs:
        st.warning("No Glue jobs found in this region.")
        st.stop()

    selected = st.selectbox("Glue jobs in your account", jobs, index=0)

    try:
        details = get_glue_job_details(selected)
    except Exception as e:
        st.error(f"Failed to fetch Glue job details: {e}")
        st.stop()
    command = details.get("Command", {})
    script_location = command.get("ScriptLocation", "")
    job_type = command.get("Name", "")
    worker_type = details.get("WorkerType")
    num_workers = details.get("NumberOfWorkers")
    timeout = details.get("Timeout")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Job type", job_type)
    col_b.metric("Worker type", worker_type or "—")
    col_c.metric("Workers", num_workers or "—")
    col_d.metric("Timeout (min)", timeout or "—")
    st.caption(f"Script location: `{script_location}`")

    try:
        original_script = fetch_script_from_s3(script_location)
    except Exception as e:
        st.error(f"Failed to fetch script from S3: {e}")
        st.stop()
    st.session_state["script_s3_uri"] = script_location
    st.session_state["original_script_content"] = original_script

else:  # Upload a file
    uploaded_file = st.file_uploader(
        "Upload a script — PySpark, Python, SQL, or Jupyter Notebook",
        type=["py", "sql", "ipynb"],
        key="script_uploader",
    )
    if uploaded_file is None:
        st.info("Upload a .py, .sql, or .ipynb file to review.")
        st.stop()

    _raw_content = uploaded_file.getvalue().decode("utf-8")
    filename = uploaded_file.name

    if filename.lower().endswith(".ipynb"):
        try:
            _nb = json.loads(_raw_content)
            _code_cells = []
            for cell in _nb.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                source = source.strip()
                if source:
                    _code_cells.append(source)
            if not _code_cells:
                st.error("No code cells found in the notebook.")
                st.stop()
            original_script = "\n\n# ===== CELL BOUNDARY =====\n\n".join(_code_cells)
            st.info(
                f"📓 Jupyter Notebook detected — extracted "
                f"{len(_code_cells)} code cell(s) for review"
            )
        except Exception as _nb_exc:
            st.error(f"Failed to parse notebook: {_nb_exc}")
            st.stop()
    else:
        original_script = _raw_content

    st.session_state["original_script_content"] = original_script
    st.session_state["uploaded_filename"] = filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "py"
    job_type = "sql" if ext == "sql" else "glueetl"
    worker_type = None
    num_workers = None
    timeout = None
    selected = filename.rsplit(".", 1)[0]
    st.caption(f"File: `{filename}`")

# Build a fingerprint of the current input selection and clear stale review state
if source == "From AWS Glue":
    _current_input_key = f"glue:{selected}"
else:
    _current_input_key = f"upload:{filename}"

if st.session_state.get("_last_input_key") != _current_input_key:
    for _k in [
        "review_result", "last_decision", "similar_reviews",
        "memory_context", "assistant_messages", "active_result_tab",
        "script_s3_uri",
    ]:
        st.session_state.pop(_k, None)
    st.session_state["_last_input_key"] = _current_input_key

with st.expander("View original script"):
    lang = "sql" if job_type == "sql" else "python"
    st.code(original_script, language=lang)

# ---------------------------------------------------------------------------
# Agent toggles + Run button
# ---------------------------------------------------------------------------
st.markdown("##### 🤖 Select agents to run")

_agent_keys = ["agent_perf", "agent_docs", "agent_sec", "agent_lin"]
for _k in _agent_keys:
    if _k not in st.session_state:
        st.session_state[_k] = True

col_all, col_none, _ = st.columns([1, 1, 4])
with col_all:
    if st.button("Select all", key="agents_all", use_container_width=True):
        for _k in _agent_keys:
            st.session_state[_k] = True
        st.rerun()
with col_none:
    if st.button("Clear all", key="agents_none", use_container_width=True):
        for _k in _agent_keys:
            st.session_state[_k] = False
        st.rerun()

cols = st.columns(4)
agent_meta = [
    ("agent_perf", "⚡ Performance",   "Optimize for cost & speed"),
    ("agent_docs", "📘 Documentation", "Generate Word doc"),
    ("agent_sec",  "🔒 Security",      "Find PII, compliance risks"),
    ("agent_lin",  "🔗 Lineage",       "Map data flow"),
]
for col, (key, label, tooltip) in zip(cols, agent_meta):
    with col:
        st.checkbox(label, key=key, help=tooltip)

if st.button("⚡ Run Code Sentinel Review", type="primary"):
    selected_agents = {
        name for name, enabled in [
            ("performance",   st.session_state.get("agent_perf", True)),
            ("documentation", st.session_state.get("agent_docs", True)),
            ("security",      st.session_state.get("agent_sec",  True)),
            ("lineage",       st.session_state.get("agent_lin",  True)),
        ] if enabled
    }
    if not selected_agents:
        st.warning("Select at least one agent.")
        st.stop()

    _similar = find_similar_reviews(original_script)
    _memory_context = build_memory_context(_similar)
    st.session_state["similar_reviews"] = _similar
    st.session_state["memory_context"] = _memory_context

    config_dict = {
        "worker_type": worker_type,
        "number_of_workers": num_workers,
        "timeout": timeout,
        "job_type": job_type,
    }
    n = len(selected_agents)
    with st.status("🚀 Running Code Sentinel review...", expanded=True) as _status:
        st.write(f"🤖 Running {n} agent{'s' if n != 1 else ''} in parallel (this may take ~30s)...")
        review = review_script(original_script, config_dict, selected_agents, _memory_context)
        st.write("✓ All agents complete")
        _status.update(label="✅ Review complete", state="complete", expanded=False)
    st.session_state["review_result"] = review
    st.session_state["reviewed_job"] = selected
    st.session_state["selected_agents"] = selected_agents
    st.session_state.pop("last_decision", None)  # clear any prior decision banner

# ---------------------------------------------------------------------------
# Results guard
# ---------------------------------------------------------------------------
if "assistant_messages" not in st.session_state:
    st.session_state["assistant_messages"] = []

if "review_result" not in st.session_state:
    st.stop()

review = st.session_state["review_result"]
meta = review.get("_meta", {})
errors = review.get("errors", {})

# Status bar
agents_failed = meta.get("agents_failed", 0)
_skipped_keys = [k for k, v in errors.items() if v == "Not selected"]
_failed_keys  = [k for k, v in errors.items() if v != "Not selected"]
_skipped_visible = [k for k in _skipped_keys if k != "test_gen"]

status_msg = (
    f"**{meta.get('agents_run', 5) - agents_failed} agents succeeded** · "
    f"Total time: **{meta.get('total_seconds', '?')}s**"
)
if _failed_keys:
    status_msg += f" · ⚠️ Failed: {', '.join(f'`{k}`' for k in _failed_keys)}"
if _skipped_visible:
    status_msg += f" · ⏭ Skipped: {', '.join(f'`{k}`' for k in _skipped_visible)}"
st.markdown(status_msg)

_tab_labels = [
    "📊 Before / After",
    "📘 Documentation",
    "🔒 Security",
    "🔗 Lineage",
    "💬 Ask",
]

if "active_result_tab" not in st.session_state:
    st.session_state["active_result_tab"] = 0

_active_tab_label = option_menu(
    menu_title=None,
    options=_tab_labels,
    default_index=st.session_state["active_result_tab"],
    orientation="horizontal",
    key="result_tabs_menu",
    styles={
        "container": {
            "padding": "4px",
            "background-color": "rgba(128, 128, 128, 0.05)",
            "border-radius": "8px",
            "margin-bottom": "16px",
        },
        "icon": {"display": "none"},
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0 2px",
            "padding": "8px 14px",
            "border-radius": "6px",
            "--hover-color": "rgba(255, 75, 75, 0.08)",
        },
        "nav-link-selected": {
            "background-color": "rgba(255, 75, 75, 0.15)",
            "color": "#ff4b4b",
            "font-weight": "600",
        },
    },
)

st.session_state["active_result_tab"] = _tab_labels.index(_active_tab_label)

# ---------------------------------------------------------------------------
# TAB 1 — Before / After
# ---------------------------------------------------------------------------
if _active_tab_label == "📊 Before / After":
    perf = review.get("performance")
    if perf is None:
        if errors.get("performance") == "Not selected":
            st.info("Performance agent skipped — re-run with the checkbox enabled to see results.")
        else:
            st.error(f"Performance agent failed: {errors.get('performance', 'unknown error')}")
    else:
        _job_slug = st.session_state.get("reviewed_job", "job").replace(" ", "_").lower()
        _paradigm = perf.get("detected_paradigm", "")
        _code_lang = "sql" if _paradigm == "sql" else "python"

        # ---- Decision banner (persists after Approve/Reject) ----
        _last_dec = st.session_state.get("last_decision")
        if _last_dec:
            _dec = _last_dec.get("decision", "")
            _dec_color = "#44ff88" if _dec == "APPROVED" else "#ff4444"
            _dec_bg    = "#1a4a1a" if _dec == "APPROVED" else "#4a1a1a"
            _dec_icon  = "✅" if _dec == "APPROVED" else "❌"
            _dec_ts    = _fmt_ts(_last_dec.get("timestamp", ""))
            _banner_lines = [
                f"{_dec_icon} {_dec} — {_last_dec.get('job_name', '')}",
                f"Recorded at {_dec_ts}",
            ]
            if _dec == "APPROVED":
                if _last_dec.get("s3_backup_path"):
                    _banner_lines.append(f"Backup: {_last_dec['s3_backup_path']}")
                _s3_uri_display = st.session_state.get("script_s3_uri", "")
                if _s3_uri_display:
                    _banner_lines.append(f"Updated: {_s3_uri_display}")
                _docs_uri = _last_dec.get("s3_doc_path") or _last_dec.get("s3_docs_path")
                if _docs_uri:
                    _banner_lines.append(f"Docs: {_docs_uri}")
            else:
                _banner_lines.append(f"Reason: {_last_dec.get('reason', '')}")
            st.markdown(
                f'<div style="background:{_dec_bg};border:2px solid {_dec_color};'
                f'border-radius:6px;padding:0.75rem 1rem;margin-bottom:0.5rem">'
                + "<br>".join(_esc(ln) for ln in _banner_lines)
                + "</div>",
                unsafe_allow_html=True,
            )
            if st.button("▶ Start New Review", key="new_review_btn"):
                st.session_state.pop("review_result", None)
                st.session_state.pop("last_decision", None)
                st.session_state.pop("similar_reviews", None)
                st.session_state.pop("memory_context", None)
                st.session_state["assistant_messages"] = []
                st.rerun()
            st.markdown("---")

        # ---- Memory match panel ----
        _similar_ui = st.session_state.get("similar_reviews")
        if _similar_ui:
            st.markdown(f"""
<div style="
    background: linear-gradient(90deg, rgba(168, 85, 247, 0.08), rgba(168, 85, 247, 0.02));
    border-left: 3px solid #a855f7;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 16px;
">
    <div style="font-weight: 600; color: #a855f7; margin-bottom: 4px;">
        🧠 Organizational memory active
    </div>
    <div style="font-size: 14px; color: inherit;">
        Found {len(_similar_ui)} similar past review(s).
        Context was injected into all agent prompts.
    </div>
</div>
""", unsafe_allow_html=True)
            with st.expander("View memory context (what the agents saw)", expanded=False):
                for _mi, _m in enumerate(_similar_ui, 1):
                    _badge = "🟢 APPROVED" if _m["decision"] == "APPROVED" else "🔴 REJECTED"
                    st.markdown(
                        f"**[{_mi}] {_badge} · `{_m['job_name']}` · "
                        f"{_m['timestamp'][:10]} · "
                        f"similarity {int(_m['similarity'] * 100)}%**"
                    )
                    if _m["decision"] == "REJECTED" and _m["reason"]:
                        st.markdown(f"> Reason: {_m['reason']}")
                    if _m["findings_summary"]:
                        st.caption(f"Findings: {_m['findings_summary']}")
                    st.divider()

        # ---- Download ----
        _dl_input_source = st.session_state.get("input_source", "glue")
        _uploaded_fname = st.session_state.get("uploaded_filename", "")
        _is_notebook_source = _dl_input_source == "upload" and _uploaded_fname.lower().endswith(".ipynb")

        if _is_notebook_source:
            _dl_data = build_ipynb(perf.get("optimized_code", ""), _uploaded_fname)
            _dl_mime = "application/x-ipynb+json"
            _dl_ext = ".ipynb"
            _dl_stem = Path(_uploaded_fname).stem
        elif _dl_input_source == "upload" and _uploaded_fname:
            _dl_data = perf.get("optimized_code", "")
            _dl_mime = "text/x-python"
            _dl_ext = Path(_uploaded_fname).suffix or ".py"
            _dl_stem = Path(_uploaded_fname).stem
        else:
            _dl_data = perf.get("optimized_code", "")
            _dl_mime = "text/x-python"
            _dl_ext = ".py"
            _dl_stem = _job_slug

        st.download_button(
            f"⬇️ Download optimized code ({_dl_ext})",
            data=_dl_data,
            file_name=f"{_dl_stem}_optimized{_dl_ext}",
            mime=_dl_mime,
            key="dl_perf",
        )
        if _paradigm:
            st.caption(f"Detected paradigm: `{_paradigm}` — optimized within this paradigm only")

        rec = perf.get("config_recommendation", {})
        savings = perf.get("estimated_savings", {})
        optimized_code = perf.get("optimized_code", "")

        # Config & cost header
        h1, h2, h3, h4 = st.columns(4)
        h1.metric(
            "Worker type",
            str(rec.get("worker_type", "—")),
            delta=f"was {worker_type or 'n/a'}",
        )
        h2.metric(
            "Workers",
            str(rec.get("number_of_workers", "—")),
            delta=f"was {num_workers or 'n/a'}",
        )
        h3.metric("$ before/run", f"${savings.get('before_per_run_usd', 0):.2f}")
        h4.metric(
            "$ after/run",
            f"${savings.get('after_per_run_usd', 0):.2f}",
            delta=f"-${savings.get('before_per_run_usd', 0) - savings.get('after_per_run_usd', 0):.2f}",
            delta_color="inverse",
        )

        before_usd = savings.get("before_per_run_usd", 0)
        after_usd  = savings.get("after_per_run_usd", 0)
        if before_usd and before_usd > 0:
            pct = (before_usd - after_usd) / before_usd * 100
            st.markdown(
                f"<h3 style='color:#6bff8a'>Estimated saving: "
                f"${before_usd - after_usd:.2f} per run ({pct:.0f}%)</h3>",
                unsafe_allow_html=True,
            )
        st.info(rec.get("rationale", ""))

        st.markdown("---")

        # Always side-by-side diff
        st.markdown("**Original** vs **Optimized**")
        _diff_code_lang = "sql" if _paradigm == "sql" else "python"
        diff_l, diff_r = st.columns(2)
        with diff_l:
            st.markdown("**Original**")
            st.code(original_script, language=_diff_code_lang)
        with diff_r:
            st.markdown("**Optimized**")
            st.code(optimized_code, language=_diff_code_lang)

        # Summary of changes
        st.markdown("#### Summary of changes")
        for change in perf.get("summary_of_changes", []):
            st.markdown(f"- {change}")

        # ---- Approve / Reject ----
        st.markdown("---")
        st.subheader("🚀 Apply changes")

        _input_source = st.session_state.get("input_source", "upload")

        if _last_dec:
            st.info("Decision recorded — start a new review to act again.")
        elif _input_source == "upload":
            st.subheader("💾 Save to organizational memory")
            st.caption(
                "Upload-source reviews can't overwrite S3, but you can log this decision "
                "to memory so future similar reviews benefit."
            )
            _up_col_a, _up_col_b = st.columns(2)
            with _up_col_a:
                _save_clicked = st.button("💾 Save review to memory", type="primary", key="save_upload_btn")
            with _up_col_b:
                _discard_clicked = st.button("❌ Discard", key="discard_upload_btn")
            _discard_reason = st.text_area(
                "Discard reason (required for Discard)",
                value="",
                height=80,
                placeholder="e.g. 'optimization changes break our downstream consumer'",
                key="discard_reason_upload",
            )

            if _save_clicked:
                try:
                    _up_name = st.session_state.get("uploaded_filename", "uploaded_script")
                    _up_findings: list[str] = []
                    _up_sec = review.get("security")
                    if _up_sec and isinstance(_up_sec, dict):
                        _up_issues = _up_sec.get("issues") or _up_sec.get("findings") or []
                        if _up_issues:
                            _up_findings.append(f"Security: {len(_up_issues)} issue(s)")
                    if perf and perf.get("summary_of_changes"):
                        _up_findings.append(f"Perf: {len(perf['summary_of_changes'])} optimization(s)")
                    _up_findings_summary = "; ".join(_up_findings) if _up_findings else None
                    _sv = perf.get("estimated_savings", {}) if perf else {}
                    _decision_record = {
                        "timestamp":           datetime.now(timezone.utc).isoformat(),
                        "job_name":            _up_name,
                        "source":              "upload",
                        "decision":            "APPROVED",
                        "agents_run":          list(st.session_state.get("selected_agents", set())),
                        "reason":              None,
                        "s3_backup_path":      None,
                        "s3_review_folder":    None,
                        "s3_doc_path":         None,
                        "savings_per_run_usd": _sv.get("before_per_run_usd", 0) - _sv.get("after_per_run_usd", 0),
                    }
                    log_decision(_decision_record)
                    store_memory(_decision_record, st.session_state.get("original_script_content"), _up_findings_summary)
                    st.session_state["last_decision"] = _decision_record
                    st.success(f"Saved to memory. Job: {_up_name}")
                except Exception as _up_exc:
                    st.error(f"Failed to save: {_up_exc}")

            if _discard_clicked:
                if not _discard_reason.strip():
                    st.warning("Provide a discard reason.")
                else:
                    _up_name = st.session_state.get("uploaded_filename", "uploaded_script")
                    _sv = perf.get("estimated_savings", {}) if perf else {}
                    _decision_record = {
                        "timestamp":           datetime.now(timezone.utc).isoformat(),
                        "job_name":            _up_name,
                        "source":              "upload",
                        "decision":            "REJECTED",
                        "agents_run":          list(st.session_state.get("selected_agents", set())),
                        "reason":              _discard_reason.strip(),
                        "s3_backup_path":      None,
                        "s3_review_folder":    None,
                        "s3_doc_path":         None,
                        "savings_per_run_usd": _sv.get("before_per_run_usd", 0) - _sv.get("after_per_run_usd", 0),
                    }
                    log_decision(_decision_record)
                    store_memory(_decision_record, st.session_state.get("original_script_content"), None)
                    st.session_state["last_decision"] = _decision_record
                    st.success("Discarded. Reason logged to memory.")
        else:
            confirm = st.checkbox(
                "I understand this will overwrite the production script in S3 "
                "(a timestamped backup will be saved first)",
                key="approve_confirm",
            )
            _btn_l, _btn_r = st.columns(2)
            with _btn_l:
                approve_clicked = st.button(
                    "✅ Approve & Apply to Glue",
                    type="primary",
                    disabled=not confirm,
                    key="approve_btn",
                )
            with _btn_r:
                reject_clicked = st.button("❌ Reject")
            reject_reason = st.text_area(
                "Rejection reason (required for Reject)",
                value="",
                height=80,
                placeholder="e.g. 'we don't want this column dropped — downstream BI depends on it'",
            )

            if approve_clicked:
                _ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                _s3_uri   = st.session_state.get("script_s3_uri", "")
                _parsed   = urlparse(_s3_uri)
                _bucket   = _parsed.netloc
                _orig_key = _parsed.path.lstrip("/")
                _stem     = _orig_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                _prefix   = _orig_key.rsplit("/", 1)[0] if "/" in _orig_key else ""
                _backup_uri    = ""
                _docs_s3_path  = None
                try:
                    with st.spinner("Backing up original and applying optimized script…"):
                        _bp = (f"s3://{_bucket}/{_prefix}/_backups/{_ts}"
                               if _prefix else f"s3://{_bucket}/_backups/{_ts}")
                        _backup_uri = backup_s3_object(_s3_uri, _bp)
                        overwrite_s3_object(_s3_uri, perf["optimized_code"])

                        # Audit record logged immediately after overwrite so the
                        # trail is safe even if the optional doc upload fails.
                        _sv = perf.get("estimated_savings", {})
                        _decision_record = {
                            "timestamp":           datetime.now(timezone.utc).isoformat(),
                            "job_name":            st.session_state.get("reviewed_job", "unknown"),
                            "source":              _input_source,
                            "decision":            "APPROVED",
                            "agents_run":          list(st.session_state.get("selected_agents", set())),
                            "reason":              None,
                            "s3_backup_path":      _backup_uri,
                            "s3_review_folder":    None,
                            "s3_doc_path":         None,
                            "savings_per_run_usd": _sv.get("before_per_run_usd", 0) - _sv.get("after_per_run_usd", 0),
                        }
                        log_decision(_decision_record)

                        try:
                            _docs_result = review.get("documentation")
                            if _docs_result:
                                _docx = build_docx(
                                    _docs_result, st.session_state.get("reviewed_job", ""))
                                _dk  = (f"{_prefix}/_docs/{_stem}_{_ts}.docx"
                                        if _prefix else f"_docs/{_stem}_{_ts}.docx")
                                upload_s3_artifact(_bucket, _dk, _docx)
                                _docs_s3_path = f"s3://{_bucket}/{_dk}"
                                _decision_record["s3_doc_path"] = _docs_s3_path
                        except Exception as _doc_exc:
                            logger.warning(
                                "Doc upload failed (audit record already saved): %s", _doc_exc
                            )

                        _findings_parts = []
                        _sec_result = review.get("security")
                        if _sec_result and isinstance(_sec_result, dict):
                            _issues = _sec_result.get("issues") or _sec_result.get("findings") or []
                            if _issues:
                                _findings_parts.append(f"Security: {len(_issues)} issue(s)")
                        if perf and perf.get("summary_of_changes"):
                            _findings_parts.append(f"Perf: {len(perf['summary_of_changes'])} optimization(s)")
                        _findings_summary = "; ".join(_findings_parts) if _findings_parts else None
                        _original_for_memory = st.session_state.get("original_script_content")
                        store_memory(_decision_record, _original_for_memory, _findings_summary)
                except Exception as _exc:
                    st.error(f"Approval failed: {_exc}")
                else:
                    _lines = [
                        "**Approved & applied to Glue.**",
                        f"- Backup: `{_backup_uri}`",
                        f"- Updated: `{_s3_uri}`",
                    ]
                    if _docs_s3_path:
                        _lines.append(f"- Docs: `{_docs_s3_path}`")
                    st.success("\n".join(_lines))
                    st.session_state["last_decision"] = _decision_record

            if reject_clicked:
                if not reject_reason.strip():
                    st.warning("Provide a reason before rejecting.")
                else:
                    _sv = perf.get("estimated_savings", {})
                    _decision_record = {
                        "timestamp":           datetime.now(timezone.utc).isoformat(),
                        "job_name":            st.session_state.get("reviewed_job", "unknown"),
                        "source":              _input_source,
                        "decision":            "REJECTED",
                        "agents_run":          list(st.session_state.get("selected_agents", set())),
                        "reason":              reject_reason.strip(),
                        "s3_backup_path":      None,
                        "s3_review_folder":    None,
                        "s3_doc_path":         None,
                        "savings_per_run_usd": _sv.get("before_per_run_usd", 0) - _sv.get("after_per_run_usd", 0),
                    }
                    log_decision(_decision_record)
                    _findings_parts = []
                    _sec_result = review.get("security")
                    if _sec_result and isinstance(_sec_result, dict):
                        _issues = _sec_result.get("issues") or _sec_result.get("findings") or []
                        if _issues:
                            _findings_parts.append(f"Security: {len(_issues)} issue(s)")
                    if perf and perf.get("summary_of_changes"):
                        _findings_parts.append(f"Perf: {len(perf['summary_of_changes'])} optimization(s)")
                    _findings_summary = "; ".join(_findings_parts) if _findings_parts else None
                    _original_for_memory = st.session_state.get("original_script_content")
                    store_memory(_decision_record, _original_for_memory, _findings_summary)
                    st.success("Rejection logged. Reason saved for future reference.")
                    st.session_state["last_decision"] = _decision_record


# ---------------------------------------------------------------------------
# TAB 2 — Documentation
# ---------------------------------------------------------------------------
if _active_tab_label == "📘 Documentation":
    docs = review.get("documentation")
    if docs is None:
        if errors.get("documentation") == "Not selected":
            st.info("Documentation agent skipped — re-run with the checkbox enabled to see results.")
        else:
            st.error(f"Documentation agent failed: {errors.get('documentation', 'unknown error')}")
    else:
        _job_slug_d = st.session_state.get("reviewed_job", "job").replace(" ", "_").lower()
        try:
            _docx_bytes = build_docx(docs, st.session_state.get("reviewed_job", ""))
            st.download_button(
                label="⬇️ Download documentation (.docx)",
                data=_docx_bytes,
                file_name=f"{_job_slug_d}_docs.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_docs_docx_btn",
            )
        except Exception as _docx_err:
            st.warning(f"Could not build .docx: {_docx_err}. Showing preview only.")

        st.markdown(f"> {docs.get('business_summary', '')}")
        st.markdown("---")
        col_in, col_out = st.columns(2)
        with col_in:
            st.markdown("**Inputs**")
            for item in docs.get("inputs", []):
                st.markdown(f"- **{item.get('source', '—')}**: {item.get('description', '')}")
        with col_out:
            st.markdown("**Outputs**")
            for item in docs.get("outputs", []):
                st.markdown(f"- **{item.get('destination', '—')}**: {item.get('description', '')}")

        st.markdown("#### Transformation Logic")
        for i, step in enumerate(docs.get("transformation_logic", []), 1):
            st.markdown(f"{i}. {step}")

        st.markdown("#### Operational Notes")
        for note in docs.get("operational_notes", []):
            st.markdown(f"- {note}")

        with st.expander("Generated Docstring"):
            raw_ds = docs.get("generated_docstring", "")
            st.code(raw_ds.replace("\\n", "\n"), language="python")


# ---------------------------------------------------------------------------
# TAB 3 — Security
# ---------------------------------------------------------------------------
if _active_tab_label == "🔒 Security":
    sec = review.get("security")
    if sec is None:
        if errors.get("security") == "Not selected":
            st.info("Security agent skipped — re-run with the checkbox enabled to see results.")
        else:
            st.error(f"Security agent failed: {errors.get('security', 'unknown error')}")
    else:
        _job_slug_s = st.session_state.get("reviewed_job", "job").replace(" ", "_").lower()
        st.download_button(
            "⬇️ Download security report",
            data=json.dumps(sec, indent=2),
            file_name=f"{_job_slug_s}_security.json",
            mime="application/json",
            key="dl_sec",
        )

        risk = sec.get("overall_risk", "NONE")
        fg, bg = _RISK_COLOURS.get(risk, ("#ffffff", "#333333"))
        st.markdown(
            f'<div style="background:{bg};border:2px solid {fg};border-radius:6px;'
            f'padding:0.75rem 1rem;margin-bottom:1rem">'
            f'<span style="color:{fg};font-size:1.4rem;font-weight:700">'
            f"Overall Risk: {risk}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(sec.get("summary", ""))

        findings = sec.get("findings", [])
        if not findings:
            st.success("No security issues detected.")
        else:
            st.markdown(f"#### Findings ({len(findings)})")
            for f in findings:
                sev = f.get("severity", "LOW")
                badge = _SEV_BADGE.get(sev, "⚪")
                line_info = f"line {f['line_number']}" if f.get("line_number") else "n/a"
                st.markdown(
                    f"{badge} **{sev}** · `{f.get('category', '—')}` · {line_info}  \n"
                    f"**Issue:** {f.get('issue', '')}  \n"
                    f"**Fix:** {f.get('suggested_fix', '')}"
                )
                st.divider()

        pii_cols = sec.get("pii_columns_detected", [])
        if pii_cols:
            st.markdown("#### PII Columns Detected")
            chips = " ".join(
                f'<span style="border:1px solid #ff4444;border-radius:4px;'
                f'padding:2px 8px;margin:2px;color:#ff6b6b">{c}</span>'
                for c in pii_cols
            )
            st.markdown(chips, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TAB 4 — Lineage
# ---------------------------------------------------------------------------
if _active_tab_label == "🔗 Lineage":
    lin = review.get("lineage")
    if lin is None:
        if errors.get("lineage") == "Not selected":
            st.info("Lineage agent skipped — re-run with the checkbox enabled to see results.")
        else:
            st.error(f"Lineage agent failed: {errors.get('lineage', 'unknown error')}")
    else:
        _job_slug_l = st.session_state.get("reviewed_job", "job").replace(" ", "_").lower()
        diagram_raw = lin.get("mermaid_diagram", "")
        diagram = diagram_raw.replace("\\n", "\n")
        _lin_dl_col, _lin_live_col = st.columns(2)
        with _lin_dl_col:
            st.download_button(
                "⬇️ Download diagram source (.mmd)",
                data=diagram,
                file_name=f"{_job_slug_l}_lineage.mmd",
                mime="text/plain",
                key="dl_lin",
            )
        with _lin_live_col:
            _live_state = json.dumps({"code": diagram, "mermaid": {"theme": "dark"}})
            _live_b64 = base64.urlsafe_b64encode(_live_state.encode("utf-8")).decode("ascii")
            st.link_button(
                "🔗 Open in mermaid.live",
                url=f"https://mermaid.live/edit#base64:{_live_b64}",
            )

        st.markdown(f"> {lin.get('dependency_summary', '')}")

        st.markdown("#### Data Flow Diagram")
        try:
            render_mermaid(diagram)
        except Exception as e:
            st.code(diagram, language="text")
            st.caption(f"(Mermaid render failed: {e})")

        with st.expander("🔧 Debug: raw mermaid source"):
            st.code(diagram, language="text")
            st.caption(f"Length: {len(diagram)} chars")

        st.markdown("---")
        lin_l, lin_r = st.columns(2)
        with lin_l:
            st.markdown("**Input Tables**")
            for t in lin.get("input_tables", []):
                size = t.get("estimated_size")
                size_str = f" *(~{size})*" if size else ""
                st.markdown(f"- **{t.get('name', '—')}**{size_str}: `{t.get('source', '—')}`")
        with lin_r:
            st.markdown("**Output Tables**")
            for t in lin.get("output_tables", []):
                st.markdown(f"- **{t.get('name', '—')}**: `{t.get('destination', '—')}`")

        st.markdown("#### Transformation Steps")
        for i, step in enumerate(lin.get("transformations", []), 1):
            st.markdown(f"{i}. {step}")

# ---------------------------------------------------------------------------
# TAB 5 — Ask
# ---------------------------------------------------------------------------
if _active_tab_label == "💬 Ask":
    if not st.session_state.get("review_result"):
        st.info(
            "💬 Run a review first to enable the assistant. "
            "Once a review completes, you can ask questions about "
            "the findings, explore past similar reviews, and "
            "reason about whether to approve."
        )
    else:
        _ask_review = st.session_state["review_result"]
        _ask_job = (
            st.session_state.get("reviewed_job")
            or st.session_state.get("uploaded_filename")
            or "current review"
        )
        _ask_paradigm = (_ask_review.get("performance") or {}).get("detected_paradigm")
        _ask_script = st.session_state.get("original_script_content")
        _ask_similar = st.session_state.get("similar_reviews") or []

        _system_prompt = build_system_prompt(
            job_name=_ask_job,
            paradigm=_ask_paradigm,
            original_script=_ask_script,
            review_result=_ask_review,
            similar_reviews=_ask_similar,
        )

        # Example prompts as quick-start chips
        st.caption("💡 Suggested questions:")
        _ask_col1, _ask_col2, _ask_col3 = st.columns(3)
        _example_prompts = [
            "What's the most important change to approve?",
            "Have we seen this pattern before?",
            "Explain the security findings in plain English",
        ]
        _clicked_example = None
        for _col, _prompt in zip([_ask_col1, _ask_col2, _ask_col3], _example_prompts):
            with _col:
                if st.button(_prompt, key=f"example_{hash(_prompt)}", use_container_width=True):
                    _clicked_example = _prompt

        st.divider()

        # Render existing chat history
        for _msg in st.session_state["assistant_messages"]:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])

        # Chat input — handles both example clicks and typed input
        _user_input = _clicked_example or st.chat_input("Ask about this review...")

        if _user_input:
            st.session_state["assistant_messages"].append(
                {"role": "user", "content": _user_input}
            )
            with st.chat_message("user"):
                st.markdown(_user_input)

            # Build messages array for Bedrock
            _history = trim_history(st.session_state["assistant_messages"])
            _bedrock_messages = [
                {"role": m["role"], "content": [{"text": m["content"]}]}
                for m in _history
            ]

            # Stream the response
            with st.chat_message("assistant"):
                _placeholder = st.empty()
                _accumulated = ""
                for _chunk in chat_stream(_bedrock_messages, _system_prompt):
                    _accumulated += _chunk
                    _placeholder.markdown(_accumulated + "▌")
                _placeholder.markdown(_accumulated)

            st.session_state["assistant_messages"].append(
                {"role": "assistant", "content": _accumulated}
            )
