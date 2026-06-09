"""
diagnostic_ui.py
----------------
Caterpillar C7 — Diagnostic Scan Tool (Detection Side)

Run with:  streamlit run diagnostic_ui.py
           streamlit run diagnostic_ui.py --server.port 8502

Data Sources
------------
  1. Local File   — reads live_stream.json written by test_cell_ui.py (same machine)
  2. Network URL  — fetches http://<ip>:<port>/stream from test_cell_ui.py over Wi-Fi
  3. CSV Playback — replays a batch CSV row-by-row at 10 Hz in a background thread

Detection Pipeline
------------------
  Raw sensor dict  ->  DataPipeline (median filter + steady-state)
                   ->  PhysicsEngine (Zone A/B/C residuals + cascade rules)
                   ->  MLEngine (autoencoder reconstruction error)
                   ->  fuse() (60% physics + 40% ML -> FusionDecision)

Do NOT modify: simulator.py, pipeline.py, physics_engine.py, ml_engine.py, fusion.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import json
import urllib.request
import urllib.error
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from pipeline       import DataPipeline
from physics_engine import PhysicsEngine, ZoneResult
from ml_engine      import MLEngine, MLResult
from fusion         import fuse, FusionDecision

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAT C7 — Diagnostic Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }
  .main .block-container { padding-top: 1rem; max-width: 100%; }

  /* ── Header ── */
  .cat-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d2040 50%, #091525 100%);
    border: 1px solid #0e7490;
    border-radius: 12px;
    padding: 1.2rem 2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .cat-title    { font-size:1.5rem; font-weight:900; color:#22d3ee;
                  letter-spacing:0.05em; text-transform:uppercase; }
  .cat-subtitle { font-size:0.75rem; color:#94a3b8; margin-top:2px; }

  /* ── Badges ── */
  .badge { display:inline-block; padding:0.4rem 1.2rem; border-radius:999px;
           font-weight:700; font-size:1rem; letter-spacing:0.08em; text-transform:uppercase; }
  .badge-pass    { background:#052e16; color:#4ade80; border:1px solid #16a34a; }
  .badge-monitor { background:#422006; color:#fb923c; border:1px solid #ea580c; }
  .badge-alert   { background:#450a0a; color:#f87171; border:1px solid #dc2626;
                   animation:pulse-red 1s ease-in-out infinite; }
  .badge-log     { background:#172554; color:#93c5fd; border:1px solid #3b82f6; }
  .badge-warm    { background:#1a1a2e; color:#94a3b8; border:1px solid #475569; }
  @keyframes pulse-red { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.4);}
                         50%    {box-shadow:0 0 0 8px rgba(239,68,68,0);} }

  /* ── Source selector panel ── */
  .source-panel { background:#0d1526; border:1px solid #0e7490; border-radius:10px;
                  padding:0.8rem 1.2rem; margin-bottom:1rem; }

  /* ── Sensor cards ── */
  .sensor-card { background:#0f1929; border:1px solid #1e3a5f; border-radius:10px;
                 padding:0.75rem 1rem; margin-bottom:0.45rem;
                 transition:border-color 0.3s; }
  .sensor-card.warn { border-color:#dc2626; background:#1a0a0a; }
  .sensor-label { font-size:0.68rem; color:#64748b; text-transform:uppercase;
                  letter-spacing:0.05em; }
  .sensor-value { font-family:'JetBrains Mono',monospace; font-size:1.1rem;
                  font-weight:700; color:#e2e8f0; }
  .sensor-unit  { font-size:0.68rem; color:#94a3b8; }

  /* ── Residual gauge ── */
  .gauge-label { font-size:0.7rem; color:#64748b; text-transform:uppercase; margin-bottom:4px; }
  .gauge-value { font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; }
  .gauge-ok    { color:#4ade80; }
  .gauge-warn  { color:#fb923c; }
  .gauge-crit  { color:#f87171; }

  /* ── Alert box ── */
  .alert-box { border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.75rem;
               background:#0f1929; border:1px solid #1e3a5f; }
  .alert-box.red   { background:#1a0808; border-color:#dc2626;
                     animation:pulse-border 1.2s ease-in-out infinite; }
  .alert-box.orange{ background:#1a1008; border-color:#ea580c; }
  .alert-box.green { background:#06110e; border-color:#16a34a; }
  .alert-zone   { font-size:2rem; font-weight:900; }
  .alert-sub    { font-size:0.75rem; color:#94a3b8; margin-top:4px; word-break:break-word; }
  .alert-action { font-size:0.78rem; color:#e2e8f0; margin-top:8px; padding-top:8px;
                  border-top:1px solid #1e293b; }
  @keyframes pulse-border { 0%,100%{box-shadow:0 0 0 0 rgba(220,38,38,0.3);}
                             50%    {box-shadow:0 0 15px rgba(220,38,38,0.3);} }

  /* ── Playback panel ── */
  .playback-panel { background:#071520; border:1px solid #0e7490; border-radius:10px;
                    padding:0.8rem 1.2rem; margin-bottom:0.8rem; }
  .playback-running { border-color:#22d3ee !important;
                      box-shadow:0 0 12px rgba(34,211,238,0.15); }

  /* ── Network panel ── */
  .net-box { background:#0a1225; border:1px solid #1e40af; border-radius:8px;
             padding:0.7rem 1rem; font-family:'JetBrains Mono',monospace;
             font-size:0.8rem; color:#93c5fd; line-height:1.9; margin-bottom:0.6rem; }

  /* ── Progress bar track ── */
  .bar-track { background:#1e293b; border-radius:999px; height:7px; margin-top:5px; }
  .bar-fill  { height:7px; border-radius:999px; transition:width 0.4s ease; }

  /* ── Confidence bar ── */
  .conf-card { background:#0f1929; border:1px solid #1e3a5f; border-radius:10px;
               padding:0.8rem 1rem; margin-bottom:0.45rem; }

  /* ── Section title ── */
  .sec-title { font-size:0.75rem; font-weight:600; color:#22d3ee;
               text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem; }

  /* scrollbar */
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:#0a0e1a; }
  ::-webkit-scrollbar-thumb { background:#0e7490; border-radius:3px; }

  [data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace;
                                   font-size:1.1rem !important; color:#e2e8f0 !important; }
  [data-testid="stMetricLabel"] { color:#64748b !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Boot ────────────────────────────────────────────────────────

def _init_session():
    if "diag_initialized" in st.session_state:
        return

    pipe = DataPipeline()
    pe   = PhysicsEngine()
    ml   = MLEngine()
    try:
        ml.load()
    except FileNotFoundError:
        pass  # physics-only mode

    HIST = 600   # 60 s × 10 Hz
    shared = {
        # Detection output (written by background thread, read by UI)
        "decision":  FusionDecision(status="WARMING_UP"),
        "raw":       {},
        "filt":      {},
        "res_a":     0.0,
        "res_b":     0.0,
        "res_c":     0.0,
        "hist": {
            "t":          deque(maxlen=HIST),
            "res_a":      deque(maxlen=HIST),
            "res_b":      deque(maxlen=HIST),
            "res_c":      deque(maxlen=HIST),
            "confidence": deque(maxlen=HIST),
        },
        # Thread control
        "running":          True,
        "source":           "local",   # "local" | "network" | "csv"
        "net_url":          "",
        "csv_df":           None,
        "csv_idx":          0,
        "playback_running": False,
        "step_count":       0,
        "stream_error":     "",
        # Physics zone tracking (for inter-run cascade continuity)
        "last_ra": ZoneResult("A"),
        "last_rb": ZoneResult("B"),
        "last_rc": ZoneResult("C"),
    }
    lock = threading.Lock()

    # ── Background Detection Thread ──────────────────────────────────────────
    # Polls the data source at 10 Hz, routes through full pipeline, stores result.
    # Key design rules:
    #   - Tracks prev_source; resets pipeline+engines when source changes so
    #     stale rolling buffers from the old source never contaminate the new one.
    #   - For local file: tracks mtime so a stopped broadcast doesn't keep
    #     re-feeding the same stale row into the pipeline.
    #   - PermissionError on Windows file rename is silently skipped (use cached).
    def diag_loop():
        prev_source   = None
        local_mtime   = 0.0   # mtime of last successfully read live_stream.json

        while shared["running"]:
            t0     = time.perf_counter()
            source = shared["source"]
            raw    = None

            # ── 0. Reset pipeline if source changed ──────────────────────────
            if source != prev_source:
                pipe.reset()
                pe.__init__()           # fresh PhysicsEngine (clears cascade state)
                with lock:
                    shared["step_count"] = 0
                    shared["decision"]   = FusionDecision(status="WARMING_UP")
                    shared["res_a"]      = 0.0
                    shared["res_b"]      = 0.0
                    shared["res_c"]      = 0.0
                    shared["last_ra"]    = ZoneResult("A")
                    shared["last_rb"]    = ZoneResult("B")
                    shared["last_rc"]    = ZoneResult("C")
                    h = shared["hist"]
                    for q in h.values():
                        q.clear()
                local_mtime = 0.0
                prev_source = source

            # ── 1. Acquire one sensor row ───────────────────────────────────
            if source == "local":
                try:
                    p = Path(__file__).parent / "live_stream.json"
                    if p.exists():
                        cur_mtime = p.stat().st_mtime
                        if cur_mtime > local_mtime:
                            # Only read if file was updated since last read.
                            # Use raw_decode() so a Windows rename race that
                            # delivers two concatenated JSON objects (Extra data)
                            # only consumes the first valid one and ignores
                            # any trailing bytes.
                            text = p.read_text()
                            obj, _ = json.JSONDecoder().raw_decode(text)
                            raw = obj
                            local_mtime = cur_mtime
                            shared["stream_error"] = ""
                        # else: file unchanged (DAQ stopped) — skip stale data
                    else:
                        shared["stream_error"] = "Waiting for live_stream.json …"
                except PermissionError:
                    pass   # Windows atomic rename race — skip this tick silently
                except json.JSONDecodeError:
                    pass   # Transient corruption mid-write — retry next 100 ms tick
                except Exception as e:
                    shared["stream_error"] = f"File read error: {e}"

            elif source == "network":
                url = shared.get("net_url", "")
                if url:
                    try:
                        req = urllib.request.urlopen(url, timeout=1)
                        raw = json.loads(req.read().decode())
                        shared["stream_error"] = ""
                    except urllib.error.URLError as e:
                        shared["stream_error"] = f"Network error: {e.reason}"
                    except Exception as e:
                        shared["stream_error"] = f"Fetch error: {e}"
                else:
                    shared["stream_error"] = "Enter a network URL above."

            elif source == "csv":
                if shared.get("playback_running"):
                    df  = shared.get("csv_df")
                    idx = shared.get("csv_idx", 0)
                    if df is not None and idx < len(df):
                        raw = df.iloc[idx].to_dict()
                        shared["csv_idx"] = idx + 1
                        shared["stream_error"] = ""
                    elif df is not None:
                        shared["playback_running"] = False
                        shared["stream_error"] = "✅ Playback complete"

            # ── 2. Run pipeline if we have new data ─────────────────────────
            if raw:
                pr = pipe.process(raw)
                shared["step_count"] += 1
                sc = shared["step_count"]

                # Detection cadence:
                #   Live / Network — every 10 rows (1 Hz) AND need full 10-s
                #                    steady-state window (100 rows of stable RPM/Load).
                #   CSV Playback   — every row AND only need the median-filter
                #                    window (≥15 rows).  The data is pre-recorded at
                #                    a stable point; waiting for the 10-s steady-state
                #                    window would waste 15% of a 60-s CSV as warmup.
                is_csv       = (source == "csv")
                detect_now   = (sc % 10 == 0) if not is_csv else True
                data_ready   = pr.steady_state if not is_csv else (sc >= 15)

                if detect_now:
                    if data_ready:
                        ra, rb, rc = pe.run(pr.filt, pr.raw)
                        ml_res     = ml.run(pr.filt) if ml.is_loaded else MLResult()
                        decision   = fuse(ra, rb, rc, ml_res,
                                         raw.get("timestamp", 0.0))
                        with lock:
                            shared["decision"]  = decision
                            shared["raw"]       = raw
                            shared["filt"]      = pr.filt
                            shared["res_a"]     = ra.residual_pct
                            shared["res_b"]     = rb.residual_pct
                            shared["res_c"]     = rc.residual_pct
                            shared["last_ra"]   = ra
                            shared["last_rb"]   = rb
                            shared["last_rc"]   = rc
                            h = shared["hist"]
                            h["t"].append(raw.get("timestamp", 0.0))
                            h["res_a"].append(ra.residual_pct)
                            h["res_b"].append(rb.residual_pct)
                            h["res_c"].append(rc.residual_pct)
                            h["confidence"].append(decision.confidence_pct)
                    else:
                        with lock:
                            shared["decision"] = FusionDecision(
                                status="WARMING_UP",
                                timestamp=raw.get("timestamp", 0.0),
                            )

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 0.1 - elapsed))

    threading.Thread(target=diag_loop, daemon=True, name="DiagLoop").start()

    st.session_state.update({
        "diag_initialized": True,
        "pipe":             pipe,
        "pe":               pe,
        "ml":               ml,
        "shared":           shared,
        "lock":             lock,
        "ml_loaded":        ml.is_loaded,
        "selected_source":  "Local File  (same laptop)",
        "net_url_input":    "http://192.168.1.100:5000/stream",
    })


_init_session()

shared = st.session_state.shared
lock   = st.session_state.lock

# ── Single atomic snapshot of ALL shared state ───────────────────────────────
# Everything the UI renders must come from ONE lock acquisition.
# A second lock later in the same rerun would race the background thread and
# produce mismatched values (e.g. Zone C showing two different residuals).
with lock:
    d           = shared["decision"]
    raw         = dict(shared.get("raw",  {}))
    filt        = dict(shared.get("filt", {}))
    res_a       = shared["res_a"]
    res_b       = shared["res_b"]
    res_c       = shared["res_c"]
    # Zone objects — captured here so zone_detail() uses the same snapshot
    snap_ra     = shared["last_ra"]
    snap_rb     = shared["last_rb"]
    snap_rc     = shared["last_rc"]
    h           = {k: list(v) for k, v in shared["hist"].items()}
    s_err       = shared.get("stream_error", "")
    step_n      = shared.get("step_count", 0)
    csv_running = shared.get("playback_running", False)
    csv_idx     = shared.get("csv_idx", 0)
    csv_df_len  = len(shared["csv_df"]) if shared.get("csv_df") is not None else 0


# ─── Header ───────────────────────────────────────────────────────────────────
status     = d.status or "WARMING_UP"
badge_cls  = {
    "PASS":       "badge-pass",
    "MONITOR":    "badge-monitor",
    "ALERT":      "badge-alert",
    "LOG":        "badge-log",
    "WARMING_UP": "badge-warm",
    "SUPPRESSED": "badge-warm",
}.get(status, "badge-warm")

ts_str = f"{d.timestamp:.1f} s" if d.timestamp else "—"

st.markdown(f"""
<div class="cat-header">
  <div>
    <div class="cat-title">🔬 CAT C7 — Diagnostic Scan Tool</div>
    <div class="cat-subtitle">
      Physics Engine · ML Autoencoder · Fusion Layer · Real-time Anomaly Detection
    </div>
  </div>
  <div style="text-align:right">
    <span class="badge {badge_cls}">{status}</span>
    <div class="cat-subtitle" style="margin-top:6px">
      Engine time: {ts_str} &nbsp;|&nbsp;
      ML: {"✅ Loaded" if st.session_state.ml_loaded else "⚠ Physics-only"} &nbsp;|&nbsp;
      Samples: {step_n:,}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Data Source Selector ──────────────────────────────────────────────────────
# Use a CSS-styled container rather than unclosed HTML div wrappers.
# Unclosed st.markdown divs break Streamlit's React reconciliation, causing
# layout shifts, duplicate widget renders, and labels below their borders.
with st.container(border=True):
    st.markdown('<div class="sec-title">🔌 Data Source</div>', unsafe_allow_html=True)

src_col, cfg_col = st.columns([1, 2], gap="large")

with src_col:
    source_choice = st.radio(
        "Select data origin",
        options=[
            "Local File  (same laptop)",
            "Network URL  (different laptop)",
            "CSV Playback  (batch replay)",
        ],
        key="selected_source",
        label_visibility="collapsed",
    )

with cfg_col:
    if "Local File" in source_choice:
        shared["source"] = "local"
        stream_path = Path(__file__).parent / "live_stream.json"
        exists = stream_path.exists()
        age_s  = ""
        if exists:
            age = time.time() - stream_path.stat().st_mtime
            age_s = f"  (last written {age:.1f}s ago)"
        st.markdown(f"""
        <div class="net-box">
          📁 Reading: <b>live_stream.json</b>{age_s}<br>
          {"✅ File found — stream active" if exists else "⏳ Waiting for test_cell_ui.py to start broadcasting…"}
        </div>
        """, unsafe_allow_html=True)
        if s_err:
            st.warning(s_err)

    elif "Network URL" in source_choice:
        shared["source"] = "network"
        net_url = st.text_input(
            "Network stream URL",
            value=st.session_state.get("net_url_input", "http://192.168.1.100:5000/stream"),
            key="net_url_input",
            placeholder="http://<test-cell-IP>:<port>/stream",
        )
        shared["net_url"] = net_url
        st.markdown(
            '<p style="font-size:0.75rem;color:#64748b;margin-top:0.2rem">'
            'Enter the URL shown in the header of test_cell_ui.py running on the other laptop.'
            '</p>',
            unsafe_allow_html=True,
        )
        if s_err:
            st.error(s_err)
        elif step_n > 0:
            st.success(f"✅ Receiving data — {step_n:,} samples processed")

    elif "CSV Playback" in source_choice:
        shared["source"] = "csv"
        uploaded = st.file_uploader(
            "Upload engine log CSV",
            type=["csv"],
            key="csv_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            # Always refresh the DataFrame reference when a file is uploaded
            # (Streamlit re-runs the script on every upload)
            if not csv_running:
                shared["csv_df"]  = df_up
                shared["csv_idx"] = 0
            pb1, pb2 = st.columns([2, 1])
            with pb1:
                if not csv_running:
                    if st.button("▶ Start Playback", key="pb_start",
                                 type="primary", use_container_width=True):
                        # Reset pipeline buffers so CSV starts clean
                        pipe = st.session_state.pipe
                        pipe.reset()
                        shared["csv_df"]           = df_up
                        shared["csv_idx"]           = 0
                        shared["step_count"]        = 0
                        shared["playback_running"]  = True
                        shared["stream_error"]      = ""
                        st.rerun()
                else:
                    if st.button("⏹ Stop Playback", key="pb_stop", use_container_width=True):
                        shared["playback_running"] = False
            with pb2:
                if csv_df_len > 0:
                    pct = min(100, csv_idx / csv_df_len * 100)
                    st.markdown(f"""
                    <div style="padding-top:0.4rem">
                      <div style="font-size:0.7rem;color:#64748b">{csv_idx}/{csv_df_len} rows ({pct:.0f}%)</div>
                      <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:#22d3ee"></div></div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Upload a CSV generated by the Test Cell DAQ Controller to begin playback.", icon="📂")


# ─── Module-level helpers (defined OUTSIDE any column context) ────────────────
# CRITICAL: defining these inside a `with col:` block causes Streamlit to lose
# track of which column `st.markdown` should render into during rapid reruns,
# producing duplicate Zone B / Zone C cards.

THRESH = {"A": 8.0, "B": 6.0, "C": 12.0}


def _zone_detail_html(label: str, result, threshold: float) -> str:
    """Return the HTML for one zone detail card (does NOT call st.markdown)."""
    pct    = min(100, abs(result.residual_pct) / max(threshold, 0.01) * 100)
    colour = "#4ade80" if pct < 50 else "#fb923c" if pct < 85 else "#f87171"
    flag_t = "🚩 FLAGGED" if result.flag else ("🔇 SUPPRESSED" if result.suppressed else "✅ OK")
    flag_c = "#f87171" if result.flag else "#94a3b8"
    drift_t = " | 📈 DRIFT" if result.drift else ""
    sign   = "+" if result.residual_pct > 0 else ""
    return f"""
    <div class="sensor-card" style="margin-bottom:0.4rem;padding:0.6rem 1rem">
      <div style="display:flex;justify-content:space-between">
        <div style="font-size:0.72rem;color:#94a3b8;font-weight:600">{label}</div>
        <div style="font-size:0.72rem;color:{flag_c};font-weight:700">{flag_t}{drift_t}</div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px">
        <div style="font-size:0.68rem;color:#475569">
          Exp: {result.expected:.1f} &nbsp; Act: {result.actual:.1f}
        </div>
        <div style="font-family:'JetBrains Mono';font-size:0.95rem;font-weight:700;color:{colour}">
          {sign}{result.residual_pct:.1f}%
        </div>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{pct:.1f}%;background:{colour}"></div>
      </div>
    </div>"""


# ─── Main Output Panel ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_telem, col_analytics, col_alert = st.columns([1.2, 1.2, 1.4], gap="large")


# ────────────────────────────────────────────────────────────────────────────────
# COLUMN 1 — Telemetry Panel
# ────────────────────────────────────────────────────────────────────────────────
with col_telem:
    st.markdown('<div class="sec-title">📡 Telemetry — Live Sensor Values</div>',
                unsafe_allow_html=True)

    def sensor_card(label, value, unit, warn=False):
        cls = "sensor-card warn" if warn else "sensor-card"
        st.markdown(f"""
        <div class="{cls}" style="padding:0.4rem 1rem;margin-bottom:0.35rem">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div class="sensor-label" style="margin-bottom:0">{label}</div>
            <div><span class="sensor-value" style="font-size:0.95rem">{value}</span>
                 <span class="sensor-unit"> {unit}</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Primary sensors
    rpm_v  = filt.get("rpm",  raw.get("rpm",  0))
    load_v = raw.get("load_pct", filt.get("load_pct", 0))
    maf_v  = filt.get("maf_gs", 0)
    map_v  = filt.get("map_kpa", 0)
    ebp_v  = filt.get("ebp_kpa", 0)
    iat_v  = filt.get("iat_c", 0)
    ic_v   = filt.get("intercooler_outlet_c", 0)
    bst_v  = filt.get("boost_temp_c", 0)
    lam_v  = filt.get("lambda_ratio", 0)
    trb_v  = filt.get("n_turbo_rpm", 0)
    fuel_v = filt.get("fuel_rate_gs", 0)

    sensor_card("Engine Speed",    f"{rpm_v:,.0f}",  "RPM")
    sensor_card("Engine Load",     f"{load_v:.1f}",  "%")
    sensor_card("Mass Air Flow",   f"{maf_v:.1f}",   "g/s",
                warn=(d.zone == "A" and d.leak_detected))
    sensor_card("Intake Pressure", f"{map_v:.1f}",   "kPa",
                warn=(d.zone == "B" and d.leak_detected))
    sensor_card("Exh. Backpressure", f"{ebp_v:.1f}", "kPa",
                warn=(d.zone == "C" and d.leak_detected))
    sensor_card("Intake Air Temp", f"{iat_v:.1f}",   "°C")
    sensor_card("Boost Temp",      f"{bst_v:.1f}",   "°C")
    sensor_card("Intercooler Out", f"{ic_v:.1f}",    "°C")
    sensor_card("Turbo Speed",     f"{trb_v/1000:.1f}k", "RPM")
    sensor_card("Lambda λ",        f"{lam_v:.3f}",   "")
    sensor_card("Fuel Rate",       f"{fuel_v:.2f}",  "g/s")

    # EGT cascade strip
    st.markdown('<div class="sec-title" style="margin-top:0.6rem">🌡️ EGT Cascade</div>',
                unsafe_allow_html=True)
    egt_vals = [
        ("EGT-1 Manifold", filt.get("egt_1_c", 0)),
        ("EGT-2 Turbine",  filt.get("egt_2_c", 0)),
        ("EGT-3 DOC",      filt.get("egt_3_c", 0)),
        ("EGT-4 DPF",      filt.get("egt_4_c", 0)),
        ("EGT-5 SCR",      filt.get("egt_5_c", 0)),
    ]
    for name, val in egt_vals:
        sensor_card(name, f"{val:.0f}", "°C")


# ────────────────────────────────────────────────────────────────────────────────
# COLUMN 2 — Analytics Grid (Residuals + ML)
# ────────────────────────────────────────────────────────────────────────────────
with col_analytics:
    st.markdown('<div class="sec-title">📊 Zone Residuals vs Threshold</div>',
                unsafe_allow_html=True)

    # THRESH is defined at module level above — do NOT redefine it here.

    def residual_gauge(zone_name, residual, threshold, sensor_label):
        pct    = min(100, abs(residual) / max(threshold, 0.01) * 100)
        colour = "#4ade80" if pct < 50 else "#fb923c" if pct < 85 else "#f87171"
        cls    = "gauge-ok"  if pct < 50 else "gauge-warn" if pct < 85 else "gauge-crit"
        sign   = "+" if residual > 0 else ""
        st.markdown(f"""
        <div class="sensor-card" style="margin-bottom:0.5rem;padding:0.6rem 1rem">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div class="gauge-label" style="margin-bottom:0">
              Zone {zone_name} — {sensor_label}
            </div>
            <div class="{cls}" style="font-family:'JetBrains Mono',monospace;
                 font-size:1.1rem;font-weight:700">
              {sign}{residual:.1f}%
            </div>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{colour}"></div>
          </div>
          <div style="font-size:0.6rem;color:#475569;margin-top:2px">
            Threshold: ±{threshold:.1f}%
          </div>
        </div>""", unsafe_allow_html=True)

    residual_gauge("A", res_a, THRESH["A"], "MAF")
    residual_gauge("B", res_b, THRESH["B"], "MAP")
    residual_gauge("C", res_c, THRESH["C"], "EBP")

    # ML Engine block
    st.markdown('<div class="sec-title" style="margin-top:0.8rem">🤖 ML Autoencoder</div>',
                unsafe_allow_html=True)

    ml_err  = d.ml_recon_error or 0.0
    ml_flag = d.ml_flag
    ml_col  = "#f87171" if ml_flag else "#4ade80"
    ml_txt  = f"⚠ ANOMALY — {d.ml_worst_feature}" if ml_flag else "✅ Normal reconstruction"

    st.markdown(f"""
    <div class="sensor-card">
      <div class="gauge-label">Reconstruction Error</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;
           color:{ml_col};font-weight:700">{ml_err:.6f}</div>
      <div style="font-size:0.8rem;color:#94a3b8;margin-top:4px">{ml_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    # Fusion scores
    st.markdown(f"""
    <div class="sensor-card" style="margin-top:0.5rem">
      <div class="gauge-label">Fusion Scores (Physics 60% · ML 40%)</div>
      <div style="display:flex;justify-content:space-between;margin-top:8px;gap:0.5rem">
        <div style="flex:1;text-align:center">
          <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase">Physics</div>
          <div style="font-family:'JetBrains Mono';font-size:1.2rem;color:#facc15;font-weight:700">
            {d.physics_score:.1f}
          </div>
        </div>
        <div style="flex:1;text-align:center">
          <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase">ML</div>
          <div style="font-family:'JetBrains Mono';font-size:1.2rem;color:#facc15;font-weight:700">
            {d.ml_score:.1f}
          </div>
        </div>
        <div style="flex:1;text-align:center;border-left:1px solid #1e3a5f;padding-left:0.5rem">
          <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase">Fused</div>
          <div style="font-family:'JetBrains Mono';font-size:1.4rem;color:#e2e8f0;font-weight:900">
            {d.fused_score:.1f}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ECU flags
    st.markdown('<div class="sec-title" style="margin-top:0.8rem">⚙️ ECU Status</div>',
                unsafe_allow_html=True)
    dpf_on = bool(raw.get("dpf_regen", 0))
    egr_p  = raw.get("egr_pct", 15)
    cool   = raw.get("coolant_temp_c", 88)
    dpf_c  = "#fb923c" if dpf_on else "#4ade80"
    st.markdown(f"""
    <div class="sensor-card">
      <div style="font-size:0.88rem;line-height:2.1">
        DPF Regen: <b style="float:right;color:{dpf_c}">{'ACTIVE' if dpf_on else 'OFF'}</b><br>
        EGR Position: <b style="float:right;color:#e2e8f0">{egr_p:.1f}%</b><br>
        Coolant Temp: <b style="float:right;color:#e2e8f0">{cool:.1f} °C</b>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# COLUMN 3 — Alert Matrix
# ────────────────────────────────────────────────────────────────────────────────
with col_alert:
    st.markdown('<div class="sec-title">🚨 Detection Output / Alert Matrix</div>',
                unsafe_allow_html=True)

    if d.leak_detected:
        # Choose colour severity
        if status == "ALERT":
            box_cls   = "alert-box red"
            hdr_color = "#f87171"
        else:
            box_cls   = "alert-box orange"
            hdr_color = "#fb923c"

        zone_emoji = {"A": "💨", "B": "💨", "C": "🔥"}.get(d.zone, "⚠")
        sub_display = (d.sub_location
                       .replace("_", " ").title()
                       .replace("Doc", "DOC").replace("Dpf", "DPF")
                       .replace("Scr", "SCR"))

        st.markdown(f"""
        <div class="{box_cls}">
          <div class="alert-zone" style="color:{hdr_color}">
            {zone_emoji} ZONE {d.zone} LEAK
          </div>
          <div style="font-size:0.8rem;color:#94a3b8;margin-top:4px">
            Status: <b style="color:#e2e8f0">{status}</b>
            &nbsp;|&nbsp; Triggered by:
            <b style="color:#e2e8f0">{d.triggered_by.upper()}</b>
          </div>
          <div class="alert-sub">📍 {sub_display}</div>
          <div class="alert-action">🔧 {d.action}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        conf      = d.confidence_pct
        conf_col  = "#dc2626" if conf > 80 else "#ea580c" if conf > 50 else "#facc15"
        st.markdown(f"""
        <div class="conf-card">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div class="gauge-label">Detection Confidence</div>
            <div style="font-family:'JetBrains Mono';font-size:1.5rem;
                 font-weight:900;color:{conf_col}">{conf:.1f}%</div>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{conf:.1f}%;background:{conf_col}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Cascade / suppression note
        if d.cascade_zones:
            st.markdown(f"""
            <div class="alert-box" style="padding:0.6rem 1rem;margin-top:0.4rem">
              <div style="font-size:0.72rem;color:#94a3b8">
                ⚡ <b>Cascade Note:</b> {d.cascade_zones}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Confirming sensors
        if d.confirmed_by:
            st.markdown(f"""
            <div class="sensor-card" style="margin-top:0.4rem">
              <div class="gauge-label">Confirming Sensors</div>
              <div style="font-size:0.85rem;color:#93c5fd;margin-top:4px">
                {d.confirmed_by.replace('+', ' · ')}
              </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Nominal / warm-up / suppressed states
        if status == "WARMING_UP":
            icon = "⏳"
            msg  = "Warming Up…"
            is_csv_mode = shared.get("source") == "csv"
            if is_csv_mode:
                rows_left = max(0, 15 - step_n)
                desc = (
                    f"Filling median filter — {rows_left} more row(s) before detection starts."
                    if rows_left > 0
                    else "Median filter ready. Running detection…"
                )
            else:
                desc = "Collecting steady-state history. Detection begins after 10 s of stable RPM/Load."
            box_c = "alert-box"
        elif status == "SUPPRESSED":
            icon = "🔇"; msg = "Detection Suppressed"; desc = f"Reason: {d.suppression_reason or 'Edge-case condition active (DPF regen / transient).'}"
            box_c = "alert-box"
        else:
            icon = "✅"; msg = "System Nominal"; desc = "All zone residuals within normal bounds. No leaks detected."
            box_c = "alert-box green"

        st.markdown(f"""
        <div class="{box_c}">
          <div class="alert-zone">{icon} {msg}</div>
          <div class="alert-sub">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Zone result detail cards ─────────────────────────────────────────────
    # col_alert.markdown() is used DIRECTLY (not st.markdown inside a with block)
    # to guarantee the HTML lands in col_alert — immune to column-context drift
    # during rapid st.rerun() cycles that caused duplicated Zone B / Zone C cards.
    col_alert.markdown(
        '<div class="sec-title" style="margin-top:0.8rem">🔍 Zone Detail</div>',
        unsafe_allow_html=True,
    )
    col_alert.markdown(
        _zone_detail_html("Zone A — MAF (Pre-Turbo)",  snap_ra, THRESH["A"]) +
        _zone_detail_html("Zone B — MAP (Charge-Air)",  snap_rb, THRESH["B"]) +
        _zone_detail_html("Zone C — EBP (Exhaust)",     snap_rc, THRESH["C"]),
        unsafe_allow_html=True,
    )




# ─── History Chart ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-title">📈 60-Second Residual History</div>',
            unsafe_allow_html=True)

if len(h["t"]) > 5:
    df_hist = pd.DataFrame({
        "Time (s)":             h["t"],
        "Zone A residual (%)":  h["res_a"],
        "Zone B residual (%)":  h["res_b"],
        "Zone C residual (%)":  h["res_c"],
    }).set_index("Time (s)")

    if df_hist.index[-1] - df_hist.index[0] > 60:
        df_hist = df_hist[df_hist.index >= df_hist.index[-1] - 60]

    st.line_chart(df_hist, height=200, use_container_width=True)

    # Confidence history overlay
    if len(h["confidence"]) > 5:
        df_conf = pd.DataFrame({
            "Time (s)":           h["t"][-len(h["confidence"]):],
            "Confidence (%)":     h["confidence"],
        }).set_index("Time (s)")
        st.line_chart(df_conf, height=120, use_container_width=True)
else:
    st.info("History will appear after warm-up (~10 s of steady-state data).", icon="⏳")


# ─── Auto-refresh ─────────────────────────────────────────────────────────────
# Use shorter sleep during CSV playback (closer to real-time feel)
sleep_s = 0.15 if (shared.get("source") == "csv" and csv_running) else 0.5
time.sleep(sleep_s)
st.rerun()
