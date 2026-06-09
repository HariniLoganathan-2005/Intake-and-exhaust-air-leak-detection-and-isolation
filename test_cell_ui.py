"""
test_cell_ui.py
---------------
Caterpillar C7 — Test Cell DAQ Controller (Simulator Side)

Run with:  streamlit run test_cell_ui.py
           streamlit run test_cell_ui.py --server.port 8501

Architecture
------------
  • Always steps the engine simulator at 10 Hz in a background thread.
  • State HTTP server (random free port) feeds the embedded flow animation
    AND acts as a network data endpoint for diagnostic_ui.py on another machine.
  • Live DAQ toggle: when ON, atomically writes raw sensor dict to
    live_stream.json every 100 ms (local same-laptop comms).
  • Network clients connect to http://<this-machine-IP>:<port>/stream.
  • CSV Batch Generator: run_batch() in a snapshot simulator, instant download.

Do NOT modify: simulator.py, pipeline.py, physics_engine.py, ml_engine.py, fusion.py
"""

import streamlit as st
import pandas as pd
import threading
import time
import json
import io
import sys
import socket
import socketserver
import http.server
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simulator import EngineSimulator
import streamlit.components.v1 as components

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAT C7 — Test Cell DAQ",
    page_icon="🏭",
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
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #4f46e5;
    border-radius: 12px;
    padding: 1.2rem 2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .cat-title { font-size: 1.5rem; font-weight: 900; color: #facc15;
               letter-spacing: 0.05em; text-transform: uppercase; }
  .cat-subtitle { font-size: 0.75rem; color: #94a3b8; margin-top: 2px; }

  /* ── Badges ── */
  .badge { display: inline-block; padding: 0.4rem 1.2rem; border-radius: 999px;
           font-weight: 700; font-size: 1rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .badge-pass   { background:#052e16; color:#4ade80; border:1px solid #16a34a; }
  .badge-warm   { background:#1a1a2e; color:#94a3b8; border:1px solid #475569; }
  .badge-purple { background:#2e1065; color:#c084fc; border:1px solid #7c3aed;
                  animation: pulse-purple 1.5s ease-in-out infinite; }
  .badge-red    { background:#450a0a; color:#f87171; border:1px solid #dc2626;
                  animation: pulse-red 1s ease-in-out infinite; }
  @keyframes pulse-purple { 0%,100%{box-shadow:0 0 0 0 rgba(124,58,237,0.4);} 50%{box-shadow:0 0 0 8px rgba(124,58,237,0);} }
  @keyframes pulse-red    { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.4);}  50%{box-shadow:0 0 0 8px rgba(239,68,68,0);} }

  /* ── Cards ── */
  .sensor-card { background:#0f1929; border:1px solid #1e3a5f; border-radius:10px;
                 padding:0.8rem 1rem; margin-bottom:0.5rem; }
  .sensor-label { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }
  .sensor-value { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:#e2e8f0; }
  .sensor-unit  { font-size:0.7rem; color:#94a3b8; }

  /* ── Panels ── */
  .control-section { background:#0d1526; border:1px solid #1e3a5f;
                     border-radius:10px; padding:1rem 1.2rem; height:100%; }
  .control-title   { font-size:0.75rem; font-weight:600; color:#facc15;
                     text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem; }
  .leak-active     { border-color:#dc2626 !important; background:#120808 !important; }

  /* ── DAQ panel ── */
  .daq-panel { background:linear-gradient(135deg,#0d0d2b 0%,#1a1040 100%);
               border:2px solid #7c3aed; border-radius:12px; padding:1.2rem 1.5rem; }
  .daq-live  { border-color:#c084fc !important;
               box-shadow:0 0 20px rgba(192,132,252,0.15);
               animation:pulse-glow 2s ease-in-out infinite; }
  @keyframes pulse-glow { 0%,100%{box-shadow:0 0 10px rgba(192,132,252,0.15);}
                           50%    {box-shadow:0 0 25px rgba(192,132,252,0.35);} }

  /* ── CSV panel ── */
  .csv-panel { background:#071520; border:2px solid #0e7490;
               border-radius:12px; padding:1.2rem 1.5rem; }

  /* ── Network info box ── */
  .net-box { background:#0a1225; border:1px solid #1e40af; border-radius:8px;
             padding:0.7rem 1rem; font-family:'JetBrains Mono',monospace;
             font-size:0.8rem; color:#93c5fd; line-height:1.9; margin-top:0.6rem; }

  /* ── Mini metric cards ── */
  .mini-metric { background:#0f1929; border:1px solid #1e3a5f; border-radius:8px;
                 padding:0.5rem 0.4rem; text-align:center; }
  .mini-label  { font-size:0.58rem; color:#64748b; text-transform:uppercase; letter-spacing:0.04em; }
  .mini-val    { font-family:'JetBrains Mono',monospace; font-size:0.95rem;
                 font-weight:700; color:#e2e8f0; }

  /* ── Zone injection highlight ── */
  .zone-a { border-left:3px solid #3b82f6 !important; }
  .zone-b { border-left:3px solid #f87171 !important; }
  .zone-c { border-left:3px solid #c084fc !important; }

  /* scrollbar */
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:#0a0e1a; }
  ::-webkit-scrollbar-thumb { background:#7c3aed; border-radius:3px; }

  [data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace;
                                   font-size:1.1rem !important; color:#e2e8f0 !important; }
  [data-testid="stMetricLabel"] { color:#64748b !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _has_leak(sim: EngineSimulator) -> bool:
    return (sim.leak.zone_a_severity > 0 or
            sim.leak.zone_b_severity > 0 or
            sim.leak.zone_c_severity > 0)

def _active_zone(sim: EngineSimulator):
    if sim.leak.zone_a_severity > 0: return "A"
    if sim.leak.zone_b_severity > 0: return "B"
    if sim.leak.zone_c_severity > 0: return "C"
    return None

def _active_sub(sim: EngineSimulator):
    if sim.leak.zone_b_severity > 0: return sim.leak.zone_b_location
    if sim.leak.zone_c_severity > 0: return sim.leak.zone_c_location
    return None

def _own_ip() -> str:
    try:
        # Connect to an external address to find the LAN IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─── Session State Boot ────────────────────────────────────────────────────────

def _init_session():
    if "tc_initialized" in st.session_state:
        return

    sim    = EngineSimulator(initial_rpm=2000, initial_load=60)
    shared = {"raw": {}, "running": True, "daq_active": False}
    lock   = threading.Lock()

    # ── Engine Loop ─────────────────────────────────────────────────────────
    # Runs simulator at 10 Hz indefinitely. Writes to live_stream.json when
    # DAQ is active (atomic rename to prevent half-written reads).
    def engine_loop():
        stream_path = Path(__file__).parent / "live_stream.json"
        tmp_path    = stream_path.with_suffix(".tmp")
        while shared["running"]:
            t0  = time.perf_counter()
            raw = sim.step()
            with lock:
                shared["raw"] = raw
            if shared["daq_active"]:
                try:
                    tmp_path.write_text(json.dumps(raw))
                    tmp_path.replace(stream_path)
                except Exception:
                    pass
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 0.1 - elapsed))

    threading.Thread(target=engine_loop, daemon=True, name="TC_EngineLoop").start()

    # ── State HTTP Server ────────────────────────────────────────────────────
    # Serves two endpoints:
    #   /state.json  — consumed by flow_animation.html iframe (same machine)
    #   /stream      — consumed by diagnostic_ui.py on another machine (Wi-Fi)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    class StateHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args): pass  # suppress noise

        def do_GET(self):
            if self.path in ("/state.json", "/stream"):
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with lock:
                    raw = dict(shared.get("raw", {}))
                has_lk = _has_leak(sim)
                payload = {
                    **raw,                                  # full sensor dict for /stream
                    "leak_zone": _active_zone(sim) if has_lk else None,
                    "leak_sub":  _active_sub(sim)  if has_lk else None,
                    "status":    "ALERT" if has_lk else "PASS",
                    # Aliases for flow_animation.html
                    "n_turbo":   raw.get("n_turbo_rpm", 80000),
                    "load":      raw.get("load_pct", 60),
                    "maf":       raw.get("maf_gs", 150),
                }
                self.wfile.write(json.dumps(payload).encode())
            else:
                self.send_response(404)
                self.end_headers()

    server = socketserver.TCPServer(("", port), StateHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="TC_StateServer").start()

    st.session_state.update({
        "tc_initialized": True,
        "sim":            sim,
        "shared":         shared,
        "lock":           lock,
        "port":           port,
        "daq_active":     False,
        "own_ip":         _own_ip(),
        "csv_ready":      False,
        "csv_buf":        None,
        "csv_rows":       0,
    })


_init_session()

sim    = st.session_state.sim
shared = st.session_state.shared
lock   = st.session_state.lock
port   = st.session_state.port
own_ip = st.session_state.own_ip

with lock:
    raw = dict(shared.get("raw", {}))

is_daq_on = st.session_state.get("daq_active", False)
has_lk    = _has_leak(sim)


# ─── Header ───────────────────────────────────────────────────────────────────
badge_cls  = "badge-purple" if is_daq_on else "badge-warm"
badge_text = "📡 DAQ LIVE" if is_daq_on else "⭕ DAQ STANDBY"
lk_badge   = ' &nbsp; <span class="badge badge-red">🔴 FAULT ACTIVE</span>' if has_lk else ""

st.markdown(f"""
<div class="cat-header">
  <div>
    <div class="cat-title">🏭 CAT C7 — Test Cell DAQ Controller</div>
    <div class="cat-subtitle">
      Digital Twin · Engine Simulator · Fault Injector · CSV Exporter
    </div>
  </div>
  <div style="text-align:right">
    <span class="badge {badge_cls}">{badge_text}</span>{lk_badge}
    <div class="cat-subtitle" style="margin-top:6px">
      🌐 Network stream →
      <b style="color:#a5b4fc">http://{own_ip}:{port}/stream</b>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Flow Animation ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="control-title" style="font-size:1.05rem;text-align:center;margin-bottom:0">'
    '🌬️ &nbsp; LIVE ENGINE FLOW MAP &nbsp; — &nbsp; Real-time Digital Twin Visualization'
    '</div>',
    unsafe_allow_html=True,
)
html_str = open(Path(__file__).parent / "flow_animation.html").read().replace("8502", str(port))
components.html(html_str, height=600)
st.markdown("---")


# ─── Controls Row ─────────────────────────────────────────────────────────────
col_op, col_ecu, col_leak = st.columns([1.1, 1.1, 2.8], gap="large")

# ── Operating Point ──────────────────────────────────────────────────────────
with col_op:
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.markdown('<div class="control-title">⚙️ Engine Operating Point</div>', unsafe_allow_html=True)
    new_rpm  = st.slider("Engine Speed (RPM)", 600, 3000,
                         int(sim.state.rpm), step=50, key="rpm_sl")
    new_load = st.slider("Engine Load (%)", 0, 100,
                         int(sim.state.load_pct), step=5, key="load_sl")
    if st.button("✅ Apply Operating Point", key="apply_op", use_container_width=True):
        sim.set_operating_point(new_rpm, new_load)
    # Live readout strip
    st.markdown(f"""
    <div style="display:flex;gap:0.5rem;margin-top:0.7rem">
      <div class="mini-metric" style="flex:1">
        <div class="mini-label">Live RPM</div>
        <div class="mini-val">{raw.get('rpm', 2000):,.0f}</div>
      </div>
      <div class="mini-metric" style="flex:1">
        <div class="mini-label">Live Load</div>
        <div class="mini-val">{raw.get('load_pct', 60):.0f}%</div>
      </div>
      <div class="mini-metric" style="flex:1">
        <div class="mini-label">Turbo</div>
        <div class="mini-val">{raw.get('n_turbo_rpm', 0)/1000:.0f}k</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── ECU Flags ─────────────────────────────────────────────────────────────────
with col_ecu:
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.markdown('<div class="control-title">🔌 ECU Status Flags</div>', unsafe_allow_html=True)
    dpf_tog = st.toggle("DPF Regen Active", value=sim.state.dpf_regen_active, key="dpf_tog")
    egr_sl  = st.slider("EGR Position (%)", 0, 60,
                         int(sim.state.egr_position_pct), key="egr_sl")
    if st.button("✅ Apply ECU Flags", key="apply_flags", use_container_width=True):
        sim.set_engine_flags(
            dpf_regen=dpf_tog,
            egr_pct=egr_sl,
            coolant_temp_c=sim.state.coolant_temp_c,
        )
    dpf_color = "#fb923c" if sim.state.dpf_regen_active else "#4ade80"
    dpf_text  = "ACTIVE" if sim.state.dpf_regen_active else "OFF"
    st.markdown(f"""
    <div class="net-box" style="background:#0a1225;border-color:#1e3a5f">
      DPF Regen: <b style="float:right;color:{dpf_color}">{dpf_text}</b><br>
      EGR Position: <b style="float:right;color:#e2e8f0">{sim.state.egr_position_pct:.0f}%</b><br>
      Coolant Temp: <b style="float:right;color:#e2e8f0">{sim.state.coolant_temp_c:.0f} °C</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Leak / Fault Injection ────────────────────────────────────────────────────
with col_leak:
    lk_cls = "control-section leak-active" if has_lk else "control-section"
    lk_hdr = f'💉 Fault Injection &nbsp;<span style="color:#f87171;font-size:0.7rem">● FAULT ACTIVE</span>' if has_lk else "💉 Fault Injection — No Faults"
    st.markdown(f'<div class="{lk_cls}">', unsafe_allow_html=True)
    st.markdown(f'<div class="control-title">{lk_hdr}</div>', unsafe_allow_html=True)

    lc1, lc2, lc3 = st.columns(3)

    with lc1:
        st.markdown('<div style="border-left:3px solid #3b82f6;padding-left:0.5rem">', unsafe_allow_html=True)
        st.markdown("**Zone A** — Pre-Turbo")
        sev_a = st.slider("Severity", 0.0, 1.0, sim.leak.zone_a_severity, 0.05,
                          key="sev_a", help="MAF reduction fraction")
        if st.button("Inject A", key="inj_a",
                     type="primary" if sev_a > 0 else "secondary",
                     use_container_width=True):
            sim.inject_leak("A", sev_a)
        st.markdown('</div>', unsafe_allow_html=True)

    with lc2:
        st.markdown('<div style="border-left:3px solid #f87171;padding-left:0.5rem">', unsafe_allow_html=True)
        st.markdown("**Zone B** — Charge-Air")
        sev_b = st.slider("Severity", 0.0, 1.0, sim.leak.zone_b_severity, 0.05,
                          key="sev_b", help="MAP reduction fraction")
        loc_b = st.selectbox("Location",
                             ["after_intercooler", "before_intercooler"], key="loc_b")
        if st.button("Inject B", key="inj_b",
                     type="primary" if sev_b > 0 else "secondary",
                     use_container_width=True):
            sim.inject_leak("B", sev_b, b_location=loc_b)
        st.markdown('</div>', unsafe_allow_html=True)

    with lc3:
        st.markdown('<div style="border-left:3px solid #c084fc;padding-left:0.5rem">', unsafe_allow_html=True)
        st.markdown("**Zone C** — Exhaust")
        sev_c = st.slider("Severity", 0.0, 1.0, sim.leak.zone_c_severity, 0.05,
                          key="sev_c", help="EBP cascade effect")
        loc_c = st.selectbox("Location",
                             ["exhaust_manifold", "manifold_to_turbine",
                              "turbine_to_doc", "doc_to_dpf", "dpf_to_scr"],
                             key="loc_c")
        if st.button("Inject C", key="inj_c",
                     type="primary" if sev_c > 0 else "secondary",
                     use_container_width=True):
            sim.inject_leak("C", sev_c, c_location=loc_c)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔴 CLEAR ALL FAULTS", key="clear_all", use_container_width=True):
        sim.clear_leak()
    st.markdown("</div>", unsafe_allow_html=True)


# ─── DAQ Broadcaster + CSV Generator ──────────────────────────────────────────
st.markdown("---")
col_daq, col_csv = st.columns([1, 1], gap="large")

# ── DAQ Broadcaster ───────────────────────────────────────────────────────────
with col_daq:
    daq_panel_cls = "daq-panel daq-live" if is_daq_on else "daq-panel"
    st.markdown(f'<div class="{daq_panel_cls}">', unsafe_allow_html=True)
    st.markdown(
        '<div class="control-title" style="color:#c084fc;font-size:0.85rem">'
        '📡 Live DAQ Broadcaster</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.78rem;color:#94a3b8;margin:0 0 0.8rem 0">'
        'When active, writes raw sensor data to <code>live_stream.json</code> every 100 ms. '
        'The Diagnostic UI — on this machine or across Wi-Fi — reads this stream.</p>',
        unsafe_allow_html=True,
    )

    if not is_daq_on:
        if st.button("▶  START BROADCASTING", key="daq_start",
                     use_container_width=True, type="primary"):
            st.session_state.daq_active = True
            shared["daq_active"] = True
            st.rerun()
    else:
        if st.button("⏹  STOP BROADCASTING", key="daq_stop", use_container_width=True):
            st.session_state.daq_active = False
            shared["daq_active"] = False
            st.rerun()

    if is_daq_on:
        st.markdown(f"""
        <div class="net-box">
          📁 &nbsp;<b>Same laptop</b> → reads <code>live_stream.json</code><br>
          🌐 &nbsp;<b>Network laptop</b> → <code>http://{own_ip}:{port}/stream</code><br>
          📶 &nbsp;Broadcast rate: <b>10 Hz (100 ms/frame)</b><br>
          🔑 &nbsp;No auth required — same Wi-Fi network only
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="net-box" style="opacity:0.5">
          📁 &nbsp;Local file: <code>live_stream.json</code> (not writing)<br>
          🌐 &nbsp;Network URL: <code>http://{own_ip}:{port}/stream</code> (ready)<br>
          📶 &nbsp;Start broadcasting to activate the data feed
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── CSV Batch Generator ───────────────────────────────────────────────────────
with col_csv:
    st.markdown('<div class="csv-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="control-title" style="color:#22d3ee;font-size:0.85rem">'
        '📊 CSV Batch Generator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.78rem;color:#94a3b8;margin:0 0 0.8rem 0">'
        'Instantly simulate a full engine run at current conditions and download '
        'the structured sensor log as a CSV for offline analysis or CSV Playback mode.</p>',
        unsafe_allow_html=True,
    )

    dur_col, btn_col = st.columns([1.4, 1])
    with dur_col:
        dur_s = st.number_input("Duration (seconds)", min_value=5, max_value=600,
                                value=60, step=5, key="batch_dur")
    with btn_col:
        st.write("")  # vertical alignment nudge
        st.write("")
        gen_clicked = st.button("⚡ Generate", key="gen_csv",
                                use_container_width=True, type="primary")

    if gen_clicked:
        with st.spinner(f"Simulating {dur_s} s of engine data (no realtime delay)..."):
            # Snapshot simulator: mirrors current state but runs independently
            snap = EngineSimulator(
                initial_rpm=sim.state.rpm,
                initial_load=sim.state.load_pct,
            )
            snap.state.dpf_regen_active = sim.state.dpf_regen_active
            snap.state.egr_position_pct = sim.state.egr_position_pct
            snap.state.coolant_temp_c   = sim.state.coolant_temp_c
            snap.leak.zone_a_severity   = sim.leak.zone_a_severity
            snap.leak.zone_b_severity   = sim.leak.zone_b_severity
            snap.leak.zone_c_severity   = sim.leak.zone_c_severity
            snap.leak.zone_b_location   = sim.leak.zone_b_location
            snap.leak.zone_c_location   = sim.leak.zone_c_location
            df_batch = snap.run_batch(duration_s=float(dur_s), realtime=False)

        buf = io.BytesIO()
        df_batch.to_csv(buf, index=False)
        buf.seek(0)
        st.session_state.csv_buf  = buf.getvalue()
        st.session_state.csv_rows = len(df_batch)
        st.session_state.csv_ready = True

    if st.session_state.get("csv_ready"):
        st.success(f"✅  {st.session_state.csv_rows} rows ready  "
                   f"({st.session_state.csv_rows * 0.1:.0f} s of data)")
        ts = int(time.time())
        rpm_tag  = int(sim.state.rpm)
        load_tag = int(sim.state.load_pct)
        st.download_button(
            label="⬇  Download CSV",
            data=st.session_state.csv_buf,
            file_name=f"cat_c7_{rpm_tag}rpm_{load_tag}pct_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Live Sensor Strip ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="control-title">📡 Live Sensor Readings</div>', unsafe_allow_html=True)

def _mini(label, val, unit=""):
    return (
        f'<div class="mini-metric">'
        f'<div class="mini-label">{label}</div>'
        f'<div class="mini-val">{val}'
        f'<span style="font-size:0.6rem;color:#94a3b8"> {unit}</span></div>'
        f'</div>'
    )

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
strips = [
    (c1, "MAF",        f"{raw.get('maf_gs', 0):.1f}",          "g/s"),
    (c2, "MAP",        f"{raw.get('map_kpa', 0):.1f}",         "kPa"),
    (c3, "EBP",        f"{raw.get('ebp_kpa', 0):.1f}",         "kPa"),
    (c4, "EGT-1",      f"{raw.get('egt_1_c', 0):.0f}",         "°C"),
    (c5, "EGT-3",      f"{raw.get('egt_3_c', 0):.0f}",         "°C"),
    (c6, "Turbo RPM",  f"{raw.get('n_turbo_rpm', 0)/1000:.1f}k","RPM"),
    (c7, "Lambda λ",   f"{raw.get('lambda_ratio', 1.65):.3f}", ""),
    (c8, "Fuel Rate",  f"{raw.get('fuel_rate_gs', 0):.2f}",    "g/s"),
]
for col, lbl, val, unit in strips:
    with col:
        st.markdown(_mini(lbl, val, unit), unsafe_allow_html=True)


# ─── Auto-refresh ─────────────────────────────────────────────────────────────
time.sleep(0.5)
st.rerun()
