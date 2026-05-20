import streamlit as st
import numpy as np
from datetime import datetime

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangladesh AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    import tensorflow as tf
    import joblib
    model = tf.keras.models.load_model("bangladesh_aqi_ann_model.keras")
    scaler = joblib.load("aqi_scaler.joblib")
    return model, scaler

try:
    model, scaler = load_assets()
    model_loaded = True
except Exception:
    model_loaded = False

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_aqi_config(aqi: float) -> dict:
    if aqi <= 50:
        return dict(color="#22c55e", status="Good", icon="😄", risk_pct=10,
                    risk_desc="Air quality is satisfactory")
    elif aqi <= 100:
        return dict(color="#f59e0b", status="Moderate", icon="😐", risk_pct=35,
                    risk_desc="Moderate risk due to weather conditions")
    elif aqi <= 150:
        return dict(color="#f97316", status="Unhealthy for Sensitive Groups", icon="😷",
                    risk_pct=55, risk_desc="Sensitive groups at elevated risk")
    elif aqi <= 200:
        return dict(color="#ef4444", status="Unhealthy", icon="🚨", risk_pct=78,
                    risk_desc="Health effects may be experienced")
    else:
        return dict(color="#a855f7", status="Hazardous", icon="☠️", risk_pct=95,
                    risk_desc="Serious health effects for all groups")

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── Reset & root ── */
*, *::before, *::after { box-sizing: border-box; }
:root {
  --navy: #040d1f;
  --panel: rgba(255,255,255,0.04);
  --panel-border: rgba(255,255,255,0.08);
  --accent-blue: #3b8beb;
  --accent-cyan: #00d4ff;
  --text-primary: #f0f4ff;
  --text-secondary: rgba(200,215,255,0.6);
  --text-muted: rgba(150,170,220,0.4);
}

/* ── Streamlit shell overrides ── */
html, body, [data-testid="stApp"] {
  background: #040d1f !important;
  color: var(--text-primary) !important;
  font-family: 'Syne', sans-serif !important;
}
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
section.main > div { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ────────────────────────────────────────────────────────────────────
now = datetime.now()
dt_str = now.strftime("%d/%m/%Y %H:%M")

# ── PAGE LAYOUT: LEFT | RIGHT ─────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.2], gap="small")

# ════════════════════════════════════════════════════════════════
# LEFT SIDEBAR PANEL
# ════════════════════════════════════════════════════════════════
with left_col:
    st.markdown(f"""
    <div style="
      padding: 0.2rem 1.5rem;
      border-right: 0.5px solid rgba(255,255,255,0.08);
      display: flex; flex-direction: column; gap: 1.1rem;
      background: rgba(4,13,31,0.5);
      position: relative; z-index: 1;
    ">
      <!-- Title -->
      <div>
        <h1 style="font-size:26px; font-weight:800; line-height:1.1; letter-spacing:-0.02em; color:#f0f4ff; margin:0;">
          <span style="color:#3b8beb;">Air Quality</span> Index
        </h1>
        <div style="font-size:11px; color:rgba(200,215,255,0.6); font-family:'DM Mono',monospace; margin-top:5px; display:flex; align-items:center; gap:6px;">
          {dt_str} &nbsp;·&nbsp;
          <span style="width:6px;height:6px;border-radius:50%;background:#22c55e;display:inline-block;"></span>
          Local → BGD
        </div>
      </div>
    """, unsafe_allow_html=True)

    # We'll inject dynamic AQI stats after prediction via session state
    if "predicted_aqi" in st.session_state:
        aqi = st.session_state.predicted_aqi
        cfg = get_aqi_config(aqi)
        dominant = st.session_state.get("dominant", "PM2.5")
        risk_pct = cfg["risk_pct"]
        risk_bar_width = risk_pct

        st.markdown(f"""
        <!-- Main Statistics Card -->
        <div style="background:rgba(255,255,255,0.04); border:0.5px solid rgba(255,255,255,0.08); border-radius:14px; padding:1rem 1.2rem; margin-top:1rem;">
          <div style="font-size:10px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:rgba(150,170,220,0.4); margin-bottom:0.7rem;">Main Statistics</div>
          <div style="font-size:10px; color:rgba(150,170,220,0.4); letter-spacing:0.06em; text-transform:uppercase; margin-bottom:3px;">AQI</div>
          <div style="font-size:48px; font-weight:800; line-height:1; letter-spacing:-0.03em; color:{cfg['color']};">{aqi:.1f}</div>
          <div style="font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; letter-spacing:0.02em; display:inline-block; margin-top:5px; background:{cfg['color']}22; color:{cfg['color']};">
            {cfg['status']}
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:10px; border-top:0.5px solid rgba(255,255,255,0.08); font-size:11px; color:rgba(200,215,255,0.6);">
            <span>Dominant Pollutant</span>
            <span style="color:#f0f4ff; font-weight:600; font-family:'DM Mono',monospace;">{dominant} — Wind</span>
          </div>
        </div>

        <!-- Risk Card -->
        <div style="background:rgba(255,255,255,0.04); border:0.5px solid rgba(255,255,255,0.08); border-radius:14px; padding:1rem 1.2rem; margin-top:0.8rem;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:13px; font-weight:600; color:#f0f4ff;">Risk of Pollution</span>
            <span style="font-size:10px; font-weight:700; padding:3px 8px; border-radius:20px; background:rgba(59,139,235,0.2); color:#00d4ff;">Details</span>
          </div>
          <div style="font-size:10px; color:rgba(150,170,220,0.4); letter-spacing:0.06em; text-transform:uppercase; margin-bottom:3px;">Risk</div>
          <div style="font-size:38px; font-weight:800; letter-spacing:-0.03em; color:#f0f4ff;">{risk_pct}%</div>
          <div style="font-size:11px; color:rgba(200,215,255,0.6); margin-top:3px;">{cfg['risk_desc']}</div>
          <div style="height:4px; background:rgba(255,255,255,0.08); border-radius:4px; margin-top:10px; overflow:hidden;">
            <div style="height:100%; width:{risk_bar_width}%; border-radius:4px; background:{cfg['color']};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <!-- Main Statistics Card (empty state) -->
        <div style="background:rgba(255,255,255,0.04); border:0.5px solid rgba(255,255,255,0.08); border-radius:14px; padding:1rem 1.2rem; margin-top:1rem;">
          <div style="font-size:10px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:rgba(150,170,220,0.4); margin-bottom:0.7rem;">Main Statistics</div>
          <div style="font-size:10px; color:rgba(150,170,220,0.4); letter-spacing:0.06em; text-transform:uppercase; margin-bottom:3px;">AQI</div>
          <div style="font-size:48px; font-weight:800; line-height:1; letter-spacing:-0.03em; color:#f0f4ff;">--</div>
          <div style="font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; letter-spacing:0.02em; display:inline-block; margin-top:5px; background:rgba(255,255,255,0.08); color:rgba(200,215,255,0.6);">
            Awaiting Input
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:10px; border-top:0.5px solid rgba(255,255,255,0.08); font-size:11px; color:rgba(200,215,255,0.6);">
            <span>Dominant Pollutant</span>
            <span style="color:#f0f4ff; font-weight:600; font-family:'DM Mono',monospace;">PM2.5 — Wind</span>
          </div>
        </div>
        <!-- Risk Card (empty) -->
        <div style="background:rgba(255,255,255,0.04); border:0.5px solid rgba(255,255,255,0.08); border-radius:14px; padding:1rem 1.2rem; margin-top:0.8rem;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:13px; font-weight:600; color:#f0f4ff;">Risk of Pollution</span>
            <span style="font-size:10px; font-weight:700; padding:3px 8px; border-radius:20px; background:rgba(59,139,235,0.2); color:#00d4ff;">Details</span>
          </div>
          <div style="font-size:10px; color:rgba(150,170,220,0.4); letter-spacing:0.06em; text-transform:uppercase; margin-bottom:3px;">Risk</div>
          <div style="font-size:38px; font-weight:800; letter-spacing:-0.03em; color:#f0f4ff;">--</div>
          <div style="font-size:11px; color:rgba(200,215,255,0.6); margin-top:3px;">Enter values to calculate risk</div>
          <div style="height:4px; background:rgba(255,255,255,0.08); border-radius:4px; margin-top:10px;"></div>
        </div>
        """, unsafe_allow_html=True)

    # AQI Scale + Legend (always shown)
    lat_val = st.session_state.get("lat_val", 23.7298)
    lon_val = st.session_state.get("lon_val", 90.3854)
    st.markdown(f"""
      <!-- AQI Scale -->
      <div style="margin-top:0.8rem;">
        <div style="font-size:10px; color:rgba(150,170,220,0.4); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:7px;">AQI Scale</div>
        <div style="display:flex; gap:3px; border-radius:6px; overflow:hidden;">
          <div style="flex:1; height:6px; background:#22c55e;"></div>
          <div style="flex:1; height:6px; background:#f59e0b;"></div>
          <div style="flex:1; height:6px; background:#f97316;"></div>
          <div style="flex:1; height:6px; background:#ef4444;"></div>
          <div style="flex:1; height:6px; background:#a855f7;"></div>
        </div>
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:8px;">
          <div style="display:flex; align-items:center; gap:5px; font-size:10px; color:rgba(200,215,255,0.6);">
            <div style="width:7px;height:7px;border-radius:50%;background:#22c55e;"></div>Good
          </div>
          <div style="display:flex; align-items:center; gap:5px; font-size:10px; color:rgba(200,215,255,0.6);">
            <div style="width:7px;height:7px;border-radius:50%;background:#f59e0b;"></div>Moderate
          </div>
          <div style="display:flex; align-items:center; gap:5px; font-size:10px; color:rgba(200,215,255,0.6);">
            <div style="width:7px;height:7px;border-radius:50%;background:#f97316;"></div>Sensitive
          </div>
          <div style="display:flex; align-items:center; gap:5px; font-size:10px; color:rgba(200,215,255,0.6);">
            <div style="width:7px;height:7px;border-radius:50%;background:#ef4444;"></div>Unhealthy
          </div>
          <div style="display:flex; align-items:center; gap:5px; font-size:10px; color:rgba(200,215,255,0.6);">
            <div style="width:7px;height:7px;border-radius:50%;background:#a855f7;"></div>Hazardous
          </div>
        </div>
      </div>

      <!-- Coordinates -->
      <div style="margin-top:auto; padding-top:1rem; border-top:0.5px solid rgba(255,255,255,0.08);">
        <div style="font-family:'DM Mono',monospace; font-size:11px; color:rgba(150,170,220,0.4); display:flex; gap:1.5rem;">
          <span>X: <span style="color:rgba(200,215,255,0.6);">{lat_val:.4f}</span></span>
          <span>Y: <span style="color:rgba(200,215,255,0.6);">{lon_val:.4f}</span></span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# RIGHT MAIN PANEL
# ════════════════════════════════════════════════════════════════
with right_col:
    # Note banner
    model_note = " Enter the coordinates and pollutant levels below to estimate the real-time Air Quality Index using our deep learning model." if model_loaded else \
        "⚠️ Model files not found. Place <b>bangladesh_aqi_ann_model.keras</b> and <b>aqi_scaler.joblib</b> in the same folder."
    st.markdown(f"""
    <div style="background:rgba(59,139,235,0.08); border:0.5px solid rgba(59,139,235,0.2); border-radius:10px; padding:10px 14px; font-size:12px; color:rgba(200,215,255,0.6); margin-bottom:1.5rem;">
      <b style="color:#00d4ff;">Note:</b> {model_note}
    </div>
    """, unsafe_allow_html=True)

    # ── Section: Coordinates ──────────────────────────────────────
    st.markdown("""
    <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:rgba(150,170,220,0.4); margin-bottom:0.9rem; display:flex; align-items:center; gap:8px;">
      Location Coordinates
      <div style="flex:1; height:0.5px; background:rgba(255,255,255,0.08);"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input("Latitude", value=23.7298, format="%.4f", key="lat_input")
    with c2:
        lon = st.number_input("Longitude", value=90.3854, format="%.4f", key="lon_input")

    st.session_state["lat_val"] = lat
    st.session_state["lon_val"] = lon

    # ── Section: Particulate Matter ───────────────────────────────
    st.markdown("""
    <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:rgba(150,170,220,0.4); margin:1.2rem 0 0.9rem; display:flex; align-items:center; gap:8px;">
      Particulate Matter
      <div style="flex:1; height:0.5px; background:rgba(255,255,255,0.08);"></div>
    </div>
    """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        pm10 = st.number_input("PM10 (µg/m³)", value=25.0, min_value=0.0, step=0.1)
    with c4:
        pm25 = st.number_input("PM2.5 (µg/m³)", value=16.7, min_value=0.0, step=0.1)

    # ── Section: Gas Pollutants ───────────────────────────────────
    st.markdown("""
    <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:rgba(150,170,220,0.4); margin:1.2rem 0 0.9rem; display:flex; align-items:center; gap:8px;">
      Gas Pollutants
      <div style="flex:1; height:0.5px; background:rgba(255,255,255,0.08);"></div>
    </div>
    """, unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        co  = st.number_input("Carbon Monoxide (CO)", value=252.0, min_value=0.0, step=0.1)
        so2 = st.number_input("Sulphur Dioxide (SO₂)", value=5.2,  min_value=0.0, step=0.1)
    with c6:
        no2   = st.number_input("Nitrogen Dioxide (NO₂)", value=18.6, min_value=0.0, step=0.1)
        ozone = st.number_input("Ozone (O₃)",             value=13.0, min_value=0.0, step=0.1)

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

    # ── PREDICT BUTTON ────────────────────────────────────────────
    if st.button("Predict AQI", use_container_width=True):
        with st.spinner("Analyzing..."):

            if model_loaded:
                # ── Real model path ──────────────────────────────
                pm10_log = np.log1p(pm10)
                pm25_log = np.log1p(pm25)
                co_log   = np.log1p(co)
                no2_log  = np.log1p(no2)
                so2_log  = np.log1p(so2)

                raw_features = np.array([[lat, lon, pm10_log, pm25_log, co_log, no2_log, so2_log, ozone]])
                scaled       = scaler.transform(raw_features)
                prediction   = model.predict(scaled)
                predicted_aqi = float(prediction[0][0])
            else:
                # ── Fallback approximation (same logic as HTML) ──
                pm25_idx = (301 if pm25 > 250 else
                            201 + (pm25 - 150) * 0.8 if pm25 > 150 else
                            151 + (pm25 - 55)  * 0.94 if pm25 > 55 else
                            101 + (pm25 - 35)  * 2.5  if pm25 > 35 else
                            51  + (pm25 - 12)  * 2.1  if pm25 > 12 else
                            (pm25 / 12) * 50)
                pm10_idx = (301 if pm10 > 354 else
                            201 + (pm10 - 254) * 0.5 if pm10 > 254 else
                            151 + (pm10 - 154) * 0.5 if pm10 > 154 else
                            101 + (pm10 - 54)  * 0.5 if pm10 > 54 else
                            (pm10 / 54) * 100)
                co_log  = np.log1p(co)
                no2_log = np.log1p(no2)
                predicted_aqi = float(np.clip(
                    max(pm25_idx, pm10_idx * 0.7) + (no2_log * 5) + (co_log * 2) - (ozone * 0.1),
                    0, 500
                ))

            dominant = "PM2.5" if pm25 > pm10 * 0.5 else "PM10"
            st.session_state["predicted_aqi"] = round(predicted_aqi, 1)
            st.session_state["dominant"]      = dominant
            st.session_state["inputs"]        = dict(pm25=pm25, pm10=pm10, no2=no2, co=co)
            st.rerun()

    # ── RESULT CARD ───────────────────────────────────────────────
    if "predicted_aqi" in st.session_state:
        aqi    = st.session_state["predicted_aqi"]
        cfg    = get_aqi_config(aqi)
        inputs = st.session_state.get("inputs", {})
        health_pct = min(100, (aqi / 500) * 100)
        time_str = datetime.now().strftime("%H:%M")

        st.markdown(f"""
        <div style="
          margin-top: 1.4rem;
          background: rgba(255,255,255,0.04);
          border: 0.5px solid rgba(255,255,255,0.08);
          border-radius: 16px;
          padding: 1.4rem;
          animation: slideUp 0.4s ease;
        ">
          <style>
            @keyframes slideUp {{
              from {{ opacity: 0; transform: translateY(14px); }}
              to   {{ opacity: 1; transform: translateY(0); }}
            }}
          </style>

          <!-- Header row -->
          <div style="display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:1rem;">
            <div>
              <div style="font-size:10px; color:rgba(150,170,220,0.4); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:5px;">
                Predicted AQI Score
              </div>
              <div style="font-size:72px; font-weight:800; line-height:1; letter-spacing:-0.04em; color:{cfg['color']};">
                {aqi:.1f}
              </div>
              <div style="font-size:17px; font-weight:700; margin-top:5px; color:{cfg['color']};">
                {cfg['status']}
              </div>
              <div style="font-size:11px; color:rgba(200,215,255,0.6); margin-top:3px; font-family:'DM Mono',monospace;">
                Bangladesh · {time_str}
              </div>
            </div>
            <div style="width:56px; height:56px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:28px; background:rgba(255,255,255,0.06);">
              {cfg['icon']}
            </div>
          </div>

          <!-- Health bar -->
          <div style="height:7px; border-radius:8px; overflow:hidden; background:rgba(255,255,255,0.05);">
            <div style="height:100%; width:{health_pct:.1f}%; border-radius:8px; background:{cfg['color']};"></div>
          </div>

          <!-- Breakdown grid -->
          <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:1.1rem; padding-top:1.1rem; border-top:0.5px solid rgba(255,255,255,0.08);">
            <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:10px; text-align:center;">
              <div style="font-size:15px; font-weight:700; font-family:'DM Mono',monospace; margin-bottom:3px; color:#f0f4ff;">{inputs.get('pm25', pm25):.1f}</div>
              <div style="font-size:10px; color:rgba(150,170,220,0.4); letter-spacing:0.04em;">PM2.5</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:10px; text-align:center;">
              <div style="font-size:15px; font-weight:700; font-family:'DM Mono',monospace; margin-bottom:3px; color:#f0f4ff;">{inputs.get('pm10', pm10):.1f}</div>
              <div style="font-size:10px; color:rgba(150,170,220,0.4); letter-spacing:0.04em;">PM10</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:10px; text-align:center;">
              <div style="font-size:15px; font-weight:700; font-family:'DM Mono',monospace; margin-bottom:3px; color:#f0f4ff;">{inputs.get('no2', no2):.1f}</div>
              <div style="font-size:10px; color:rgba(150,170,220,0.4); letter-spacing:0.04em;">NO₂</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:10px; text-align:center;">
              <div style="font-size:15px; font-weight:700; font-family:'DM Mono',monospace; margin-bottom:3px; color:#f0f4ff;">{inputs.get('co', co):.0f}</div>
              <div style="font-size:10px; color:rgba(150,170,220,0.4); letter-spacing:0.04em;">CO</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)