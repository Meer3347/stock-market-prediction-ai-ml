"""
Stock Forecast Dashboard — Static Version
==========================================
Loads pre-computed forecasts from data/dashboard_static.json
No model running at load time — loads in 2-3 seconds.

To refresh data: run generate_static_data.py in Jupyter,
then redeploy to Streamlit Cloud.
"""

import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Prediction with AI & ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1rem 1rem 2rem 1rem !important;max-width:1400px;}
.app-title{font-size:1.5rem;font-weight:600;color:#1a1f2e;}
.app-sub{font-size:0.75rem;color:#57606a;font-family:'IBM Plex Mono',monospace;margin-top:0.2rem;}
.live-badge{display:inline-flex;align-items:center;gap:6px;background:#e8f5e9;border:1px solid #a5d6a7;color:#2e7d32;font-size:0.7rem;font-family:'IBM Plex Mono',monospace;padding:3px 10px;border-radius:20px;margin-top:0.4rem;}
.static-badge{display:inline-flex;align-items:center;gap:6px;background:#e3f2fd;border:1px solid #90caf9;color:#1565c0;font-size:0.7rem;font-family:'IBM Plex Mono',monospace;padding:3px 10px;border-radius:20px;margin-top:0.4rem;margin-left:6px;}
.sec-hdr{font-size:0.7rem;font-family:'IBM Plex Mono',monospace;color:#57606a;text-transform:uppercase;letter-spacing:0.7px;font-weight:500;margin:0.75rem 0 0.5rem 0;padding-bottom:0.3rem;border-bottom:1px solid #e8eaed;}
.mcard{background:#fff;border:1.5px solid #d0d7de;border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.5rem;border-top:3px solid;transition:box-shadow 0.15s;}
.mcard:hover{box-shadow:0 2px 10px rgba(0,0,0,0.08);}
.mticker{font-size:0.65rem;font-family:'IBM Plex Mono',monospace;color:#57606a;}
.mprice{font-size:1.25rem;font-weight:600;margin:0.2rem 0;}
.mchange{font-size:0.75rem;font-weight:600;}
.msignal{font-size:0.7rem;margin-top:0.3rem;font-weight:500;}
.sent-row{display:flex;align-items:center;gap:10px;margin-bottom:0.7rem;}
.sent-lbl{font-size:0.75rem;color:#1a1f2e;font-weight:500;}
.sent-sub{font-size:0.65rem;color:#8c959f;margin-top:1px;}
.sent-track{flex:1;height:6px;background:#e8eaed;border-radius:3px;overflow:hidden;}
.sent-fill{height:100%;border-radius:3px;}
.sent-val{font-size:0.75rem;font-family:'IBM Plex Mono',monospace;font-weight:600;width:52px;text-align:right;}
.score-box{background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:0.75rem;font-size:0.73rem;color:#57606a;line-height:1.8;margin-top:0.75rem;}
.score-box b{color:#1a1f2e;}
.ni{background:#f6f8fa;border:1px solid #d0d7de;border-left:3px solid;border-radius:6px;padding:0.6rem 0.75rem;margin-bottom:0.5rem;}
.nt{font-size:0.78rem;color:#1a1f2e;font-weight:500;line-height:1.4;margin-bottom:0.3rem;}
.nm{font-size:0.68rem;color:#57606a;display:flex;gap:10px;flex-wrap:wrap;align-items:center;}
.nl{color:#1565c0;text-decoration:none;font-weight:500;margin-left:auto;}
.disc{background:#fffbf0;border:1px solid #f0c060;border-radius:8px;padding:0.75rem 1rem;font-size:0.72rem;color:#7a5500;line-height:1.7;margin-top:1rem;}
.disc b{color:#e65100;}
.gen-info{background:#f0f7ff;border:1px solid #b3d4f5;border-radius:8px;padding:0.6rem 1rem;font-size:0.72rem;color:#1565c0;margin-bottom:0.75rem;font-family:'IBM Plex Mono',monospace;}
@media(max-width:768px){
  .app-title{font-size:1.1rem;}
  .block-container{padding:0.5rem !important;}
  .mprice{font-size:1rem;}
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
COLORS = {
    "AAPL":"#1565C0","GOOGL":"#2E7D32","MSFT":"#6A1B9A",
    "META":"#E65100","AMZN":"#00695C","NFLX":"#B71C1C",
}

def hex_rgba(h, a):
    return f"rgba({int(h[1:3],16)},{int(h[3:5],16)},{int(h[5:7],16)},{a})"

def sig_color(v):
    return "#2e7d32" if v > 0.05 else "#c62828" if v < -0.05 else "#e65100"

def pct_bar(v):
    return max(2, min(98, int((v + 1) / 2 * 100)))

def fmt_score(v):
    return f"{'+' if v >= 0 else ''}{v:.3f}"

# ── Load static data ──────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, "data", "dashboard_static.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(base, "dashboard_static.json")
    if not os.path.exists(data_path):
        data_path = "dashboard_static.json"
    with open(data_path, "r") as f:
        return json.load(f)

data = load_data()
tickers_data = data["tickers"]
generated_at = data["generated_at"]
fore_end     = data["forecast_end"]

# ── Header ────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:0.5rem 0 0.75rem 0;border-bottom:1px solid #d0d7de;margin-bottom:1rem;">
  <div class="app-title">📈 Stock Market Prediction with AI & ML</div>
  <div class="app-sub">Bidirectional LSTM · Monte Carlo Simulation · Technical + Sentiment Analysis</div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:0.4rem;">
    <div class="live-badge">● Yahoo Finance · Google Trends · VADER NLP</div>
    <div class="static-badge">📅 Data as of {generated_at}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Filters ───────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.2, 2.5, 1.8])

with c1:
    date_range = st.selectbox("📅 Date range",
                               ["1Y", "2Y", "3Y", "Max"], index=2)

with c2:
    company_opts = ["All"] + [f"{t} — {d['company']}" for t, d in tickers_data.items()]
    sel_company  = st.selectbox("🏢 Company", company_opts)

with c3:
    model_choice = st.radio("🤖 Forecast model",
                             ["Both", "Technical only", "Sentiment only"],
                             horizontal=True)

sel_tickers = list(tickers_data.keys()) if sel_company == "All" \
    else [sel_company.split(" — ")[0]]
sel_label   = "All Companies" if sel_company == "All" \
    else sel_company.split(" — ")[1]

range_pts = {"1Y": 252, "2Y": 504, "3Y": 756, "Max": 9999}[date_range]

# ── Generation notice ─────────────────────────────────────────
st.markdown(f"""
<div class="gen-info">
  📊 Forecasts generated on {generated_at} · 
  Forecast horizon: {data['forecast_start']} → {fore_end} · 
  10 Monte Carlo simulations per model
</div>
""", unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Prices & signals — as of data generation date</div>',
            unsafe_allow_html=True)

cols = st.columns(6)
for i, (ticker, td) in enumerate(tickers_data.items()):
    col = COLORS[ticker]
    sc  = "#2e7d32" if td["signal"] == "BULLISH" else \
          "#c62828" if td["signal"] == "BEARISH" else "#e65100"
    with cols[i]:
        st.markdown(f"""
        <div class="mcard" style="border-top-color:{col}">
          <div class="mticker">{ticker} · {td['company']}</div>
          <div class="mprice" style="color:{col}">${td['current']:.2f}</div>
          <div class="mchange" style="color:{'#2e7d32' if td['day_chg']>=0 else '#c62828'}">
            {'+' if td['day_chg']>=0 else ''}{td['day_chg']:.1f}% that day
          </div>
          <div class="mchange" style="color:{'#2e7d32' if td['m1_6m_pct']>=0 else '#c62828'}">
            {'+' if td['m1_6m_pct']>=0 else ''}{td['m1_6m_pct']:.1f}% 6M forecast
          </div>
          <div class="msignal" style="color:{sc}">
            ● {td['signal']} · {td['confidence']:.0f}% confidence
          </div>
        </div>""", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Price chart — historical + 6-month forecast</div>',
            unsafe_allow_html=True)

fig = go.Figure()

for ticker in sel_tickers:
    td    = tickers_data[ticker]
    color = COLORS[ticker]
    name  = td["company"]

    # Slice historical to date range
    hist_dates  = td["hist_dates"][-range_pts:]
    hist_prices = td["hist_prices"][-range_pts:]
    fore_dates  = td["fore_dates"]
    m1_med      = td["m1_med"]
    m1_p05      = td["m1_p05"]
    m1_p95      = td["m1_p95"]
    m1_p25      = td["m1_p25"]
    m1_p75      = td["m1_p75"]
    m2_med      = td["m2_med"]

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices,
        name=f"{name} — Historical",
        line=dict(color=color, width=2),
        hovertemplate=f"<b>%{{x}}</b><br>{name} Historical: $%{{y:.2f}}<extra></extra>",
    ))

    if model_choice in ["Both", "Technical only"]:
        # 90% confidence band
        fig.add_trace(go.Scatter(
            x=fore_dates + fore_dates[::-1],
            y=m1_p95 + m1_p05[::-1],
            fill="toself",
            fillcolor=hex_rgba(color, 0.08),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        # 50% confidence band
        fig.add_trace(go.Scatter(
            x=fore_dates + fore_dates[::-1],
            y=m1_p75 + m1_p25[::-1],
            fill="toself",
            fillcolor=hex_rgba(color, 0.15),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Technical forecast line
        fig.add_trace(go.Scatter(
            x=fore_dates, y=m1_med,
            name=f"{name} — Technical Analysis Forecast",
            line=dict(color=color, width=2.5, dash="dash"),
            hovertemplate=f"<b>📈 Technical Forecast — %{{x}}</b><br>"
                          f"{name}: $%{{y:.2f}}<extra></extra>",
        ))

    if model_choice in ["Both", "Sentiment only"]:
        # Sentiment forecast line
        fig.add_trace(go.Scatter(
            x=fore_dates, y=m2_med,
            name=f"{name} — Sentiment-Driven Forecast",
            line=dict(color="#E65100", width=2.5, dash="dot"),
            hovertemplate=f"<b>📊 Sentiment Forecast — %{{x}}</b><br>"
                          f"{name}: $%{{y:.2f}}<extra></extra>",
        ))

# Today vertical line — use numeric timestamp
import pandas as pd
today_ts = pd.Timestamp(data["forecast_start"]).timestamp() * 1000
fig.add_vline(
    x=today_ts,
    line_dash="dot",
    line_color="#57606a",
    line_width=1.5,
    annotation_text="Forecast starts ▼",
    annotation_position="top",
    annotation_font=dict(size=11, color="#57606a"),
)

fig.update_layout(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="IBM Plex Sans", size=12, color="#1a1f2e"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.01,
        xanchor="left",   x=0,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#d0d7de", borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(
        showgrid=True, gridcolor="#e8eaed",
        tickfont=dict(family="IBM Plex Mono", size=11, color="#1a1f2e"),
        tickformat="%b %Y",
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#e8eaed",
        tickfont=dict(family="IBM Plex Mono", size=11, color="#1a1f2e"),
        tickprefix="$",
        zeroline=False,
    ),
    height=440,
)

st.plotly_chart(fig, use_container_width=True)

# ── Bottom panels ─────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown(
        f'<div class="sec-hdr">Sentiment scoring — {sel_label}</div>',
        unsafe_allow_html=True)

    avg_n = np.mean([tickers_data[t]["sentiment"]["news"]   for t in sel_tickers])
    avg_t = np.mean([tickers_data[t]["sentiment"]["trends"] for t in sel_tickers])
    avg_m = np.mean([tickers_data[t]["sentiment"]["macro"]  for t in sel_tickers])
    avg_f = np.mean([tickers_data[t]["sentiment"]["final"]  for t in sel_tickers])

    rows = [
        ("News sentiment",    "VADER NLP on headlines · 35% weight",  avg_n),
        ("Google Trends",     "Search interest rising/falling · 30%", avg_t),
        ("Macro environment", "FRED: rates, CPI, VIX, yields · 35%",  avg_m),
    ]

    html = ""
    for label, sub, val in rows:
        c = sig_color(val)
        html += f"""
        <div class="sent-row">
          <div style="width:130px;flex-shrink:0">
            <div class="sent-lbl">{label}</div>
            <div class="sent-sub">{sub}</div>
          </div>
          <div class="sent-track">
            <div class="sent-fill" style="width:{pct_bar(val)}%;background:{c}"></div>
          </div>
          <div class="sent-val" style="color:{c}">{fmt_score(val)}</div>
        </div>"""

    c = sig_color(avg_f)
    html += f"""
    <div style="border-top:1px solid #d0d7de;padding-top:8px;margin-top:4px;">
    <div class="sent-row">
      <div style="width:130px;flex-shrink:0">
        <div class="sent-lbl" style="font-weight:600">Overall score</div>
        <div class="sent-sub">Weighted combination of all 3</div>
      </div>
      <div class="sent-track">
        <div class="sent-fill" style="width:{pct_bar(avg_f)}%;background:{c}"></div>
      </div>
      <div class="sent-val" style="color:{c};font-weight:700">{fmt_score(avg_f)}</div>
    </div></div>"""

    st.markdown(html, unsafe_allow_html=True)

    fdir = "bullish" if avg_f > 0.05 else "bearish" if avg_f < -0.05 else "neutral"
    st.markdown(f"""
    <div class="score-box">
      <b>Score range: −1.0 (very bearish) → 0.0 (neutral) → +1.0 (very bullish)</b><br>
      <b>News</b> = NLP tone detection on Yahoo Finance headlines.<br>
      <b>Trends</b> = Google search interest over 90 days — rising = bullish signal.<br>
      <b>Macro</b> = Fed rate, CPI inflation, VIX fear index, 10Y Treasury via FRED.<br>
      <b>Overall</b> = weighted average. Currently
      <span style="color:{sig_color(avg_f)};font-weight:600">{fdir} at {fmt_score(avg_f)}</span>
      — this shifts the sentiment forecast line on the chart up or down.
    </div>""", unsafe_allow_html=True)

with right:
    st.markdown(
        f'<div class="sec-hdr">Latest headlines — {sel_label}</div>',
        unsafe_allow_html=True)

    articles = []
    for t in sel_tickers:
        for a in tickers_data[t]["news"]:
            articles.append({**a, "company": tickers_data[t]["company"]})

    articles.sort(key=lambda x: abs(x["score"]), reverse=True)

    for a in articles[:5]:
        tc  = "#2e7d32" if a["tone"] == "bull" else \
              "#c62828" if a["tone"] == "bear" else "#e65100"
        ico = "↑ Bullish" if a["tone"] == "bull" else \
              "↓ Bearish" if a["tone"] == "bear" else "→ Neutral"
        st.markdown(f"""
        <div class="ni" style="border-left-color:{tc}">
          <div class="nt">{a['title']}</div>
          <div class="nm">
            <span style="color:{tc};font-weight:600">{ico}</span>
            <span>{a['company']}</span>
            <span style="font-family:'IBM Plex Mono',monospace;color:{tc}">
              {'+' if a['score']>=0 else ''}{a['score']:.3f}</span>
            <a class="nl" href="{a['url']}" target="_blank">Read →</a>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────
st.markdown(f"""
<div class="disc">
  <b>⚠ Not financial advice.</b>
  Forecasts were pre-computed on <b>{generated_at}</b> using live Yahoo Finance prices,
  Google Trends, and VADER NLP sentiment analysis.
  Running Bidirectional LSTM models with Monte Carlo simulation is computationally intensive,
  so this dashboard uses pre-computed results for fast loading.
  To refresh forecasts, retrain and redeploy.
  Confidence bands show uncertainty growing over time —
  short-term (30–60 day) forecasts are more reliable than 6-month ones.
  Always conduct your own research before making any investment decisions.
</div>
<div style="text-align:center;font-size:0.7rem;color:#8c959f;margin-top:1rem;
            font-family:'IBM Plex Mono',monospace;">
  Built with TensorFlow · Yahoo Finance · Google Trends · VADER NLP · Streamlit
  &nbsp;·&nbsp;
  Data as of {generated_at}
</div>
""", unsafe_allow_html=True)
