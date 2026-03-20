"""
Stock Market Prediction with AI & ML
"""
import os, json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Stock Market Prediction with AI & ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body { background:#13151f !important; overflow-x:hidden; font-family:'IBM Plex Sans',sans-serif; }
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > section > div { background:#13151f !important; }
.block-container {
    background:#13151f !important;
    padding:1.2rem 2rem 3rem 2rem !important;
    max-width:900px !important;
}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"] {
    display:none !important; visibility:hidden !important;
}

/* Selectbox dark */
div[data-baseweb="select"] > div {
    background:#13151f !important; border:1px solid #2a2d3e !important; border-radius:8px !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color:#c8cfe0 !important; }
div[data-baseweb="select"] svg { fill:#6b7590 !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { background:#1e2133 !important; border:1px solid #2a2d3e !important; }
li[role="option"] { background:#1e2133 !important; color:#c8cfe0 !important; }
li[role="option"]:hover { background:#2a2d3e !important; }
[data-testid="stSelectbox"] label {
    color:#a0a8c0 !important; font-size:0.65rem !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:0.8px !important;
    font-family:'IBM Plex Mono',monospace !important;
}
[data-testid="stSelectbox"] { margin-bottom:0 !important; }

/* Remove extra spacing Streamlit adds */
[data-testid="stVerticalBlock"] > div { gap:0 !important; }

/* Cards grid */
.cards-grid {
    display:grid; grid-template-columns:repeat(6,1fr);
    gap:6px; margin-bottom:12px; width:100%;
}
@media(max-width:768px) {
    .cards-grid { grid-template-columns:repeat(3,1fr); }
    .block-container { padding:0.75rem 0.75rem 3rem 0.75rem !important; }
}

/* Bottom grid */
.bottom-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:8px; }
@media(max-width:768px) { .bottom-grid { grid-template-columns:1fr; } }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
COLORS   = {"AAPL":"#1565C0","GOOGL":"#2E7D32","MSFT":"#6A1B9A","META":"#E65100","AMZN":"#00695C","NFLX":"#B71C1C"}
CO_NAMES = {"AAPL":"Apple","GOOGL":"Google","MSFT":"Microsoft","META":"Meta","AMZN":"Amazon","NFLX":"Netflix"}

def hex_rgba(h,a): return f"rgba({int(h[1:3],16)},{int(h[3:5],16)},{int(h[5:7],16)},{a})"
def sig_color(v):  return "#2e7d32" if v>0.05 else "#c62828" if v<-0.05 else "#e65100"
def pct_bar(v):    return max(2,min(98,int((v+1)/2*100)))
def fmt(v):        return f"{'+' if v>=0 else ''}{v:.3f}"
def H(html):       st.markdown(html, unsafe_allow_html=True)

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [
        os.path.join(base,"data","dashboard_static.json"),
        os.path.join(base,"dashboard_static.json"),
        "dashboard_static.json",
    ]:
        if os.path.exists(p):
            with open(p,"r") as f: return json.load(f)
    raise FileNotFoundError("dashboard_static.json not found")

data         = load_data()
tickers_data = data["tickers"]
generated_at = data["generated_at"]
fore_end     = data["forecast_end"]

# ── Header ────────────────────────────────────────────────────
H(f"""
<div style="background:#1e2133;border-radius:12px;padding:14px 16px;
            margin-bottom:10px;border:1px solid #2a2d3e;border-left:4px solid #1565c0;">
  <div style="font-size:1.05rem;font-weight:800;color:#e8ecf4;line-height:1.3;">
    📈 Stock Market Prediction with AI & ML
  </div>
  <div style="font-size:0.63rem;color:#6b7590;font-family:'IBM Plex Mono',monospace;margin-top:3px;">
    Bidirectional LSTM · Monte Carlo · Technical + Sentiment Analysis
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">
    <span style="font-size:0.6rem;padding:3px 9px;border-radius:20px;font-weight:600;
                 background:#1b5e2025;border:1px solid #2e7d3240;color:#66bb6a;
                 font-family:'IBM Plex Mono',monospace;">● Yahoo Finance · Google Trends · VADER NLP</span>
    <span style="font-size:0.6rem;padding:3px 9px;border-radius:20px;font-weight:600;
                 background:#0d47a125;border:1px solid #1565c040;color:#64b5f6;
                 font-family:'IBM Plex Mono',monospace;">📅 {generated_at}</span>
  </div>
</div>
<div style="background:#1565c012;border:1px solid #1565c025;border-radius:8px;
            padding:7px 14px;font-size:0.62rem;color:#6b8aaa;
            font-family:'IBM Plex Mono',monospace;margin-bottom:10px;line-height:1.6;">
  📊 Pre-computed {generated_at} &nbsp;·&nbsp; Forecast: {data['forecast_start']} → {fore_end}
</div>
""")

# ── Price cards ───────────────────────────────────────────────
H("""<div style="font-size:0.6rem;font-family:'IBM Plex Mono',monospace;color:#4a5270;
      text-transform:uppercase;letter-spacing:1px;font-weight:700;
      margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #2a2d3e;">
  Prices & signals</div>""")

cards = '<div class="cards-grid">'
for ticker, td in tickers_data.items():
    col = COLORS[ticker]
    sc  = "#2e7d32" if td["signal"]=="BULLISH" else "#c62828" if td["signal"]=="BEARISH" else "#e65100"
    cards += f"""
    <div style="background:#1e2133;border:1px solid #2a2d3e;border-radius:10px;
                padding:8px 9px;border-top:3px solid {col};">
      <div style="font-size:0.58rem;font-family:'IBM Plex Mono',monospace;color:#4a5270;">{ticker}</div>
      <div style="font-size:0.95rem;font-weight:800;color:{col};margin:3px 0 1px;">${td['current']:.2f}</div>
      <div style="font-size:0.62rem;font-weight:700;color:{'#4caf50' if td['day_chg']>=0 else '#ef5350'};">{'+' if td['day_chg']>=0 else ''}{td['day_chg']:.1f}%</div>
      <div style="font-size:0.62rem;font-weight:700;color:{'#4caf50' if td['m1_6m_pct']>=0 else '#ef5350'};">{'+' if td['m1_6m_pct']>=0 else ''}{td['m1_6m_pct']:.1f}% 6M</div>
      <div style="font-size:0.6rem;font-weight:700;margin-top:4px;display:flex;align-items:center;gap:3px;">
        <span style="width:5px;height:5px;border-radius:50%;background:{sc};display:inline-block;flex-shrink:0;"></span>
        <span style="color:{sc};">{td['signal'][:4]} {td['confidence']:.0f}%</span>
      </div>
    </div>"""
cards += '</div>'
H(cards)

# ── Filters block ─────────────────────────────────────────────
H("""<div style="background:#1e2133;border-radius:12px;padding:12px 14px;
               border:1px solid #2a2d3e;margin-bottom:8px;">
  <div style="font-size:0.6rem;font-family:'IBM Plex Mono',monospace;color:#4a5270;
              text-transform:uppercase;letter-spacing:1px;font-weight:700;
              margin-bottom:10px;padding-bottom:4px;border-bottom:1px solid #2a2d3e;">
    Filters</div>""")

rng_opts = ["1Y","2Y","3Y","Max"]
co_opts  = ["All Companies"] + [f"{t} — {n}" for t,n in CO_NAMES.items()]
mdl_opts = [
    "Both — Technical + Sentiment",
    "Technical only — Price + indicators",
    "Sentiment only — News + trends",
]

rng    = st.selectbox("Date range",  rng_opts, index=2)
co_sel = st.selectbox("Company",     co_opts,  index=0)
mdl    = st.selectbox("Forecast model", mdl_opts, index=0)

H('</div>')

# Model description
model_descs = {
    "Both — Technical + Sentiment":
        "<b>Both</b> — shows Technical and Sentiment forecasts together on the chart.",
    "Technical only — Price + indicators":
        "<b>Technical</b> — LSTM trained on price history, volume, RSI, MACD and Bollinger Bands. Stable and data-driven.",
    "Sentiment only — News + trends":
        "<b>Sentiment</b> — adds Yahoo Finance news (NLP), Google Trends search interest and FRED macro signals (interest rates, CPI, VIX) on top of price data.",
}

H(f"""
<div style="background:#1e2133;border-radius:10px;padding:10px 14px;
            border:1px solid #2a2d3e;border-left:3px solid #1565c040;
            margin-bottom:8px;font-size:0.67rem;color:#6b7590;line-height:1.8;">
  {model_descs[mdl]}
</div>
""")

# Resolve selections
model       = mdl.split(" — ")[0]   # "Both", "Technical", "Sentiment"
company     = "All" if co_sel=="All Companies" else co_sel.split(" — ")[0]
sel_tickers = list(tickers_data.keys()) if company=="All" else [company]
sel_label   = "All Companies" if company=="All" else CO_NAMES.get(company, company)
range_pts   = {"1Y":252,"2Y":504,"3Y":756,"Max":9999}[rng]

# ── Chart ─────────────────────────────────────────────────────
H(f"""
<div style="background:#1e2133;border-radius:12px;padding:12px 14px 8px;
            border:1px solid #2a2d3e;margin-bottom:8px;">
  <div style="font-size:0.85rem;font-weight:700;color:#e8ecf4;margin-bottom:2px;">
    {sel_label} — {rng} history + 6M forecast
  </div>
  <div style="font-size:0.62rem;color:#6b7590;margin-bottom:8px;">
    Forecast ends {fore_end} · {mdl}
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:6px;">
    <span style="display:flex;align-items:center;gap:4px;font-size:0.6rem;color:#6b7590;">
      <span style="width:14px;height:2px;background:#6b7590;border-radius:1px;display:inline-block;"></span>Historical
    </span>
    <span style="display:flex;align-items:center;gap:4px;font-size:0.6rem;color:#6b7590;">
      <span style="width:14px;border-top:2px dashed #64b5f6;display:inline-block;"></span>Technical
    </span>
    <span style="display:flex;align-items:center;gap:4px;font-size:0.6rem;color:#6b7590;">
      <span style="width:14px;border-top:2px dotted #E65100;display:inline-block;"></span>Sentiment
    </span>
    <span style="display:flex;align-items:center;gap:4px;font-size:0.6rem;color:#6b7590;">
      <span style="width:14px;height:8px;background:#1565c0;opacity:0.3;border-radius:2px;display:inline-block;"></span>90% CI
    </span>
  </div>
""")

fig = go.Figure()
for ticker in sel_tickers:
    td    = tickers_data[ticker]
    color = COLORS[ticker]
    name  = td["company"]
    hd    = td["hist_dates"][-range_pts:]
    hp    = td["hist_prices"][-range_pts:]
    fd    = td["fore_dates"]
    m1p=td["m1_med"]; p05=td["m1_p05"]; p95=td["m1_p95"]
    p25=td["m1_p25"]; p75=td["m1_p75"]; m2p=td["m2_med"]

    fig.add_trace(go.Scatter(
        x=hd, y=hp, name=name, line=dict(color=color, width=2),
        hovertemplate=f"<b>%{{x}}</b><br>{name}: $%{{y:.2f}}<extra></extra>"))

    if model in ["Both","Technical"]:
        fig.add_trace(go.Scatter(x=fd+fd[::-1], y=p95+p05[::-1],
            fill="toself", fillcolor=hex_rgba(color,0.07),
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fd+fd[::-1], y=p75+p25[::-1],
            fill="toself", fillcolor=hex_rgba(color,0.13),
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fd, y=m1p,
            name=f"{name} — Technical", showlegend=False,
            line=dict(color=color, width=2, dash="dash"),
            hovertemplate=f"<b>📈 Technical — %{{x}}</b><br>{name}: $%{{y:.2f}}<extra></extra>"))

    if model in ["Both","Sentiment"]:
        fig.add_trace(go.Scatter(x=fd, y=m2p,
            name=f"{name} — Sentiment", showlegend=False,
            line=dict(color="#E65100", width=2, dash="dot"),
            hovertemplate=f"<b>📊 Sentiment — %{{x}}</b><br>{name}: $%{{y:.2f}}<extra></extra>"))

vts = pd.Timestamp(data["forecast_start"]).timestamp()*1000
fig.add_vline(x=vts, line_dash="dot", line_color="#4a5270", line_width=1,
    annotation_text="Forecast →", annotation_position="top right",
    annotation_font=dict(size=9, color="#6b7590"))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", size=11, color="#c8cfe0"),
    hovermode="x",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                bgcolor="rgba(30,33,51,0.95)", bordercolor="#2a2d3e",
                borderwidth=1, font=dict(size=10, color="#c8cfe0")),
    margin=dict(l=0, r=0, t=28, b=0),
    xaxis=dict(showgrid=True, gridcolor="rgba(42,45,62,0.8)",
               tickfont=dict(family="IBM Plex Mono", size=10, color="#6b7590"),
               tickformat="%b %Y", zeroline=False, linecolor="#2a2d3e", fixedrange=True),
    yaxis=dict(showgrid=True, gridcolor="rgba(42,45,62,0.8)",
               tickfont=dict(family="IBM Plex Mono", size=10, color="#6b7590"),
               tickprefix="$", zeroline=False, linecolor="#2a2d3e", fixedrange=True),
    height=380, dragmode=False, clickmode="none",
)

st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar":False, "scrollZoom":False, "doubleClick":False, "showTips":False,
})

H("""<div style="font-size:0.6rem;color:#4a5270;text-align:center;
      margin-top:4px;font-style:italic;padding-bottom:4px;">
  Hover / tap chart for prices · Tap outside to dismiss
</div></div>""")

# ── Sentiment + News — side by side desktop, stacked mobile ──
avg_n=np.mean([tickers_data[t]["sentiment"]["news"]   for t in sel_tickers])
avg_t=np.mean([tickers_data[t]["sentiment"]["trends"] for t in sel_tickers])
avg_m=np.mean([tickers_data[t]["sentiment"]["macro"]  for t in sel_tickers])
avg_f=np.mean([tickers_data[t]["sentiment"]["final"]  for t in sel_tickers])
fdir ="bullish" if avg_f>0.05 else "bearish" if avg_f<-0.05 else "neutral"

rows=[("News sentiment","VADER NLP · 35%",avg_n),
      ("Google Trends","Search trend · 30%",avg_t),
      ("Macro (FRED)","Rates/CPI/VIX · 35%",avg_m)]

sent=f"""<div style="background:#1e2133;border-radius:12px;padding:12px 14px;border:1px solid #2a2d3e;">
  <div style="font-size:0.6rem;font-family:'IBM Plex Mono',monospace;color:#4a5270;
              text-transform:uppercase;letter-spacing:0.8px;font-weight:700;
              margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid #2a2d3e;">
    Sentiment — {sel_label}</div>"""

for label,sub,val in rows:
    c=sig_color(val)
    sent+=f"""
  <div style="margin-bottom:9px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
      <div><div style="font-size:0.72rem;font-weight:700;color:#c8cfe0;">{label}</div>
           <div style="font-size:0.6rem;color:#4a5270;">{sub}</div></div>
      <div style="font-size:0.72rem;font-family:'IBM Plex Mono',monospace;font-weight:800;color:{sig_color(val)};">{fmt(val)}</div>
    </div>
    <div style="height:6px;background:#13151f;border-radius:3px;overflow:hidden;">
      <div style="height:100%;width:{pct_bar(val)}%;background:{sig_color(val)};border-radius:3px;"></div>
    </div>
  </div>"""

c=sig_color(avg_f)
sent+=f"""
  <div style="background:#13151f;border-radius:9px;padding:10px 12px;margin-top:2px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
      <div><div style="font-size:0.72rem;font-weight:700;color:#e8ecf4;">Overall score</div>
           <div style="font-size:0.6rem;color:#4a5270;">Weighted combination</div></div>
      <div style="font-size:0.85rem;font-family:'IBM Plex Mono',monospace;font-weight:800;color:{c};">{fmt(avg_f)}</div>
    </div>
    <div style="height:8px;background:#1e2133;border-radius:4px;overflow:hidden;">
      <div style="height:100%;width:{pct_bar(avg_f)}%;background:{c};border-radius:4px;"></div>
    </div>
  </div>
  <div style="background:#13151f;border:1px solid #2a2d3e;border-radius:8px;
              padding:10px 12px;font-size:0.65rem;color:#6b7590;line-height:1.9;margin-top:8px;">
    <span style="color:#a0a8c0;font-weight:700;">−1.0</span> very bearish &nbsp;·&nbsp;
    <span style="color:#a0a8c0;font-weight:700;">0.0</span> neutral &nbsp;·&nbsp;
    <span style="color:#a0a8c0;font-weight:700;">+1.0</span> very bullish<br>
    <span style="color:#a0a8c0;font-weight:700;">News</span> = NLP tone on Yahoo Finance headlines<br>
    <span style="color:#a0a8c0;font-weight:700;">Trends</span> = Google search interest (90 days)<br>
    <span style="color:#a0a8c0;font-weight:700;">Macro</span> = Fed rate · CPI · VIX · 10Y Treasury<br>
    Currently <span style="color:{sig_color(avg_f)};font-weight:700;">{fdir} at {fmt(avg_f)}</span>
  </div>
</div>"""

articles=[]
for t in sel_tickers:
    for a in tickers_data[t]["news"]:
        articles.append({**a,"company":tickers_data[t]["company"]})
articles.sort(key=lambda x:abs(x["score"]),reverse=True)

news=f"""<div style="background:#1e2133;border-radius:12px;padding:12px 14px;border:1px solid #2a2d3e;">
  <div style="font-size:0.6rem;font-family:'IBM Plex Mono',monospace;color:#4a5270;
              text-transform:uppercase;letter-spacing:0.8px;font-weight:700;
              margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid #2a2d3e;">
    Headlines — {sel_label}</div>"""

for a in articles[:5]:
    tc ="2e7d32" if a["tone"]=="bull" else "c62828" if a["tone"]=="bear" else "e65100"
    ico="↑ Bullish" if a["tone"]=="bull" else "↓ Bearish" if a["tone"]=="bear" else "→ Neutral"
    news+=f"""
  <div style="background:#13151f;border:1px solid #2a2d3e;border-left:3px solid #{tc};
              border-radius:8px;padding:8px 10px;margin-bottom:7px;">
    <div style="font-size:0.72rem;color:#c8cfe0;font-weight:500;line-height:1.4;margin-bottom:5px;">{a['title']}</div>
    <div style="font-size:0.62rem;color:#6b7590;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
      <span style="color:#{tc};font-weight:700;">{ico}</span>
      <span style="color:#a0a8c0;">{a['company']}</span>
      <span style="font-family:'IBM Plex Mono',monospace;color:#{tc};">{'+' if a['score']>=0 else ''}{a['score']:.3f}</span>
      <a href="{a['url']}" target="_blank"
         style="color:#64b5f6;text-decoration:none;font-weight:700;margin-left:auto;">Read →</a>
    </div>
  </div>"""
news+="</div>"

H(f'<div class="bottom-grid">{sent}{news}</div>')

# ── Disclaimer ────────────────────────────────────────────────
H(f"""
<div style="background:#1e2133;border:1px solid #2a2d3e;border-left:3px solid #f0a00060;
            border-radius:10px;padding:10px 14px;font-size:0.65rem;color:#6b7590;
            line-height:1.8;margin-bottom:8px;">
  <span style="color:#ffb74d;font-weight:700;">⚠ Not financial advice.</span>
  Forecasts pre-computed on <span style="color:#a0a8c0;font-weight:700;">{generated_at}</span>
  using Yahoo Finance prices, Google Trends and VADER NLP.
  Short-term forecasts are more reliable than 6-month projections.
  Always conduct your own research before investing.
</div>
<div style="text-align:center;font-size:0.58rem;color:#3a4260;
            font-family:'IBM Plex Mono',monospace;padding-bottom:1.5rem;">
  TensorFlow · Yahoo Finance · Google Trends · VADER NLP · Streamlit · {generated_at}
</div>""")
