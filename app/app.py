"""
Bearing RUL Diagnostic System — Streamlit Application.
Optimized for both non-technical plant operators and technical research evaluators.

UI: two themes selectable from the sidebar toggle
    - Light  : "Foodaily"-style green sidebar, soft grey canvas, white rounded cards
    - Dark   : "Concept 6" data-driven particle terminal (#0B0E14 canvas, neon
               cyan / amber / violet glow, particle RUL trend, fleet-health aura
               vortex, SHAP constellation map, holographic metric rings)

PIPELINE:
  FEMTO-ST / PRONOSTIA (25.6 kHz, 2560-sample / 0.1s windows)
  -> Time / Frequency / Wavelet Features
  -> CatBoost Point Prediction
  -> XGBoost Quantile Models + CQR (90% interval)
  -> SHAP Explainability
  -> 4-Tier Risk-Aware Maintenance Policy:
      - Low: Continue monitoring
      - Medium: Inspect before replacement
      - High: Schedule maintenance
      - Critical: Replace immediately
"""

# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import os
import json
import io
import pandas as pd  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import streamlit as st  # type: ignore[import-untyped]
import streamlit.components.v1 as components  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
from datetime import datetime
from ollama import chat  # type: ignore[import-untyped]

# Upload & inference dependencies
from scipy import stats  # type: ignore[import-untyped]
from scipy.fft import fft, fftfreq  # type: ignore[import-untyped]
import pywt  # type: ignore[import-untyped]
import joblib  # type: ignore[import-untyped]
import shap  # type: ignore[import-untyped]


# ============================================================
# CONFIGURATION & ARTIFACT DIRECTORIES
# ============================================================
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)

_default_dirs = [
    os.path.join(_PROJECT_ROOT, "deployment_artifacts"),
    os.path.join(_APP_DIR, "deployment_artifacts"),
    os.path.join(_PROJECT_ROOT, "outputs"),
    os.path.join(_PROJECT_ROOT, "chapter7_outputs"),
    os.path.join(_PROJECT_ROOT, "chapter6_outputs"),
    os.path.join(_PROJECT_ROOT, "chapter5_outputs"),
    os.path.join(_PROJECT_ROOT, "processed_features"),
    "deployment_artifacts",
    "outputs",
    "chapter7_outputs",
    "chapter6_outputs",
    "chapter5_outputs",
    "processed_features",
]

if os.environ.get("BEARING_APP_OUTPUT_DIRS"):
    OUTPUT_DIRS = [
        d.strip()
        for d in os.environ["BEARING_APP_OUTPUT_DIRS"].split(",")
        if d.strip()
    ]
elif os.environ.get("BEARING_APP_OUTPUT_DIR"):
    OUTPUT_DIRS = [os.environ["BEARING_APP_OUTPUT_DIR"]]
else:
    OUTPUT_DIRS = _default_dirs

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bearing RUL Diagnostic & Maintenance System",
    page_icon=":material/settings:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME (Light = green template, Dark = particle terminal)
# The toggle itself lives in the sidebar; its state is read here
# so the palette is known before any styling is injected.
# ============================================================
DARK_MODE = bool(st.session_state.get("ui_dark_mode", False))

_LIGHT = {
    "bg": "#EEF2F4",
    "surface": "#FFFFFF",
    "surface_alt": "#F3F6F8",
    "border": "#E6EAED",
    "text": "#1E2A32",
    "text_dim": "#8A949C",
    "accent": "#439A77",
    "accent_dim": "#E3F1EA",
    "info": "#3B82C4",
    "healthy": "#43A07A",
    "caution": "#E2A23B",
    "warning": "#F28C4B",
    "critical": "#D64545",
    "violet": "#7C6BD6",
}

_DARK = {
    "bg": "#0B0E14",
    "surface": "#11161F",
    "surface_alt": "#161C27",
    "border": "#1F2836",
    "text": "#E6EDF3",
    "text_dim": "#7F8C9B",
    "accent": "#19E3F2",
    "accent_dim": "#0F2A33",
    "info": "#7C9CFF",
    "healthy": "#2EE6B8",
    "caution": "#FFC24B",
    "warning": "#FF8A4C",
    "critical": "#FF4D6D",
    "violet": "#A78BFA",
}

_LIGHT_THEME = {
    "app_bg": _LIGHT["bg"],
    "shadow": "0 2px 14px rgba(30, 42, 50, 0.05)",
    "card_border": "none",
    "on_accent": "#FFFFFF",
    "grid": "#EDF0F2",
    "band": "rgba(67,154,119,0.25)",
    "sidebar_bg": _LIGHT["accent"],
    "sidebar_text": "#FFFFFF",
    "sidebar_eyebrow": "#CFEBDD",
    "sidebar_hr": "rgba(255,255,255,0.25)",
    "sidebar_border": "none",
    "nav_hover": "rgba(255,255,255,0.12)",
    "nav_active": "rgba(255,255,255,0.24)",
    "nav_active_ring": "transparent",
    "btn_bg": _LIGHT["accent"],
    "btn_text": "#FFFFFF",
    "btn_border": _LIGHT["accent"],
    "btn_hover_bg": "#378266",
    "btn_hover_text": "#FFFFFF",
    "btn_glow": "none",
    "sbtn_bg": "#FFFFFF",
    "sbtn_text": _LIGHT["accent"],
    "sbtn_border": "#FFFFFF",
    "sbtn_hover_bg": _LIGHT["accent_dim"],
    "hl_bg": _LIGHT["accent"],
    "hl_border": "none",
    "hl_text": "#FFFFFF",
    "upload_bg": _LIGHT["accent_dim"],
    "upload_border": _LIGHT["accent"],
    "upload_btn_text": "#FFFFFF",
    "input_bg": "#FFFFFF",
}

_DARK_THEME = {
    "app_bg": (
        "radial-gradient(circle at 12% -10%, rgba(167,139,250,0.13), transparent 42%), "
        "radial-gradient(circle at 100% 0%, rgba(25,227,242,0.09), transparent 38%), "
        "#0B0E14"
    ),
    "shadow": "0 0 0 1px rgba(255,255,255,0.03), 0 0 24px rgba(25,227,242,0.04)",
    "card_border": "1px solid rgba(255,255,255,0.07)",
    "on_accent": "#0B0E14",
    "grid": "#1B2330",
    "band": "rgba(167,139,250,0.28)",
    "sidebar_bg": "#0A0D13",
    "sidebar_text": "#E6EDF3",
    "sidebar_eyebrow": "#19E3F2",
    "sidebar_hr": "rgba(255,255,255,0.10)",
    "sidebar_border": "1px solid rgba(25,227,242,0.18)",
    "nav_hover": "rgba(255,255,255,0.05)",
    "nav_active": "rgba(25,227,242,0.10)",
    "nav_active_ring": "rgba(25,227,242,0.45)",
    "btn_bg": "rgba(25,227,242,0.10)",
    "btn_text": "#19E3F2",
    "btn_border": "#19E3F2",
    "btn_hover_bg": "#19E3F2",
    "btn_hover_text": "#0B0E14",
    "btn_glow": "0 0 16px rgba(25,227,242,0.25)",
    "sbtn_bg": "rgba(25,227,242,0.10)",
    "sbtn_text": "#19E3F2",
    "sbtn_border": "rgba(25,227,242,0.55)",
    "sbtn_hover_bg": "rgba(25,227,242,0.22)",
    "hl_bg": "rgba(25,227,242,0.08)",
    "hl_border": "1px solid rgba(25,227,242,0.45)",
    "hl_text": "#19E3F2",
    "upload_bg": "rgba(25,227,242,0.05)",
    "upload_border": "#19E3F2",
    "upload_btn_text": "#0B0E14",
    "input_bg": "#11161F",
}

COLOR = _DARK if DARK_MODE else _LIGHT
THEME = _DARK_THEME if DARK_MODE else _LIGHT_THEME

# Neon palette used by the particle / canvas widgets (dark theme only).
_FX = {
    "cyan": "#19E3F2",
    "violet": "#A78BFA",
    "amber": "#FFC24B",
    "red": "#FF4D6D",
    "teal": "#2EE6B8",
    "orange": "#FF8A4C",
    "text": "#E6EDF3",
    "dim": "#7F8C9B",
    "grid": "rgba(255,255,255,0.07)",
}

# Maintenance action colours. Defined here so every page can safely use
# action_color() without relying on a later declaration.
ACTION_COLOR = {
    "Replace immediately": COLOR["critical"],
    "Inspect before replacement": COLOR["caution"],
    "Schedule maintenance": COLOR["warning"],
    "Continue monitoring": COLOR["healthy"],
}

COLORWAY = [
    COLOR["accent"], COLOR["violet"], COLOR["caution"],
    COLOR["info"], COLOR["warning"], COLOR["critical"],
]

# Plotly Express defaults follow the active theme.
px.defaults.template = "plotly_dark" if DARK_MODE else "plotly"
px.defaults.color_discrete_sequence = COLORWAY


def apply_plotly_readability(fig):
    """Keep Plotly text readable on the active dashboard background."""
    fig.update_layout(
        font={"color": COLOR["text"], "family": "Poppins, sans-serif"},
        legend={"font": {"color": COLOR["text"]}},
        hoverlabel={"font": {"color": COLOR["text"]}, "bgcolor": COLOR["surface"]},
        colorway=COLORWAY,
    )
    # A title object that carries only styling makes Plotly render the literal
    # string "undefined" above the plot, so the title font is restyled only
    # when the figure actually has a title.
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_font={"color": COLOR["text"]})
    fig.update_xaxes(
        title_font={"color": COLOR["text"]},
        tickfont={"color": COLOR["text"]},
        linecolor=COLOR["border"],
        tickcolor=COLOR["border"],
        gridcolor=THEME["grid"],
        zeroline=False,
    )
    fig.update_yaxes(
        title_font={"color": COLOR["text"]},
        tickfont={"color": COLOR["text"]},
        linecolor=COLOR["border"],
        tickcolor=COLOR["border"],
        gridcolor=THEME["grid"],
        zeroline=False,
    )
    return fig


# ------------------------------------------------------------
# GLOBAL CSS (variables are generated from the active theme)
# ------------------------------------------------------------
def _css_variables():
    merged = {**COLOR, **THEME}
    return "\n".join(
        f"--{key.replace('_', '-')}: {value};" for key, value in merged.items()
    )


_BASE_CSS = """
/* ---------- Canvas ---------- */
.stApp {
    background: var(--app-bg);
    color: var(--text);
    font-family: 'Poppins', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2.2rem;
}

h1, h2, h3, h4, .panel-title {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    color: var(--text) !important;
}

.stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"] {
    color: var(--text);
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: var(--text-dim) !important;
}

.mono {
    font-family: 'Poppins', sans-serif !important;
}

code {
    font-family: 'IBM Plex Mono', monospace !important;
}

.stMetric [data-testid="stMetricValue"] {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--btn-bg) !important;
    color: var(--btn-text) !important;
    border: 1px solid var(--btn-border) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.1rem !important;
    box-shadow: var(--btn-glow);
}

.stButton > button * {
    color: var(--btn-text) !important;
}

.stButton > button:hover {
    background: var(--btn-hover-bg) !important;
    color: var(--btn-hover-text) !important;
    border-color: var(--btn-hover-bg) !important;
}

.stButton > button:hover * {
    color: var(--btn-hover-text) !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: var(--sidebar-bg) !important;
    color: var(--sidebar-text) !important;
    border-right: var(--sidebar-border);
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label div,
section[data-testid="stSidebar"] [data-testid="stRadio"] label span {
    color: var(--sidebar-text) !important;
}

section[data-testid="stSidebar"] .eyebrow {
    color: var(--sidebar-eyebrow) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--sidebar-hr) !important;
}

/* Navigation: radio becomes a vertical menu with a highlight bar */
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.15rem;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    width: 100%;
    padding: 0.62rem 0.85rem;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.15s ease, box-shadow 0.15s ease;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 0.9rem;
    font-weight: 500;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: var(--nav-hover);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: var(--nav-active);
    box-shadow: inset 0 0 0 1px var(--nav-active-ring);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    font-weight: 600;
}

/* Sidebar button */
section[data-testid="stSidebar"] .stButton > button {
    background: var(--sbtn-bg) !important;
    border-color: var(--sbtn-border) !important;
    width: 100%;
}

section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button * {
    color: var(--sbtn-text) !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--sbtn-hover-bg) !important;
    border-color: var(--sbtn-hover-bg) !important;
}

/* ---------- Typography helpers ---------- */
.eyebrow {
    font-family: 'Poppins', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* ---------- Cards ---------- */
.panel {
    background: var(--surface);
    border: var(--card-border);
    border-radius: 18px;
    padding: 1.15rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}

.panel-accent {
    border-left: 4px solid var(--accent);
}

.badge {
    display: inline-block;
    font-family: 'Poppins', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    border: 1px solid currentColor;
}
.badge-live { color: var(--healthy); }
.badge-demo { color: var(--caution); }

.action-card {
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    border: var(--card-border);
    border-top: 4px solid var(--zone-color, var(--accent));
    background: var(--surface);
    box-shadow: var(--shadow);
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.15rem;
}

.hero-subtitle {
    color: var(--text-dim);
    font-size: 0.98rem;
    margin-bottom: 0.6rem;
}

.exec-summary {
    background: var(--surface);
    border: var(--card-border);
    border-left: 4px solid var(--accent);
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    font-size: 0.98rem;
    line-height: 1.6;
    color: var(--text);
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
}

.kpi-card {
    background: var(--surface);
    border: var(--card-border);
    border-radius: 18px;
    padding: 1rem 1.15rem;
    height: 100%;
    box-shadow: var(--shadow);
}

.kpi-label {
    font-family: 'Poppins', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-family: 'Poppins', sans-serif;
    font-size: 1.85rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.1;
}

.kpi-help {
    color: var(--text-dim);
    font-size: 0.8rem;
    margin-top: 0.35rem;
    line-height: 1.4;
}

.status-pill {
    display: inline-block;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    color: var(--on-accent);
}
.status-pill-healthy { background: var(--healthy); }
.status-pill-attention { background: var(--caution); }
.status-pill-warning { background: var(--warning); }
.status-pill-critical { background: var(--critical); color: #fff; }
.status-pill-info { background: var(--info); }

.result-hero {
    background: var(--surface);
    border: var(--card-border);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    text-align: center;
    box-shadow: var(--shadow);
}
.result-hero-label {
    font-family: 'Poppins', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
}
.result-hero-status {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    margin-bottom: 0.2rem;
}
.result-hero-value {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 3rem;
    line-height: 1.05;
}

.step-card {
    background: var(--surface);
    border: var(--card-border);
    border-radius: 14px;
    padding: 0.7rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.7rem;
    box-shadow: var(--shadow);
}
.step-number {
    background: var(--accent);
    color: var(--on-accent);
    font-family: 'Poppins', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.step-text {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text) !important;
}

/* ---------- Metric cards: first one in each row is highlighted ---------- */
[data-testid="stMetric"] {
    background: var(--surface);
    border: var(--card-border);
    border-radius: 18px;
    padding: 1rem 1.15rem;
    box-shadow: var(--shadow);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: var(--text-dim) !important;
    font-size: 0.82rem;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: var(--text) !important;
}

div[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stMetric"] {
    background: var(--hl-bg);
    border: var(--hl-border);
}

div[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stMetricLabel"],
div[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stMetricLabel"] *,
div[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stMetricValue"],
div[data-testid="stHorizontalBlock"] > div:first-child [data-testid="stMetricValue"] * {
    color: var(--hl-text) !important;
}

/* ---------- Expanders, tables, alerts, inputs ---------- */
div[data-testid="stExpander"] {
    background: var(--surface);
    border: var(--card-border) !important;
    border-radius: 16px;
    box-shadow: var(--shadow);
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary * {
    color: var(--text) !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--shadow);
}

div[data-testid="stAlert"] {
    border-radius: 14px;
    border: none;
}

div[data-baseweb="select"] > div {
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

hr {
    border-color: var(--border);
}

/* ---------- File uploader ---------- */
div[data-testid="stFileUploader"] > label,
div[data-testid="stFileUploader"] > label p,
div[data-testid="stFileUploader"] > label span {
    color: var(--text) !important;
}

div[data-testid="stFileUploader"] section {
    background: var(--upload-bg) !important;
    border: 1.5px dashed var(--upload-border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
}

div[data-testid="stFileUploader"] section *,
div[data-testid="stFileUploader"] section p,
div[data-testid="stFileUploader"] section span {
    color: var(--text) !important;
}

div[data-testid="stFileUploader"] button {
    color: var(--upload-btn-text) !important;
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    border-radius: 10px !important;
}

div[data-testid="stFileUploader"] button * {
    color: var(--upload-btn-text) !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
    color: var(--text) !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
"""

# Extra rules that only apply to the dark particle theme.
_DARK_CSS = """
/* ---------- Dark particle theme extras ---------- */
@keyframes hum {
    0%, 100% { box-shadow: 0 0 0 1px rgba(25,227,242,0.10), 0 0 18px rgba(25,227,242,0.06); }
    50%      { box-shadow: 0 0 0 1px rgba(25,227,242,0.24), 0 0 34px rgba(25,227,242,0.16); }
}

/* ambient "hum": result cards pulse slowly */
.result-hero, .action-card {
    animation: hum 5s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
    .result-hero, .action-card { animation: none; }
}

.panel, .exec-summary, .step-card, .kpi-card, [data-testid="stMetric"] {
    backdrop-filter: blur(6px);
}

/* Streamlit's table canvas cannot be restyled directly: invert it for dark */
div[data-testid="stDataFrame"] {
    filter: invert(0.92) hue-rotate(180deg);
}

div[data-testid="stAlert"] {
    background: rgba(25,227,242,0.07) !important;
    border: 1px solid rgba(25,227,242,0.22) !important;
}

div[data-testid="stAlert"] * {
    color: var(--text) !important;
}

div[data-testid="stPlotlyChart"] {
    background: var(--surface);
    border: var(--card-border);
    border-radius: 18px;
    padding: 0.6rem;
    box-shadow: var(--shadow);
}

div[data-baseweb="select"] * {
    color: var(--text) !important;
}

div[data-testid="stSlider"] * {
    color: var(--text);
}

code {
    background: rgba(255,255,255,0.08) !important;
    color: var(--violet) !important;
}

a { color: var(--accent) !important; }

[data-testid="stToolbar"] *,
[data-testid="stExpandSidebarButton"] * {
    color: var(--text-dim) !important;
}
"""

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700'
    '&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">\n\n'
    "<style>\n:root {\n"
    + _css_variables()
    + "\n}\n"
    + _BASE_CSS
    + (_DARK_CSS if DARK_MODE else "")
    + "\n</style>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER DATA LOADERS
# ============================================================
def _find_file(filename):
    for d in OUTPUT_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

def _try_load_csv(filename):
    path = _find_file(filename)
    if path:
        try:
            return pd.read_csv(path), True
        except Exception:
            return None, False
    return None, False

@st.cache_data
def load_model_comparison():
    df, live = _try_load_csv("model_performance_final.csv")
    if df is None:
        df, live = _try_load_csv("model_comparison.csv")
    if df is not None:
        return df, live

    demo = pd.DataFrame({
        "model": ["CatBoost", "Random Forest", "XGBoost", "LightGBM"],
        "MAE": [18.727481, 19.152684, 19.362556, 20.625481],
        "RMSE": [22.244893, 24.681066, 23.724841, 24.580809],
        "R2": [0.406197, 0.269014, 0.324558, 0.274940],
    })
    return demo, False

@st.cache_data
def load_uncertainty_metrics():
    df, live = _try_load_csv("uncertainty_metrics.csv")
    if df is not None:
        return df.iloc[0].to_dict(), live
    demo = {
        "PICP": 0.889,
        "target_coverage": 0.90,
        "MPIW": 80.74,
        "CWC": 1.709
    }
    return demo, False

@st.cache_data
def load_predictions():
    df, live = _try_load_csv("framework_results.csv")
    if df is None:
        df, live = _try_load_csv("maintenance_recommendations.csv")
    if df is not None:
        df = _synchronize_prediction_actions(df)
        return df, live

    rng = np.random.default_rng(7)
    n = 60
    bearings = rng.choice(["Bearing1_2", "Bearing2_2", "Bearing3_2"], n)
    pred = np.clip(rng.normal(35, 25, n), 0, 100)
    width = np.clip(rng.normal(55, 20, n), 15, 95)
    lower = np.clip(pred - width / 2, 0, 100)
    upper = np.clip(pred + width / 2, 0, 100)

    demo = pd.DataFrame({
        "bearing": bearings,
        "snapshot_idx": rng.integers(1, 900, n),
        "predicted_RUL_pct": pred,
        "lower_bound_pct": lower,
        "upper_bound_pct": upper,
        "interval_width_pct": upper - lower,
        "top_driver_1_feature": rng.choice(
            ["horiz_band_energy_2", "horiz_wavelet_detail_2_energy_ratio", "horiz_std"], n
        ),
    })
    return _synchronize_prediction_actions(demo), False

@st.cache_data
def load_feature_importance():
    df, live = _try_load_csv("shap_values.csv")
    if df is not None:
        shap_cols = [c for c in df.columns if c.startswith("shap_")]
        importance = df[shap_cols].abs().mean().sort_values(ascending=False)
        out = pd.DataFrame({
            "feature": [c.replace("shap_", "") for c in importance.index],
            "importance": importance.values
        })
        return out.head(10), live

    demo = pd.DataFrame({
        "feature": [
            "horiz_band_energy_2", "horiz_band_energy_5",
            "horiz_std", "horiz_shape_factor", "horiz_variance"
        ],
        "importance": [5.48, 2.91, 2.88, 2.01, 1.94],
    })
    return demo, False


# ============================================================
# POLICY & CLASSIFICATION HELPERS
# ============================================================
def zone_for_rul(rul_pct):
    if rul_pct < 20:
        return COLOR["critical"], "CRITICAL"
    elif rul_pct < 40:
        return COLOR["caution"], "MODERATE"
    return COLOR["healthy"], "HEALTHY"

def health_status(rul_pct):
    if rul_pct is None or (isinstance(rul_pct, float) and np.isnan(rul_pct)):
        return "Unknown", "Unknown", COLOR["text_dim"], "status-pill-info"
    if rul_pct < 20:
        return "Critical", "High", COLOR["critical"], "status-pill-critical"
    elif rul_pct < 40:
        return "Needs Attention", "Medium", COLOR["caution"], "status-pill-attention"
    return "Healthy", "Low", COLOR["healthy"], "status-pill-healthy"

def uncertainty_risk_level(point_rul, lower_bound):
    """
    Automatic 3-tier risk classification.

    The user never selects the risk level.

    Low:
        Point RUL >= 40% AND calibrated 90% lower bound >= 20%

    Medium:
        Point RUL >= 40% AND calibrated 90% lower bound < 20%

    High:
        Point RUL < 40%

    A very severe High-risk case (point RUL < 20% and lower bound < 20%)
    can still receive "Replace immediately" as the recommended action.
    The risk label remains High so the presentation uses only
    Low / Medium / High.
    """
    if point_rul is None or lower_bound is None:
        return "Unknown", COLOR["text_dim"], "status-pill-info"

    point_rul = float(point_rul)
    lower_bound = float(lower_bound)

    if point_rul < 40:
        return "High", COLOR["warning"], "status-pill-warning"

    if lower_bound < 20:
        return "Medium", COLOR["caution"], "status-pill-attention"

    return "Low", COLOR["healthy"], "status-pill-healthy"


def maintenance_action(point_rul, lower_bound):
    """
    Map the automatically calculated risk state to an operator action.

    High risk with both point RUL and lower bound below 20% is escalated
    to immediate replacement. This is an action severity, not a fourth
    displayed risk category.
    """
    risk, _, _ = uncertainty_risk_level(point_rul, lower_bound)

    if risk == "High" and float(point_rul) < 20 and float(lower_bound) < 20:
        return "Replace immediately"

    return {
        "High": "Schedule maintenance",
        "Medium": "Inspect before replacement",
        "Low": "Continue monitoring",
    }.get(risk, "Continue monitoring")


def _synchronize_prediction_actions(df):
    """Make displayed fleet actions follow the same automatic risk policy."""
    required = {"predicted_RUL_pct", "lower_bound_pct"}
    if df is None or not required.issubset(df.columns):
        return df

    df = df.copy()
    actions = [
        maintenance_action(point, lower)
        for point, lower in zip(
            df["predicted_RUL_pct"],
            df["lower_bound_pct"],
        )
    ]

    df["recommended_action"] = actions

    if "risk_aware_action" in df.columns:
        df["risk_aware_action"] = actions

    return df


def risk_explanation(point_rul, lower_bound):
    risk, _, _ = uncertainty_risk_level(point_rul, lower_bound)

    point_rul = float(point_rul)
    lower_bound = float(lower_bound)

    if risk == "High":
        if point_rul < 20 and lower_bound < 20:
            return (
                "The predicted RUL is below 20% and the calibrated 90% "
                "lower bound is also below 20%. The system therefore "
                "classifies the bearing as High Risk and escalates the "
                "recommended action to immediate replacement according "
                "to site procedures."
            )
        return (
            "The predicted RUL is below 40%. The system automatically "
            "classifies the bearing as High Risk and recommends scheduling "
            "maintenance."
        )

    if risk == "Medium":
        return (
            "The predicted RUL is at least 40%, but the calibrated 90% "
            "lower bound is below 20%. The uncertainty indicates elevated "
            "risk, so the system classifies the bearing as Medium Risk "
            "and recommends inspection before replacement."
        )

    if risk == "Low":
        return (
            "The predicted RUL is at least 40% and the calibrated 90% "
            "lower bound is at least 20%. The system automatically "
            "classifies the bearing as Low Risk and recommends continued "
            "monitoring."
        )

    return "Risk evaluation unavailable."


def action_color(action):
    return ACTION_COLOR.get(action, COLOR["accent"])

def get_action_col(df):
    return "recommended_action" if "recommended_action" in df.columns else "risk_aware_action"

def data_badge(is_live, label="pipeline output"):
    if is_live:
        st.markdown(f'<span class="badge badge-live">● LIVE — {label}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge badge-demo">○ FALLBACK — {label} artifact not found</span>', unsafe_allow_html=True)


def fleet_health_index(df):
    """
    Fleet health index used by the aura vortex (dark theme).

    Low-risk windows count fully, Medium-risk windows count half, High-risk
    windows count zero. Returns (score 0..1, risk counts) or (None, {}).
    """
    needed = {"predicted_RUL_pct", "lower_bound_pct"}
    if df is None or df.empty or not needed.issubset(df.columns):
        return None, {}

    counts = {"Low": 0, "Medium": 0, "High": 0}
    for point, lower in zip(df["predicted_RUL_pct"], df["lower_bound_pct"]):
        level, _, _ = uncertainty_risk_level(point, lower)
        if level in counts:
            counts[level] += 1

    total = sum(counts.values())
    if total == 0:
        return None, counts
    return (counts["Low"] + 0.5 * counts["Medium"]) / total, counts


# ============================================================
# PARTICLE / CANVAS WIDGETS (dark theme)
# Each widget is a self-contained HTML + canvas/SVG page rendered
# through streamlit components (no extra Python packages needed).
# ============================================================
_FX_HEAD = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html,body{margin:0;padding:0;background:transparent;font-family:'Poppins',sans-serif;color:__TEXT__;overflow:hidden}
.wrap{position:relative;width:100%;height:__H__px;box-sizing:border-box;border-radius:18px;background:__SURF__;border:1px solid __BORDER__;overflow:hidden}
canvas{display:block;width:100%;height:100%}
__EXTRA_CSS__
</style></head><body>"""


def _fx_html(extra_css, body, script, height, **placeholders):
    html = _FX_HEAD + body + "<script>" + script + "</script></body></html>"
    html = html.replace("__EXTRA_CSS__", extra_css)
    for key, value in placeholders.items():
        html = html.replace(f"__{key}__", value)
    common = {
        "TEXT": COLOR["text"],
        "DIM": COLOR["text_dim"],
        "SURF": COLOR["surface"],
        "BORDER": "rgba(255,255,255,0.08)",
        "H": str(height),
    }
    for key, value in common.items():
        html = html.replace(f"__{key}__", value)
    return html


def _downsample(arrays, max_pts=300):
    n = len(arrays[0])
    if n <= max_pts:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_pts).astype(int))
    return [np.asarray(a)[idx] for a in arrays]


# ---------------- 1. Particle-field RUL trend ----------------
_TREND_JS = r"""
(function(){
var D = __DATA__, C = __COLORS__, H = __H__;
var n = D.y.length;
var wrap = document.getElementById('wrap'), cv = document.getElementById('cv'), ctx = cv.getContext('2d');
var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
var M = {l:44, r:18, t:30, b:32};
var W = 600, dpr = Math.min(window.devicePixelRatio || 1, 2);
function resize(){ W = wrap.clientWidth || 600; cv.width = W*dpr; cv.height = H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); }
resize();
if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);

function plotW(){ return W - M.l - M.r; }
function px(f){ return M.l + (n > 1 ? f/(n-1) : 0.5) * plotW(); }
function py(v){ return M.t + (1 - v/100) * (H - M.t - M.b); }
function interp(a, f){ f = Math.max(0, Math.min(n-1, f)); var i = Math.floor(f), j = Math.min(n-1, i+1), t = f - i; return a[i]*(1-t) + a[j]*t; }
function gauss(){ return (Math.random()+Math.random()+Math.random()-1.5)*0.8; }

// Density is higher where the interval is narrow (model is confident).
var cum = [], tot = 0;
for (var j=0; j<n-1; j++){ var w = 1/(8 + ((D.hi[j]-D.lo[j]) + (D.hi[j+1]-D.lo[j+1]))/2); tot += w; cum.push(tot); }
function pickF(){
  if (!cum.length) return 0;
  var r = Math.random()*tot, lo = 0, hi = cum.length-1;
  while (lo < hi){ var m = (lo+hi) >> 1; if (cum[m] < r) lo = m+1; else hi = m; }
  return lo + Math.random();
}

var N = 1700, P = [];
for (var i=0; i<N; i++){
  var band = (i % 2 === 1);
  P.push({f: pickF(), band: band, u: Math.random(), g: gauss(), ph: Math.random()*6.283, x: 0, y: 0, init: false, sz: band ? 1.3 : 1.9});
}

var mx = 0, my = 0, hover = false, tip = 0, hIdx = 0;
function onMove(e){ var r = cv.getBoundingClientRect(); var p = e.touches ? e.touches[0] : e; mx = p.clientX - r.left; my = p.clientY - r.top; hover = (mx >= M.l - 10 && mx <= W - M.r + 10); }
function onLeave(){ hover = false; }
cv.addEventListener('mousemove', onMove); cv.addEventListener('mouseleave', onLeave);
cv.addEventListener('touchstart', onMove, {passive:true}); cv.addEventListener('touchmove', onMove, {passive:true}); cv.addEventListener('touchend', onLeave);

function rr(x,y,w,h,r){ ctx.beginPath(); ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath(); }

function drawGrid(){
  ctx.font = '11px Poppins, sans-serif'; ctx.lineWidth = 1; ctx.textBaseline = 'alphabetic';
  for (var v=0; v<=100; v+=20){
    var y = py(v);
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(W-M.r, y); ctx.stroke();
    ctx.fillStyle = C.dim; ctx.textAlign = 'right'; ctx.fillText(v+'%', M.l-8, y+4);
  }
  [[40, C.amber, '40% high-risk'], [20, C.red, '20% severe']].forEach(function(t){
    ctx.setLineDash([5,5]); ctx.strokeStyle = t[1]; ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(M.l, py(t[0])); ctx.lineTo(W-M.r, py(t[0])); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 0.9; ctx.fillStyle = t[1]; ctx.textAlign = 'right';
    ctx.fillText(t[2], W-M.r-4, py(t[0])-5); ctx.globalAlpha = 1;
  });
  var k = Math.min(6, n); ctx.fillStyle = C.dim; ctx.textAlign = 'center';
  for (var q=0; q<k; q++){ var ii = k > 1 ? Math.round(q*(n-1)/(k-1)) : 0; ctx.fillText(D.x[ii], px(ii), H-10); }
  ctx.textAlign = 'left';
  ctx.fillStyle = C.cyan; ctx.fillRect(M.l, 12, 8, 8); ctx.fillStyle = C.dim; ctx.fillText('Point RUL', M.l+14, 20);
  ctx.fillStyle = C.violet; ctx.fillRect(M.l+84, 12, 8, 8); ctx.fillStyle = C.dim; ctx.fillText('90% range', M.l+98, 20);
}

function frame(ts){
  var t = ts/1000, amp = reduce ? 0 : 1;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  ctx.clearRect(0, 0, W, H);
  drawGrid();

  var sx = 0, sy = 0;
  if (hover){
    hIdx = Math.max(0, Math.min(n-1, Math.round((mx - M.l)/plotW()*(n-1))));
    sx = px(hIdx); sy = py(D.y[hIdx]);
  }
  tip += ((hover ? 1 : 0) - tip) * 0.12;

  ctx.globalCompositeOperation = 'lighter';
  for (var i=0; i<P.length; i++){
    var p = P[i];
    var fe = p.f + amp*0.35*Math.sin(t*0.5 + p.ph);
    var tx = px(fe) + amp*1.2*Math.sin(t*0.8 + p.ph*2), ty;
    if (p.band){
      var lo = interp(D.lo, fe), hi = interp(D.hi, fe);
      ty = py(lo + (p.u + amp*0.04*Math.sin(t*0.7 + p.ph))*(hi - lo));
    } else {
      ty = py(interp(D.y, fe)) + p.g*3 + amp*1.5*Math.cos(t*0.9 + p.ph);
    }
    var snapped = 0;
    if (tip > 0.02){
      var dx = tx - mx, dy = ty - my, d = Math.sqrt(dx*dx + dy*dy);
      if (d < 120){
        var k2 = 1 - d/120, wave = Math.sin(d*0.14 - t*7)*9*k2*tip;
        tx += (dx/(d+0.001))*wave; ty += (dy/(d+0.001))*wave;
      }
      var ex = tx - sx, ey = ty - sy, ds = Math.sqrt(ex*ex + ey*ey);
      if (ds < 80){
        var s = (1 - ds/80)*tip, ring = 7 + 9*((p.ph*7) % 1), a = p.ph*11 + t*1.2;
        tx = tx*(1-s) + (sx + Math.cos(a)*ring)*s;
        ty = ty*(1-s) + (sy + Math.sin(a)*ring)*s;
        snapped = s;
      }
    }
    if (!p.init){ if (reduce){ p.x = tx; p.y = ty; } else { p.x = Math.random()*W; p.y = Math.random()*H; } p.init = true; }
    p.x += (tx - p.x)*0.14; p.y += (ty - p.y)*0.14;
    ctx.globalAlpha = p.band ? 0.38 : 0.85;
    ctx.fillStyle = snapped > 0.35 ? C.amber : (p.band ? C.violet : C.cyan);
    ctx.fillRect(p.x, p.y, p.sz, p.sz);
  }
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;

  if (tip > 0.03){
    ctx.globalAlpha = tip;
    ctx.strokeStyle = 'rgba(255,194,75,0.35)'; ctx.setLineDash([3,4]);
    ctx.beginPath(); ctx.moveTo(sx, M.t); ctx.lineTo(sx, H-M.b); ctx.stroke(); ctx.setLineDash([]);
    var lines = ['Window ' + D.x[hIdx], 'RUL ' + D.y[hIdx].toFixed(1) + '%', '90% range ' + D.lo[hIdx].toFixed(1) + ' - ' + D.hi[hIdx].toFixed(1) + '%'];
    var bw = 168, bh = 60, bx = sx + 16, by = sy - bh - 14;
    if (bx + bw > W - 6) bx = sx - bw - 16;
    if (by < 4) by = sy + 16;
    rr(bx, by, bw, bh, 10); ctx.fillStyle = 'rgba(11,14,20,0.9)'; ctx.fill();
    ctx.strokeStyle = 'rgba(255,194,75,0.6)'; ctx.lineWidth = 1; ctx.stroke();
    ctx.textAlign = 'left';
    ctx.font = '600 12px Poppins, sans-serif'; ctx.fillStyle = C.amber; ctx.fillText(lines[0], bx+12, by+19);
    ctx.font = '500 12px Poppins, sans-serif'; ctx.fillStyle = C.text; ctx.fillText(lines[1], bx+12, by+36);
    ctx.font = '400 11px Poppins, sans-serif'; ctx.fillStyle = C.dim; ctx.fillText(lines[2], bx+12, by+52);
    ctx.globalAlpha = 1;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
"""


def render_particle_trend(result_df, height=380):
    """Particle-field RUL trend: point RUL stream plus 90% interval cloud."""
    x, y, lo, hi = _downsample([
        result_df["Window"].values,
        result_df["Predicted RUL (%)"].values,
        result_df["Lower Bound (%)"].values,
        result_df["Upper Bound (%)"].values,
    ])
    data = {
        "x": [int(v) for v in x],
        "y": [round(float(v), 2) for v in y],
        "lo": [round(float(v), 2) for v in lo],
        "hi": [round(float(v), 2) for v in hi],
    }
    html = _fx_html(
        "",
        '<div class="wrap" id="wrap"><canvas id="cv"></canvas></div>',
        _TREND_JS,
        height,
        DATA=json.dumps(data),
        COLORS=json.dumps(_FX),
    )
    components.html(html, height=height + 4)


# ---------------- 2. Central aura vortex (fleet health) ----------------
_AURA_JS = r"""
(function(){
var A = __PARAMS__, C = __COLORS__, H = __H__;
var wrap = document.getElementById('wrap'), cv = document.getElementById('cv'), ctx = cv.getContext('2d');
var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
var W = 400, dpr = Math.min(window.devicePixelRatio || 1, 2);
var BG = '13,17,25';
function resize(){ W = wrap.clientWidth || 400; cv.width = W*dpr; cv.height = H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); ctx.fillStyle = 'rgb('+BG+')'; ctx.fillRect(0,0,W,H); }
resize();
if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);

document.getElementById('val').textContent = A.value;
document.getElementById('lab').textContent = A.label;
var st = document.getElementById('stat'); st.textContent = A.status; st.style.color = A.statusColor;
document.getElementById('sub').textContent = A.sub;

function mix(a,b,m){ return [a[0]+(b[0]-a[0])*m, a[1]+(b[1]-a[1])*m, a[2]+(b[2]-a[2])*m]; }
var GOLD = [255,194,75], TEAL = [25,227,200], ORANGE = [255,138,76], RED = [255,77,109];
// s = 0 -> calm coherent orbit, s = 1 -> scattered / flared
var s = Math.pow(Math.max(0, Math.min(1, 1 - A.health)), 0.85);

var N = 1100, P = [];
for (var i=0; i<N; i++){
  P.push({r: 0.2 + 0.8*Math.pow(Math.random(), 0.75), a: Math.random()*6.283, arm: i % 2, sd: Math.random(), sd2: Math.random(), sz: 0.9 + Math.random()*1.6});
}

function frame(ts){
  var t = ts/1000;
  var R = Math.min(W, H)/2 - 12, cx = W/2, cy = H/2;
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = 'rgba('+BG+','+(reduce ? 1 : 0.2)+')';
  ctx.fillRect(0, 0, W, H);

  ctx.globalCompositeOperation = 'lighter';
  var pulse = 1 + (reduce ? 0 : 0.06*Math.sin(t*1.57));
  var gc = mix(GOLD, RED, s);
  var g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R*0.8*pulse);
  g.addColorStop(0, 'rgba('+(gc[0]|0)+','+(gc[1]|0)+','+(gc[2]|0)+',0.16)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

  for (var i=0; i<P.length; i++){
    var p = P[i];
    var w = 0.55 + 0.9*(1 - p.r);
    var ang = p.a + (reduce ? 0 : t*w) + p.r*3.2 + p.arm*3.1416;
    var rr = p.r*R, ej = 0;
    if (p.sd < s*0.55){
      var cyc = (t*(0.07 + 0.12*p.sd2) + p.sd2) % 1;
      rr += R*0.8*s*cyc; ej = cyc*s;
    }
    if (!reduce){
      ang += s*0.5*Math.sin(t*1.1 + p.sd*20);
      rr += s*R*0.12*Math.sin(t*1.7 + p.sd2*30);
    }
    var x = cx + Math.cos(ang)*rr, y = cy + Math.sin(ang)*rr*0.82;
    var base = p.arm ? TEAL : GOLD;
    var target = p.sd2 > 0.5 ? ORANGE : RED;
    var m = ej > 0 ? Math.min(1, s*1.3) : s*(0.35 + 0.65*p.sd2);
    var col = mix(base, target, m);
    var alpha = (0.35 + 0.55*(1 - p.r))*(1 - ej*0.8);
    ctx.fillStyle = 'rgba('+(col[0]|0)+','+(col[1]|0)+','+(col[2]|0)+','+alpha.toFixed(3)+')';
    ctx.fillRect(x, y, p.sz, p.sz);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
"""

_AURA_CSS = r"""
.wrap{background:#0D1119}
.center{position:absolute;left:0;top:0;right:0;bottom:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none;text-align:center}
.val{font-size:34px;font-weight:700;color:#E6EDF3;line-height:1.05;text-shadow:0 0 18px rgba(255,194,75,0.35)}
.lab{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#7F8C9B;margin-top:2px}
.stat{font-size:12px;font-weight:600;margin-top:8px}
.sub{font-size:10px;color:#7F8C9B;margin-top:3px;max-width:150px}
"""


def render_aura(health, counts, height=300):
    """Swirling dual-tone particle vortex that reflects fleet health."""
    if health >= 0.7:
        status, status_color = "Coherent orbit", _FX["teal"]
    elif health >= 0.4:
        status, status_color = "Drifting orbit", _FX["amber"]
    else:
        status, status_color = "Turbulence detected", _FX["red"]

    params = {
        "health": float(health),
        "value": f"{health * 100:.0f}%",
        "label": "Fleet health index",
        "status": status,
        "statusColor": status_color,
        "sub": (
            f"Low {counts.get('Low', 0)} / Medium {counts.get('Medium', 0)} "
            f"/ High {counts.get('High', 0)} windows"
        ),
    }
    body = (
        '<div class="wrap" id="wrap"><canvas id="cv"></canvas>'
        '<div class="center"><div class="val" id="val"></div>'
        '<div class="lab" id="lab"></div><div class="stat" id="stat"></div>'
        '<div class="sub" id="sub"></div></div></div>'
    )
    html = _fx_html(
        _AURA_CSS,
        body,
        _AURA_JS,
        height,
        PARAMS=json.dumps(params),
        COLORS=json.dumps(_FX),
    )
    components.html(html, height=height + 4)


# ---------------- 3. Constellation map (SHAP drivers) ----------------
_CONSTELLATION_JS = r"""
(function(){
var NODES = __NODES__, C = __COLORS__, H = __H__;
var wrap = document.getElementById('wrap'), cv = document.getElementById('cv'), ctx = cv.getContext('2d');
var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
var W = 600, dpr = Math.min(window.devicePixelRatio || 1, 2);
function resize(){ W = wrap.clientWidth || 600; cv.width = W*dpr; cv.height = H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); }
resize();
if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
if (!NODES.length) return;

var FAMC = {Time: C.cyan, Frequency: C.violet, Wavelet: C.amber};
function famColor(f){ return FAMC[f] || C.cyan; }
function rgba(hex, a){ var h = hex.replace('#',''); return 'rgba('+parseInt(h.substr(0,2),16)+','+parseInt(h.substr(2,2),16)+','+parseInt(h.substr(4,2),16)+','+a+')'; }

var maxImp = Math.max.apply(null, NODES.map(function(d){ return d.imp; })) || 1;
var totalImp = NODES.reduce(function(a,d){ return a + d.imp; }, 0) || 1;
var fams = [];
NODES.forEach(function(d){ if (fams.indexOf(d.fam) < 0) fams.push(d.fam); });
var byFam = {};
fams.forEach(function(f){ byFam[f] = NODES.filter(function(d){ return d.fam === f; }).sort(function(a,b){ return b.imp - a.imp; }); });

var items = [], leader = {};
fams.forEach(function(f, fi){
  byFam[f].forEach(function(d, i){
    var it = {d: d, fam: f, fi: fi, i: i, ang0: i*2.39996 + fi*1.3, rad0: i === 0 ? 0 : 28 + 24*Math.sqrt(i), rN: 3.5 + 13*Math.sqrt(d.imp/maxImp), dir: (fi % 2 ? -1 : 1), dust: []};
    if (i === 0) leader[f] = items.length;
    items.push(it);
  });
});
items.forEach(function(it){
  var cnt = Math.round(6 + 34*it.d.imp/maxImp);
  for (var k=0; k<cnt; k++) it.dust.push({orb: it.rN*(1.5 + Math.random()*2.4), a: Math.random()*6.283, sp: (0.25 + Math.random()*0.9)*(Math.random() < 0.5 ? -1 : 1), sz: 0.6 + Math.random()*1.1});
});
var ranked = items.slice().sort(function(a,b){ return b.d.imp - a.d.imp; });
var topSet = ranked.slice(0, 6);

var stars = [];
for (var s=0; s<110; s++) stars.push({x: Math.random(), y: Math.random(), r: 0.4 + Math.random()*0.9, ph: Math.random()*6.283});

var mx = -1, my = -1, inside = false;
cv.addEventListener('mousemove', function(e){ var r = cv.getBoundingClientRect(); mx = e.clientX - r.left; my = e.clientY - r.top; inside = true; });
cv.addEventListener('mouseleave', function(){ inside = false; });

function center(fi){
  var m = fams.length;
  return {x: W*(fi+1)/(m+1), y: H/2 - 6 + (m > 1 ? (fi % 2 ? 1 : -1)*H*0.05 : 0)};
}
function place(it, t){
  var c = center(it.fi), m = fams.length;
  var sc = Math.min(1, (W/(m+1)*0.5)/105);
  var a = it.ang0 + (reduce ? 0 : t*0.06*it.dir);
  return {x: c.x + Math.cos(a)*it.rad0*sc, y: c.y + Math.sin(a)*it.rad0*sc*0.9};
}
function rr(x,y,w,h,r){ ctx.beginPath(); ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath(); }

function frame(ts){
  var t = ts/1000;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  ctx.clearRect(0, 0, W, H);

  for (var si=0; si<stars.length; si++){
    var st = stars[si];
    ctx.globalAlpha = 0.12 + 0.3*(0.5 + 0.5*Math.sin(t*1.1 + st.ph));
    ctx.fillStyle = '#FFFFFF'; ctx.fillRect(st.x*W, st.y*H, st.r, st.r);
  }
  ctx.globalAlpha = 1;

  var pos = items.map(function(it){ return place(it, t); });

  ctx.globalCompositeOperation = 'lighter';
  fams.forEach(function(f, fi){
    var sum = byFam[f].reduce(function(a,d){ return a + d.imp; }, 0);
    var c = center(fi), rad = 60 + 110*Math.sqrt(sum/totalImp);
    var g = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, rad);
    g.addColorStop(0, rgba(famColor(f), 0.10)); g.addColorStop(1, rgba(famColor(f), 0));
    ctx.fillStyle = g; ctx.fillRect(c.x-rad, c.y-rad, rad*2, rad*2);
  });

  ctx.lineWidth = 1;
  items.forEach(function(it, idx){
    if (it.i === 0) return;
    var l = leader[it.fam];
    ctx.strokeStyle = rgba(famColor(it.fam), 0.16);
    ctx.beginPath(); ctx.moveTo(pos[idx].x, pos[idx].y); ctx.lineTo(pos[l].x, pos[l].y); ctx.stroke();
    ctx.strokeStyle = rgba(famColor(it.fam), 0.08);
    ctx.beginPath(); ctx.moveTo(pos[idx].x, pos[idx].y); ctx.lineTo(pos[idx-1].x, pos[idx-1].y); ctx.stroke();
  });

  var hov = -1, best = 1e9;
  if (inside){
    items.forEach(function(it, idx){
      var dx = pos[idx].x - mx, dy = pos[idx].y - my, d = Math.sqrt(dx*dx + dy*dy);
      if (d < it.rN + 14 && d < best){ best = d; hov = idx; }
    });
  }

  items.forEach(function(it, idx){
    var p = pos[idx], col = famColor(it.fam), rel = it.d.imp/maxImp;
    var boost = idx === hov ? 0.3 : 0;
    var gr = it.rN*3.6;
    var g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, gr);
    g.addColorStop(0, rgba(col, Math.min(1, 0.18 + 0.5*rel + boost))); g.addColorStop(1, rgba(col, 0));
    ctx.fillStyle = g; ctx.fillRect(p.x-gr, p.y-gr, gr*2, gr*2);
    for (var k=0; k<it.dust.length; k++){
      var dd = it.dust[k], a = dd.a + (reduce ? 0 : t*dd.sp);
      ctx.globalAlpha = 0.35 + 0.5*rel;
      ctx.fillStyle = col;
      ctx.fillRect(p.x + Math.cos(a)*dd.orb, p.y + Math.sin(a)*dd.orb*0.9, dd.sz, dd.sz);
    }
    ctx.globalAlpha = 1;
  });

  ctx.globalCompositeOperation = 'source-over';
  items.forEach(function(it, idx){
    var p = pos[idx], col = famColor(it.fam), rel = it.d.imp/maxImp;
    ctx.fillStyle = rgba(col, 0.55 + 0.4*rel);
    ctx.beginPath(); ctx.arc(p.x, p.y, it.rN, 0, 6.2832); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath(); ctx.arc(p.x, p.y, it.rN*0.38, 0, 6.2832); ctx.fill();
  });

  ctx.font = '10.5px Poppins, sans-serif'; ctx.textBaseline = 'alphabetic';
  items.forEach(function(it, idx){
    if (topSet.indexOf(it) < 0 && idx !== hov) return;
    var p = pos[idx], name = it.d.name.length > 26 ? it.d.name.slice(0, 25) + '...' : it.d.name;
    var right = p.x < W - 170;
    ctx.textAlign = right ? 'left' : 'right';
    ctx.fillStyle = idx === hov ? '#FFFFFF' : C.dim;
    ctx.fillText(name, p.x + (right ? 1 : -1)*(it.rN + 7), p.y + 4);
  });

  ctx.font = '600 11px Poppins, sans-serif'; ctx.textAlign = 'center';
  fams.forEach(function(f, fi){
    var c = center(fi);
    ctx.fillStyle = rgba(famColor(f), 0.85);
    ctx.fillText(f + ' domain', c.x, H - 12);
  });

  if (hov >= 0){
    var it2 = items[hov], p2 = pos[hov];
    var lines = [it2.d.name, 'importance ' + it2.d.imp.toFixed(3), it2.fam + ' domain'];
    ctx.font = '600 12px Poppins, sans-serif';
    var bw = Math.max(150, ctx.measureText(lines[0]).width + 24), bh = 60;
    var bx = p2.x + it2.rN + 14, by = p2.y - bh/2;
    if (bx + bw > W - 6) bx = p2.x - it2.rN - 14 - bw;
    by = Math.max(6, Math.min(H - bh - 6, by));
    rr(bx, by, bw, bh, 10); ctx.fillStyle = 'rgba(11,14,20,0.92)'; ctx.fill();
    ctx.strokeStyle = rgba(famColor(it2.fam), 0.7); ctx.lineWidth = 1; ctx.stroke();
    ctx.textAlign = 'left';
    ctx.fillStyle = famColor(it2.fam); ctx.fillText(lines[0], bx+12, by+19);
    ctx.font = '500 12px Poppins, sans-serif'; ctx.fillStyle = C.text; ctx.fillText(lines[1], bx+12, by+36);
    ctx.font = '400 11px Poppins, sans-serif'; ctx.fillStyle = C.dim; ctx.fillText(lines[2], bx+12, by+52);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
"""


def _feature_family(name):
    lowered = str(name).lower()
    if "wavelet" in lowered:
        return "Wavelet"
    if "band_energy" in lowered or "spectral" in lowered:
        return "Frequency"
    return "Time"


def render_constellation(feature_df, value_col="importance", height=340):
    """
    Constellation map of the top vibration features.
    Node size and brightness follow feature importance; features of the
    same domain (time / frequency / wavelet) form a cluster.
    """
    if feature_df is None or feature_df.empty or value_col not in feature_df.columns:
        return
    top = feature_df.head(10)
    nodes = [
        {
            "name": str(row["feature"]),
            "imp": abs(float(row[value_col])),
            "fam": _feature_family(row["feature"]),
        }
        for _, row in top.iterrows()
    ]
    html = _fx_html(
        "",
        '<div class="wrap" id="wrap"><canvas id="cv"></canvas></div>',
        _CONSTELLATION_JS,
        height,
        NODES=json.dumps(nodes),
        COLORS=json.dumps(_FX),
    )
    components.html(html, height=height + 4)


# ---------------- 4. Holographic metric rings ----------------
_RINGS_CSS = r"""
body{overflow:visible}
.grid{display:grid;gap:14px;padding:2px;grid-template-columns:repeat(__N__,1fr)}
.card{--c:#19E3F2;background:__SURF__;border:1px solid __BORDER__;border-radius:18px;padding:14px 8px 12px;text-align:center;box-sizing:border-box;animation:hum 5s ease-in-out infinite}
@keyframes hum{0%,100%{box-shadow:0 0 14px color-mix(in srgb,var(--c) 8%,transparent)}50%{box-shadow:0 0 30px color-mix(in srgb,var(--c) 24%,transparent)}}
svg{width:100%;max-width:140px;height:auto;display:block;margin:0 auto}
.outer{transform-origin:60px 60px;animation:spin 28s linear infinite}
.inner{transform-origin:60px 60px;animation:spin 20s linear infinite reverse}
@keyframes spin{to{transform:rotate(360deg)}}
.arc{animation:draw 1.5s cubic-bezier(.2,.7,.2,1) forwards}
@keyframes draw{from{stroke-dashoffset:276.46}to{stroke-dashoffset:var(--to)}}
.val{font-weight:700;fill:__TEXT__;font-family:Poppins,sans-serif}
.lab{font-size:12px;font-weight:500;margin-top:6px;color:__TEXT__}
.sub{font-size:10.5px;margin-top:2px;color:__DIM__}
@media (prefers-reduced-motion:reduce){.card,.outer,.inner{animation:none}.arc{animation:none;stroke-dashoffset:var(--to)}}
"""

_RINGS_JS = r"""
(function(){
var ITEMS = __ITEMS__;
var C2 = 2*Math.PI*44;
var root = document.getElementById('root');
root.innerHTML = ITEMS.map(function(it, i){
  var f = Math.max(0, Math.min(1, isFinite(it.frac) ? it.frac : 0));
  var len = String(it.value).length;
  var fs = len > 8 ? 11 : (len > 5 ? 14 : 22);
  return '<div class="card" style="--c:' + it.color + ';animation-delay:' + (i*0.5) + 's">' +
    '<svg viewBox="0 0 120 120">' +
    '<defs><radialGradient id="g' + i + '" cx="50%" cy="42%" r="60%"><stop offset="0%" stop-color="' + it.color + '" stop-opacity="0.22"/><stop offset="100%" stop-color="' + it.color + '" stop-opacity="0"/></radialGradient>' +
    '<filter id="f' + i + '" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>' +
    '<circle cx="60" cy="60" r="43" fill="url(#g' + i + ')"/>' +
    '<circle class="outer" cx="60" cy="60" r="55" fill="none" stroke="' + it.color + '" stroke-opacity="0.45" stroke-width="1.2" stroke-dasharray="2 7"/>' +
    '<circle class="inner" cx="60" cy="60" r="33" fill="none" stroke="' + it.color + '" stroke-opacity="0.35" stroke-width="1" stroke-dasharray="1 8"/>' +
    '<circle cx="60" cy="60" r="44" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="7"/>' +
    '<circle class="arc" cx="60" cy="60" r="44" fill="none" stroke="' + it.color + '" stroke-width="7" stroke-linecap="round" stroke-dasharray="' + C2 + '" style="--to:' + (C2*(1-f)) + ';stroke-dashoffset:' + C2 + '" transform="rotate(-90 60 60)" filter="url(#f' + i + ')"/>' +
    '<path d="M 24 46 A 40 40 0 0 1 46 24" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="2" stroke-linecap="round"/>' +
    '<text class="val" x="60" y="' + (60 + fs*0.35) + '" text-anchor="middle" font-size="' + fs + '">' + it.value + '</text>' +
    '</svg>' +
    '<div class="lab">' + it.label + '</div>' +
    '<div class="sub">' + (it.sub || '') + '</div></div>';
}).join('');
})();
"""


def render_metric_rings(items, height=220):
    """Glass KPI rings with a rotating outer ring and a glowing progress arc."""
    clean = []
    for it in items:
        frac = it.get("frac", 0.0)
        try:
            frac = float(frac)
        except Exception:
            frac = 0.0
        if not np.isfinite(frac):
            frac = 0.0
        clean.append({
            "label": str(it.get("label", "")),
            "value": str(it.get("value", "")),
            "frac": max(0.0, min(1.0, frac)),
            "color": it.get("color", _FX["cyan"]),
            "sub": str(it.get("sub", "")),
        })
    html = _fx_html(
        _RINGS_CSS.replace("__N__", str(max(1, len(clean)))),
        '<div class="grid" id="root"></div>',
        _RINGS_JS,
        height,
        ITEMS=json.dumps(clean),
    )
    components.html(html, height=height + 4)


# ============================================================
# USER UPLOAD & SIGNAL FEATURE EXTRACTION (FEMTO-ST 25.6 kHz)
# ============================================================
FS_ACC = 25600
WINDOW_SAMPLES = 2560
WINDOW_SECONDS = WINDOW_SAMPLES / FS_ACC
UPLOAD_MAX_MB = 200

def _find_first_existing(filenames):
    for fname in filenames:
        path = _find_file(fname)
        if path:
            return path
    return None

def _read_uploaded_csv(uploaded_file):
    raw = uploaded_file.getvalue()
    sample = raw[:4096].decode("utf-8", errors="ignore")
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    sep = ";" if ";" in first_line else ","

    attempts = []
    for kwargs in [
        {"header": None, "sep": sep},
        {"header": 0, "sep": sep},
        {"header": None, "sep": None, "engine": "python"},
        {"header": 0, "sep": None, "engine": "python"},
    ]:
        try:
            attempts.append(pd.read_csv(io.BytesIO(raw), **kwargs))
        except Exception:
            pass

    if not attempts:
        raise ValueError("The uploaded file could not be read as CSV.")

    # Check named columns
    for df in attempts:
        cols = {str(c).strip().lower(): c for c in df.columns}
        horiz = [c for k, c in cols.items() if any(t in k for t in ["horiz", "horizontal", "x_axis", "acc_x"])]
        vert = [c for k, c in cols.items() if any(t in k for t in ["vert", "vertical", "y_axis", "acc_y"])]
        if horiz and vert:
            out = pd.DataFrame({
                "horiz": pd.to_numeric(df[horiz[0]], errors="coerce"),
                "vert": pd.to_numeric(df[vert[0]], errors="coerce"),
            }).dropna()
            if len(out): return out

    # FEMTO-ST 6-column format
    for df in attempts:
        if df.shape[1] >= 6:
            out = pd.DataFrame({
                "horiz": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
                "vert": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
            }).dropna()
            if len(out): return out

    # Exactly 2 columns
    for df in attempts:
        if df.shape[1] == 2:
            out = pd.DataFrame({
                "horiz": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
                "vert": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
            }).dropna()
            if len(out): return out

    raise ValueError("Unsupported format. Use FEMTO-ST format or a 2-column CSV (Horizontal, Vertical vibration).")

def _time_domain_features_upload(sig, prefix=""):
    sig = np.asarray(sig, dtype=np.float64)
    rms = np.sqrt(np.mean(sig ** 2))
    peak = np.max(np.abs(sig))
    mean_abs = np.mean(np.abs(sig))
    return {
        f"{prefix}mean": np.mean(sig),
        f"{prefix}std": np.std(sig),
        f"{prefix}rms": rms,
        f"{prefix}variance": np.var(sig),
        f"{prefix}skewness": stats.skew(sig),
        f"{prefix}kurtosis": stats.kurtosis(sig),
        f"{prefix}peak": peak,
        f"{prefix}peak_to_peak": np.max(sig) - np.min(sig),
        f"{prefix}crest_factor": peak / (rms + 1e-12),
        f"{prefix}shape_factor": rms / (mean_abs + 1e-12),
        f"{prefix}impulse_factor": peak / (mean_abs + 1e-12),
        f"{prefix}clearance_factor": (peak / (np.mean(np.sqrt(np.abs(sig))) ** 2 + 1e-12)),
    }

def _freq_domain_features_upload(sig, fs=FS_ACC, prefix=""):
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    freqs = fftfreq(n, 1 / fs)[: n // 2]
    mags = np.abs(fft(sig))[: n // 2]
    mags_norm = mags / (np.sum(mags) + 1e-12)
    mean_freq = np.sum(freqs * mags_norm)

    feats = {
        f"{prefix}spectral_mean_freq": mean_freq,
        f"{prefix}spectral_rms_freq": np.sqrt(np.sum((freqs ** 2) * mags_norm)),
        f"{prefix}spectral_std_freq": np.sqrt(np.sum(((freqs - mean_freq) ** 2) * mags_norm)),
        f"{prefix}spectral_energy": np.sum(mags ** 2),
        f"{prefix}spectral_entropy": -np.sum(mags_norm * np.log(mags_norm + 1e-12)),
    }
    band_edges = np.linspace(0, fs / 2, 6)
    for i in range(5):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        feats[f"{prefix}band_energy_{i+1}"] = np.sum(mags[mask] ** 2)
    return feats

def _wavelet_features_upload(sig, wavelet="db4", level=4, prefix=""):
    sig = np.asarray(sig, dtype=np.float64).copy()
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    total_energy = sum(np.sum(c ** 2) for c in coeffs) + 1e-12
    feats = {}
    for i, c in enumerate(coeffs):
        name = "approx" if i == 0 else f"detail_{level - i + 1}"
        energy = np.sum(c ** 2)
        feats[f"{prefix}wavelet_{name}_energy_ratio"] = energy / total_energy
        feats[f"{prefix}wavelet_{name}_std"] = np.std(c)
    return feats

def _extract_upload_features(window_df):
    row = {}
    row.update(_time_domain_features_upload(window_df["horiz"].values, "horiz_"))
    row.update(_time_domain_features_upload(window_df["vert"].values, "vert_"))
    row.update(_freq_domain_features_upload(window_df["horiz"].values, prefix="horiz_"))
    row.update(_freq_domain_features_upload(window_df["vert"].values, prefix="vert_"))
    row.update(_wavelet_features_upload(window_df["horiz"].values, prefix="horiz_"))
    row.update(_wavelet_features_upload(window_df["vert"].values, prefix="vert_"))
    return row

def _make_upload_windows(signal_df):
    n_complete = len(signal_df) // WINDOW_SAMPLES
    if n_complete < 1:
        raise ValueError(
            f"The file contains {len(signal_df):,} samples. At least {WINDOW_SAMPLES:,} "
            f"samples (0.1 s at 25.6 kHz) are required for a complete feature window."
        )
    usable = n_complete * WINDOW_SAMPLES
    trimmed = signal_df.iloc[:usable].reset_index(drop=True)
    rows = []
    for i in range(n_complete):
        start = i * WINDOW_SAMPLES
        rows.append(_extract_upload_features(trimmed.iloc[start:start + WINDOW_SAMPLES]))
    return pd.DataFrame(rows), n_complete, len(signal_df) - usable

@st.cache_resource(show_spinner=False)
def _load_upload_model():
    model_path = _find_first_existing(["best_model.pkl"])
    features_path = _find_first_existing(["selected_feature_names.json"])
    config_path = _find_first_existing(["deployment_config.json"])

    if not model_path: return None, None, None, "Frozen artifact `best_model.pkl` not found."
    if not features_path: return None, None, None, "Frozen artifact `selected_feature_names.json` not found."
    try:
        model = joblib.load(model_path)
        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        params = None
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                params = json.load(f)
        return model, params, feature_names, None
    except Exception as exc:
        return None, None, None, f"Could not load deployment model: {exc}"

@st.cache_resource(show_spinner=False)
def _load_upload_cqr_artifacts():
    q_lo_path = _find_first_existing(["q_lo_model.pkl"])
    q_hi_path = _find_first_existing(["q_hi_model.pkl"])
    config_path = _find_first_existing(["conformal_config.json"])
    features_path = _find_first_existing(["selected_feature_names.json"])

    missing = []
    if not q_lo_path: missing.append("q_lo_model.pkl")
    if not q_hi_path: missing.append("q_hi_model.pkl")
    if not config_path: missing.append("conformal_config.json")
    if not features_path: missing.append("selected_feature_names.json")

    if missing:
        return None, None, None, None, "Required CQR artifacts missing: " + ", ".join(missing)

    try:
        q_lo = joblib.load(q_lo_path)
        q_hi = joblib.load(q_hi_path)
        with open(config_path, "r", encoding="utf-8") as f:
            conformal_config = json.load(f)
        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        q_hat = float(conformal_config["q_hat"])
        return q_lo, q_hi, q_hat, feature_names, None
    except Exception as exc:
        return None, None, None, None, f"Could not load CQR artifacts: {exc}"

def _upload_action(point_rul, lower_bound):
    return maintenance_action(point_rul, lower_bound)


def _prepare_upload_prediction(features_df):
    model, params, selected_features, model_error = _load_upload_model()
    if model is None or selected_features is None:
        return None, model_error

    X = features_df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    point = np.clip(np.asarray(model.predict(X), dtype=float), 0, 100)

    q_lo, q_hi, q_hat, cqr_features, cqr_error = _load_upload_cqr_artifacts()
    if q_lo is None:
        return None, f"Frozen 90% CQR artifacts missing: {cqr_error}"

    X_cqr = features_df[cqr_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_lower = np.asarray(q_lo.predict(X_cqr), dtype=float)
    raw_upper = np.asarray(q_hi.predict(X_cqr), dtype=float)
    lower = np.clip(raw_lower - q_hat, 0, 100)
    upper = np.clip(raw_upper + q_hat, 0, 100)

    result = pd.DataFrame({
        "Window": np.arange(1, len(features_df) + 1),
        "Predicted RUL (%)": point,
        "Lower Bound (%)": lower,
        "Upper Bound (%)": upper,
        "Interval Width (%)": upper - lower,
    })
    result["Recommended Action"] = [
        _upload_action(p, l) for p, l in zip(result["Predicted RUL (%)"], result["Lower Bound (%)"])
    ]
    return result, None

def _compute_upload_shap(features_df, result_df):
    model, _, selected_features, model_error = _load_upload_model()
    if model is None or selected_features is None:
        return None, model_error or "Frozen model unavailable."
    try:
        X = features_df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(X.tail(1)))
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        values = shap_values[-1]

        shap_df = pd.DataFrame({
            "feature": selected_features,
            "shap_value": values,
            "abs_shap_value": np.abs(values),
            "feature_value": X.tail(1).iloc[0].values,
        }).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)
        return shap_df, None
    except Exception as exc:
        return None, f"SHAP calculation failed: {exc}"


# ============================================================
# OLLAMA GROUNDED EXPLANATION
# ============================================================
def generate_ollama_explanation(prediction_row, feature_df):
    top_features = feature_df.head(5).copy()
    feature_lines = []
    for _, r in top_features.iterrows():
        feat = str(r.get("feature", "Unknown"))
        val = r.get("shap_value", r.get("importance", 0.0))
        try:
            feature_lines.append(f"- {feat}: impact = {float(val):.4f}")
        except Exception:
            feature_lines.append(f"- {feat}: impact = {val}")

    feature_text = "\n".join(feature_lines) if feature_lines else "- No SHAP features available"
    bearing = prediction_row.get("bearing", "Asset")
    snapshot = prediction_row.get("snapshot_idx", "Latest")
    predicted_rul = float(prediction_row["predicted_RUL_pct"])
    lower_bound = float(prediction_row["lower_bound_pct"])
    upper_bound = float(prediction_row["upper_bound_pct"])
    action = prediction_row.get("recommended_action", "Continue monitoring")
    risk_level, _, _ = uncertainty_risk_level(predicted_rul, lower_bound)
    risk_reason = risk_explanation(predicted_rul, lower_bound)

    system_prompt = """
You are an expert AI maintenance engineer explaining bearing diagnostics.
Explain ONLY the provided inputs in concise, professional language suitable for plant technicians.
Do not invent unprovided numbers, sensors, dates, failure causes, or new predictions.
The supplied risk level and maintenance action are authoritative.
Do not call the asset healthy when the supplied risk level is Medium, High, or Critical.
SHAP values indicate model influence, not a confirmed physical fault cause.
"""
    user_prompt = f"""
ASSET DIAGNOSTIC SUMMARY:
Asset: {bearing} (Window/Snapshot: {snapshot})
Predicted Remaining Useful Life (RUL): {predicted_rul:.1f}%
Calibrated 90% Prediction Interval: [{lower_bound:.1f}%, {upper_bound:.1f}%]
Risk Level: {risk_level}
Risk Rationale: {risk_reason}
Prescribed Maintenance Action: {action}
Top Vibration Drivers (SHAP):
{feature_text}

Provide a concise breakdown:
1. Operational Condition Summary
2. RUL & Prediction Range Interpretation
3. Maintenance Action Rationale
"""
    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.message.content


# ============================================================
# GAUGE VISUALIZATION
# ============================================================
def rul_gauge(rul_pct, lower, upper):
    zone_color, zone_label = zone_for_rul(rul_pct)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul_pct,
            number={"suffix": "%", "font": {"family": "Poppins", "size": 42, "color": COLOR["text"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLOR["text_dim"]},
                "bar": {"color": zone_color, "thickness": 0.28},
                "bgcolor": COLOR["surface_alt"],
                "steps": [
                    {"range": [0, 20], "color": "rgba(214, 69, 69, 0.22)"},
                    {"range": [20, 40], "color": "rgba(242, 140, 75, 0.22)"},
                    {"range": [40, 100], "color": "rgba(67, 160, 122, 0.20)"},
                ],
                "threshold": {"line": {"color": COLOR["text"], "width": 2}, "thickness": 0.75, "value": rul_pct},
            },
        )
    )
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": COLOR["text"]}
    )
    return fig, zone_color, zone_label


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown('<div class="eyebrow">SYSTEM // ONLINE</div>', unsafe_allow_html=True)
st.sidebar.markdown("## Bearing RUL Console")
st.sidebar.caption("PRONOSTIA / FEMTO-ST Diagnostic Suite")
st.sidebar.markdown("---")

# Theme switch: off = green template (Light), on = particle terminal (Dark).
st.sidebar.toggle("Dark particle mode", key="ui_dark_mode")

NAV_ITEMS = [
    "Upload Bearing Data",
    "Dashboard",
    "Bearing Health",
    "Model Performance",
    "Maintenance Recommendations",
    "Prediction Confidence",
    "Why This Prediction?",
    "AI Maintenance Assistant",
]

# Icons shown beside each menu label (display only; page names are unchanged).
NAV_ICONS = {
    "Upload Bearing Data": ":material/upload_file:",
    "Dashboard": ":material/dashboard:",
    "Bearing Health": ":material/monitor_heart:",
    "Model Performance": ":material/insights:",
    "Maintenance Recommendations": ":material/build:",
    "Prediction Confidence": ":material/target:",
    "Why This Prediction?": ":material/manage_search:",
    "AI Maintenance Assistant": ":material/smart_toy:",
}

page = st.sidebar.radio(
    "Console Navigation",
    NAV_ITEMS,
    format_func=lambda name: f"{NAV_ICONS.get(name, '•')}  {name}",
    label_visibility="collapsed",
)



if st.sidebar.button("Refresh Data & Cache", icon=":material/refresh:"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# PAGE: UPLOAD BEARING DATA (AUTOMATIC INFERENCE WORKFLOW)
# ============================================================
if page == "Upload Bearing Data":
    st.markdown(
        '<div class="eyebrow">DIAGNOSTIC WORKFLOW // AUTOMATIC INFERENCE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Upload & Inspect Bearing Data")
    st.caption(
        "Upload vibration data and let the trained pipeline automatically "
        "extract features, predict RUL, estimate uncertainty, and classify "
        "the bearing as Low, Medium, or High Risk."
    )

    # ------------------------------------------------------------------
    # STEP 1: UPLOAD DATA
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="step-card">
            <span class="step-number">1</span>
            <span class="step-text">
                Upload Accelerometer Sensor Data (25.6 kHz / FEMTO-ST Format)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose a vibration CSV file",
        type=["csv"],
        help=(
            "FEMTO-ST format (hour, min, sec, usec, horizontal, vertical) "
            "or a CSV with Horizontal and Vertical vibration columns."
        ),
        key="bearing_upload_widget",
    )

    # --------------------------------------------------------------
    # SESSION-PERSISTENT UPLOAD
    # Streamlit reruns the script whenever the sidebar navigation changes.
    # Keep the uploaded CSV bytes + filename in session_state so the
    # faculty demo can move between pages without uploading the file again.
    # --------------------------------------------------------------
    if uploaded is not None:
        try:
            _new_upload_bytes = uploaded.getvalue()
            _new_upload_name = uploaded.name

            # Only replace the saved upload when a genuinely new file is selected.
            _saved_name = st.session_state.get("bearing_upload_name")
            _saved_bytes = st.session_state.get("bearing_upload_bytes")
            if _saved_name != _new_upload_name or _saved_bytes != _new_upload_bytes:
                st.session_state["bearing_upload_name"] = _new_upload_name
                st.session_state["bearing_upload_bytes"] = _new_upload_bytes
                # Clear upload-specific generated text when the bearing changes.
                st.session_state.pop("upload_ai_text", None)
                st.session_state.pop("upload_ai_file", None)
        except Exception as exc:
            st.error(f"Unable to store the uploaded CSV for this session: {exc}")
            st.stop()

    _saved_upload_bytes = st.session_state.get("bearing_upload_bytes")
    _saved_upload_name = st.session_state.get("bearing_upload_name")

    if _saved_upload_bytes is None or _saved_upload_name is None:
        st.info(
            "Upload a CSV file to begin. Risk classification is generated "
            "only from the uploaded vibration data and the trained model."
        )
        st.stop()

    # Recreate a file-like object from the session copy. This means the
    # original uploader does not have to be populated again after navigation.
    _session_file = io.BytesIO(_saved_upload_bytes)
    _session_file.name = _saved_upload_name
    file_label = _saved_upload_name
    size_mb = len(_saved_upload_bytes) / (1024 * 1024)

    if size_mb > UPLOAD_MAX_MB:
        st.error(
            f"File size ({size_mb:.1f} MB) exceeds the maximum limit "
            f"of {UPLOAD_MAX_MB} MB."
        )
        st.stop()

    try:
        signal_df = _read_uploaded_csv(_session_file)
    except Exception as exc:
        st.error(f"Unable to parse CSV: {exc}")
        st.stop()

    if uploaded is None:
        st.success(
            f"Using the saved upload from this session: **{file_label}**. "
            "You do not need to upload it again while navigating the app."
        )

    # ------------------------------------------------------------------
    # STEP 2: VALIDATE DATA
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="step-card">
            <span class="step-number">2</span>
            <span class="step-text">
                Data Validation & Sampling Integrity
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Uploaded File", file_label)
    v2.metric("Valid Samples", f"{len(signal_df):,}")
    v3.metric("Window Size", f"{WINDOW_SAMPLES:,} samples")
    v4.metric("Sampling Frequency", f"{FS_ACC / 1000:.1f} kHz")

    with st.expander("View Raw Signal Table Preview", icon=":material/search:"):
        st.dataframe(
            signal_df.head(15),
            width="stretch",
            hide_index=True,
        )

    # ------------------------------------------------------------------
    # STEP 3: VIBRATION PREVIEW
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="step-card">
            <span class="step-number">3</span>
            <span class="step-text">
                Vibration Signal Preview — Horizontal & Vertical Axes
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_n = min(len(signal_df), 5000)

    fig_signal = go.Figure()

    fig_signal.add_trace(
        go.Scatter(
            y=signal_df["horiz"].iloc[:preview_n],
            mode="lines",
            name="Horizontal Vibration (G)",
            line={"color": COLOR["accent"], "width": 1.2},
        )
    )

    fig_signal.add_trace(
        go.Scatter(
            y=signal_df["vert"].iloc[:preview_n],
            mode="lines",
            name="Vertical Vibration (G)",
            line={"color": COLOR["caution"], "width": 1.2},
        )
    )

    fig_signal.update_layout(
        xaxis_title="Time Sample Index",
        yaxis_title="Acceleration (G)",
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR["text"]},
        legend={"orientation": "h", "y": 1.05, "x": 0},
        margin=dict(l=10, r=10, t=10, b=10),
    )

    apply_plotly_readability(fig_signal)
    st.plotly_chart(fig_signal, width="stretch")

    st.caption(
        "The uploaded horizontal and vertical vibration channels are "
        "processed using the same 25.6 kHz / 0.1-second window convention "
        "used by the deployment pipeline."
    )

    # ------------------------------------------------------------------
    # STEP 4: FEATURE EXTRACTION
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="step-card">
            <span class="step-number">4</span>
            <span class="step-text">
                Automatic Feature Extraction — Time, Frequency & Wavelet
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        features_df, n_windows, remainder = _make_upload_windows(signal_df)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Analysis Windows", f"{n_windows}")
    f2.metric("Window Duration", f"{WINDOW_SECONDS:.2f} s")
    f3.metric("Extracted Features", f"{features_df.shape[1]}")
    f4.metric("Unused Samples", f"{remainder:,}")

    with st.expander(
        "View Extracted Engineering Features",
        icon=":material/insights:",
    ):
        st.markdown(
            """
            - **Time Domain:** RMS, variance, kurtosis, crest factor,
              shape factor, peak-to-peak and related statistics.
            - **Frequency Domain:** FFT spectral energy, entropy and
              frequency sub-band energy.
            - **Wavelet Domain:** db4 level-4 energy ratios and coefficient
              statistics.
            """
        )
        st.dataframe(
            features_df.head(5),
            width="stretch",
            hide_index=True,
        )

    # ------------------------------------------------------------------
    # STEP 5: AUTOMATIC MODEL INFERENCE
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="step-card">
            <span class="step-number">5</span>
            <span class="step-text">
                Automatic RUL Prediction & 90% CQR Uncertainty Estimation
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "No risk level is selected by the user. The uploaded vibration "
        "features are passed directly to the frozen CatBoost RUL model and "
        "the calibrated XGBoost CQR models."
    )

    with st.spinner(
        "Executing trained CatBoost inference and calibrated XGBoost CQR..."
    ):
        result_df, err = _prepare_upload_prediction(features_df)

    if result_df is None:
        st.error(
            f"Live inference could not be completed: {err}"
        )
        st.markdown(
            """
            **Required deployment artifacts**
            - `best_model.pkl`
            - `selected_feature_names.json`
            - `q_lo_model.pkl`
            - `q_hi_model.pkl`
            - `conformal_config.json`
            """
        )
        st.stop()

    # Use the latest completed analysis window as the current assessment.
    last = result_df.iloc[-1]

    pt = float(last["Predicted RUL (%)"])
    lo = float(last["Lower Bound (%)"])
    hi = float(last["Upper Bound (%)"])

    risk_level, risk_color, risk_css = uncertainty_risk_level(pt, lo)
    act = maintenance_action(pt, lo)
    act_col = action_color(act)

    # ------------------------------------------------------------------
    # STEP 6: AUTOMATIC RISK CLASSIFICATION
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="step-card">
            <span class="step-number">6</span>
            <span class="step-text">
                Automatic Risk Classification — Low / Medium / High
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel panel-accent">
            <div class="eyebrow">AUTOMATIC DECISION LOGIC</div>
            <div style="font-size:0.92rem; line-height:1.75;">
                <b>Low Risk:</b> Predicted RUL ≥ 40% and 90% lower bound ≥ 20%
                &nbsp; | &nbsp;
                <b>Medium Risk:</b> Predicted RUL ≥ 40% and 90% lower bound &lt; 20%
                &nbsp; | &nbsp;
                <b>High Risk:</b> Predicted RUL &lt; 40%
            </div>
            <div class="kpi-help" style="margin-top:0.55rem;">
                The risk level is calculated automatically from the model
                outputs. The user does not select or enter the risk category.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if DARK_MODE:
        render_metric_rings(
            [
                {
                    "label": "Predicted RUL",
                    "value": f"{pt:.1f}%",
                    "frac": pt / 100,
                    "color": risk_color,
                    "sub": "point estimate",
                },
                {
                    "label": "90% CQR Range",
                    "value": f"{lo:.0f}-{hi:.0f}%",
                    "frac": (hi - lo) / 100,
                    "color": COLOR["violet"],
                    "sub": f"interval width {hi - lo:.1f}%",
                },
                {
                    "label": "Automatic Risk",
                    "value": risk_level.upper(),
                    "frac": {"Low": 1.0, "Medium": 0.6, "High": 0.3}.get(risk_level, 0.0),
                    "color": risk_color,
                    "sub": act,
                },
            ],
            height=230,
        )
    else:
        r1, r2, r3, r4 = st.columns(4)

        r1.metric("Predicted RUL", f"{pt:.1f}%")
        r2.metric("90% CQR Range", f"{lo:.1f}% — {hi:.1f}%")
        r3.metric("Automatic Risk", risk_level.upper())
        r4.metric("Recommended Action", act)

    st.markdown(
        f"""
        <div class="result-hero"
             style="border-top:4px solid {risk_color}; margin-top:1rem;">
            <div class="result-hero-label">
                Automatic Bearing Risk Assessment
            </div>
            <div class="result-hero-status" style="color:{risk_color};">
                {risk_level.upper()} RISK
            </div>
            <div class="result-hero-label" style="margin-top:10px;">
                Predicted Remaining Useful Life
            </div>
            <div class="result-hero-value" style="color:{risk_color};">
                {pt:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="action-card"
             style="--zone-color:{act_col}; margin-top:1rem;">
            <div class="eyebrow">SYSTEM-GENERATED RECOMMENDATION</div>
            <div style="
                font-family:'Poppins';
                font-size:1.45rem;
                color:{act_col};
                font-weight:700;
            ">
                {act.upper()}
            </div>
            <div class="kpi-help"
                 style="font-size:0.92rem; margin-top:0.4rem;">
                {risk_explanation(pt, lo)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # RUL TREND ACROSS ALL UPLOADED WINDOWS
    # ------------------------------------------------------------------
    if len(result_df) > 1:
        st.markdown("### RUL Trend Across Uploaded Data")

        if DARK_MODE:
            # Particle-field version of the trend (replaces the vector chart).
            render_particle_trend(result_df)
        else:
            fig_trend = go.Figure()

            fig_trend.add_trace(
                go.Scatter(
                    x=result_df["Window"],
                    y=result_df["Upper Bound (%)"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig_trend.add_trace(
                go.Scatter(
                    x=result_df["Window"],
                    y=result_df["Lower Bound (%)"],
                    mode="lines",
                    fill="tonexty",
                    name="90% Prediction Range",
                    line={"color": THEME["band"]},
                )
            )

            fig_trend.add_trace(
                go.Scatter(
                    x=result_df["Window"],
                    y=result_df["Predicted RUL (%)"],
                    mode="lines+markers",
                    name="Point RUL (%)",
                    line={"color": COLOR["accent"], "width": 2},
                )
            )

            fig_trend.add_hline(
                y=40,
                line_dash="dash",
                line_color=COLOR["caution"],
                annotation_text="40% High-Risk Threshold",
            )

            fig_trend.add_hline(
                y=20,
                line_dash="dash",
                line_color=COLOR["critical"],
                annotation_text="20% Severe-Action Threshold",
            )

            fig_trend.update_layout(
                xaxis_title="Analysis Window (0.1s increments)",
                yaxis_title="Remaining Useful Life (%)",
                yaxis_range=[0, 100],
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLOR["text"]},
            )

            apply_plotly_readability(fig_trend)
            st.plotly_chart(fig_trend, width="stretch")

    # ------------------------------------------------------------------
    # EXPLAINABILITY
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Why Did the Model Produce This Result?")

    shap_df, shap_err = _compute_upload_shap(
        features_df,
        result_df,
    )

    if shap_df is not None:
        st.caption(
            "Top vibration features influencing the latest prediction "
            "according to TreeSHAP:"
        )
        if DARK_MODE:
            render_constellation(shap_df, "abs_shap_value", height=320)
        st.dataframe(
            shap_df.head(5)[
                ["feature", "feature_value", "shap_value"]
            ],
            width="stretch",
            hide_index=True,
        )
    elif shap_err:
        st.caption(f"SHAP explanation unavailable: {shap_err}")

    # ------------------------------------------------------------------
    # OPTIONAL GROUNDED AI EXPLANATION
    # ------------------------------------------------------------------
    if st.button(
        "Generate AI Maintenance Explanation",
        icon=":material/smart_toy:",
        key="btn_upload_ollama",
    ):
        try:
            with st.spinner(
                "Generating grounded maintenance explanation..."
            ):
                row_data = {
                    "bearing": file_label,
                    "snapshot_idx": int(last["Window"]),
                    "predicted_RUL_pct": pt,
                    "lower_bound_pct": lo,
                    "upper_bound_pct": hi,
                    "recommended_action": act,
                }

                ai_resp = generate_ollama_explanation(
                    row_data,
                    shap_df
                    if shap_df is not None
                    else pd.DataFrame(),
                )

                st.session_state["upload_ai_text"] = ai_resp

        except Exception as exc:
            st.error(f"Local Ollama error: {exc}")

    if "upload_ai_text" in st.session_state:
        with st.expander(
            "AI Maintenance Explanation",
            expanded=True,
            icon=":material/smart_toy:",
        ):
            st.markdown(st.session_state["upload_ai_text"])


# ============================================================

# ============================================================
elif page == "Dashboard":
    st.markdown('<div class="eyebrow">OVERVIEW // EXECUTIVE FLEET DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Bearing Health &amp; Fleet Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">High-level condition monitoring for maintenance planners</div>', unsafe_allow_html=True)

    pred_df, pred_live = load_predictions()
    model_df, model_live = load_model_comparison()

    has_pred = pred_df is not None and not pred_df.empty and "predicted_RUL_pct" in pred_df.columns
    if has_pred:
        n_monitored = pred_df["bearing"].nunique() if "bearing" in pred_df.columns else len(pred_df)
        avg_rul = pred_df["predicted_RUL_pct"].mean()
        attention_mask = pred_df["predicted_RUL_pct"] < 40
        n_attention = int(attention_mask.sum())
        lowest_row = pred_df.sort_values("predicted_RUL_pct", ascending=True).iloc[0]
        lowest_bearing = lowest_row.get("bearing", "—")
        lowest_rul = float(lowest_row["predicted_RUL_pct"])
    else:
        n_monitored, avg_rul, n_attention, lowest_bearing, lowest_rul = 0, float("nan"), 0, "—", float("nan")

    summary_html = f"""
        <div class="exec-summary">
            Fleet Status: <b>{n_monitored}</b> bearing units represented in the prediction table. 
            <b>{n_attention}</b> prediction windows require proactive attention. 
            Unit <b>{lowest_bearing}</b> has the lowest remaining life at <b>{lowest_rul:.1f}%</b>.
        </div>
        """

    fleet_health, fleet_counts = fleet_health_index(pred_df) if has_pred else (None, {})

    if DARK_MODE and fleet_health is not None:
        col_aura, col_main = st.columns([1, 2])
        with col_aura:
            render_aura(fleet_health, fleet_counts, height=300)
        with col_main:
            st.markdown(summary_html, unsafe_allow_html=True)
            total_windows = max(len(pred_df), 1)
            low_color, _ = zone_for_rul(lowest_rul)
            render_metric_rings(
                [
                    {
                        "label": "Bearings Tracked",
                        "value": f"{n_monitored}",
                        "frac": 1.0,
                        "color": COLOR["accent"],
                        "sub": "units monitored",
                    },
                    {
                        "label": "Needs Attention",
                        "value": f"{n_attention}",
                        "frac": n_attention / total_windows,
                        "color": COLOR["caution"] if n_attention / total_windows < 0.5 else COLOR["critical"],
                        "sub": "windows with RUL < 40%",
                    },
                    {
                        "label": "Fleet Average RUL",
                        "value": f"{avg_rul:.1f}%",
                        "frac": avg_rul / 100,
                        "color": COLOR["healthy"],
                        "sub": "mean predicted RUL",
                    },
                    {
                        "label": "Lowest Asset Life",
                        "value": f"{lowest_rul:.1f}%",
                        "frac": lowest_rul / 100,
                        "color": low_color,
                        "sub": f"unit {lowest_bearing}",
                    },
                ],
                height=215,
            )
    else:
        st.markdown(summary_html, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Bearings Tracked", f"{n_monitored}")
        with k2: st.metric("Prediction Windows Requiring Attention", f"{n_attention}")
        with k3: st.metric("Fleet Average RUL", f"{avg_rul:.1f}%" if not np.isnan(avg_rul) else "—")
        with k4: st.metric("Lowest Asset Life", f"{lowest_rul:.1f}%" if not np.isnan(lowest_rul) else "—")

    st.markdown("---")
    st.markdown("### Low Remaining Life Priority Queue")
    st.caption("Bearings with predicted RUL below 40%, sorted by urgency:")

    if has_pred:
        action_col = get_action_col(pred_df)
        low_df = pred_df[pred_df["predicted_RUL_pct"] < 40].sort_values("predicted_RUL_pct").reset_index(drop=True)
        if low_df.empty:
            st.success("All monitored bearings are currently above 40% RUL.")
        else:
            low_df["Priority"] = range(1, len(low_df) + 1)
            display_table = low_df[["Priority", "bearing", "predicted_RUL_pct", "lower_bound_pct", "upper_bound_pct", action_col]].rename(
                columns={
                    "bearing": "Bearing Unit",
                    "predicted_RUL_pct": "Point RUL (%)",
                    "lower_bound_pct": "90% Lower Bound (%)",
                    "upper_bound_pct": "90% Upper Bound (%)",
                    action_col: "Recommended Action"
                }
            )
            st.dataframe(display_table, width="stretch", hide_index=True)
    data_badge(pred_live, "predictions")


# ============================================================
# PAGE: BEARING HEALTH
# ============================================================
elif page == "Bearing Health":
    st.markdown('<div class="eyebrow">MODULE // BEARING HEALTH ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown("## Individual Bearing Health & RUL")

    pred_df, pred_live = load_predictions()
    if pred_df is None or pred_df.empty:
        st.warning("No prediction data available.")
        st.stop()

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        bearing = st.selectbox("Select Bearing Unit:", sorted(pred_df["bearing"].unique()))
    bearing_df = pred_df[pred_df["bearing"] == bearing].reset_index(drop=True)

    with b_col2:
        idx_col = "snapshot_idx" if "snapshot_idx" in bearing_df.columns else bearing_df.index
        snapshot = st.select_slider("Select Inspection Snapshot:", options=sorted(bearing_df[idx_col].unique()))

    row = bearing_df[bearing_df[idx_col] == snapshot].iloc[0]
    rul = float(row["predicted_RUL_pct"])
    lower = float(row["lower_bound_pct"])
    upper = float(row["upper_bound_pct"])
    action = row.get(get_action_col(bearing_df), "Continue monitoring")
    act_col = action_color(action)
    risk, color, _ = uncertainty_risk_level(rul, lower)

    st.markdown(
        f"""
        <div class="result-hero" style="border-top:4px solid {color};">
            <div class="result-hero-label">Bearing Health Status</div>
            <div class="result-hero-status" style="color:{color};">{risk.upper()} RISK</div>
            <div class="result-hero-label" style="margin-top:10px;">Predicted Remaining Useful Life</div>
            <div class="result-hero-value" style="color:{color};">{rul:.1f}%</div>
        </div>
        """, unsafe_allow_html=True
    )

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            f"""
            <div class="panel">
                <div class="eyebrow">90% Prediction Range (CQR)</div>
                <div style="font-family:'Poppins'; font-size:1.4rem; font-weight:600;">{lower:.1f}% — {upper:.1f}%</div>
                <div class="kpi-help">The calibrated 90% confidence range based on vibration features.</div>
            </div>
            """, unsafe_allow_html=True
        )
    with r2:
        st.markdown(
            f"""
            <div class="action-card" style="--zone-color:{act_col};">
                <div class="eyebrow">Prescribed Action</div>
                <div style="font-family:'Poppins'; font-size:1.3rem; color:{act_col}; font-weight:700;">
                    {action.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    with st.expander("View Gauge & Details", icon=":material/speed:"):
        fig_g, _, _ = rul_gauge(rul, lower, upper)
        apply_plotly_readability(fig_g)
        st.plotly_chart(fig_g, width="stretch")

    # Dark theme: particle-field trend of this bearing across its snapshots.
    if DARK_MODE and "snapshot_idx" in bearing_df.columns and len(bearing_df) > 1:
        st.markdown("### RUL Trend Across Snapshots")
        _trend = bearing_df.sort_values("snapshot_idx")
        render_particle_trend(
            pd.DataFrame({
                "Window": _trend["snapshot_idx"].values,
                "Predicted RUL (%)": _trend["predicted_RUL_pct"].values,
                "Lower Bound (%)": _trend["lower_bound_pct"].values,
                "Upper Bound (%)": _trend["upper_bound_pct"].values,
            })
        )


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.markdown('<div class="eyebrow">MODULE // BENCHMARKING</div>', unsafe_allow_html=True)
    st.markdown("## Model Performance Comparison")
    st.caption("Comparison of candidate regression models evaluated on the unseen FEMTO-ST test bearings.")

    model_df, model_live = load_model_comparison()
    data_badge(model_live, "model performance")
    st.dataframe(model_df, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_mae = px.bar(
            model_df,
            x="model",
            y="MAE",
            title="Mean Absolute Error (Lower is Better)",
            color="model",
        )
        fig_mae.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLOR["text"]},
        )
        apply_plotly_readability(fig_mae)
        st.plotly_chart(fig_mae, width="stretch")
    with c2:
        fig_r2 = px.bar(
            model_df,
            x="model",
            y="R2",
            title="R² Score (Higher is Better)",
            color="model",
        )
        fig_r2.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLOR["text"]},
        )
        apply_plotly_readability(fig_r2)
        st.plotly_chart(fig_r2, width="stretch")

# ============================================================
# PAGE: MAINTENANCE RECOMMENDATIONS
# ============================================================
elif page == "Maintenance Recommendations":
    st.markdown('<div class="eyebrow">MODULE // DECISION LAYER</div>', unsafe_allow_html=True)
    st.markdown("## Maintenance Policy & Actions")

    pred_df, pred_live = load_predictions()
    if pred_df is not None:
        action_col = get_action_col(pred_df)
        counts = pred_df[action_col].value_counts()

        n_cont = int(counts.get("Continue monitoring", 0))
        n_insp = int(counts.get("Inspect before replacement", 0))
        n_sched = int(counts.get("Schedule maintenance", 0))
        n_repl = int(counts.get("Replace immediately", 0))

        if DARK_MODE:
            total_actions = max(n_cont + n_insp + n_sched + n_repl, 1)
            render_metric_rings(
                [
                    {"label": "Continue Monitoring", "value": f"{n_cont}", "frac": n_cont / total_actions, "color": COLOR["healthy"], "sub": "windows"},
                    {"label": "Inspect Before Replacement", "value": f"{n_insp}", "frac": n_insp / total_actions, "color": COLOR["caution"], "sub": "windows"},
                    {"label": "Schedule Maintenance", "value": f"{n_sched}", "frac": n_sched / total_actions, "color": COLOR["warning"], "sub": "windows"},
                    {"label": "Replace Immediately", "value": f"{n_repl}", "frac": n_repl / total_actions, "color": COLOR["critical"], "sub": "windows"},
                ],
                height=215,
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Continue Monitoring", n_cont)
            c2.metric("Inspect Before Replacement", n_insp)
            c3.metric("Schedule Maintenance", n_sched)
            c4.metric("Replace Immediately", n_repl)

        fig_pie = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.55,
            title="Recommended Action Distribution",
            color=counts.index,
            color_discrete_map={
                "Continue monitoring": COLOR["healthy"],
                "Inspect before replacement": COLOR["caution"],
                "Schedule maintenance": COLOR["warning"],
                "Replace immediately": COLOR["critical"],
            },
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": COLOR["text"]},
        )
        apply_plotly_readability(fig_pie)
        st.plotly_chart(fig_pie, width="stretch")
    else:
        st.warning("No prediction data available.")

# ============================================================
# PAGE: PREDICTION CONFIDENCE
# ============================================================
elif page == "Prediction Confidence":
    st.markdown('<div class="eyebrow">MODULE // UNCERTAINTY QUANTIFICATION</div>', unsafe_allow_html=True)
    st.markdown("## Prediction Interval (90% CQR)")

    unc, unc_live = load_uncertainty_metrics()
    if DARK_MODE:
        _target = float(unc.get("target_coverage", 0.90))
        _picp = float(unc.get("PICP", 0.889))
        _mpiw = float(unc.get("MPIW", 80.74))
        render_metric_rings(
            [
                {"label": "Target Coverage", "value": f"{_target * 100:.0f}%", "frac": _target, "color": COLOR["accent"], "sub": "nominal level"},
                {"label": "Empirical Coverage (PICP)", "value": f"{_picp * 100:.1f}%", "frac": _picp, "color": COLOR["healthy"], "sub": "observed on test bearings"},
                {"label": "Mean Interval Width (MPIW)", "value": f"{_mpiw:.2f}%", "frac": _mpiw / 100, "color": COLOR["violet"], "sub": "narrower is sharper"},
            ],
            height=215,
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Target Coverage", f"{unc.get('target_coverage', 0.90)*100:.0f}%")
        c2.metric("Empirical Coverage (PICP)", f"{unc.get('PICP', 0.889)*100:.1f}%")
        c3.metric("Mean Interval Width (MPIW)", f"{unc.get('MPIW', 80.74):.2f}%")
    _target_pct = float(unc.get("target_coverage", 0.90)) * 100
    _picp_pct = float(unc.get("PICP", 0.889)) * 100
    _mpiw_pct = float(unc.get("MPIW", 80.74))
    st.info(
        f"The CQR interval has a nominal {_target_pct:.0f}% coverage target. "
        f"On the evaluated test bearings the empirical coverage (PICP) was "
        f"{_picp_pct:.1f}%, with a mean interval width (MPIW) of "
        f"{_mpiw_pct:.2f}%, reported separately."
    )

# ============================================================
# PAGE: WHY THIS PREDICTION? (SHAP)
# ============================================================
elif page == "Why This Prediction?":
    st.markdown('<div class="eyebrow">MODULE // SHAP EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.markdown("## Key Degradation Drivers")

    imp_df, imp_live = load_feature_importance()
    if DARK_MODE:
        render_constellation(imp_df, "importance", height=360)
    fig_imp = px.bar(
        imp_df.sort_values("importance"), x="importance", y="feature", orientation="h",
        title="Global Mean |SHAP| Importance", color_discrete_sequence=[COLOR["accent"]]
    )
    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": COLOR["text"]})
    apply_plotly_readability(fig_imp)
    st.plotly_chart(fig_imp, width="stretch")
# ============================================================
# PAGE: AI MAINTENANCE ASSISTANT
# ============================================================
elif page == "AI Maintenance Assistant":
    st.markdown('<div class="eyebrow">AI AGENT // LOCAL OLLAMA INTERFACE</div>', unsafe_allow_html=True)
    st.markdown("## Grounded AI Maintenance Assistant")

    pred_df, _ = load_predictions()
    imp_df, _ = load_feature_importance()

    if pred_df is not None:
        sel_bearing = st.selectbox("Select Asset for Explanation:", sorted(pred_df["bearing"].unique()))
        b_slice = pred_df[pred_df["bearing"] == sel_bearing].iloc[0]

        if st.button("Generate Plant Engineer Summary", use_container_width=True):
            try:
                with st.spinner("Generating grounded explanation..."):
                    expl = generate_ollama_explanation(b_slice, imp_df)
                    st.session_state["fleet_ai_text"] = expl
            except Exception as e:
                st.error(f"Ollama execution failed: {e}")

        if "fleet_ai_text" in st.session_state:
            with st.expander(
                "AI Maintenance Explanation",
                expanded=True,
                icon=":material/smart_toy:",
            ):
                st.markdown(st.session_state["fleet_ai_text"])

# FINAL PRESENTATION BUILD: synthetic low-risk generator removed; live risk is data-driven.
