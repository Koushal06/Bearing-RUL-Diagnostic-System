"""
Bearing RUL Diagnostic System — Streamlit Application.
Optimized for both non-technical plant operators and technical research evaluators.

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

import os
import json
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from ollama import chat

# Upload & inference dependencies
from scipy import stats
from scipy.fft import fft, fftfreq
import pywt
import joblib
import shap


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
# STREAMLIT PAGE CONFIG & STYLES
# ============================================================
st.set_page_config(
    page_title="Bearing RUL Diagnostic & Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR = {
    "bg": "#F4F5F1",
    "surface": "#FFFFFF",
    "surface_alt": "#15171A",
    "border": "#E3E5DE",
    "text": "#15171A",
    "text_dim": "#6B7177",
    "accent": "#7ED321",
    "accent_dim": "#E8F7D4",
    "info": "#2E7DD1",
    "healthy": "#5FB825",
    "caution": "#D69A2D",
    "warning": "#FF8C42",
    "critical": "#D14343",
}

# Maintenance action colours. Defined here so every page can safely use
# action_color() without relying on a later declaration.
ACTION_COLOR = {
    "Replace immediately": COLOR["critical"],
    "Inspect before replacement": COLOR["caution"],
    "Schedule maintenance": COLOR["warning"],
    "Continue monitoring": COLOR["healthy"],
}


def apply_plotly_readability(fig):
    """Keep Plotly text readable on the light dashboard background."""
    fig.update_layout(
        font={"color": COLOR["text"]},
        title={"font": {"color": COLOR["text"]}},
        legend={"font": {"color": COLOR["text"]}},
        hoverlabel={"font": {"color": COLOR["text"]}},
    )
    fig.update_xaxes(
        title_font={"color": COLOR["text"]},
        tickfont={"color": COLOR["text"]},
        linecolor=COLOR["text"],
        tickcolor=COLOR["text"],
    )
    fig.update_yaxes(
        title_font={"color": COLOR["text"]},
        tickfont={"color": COLOR["text"]},
        linecolor=COLOR["text"],
        tickcolor=COLOR["text"],
    )
    return fig

st.markdown(
    f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {{
    --bg: {COLOR['bg']};
    --surface: {COLOR['surface']};
    --surface-alt: {COLOR['surface_alt']};
    --border: {COLOR['border']};
    --text: {COLOR['text']};
    --text-dim: {COLOR['text_dim']};
    --accent: {COLOR['accent']};
    --info: {COLOR['info']};
    --healthy: {COLOR['healthy']};
    --caution: {COLOR['caution']};
    --warning: {COLOR['warning']};
    --critical: {COLOR['critical']};
}}

.stApp {{
    background: radial-gradient(circle at 12% -10%, rgba(46,125,209,0.07), transparent 45%),
                radial-gradient(circle at 100% 0%, rgba(46,125,209,0.04), transparent 35%),
                var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}}

h1, h2, h3, .panel-title {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}

.mono, .stMetric [data-testid="stMetricValue"], code {{
    font-family: 'IBM Plex Mono', monospace !important;
}}

.stButton > button {{
    background: #15171A !important;
    color: #FFFFFF !important;
    border: 1px solid #15171A !important;
    font-weight: 600 !important;
}}

.stButton > button:hover {{
    background: #7ED321 !important;
    color: #15171A !important;
    border-color: #7ED321 !important;
}}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {{
    background: #242424 !important;
    color: #F5F7FA !important;
    border-right: 1px solid #333333;
}}

/* Sidebar readability: Streamlit may inherit the app's text color from
   the active theme. Keep every navigation/control label explicitly light
   against the dark reference-style sidebar. */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label div,
section[data-testid="stSidebar"] [data-testid="stRadio"] label span {{
    color: #F5F7FA !important;
}}

section[data-testid="stSidebar"] .eyebrow {{
    color: var(--accent) !important;
}}

section[data-testid="stSidebar"] hr {{
    border-color: #3A3A3A !important;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    color: #FFFFFF !important;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div {{
    color: #FFFFFF !important;
}}

.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}}

.panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.15rem 1.3rem;
    margin-bottom: 1rem;
}}

.panel-accent {{
    border-left: 4px solid var(--accent);
}}

.badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    border: 1px solid currentColor;
}}
.badge-live {{ color: var(--healthy); }}
.badge-demo {{ color: var(--caution); }}

.action-card {{
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    border: 1px solid var(--border);
    border-top: 4px solid var(--zone-color, var(--accent));
    background: var(--surface);
}}

.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.15rem;
}}

.hero-subtitle {{
    color: var(--text-dim);
    font-size: 1.02rem;
    margin-bottom: 0.6rem;
}}

.exec-summary {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    font-size: 1.02rem;
    line-height: 1.6;
    color: var(--text);
    margin-bottom: 1.2rem;
}}

.kpi-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    height: 100%;
}}

.kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.35rem;
}}

.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}}

.kpi-help {{
    color: var(--text-dim);
    font-size: 0.8rem;
    margin-top: 0.35rem;
    line-height: 1.35;
}}

.status-pill {{
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    color: #0B0F14;
}}
.status-pill-healthy {{ background: var(--healthy); }}
.status-pill-attention {{ background: var(--caution); }}
.status-pill-warning {{ background: var(--warning); }}
.status-pill-critical {{ background: var(--critical); color: #fff; }}
.status-pill-info {{ background: var(--info); }}

.result-hero {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    text-align: center;
}}
.result-hero-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
}}
.result-hero-status {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    margin-bottom: 0.2rem;
}}
.result-hero-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 3rem;
    line-height: 1.05;
}}

.step-card {{
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.step-number {{
    background: var(--accent);
    color: #fff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}}
.step-text {{
    font-size: 0.88rem;
    font-weight: 600;
    color: #15171A !important;
}}

[data-testid="stMetric"] {{
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
}}

/* File uploader readability on the light dashboard. */
div[data-testid="stFileUploader"] > label,
div[data-testid="stFileUploader"] > label p,
div[data-testid="stFileUploader"] > label span {{
    color: #15171A !important;
}}

div[data-testid="stFileUploader"] section {{
    background: #24262D !important;
    color: #FFFFFF !important;
}}

div[data-testid="stFileUploader"] section *,
div[data-testid="stFileUploader"] section p,
div[data-testid="stFileUploader"] section span {{
    color: #FFFFFF !important;
}}

div[data-testid="stFileUploader"] button {{
    color: #15171A !important;
    background: #FFFFFF !important;
}}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {{
    color: #15171A !important;
}}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
    background: #FFFFFF !important;
    border: 1px solid #D7DCE2 !important;
}}
</style>
""",
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
            number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 42, "color": COLOR["text"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLOR["text_dim"]},
                "bar": {"color": zone_color, "thickness": 0.28},
                "bgcolor": COLOR["surface_alt"],
                "steps": [
                    {"range": [0, 20], "color": "rgba(209, 67, 67, 0.25)"},
                    {"range": [20, 40], "color": "rgba(255, 140, 66, 0.25)"},
                    {"range": [40, 100], "color": "rgba(47, 168, 112, 0.20)"},
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

page = st.sidebar.radio("Console Navigation", NAV_ITEMS)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div class="mono" style="font-size:0.75rem; line-height:1.8; color:var(--text-dim);">
<b>Pipeline Architecture:</b><br>
FEMTO-ST 25.6 kHz &rarr; 0.1s Windows<br>
&rarr; Time/FFT/Wavelet Features<br>
&rarr; CatBoost RUL Point Model<br>
&rarr; XGBoost CQR (90% Interval)<br>
&rarr; SHAP Explainability<br>
&rarr; Automatic Low / Medium / High Risk
</div>
""", unsafe_allow_html=True
)

if st.sidebar.button("🔄 Refresh Data & Cache"):
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
    )

    if uploaded is None:
        st.info(
            "Upload a CSV file to begin. Risk classification is generated "
            "only from the uploaded vibration data and the trained model."
        )
        st.stop()

    size_mb = uploaded.size / (1024 * 1024)

    if size_mb > UPLOAD_MAX_MB:
        st.error(
            f"File size ({size_mb:.1f} MB) exceeds the maximum limit "
            f"of {UPLOAD_MAX_MB} MB."
        )
        st.stop()

    try:
        signal_df = _read_uploaded_csv(uploaded)
        file_label = uploaded.name
    except Exception as exc:
        st.error(f"Unable to parse CSV: {exc}")
        st.stop()

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

    with st.expander("🔍 View Raw Signal Table Preview"):
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

    with st.expander("📊 View Extracted Engineering Features"):
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
                font-family:'Space Grotesk';
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
                line={"color": "rgba(46,125,209,0.3)"},
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
        "🤖 Generate AI Maintenance Explanation",
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
        st.markdown(
            f'<div class="panel">{st.session_state["upload_ai_text"]}</div>',
            unsafe_allow_html=True,
        )


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

    st.markdown(
        f"""
        <div class="exec-summary">
            Fleet Status: <b>{n_monitored}</b> bearing units represented in the prediction table. 
            <b>{n_attention}</b> prediction windows require proactive attention. 
            Unit <b>{lowest_bearing}</b> has the lowest remaining life at <b>{lowest_rul:.1f}%</b>.
        </div>
        """, unsafe_allow_html=True
    )

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
                <div style="font-family:'IBM Plex Mono'; font-size:1.4rem;">{lower:.1f}% — {upper:.1f}%</div>
                <div class="kpi-help">The calibrated 90% confidence range based on vibration features.</div>
            </div>
            """, unsafe_allow_html=True
        )
    with r2:
        st.markdown(
            f"""
            <div class="action-card" style="--zone-color:{act_col};">
                <div class="eyebrow">Prescribed Action</div>
                <div style="font-family:'Space Grotesk'; font-size:1.3rem; color:{act_col}; font-weight:700;">
                    {action.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    with st.expander("📊 View Gauge & Details"):
        fig_g, _, _ = rul_gauge(rul, lower, upper)
        apply_plotly_readability(fig_g)
    st.plotly_chart(fig_g, width="stretch")


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
        fig_mae = px.bar(model_df, x="model", y="MAE", title="Mean Absolute Error (Lower is Better)", color="model")
        fig_mae.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": COLOR["text"]})
        apply_plotly_readability(fig_mae)
    st.plotly_chart(fig_mae, width="stretch")
    with c2:
        fig_r2 = px.bar(model_df, x="model", y="R2", title="R² Score (Higher is Better)", color="model")
        fig_r2.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": COLOR["text"]})
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Continue Monitoring", int(counts.get("Continue monitoring", 0)))
        c2.metric("Inspect Before Replacement", int(counts.get("Inspect before replacement", 0)))
        c3.metric("Schedule Maintenance", int(counts.get("Schedule maintenance", 0)))
        c4.metric("Replace Immediately", int(counts.get("Replace immediately", 0)))

        fig_pie = px.pie(
            values=counts.values, names=counts.index, hole=0.55,
            color=counts.index,
            color_discrete_map={
                "Continue monitoring": COLOR["healthy"],
                "Inspect before replacement": COLOR["caution"],
                "Schedule maintenance": COLOR["warning"],
                "Replace immediately": COLOR["critical"],
            }
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": COLOR["text"]})
        apply_plotly_readability(fig_pie)
    st.plotly_chart(fig_pie, width="stretch")

# ============================================================
# PAGE: PREDICTION CONFIDENCE
# ============================================================
elif page == "Prediction Confidence":
    st.markdown('<div class="eyebrow">MODULE // UNCERTAINTY QUANTIFICATION</div>', unsafe_allow_html=True)
    st.markdown("## Prediction Interval (90% CQR)")

    unc, unc_live = load_uncertainty_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("Target Coverage", f"{unc.get('target_coverage', 0.90)*100:.0f}%")
    c2.metric("Empirical Coverage (PICP)", f"{unc.get('PICP', 0.889)*100:.1f}%")
    c3.metric("Mean Interval Width (MPIW)", f"{unc.get('MPIW', 80.74):.2f}%")
    st.info("The CQR interval has a nominal 90% coverage target. In the reported evaluation, empirical PICP was 88.9%, close to the target, with interval width reported separately.")

# ============================================================
# PAGE: WHY THIS PREDICTION? (SHAP)
# ============================================================
elif page == "Why This Prediction?":
    st.markdown('<div class="eyebrow">MODULE // SHAP EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.markdown("## Key Degradation Drivers")

    imp_df, imp_live = load_feature_importance()
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
            st.markdown(f'<div class="panel">{st.session_state["fleet_ai_text"]}</div>', unsafe_allow_html=True)

# FINAL PRESENTATION BUILD: synthetic low-risk generator removed; live risk is data-driven.