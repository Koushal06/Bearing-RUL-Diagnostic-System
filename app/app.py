"""
Bearing RUL Diagnostic System — Streamlit application.

Design: professional industrial analytics dashboard. Every page tries to load
real pipeline output first (from OUTPUT_DIR) and falls back to clearly-labeled
demo values if that stage of the pipeline hasn't been (re-)run yet — so the
app upgrades itself automatically as Chapters 5-7 complete on the full
FAST_DEMO=False data.

UI PRINCIPLE ("two-layer" information architecture):
  Layer 1 — Simple business view (Bearing Health, Remaining Life, Risk Level,
            Recommended Action) that a non-technical stakeholder can read in
            seconds.
  Layer 2 — Technical details (MAE, RMSE, R², PICP, MPIW, CWC, SHAP, feature
            names, model internals) tucked behind "Technical Details"
            expanders. Nothing technical is removed — it's just not forced
            on the reader first.

This file keeps the ORIGINAL data-loading, caching, and risk-aware policy
logic untouched wherever possible. Only presentation/navigation/layout was
redesigned per the UI/UX brief. Existing output files, column names, and
directory search order are fully preserved (see CONFIG below).
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from ollama import chat


# ============================================================
# CONFIG
# ============================================================
# The four project notebooks each write to their OWN folder rather than a
# single shared one:
#
#   MA_Project_FE__  (Ch.4 Features)         -> processed_features/
#   MA_Project_MD__  (Ch.5 Modeling)         -> chapter5_outputs/
#   MA_Project_MDE__ (Ch.6 Uncertainty/SHAP) -> chapter6_outputs/
#   MA_Project_EE__  (Ch.7 Evaluation)       -> chapter7_outputs/
#
# We search all of them in this order.
# Override/extend via BEARING_APP_OUTPUT_DIRS (comma-separated list),
# or point BEARING_APP_OUTPUT_DIR at a single folder.

_default_dirs = [
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


# ============================================================
# OLLAMA LOCAL AI CONFIGURATION
# ============================================================
# Default model: llama3
# You can change it later using:
# OLLAMA_MODEL=another_model_name
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


def generate_ollama_explanation(prediction_row, feature_df):
    """
    Generate a grounded maintenance explanation using the local Ollama model.

    The AI receives ONLY:
    - Existing predicted RUL
    - Prediction interval
    - Existing maintenance recommendation
    - Existing SHAP feature importance

    It is instructed not to invent new sensor values, predictions,
    failure causes, dates, or measurements.
    """

    top_features = feature_df.head(5).copy()

    feature_lines = []

    for _, feature_row in top_features.iterrows():

        feature_name = str(
            feature_row.get("feature", "Unknown feature")
        )

        importance = feature_row.get("importance", np.nan)

        try:
            feature_lines.append(
                f"- {feature_name}: mean |SHAP| = "
                f"{float(importance):.4f}"
            )

        except (TypeError, ValueError):
            feature_lines.append(
                f"- {feature_name}: mean |SHAP| = {importance}"
            )

    feature_text = (
        "\n".join(feature_lines)
        if feature_lines
        else "- No SHAP features available"
    )

    bearing = prediction_row.get("bearing", "Unknown")

    snapshot = prediction_row.get(
        "snapshot_idx",
        "Unknown"
    )

    predicted_rul = float(
        prediction_row["predicted_RUL_pct"]
    )

    lower_bound = float(
        prediction_row["lower_bound_pct"]
    )

    upper_bound = float(
        prediction_row["upper_bound_pct"]
    )

    action = prediction_row.get(
        "recommended_action",
        prediction_row.get(
            "risk_aware_action",
            "Not available"
        ),
    )

    top_driver = prediction_row.get(
        "top_driver_1_feature",
        (
            top_features.iloc[0]["feature"]
            if not top_features.empty
            else "Not available"
        ),
    )

    system_prompt = """
You are an AI assistant for an industrial bearing Remaining Useful Life
(RUL) diagnostic and maintenance decision-support system.

Your job is to explain ONLY the project outputs supplied by the user.

Rules:
- Do not invent sensor readings.
- Do not invent vibration measurements.
- Do not invent failure causes.
- Do not invent dates or maintenance schedules.
- Do not invent percentages.
- Do not create a new RUL prediction.
- Treat the maintenance action as an existing recommendation from
  the project's decision layer.
- SHAP features indicate model influence and should not automatically
  be described as direct physical causes.
- Use clear, professional language suitable for a maintenance engineer.
- Keep the explanation grounded in the supplied information.
"""

    user_prompt = f"""
PROJECT OUTPUTS

Bearing:
{bearing}

Snapshot:
{snapshot}

Predicted Remaining Useful Life:
{predicted_rul:.1f}%

Prediction interval:
Lower bound: {lower_bound:.1f}%
Upper bound: {upper_bound:.1f}%

Recommended maintenance action:
{action}

Top instance-level driving feature:
{top_driver}

Top global SHAP features:
{feature_text}

Write a concise maintenance explanation in the following sections:

1. Bearing Health Summary
2. RUL Interpretation
3. Uncertainty Interpretation
4. SHAP Feature Insights
5. Maintenance Recommendation

Use only the supplied information.
Keep the answer concise, practical, and professional.
"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.message.content


# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Bearing Health & Remaining Life Dashboard",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN TOKENS + GLOBAL STYLE
# ============================================================
COLOR = {
    "bg": "#14181F",
    "surface": "#1B212C",
    "surface_alt": "#212836",
    "border": "#2C3547",
    "text": "#E8ECF1",
    "text_dim": "#8A94A6",
    "accent": "#4FC3D9",
    "accent_dim": "#2E5A66",
    "info": "#4FC3D9",
    "healthy": "#3DDC84",
    "caution": "#F5A623",
    "critical": "#E74C3C",
}


st.markdown(
    f"""
<link rel="preconnect" href="https://fonts.googleapis.com">

<link
href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
rel="stylesheet"
>

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
    --critical: {COLOR['critical']};
}}

.stApp {{
    background:
        radial-gradient(
            circle at 15% 0%,
            rgba(79,195,217,0.05),
            transparent 40%
        ),
        var(--bg);

    color: var(--text);

    font-family: 'IBM Plex Sans', sans-serif;
}}

h1,
h2,
h3,
.panel-title {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}

/* mono for every number/data readout */

.mono,
.stMetric [data-testid="stMetricValue"],
code {{
    font-family: 'IBM Plex Mono', monospace !important;
}}

section[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
}}

/* eyebrow label */

.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}}

/* panel card */

.panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
}}

.panel-accent {{
    border-left: 3px solid var(--accent);
}}

/* status badge */

.badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.55rem;
    border-radius: 3px;
    border: 1px solid currentColor;
}}

.badge-live {{
    color: var(--healthy);
}}

.badge-demo {{
    color: var(--caution);
}}

/* status action card */

.action-card {{
    border-radius: 6px;
    padding: 1rem 1.2rem;
    border: 1px solid var(--border);
    border-top: 3px solid var(--zone-color, var(--accent));
    background: var(--surface);
}}

/* -------------------------------------------------- */
/* NEW: executive / non-technical presentation styles  */
/* -------------------------------------------------- */

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
    margin-bottom: 0.5rem;
}}

.exec-summary {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    font-size: 1.02rem;
    line-height: 1.6;
    color: var(--text);
    margin-bottom: 1.2rem;
}}

.kpi-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
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
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}}

.kpi-help {{
    color: var(--text-dim);
    font-size: 0.78rem;
    margin-top: 0.35rem;
    line-height: 1.35;
}}

/* status pill — used for Healthy / Needs Attention / Critical everywhere */

.status-pill {{
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    color: #0B0F14;
}}

.status-pill-healthy {{ background: var(--healthy); }}
.status-pill-attention {{ background: var(--caution); }}
.status-pill-critical {{ background: var(--critical); color: #fff; }}
.status-pill-info {{ background: var(--info); }}

/* big result block for the simple RUL / Bearing Health page */

.result-hero {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    text-align: center;
}}

.result-hero-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.4rem;
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
    font-size: 3.1rem;
    line-height: 1.05;
}}

.result-hero-caption {{
    color: var(--text-dim);
    font-size: 0.92rem;
    margin-top: 0.3rem;
}}

/* info tooltip glyph — hover shows plain-English definition */

.info-tip {{
    display: inline-block;
    font-size: 0.78rem;
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    border-radius: 50%;
    width: 1.05rem;
    height: 1.05rem;
    line-height: 1.0rem;
    text-align: center;
    cursor: help;
    margin-left: 0.3rem;
}}

.category-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    height: 100%;
}}

.category-card .count {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.3rem 0;
}}

.model-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    height: 100%;
}}

.model-card-best {{
    border: 1px solid var(--healthy);
}}

.best-tag {{
    display: inline-block;
    background: var(--healthy);
    color: #0B0F14;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    margin-bottom: 0.5rem;
}}

hr {{
    border-color: var(--border);
}}

[data-testid="stMetric"] {{
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.7rem 0.9rem 0.5rem 0.9rem;
}}

[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
}}

.stDataFrame {{
    border: 1px solid var(--border);
    border-radius: 6px;
}}

::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-thumb {{
    background: var(--border);
    border-radius: 4px;
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# Real file if present, labeled demo fallback otherwise
# (Unchanged from the original pipeline-integration logic.)
# ============================================================

def _find_file(filename):
    """
    Search all configured output directories for filename.
    Return first matching path.
    """

    for d in OUTPUT_DIRS:

        path = os.path.join(
            d,
            filename
        )

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


def _try_load_json(filename):

    path = _find_file(filename)

    if path:

        try:

            with open(path) as f:
                return json.load(f), True

        except Exception:
            return None, False

    return None, False


@st.cache_data
def load_model_comparison():

    df, live = _try_load_csv(
        "model_performance_final.csv"
    )

    if df is None:
        df, live = _try_load_csv(
            "model_comparison.csv"
        )

    if df is not None:
        return df, live

    demo = pd.DataFrame({
        "model": [
            "CatBoost",
            "Random Forest",
            "XGBoost",
            "LightGBM"
        ],

        "MAE": [
            18.727481,
            19.152684,
            19.362556,
            20.625481
        ],

        "RMSE": [
            22.244893,
            24.681066,
            23.724841,
            24.580809
        ],

        "R2": [
            0.406197,
            0.269014,
            0.324558,
            0.274940
        ],
    })

    return demo, False


@st.cache_data
def load_uncertainty_metrics():

    df, live = _try_load_csv(
        "uncertainty_metrics.csv"
    )

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

    df, live = _try_load_csv(
        "framework_results.csv"
    )

    if df is None:

        df, live = _try_load_csv(
            "maintenance_recommendations.csv"
        )

    if df is not None:
        return df, live

    rng = np.random.default_rng(7)

    n = 60

    bearings = rng.choice(
        [
            "Bearing1_2",
            "Bearing2_2",
            "Bearing3_2"
        ],
        n
    )

    pred = np.clip(
        rng.normal(35, 25, n),
        -5,
        100
    )

    width = np.clip(
        rng.normal(55, 20, n),
        15,
        95
    )

    lower = np.clip(
        pred - width / 2,
        0,
        100
    )

    upper = np.clip(
        pred + width / 2,
        0,
        100
    )

    def action(p, w):

        if p < 20:
            return (
                "Replace immediately"
                if w < 50
                else "Inspect before replacement"
            )

        elif p < 40:
            return "Schedule maintenance"

        return "Continue monitoring"

    demo = pd.DataFrame({

        "bearing": bearings,

        "snapshot_idx": rng.integers(
            1,
            900,
            n
        ),

        "predicted_RUL_pct": pred,

        "lower_bound_pct": lower,

        "upper_bound_pct": upper,

        "interval_width_pct": upper - lower,

        "recommended_action": [
            action(p, w)
            for p, w in zip(
                pred,
                upper - lower
            )
        ],

        "top_driver_1_feature": rng.choice(
            [
                "horiz_band_energy_2",
                "horiz_wavelet_detail_2_energy_ratio",
                "horiz_std"
            ],
            n
        ),
    })

    return demo, False


@st.cache_data
def load_feature_importance():

    df, live = _try_load_csv(
        "shap_values.csv"
    )

    if df is not None:

        shap_cols = [
            c
            for c in df.columns
            if c.startswith("shap_")
        ]

        importance = (
            df[shap_cols]
            .abs()
            .mean()
            .sort_values(
                ascending=False
            )
        )

        out = pd.DataFrame({
            "feature": [
                c.replace("shap_", "")
                for c in importance.index
            ],

            "importance": importance.values
        })

        return out.head(10), live

    demo = pd.DataFrame({

        "feature": [
            "horiz_band_energy_2",
            "horiz_band_energy_5",
            "horiz_std",
            "horiz_shape_factor",
            "horiz_variance"
        ],

        "importance": [
            5.48,
            2.91,
            2.88,
            2.01,
            1.94
        ],
    })

    return demo, False


@st.cache_data
def load_model_specific_predictions(model_name):
    """
    Try to load a model-specific prediction output file, e.g. for CatBoost:
    'catboost_predictions.csv', 'predictions_catboost.csv', etc.

    IMPORTANT — data integrity:
    This project's pipeline (as delivered) writes one COMBINED prediction
    file (framework_results.csv / maintenance_recommendations.csv) rather
    than one file per model. We therefore search a set of plausible
    per-model filenames; if none exist we return (None, False) and the
    calling page must show the "not available" message rather than
    fabricate numbers.
    """

    key = model_name.lower().replace(" ", "_").replace("-", "_")

    candidates = [
        f"{key}_predictions.csv",
        f"predictions_{key}.csv",
        f"{key}_output.csv",
        f"{key}_results.csv",
        f"{key}.csv",
    ]

    for fname in candidates:
        df, live = _try_load_csv(fname)
        if df is not None:
            return df, live

    return None, False


def data_badge(
    is_live,
    label="pipeline output"
):

    if is_live:

        st.markdown(
            f'<span class="badge badge-live">'
            f'● live — {label}'
            f'</span>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<span class="badge badge-demo">'
            f'○ demo — {label} not yet generated'
            f'</span>',
            unsafe_allow_html=True
        )


# ============================================================
# SHARED CLASSIFICATION / POLICY HELPERS
#
# NOTE ON POLICY: this project already has TWO layers of decisioning:
#
#   1. `recommended_action` (or `risk_aware_action`) — the project's
#      existing RISK-AWARE policy, computed from the prediction interval's
#      LOWER bound (not just the point estimate). This is preserved
#      exactly as-is; it is authoritative for "what should we do."
#
#   2. A simple point-estimate-only HEALTH STATUS (Healthy / Needs
#      Attention / Critical) used purely for quick-glance labels, KPI
#      tiles, and the health-distribution chart. Thresholds (<20 / <40 /
#      >=40) match the project's existing `zone_for_rul()` boundaries, so
#      the two layers never contradict each other — they just answer
#      different questions ("how much life is left" vs "what should we
#      actually do about it, given uncertainty").
# ============================================================

def zone_for_rul(rul_pct):

    if rul_pct < 20:
        return COLOR["critical"], "CRITICAL"

    elif rul_pct < 40:
        return COLOR["caution"], "MODERATE"

    return COLOR["healthy"], "HEALTHY"


def health_status(rul_pct):
    """
    Simple, plain-English health classification derived only from the
    predicted RUL percentage. Returns (label, risk_level, color, css_class).
    """

    if rul_pct is None or (isinstance(rul_pct, float) and np.isnan(rul_pct)):
        return "Unknown", "Unknown", COLOR["text_dim"], "status-pill-info"

    if rul_pct < 20:
        return "Critical", "High", COLOR["critical"], "status-pill-critical"

    elif rul_pct < 40:
        return "Needs Attention", "Medium", COLOR["caution"], "status-pill-attention"

    return "Healthy", "Low", COLOR["healthy"], "status-pill-healthy"


ACTION_COLOR = {
    "Replace immediately": COLOR["critical"],
    "Inspect before replacement": COLOR["caution"],
    "Schedule maintenance": COLOR["caution"],
    "Continue monitoring": COLOR["healthy"],
}


def action_color(action):
    return ACTION_COLOR.get(action, COLOR["accent"])


def get_action_col(df):
    return (
        "recommended_action"
        if "recommended_action" in df.columns
        else "risk_aware_action"
    )


def status_pill_html(label, css_class):
    return (
        f'<span class="status-pill {css_class}">{label}</span>'
    )


# Plain-English glossary for technical terms (Section 10 of the brief)
GLOSSARY = {
    "MAE": "Average prediction error. Lower is better.",
    "RMSE": "Measures prediction error while giving more weight to large errors.",
    "R2": "Shows how well the model explains variation in the data. Higher is generally better.",
    "R²": "Shows how well the model explains variation in the data. Higher is generally better.",
    "PICP": "Percentage of actual values captured by the prediction interval.",
    "MPIW": "Average width of the prediction interval. Narrower intervals are generally more precise.",
    "CWC": "A combined score that penalizes both poor coverage and overly wide intervals.",
    "SHAP": "Shows which input features influenced the model's prediction.",
    "RUL": "Remaining Useful Life — the estimated percentage of useful operating life remaining.",
    "CQR": "Conformalized Quantile Regression — the method used to produce a calibrated prediction range.",
}


def info_tip(term):
    text = GLOSSARY.get(term, "")
    text_escaped = text.replace('"', "&quot;")
    return f'<span class="info-tip" title="{text_escaped}">ⓘ</span>'


def kpi_card(label, value, help_text=None):
    help_html = (
        f'<div class="kpi-help">{help_text}</div>'
        if help_text else ""
    )
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{help_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def rul_gauge(
    rul_pct,
    lower,
    upper
):

    zone_color, zone_label = zone_for_rul(
        rul_pct
    )

    fig = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=rul_pct,

            number={
                "suffix": "%",
                "font": {
                    "family": "IBM Plex Mono",
                    "size": 44,
                    "color": COLOR["text"]
                }
            },

            gauge={

                "axis": {
                    "range": [0, 100],
                    "tickcolor": COLOR["text_dim"],
                    "tickfont": {
                        "color": COLOR["text_dim"]
                    }
                },

                "bar": {
                    "color": zone_color,
                    "thickness": 0.28
                },

                "bgcolor": COLOR["surface_alt"],

                "borderwidth": 1,

                "bordercolor": COLOR["border"],

                "steps": [

                    {
                        "range": [0, 20],
                        "color": "rgba(231,76,60,0.18)"
                    },

                    {
                        "range": [20, 40],
                        "color": "rgba(245,166,35,0.18)"
                    },

                    {
                        "range": [40, 100],
                        "color": "rgba(61,220,132,0.12)"
                    },
                ],

                "threshold": {

                    "line": {
                        "color": COLOR["text"],
                        "width": 2
                    },

                    "thickness": 0.75,

                    "value": rul_pct,
                },
            },
        )
    )

    fig.update_layout(
        height=260,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=10
        ),

        paper_bgcolor="rgba(0,0,0,0)",

        font={
            "color": COLOR["text"]
        },
    )

    return (
        fig,
        zone_color,
        zone_label
    )


def build_low_life_table(pred_df, top_n=None, low_life_threshold=40):
    """
    Build a table containing ONLY low-life bearings.

    A low-life bearing is defined here as predicted_RUL_pct below
    ``low_life_threshold`` (default: 40%). Results are sorted in ascending
    order so the bearing with the LOWEST predicted remaining life appears first.
    All values come from the prediction dataframe; nothing is hard-coded.
    """

    if pred_df is None or pred_df.empty or "predicted_RUL_pct" not in pred_df.columns:
        return pd.DataFrame()

    action_col = get_action_col(pred_df)

    work = pred_df.copy()
    work["predicted_RUL_pct"] = pd.to_numeric(
        work["predicted_RUL_pct"], errors="coerce"
    )
    work = work.dropna(subset=["predicted_RUL_pct"])

    # Keep ONLY low-life bearings.
    work = work[
        work["predicted_RUL_pct"] < low_life_threshold
    ].copy()

    # Lowest remaining life first.
    work = work.sort_values(
        "predicted_RUL_pct", ascending=True
    ).reset_index(drop=True)

    if top_n is not None:
        work = work.head(top_n).reset_index(drop=True)

    rows = []
    for i, r in work.iterrows():
        rul = float(r["predicted_RUL_pct"])
        label, risk, color, css = health_status(rul)
        action = r.get(action_col, "—") if action_col in r.index else "—"
        bearing = r.get("bearing", "—")

        rows.append({
            "Priority": i + 1,
            "Bearing": bearing,
            "Remaining Life (%)": round(rul, 1),
            "Health Status": label,
            "Risk Level": risk,
            "Recommended Action": action,
        })

    return pd.DataFrame(rows)


def style_status_column(df, col="Health Status"):
    """Return a Styler that color-codes the Health Status column."""

    def _color(val):
        if val == "Critical":
            return f"color: {COLOR['critical']}; font-weight: 700;"
        elif val == "Needs Attention":
            return f"color: {COLOR['caution']}; font-weight: 700;"
        elif val == "Healthy":
            return f"color: {COLOR['healthy']}; font-weight: 700;"
        return ""

    try:
        return df.style.applymap(_color, subset=[col])
    except Exception:
        return df


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="eyebrow">SYSTEM // ONLINE</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "## ⚙ Bearing RUL"
)

st.sidebar.caption(
    "Diagnostic & Maintenance Console"
)

st.sidebar.markdown("---")


NAV_ITEMS = [
    "🏠 Dashboard",
    "🔧 Bearing Health",
    "🤖 Model Performance",
    "⚠️ Maintenance Recommendations",
    "📊 Prediction Confidence",
    "🔎 Why This Prediction?",
    "🧠 AI Maintenance Assistant",
]

page_raw = st.sidebar.radio(
    "Console",
    NAV_ITEMS,
    label_visibility="collapsed",
)

# Strip the emoji prefix for internal routing
page = page_raw.split(" ", 1)[1].strip()


st.sidebar.markdown("---")


st.sidebar.markdown(
    '<div class="eyebrow">Pipeline</div>',
    unsafe_allow_html=True
)


st.sidebar.markdown(
    """
<div class="mono"
style="
font-size:0.78rem;
line-height:1.9;
color:var(--text-dim);
">

FEMTO-ST &rarr; Features<br>
&rarr; XGBoost / CatBoost<br>
&rarr; CQR Interval<br>
&rarr; SHAP<br>
&rarr; Risk-Aware Policy

</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar.expander(
    "🔧 Technical Details — Data Source Diagnostics"
):

    _expected_files = [

        "model_performance_final.csv",

        "model_comparison.csv",

        "uncertainty_metrics.csv",

        "framework_results.csv",

        "maintenance_recommendations.csv",

        "shap_values.csv",
    ]

    for fname in _expected_files:

        hit = _find_file(fname)

        if hit:

            st.markdown(
                f"✅ `{fname}`  \n"
                f"&nbsp;&nbsp;`{hit}`"
            )

        else:

            st.markdown(
                f"❌ `{fname}` — not found"
            )

    st.caption(
        "Searched: " +
        ", ".join(
            f"`{d}`"
            for d in OUTPUT_DIRS
        )
    )

    st.markdown("---")

    if st.button(
        "🔄 Refresh data (clear cache)"
    ):

        st.cache_data.clear()

        st.rerun()

    st.caption(
        "Data is cached for the session. "
        "If you just copied new CSVs into outputs/, "
        "click Refresh."
    )


# ============================================================
# PAGE: DASHBOARD  (formerly "Overview")
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="eyebrow">'
        'BEARING DIAGNOSTIC SYSTEM // EXECUTIVE DASHBOARD'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-title">Bearing Health &amp; Remaining Life Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">Predictive maintenance decision-support system</div>',
        unsafe_allow_html=True,
    )

    # ---- Load everything up front -------------------------------------
    model_df, model_live = load_model_comparison()
    unc, unc_live = load_uncertainty_metrics()
    pred_df, pred_live = load_predictions()

    mae_col = "MAE" if "MAE" in model_df.columns else model_df.columns[1]
    best_model_row = model_df.sort_values(mae_col).iloc[0]
    best_model_name = best_model_row.get("model", best_model_row.get("Model", "—"))

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
        n_monitored = 0
        avg_rul = float("nan")
        n_attention = 0
        lowest_bearing = "—"
        lowest_rul = float("nan")

    # ---- Executive summary (auto-generated from real data) -------------
    if has_pred:
        summary_text = (
            f"<b>{n_monitored}</b> bearings are currently monitored. "
            f"<b>{n_attention}</b> require maintenance attention. "
            f"<b>{lowest_bearing}</b> has the lowest predicted remaining life "
            f"at <b>{lowest_rul:.1f}%</b>. "
            f"<b>{best_model_name}</b> provides the best overall prediction "
            f"performance with an MAE of <b>{best_model_row[mae_col]:.2f}%</b>."
        )
    else:
        summary_text = (
            "No prediction data is currently available. The figures below "
            "are demo placeholders until the pipeline output is generated."
        )

    st.markdown(
        f'<div class="exec-summary">{summary_text}</div>',
        unsafe_allow_html=True,
    )

    if not (model_live and pred_live):
        st.markdown(
            '<span class="badge badge-demo">○ DEMO — Pipeline Output Not Available</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-live">● LIVE — Pipeline Output</span>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ---- KPI cards -------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        kpi_card(
            "Bearings Monitored",
            f"{n_monitored}" if has_pred else "—",
            "Total number of bearing units currently tracked by the system.",
        )

    with k2:
        kpi_card(
            "Bearings Requiring Attention",
            f"{n_attention}" if has_pred else "—",
            "Bearings with low predicted remaining life or elevated maintenance risk.",
        )

    with k3:
        kpi_card(
            "Average Remaining Life",
            f"{avg_rul:.1f}%" if has_pred and not np.isnan(avg_rul) else "—",
            "Mean predicted Remaining Useful Life across all monitored bearings.",
        )

    with k4:
        kpi_card(
            "Selected Model",
            str(best_model_name),
            f"Best-performing model by lowest MAE ({mae_col}).",
        )

    st.markdown("")
    st.markdown("---")

    # ---- Health distribution + explanation --------------------------------
    col_chart, col_text = st.columns([1, 1])

    if has_pred:
        labels = pred_df["predicted_RUL_pct"].apply(lambda x: health_status(x)[0])
        counts = labels.value_counts().reindex(
            ["Healthy", "Needs Attention", "Critical"]
        ).fillna(0).astype(int)

        with col_chart:
            st.markdown(
                '<div class="eyebrow">Bearing Health Distribution</div>',
                unsafe_allow_html=True,
            )

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.55,
                color=counts.index,
                color_discrete_map={
                    "Healthy": COLOR["healthy"],
                    "Needs Attention": COLOR["caution"],
                    "Critical": COLOR["critical"],
                },
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": COLOR["text"]},
                legend={"orientation": "h", "y": -0.1},
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
            )

            st.plotly_chart(fig, width="stretch")

        with col_text:
            st.markdown(
                '<div class="eyebrow">What This Means</div>',
                unsafe_allow_html=True,
            )

            n_healthy = int(counts.get("Healthy", 0))
            n_needs = int(counts.get("Needs Attention", 0))
            n_crit = int(counts.get("Critical", 0))
            total = max(n_healthy + n_needs + n_crit, 1)

            if n_crit == 0 and n_needs == 0:
                plain_text = (
                    "All monitored bearings currently have sufficient "
                    "predicted life remaining. No immediate action is required."
                )
            else:
                plain_text = (
                    f"Most bearings currently have sufficient predicted life "
                    f"remaining ({n_healthy} of {total} healthy). "
                    f"A smaller group requires maintenance attention "
                    f"({n_needs} needs attention, {n_crit} critical)."
                )

            st.markdown(
                f'<div class="panel">{plain_text}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="eyebrow">Status Definitions</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'{status_pill_html("Healthy", "status-pill-healthy")} '
                f'Remaining life ≥ 40%<br><br>'
                f'{status_pill_html("Needs Attention", "status-pill-attention")} '
                f'Remaining life 20–40%<br><br>'
                f'{status_pill_html("Critical", "status-pill-critical")} '
                f'Remaining life &lt; 20%',
                unsafe_allow_html=True,
            )
    else:
        st.info("No prediction data available yet — health distribution will appear once the pipeline runs.")

    st.markdown("---")

    # ---- LOW-LIFE bearings table -------------------------------------------
    st.markdown("### ⚠️ Low Remaining Life Bearings")
    st.caption(
        "Only bearings with predicted remaining life below 40% are shown. "
        "The bearing with the LOWEST remaining life appears first."
    )

    if has_pred:
        low_life_all = build_low_life_table(
            pred_df,
            top_n=None,
            low_life_threshold=40,
        )

        if low_life_all.empty:
            st.success(
                "No bearings currently have predicted remaining life below 40%."
            )
        else:
            low_life_top10 = low_life_all.head(10)

            st.dataframe(
                style_status_column(low_life_top10),
                width="stretch",
                hide_index=True,
                column_config={
                    "Priority": st.column_config.NumberColumn(
                        "Priority",
                        format="%d",
                    ),
                    "Remaining Life (%)": st.column_config.NumberColumn(
                        "Remaining Life (%)",
                        format="%.1f%%",
                    ),
                },
            )

            if len(low_life_all) > 10:
                with st.expander(
                    f"View All {len(low_life_all)} Low-Life Bearings"
                ):
                    st.dataframe(
                        style_status_column(low_life_all),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Priority": st.column_config.NumberColumn(
                                "Priority",
                                format="%d",
                            ),
                            "Remaining Life (%)": st.column_config.NumberColumn(
                                "Remaining Life (%)",
                                format="%.1f%%",
                            ),
                        },
                    )
    else:
        st.info(
            "Prediction output not available — showing no data rather than fabricated values."
        )

    data_badge(pred_live, "predictions")


# ============================================================
# PAGE: BEARING HEALTH  (formerly "RUL Prediction")
# ============================================================

elif page == "Bearing Health":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // BEARING HEALTH'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Bearing Health &amp; Remaining Life")

    pred_df, pred_live = load_predictions()

    data_badge(
        pred_live,
        "predictions"
    )

    st.markdown("")

    if pred_df is None or pred_df.empty or "bearing" not in pred_df.columns:
        st.warning("No bearing prediction data is currently available.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:

        bearing = st.selectbox(
            "Select a bearing",
            sorted(
                pred_df["bearing"].unique()
            )
        )

    bearing_df = (
        pred_df[
            pred_df["bearing"] == bearing
        ]
        .reset_index(drop=True)
    )

    with col2:

        idx_col = (
            "snapshot_idx"
            if "snapshot_idx" in bearing_df.columns
            else bearing_df.index
        )

        snapshot = st.select_slider(
            "Select a snapshot in time",

            options=(
                sorted(
                    bearing_df[idx_col].unique()
                )
                if "snapshot_idx" in bearing_df.columns
                else list(bearing_df.index)
            )
        )

    if "snapshot_idx" in bearing_df.columns:

        row = (
            bearing_df[
                bearing_df["snapshot_idx"] == snapshot
            ]
            .iloc[0]
        )

    else:

        row = bearing_df.iloc[
            snapshot
        ]

    rul = float(row["predicted_RUL_pct"])
    lower = float(row["lower_bound_pct"])
    upper = float(row["upper_bound_pct"])
    label, risk, color, css = health_status(rul)

    action = row.get(
        "recommended_action",
        row.get("risk_aware_action", "—")
    )
    act_color = action_color(action)

    st.markdown("---")

    # ---- LAYER 1: big, simple result -------------------------------------
    st.markdown(
        f'''
        <div class="result-hero">
            <div class="result-hero-label">Bearing Health</div>
            <div class="result-hero-status" style="color:{color};">{label.upper()}</div>
            <div style="margin: 0.8rem 0;">
                <div class="result-hero-label">Remaining Useful Life</div>
                <div class="result-hero-value" style="color:{color};">{rul:.1f}%</div>
                <div class="result-hero-caption">
                    Approximately {rul:.1f}% of the estimated useful life remains.
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown("")

    r1, r2 = st.columns(2)

    with r1:
        st.markdown(
            f'''
            <div class="panel">
                <div class="eyebrow">Prediction Range</div>
                <div style="font-family:'IBM Plex Mono'; font-size:1.4rem;">
                    {lower:.1f}% — {upper:.1f}%
                </div>
                <div class="kpi-help">
                    The model's best estimate falls somewhere in this range.
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            f'''
            <div class="action-card" style="--zone-color:{act_color};">
                <div class="eyebrow">Recommended Action</div>
                <div style="font-family:'Space Grotesk'; font-size:1.3rem; color:{act_color};">
                    {str(action).upper()}
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ---- Supporting gauge (below the plain-English result) ----------------
    with st.expander("View Gauge Visualization", expanded=False):

        gcol, dcol = st.columns([1, 1])

        with gcol:
            fig, zone_color, zone_label = rul_gauge(rul, lower, upper)
            st.plotly_chart(fig, width="stretch")

        with dcol:
            m1, m2, m3 = st.columns(3)
            m1.metric("Lower Bound", f"{lower:.1f}%")
            m2.metric("Predicted", f"{rul:.1f}%")
            m3.metric("Upper Bound", f"{upper:.1f}%")

            top_feat = row.get("top_driver_1_feature", "—")
            st.markdown(
                f'<div class="panel" style="margin-top:0.6rem;">'
                f'<div class="eyebrow">Top Driving Feature</div>'
                f'<code style="color:var(--accent);">{top_feat}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---- LAYER 2: technical details -----------------------------------
    with st.expander("🔧 View Technical Details"):

        st.markdown(
            f"**Risk Level:** {risk}  \n"
            f"**Health classification thresholds:** "
            f"Critical &lt; 20% · Needs Attention 20–40% · Healthy ≥ 40%"
        )

        st.markdown("**Raw record:**")
        st.dataframe(row.to_frame().T, width="stretch", hide_index=True)


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // MODEL COMPARISON'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Model Performance")
    st.caption(
        "Which model is being used to generate predictions, and how well "
        "does it perform?"
    )

    model_df, model_live = load_model_comparison()

    data_badge(model_live, "model comparison")

    st.markdown("")

    # Normalize column names we rely on
    df = model_df.copy()

    model_col = "model" if "model" in df.columns else ("Model" if "Model" in df.columns else df.columns[0])
    mae_col = "MAE" if "MAE" in df.columns else None
    rmse_col = "RMSE" if "RMSE" in df.columns else None
    r2_col = "R2" if "R2" in df.columns else ("R²" if "R²" in df.columns else None)
    time_col = None
    for cand in ["inference_time", "Inference Time", "inference_time_ms", "latency", "Latency"]:
        if cand in df.columns:
            time_col = cand
            break

    if mae_col is None:
        st.warning("Model comparison data is missing an MAE column — cannot rank models.")
    else:
        df = df.sort_values(mae_col, ascending=True).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        best_name = df.iloc[0][model_col]

        # ---- Comparison table -------------------------------------------
        st.markdown("### Model Performance Comparison")

        display_cols = ["Rank", model_col]
        if mae_col: display_cols.append(mae_col)
        if rmse_col: display_cols.append(rmse_col)
        if r2_col: display_cols.append(r2_col)
        if time_col: display_cols.append(time_col)

        disp = df[display_cols].copy()
        disp["Overall Assessment"] = disp[model_col].apply(
            lambda m: "🏆 BEST OVERALL" if m == best_name else ""
        )

        st.dataframe(disp, width="stretch", hide_index=True)

        with st.expander("ⓘ What do these columns mean?"):
            st.markdown(
                f"- **MAE** {info_tip('MAE')} — {GLOSSARY['MAE']}\n"
                f"- **RMSE** {info_tip('RMSE')} — {GLOSSARY['RMSE']}\n"
                f"- **R²** {info_tip('R2')} — {GLOSSARY['R2']}\n"
                f"- **Inference Time** — how long the model takes to produce "
                f"one prediction; lower is better for edge deployment.",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ---- Charts --------------------------------------------------------
        c1, c2 = st.columns(2)

        with c1:
            if mae_col:
                fig = px.bar(
                    df, x=model_col, y=mae_col, color=model_col,
                    color_discrete_sequence=[COLOR["accent"], COLOR["text_dim"], COLOR["caution"], COLOR["healthy"]],
                )
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": COLOR["text"]},
                    title="MAE by Model (lower is better)",
                )
                st.plotly_chart(fig, width="stretch")

        with c2:
            if r2_col:
                fig = px.bar(
                    df, x=model_col, y=r2_col, color=model_col,
                    color_discrete_sequence=[COLOR["accent"], COLOR["text_dim"], COLOR["caution"], COLOR["healthy"]],
                )
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": COLOR["text"]},
                    title="R² by Model (higher is better)",
                )
                st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # ---- Individual model cards -----------------------------------
        st.markdown("### Individual Model Cards")

        def plain_english_assessment(row):
            """
            Generate an assessment from the ACTUAL metrics rather than a
            fixed script. Falls back to neutral language if a metric is
            missing.
            """
            is_best = row[model_col] == best_name
            bits = []

            if is_best:
                bits.append("Best overall balance of accuracy and speed among the compared models.")
            else:
                if mae_col:
                    gap = row[mae_col] - df.iloc[0][mae_col]
                    if gap > 0:
                        bits.append(f"Prediction error is {gap:.2f} points higher (MAE) than the best model.")
                if time_col:
                    try:
                        if row[time_col] > df[time_col].median():
                            bits.append("Slower at inference than most other models.")
                    except Exception:
                        pass

            if not bits:
                bits.append("Performance is comparable to the other models evaluated.")

            return " ".join(bits)

        cols = st.columns(min(len(df), 4))

        for i, (_, row) in enumerate(df.iterrows()):
            col = cols[i % len(cols)]
            is_best = row[model_col] == best_name

            with col:
                card_class = "model-card model-card-best" if is_best else "model-card"
                best_tag = '<div class="best-tag">🏆 BEST OVERALL</div>' if is_best else ""

                metrics_html = ""
                if mae_col:
                    metrics_html += f"<div><b>MAE</b>: {row[mae_col]:.2f}%</div>"
                if rmse_col:
                    metrics_html += f"<div><b>RMSE</b>: {row[rmse_col]:.2f}%</div>"
                if r2_col:
                    metrics_html += f"<div><b>R²</b>: {row[r2_col]:.3f}</div>"
                if time_col:
                    metrics_html += f"<div><b>Inference Time</b>: {row[time_col]}</div>"

                assessment = plain_english_assessment(row)

                st.markdown(
                    f'''
                    <div class="{card_class}">
                        {best_tag}
                        <div style="font-family:'Space Grotesk'; font-weight:700; font-size:1.15rem; margin-bottom:0.4rem;">
                            {row[model_col]}
                        </div>
                        <div style="font-size:0.85rem; line-height:1.7; color:var(--text-dim);">
                            {metrics_html}
                        </div>
                        <div style="margin-top:0.6rem; font-size:0.88rem; color:var(--text);">
                            {assessment}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ---- Model-specific outputs ------------------------------------
        st.markdown("### Model Prediction Outputs")
        st.caption("Inspect predictions and error metrics for one model at a time.")

        model_names = list(df[model_col].unique())
        selected_model = st.selectbox("Select a model", model_names, key="model_output_select")

        msp_df, msp_live = load_model_specific_predictions(selected_model)

        model_row = df[df[model_col] == selected_model].iloc[0]

        if msp_df is not None and not msp_df.empty:
            data_badge(msp_live, f"{selected_model} predictions")

            mm1, mm2, mm3 = st.columns(3)
            if mae_col:
                mm1.metric("MAE", f"{model_row[mae_col]:.2f}%")
            if rmse_col:
                mm2.metric("RMSE", f"{model_row[rmse_col]:.2f}%")
            if r2_col:
                mm3.metric("R²", f"{model_row[r2_col]:.3f}")

            st.metric("Number of Predictions", len(msp_df))

            if "predicted_RUL_pct" in msp_df.columns:
                fig = px.histogram(
                    msp_df, x="predicted_RUL_pct", nbins=25,
                    color_discrete_sequence=[COLOR["accent"]],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": COLOR["text"]},
                    title=f"{selected_model} — Predicted RUL Distribution",
                )
                st.plotly_chart(fig, width="stretch")

            st.dataframe(msp_df.head(50), width="stretch", hide_index=True)

        else:
            st.markdown(
                f'''
                <div class="panel">
                    <b>{selected_model}</b><br>
                    Model-specific prediction output is not available in the
                    current pipeline. Only the aggregate metrics (MAE, RMSE, R²)
                    from the model comparison file are shown below — individual
                    predictions were not saved separately for this model.
                </div>
                ''',
                unsafe_allow_html=True,
            )

            mm1, mm2, mm3 = st.columns(3)
            if mae_col:
                mm1.metric("MAE", f"{model_row[mae_col]:.2f}%")
            if rmse_col:
                mm2.metric("RMSE", f"{model_row[rmse_col]:.2f}%")
            if r2_col:
                mm3.metric("R²", f"{model_row[r2_col]:.3f}")


# ============================================================
# PAGE: MAINTENANCE RECOMMENDATIONS  (formerly "Maintenance Policy")
# ============================================================

elif page == "Maintenance Recommendations":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // RISK-AWARE POLICY'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Maintenance Recommendations")

    st.caption(
        "What should be done about each bearing, based on its predicted "
        "remaining life and the uncertainty around that prediction."
    )

    pred_df, pred_live = load_predictions()

    data_badge(pred_live, "recommendations")

    st.markdown("")

    if pred_df is None or pred_df.empty:
        st.warning("No maintenance recommendation data is currently available.")
        st.stop()

    action_col = get_action_col(pred_df)

    counts = pred_df[action_col].value_counts()

    zone_meta = {
        "Continue monitoring": (
            COLOR["healthy"], "HEALTHY", "Continue Monitoring",
            "Sufficient RUL remains. Continue normal monitoring.",
        ),
        "Schedule maintenance": (
            COLOR["caution"], "ATTENTION", "Schedule Maintenance",
            "Moderate risk. Plan maintenance in the near term.",
        ),
        "Inspect before replacement": (
            COLOR["caution"], "INSPECTION", "Inspect Before Replacement",
            "Point estimate is optimistic but the interval signals real "
            "downside risk. Inspect before deciding.",
        ),
        "Replace immediately": (
            COLOR["critical"], "CRITICAL", "Replace Immediately",
            "Point estimate and worst-case downside agree: critical.",
        ),
    }

    cols = st.columns(len(zone_meta))

    for col, (action, (color, tag, title, desc)) in zip(cols, zone_meta.items()):

        n = int(counts.get(action, 0))

        with col:
            st.markdown(
                f'''
                <div class="category-card" style="border-top:3px solid {color};">
                    <div class="eyebrow" style="color:{color};">{tag}</div>
                    <div style="font-family:'Space Grotesk'; font-weight:700; font-size:1.05rem;">
                        {title}
                    </div>
                    <div class="count" style="color:{color};">{n}</div>
                    <div style="font-size:0.82rem; color:var(--text-dim);">{desc}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    st.markdown("")

    with st.expander("Why these recommendations?", expanded=True):
        st.markdown(
            "These recommendations consider both the predicted remaining "
            "life **and** the uncertainty around the prediction. A bearing "
            "whose point estimate looks fine but whose prediction interval "
            "reaches dangerously low values is still flagged for inspection "
            "— we don't wait for the average case to go wrong."
        )

    st.markdown("---")

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.55,
        color=counts.index,
        color_discrete_map={
            "Continue monitoring": COLOR["healthy"],
            "Schedule maintenance": COLOR["caution"],
            "Inspect before replacement": "#FF8C42",
            "Replace immediately": COLOR["critical"],
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR["text"]},
        legend={"orientation": "h", "y": -0.1},
        title="Distribution of Recommended Actions",
    )

    st.plotly_chart(fig, width="stretch")

    with st.expander("🔧 View Technical Details"):
        st.markdown(
            "The risk-aware policy decides the action from the prediction "
            "interval's **lower bound**, not the point estimate alone — a "
            "wide interval on an optimistic prediction can still trigger "
            "an inspection recommendation."
        )
        st.dataframe(pred_df, width="stretch", hide_index=True)


# ============================================================
# PAGE: PREDICTION CONFIDENCE  (formerly "Uncertainty")
# ============================================================

elif page == "Prediction Confidence":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // CQR CALIBRATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("## How Confident Is the Prediction?")

    st.caption(
        "Every prediction comes with a range, not just a single number. "
        "This page shows how trustworthy that range actually is."
    )

    unc, unc_live = load_uncertainty_metrics()

    data_badge(unc_live, "uncertainty metrics")

    st.markdown("")

    target_cov = unc.get("target_coverage", 0.9)
    actual_cov = unc["PICP"]
    gap = abs(actual_cov - target_cov)
    gap_color = COLOR["healthy"] if gap <= 0.03 else COLOR["caution"]

    c1, c2 = st.columns(2)

    with c1:
        kpi_card(
            "Prediction Coverage (Target)",
            f"{target_cov * 100:.0f}%",
            "The percentage of true outcomes the prediction range is designed to capture.",
        )

    with c2:
        kpi_card(
            "Actual Coverage",
            f"{actual_cov * 100:.1f}%",
            "The percentage of true outcomes the prediction range actually captured, measured on test data.",
        )

    st.markdown("")

    st.markdown(
        f'''
        <div class="panel" style="border-left:3px solid {gap_color};">
            Empirical coverage sits <b style="color:{gap_color};">{gap * 100:.1f} points</b>
            from the {target_cov * 100:.0f}% target —
            {"within" if gap <= 0.03 else "outside"} tolerance.
        </div>
        ''',
        unsafe_allow_html=True,
    )

    pred_df, pred_live = load_predictions()

    has_bounds = (
        pred_df is not None
        and {"predicted_RUL_pct", "lower_bound_pct", "upper_bound_pct"}.issubset(pred_df.columns)
    )

    if has_bounds:
        avg_lower = pred_df["lower_bound_pct"].mean()
        avg_upper = pred_df["upper_bound_pct"].mean()

        st.markdown(
            f'''
            <div class="panel">
                <div class="eyebrow">Typical Prediction Range</div>
                <div style="font-family:'IBM Plex Mono'; font-size:1.4rem;">
                    {avg_lower:.1f}% — {avg_upper:.1f}%
                </div>
                <div class="kpi-help">Average lower and upper bound across all current predictions.</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        st.markdown("")

        st.markdown(
            '<div class="eyebrow">Interval width across samples</div>',
            unsafe_allow_html=True,
        )

        widths = pred_df["upper_bound_pct"] - pred_df["lower_bound_pct"]

        fig = px.histogram(
            widths, nbins=25,
            color_discrete_sequence=[COLOR["accent"]],
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLOR["text"]},
            xaxis_title="Interval width (%)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("🔧 Technical Details"):
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("PICP", f"{unc['PICP']:.3f}")
        t2.metric("Target Coverage", f"{target_cov:.3f}")
        t3.metric("MPIW", f"{unc['MPIW']:.2f}")
        t4.metric(
            "CWC",
            f"{unc.get('CWC', float('nan')):.3f}" if unc.get("CWC") is not None else "—",
        )
        st.markdown(
            f"- **PICP** — {GLOSSARY['PICP']}\n"
            f"- **MPIW** — {GLOSSARY['MPIW']}\n"
            f"- **CWC** — {GLOSSARY['CWC']}\n"
            f"- **CQR** — {GLOSSARY['CQR']}"
        )


# ============================================================
# PAGE: WHY THIS PREDICTION?  (formerly "Explainability" / SHAP)
# ============================================================

elif page == "Why This Prediction?":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // SHAP'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Why Did the Model Predict This?")

    st.markdown(
        "The chart below shows which vibration characteristics had the "
        "strongest influence on the model's prediction."
    )

    imp_df, imp_live = load_feature_importance()

    data_badge(imp_live, "SHAP values")

    st.markdown("")

    fig = px.bar(
        imp_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[COLOR["accent"]],
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR["text"]},
        yaxis_title="",
        xaxis_title="Mean |SHAP value|",
        height=380,
    )

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Note: a feature's influence on the model's prediction is not "
        "automatically evidence of the physical root cause of wear or "
        "failure — it reflects statistical association learned by the model."
    )

    with st.expander("🔧 Technical SHAP Details"):
        st.markdown(
            f"**SHAP** {info_tip('SHAP')} — {GLOSSARY['SHAP']}\n\n"
            "Values below are the mean absolute SHAP value per feature, "
            "ranking features by their average influence on the model's "
            "output across the evaluated samples.",
            unsafe_allow_html=True,
        )
        st.dataframe(imp_df, width="stretch", hide_index=True)


# ============================================================
# PAGE: AI MAINTENANCE ASSISTANT  (formerly bottom-of-Explainability Ollama)
# ============================================================

elif page == "AI Maintenance Assistant":

    st.markdown(
        '<div class="eyebrow">'
        'AI-ASSISTED EXPLANATION // LOCAL OLLAMA'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("## AI Maintenance Assistant")

    st.markdown(
        "Ask the local AI assistant to explain the bearing prediction in "
        "plain English."
    )

    pred_df, pred_live = load_predictions()
    imp_df, imp_live = load_feature_importance()

    if pred_df is not None and not pred_df.empty:

        required_cols = {
            "bearing",
            "predicted_RUL_pct",
            "lower_bound_pct",
            "upper_bound_pct",
        }

        if required_cols.issubset(pred_df.columns):

            st.markdown(
                '<div class="panel panel-accent">'
                'Select a bearing snapshot and generate a grounded '
                'maintenance explanation using the existing RUL prediction, '
                'uncertainty interval, maintenance recommendation, '
                'and SHAP outputs. '
                'The explanation is generated locally using Ollama.'
                '</div>',
                unsafe_allow_html=True,
            )

            gcol1, gcol2 = st.columns(2)

            with gcol1:
                explain_bearing = st.selectbox(
                    "Bearing for AI explanation",
                    sorted(pred_df["bearing"].astype(str).unique()),
                    key="ollama_bearing",
                )

            explain_df = (
                pred_df[pred_df["bearing"].astype(str) == str(explain_bearing)]
                .reset_index(drop=True)
            )

            with gcol2:
                if "snapshot_idx" in explain_df.columns:
                    snapshot_options = sorted(explain_df["snapshot_idx"].unique())
                    explain_snapshot = st.selectbox(
                        "Snapshot for AI explanation",
                        snapshot_options,
                        key="ollama_snapshot",
                    )
                    explain_row = explain_df[explain_df["snapshot_idx"] == explain_snapshot].iloc[0]
                else:
                    explain_snapshot = st.selectbox(
                        "Snapshot for AI explanation",
                        list(range(len(explain_df))),
                        key="ollama_snapshot",
                    )
                    explain_row = explain_df.iloc[int(explain_snapshot)]

            st.markdown("")

            rul_val = float(explain_row["predicted_RUL_pct"])
            label, risk, color, css = health_status(rul_val)

            d0, d1, d2, d3 = st.columns(4)

            d0.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Bearing Health</div>'
                f'<div class="kpi-value" style="color:{color};">{label}</div></div>',
                unsafe_allow_html=True,
            )

            d1.metric("Predicted RUL", f"{rul_val:.1f}%")

            d2.metric(
                "Prediction Interval",
                f"{float(explain_row['lower_bound_pct']):.1f}% – "
                f"{float(explain_row['upper_bound_pct']):.1f}%",
            )

            explanation_action = explain_row.get(
                "recommended_action",
                explain_row.get("risk_aware_action", "—"),
            )

            d3.metric("Recommended Action", str(explanation_action))

            st.markdown(
                '<span class="badge badge-live">● LOCAL OLLAMA AI</span>',
                unsafe_allow_html=True,
            )

            st.caption(
                f"Model: {OLLAMA_MODEL} • Runs locally • No API credits required"
            )

            st.markdown("")

            if st.button(
                "✨ Generate AI Maintenance Explanation",
                key="generate_ollama_explanation",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Local AI is analyzing the existing prediction, "
                        "uncertainty, maintenance action, and SHAP outputs..."
                    ):
                        explanation = generate_ollama_explanation(explain_row, imp_df)

                    st.session_state["ollama_explanation"] = explanation
                    st.session_state["ollama_explanation_context"] = (
                        str(explain_bearing),
                        str(explain_snapshot),
                    )

                except Exception as e:
                    st.error("Local Ollama AI could not generate the explanation.")
                    st.info(
                        "Make sure Ollama is installed and running, then ensure "
                        f"the model is available by running `ollama run {OLLAMA_MODEL}` "
                        "in a terminal."
                    )
                    st.code(str(e))

            current_context = (str(explain_bearing), str(explain_snapshot))
            saved_context = st.session_state.get("ollama_explanation_context")

            if (
                "ollama_explanation" in st.session_state
                and saved_context == current_context
            ):
                st.markdown("")
                st.markdown(
                    '<div class="eyebrow">AI Maintenance Explanation</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state["ollama_explanation"])

            with st.expander("🔧 What information does the AI receive?"):
                st.markdown(
                    "The assistant only receives the values already computed "
                    "by the pipeline for this bearing/snapshot:\n\n"
                    "- Existing predicted RUL\n"
                    "- Existing prediction interval (lower/upper bound)\n"
                    "- Existing maintenance recommendation\n"
                    "- Existing SHAP feature importance\n\n"
                    "It is explicitly instructed **not** to invent sensor "
                    "values, causes, dates, predictions, or percentages."
                )

        else:
            st.warning(
                "The prediction output is missing one or more required "
                "columns: bearing, predicted_RUL_pct, lower_bound_pct, "
                "or upper_bound_pct."
            )

    else:
        st.warning(
            "Prediction data is required before an AI-assisted explanation "
            "can be generated."
        )
