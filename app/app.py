"""
Bearing RUL Diagnostic System — Streamlit application.

Design: instrument-panel aesthetic (the visual language of the equipment being
monitored, not a generic SaaS dashboard). Every page tries to load real pipeline
output first (from OUTPUT_DIR) and falls back to clearly-labeled demo values if
that stage of the pipeline hasn't been (re-)run yet — so the app upgrades itself
automatically as Chapters 5-7 complete on the full FAST_DEMO=False data.
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
    page_title="Bearing RUL Diagnostic System",
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


def zone_for_rul(rul_pct):

    if rul_pct < 20:
        return COLOR["critical"], "CRITICAL"

    elif rul_pct < 40:
        return COLOR["caution"], "MODERATE"

    return COLOR["healthy"], "HEALTHY"


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


page = st.sidebar.radio(
    "Console",

    [
        "Overview",
        "RUL Prediction",
        "Model Performance",
        "Uncertainty",
        "Maintenance Policy",
        "Explainability"
    ],

    label_visibility="collapsed",
)


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
    "🔍 Data source diagnostics"
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
# PAGE: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="eyebrow">'
        'BEARING DIAGNOSTIC SYSTEM // OVERVIEW'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "## Remaining Useful Life Prediction & Risk-Aware Maintenance"
    )

    st.caption(
        "Vibration-based RUL estimation with calibrated uncertainty "
        "and explainable, risk-aware maintenance recommendations."
    )

    st.markdown("---")

    model_df, model_live = load_model_comparison()

    unc, unc_live = load_uncertainty_metrics()

    best = (
        model_df
        .sort_values("MAE")
        .iloc[0]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Selected Model",
        best["model"]
    )

    c2.metric(
        "Test MAE",
        f"{best['MAE']:.2f}%"
    )

    c3.metric(
        "Coverage (PICP)",
        f"{unc['PICP']:.3f}"
    )

    c4.metric(
        "Interval Width",
        f"{unc['MPIW']:.1f}%"
    )

    st.markdown("")

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown(
            '<div class="panel panel-accent">'
            '<div class="eyebrow">Capability</div>'
            '<b>RUL Prediction + Uncertainty</b>'
            '<p style="color:var(--text-dim); font-size:0.9rem;">'
            'Point RUL estimate from the best-performing model, with '
            'a calibrated interval from Conformalized Quantile '
            'Regression — not just a number, a defensible range.'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with col_b:

        st.markdown(
            '<div class="panel panel-accent">'
            '<div class="eyebrow">Capability</div>'
            '<b>Risk-Aware Maintenance</b>'
            '<p style="color:var(--text-dim); font-size:0.9rem;">'
            'Maintenance action is decided from the interval lower '
            'bound, not the point estimate alone — a wide interval '
            'on an optimistic prediction can still trigger inspection.'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="eyebrow">Data Source</div>',
        unsafe_allow_html=True
    )

    data_badge(
        model_live,
        "model comparison"
    )

    data_badge(
        unc_live,
        "uncertainty metrics"
    )


# ============================================================
# PAGE: RUL PREDICTION
# ============================================================

elif page == "RUL Prediction":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // RUL PREDICTION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "## Predict Remaining Useful Life"
    )

    pred_df, pred_live = load_predictions()

    data_badge(
        pred_live,
        "predictions"
    )

    st.markdown("")

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        bearing = st.selectbox(
            "Bearing",
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
            "Snapshot",

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

    st.markdown("---")

    gcol, dcol = st.columns(
        [1, 1]
    )

    with gcol:

        fig, zone_color, zone_label = rul_gauge(
            row["predicted_RUL_pct"],
            row["lower_bound_pct"],
            row["upper_bound_pct"]
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.markdown(
            f'<div style="text-align:center;">'
            f'<span class="badge" '
            f'style="color:{zone_color};">'
            f'{zone_label} ZONE'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    with dcol:

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Lower Bound",
            f"{row['lower_bound_pct']:.1f}%"
        )

        m2.metric(
            "Predicted",
            f"{row['predicted_RUL_pct']:.1f}%"
        )

        m3.metric(
            "Upper Bound",
            f"{row['upper_bound_pct']:.1f}%"
        )

        action = row.get(
            "recommended_action",
            row.get(
                "risk_aware_action",
                "—"
            )
        )

        action_color = {

            "Replace immediately":
                COLOR["critical"],

            "Inspect before replacement":
                COLOR["caution"],

            "Schedule maintenance":
                COLOR["caution"],

            "Continue monitoring":
                COLOR["healthy"],

        }.get(
            action,
            COLOR["accent"]
        )

        st.markdown(
            f'<div class="action-card" '
            f'style="--zone-color:{action_color}; '
            f'margin-top:0.6rem;">'

            f'<div class="eyebrow">'
            f'Recommended Action'
            f'</div>'

            f'<div '
            f'style="font-family:Space Grotesk; '
            f'font-size:1.3rem; '
            f'color:{action_color};">'
            f'{action}'
            f'</div>'

            f'</div>',

            unsafe_allow_html=True
        )

        top_feat = row.get(
            "top_driver_1_feature",
            "—"
        )

        st.markdown(
            f'<div class="panel" '
            f'style="margin-top:0.8rem;">'

            f'<div class="eyebrow">'
            f'Top Driving Feature'
            f'</div>'

            f'<code '
            f'style="color:var(--accent);">'
            f'{top_feat}'
            f'</code>'

            f'</div>',

            unsafe_allow_html=True
        )


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

    st.markdown(
        "## Model Performance Comparison"
    )

    model_df, model_live = load_model_comparison()

    data_badge(
        model_live,
        "model comparison"
    )

    st.markdown("")

    disp = (
        model_df
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    st.dataframe(
        disp,
        width="stretch",
        hide_index=True
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.bar(
            disp,
            x="model",
            y="MAE",
            color="model",

            color_discrete_sequence=[
                COLOR["accent"],
                COLOR["text_dim"],
                COLOR["caution"],
                COLOR["healthy"]
            ]
        )

        fig.update_layout(
            showlegend=False,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            font={
                "color": COLOR["text"]
            },

            title="MAE by Model (lower is better)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with c2:

        fig = px.bar(
            disp,
            x="model",
            y="R2",
            color="model",

            color_discrete_sequence=[
                COLOR["accent"],
                COLOR["text_dim"],
                COLOR["caution"],
                COLOR["healthy"]
            ]
        )

        fig.update_layout(
            showlegend=False,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            font={
                "color": COLOR["text"]
            },

            title="R² by Model (higher is better)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# PAGE: UNCERTAINTY
# ============================================================

elif page == "Uncertainty":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // CQR CALIBRATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "## Prediction Interval Calibration"
    )

    unc, unc_live = load_uncertainty_metrics()

    data_badge(
        unc_live,
        "uncertainty metrics"
    )

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "PICP",
        f"{unc['PICP']:.3f}"
    )

    c2.metric(
        "Target Coverage",
        f"{unc.get('target_coverage', 0.9):.3f}"
    )

    c3.metric(
        "MPIW",
        f"{unc['MPIW']:.2f}"
    )

    c4.metric(
        "CWC",

        (
            f"{unc.get('CWC', float('nan')):.3f}"
            if unc.get("CWC") is not None
            else "—"
        )
    )

    gap = abs(
        unc["PICP"] -
        unc.get(
            "target_coverage",
            0.9
        )
    )

    gap_color = (
        COLOR["healthy"]
        if gap <= 0.03
        else COLOR["caution"]
    )

    st.markdown(
        f'<div class="panel" '
        f'style="border-left:3px solid {gap_color};">'

        f'Empirical coverage sits '
        f'<b style="color:{gap_color};">'
        f'{gap * 100:.1f} points'
        f'</b> '
        f'from the '
        f'{unc.get("target_coverage", 0.9) * 100:.0f}% '
        f'target — '

        f'{"within" if gap <= 0.03 else "outside"} '
        f'tolerance.'

        f'</div>',

        unsafe_allow_html=True
    )

    pred_df, pred_live = load_predictions()

    if {
        "predicted_RUL_pct",
        "lower_bound_pct",
        "upper_bound_pct"
    }.issubset(
        pred_df.columns
    ):

        st.markdown("")

        st.markdown(
            '<div class="eyebrow">'
            'Interval width across samples'
            '</div>',
            unsafe_allow_html=True
        )

        widths = (
            pred_df["upper_bound_pct"] -
            pred_df["lower_bound_pct"]
        )

        fig = px.histogram(
            widths,
            nbins=25,

            color_discrete_sequence=[
                COLOR["accent"]
            ]
        )

        fig.update_layout(
            showlegend=False,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            font={
                "color": COLOR["text"]
            },

            xaxis_title="Interval width (%)",

            yaxis_title="Count"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# PAGE: MAINTENANCE POLICY
# ============================================================

elif page == "Maintenance Policy":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // RISK-AWARE POLICY'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "## Risk-Aware Maintenance Recommendations"
    )

    st.caption(
        "Action is decided from the interval's lower bound, "
        "not the point estimate alone."
    )

    pred_df, pred_live = load_predictions()

    data_badge(
        pred_live,
        "recommendations"
    )

    st.markdown("")

    action_col = (
        "recommended_action"
        if "recommended_action" in pred_df.columns
        else "risk_aware_action"
    )

    counts = (
        pred_df[action_col]
        .value_counts()
    )

    zone_meta = {

        "Continue monitoring": (
            COLOR["healthy"],
            "Sufficient RUL remains. Continue normal monitoring."
        ),

        "Schedule maintenance": (
            COLOR["caution"],
            "Moderate risk. Plan maintenance in the near term."
        ),

        "Inspect before replacement": (
            COLOR["caution"],
            "Point estimate is optimistic but the interval "
            "signals real downside risk. Inspect before deciding."
        ),

        "Replace immediately": (
            COLOR["critical"],
            "Point estimate and worst-case downside agree: critical."
        ),
    }

    cols = st.columns(
        len(zone_meta)
    )

    for col, (
        action,
        (
            color,
            desc
        )
    ) in zip(
        cols,
        zone_meta.items()
    ):

        n = int(
            counts.get(
                action,
                0
            )
        )

        with col:

            st.markdown(
                f'<div class="action-card" '
                f'style="--zone-color:{color}; '
                f'min-height:170px;">'

                f'<div class="eyebrow" '
                f'style="color:{color};">'
                f'{action}'
                f'</div>'

                f'<div class="mono" '
                f'style="font-size:1.6rem;">'
                f'{n}'
                f'</div>'

                f'<div '
                f'style="font-size:0.82rem; '
                f'color:var(--text-dim); '
                f'margin-top:0.3rem;">'
                f'{desc}'
                f'</div>'

                f'</div>',

                unsafe_allow_html=True
            )

    st.markdown("")

    fig = px.pie(
        values=counts.values,

        names=counts.index,

        hole=0.55,

        color=counts.index,

        color_discrete_map={

            "Continue monitoring":
                COLOR["healthy"],

            "Schedule maintenance":
                COLOR["caution"],

            "Inspect before replacement":
                "#FF8C42",

            "Replace immediately":
                COLOR["critical"],
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",

        font={
            "color": COLOR["text"]
        },

        legend={
            "orientation": "h",
            "y": -0.1
        }
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# PAGE: EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.markdown(
        '<div class="eyebrow">'
        'MODULE // SHAP'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "## Why the Model Predicted This"
    )

    imp_df, imp_live = load_feature_importance()

    data_badge(
        imp_live,
        "SHAP values"
    )

    st.markdown("")

    fig = px.bar(
        imp_df.sort_values(
            "importance"
        ),

        x="importance",

        y="feature",

        orientation="h",

        color_discrete_sequence=[
            COLOR["accent"]
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font={
            "color": COLOR["text"]
        },

        yaxis_title="",

        xaxis_title="Mean |SHAP value|",

        height=380
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # LOCAL AI / OLLAMA SECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">'
        'AI-ASSISTED EXPLANATION // LOCAL OLLAMA'
        '</div>',
        unsafe_allow_html=True
    )

    pred_df, pred_live = load_predictions()

    if (
        pred_df is not None
        and not pred_df.empty
    ):

        required_cols = {

            "bearing",

            "predicted_RUL_pct",

            "lower_bound_pct",

            "upper_bound_pct",
        }

        if required_cols.issubset(
            pred_df.columns
        ):

            st.markdown(
                '<div class="panel panel-accent">'

                'Select a bearing snapshot and generate a grounded '
                'maintenance explanation using the existing RUL prediction, '
                'uncertainty interval, maintenance recommendation, '
                'and SHAP outputs. '

                'The explanation is generated locally using Ollama.'

                '</div>',

                unsafe_allow_html=True
            )

            gcol1, gcol2 = st.columns(2)

            with gcol1:

                explain_bearing = st.selectbox(
                    "Bearing for AI explanation",

                    sorted(
                        pred_df["bearing"]
                        .astype(str)
                        .unique()
                    ),

                    key="ollama_bearing"
                )

            explain_df = (
                pred_df[
                    pred_df["bearing"]
                    .astype(str)
                    == str(explain_bearing)
                ]
                .reset_index(drop=True)
            )

            with gcol2:

                if (
                    "snapshot_idx"
                    in explain_df.columns
                ):

                    snapshot_options = sorted(
                        explain_df[
                            "snapshot_idx"
                        ].unique()
                    )

                    explain_snapshot = st.selectbox(
                        "Snapshot for AI explanation",

                        snapshot_options,

                        key="ollama_snapshot"
                    )

                    explain_row = (
                        explain_df[
                            explain_df[
                                "snapshot_idx"
                            ]
                            == explain_snapshot
                        ]
                        .iloc[0]
                    )

                else:

                    explain_snapshot = st.selectbox(
                        "Snapshot for AI explanation",

                        list(
                            range(
                                len(explain_df)
                            )
                        ),

                        key="ollama_snapshot"
                    )

                    explain_row = explain_df.iloc[
                        int(explain_snapshot)
                    ]

            st.markdown("")

            d1, d2, d3 = st.columns(3)

            d1.metric(
                "Predicted RUL",

                f"{float(
                    explain_row[
                        'predicted_RUL_pct'
                    ]
                ):.1f}%"
            )

            d2.metric(
                "Prediction Interval",

                f"{float(
                    explain_row[
                        'lower_bound_pct'
                    ]
                ):.1f}% – "

                f"{float(
                    explain_row[
                        'upper_bound_pct'
                    ]
                ):.1f}%"
            )

            explanation_action = explain_row.get(
                "recommended_action",

                explain_row.get(
                    "risk_aware_action",
                    "—"
                ),
            )

            d3.metric(
                "Recommended Action",
                str(explanation_action)
            )

            st.markdown(
                '<span class="badge badge-live">'
                '● LOCAL OLLAMA AI'
                '</span>',

                unsafe_allow_html=True
            )

            st.caption(
                f"Model: {OLLAMA_MODEL} "
                "• Runs locally • No API credits required"
            )

            st.markdown("")

            if st.button(
                "✨ Generate AI Maintenance Explanation",

                key="generate_ollama_explanation",

                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Local AI is analyzing the existing prediction, "
                        "uncertainty, maintenance action, and SHAP outputs..."
                    ):

                        explanation = (
                            generate_ollama_explanation(
                                explain_row,
                                imp_df
                            )
                        )

                    st.session_state[
                        "ollama_explanation"
                    ] = explanation

                    st.session_state[
                        "ollama_explanation_context"
                    ] = (

                        str(explain_bearing),

                        str(explain_snapshot)
                    )

                except Exception as e:

                    st.error(
                        "Local Ollama AI could not generate the explanation."
                    )

                    st.info(
                        "Make sure Ollama is installed and running, "
                        "then ensure the model is available by running "
                        f"`ollama run {OLLAMA_MODEL}` in a terminal."
                    )

                    st.code(
                        str(e)
                    )

            current_context = (

                str(explain_bearing),

                str(explain_snapshot)
            )

            saved_context = st.session_state.get(
                "ollama_explanation_context"
            )

            if (

                "ollama_explanation"
                in st.session_state

                and saved_context
                == current_context
            ):

                st.markdown("")

                st.markdown(
                    '<div class="eyebrow">'
                    'AI Maintenance Explanation'
                    '</div>',

                    unsafe_allow_html=True
                )

                st.markdown(
                    st.session_state[
                        "ollama_explanation"
                    ]
                )

        else:

            st.warning(
                "The prediction output is missing one or more "
                "required columns: bearing, predicted_RUL_pct, "
                "lower_bound_pct, or upper_bound_pct."
            )

    else:

        st.warning(
            "Prediction data is required before an AI-assisted "
            "explanation can be generated."
        )



        