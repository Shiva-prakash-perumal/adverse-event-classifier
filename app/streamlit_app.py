"""
streamlit_app.py
----------------
Professional clinical-grade UI for the Adverse Event Intelligence Pipeline.
Run with: streamlit run app/streamlit_app.py
"""

import sys
import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adverse Event Intelligence | Medidata",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"

SEVERITY_COLORS = {
    "Mild":     "#16a34a",
    "Moderate": "#d97706",
    "Severe":   "#dc2626"
}

SEVERITY_BG = {
    "Mild":     "#f0fdf4",
    "Moderate": "#fffbeb",
    "Severe":   "#fef2f2"
}

SEVERITY_BORDER = {
    "Mild":     "#bbf7d0",
    "Moderate": "#fde68a",
    "Severe":   "#fecaca"
}

SAMPLE_NOTES = [
    {
        "label": "Case A — Severe",
        "tag": "Severe",
        "text": (
            "68 year old male patient reported severe chest pain and shortness of breath "
            "approximately 4 hours after taking 100mg of DrugX via oral route. "
            "Patient has pre-existing hypertension and type 2 diabetes. "
            "Patient was previously hospitalized for a drug reaction."
        )
    },
    {
        "label": "Case B — Moderate",
        "tag": "Moderate",
        "text": (
            "52 year old female developed moderate nausea, vomiting, and dizziness "
            "2 days after starting 50mg of DrugY via oral route. "
            "Patient is currently taking metformin for diabetes management. "
            "No prior adverse drug reactions on record."
        )
    },
    {
        "label": "Case C — Mild",
        "tag": "Mild",
        "text": (
            "35 year old male reported mild headache and slight fatigue "
            "one week after beginning 25mg of DrugZ orally. "
            "No concomitant medications. No prior conditions or drug reactions."
        )
    },
    {
        "label": "Case D — Elderly, High Risk",
        "tag": "Severe",
        "text": (
            "79 year old female with history of cardiac arrhythmia and renal insufficiency "
            "presented with acute dyspnea and hypotension 6 hours after intravenous "
            "administration of 200mg of DrugA. Patient was on warfarin and lisinopril. "
            "Transferred to ICU for monitoring."
        )
    },
    {
        "label": "Case E — Pediatric",
        "tag": "Moderate",
        "text": (
            "14 year old male reported moderate skin rash and fever of 38.5C "
            "three days after starting 10mg DrugB orally for infection treatment. "
            "No prior drug allergies. No other medications."
        )
    },
    {
        "label": "Case F — Anaphylaxis",
        "tag": "Severe",
        "text": (
            "45 year old female experienced anaphylactic reaction within 15 minutes "
            "of subcutaneous injection of 50mg DrugC. Symptoms included urticaria, "
            "angioedema, and severe hypotension. Patient was treated with epinephrine. "
            "History of penicillin allergy documented."
        )
    }
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] code {
    background: #1e293b !important;
    color: #7dd3fc !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

/* Page header */
.page-header {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.page-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0;
}
.page-subtitle {
    font-size: 0.875rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Section labels */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.75rem;
}

/* Prediction card */
.prediction-card {
    border-radius: 8px;
    padding: 24px;
    border: 1px solid;
    text-align: center;
}
.prediction-label {
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.prediction-sublabel {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Metric cards */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px 16px;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
}
.metric-value {
    font-size: 1.1rem;
    font-weight: 500;
    color: #0f172a;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 2px;
}

/* Review alert */
.review-alert {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #d97706;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #92400e;
}
.pass-alert {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid #16a34a;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #166534;
}

/* Severity indicator tags */
.severity-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-right: 4px;
}

/* Sample case cards */
.sample-card {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 14px 16px;
    cursor: pointer;
    transition: border-color 0.15s;
    background: white;
}
.sample-card:hover {
    border-color: #94a3b8;
}
.sample-tag {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 2px 6px;
    border-radius: 3px;
}
.sample-text {
    font-size: 0.78rem;
    color: #475569;
    margin-top: 6px;
    line-height: 1.5;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* Disclaimer */
.disclaimer {
    font-size: 0.72rem;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
    padding-top: 1rem;
    margin-top: 1rem;
}

/* Buttons */
.stButton > button {
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #1e293b !important;
}

/* Text area */
.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px !important;
    color: #334155 !important;
    background: #fafafa !important;
}
.stTextArea textarea:focus {
    border-color: #94a3b8 !important;
    box-shadow: none !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
}

/* Success/warning boxes */
.stSuccess, .stWarning, .stInfo {
    border-radius: 4px !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODELS_DIR / "production_model.pkl")
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
        best_model_name = joblib.load(MODELS_DIR / "best_model_name.pkl")
        return model, feature_names, best_model_name
    except FileNotFoundError:
        return None, None, None


def run_prediction(note: str) -> dict:
    try:
        from pipeline import predict_single
        return predict_single(note)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def probability_bar_chart(prob_mild, prob_moderate, prob_severe):
    fig = go.Figure(go.Bar(
        x=[prob_severe, prob_moderate, prob_mild],
        y=["Severe", "Moderate", "Mild"],
        orientation="h",
        marker_color=[
            SEVERITY_COLORS["Severe"],
            SEVERITY_COLORS["Moderate"],
            SEVERITY_COLORS["Mild"]
        ],
        text=[f"{v:.1f}%" for v in [prob_severe, prob_moderate, prob_mild]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#334155")
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 115], showgrid=False, zeroline=False,
                   tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(family="IBM Plex Sans", size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=40, t=10, b=10),
        height=180,
        showlegend=False
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Adverse Event Intelligence")
    st.markdown("---")

    st.markdown("##### About")
    st.markdown(
        "Clinical prototype for automated adverse event severity "
        "classification using LLM extraction and gradient-boosted ML."
    )

    st.markdown("---")
    st.markdown("##### Pipeline")
    st.code(
        "Clinical Note\n"
        "     ↓\n"
        "LLM Extraction\n"
        "     ↓\n"
        "Feature Engineering\n"
        "     ↓\n"
        "XGBoost Classifier\n"
        "     ↓\n"
        "Severity + Confidence",
        language=None
    )

    st.markdown("---")
    st.markdown("##### Model Status")

    model, feature_names, model_name = load_model()
    if model:
        st.success(f"Loaded: **{model_name.replace('_', ' ').title()}**")
        if feature_names:
            st.markdown(f"Features in use: **{len(feature_names)}**")
            with st.expander("View features"):
                for f in feature_names:
                    st.markdown(f"`{f}`")
    else:
        st.warning("No model found. Run `python src/pipeline.py`")

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.72rem;color:#475569'>"
        "Stack: Python · XGBoost · scikit-learn · LLM · MLflow · Docker"
        "</span>",
        unsafe_allow_html=True
    )


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <p class="page-title">Adverse Event Intelligence Pipeline</p>
    <p class="page-subtitle">
        Enter an unstructured clinical note to extract structured fields
        and predict adverse event severity.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Clinical Note Input</p>', unsafe_allow_html=True)

note_value = st.session_state.get("note", "")
clinical_note = st.text_area(
    label="clinical_note_input",
    label_visibility="collapsed",
    value=note_value,
    height=130,
    placeholder=(
        "Paste or type a clinical note here — "
        "e.g. 68 year old male patient reported severe chest pain "
        "approximately 4 hours after taking 100mg of DrugX..."
    )
)

btn_col, _ = st.columns([1, 5])
with btn_col:
    analyze = st.button("Analyze", use_container_width=True)


# ── Results ───────────────────────────────────────────────────────────────────
if analyze and clinical_note.strip():
    if not model:
        st.error("No trained model found. Run `python src/pipeline.py` first.")
        st.stop()

    with st.spinner("Running extraction and prediction..."):
        result = run_prediction(clinical_note)

    if result:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Results</p>', unsafe_allow_html=True)

        prediction   = result["prediction"]
        confidence   = result["confidence"]
        needs_review = result["needs_review"]
        extracted    = result["extracted_fields"]

        col_pred, col_conf, col_bars = st.columns([1.2, 1, 1.8])

        # ── Prediction verdict ──
        with col_pred:
            color  = SEVERITY_COLORS[prediction]
            bg     = SEVERITY_BG[prediction]
            border = SEVERITY_BORDER[prediction]
            st.markdown(
                f"""
                <div class="prediction-card" style="
                    background:{bg};
                    border-color:{border};
                ">
                    <div class="prediction-label" style="color:{color}">
                        {prediction}
                    </div>
                    <div class="prediction-sublabel">Predicted Severity</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
            if needs_review:
                st.markdown(
                    '<div class="review-alert">'
                    'Low confidence — flagged for clinical review'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="pass-alert">'
                    'Confidence above threshold'
                    '</div>',
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Confidence gauge ──
        with col_conf:
            st.markdown(
                f"""
                <div style="text-align:center;padding:20px 0 10px">
                    <div style="
                        font-family:'IBM Plex Mono',monospace;
                        font-size:2.4rem;
                        font-weight:500;
                        color:#0f172a;
                        line-height:1;
                    ">{confidence:.1f}%</div>
                    <div style="
                        font-size:0.7rem;
                        text-transform:uppercase;
                        letter-spacing:0.08em;
                        color:#94a3b8;
                        margin-top:6px;
                    ">Confidence</div>
                    <div style="
                        margin-top:12px;
                        font-size:0.72rem;
                        color:#64748b;
                    ">Threshold: 60%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ── Probability bars ──
        with col_bars:
            st.plotly_chart(
                probability_bar_chart(
                    result["prob_mild"],
                    result["prob_moderate"],
                    result["prob_severe"]
                ),
                use_container_width=True,
                config={"displayModeBar": False}
            )

        # ── Extracted fields ──
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Extracted Clinical Fields</p>', unsafe_allow_html=True)

        f1, f2, f3, f4 = st.columns(4)
        fields = [
            ("Age",              f"{extracted.get('age', 'N/A')} yrs"),
            ("Gender",           extracted.get("gender", "Unknown")),
            ("Dosage",           f"{extracted.get('dosage_mg', 'N/A')} mg"),
            ("Route",            extracted.get("route", "Unknown")),
            ("Concomitant Drugs",str(extracted.get("num_concomitant_drugs", 0))),
            ("Comorbidity",      "Yes" if extracted.get("has_comorbidity") else "No"),
            ("Prior Reaction",   "Yes" if extracted.get("has_prior_reaction") else "No"),
        ]
        cols = [f1, f2, f3, f4, f1, f2, f3, f4]
        for (label, value), col in zip(fields, cols):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div>'
                    f'</div><br>',
                    unsafe_allow_html=True
                )

        # ── Severity indicators ──
        indicators = extracted.get("severity_indicators", [])
        if indicators:
            st.markdown('<p class="section-label" style="margin-top:0.5rem">Severity Indicators Detected</p>', unsafe_allow_html=True)
            tags_html = ""
            for ind in indicators[:8]:
                tags_html += (
                    f'<span style="'
                    f'background:#fef2f2;color:#991b1b;border:1px solid #fecaca;'
                    f'padding:3px 8px;border-radius:3px;font-size:0.72rem;'
                    f'font-weight:500;margin-right:6px;font-family:IBM Plex Sans">'
                    f'{ind}</span>'
                )
            st.markdown(tags_html, unsafe_allow_html=True)

        # ── Raw JSON ──
        with st.expander("View raw extracted JSON"):
            st.json(extracted)

        # ── Disclaimer ──
        st.markdown(
            '<div class="disclaimer">'
            'This is a research prototype. All predictions must be reviewed '
            'by qualified clinical staff before informing any clinical decision.'
            '</div>',
            unsafe_allow_html=True
        )

elif analyze and not clinical_note.strip():
    st.warning("Please enter a clinical note before running analysis.")


# ── Sample Cases ──────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Sample Cases</p>', unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:0.8rem;color:#64748b;margin-bottom:1rem'>"
    "Select a case to populate the input field above.</p>",
    unsafe_allow_html=True
)

tag_styles = {
    "Severe":   "background:#fef2f2;color:#991b1b;border:1px solid #fecaca",
    "Moderate": "background:#fffbeb;color:#92400e;border:1px solid #fde68a",
    "Mild":     "background:#f0fdf4;color:#166534;border:1px solid #bbf7d0",
}

cols = st.columns(3)
for i, case in enumerate(SAMPLE_NOTES):
    with cols[i % 3]:
        tag_style = tag_styles[case["tag"]]
        st.markdown(
            f"""
            <div class="sample-card">
                <span class="sample-tag" style="{tag_style}">{case["tag"]}</span>
                <span style="font-size:0.78rem;font-weight:500;color:#334155;margin-left:6px">
                    {case["label"]}
                </span>
                <div class="sample-text">{case["text"][:120]}...</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(f"Use this case", key=f"sample_{i}"):
            st.session_state["note"] = case["text"]
            st.rerun()


# ── How It Works ──────────────────────────────────────────────────────────────
with st.expander("How this pipeline works"):
    st.markdown("""
    **Step 1 — LLM Extraction**
    The clinical note is sent to an LLM (GPT / Mistral) which extracts
    structured fields: age, dosage, route, adverse event, time to onset.
    Falls back to rule-based regex extraction if no API key is configured.

    **Step 2 — Feature Engineering**
    Extracted fields are transformed into ML-ready features including
    elderly flag, high dose flag, serious AE classification, and a
    composite clinical risk score.

    **Step 3 — XGBoost Classifier**
    A gradient-boosted classifier trained on FDA FAERS data predicts
    severity: Mild / Moderate / Severe.

    **Step 4 — Confidence and Abstain**
    Predictions below 60% confidence are flagged for mandatory human review.

    **Step 5 — MLOps**
    All runs tracked via MLflow. Model containerized with Docker.
    CI/CD via GitHub Actions with automated retraining triggers.
    """)