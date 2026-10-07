"""
tc_dann_app.py  —  TC-DANN Clinical Voice Screening Interface
=============================================================
Streamlit frontend for the TC-DANN quad-model voice biomarker system.
Replaces the MARVEL VoxClinBench frontend.

Tabs:
  1. Record  — browser mic capture → live inference
  2. Upload  — WAV/MP3 file upload → inference
  3. Results — session history
  4. About   — model architecture, SHAP, clinical notes

Run:
    streamlit run tc_dann_app.py

    API must be running:
    uvicorn tc_dann_api_server:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TC-DANN · Voice Screening",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = os.getenv("TC_DANN_API", "http://localhost:8000")

# ── Disease → model domain color ─────────────────────────────────────────────
DOMAIN_COLORS = {
    "Voice+Onco":    "#028090",
    "Neurological":  "#4472C4",
    "Respiratory":   "#00A896",
    "Psychiatric":   "#ED7D31",
}

DOMAIN_BG = {
    "Voice+Onco":    "#e1f5ee",
    "Neurological":  "#e6f1fb",
    "Respiratory":   "#e1f5ee",
    "Psychiatric":   "#faeeda",
}

TASK_OPTIONS = {
    "Sustained vowel /a/ (prolonged-vowel)": "prolonged-vowel",
    "Read speech":                            "read-speech",
    "Free speech":                            "free-speech",
    "Diadochokinesis (pa-ta-ka)":             "diadochokinesis",
}

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background: #f7f9fc; }

/* Header */
.tc-header {
    background: linear-gradient(135deg, #0D2233 0%, #1a3a52 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.tc-header-title {
    font-size: 2.2rem;
    font-weight: 600;
    color: #02C39A;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1;
}
.tc-header-sub {
    font-size: 0.95rem;
    color: #9DB4C0;
    margin: 0.3rem 0 0;
    font-weight: 300;
}

/* Model status pills */
.model-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    margin: 2px;
}
.pill-ok   { background: #e1f5ee; color: #0f6e56; }
.pill-miss { background: #fcebeb; color: #a32d2d; }

/* Confidence card */
.conf-card {
    background: #0D2233;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}
.conf-card-score {
    font-size: 3.2rem;
    font-weight: 600;
    color: #02C39A;
    line-height: 1;
}
.conf-card-label {
    font-size: 0.82rem;
    color: #9DB4C0;
    margin-top: 0.3rem;
}
.conf-card-disease {
    font-size: 1.1rem;
    font-weight: 500;
    color: #FFFFFF;
    margin-top: 0.5rem;
}

/* Top-3 result rows */
.result-row {
    background: #ffffff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.result-rank {
    font-size: 1.4rem;
    font-weight: 600;
    color: #9DB4C0;
    min-width: 2rem;
}
.result-disease { font-weight: 500; font-size: 1rem; }
.result-model   { font-size: 0.78rem; color: #607A8A; font-family: 'DM Mono', monospace; }
.result-conf    {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: #0D2233;
    min-width: 5rem;
    text-align: right;
}

/* Audit flag */
.flag-warn  { background: #faeeda; border-left: 3px solid #ED7D31; padding: 8px 12px; border-radius: 6px; margin: 4px 0; font-size: 0.85rem; }
.flag-error { background: #fcebeb; border-left: 3px solid #E24B4A; padding: 8px 12px; border-radius: 6px; margin: 4px 0; font-size: 0.85rem; }
.flag-info  { background: #e6f1fb; border-left: 3px solid #4472C4; padding: 8px 12px; border-radius: 6px; margin: 4px 0; font-size: 0.85rem; }

/* Disclaimer */
.disclaimer {
    font-size: 0.78rem;
    color: #9DB4C0;
    background: #f0f4f8;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 1rem;
    font-style: italic;
}

/* Architecture boxes */
.arch-box {
    background: #FFFFFF;
    border: 1px solid #e0e8ed;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.arch-box-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}

/* Stat pill row */
.stat-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.stat-pill {
    background: #0D2233;
    color: #02C39A;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    text-align: center;
}
.stat-pill span { display: block; color: #9DB4C0; font-size: 0.72rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def _health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception as e:
        return {"error": str(e), "models": {}}


def _predict_audio(audio_bytes: bytes, task_name: str, patient_meta: dict) -> dict:
    r = requests.post(
        f"{API_BASE}/predict",
        files={"audio_file": ("recording.wav", audio_bytes, "audio/wav")},
        data={
            "task_name":    task_name,
            "patient_meta": json.dumps(patient_meta),
            "session_id":   str(uuid.uuid4()),
        },
        timeout=30,
    )
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"API returned {r.status_code}: {detail}")
    return r.json()


def _render_results(result: dict):
    top3 = result.get("top3", [])
    if not top3:
        st.warning("No results returned from the model.")
        return

    top1 = top3[0]

    # Big confidence card
    conf_pct = round(top1.get("confidence", 0) * 100, 1)
    disease_display = top1.get("display", top1.get("disease", "")).replace("_", " ").title()
    model_label = top1.get("model", "")

    st.markdown(f"""
    <div class="conf-card">
        <div class="conf-card-score">{conf_pct}%</div>
        <div class="conf-card-label">confidence</div>
        <div class="conf-card-disease">{disease_display}</div>
        <div style="font-size:0.78rem;color:#607A8A;margin-top:4px">{model_label} model</div>
    </div>
    """, unsafe_allow_html=True)

    # Top-3 rows
    st.markdown("##### Cross-model top-3 differential")
    for r in top3:
        color = DOMAIN_COLORS.get(r.get("model", ""), "#028090")
        dis   = r.get("display", r.get("disease", "")).replace("_", " ").title()
        rank  = r.get("rank", "?")
        prob  = r.get("y_prob", 0)
        conf  = r.get("confidence", 0)
        model = r.get("model", "")
        st.markdown(f"""
        <div class="result-row" style="border-left-color:{color}">
            <span class="result-rank">#{rank}</span>
            <div style="flex:1;margin:0 1rem">
                <div class="result-disease">{dis}</div>
                <div class="result-model">{model}</div>
            </div>
            <div class="result-conf">
                conf {conf:.3f}<br>
                <span style="font-size:0.78rem;color:#9DB4C0">p={prob:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Audit flags
    flags = result.get("audit_flags", [])
    if flags:
        st.markdown("##### Audit flags")
        for f in flags:
            lvl = f.get("level", "info")
            cls = f"flag-{lvl}"
            st.markdown(f'<div class="{cls}">{f["msg"]}</div>', unsafe_allow_html=True)

    # Confidence breakdown
    with st.expander("Confidence score breakdown (top-1)"):
        st.markdown("""
        **Composite confidence = a·x + b·y − c·z + d·m + e·p**

        | Term | Meaning | Value |
        |------|---------|-------|
        | x | Subgroup cosine similarity (kNN) | — (requires training data) |
        | y | Disease head sigmoid probability | {y:.4f} |
        | z | Runner-up uncertainty (−) | {z:.4f} |
        | m | Demographic prior | {m:.4f} |
        | p | Confounder robustness | {p:.4f} |
        """.format(
            y=top1.get("y_prob", 0),
            z=top1.get("z_uncertainty", 0),
            m=top1.get("m_demo", 0),
            p=top1.get("p_confounder", 0),
        ))

    # All ranked (collapsed)
    with st.expander(f"All {len(result.get('all_ranked', []))} diseases ranked"):
        all_r = result.get("all_ranked", [])
        rows  = []
        for r in all_r[:30]:
            rows.append({
                "Rank":     r.get("rank"),
                "Disease":  r.get("display", r.get("disease", "")),
                "Model":    r.get("model"),
                "Prob":     round(r.get("y_prob", 0), 4),
                "Conf":     round(r.get("confidence", 0), 4),
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="disclaimer">
        Research-grade output from TC-DANN (Bridge2AI-Voice v6.0.0). Not a medical device.
        Not FDA-cleared. Not for diagnostic or clinical decision-making.
        Latency: {result.get("latency_ms", "—")} ms
    </div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
health = _health()
models = health.get("models", {})

st.markdown("""
<div class="tc-header">
    <div>
        <div class="tc-header-title">🎙 TC-DANN</div>
        <div class="tc-header-sub">Task-Conditioned Domain-Adversarial Voice Screening · Bridge2AI-Voice v6.0.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Model status pills
pills_html = ""
for name, info in models.items():
    ok  = info.get("loaded", False)
    cls = "pill-ok" if ok else "pill-miss"
    n   = info.get("diseases", 0)
    pills_html += f'<span class="model-pill {cls}">{name} {"✓" if ok else "✗"} {f"({n})" if ok else ""}</span>'

if "error" in health:
    st.error(f"API not reachable — is `uvicorn tc_dann_api_server:app --port 8000` running?\n\n{health['error']}")
else:
    st.markdown(f'<div style="margin-bottom:1rem">{pills_html}</div>', unsafe_allow_html=True)
    if health.get("status") == "no_models":
        st.warning(
            "The API is running but no trained TC-DANN bundles are loaded, so predictions "
            "are disabled. Trained weights are not distributed with this repository "
            "(Bridge2AI-Voice, PhysioNet data use agreement). Credentialed PhysioNet users "
            "can train them with `python tcdann/run_tc_dann.py --data_root $B2AI_DATA_ROOT "
            "--out_dir models` and restart the API, or set `TC_DANN_BUNDLE_DIR`."
        )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_record, tab_upload, tab_history, tab_about = st.tabs(
    ["🎤 Record", "📁 Upload Audio", "📋 Session History", "ℹ️ About"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECORD
# ══════════════════════════════════════════════════════════════════════════════
with tab_record:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### Patient details")
        with st.form("patient_form_record"):
            age  = st.number_input("Age", 18, 110, 55)
            sex  = st.selectbox("Sex", ["Female", "Male", "Prefer not to say"])
            task_label = st.selectbox("Recording task", list(TASK_OPTIONS.keys()))
            task_name  = TASK_OPTIONS[task_label]

            st.markdown("---")
            st.markdown("#### Recording instructions")
            task_instructions = {
                "prolonged-vowel": "Say /aaa/ at a comfortable pitch and loudness for 3-5 seconds.",
                "read-speech":     "Read the displayed sentence aloud naturally.",
                "free-speech":     "Describe what you did this morning for 30 seconds.",
                "diadochokinesis": "Repeat /pa-ta-ka/ as quickly and clearly as possible for 5 seconds.",
            }
            st.info(task_instructions.get(task_name, "Follow the instructions for this task."))

            audio_input = st.audio_input("Record audio", key="mic_recorder")
            submitted = st.form_submit_button("⚡ Run TC-DANN Inference", use_container_width=True)

    with col_r:
        st.markdown("#### Results")
        if submitted and audio_input is not None:
            audio_bytes = audio_input.read()
            patient_meta = {"age": age, "sex": sex}
            with st.spinner("Running TC-DANN quad-model inference..."):
                try:
                    result = _predict_audio(audio_bytes, task_name, patient_meta)
                    st.session_state["last_result"] = result
                    _render_results(result)
                except Exception as e:
                    st.error(f"Inference error: {e}")
        elif "last_result" in st.session_state:
            st.caption("Previous result:")
            _render_results(st.session_state["last_result"])
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#9DB4C0">
                <div style="font-size:3rem">🎙</div>
                <div>Record audio and click Run Inference</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### Upload audio file")
        uploaded = st.file_uploader(
            "WAV, MP3, M4A, OGG, FLAC",
            type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        )

        with st.form("upload_form"):
            age_u = st.number_input("Age", 18, 110, 55, key="age_u")
            sex_u = st.selectbox("Sex", ["Female", "Male", "Prefer not to say"], key="sex_u")
            task_label_u = st.selectbox("Task type", list(TASK_OPTIONS.keys()), key="task_u")
            task_name_u  = TASK_OPTIONS[task_label_u]
            run_upload   = st.form_submit_button("⚡ Run Inference", use_container_width=True)

        if uploaded:
            st.audio(uploaded)

    with col_r:
        st.markdown("#### Results")
        if run_upload and uploaded:
            audio_bytes = uploaded.read()
            patient_meta = {"age": age_u, "sex": sex_u}
            with st.spinner("Running TC-DANN inference..."):
                try:
                    result = _predict_audio(audio_bytes, task_name_u, patient_meta)
                    st.session_state["last_upload_result"] = result
                    _render_results(result)
                except Exception as e:
                    st.error(f"Inference error: {e}")
        elif "last_upload_result" in st.session_state:
            st.caption("Previous result:")
            _render_results(st.session_state["last_upload_result"])
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#9DB4C0">
                <div style="font-size:3rem">📁</div>
                <div>Upload an audio file and click Run Inference</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("#### Recent sessions")
    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        r = requests.get(f"{API_BASE}/sessions", timeout=5)
        sessions = r.json().get("sessions", [])
    except Exception:
        sessions = []

    if not sessions:
        st.info("No sessions yet. Run some inferences first.")
    else:
        import pandas as pd
        df = pd.DataFrame(sessions)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        sid = st.text_input("Load session ID")
        if sid:
            try:
                r = requests.get(f"{API_BASE}/sessions/{sid}", timeout=5)
                detail = r.json()
                st.json(detail)
            except Exception as e:
                st.error(f"Could not load session: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("#### TC-DANN Architecture")

    st.markdown("""
    <div class="stat-row">
        <div class="stat-pill">20<span>diseases</span></div>
        <div class="stat-pill">4<span>clinical domains</span></div>
        <div class="stat-pill">FC 512→256→128<span>per domain</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Four domain models")
        domain_info = {
            "Voice + Onco":  ("028090", "Laryngeal dystonia, VFP, benign lesions, MTD, glottic insufficiency, laryngeal cancer, precancerous lesions, control"),
            "Neurological":  ("4472C4", "Parkinson's disease, cognitive impairment, ALS"),
            "Respiratory":   ("00A896", "Airway stenosis, COPD/asthma, unexplained chronic cough, laryngitis"),
            "Psychiatric":   ("ED7D31", "ADHD, PTSD, anxiety, depression, bipolar disorder"),
        }
        for label, (color, diseases) in domain_info.items():
            st.markdown(f"""
            <div class="arch-box" style="border-left:4px solid #{color}">
                <div class="arch-box-title" style="color:#{color}">{label}</div>
                <div style="font-size:0.82rem;color:#607A8A">{diseases}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("##### Key design decisions")
        for title, body in [
            ("Domain separation", "One model per clinical domain eliminates cross-domain feature interference that plagued unified models (MARVEL)."),
            ("FiLM task conditioning", "Feature-wise Linear Modulation recalibrates the encoder per speech task. The same jitter value carries different diagnostic weight in a sustained vowel vs. diadochokinesis."),
            ("Gradient reversal", "A site adversary forces the 128-dim representation to be site-invariant, preventing the model from learning recording environment artefacts."),
            ("Composite confidence", "conf = a·x + b·y − c·z + d·m + e·p. Five interpretable terms with domain-specific weight sets (psychiatric: higher confounder penalty, no demographic prior)."),
            ("SHAP explainability", "DeepExplainer attribution per-disease, per-task. Top features: jitter, shimmer, alpha ratio, F0 contour — all grounded in known pathophysiology."),
        ]:
            st.markdown(f"""
            <div class="arch-box">
                <div class="arch-box-title">{title}</div>
                <div style="font-size:0.84rem;color:#44546A">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("##### SHAP top features (Voice+Onco · laryngeal dystonia)")
    shap_data = {
        "Feature":    ["jitter (local)", "alpha ratio V (norm)", "logRelF0-H1-A3",
                       "jitter (stddev)", "F2 std (loc)", "F0 falling slope",
                       "F1 bandwidth", "shimmer (local dB)", "spectral flux", "spectral skewness"],
        "Interpretation": [
            "Cycle-to-cycle pitch irregularity — involuntary VF disruption",
            "Low/high freq energy balance — vocal tract resonance change",
            "Spectral tilt relative to harmonics",
            "Variability of jitter across recording",
            "Formant frequency instability",
            "Pitch trajectory — prosodic control",
            "Formant bandwidth — resonance sharpness",
            "Amplitude irregularity — shimmer",
            "Frame-to-frame spectral change",
            "Spectral asymmetry",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(shap_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        TC-DANN is a research model trained on the Bridge2AI-Voice dataset (v6.0.0).
        All predictions are screening signals only — not diagnoses.
        Not a medical device. Not FDA-cleared.
        For research use only.
    </div>
    """, unsafe_allow_html=True)
