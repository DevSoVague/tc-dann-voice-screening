"""
VoxClinBench — Patient Voice AI Frontend (Extended)
42-657 Projects in Biomedical AI · Carnegie Mellon University

Pages:
  1. Voice Assessment  — 5-stage patient flow (OG, unchanged)
  2. Clinical AI Chat  — 4-specialist multi-agent pipeline (Pharma, Guidelines, Rehab, Comorbidity)
                         + Master synthesiser, powered by Milvus RAG + Tavily + Claude
  3. RAG Dev           — Milvus PDF indexing with LangGraph (BGE / Gemini embeddings)

Run:
    # Terminal 1 — MARVEL API
    MARVEL_CHECKPOINT=marvel_model.joblib uvicorn api_server:app --reload --port 8000
    # Terminal 2 — Frontend
    streamlit run voxclinbench_app.py

Optional env vars:
    VOXCLIN_API          — MARVEL API URL (default http://localhost:8000)
    ANTHROPIC_API_KEY    — for agent LLM calls
    GEMINI_API_KEY       — for Gemini embeddings / generation
    TAVILY_API_KEY       — for web-grounded agent search
    MILVUS_URI           — Milvus endpoint (default http://localhost:19530)
    MILVUS_TOKEN         — optional Milvus auth token
"""

import io, json, math, os, random, re, time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import requests
import streamlit as st

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# ── RAG / indexer import ──────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
try:
    from indexer import PDFIndexer, GEN_MODELS, EMBED_MODEL, PAPER_TYPES, INDEX_TYPES
    RAG_AVAILABLE = True
    _RAG_IMPORT_ERROR = None
except ImportError as _rag_err:
    RAG_AVAILABLE = False
    _RAG_IMPORT_ERROR = str(_rag_err)
    PAPER_TYPES  = ["AI / ML", "Security", "Clinical Guidelines", "Other"]
    EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
    INDEX_TYPES  = ["HNSW", "IVF_PQ", "DiskANN"]

# ── OG constants ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("VOXCLIN_API", "http://localhost:8000")
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

SPEC_FREQ = 201
MFCC_FREQ = 60
FIXED_T   = 256

TASK_NAMES = [
    "parkinsons","airway_stenosis","laryngeal_dystonia","vf_paralysis",
    "chronic_cough","mtd","benign_lesions","glottic_insuff",
    "depression","ptsd","adhd","bipolar","cognitive_impairment",
    "psychiatric_history","anxiety","precancerous","als",
    "copd_asthma","laryngitis","laryngeal_cancer",
]

TASK_BATTERIES = {
    "Structural / Motor": {
        "diseases": ["Parkinson's Disease","Laryngeal Dystonia","VF Paralysis","Airway Stenosis"],
        "color": "#028090",
        "tasks": [
            ("Sustained /a/ Phonation","Say 'ahhh' as steadily as you can for at least 5 seconds.",5),
            ("Diadochokinesis","Say 'pa-ta-ka' as rapidly and clearly as possible for 10 seconds.",10),
            ("Harvard Sentences","Read aloud: 'The birch canoe slid on the smooth planks.'",8),
            ("Spontaneous Description","Describe a busy street scene for up to 30 seconds.",30),
        ]
    },
    "Laryngeal / Vocal": {
        "diseases": ["Chronic Cough","MTD","Benign Lesions","Glottic Insufficiency"],
        "color": "#7c3aed",
        "tasks": [
            ("Sustained /a/ Low Pitch","Produce a comfortable low-pitched 'ahhh' and hold for 5 seconds.",5),
            ("Pitch Glide","Glide smoothly from your lowest to your highest comfortable pitch.",6),
            ("Counting Aloud","Count from 1 to 20 at a natural pace.",15),
            ("Rainbow Passage","Read aloud: 'When the sunlight strikes raindrops in the air...'",10),
        ]
    },
    "Psychiatric / Cognitive": {
        "diseases": ["Depression","PTSD","ADHD","Bipolar","Cognitive Impairment"],
        "color": "#b45309",
        "tasks": [
            ("Winograd Schema","Answer aloud: 'The trophy does not fit in the suitcase because it is too big. What is too big?'",20),
            ("Random Item Generation","Say as many random, unrelated items as you can think of for 60 seconds.",60),
            ("Stroop Color-Word","Name the COLOR of each word shown, not what it says.",20),
            ("Free Narrative","Describe your typical morning routine for up to 30 seconds.",30),
        ]
    },
}

AUDIT_FLAGS_STATIC = {
    "Structural / Motor": [
        ("ok",    "Parkinson's: model exceeds confounder baseline by > 0.10 -- acoustic signal likely"),
        ("amber", "VF Paralysis: sex gap detected (Female 0.96 vs Male 0.65) -- subgroup instability"),
        ("ok",    "Protocol B gap < 0.10 for all structural diseases"),
    ],
    "Laryngeal / Vocal": [
        ("warn",  "Chronic Cough: confounder baseline within 0.08 AUROC -- shortcut not ruled out"),
        ("amber", "Protocol B gap = 0.455 for chronic cough -- large curated-cohort dependency"),
        ("ok",    "MTD, Benign Lesions: acoustic signal present -- confounder gap > 0.10"),
    ],
    "Psychiatric / Cognitive": [
        ("warn",  "Depression: confounder baseline within 0.08 AUROC -- shortcut not ruled out"),
        ("warn",  "Cognitive Impairment: all test positives from Canada -- geographic leakage possible"),
        ("warn",  "ADHD Protocol B AUROC [0.413, 0.756] -- entirely below fair-protocol value"),
        ("amber", "Psychiatric labels require additional validation beyond acoustic AUROC (Rec. 5.5)"),
    ],
}

SUBGROUP_DATA = {
    "Structural / Motor": [
        ("Parkinson's","41-55","0.91","ok"),("Parkinson's","56-70","0.98","ok"),
        ("Parkinson's","71+","0.69","warn"),("VF Paralysis","Female","0.96","ok"),
        ("VF Paralysis","Male","0.65","warn"),
    ],
    "Laryngeal / Vocal": [
        ("Chronic Cough","41-55","0.98","ok"),("Chronic Cough","56-70","0.84","ok"),
        ("MTD","Female","0.84","ok"),("MTD","Male","0.99","ok"),
    ],
    "Psychiatric / Cognitive": [
        ("Depression","56-70","1.00","ok"),("Depression","71+","0.31","warn"),
        ("PTSD","Female","0.53","warn"),("PTSD","Male","0.83","ok"),
        ("Psych Hist","71+","0.19","warn"),("ADHD","all","0.58","warn"),
    ],
}

DEPLOYMENT_GATES = [
    ("Gate 1","External Transfer","Evaluated on independent external cohort.",
     False,"Not satisfied -- no external cohort evaluated for B2AI v3"),
    ("Gate 2","Confounder Separation","Acoustic model outperforms non-acoustic baseline by >= 0.10.",
     None,"Fails for Cognitive Impairment and Parkinson's; passes for structural diseases"),
    ("Gate 3","Subgroup Uniformity","No demographic subgroup < 0.65 AUROC.",
     None,"Fails for psychiatric_history (71+: 0.189), depression (71+: 0.307), PTSD (F: 0.533)"),
    ("Gate 4","Protocol Stability","Unified screening AUROC within 0.10 of task-specific binary.",
     None,"Fails for chronic_cough (-0.455), ADHD (-0.370), depression (-0.334)"),
]

PROTO_DATA = {
    "Structural / Motor":      [("Parkinson's",0.977,0.883,-0.094),("Airway Stenosis",0.985,0.909,-0.077)],
    "Laryngeal / Vocal":       [("Chronic Cough",0.926,0.470,-0.455),("MTD",0.956,0.773,-0.184)],
    "Psychiatric / Cognitive": [("ADHD",0.953,0.583,-0.370),("Depression",0.961,0.626,-0.334),("PTSD",0.961,0.687,-0.274)],
}

# ── Multi-agent prompts (from noapi_multiagent1.json) ─────────────────────────
_PHARMA_PLAN = """\
You are a clinical pharmacology specialist.
Create a focused research plan for treating the patient based on the highest-confidence diagnosis.

Patient diagnostic profile:
{patient_profile}

Format your response exactly as:

RESEARCH OBJECTIVE:
[Clear statement of research goal]

KEY SEARCH QUERIES:
- 1. Identify first-line and second-line drug treatments for each diagnosis
- 2. Check for drug-drug interactions across the conditions
- 3. Note any drugs that treat multiple conditions simultaneously
- 4. Flag contraindications given the combination of diagnoses
- 5. Weight recommendations by confidence score — higher confidence diseases take priority

SEARCH PRIORITIES:
- Clinical trial databases, FDA labels, pharmacology reviews
- Drug interaction databases (e.g. DrugBank, Lexicomp)
- Voice and neurological condition-specific literature

RULES:
- Output your pharmacological treatment proposal in structured format.
- Do NOT make a definitive diagnosis. Frame all recommendations as "for discussion with a physician."
"""

_PHARMA_EXECUTE = """\
RESEARCH PLAN:
{plan}

You are a clinical pharmacology specialist. Based on this research plan and the evidence gathered:

STEP 1 — Identify first-line and second-line drugs per diagnosis, weighted by confidence score.
STEP 2 — Check for drug-drug interactions across all diagnoses.
STEP 3 — Note any drugs that treat multiple conditions simultaneously.
STEP 4 — Flag contraindications given the combination.

Search results gathered:
{search_results}

RAG context from knowledge base:
{rag_context}

Output in this format:

PROPOSED TREATMENTS:
[First-line and second-line drugs per diagnosis, weighted by confidence score — higher confidence = higher priority]

DRUG INTERACTIONS:
[Any cross-diagnosis interactions or contraindications]

EVIDENCE QUALITY:
[Web source credibility + RAG agreement rating: SUPPORTED / PARTIAL / CONTRADICTED]

CONFLICTS:
[Any web vs RAG disagreements and how you resolved them]

Do NOT make a definitive diagnosis. All recommendations are for physician discussion only.
"""

_PHARMA_SYNTHESIS = """\
You are a Pharma research synthesis expert.

Patient diagnostic profile:
{patient_profile}

Research findings to synthesise:
{findings}

Create a comprehensive synthesis weighted by diagnostic confidence scores.

Format your response as:

EXECUTIVE SUMMARY:
[Key findings, explicitly noting which apply to the highest-confidence diagnosis first]

METHODOLOGY:
- Search Strategy Used
- Sources Analyzed
- Quality Assessment

FINDINGS & ANALYSIS:
[Detailed discussion, organised by diagnosis confidence — highest confidence first]

CONFIDENCE WEIGHTING:
[How findings shift if the lowest-confidence diagnosis turns out to be incorrect]

CONCLUSIONS:
[Main takeaways framed as recommendations for physician discussion]

IMPORTANT: For each major finding include the source link at the end of the sentence where available.
Do NOT make a definitive diagnosis. All recommendations are for physician discussion only.
"""

_GUIDELINES_PLAN = """\
You are a clinical guidelines specialist.
Create a focused research plan for the patient based on the highest-confidence diagnosis.

Patient diagnostic profile:
{patient_profile}

Format your response exactly as:

RESEARCH OBJECTIVE:
[Clear statement of research goal]

KEY SEARCH QUERIES:
- 1. Identify first-line and second-line treatments per guideline body (AAN, ASHA, APA, NICE)
- 2. Flag any conflicts between guidelines for this diagnosis combination
- 3. Specify evidence grades (A/B/C/D) for each recommendation
- 4. Note where the diagnoses create guideline ambiguity

SEARCH PRIORITIES:
- AAN, ASHA, APA, NICE, WHO current guidelines (within 5 years)
- Evidence grade A and B recommendations only
- Voice disorder and psychiatric disorder guideline overlaps

RULES:
- Output your clinical guideline treatment proposal in structured format.
- Do NOT make a definitive diagnosis. Frame all recommendations as "for discussion with a physician."
"""

_GUIDELINES_EXECUTE = """\
RESEARCH PLAN:
{plan}

You are a clinical guidelines specialist. Based on this plan and gathered evidence:

STEP 1 — Retrieve current guidelines (AAN, ASHA, APA, NICE) for each diagnosis.
STEP 2 — Check if guidelines are current (within 5 years). Flag conflicts across diagnoses.
STEP 3 — Verify recommendations against RAG knowledge base.
STEP 4 — Grade evidence level for each recommendation.

Search results:
{search_results}

RAG context:
{rag_context}

Output in this format:

GUIDELINE RECOMMENDATIONS:
[Per diagnosis, with evidence grade A/B/C, weighted by confidence score]

CONFLICTS BETWEEN GUIDELINES:
[Where guidelines for different diagnoses disagree, and how to resolve]

EVIDENCE LEVEL:
[Grade per recommendation — PASS if Grade A or B, FLAG if Grade C only]

RAG VERIFICATION:
[Which recommendations are supported by indexed literature]
"""

_GUIDELINES_SYNTHESIS = """\
You are a Clinical Guidelines research synthesis expert.

Patient diagnostic profile:
{patient_profile}

Research findings to synthesise:
{findings}

Create a comprehensive synthesis weighted by diagnostic confidence scores.

Format your response as:

EXECUTIVE SUMMARY:
[Key findings, explicitly noting which apply to the highest-confidence diagnosis first]

METHODOLOGY:
- Search Strategy Used
- Sources Analyzed
- Quality Assessment

FINDINGS & ANALYSIS:
[Detailed discussion, organised by diagnosis confidence — highest confidence first]

CONFIDENCE WEIGHTING:
[How findings shift if the lowest-confidence diagnosis turns out to be incorrect]

CONCLUSIONS:
[Main takeaways framed as recommendations for physician discussion]

Do NOT make a definitive diagnosis. All recommendations are for physician discussion only.
"""

_REHAB_PLAN = """\
You are a rehabilitation therapy specialist.
Create a focused research plan for the patient based on the highest-confidence diagnosis.

Patient diagnostic profile:
{patient_profile}

Format your response exactly as:

RESEARCH OBJECTIVE:
[Clear statement of research goal]

KEY SEARCH QUERIES:
- 1. Recommend voice and speech therapy interventions (e.g. LSVT for Parkinson's, decide accordingly)
- 2. Recommend behavioral interventions (e.g. CBT modality and frequency for depression/ADHD)
- 3. Recommend occupational therapy and lifestyle interventions
- 4. Sequence recommendations: what to stabilise first vs what to start concurrently
- 5. Flag any rehab conflicts across the diagnoses

SEARCH PRIORITIES:
- Speech-language pathology protocols
- Behavioral medicine and CBT manuals
- Voice rehabilitation evidence base

RULES:
- Output your rehabilitation proposal in structured format.
- Do NOT make a definitive diagnosis. Frame all recommendations as "for discussion with a physician."
"""

_REHAB_EXECUTE = """\
RESEARCH PLAN:
{plan}

You are a rehabilitation and behavioral medicine specialist.

STEP 1 — Search for speech therapy, CBT, occupational therapy, and lifestyle interventions for each diagnosis.
STEP 2 — Check if voice-related and psychiatric interventions are compatible and correctly sequenced.
STEP 3 — Verify interventions against RAG knowledge base.
STEP 4 — Weight by confidence score.

Search results:
{search_results}

RAG context:
{rag_context}

Output in this format:

REHABILITATION PLAN:
[Interventions per diagnosis, sequenced correctly, weighted by confidence score]

SEQUENCING RATIONALE:
[What starts immediately vs what waits for medical stabilisation]

COMPATIBILITY CHECK:
[Any conflicts between voice and psychiatric interventions]

RAG VERIFICATION:
[Which interventions are protocol-supported vs unsupported]
"""

_REHAB_SYNTHESIS = """\
You are a Rehabilitation research synthesis expert.

Patient diagnostic profile:
{patient_profile}

Research findings to synthesise:
{findings}

Format your response as:

EXECUTIVE SUMMARY:
[Key findings, explicitly noting which apply to the highest-confidence diagnosis first]

METHODOLOGY:
- Search Strategy Used
- Sources Analyzed
- Quality Assessment

FINDINGS & ANALYSIS:
[Detailed discussion, organised by diagnosis confidence — highest confidence first]

CONFIDENCE WEIGHTING:
[How findings shift if the lowest-confidence diagnosis turns out to be incorrect]

CONCLUSIONS:
[Main takeaways framed as recommendations for physician discussion]

Do NOT make a definitive diagnosis. All recommendations are for physician discussion only.
"""

_COMORBIDITY_PLAN = """\
You are a comorbidity and polypharmacy specialist.
Analyse the diagnoses as a COMBINATION, not in isolation.

Patient diagnostic profile:
{patient_profile}

Format your response exactly as:

RESEARCH OBJECTIVE:
[Clear statement of research goal]

KEY SEARCH QUERIES:
- 1. Identify emergent risks that only arise from this specific combination
- 2. List treatments contraindicated due to the combination
- 3. Assess how the combination changes clinical urgency and priority order
- 4. Flag polypharmacy risks if standard first-line drugs for each diagnosis are co-prescribed

SEARCH PRIORITIES:
- Multi-morbidity clinical literature
- Polypharmacy interaction databases
- Voice disorder + psychiatric / neurological co-occurrence studies

RULES:
- Output your comorbidity analysis in structured format.
- Do NOT make a definitive diagnosis. Frame all recommendations as "for discussion with a physician."
"""

_COMORBIDITY_EXECUTE = """\
RESEARCH PLAN:
{plan}

You are a multi-morbidity and polypharmacy specialist.

STEP 1 — Search for known interactions between each pair of diagnoses and all together.
STEP 2 — Check for emergent risks only present when all diagnoses coexist.
STEP 3 — Query RAG for comorbidity and polypharmacy evidence.
STEP 4 — Assess treatment burden.

Search results:
{search_results}

RAG context:
{rag_context}

Output in this format:

PAIRWISE INTERACTIONS:
[Risks and implications for each pair of diagnoses]

EMERGENT TRIPLE RISK:
[Risks only present when all diagnoses coexist — state NONE if not found]

TREATMENT BURDEN:
[Total burden assessment — flag if high]

CRITICAL WARNING:
[Single most important risk a physician must know before starting treatment]

RAG VERIFICATION:
[Evidence support level for each major risk finding]
"""

_COMORBIDITY_SYNTHESIS = """\
You are a Comorbidity research synthesis expert.

Patient diagnostic profile:
{patient_profile}

Research findings to synthesise:
{findings}

Format your response as:

EXECUTIVE SUMMARY:
[Key findings, explicitly noting which apply to the highest-confidence diagnosis first]

METHODOLOGY:
- Search Strategy Used
- Sources Analyzed
- Quality Assessment

FINDINGS & ANALYSIS:
[Detailed discussion, organised by diagnosis confidence — highest confidence first]

CONFIDENCE WEIGHTING:
[How findings shift if the lowest-confidence diagnosis turns out to be incorrect]

CONCLUSIONS:
[Main takeaways framed as recommendations for physician discussion]

Do NOT make a definitive diagnosis. All recommendations are for physician discussion only.
"""

_MASTER_PROMPT = """\
You are the master clinical decision support synthesiser.
You have access to four specialist reports and a RAG knowledge base.
You do NOT have web access — work ONLY from the reports and RAG context below.

Patient diagnostic profile:
{patient_profile}

SPECIALIST REPORTS:
--- PHARMACOLOGY ---
{pharma_output}

--- CLINICAL GUIDELINES ---
{guidelines_output}

--- REHABILITATION ---
{rehab_output}

--- COMORBIDITY ---
{comorbidity_output}

RAG KNOWLEDGE BASE CONTEXT:
{rag_context}

Follow this exact sequence:

STEP 1 — BAGGING
Count how many specialist agents recommended each treatment or intervention.
- 3-4 agents: HIGH CONSENSUS
- 2 agents: MODERATE CONSENSUS
- 1 agent: LOW CONSENSUS — verify before including

STEP 2 — EVIDENCE GRADING
For each HIGH CONSENSUS recommendation, check if RAG context supports it.
Mark each: SUPPORTED / PARTIAL / CONTRADICTED

STEP 3 — CONFLICT RESOLUTION
Where specialist agents disagree, use RAG evidence to arbitrate.
State explicitly which recommendation you kept and why.

STEP 4 — CONFIDENCE WEIGHTING
Adjust all recommendations by diagnostic confidence from the patient profile.
Higher confidence diagnosis = higher priority treatment.

STEP 5 — GENERATE FINAL REPORT in this exact format:

═══════════════════════════════════════
CLINICAL DECISION SUPPORT REPORT
VoxClinBench · MARVEL Voice AI · Research Grade
═══════════════════════════════════════

PATIENT PROFILE:
[Restate the diagnoses and confidence scores]

PRIMARY RECOMMENDATIONS (high consensus + literature-supported):
[Treatment | Evidence: SUPPORTED/PARTIAL | Consensus: X/4 agents | Priority: HIGH/MEDIUM]

SECONDARY RECOMMENDATIONS (moderate consensus or conditional):
[Treatment | Evidence level | Condition: only if [diagnosis] confirmed]

FLAGGED ITEMS (low consensus or contradicted — discuss with physician):
[Item | Reason for flag]

COMORBIDITY WARNINGS:
[Critical risks from treating all conditions simultaneously]

CONFIDENCE CAVEAT:
[What changes in this plan if the lowest-confidence diagnosis is incorrect]

═══════════════════════════════════════
DISCLAIMER: This report is generated by a research-grade AI clinical decision support system
built on the Bridge2AI-Voice MARVEL model. NOT a medical device. NOT FDA-cleared.
NOT for diagnostic or clinical decision-making. All recommendations must be reviewed and
approved by a licensed physician before any clinical action is taken.
═══════════════════════════════════════
"""


st.markdown("""
<style>

/* 🔴 OUTER sticky container (this is the black bar) */
div[data-testid="stChatInput"] {
    background-color: white !important;
    border-top: 1px solid #e2e8f0 !important;
}

/* Inner wrapper */
div[data-testid="stChatInput"] > div {
    background-color: white !important;
}

/* Actual text area */
div[data-testid="stChatInput"] textarea {
    background-color: white !important;
    color: black !important;
}

/* Placeholder */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}

/* Send button (arrow icon area) */
div[data-testid="stChatInput"] button {
    background-color: #f1f5f9 !important;
    color: black !important;
    border-radius: 8px !important;
}

/* Remove any dark overlay */
div[data-testid="stChatInput"] * {
    background-color: transparent;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Chat input container */
div[data-testid="stChatInput"] {
    background-color: white !important;
    border-radius: 10px;
}

/* Actual input field */
div[data-testid="stChatInput"] textarea {
    background-color: white !important;
    color: black !important;
}

/* Placeholder text */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #555 !important;
}

/* Remove dark mode styling artifacts */
div[data-testid="stChatInput"] * {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _generate_pdf_report(
    patient_profile: str,
    agent_reports: dict,
    master_report: str,
    patient: dict,
    predictions: list,
    task_family: str,
) -> bytes:
    """Generate a two-part PDF: Part 1 for patient, Part 2 for doctor."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor, black, white
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable, PageBreak)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        import io as _io

        buf = _io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        teal   = HexColor("#028090")
        dark   = HexColor("#0d2233")
        gray   = HexColor("#6b7280")
        light  = HexColor("#f0fdfa")
        warn   = HexColor("#fffbeb")

        h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                             textColor=dark, fontSize=18, spaceAfter=6, fontName="Helvetica-Bold")
        h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                             textColor=teal, fontSize=13, spaceAfter=4, fontName="Helvetica-Bold")
        h3 = ParagraphStyle("H3", parent=styles["Heading3"],
                             textColor=dark, fontSize=11, spaceAfter=3, fontName="Helvetica-Bold")
        body = ParagraphStyle("Body", parent=styles["Normal"],
                               fontSize=10, leading=15, spaceAfter=6, textColor=HexColor("#1e293b"))
        small = ParagraphStyle("Small", parent=styles["Normal"],
                                fontSize=8.5, leading=13, textColor=gray)
        disclaimer_style = ParagraphStyle("Disc", parent=styles["Normal"],
                                           fontSize=8, leading=12, textColor=HexColor("#881337"),
                                           backColor=HexColor("#fff1f2"), borderPadding=6)

        story = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        uid = patient.get("uid", "N/A")

        # ── COVER ─────────────────────────────────────────────────────────────
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph("VoxClinBench", ParagraphStyle("Cover1", parent=styles["Normal"],
            fontSize=26, fontName="Helvetica-Bold", textColor=teal, alignment=TA_CENTER)))
        story.append(Paragraph("Clinical AI Decision Support Report",
            ParagraphStyle("Cover2", parent=styles["Normal"], fontSize=14,
                           textColor=dark, alignment=TA_CENTER, spaceAfter=4)))
        story.append(Paragraph(f"Bridge2AI-Voice · MARVEL Model · Research Grade",
            ParagraphStyle("Cover3", parent=styles["Normal"], fontSize=10,
                           textColor=gray, alignment=TA_CENTER, spaceAfter=12)))
        story.append(HRFlowable(width="100%", thickness=2, color=teal, spaceAfter=12))
        story.append(Paragraph(f"Patient ID: {uid} &nbsp;|&nbsp; Generated: {ts}",
            ParagraphStyle("CoverInfo", parent=styles["Normal"], fontSize=9,
                           textColor=gray, alignment=TA_CENTER, spaceAfter=8)))
        story.append(Paragraph("NOT FOR CLINICAL USE · RESEARCH GRADE · NOT FDA-CLEARED",
            ParagraphStyle("CoverWarn", parent=styles["Normal"], fontSize=9,
                           textColor=HexColor("#dc2626"), alignment=TA_CENTER,
                           fontName="Helvetica-Bold")))
        story.append(Spacer(1, 1*cm))

        # ═════════════════════════════════════════════════════════════════════
        # PART 1 — PATIENT SUMMARY
        # ═════════════════════════════════════════════════════════════════════
        story.append(HRFlowable(width="100%", thickness=3, color=teal, spaceBefore=6, spaceAfter=12))
        story.append(Paragraph("PART 1 — Patient Summary",
            ParagraphStyle("PartHead", parent=styles["Normal"], fontSize=16,
                           fontName="Helvetica-Bold", textColor=dark, spaceAfter=4)))
        story.append(Paragraph("Prepared for the patient · Plain-language overview",
            ParagraphStyle("PartSub", parent=styles["Normal"], fontSize=10,
                           textColor=gray, spaceAfter=14)))

        # Patient info table
        p_rows = [
            ["Name", patient.get("name","—"), "Age", str(patient.get("age","—"))],
            ["Sex", patient.get("sex","—"), "Country", patient.get("country","—")],
            ["Ethnicity", patient.get("ethnicity","—"), "Task Battery", task_family],
            ["Primary Complaint", patient.get("complaint","—"), "Medications", patient.get("medications","None listed")],
        ]
        pt = Table(p_rows, colWidths=[3.5*cm, 6.5*cm, 3.5*cm, 4*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), HexColor("#f0fdfa")),
            ("BACKGROUND", (2,0), (2,-1), HexColor("#f0fdfa")),
            ("TEXTCOLOR", (0,0), (-1,-1), HexColor("#1e293b")),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, HexColor("#e5e7eb")),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [white, HexColor("#f9fafb")]),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.5*cm))

        # Top 3 findings for patient
        story.append(Paragraph("What the Voice AI Found", h2))
        story.append(Paragraph(
            "The MARVEL voice AI analyzed your voice recording and identified acoustic patterns "
            "associated with the following conditions. These are <b>research screening signals only</b> — "
            "they are not a diagnosis. Please discuss these results with your doctor.",
            body))
        story.append(Spacer(1, 0.3*cm))

        top3 = sorted(predictions, key=lambda x: x.get("probability",0), reverse=True)[:3]
        for rank, pred in enumerate(top3, 1):
            task = pred.get("task","?").replace("_"," ").title()
            pct  = pred.get("percent", round(pred.get("probability",0)*100,1))
            conf = "High" if pct >= 70 else ("Moderate" if pct >= 45 else "Low")
            conf_color = HexColor("#059669") if pct>=70 else (HexColor("#d97706") if pct>=45 else HexColor("#9ca3af"))
            rank_labels = {1:"Primary signal", 2:"Secondary signal", 3:"Tertiary signal"}
            data = [[f"#{rank} {rank_labels.get(rank,'')}",
                     f"{task}",
                     f"{pct:.0f}%",
                     f"{conf} confidence"]]
            t = Table(data, colWidths=[3.5*cm, 8*cm, 2*cm, 4*cm])
            border_color = [HexColor("#028090"), HexColor("#0369a1"), HexColor("#7c3aed")][rank-1]
            t.setStyle(TableStyle([
                ("FONTNAME", (0,0), (0,0), "Helvetica-Bold"),
                ("FONTNAME", (1,0), (1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 10),
                ("TEXTCOLOR", (0,0), (-1,-1), HexColor("#1e293b")),
                ("TEXTCOLOR", (2,0), (2,0), border_color),
                ("TEXTCOLOR", (3,0), (3,0), conf_color),
                ("FONTNAME", (2,0), (2,0), "Helvetica-Bold"),
                ("LEFTPADDING", (0,0), (0,0), 10),
                ("BOX", (0,0), (-1,-1), 0.5, HexColor("#e5e7eb")),
                ("LINEBEFORE", (0,0), (0,-1), 3, border_color),
                ("ROWBACKGROUNDS", (0,0), (-1,-1), [HexColor("#f9fafb")]),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                ("LEFTPADDING", (1,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.15*cm))

        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("What Should I Do Next?", h2))
        story.append(Paragraph(
            "This report is a <b>research screening tool</b>, not a medical diagnosis. "
            "You should <b>share this report with your doctor</b> and ask about next steps. "
            "Your doctor may recommend further tests, specialist referral, or monitoring. "
            "Do not change or stop any medications based on this report.",
            body))

        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(
            "⚠ IMPORTANT: This report is generated by a research AI and is NOT a medical device, "
            "NOT FDA-cleared, and NOT intended for clinical decision-making. "
            "All findings must be reviewed by a licensed physician before any action is taken.",
            ParagraphStyle("PatientDisc", parent=styles["Normal"],
                           fontSize=8.5, leading=13, textColor=HexColor("#7c2d12"),
                           backColor=HexColor("#fff7ed"), borderPadding=8,
                           borderColor=HexColor("#fed7aa"), borderWidth=1)))

        # ═════════════════════════════════════════════════════════════════════
        # PART 2 — CLINICAL REPORT (DOCTOR)
        # ═════════════════════════════════════════════════════════════════════
        story.append(PageBreak())
        story.append(HRFlowable(width="100%", thickness=3, color=teal, spaceBefore=6, spaceAfter=12))
        story.append(Paragraph("PART 2 — Clinical Decision Support Report",
            ParagraphStyle("PartHead2", parent=styles["Normal"], fontSize=16,
                           fontName="Helvetica-Bold", textColor=dark, spaceAfter=4)))
        story.append(Paragraph("Prepared for the treating physician · Technical clinical detail",
            ParagraphStyle("PartSub2", parent=styles["Normal"], fontSize=10,
                           textColor=gray, spaceAfter=14)))

        story.append(Paragraph(
            "NOT FOR DIAGNOSTIC USE · RESEARCH GRADE · Review required before clinical action",
            ParagraphStyle("ClinWarn", parent=styles["Normal"], fontSize=9,
                           textColor=HexColor("#dc2626"), fontName="Helvetica-Bold",
                           backColor=HexColor("#fff1f2"), spaceAfter=12, borderPadding=4)))

        # Full predictions table
        story.append(Paragraph("MARVEL Model Predictions (all tasks)", h2))
        pred_header = [["Disease", "Probability", "Confidence", "Confounder Prob", "Gap", "Flag"]]
        pred_data = []
        for p in sorted(predictions, key=lambda x: x.get("probability",0), reverse=True)[:10]:
            task = p.get("task","?").replace("_"," ").title()
            prob = p.get("percent", round(p.get("probability",0)*100,1))
            conf = "High" if prob>=70 else ("Mod" if prob>=45 else "Low")
            cp   = p.get("confounder_percent", round(p.get("confounder_prob",0)*100,1) if p.get("confounder_prob") else "—")
            gap  = p.get("gap")
            gap_s = f"{gap:+.3f}" if gap is not None else "—"
            flag = ("⚠ Shortcut risk" if gap is not None and 0 <= gap < 0.10
                    else ("✗ Confound exceeds" if gap is not None and gap < 0 else "✓ OK"))
            pred_data.append([task, f"{prob:.1f}%", conf, f"{cp}%" if cp!="—" else "—", gap_s, flag])
        full_pred_table = Table(pred_header + pred_data,
                                 colWidths=[5*cm, 2.5*cm, 2*cm, 3*cm, 2*cm, 3.5*cm])
        full_pred_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), dark),
            ("TEXTCOLOR", (0,0), (-1,0), white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, HexColor("#f9fafb")]),
            ("GRID", (0,0), (-1,-1), 0.4, HexColor("#e5e7eb")),
            ("ALIGN", (1,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(full_pred_table)
        story.append(Spacer(1, 0.6*cm))

        # Master report
        if master_report:
            story.append(Paragraph("Master Synthesiser — Integrated Clinical Report", h2))
            for line in master_report.split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.15*cm))
                elif line.startswith("═"):
                    story.append(HRFlowable(width="100%", thickness=1.5, color=teal, spaceBefore=4, spaceAfter=4))
                elif line.isupper() and len(line) < 60 and ":" not in line:
                    story.append(Paragraph(line, h3))
                elif line.endswith(":") and len(line) < 60:
                    story.append(Paragraph(line, h3))
                else:
                    story.append(Paragraph(line, body))
        story.append(Spacer(1, 0.6*cm))

        # Per-agent summaries
        agent_icons = {"Pharmacology":"💊", "Clinical Guidelines":"📋",
                       "Rehabilitation":"🏃", "Comorbidity":"⚠️"}
        for role, report in agent_reports.items():
            if not report or report.startswith("["):
                continue
            story.append(Paragraph(f"{agent_icons.get(role,'')} {role} Agent Report", h2))
            lines = report.split("\n")[:60]  # limit per agent in PDF
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.1*cm))
                elif line.isupper() and len(line) < 60:
                    story.append(Paragraph(line, h3))
                elif line.endswith(":") and len(line) < 60:
                    story.append(Paragraph(line, h3))
                else:
                    story.append(Paragraph(line[:500], body))
            story.append(Spacer(1, 0.4*cm))

        # Footer disclaimer
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e5e7eb"), spaceAfter=8))
        story.append(Paragraph(
            "Research-grade screening output from the Bridge2AI-Voice MARVEL model. "
            "NOT a medical device. NOT FDA-cleared. NOT for diagnostic or clinical decision-making. "
            "All recommendations must be reviewed and approved by a licensed physician before any clinical action is taken.",
            ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7.5,
                           textColor=gray, alignment=TA_CENTER)))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        # Fallback: plain text PDF using basic approach
        return master_report.encode("utf-8")
    except Exception as e:
        return f"PDF generation error: {e}\n\n{master_report}".encode("utf-8")


def _generate_agent_pdf(role: str, report: str, patient: dict) -> bytes:
    """Generate a single-agent PDF report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor, white
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.enums import TA_CENTER
        import io as _io

        buf = _io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        teal = HexColor("#028090")
        dark = HexColor("#0d2233")
        gray = HexColor("#6b7280")

        h1 = ParagraphStyle("H1", parent=styles["Normal"], fontSize=16,
                             fontName="Helvetica-Bold", textColor=dark, spaceAfter=6)
        h2 = ParagraphStyle("H2", parent=styles["Normal"], fontSize=12,
                             fontName="Helvetica-Bold", textColor=teal, spaceAfter=4)
        body = ParagraphStyle("Body", parent=styles["Normal"],
                               fontSize=10, leading=15, spaceAfter=5,
                               textColor=HexColor("#1e293b"))

        story = []
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M")
        uid = patient.get("uid", "N/A")
        agent_icons = {"Pharmacology":"💊","Clinical Guidelines":"📋",
                       "Rehabilitation":"🏃","Comorbidity":"⚠️","Master Synthesiser":"🧠"}
        icon = agent_icons.get(role, "🔬")

        story.append(Paragraph(f"VoxClinBench — {icon} {role} Report", h1))
        story.append(Paragraph(f"Patient: {uid} · {ts} · Research Grade",
            ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=gray, spaceAfter=10)))
        story.append(HRFlowable(width="100%", thickness=2, color=teal, spaceAfter=12))

        for line in report.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.12*cm))
            elif line.isupper() and len(line) < 60 and ":" not in line:
                story.append(Paragraph(line, h2))
            elif line.endswith(":") and len(line) < 60:
                story.append(Paragraph(line, h2))
            else:
                story.append(Paragraph(line[:600], body))

        story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e5e7eb"), spaceBefore=12, spaceAfter=6))
        story.append(Paragraph(
            "Research-grade output. NOT a medical device. NOT for clinical decision-making.",
            ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7.5,
                           textColor=gray, alignment=TA_CENTER)))

        doc.build(story)
        return buf.getvalue()
    except Exception as e:
        return f"PDF error: {e}\n\n{report}".encode("utf-8")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoxClinBench · Voice AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #ffffff; }
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none !important; }
  .block-container { max-width:1100px !important; padding-top:0 !important; }

  /* ── Nav bar ─────────────────────────────────────────────────────── */
  .top-nav {
    background: #0d2233; padding: 0.65rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    margin: -1rem -1rem 0 -1rem; border-bottom: 2px solid #028090;
    flex-wrap: nowrap; gap: 0.5rem;
    position: relative; z-index: 100;
  }
  /* Ensure stApp white bg doesn't bleed into nav */
  .block-container { background: #ffffff; }
  .nav-brand { color: #00a896; font-weight: 700; font-size: 1.05rem; letter-spacing: 0.02em; }
  .nav-sub   { color: #8ba6b8; font-size: 0.75rem; }
  .nav-badge   { background: rgba(2,128,144,0.25); color: #a7f3d0; padding: 0.22rem 0.9rem;
                 border-radius: 20px; font-size: 0.73rem; font-weight: 600; border:1px solid rgba(167,243,208,.3);
                 white-space: nowrap; }
  .nav-api-ok  { background: #022c22; color: #6ee7b7; padding: 0.22rem 0.9rem;
                 border-radius: 20px; font-size: 0.73rem; font-weight: 600; margin-left:0.4rem;
                 border:1px solid #064e3b; white-space: nowrap; }
  .nav-api-off { background: #450a0a; color: #fca5a5; padding: 0.22rem 0.9rem;
                 border-radius: 20px; font-size: 0.73rem; font-weight: 600; margin-left:0.4rem;
                 border:1px solid #7f1d1d; white-space: nowrap; }
  /* Nav page tabs — wider, more readable */
  .nav-page-btn {
    color: #8ba6b8 !important; font-size: 0.84rem; font-weight: 500; padding: 0.35rem 1.1rem;
    border-radius: 6px; cursor: default; white-space: nowrap; transition: all 0.15s;
    border: 1px solid transparent;
  }
  .nav-page-btn.active {
    color: #00a896 !important; background: rgba(2,128,144,0.18); border-color: rgba(0,168,150,0.4);
    font-weight: 700;
  }

  /* ── Stepper ─────────────────────────────────────────────────────── */
  .stepper { display:flex; align-items:center; justify-content:center; padding:1.1rem 0 0.3rem; }
  .step { display:flex; flex-direction:column; align-items:center; gap:0.3rem; min-width:110px; }
  .step-circle { width:34px; height:34px; border-radius:50%; display:flex; align-items:center;
                 justify-content:center; font-weight:700; font-size:0.85rem; border:2px solid; }
  .step-circle.done    { background:#028090; border-color:#028090; color:white; }
  .step-circle.active  { background:white; border-color:#028090; color:#028090; }
  .step-circle.pending { background:white; border-color:#d1d5db; color:#9ca3af; }
  .step-label { font-size:0.68rem; font-weight:500; text-align:center; }
  .step-label.done    { color:#028090; }
  .step-label.active  { color:#028090; font-weight:700; }
  .step-label.pending { color:#9ca3af; }
  .step-line { flex:1; height:2px; margin-bottom:20px; min-width:36px; }
  .step-line.done    { background:#028090; }
  .step-line.pending { background:#e5e7eb; }

  /* ── Cards ────────────────────────────────────────────────────────── */
  .card { background:white; border:1px solid #e5e7eb; border-radius:10px;
          padding:1.5rem; margin-bottom:1rem; }
  .card-sm { background:white; border:1px solid #e5e7eb; border-radius:8px;
             padding:1rem; margin-bottom:0.8rem; }

  /* ── Typography ──────────────────────────────────────────────────── */
  .section-title { font-size:1.25rem; font-weight:700; color:#111827 !important; margin-bottom:0.15rem; }
  .section-sub   { font-size:0.83rem; color:#4b5563 !important; margin-bottom:1.2rem; }
  .teal-line     { width:48px; height:3px; background:#028090; border-radius:2px; margin-bottom:1rem; }
  .info-pill { display:inline-block; background:#f0fdfa; color:#028090;
               border:1px solid #99f6e4; border-radius:20px; padding:0.15rem 0.65rem;
               font-size:0.72rem; font-weight:600; margin:0.15rem 0.15rem 0 0; }

  /* ── Alert boxes ─────────────────────────────────────────────────── */
  .warn-box  { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px;
               padding:0.65rem 0.9rem; font-size:0.82rem; color:#78350f; margin-top:0.8rem; }
  .note-box  { background:#f0fdfa; border:1px solid #99f6e4; border-radius:8px;
               padding:0.65rem 0.9rem; font-size:0.82rem; color:#134e4a; margin-top:0.7rem; }
  .error-box { background:#fff1f2; border:1px solid #fecdd3; border-radius:8px;
               padding:0.65rem 0.9rem; font-size:0.82rem; color:#881337; margin-top:0.7rem; }
  .na-box    { background:#faf5ff; border:1px solid #d8b4fe; border-radius:8px;
               padding:0.65rem 0.9rem; font-size:0.82rem; color:#5b21b6; margin-top:0.7rem; }

  /* ── Buttons ─────────────────────────────────────────────────────── */
  .block-container div.stButton > button {
    background:#028090 !important; color:white !important;
    border:none !important; border-radius:7px !important; font-weight:600 !important;
    padding:0.5rem 1.6rem !important; font-size:0.88rem !important;
    transition: background 0.15s; }
  .block-container div.stButton > button:hover { background:#01636f !important; }
  .block-container div.stButton > button[kind="secondary"] {
    background:white !important; color:#374151 !important;
    border:1px solid #d1d5db !important; }
  .block-container div.stButton > button[kind="secondary"]:hover {
    background:#f9fafb !important; }

  /* ── Audit flags ─────────────────────────────────────────────────── */
  .audit-flag  { font-size:0.78rem; padding:0.3rem 0.55rem; border-radius:5px; margin-bottom:0.35rem; display:block; }
  .audit-warn  { background:#fff1f2; color:#9f1239; border:1px solid #fecdd3; }
  .audit-ok    { background:#f0fdf4; color:#14532d; border:1px solid #bbf7d0; }
  .audit-amber { background:#fffbeb; color:#78350f; border:1px solid #fde68a; }
  .audit-na    { background:#faf5ff; color:#5b21b6; border:1px solid #d8b4fe; }

  /* ── Top-3 prediction cards ──────────────────────────────────────── */
  .pred-card { border:1px solid #e5e7eb; border-radius:10px; padding:1.1rem 1.3rem;
               margin-bottom:0.7rem; background:white; position:relative; overflow:hidden; }
  .pred-card-1 { border-left:4px solid #028090; }
  .pred-card-2 { border-left:4px solid #0369a1; }
  .pred-card-3 { border-left:4px solid #7c3aed; }
  .pred-rank   { font-size:0.68rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:0.08em; color:#9ca3af; margin-bottom:0.2rem; }
  .pred-task   { font-size:1.05rem; font-weight:700; color:#111827; margin-bottom:0.15rem; }
  .pred-pct    { font-size:1.9rem; font-weight:800; line-height:1; }
  .pred-label  { font-size:0.74rem; font-weight:600; margin-top:0.1rem; }
  .pred-bar-track { background:#f3f4f6; border-radius:4px; height:6px; margin:0.5rem 0 0.3rem; }
  .pred-bar-fill  { height:6px; border-radius:4px; transition:width 0.4s ease; }
  .pred-conf   { font-size:0.72rem; color:#6b7280; margin-top:0.3rem; }
  .pred-gap-ok   { color:#059669; font-weight:600; }
  .pred-gap-warn { color:#d97706; font-weight:600; }
  .pred-gap-err  { color:#dc2626; font-weight:600; }

  /* ── Subgroup table ──────────────────────────────────────────────── */
  .sg-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
  .sg-table th { background:#f9fafb; color:#374151; font-weight:600;
    padding:0.4rem 0.55rem; text-align:left; border-bottom:1px solid #e5e7eb; }
  .sg-table td { padding:0.35rem 0.55rem; border-bottom:1px solid #f3f4f6; color:#111827; }
  .sg-ok   { color:#059669; font-weight:600; }
  .sg-warn { color:#dc2626; font-weight:600; }
  .sg-mid  { color:#d97706; font-weight:600; }

  /* ── Waveform ─────────────────────────────────────────────────────── */
  [data-testid="stFileUploadDropzone"] {
    border:2px dashed #028090 !important; background:#f0fdfa !important; border-radius:8px !important; }
  @keyframes waveanim { from{height:4px;opacity:.3} to{height:var(--h);opacity:1} }
  .waveform { display:flex; align-items:center; justify-content:center; gap:3px; height:44px;
              background:#f0fdfa; border:1px solid #99f6e4; border-radius:8px;
              padding:0 1rem; margin:0.5rem 0; }
  .wbar { width:4px; border-radius:3px; background:#028090;
          animation:waveanim var(--dur) ease-in-out infinite alternate; }

  /* ── Sidebar (dark, matches nav brand) ───────────────────────────── */
  section[data-testid="stSidebar"] { background:#0d2233 !important;
    border-right:1px solid #1e3a50 !important;
    min-width:270px !important; max-width:270px !important; }
  section[data-testid="stSidebar"] > div { padding:0 12px !important; overflow-x:hidden; }
  section[data-testid="stSidebar"] .stButton > button {
    width:100% !important; background:#1e3a50 !important;
    border:1px solid #2a5060 !important; color:#9db4c0 !important;
    border-radius:5px !important; font-size:0.78rem !important; padding:6px 8px !important; }
  section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background:#028090 !important; color:white !important; border-color:#028090 !important; }
  section[data-testid="stSidebar"] .stSelectbox > div > div,
  section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
    background:#1e3a50 !important; border:1px solid #2a5060 !important;
    color:#e2e8f0 !important; border-radius:5px !important; }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label { color:#9db4c0 !important; font-size:0.78rem !important; }
  section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label { color:#9db4c0 !important; font-size:0.8rem !important; }
  section[data-testid="stSidebar"] hr { border-color:#1e3a50 !important; margin:10px 0 !important; }
  section[data-testid="stSidebar"] div[data-testid="stSlider"] label { color:#9db4c0 !important; font-size:0.76rem !important; }

  /* ── RAG Dev ─────────────────────────────────────────────────────── */
  .chunk-preview { background:#f9fafb; border-left:3px solid #028090; padding:0.6rem 0.8rem;
                   border-radius:0 6px 6px 0; margin:0.35rem 0; font-size:0.76rem; color:#374151;
                   white-space:pre-wrap; font-family:monospace; line-height:1.5; }
  .badge { display:inline-block; background:#f0fdfa; color:#028090; padding:1px 6px;
           border-radius:3px; font-size:0.64rem; font-weight:500; margin-right:2px;
           border:1px solid #99f6e4; font-family:monospace; }
  .badge-type  { background:#faf5ff; color:#7c3aed; border-color:#ddd6fe; }
  .badge-url   { background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe; }
  .badge-score { background:#f0fdf4; color:#059669; border-color:#bbf7d0; }
  .paper-card { background:white; border:1px solid #e5e7eb; border-radius:8px;
                padding:10px 12px; margin:4px 0; }
  .paper-card .ptitle { font-weight:600; font-size:0.82rem; color:#111827; margin-bottom:3px; }
  .paper-card .purl   { font-size:0.69rem; color:#028090; }
  .paper-card .pscore { font-size:0.65rem; color:#9ca3af; margin-top:2px; font-family:monospace; }

  /* ── Clinical AI agents ──────────────────────────────────────────── */
  .agent-card { background:white; border:1px solid #e5e7eb; border-radius:8px;
                padding:0.8rem 1rem; margin-bottom:0.4rem; }
  .agent-header { font-weight:600; color:#111827 !important; font-size:0.88rem; margin-bottom:0.1rem; }
  .agent-status { font-size:0.74rem; color:#4b5563 !important; }
  .agent-done { border-color:#028090; background:#f0fdfa; }
  .agent-done .agent-header { color:#065a4a !important; }
  .agent-run  { border-left:3px solid #028090; }
  .report-card { background:white; border:1px solid #e5e7eb; border-radius:10px;
                 padding:1.4rem; margin-top:0.8rem; overflow:hidden; }
  .report-card pre { background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
                     padding:0.9rem; font-size:0.8rem; color:#111827 !important; white-space:pre-wrap;
                     word-break:break-word; overflow-x:auto; line-height:1.6;
                     font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace; }

  /* ── Chat input ──────────────────────────────────────────────────── */
  div[data-testid="stChatInput"] { position:fixed !important; bottom:0 !important;
    left:calc(50% + 135px) !important; transform:translateX(-50%) !important;
    width:min(900px,calc(100vw - 290px)) !important; background:#f8fafc !important;
    padding:12px 0 24px 0 !important; z-index:999 !important;
    border-top:2px solid #028090 !important; }
  div[data-testid="stChatInput"] > div { background:#1e3a50 !important;
    border:1px solid #2a5060 !important; border-radius:12px !important;
    padding:12px 18px !important; min-height:56px !important; }
  div[data-testid="stChatInput"] > div:focus-within { border-color:#028090 !important; box-shadow:0 0 0 2px rgba(2,128,144,.25) !important; }
  div[data-testid="stChatInput"] textarea { color:#e2e8f0 !important; font-size:0.93rem !important; background:transparent !important; }
  div[data-testid="stChatInput"] textarea::placeholder { color:#7a9ab0 !important; }
  div[data-testid="stChatInput"] button { background:#028090 !important; border-radius:8px !important; }
  /* Chat messages */
  div[data-testid="stChatMessage"] { border:1px solid #e5e7eb !important;
    border-radius:10px !important; padding:12px 16px !important; margin:6px 0 !important;
    border-bottom:none !important; }
  div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background:#f0fdfa !important; border-color:#99f6e4 !important; }
  div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background:#fafafa !important; border-color:#e5e7eb !important; }

  /* ── Streamlit native alert overrides ────────────────────────────── */
  div[data-testid="stSuccess"] { background:#f0fdf4 !important; border:1px solid #bbf7d0 !important;
    color:#14532d !important; border-radius:6px !important; }
  div[data-testid="stError"]   { background:#fff1f2 !important; border:1px solid #fecdd3 !important;
    color:#881337 !important; border-radius:6px !important; }
  div[data-testid="stInfo"]    { background:#f0fdfa !important; border:1px solid #99f6e4 !important;
    color:#134e4a !important; border-radius:6px !important; }
  div[data-testid="stWarning"] { background:#fffbeb !important; border:1px solid #fde68a !important;
    color:#78350f !important; border-radius:6px !important; }
  div[data-testid="stMetric"] { background:#f9fafb !important; border:1px solid #e5e7eb !important;
    border-radius:8px !important; padding:10px !important; }
  ::-webkit-scrollbar { width:5px; height:5px; }
  ::-webkit-scrollbar-track { background:#f9fafb; }
  ::-webkit-scrollbar-thumb { background:#d1d5db; border-radius:3px; }

  /* ── Fix white-on-white text globally ────────────────────────────── */
  .stMarkdown, .stMarkdown p, .stMarkdown div, p, div { color: inherit; }
  .stApp .stMarkdown { color: #1e293b; }
  /* Ensure all text in main area is dark */
  section.main .stMarkdown, section.main p { color: #1e293b !important; }
  /* Card text */
  .card, .card-sm { color: #1e293b; }
  .card p, .card div, .card-sm p, .card-sm div { color: #1e293b; }
  /* Expander text */
  [data-testid="stExpander"] summary { color: #111827 !important; font-weight: 600; }
  [data-testid="stExpander"] [data-testid="stMarkdownContainer"] { color: #1e293b; }
  /* Remove random floating boxes from Streamlit internals */
  [data-testid="stVerticalBlock"] > div:empty { display: none; }
  /* Fix input labels */
  .stTextInput label, .stSelectbox label, .stNumberInput label,
  .stTextArea label, .stFileUploader label, .stCheckbox label,
  .stSlider label { color: #374151 !important; font-size: 0.88rem !important; }
  /* Fix expander label */
  .streamlit-expanderHeader { color: #111827 !important; }
  /* Fix selectbox text */
  .stSelectbox > div > div { color: #111827 !important; }

  /* ── Main area inputs — dark fill, white text ─────────────────── */
  section.main div[data-testid="stTextInput"] input,
  section.main div[data-testid="stNumberInput"] input {
    background: #1e3a50 !important; color: #e2e8f0 !important;
    border: 1px solid #2a5060 !important; border-radius: 6px !important; }
  section.main div[data-testid="stTextArea"] textarea {
    background: #1e3a50 !important; color: #e2e8f0 !important;
    border: 1px solid #2a5060 !important; border-radius: 6px !important; }
  section.main div[data-testid="stTextInput"] input::placeholder,
  section.main div[data-testid="stNumberInput"] input::placeholder,
  section.main div[data-testid="stTextArea"] textarea::placeholder {
    color: #7a9ab0 !important; }
  section.main .stSelectbox > div > div {
    background: #1e3a50 !important; color: #e2e8f0 !important;
    border: 1px solid #2a5060 !important; border-radius: 6px !important; }
  /* Selectbox dropdown options */
  [data-baseweb="popover"] ul li { background: #1e3a50 !important; color: #e2e8f0 !important; }
  [data-baseweb="popover"] ul li:hover { background: #023e52 !important; }
  /* Metric labels */
  [data-testid="stMetricLabel"] { color: #6b7280 !important; }
  [data-testid="stMetricValue"] { color: #111827 !important; }
  /* Tab labels */
  .stTabs [data-baseweb="tab"] { color: #4b5563 !important; }
  .stTabs [aria-selected="true"] { color: #028090 !important; font-weight: 600 !important; }
  /* Caption text */
  .stCaption, small { color: #6b7280 !important; }
  /* Download button text */
  [data-testid="stDownloadButton"] button { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Persistence paths ─────────────────────────────────────────────────────────
_CHAT_PATH    = Path("/tmp/vox_chat.json")
_INDEXED_PATH = Path("/tmp/vox_indexed.json")

def _save_chat():
    try:
        _CHAT_PATH.write_text(
            json.dumps([{"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history]),
            encoding="utf-8",
        )
    except Exception:
        pass

def _load_chat():
    try:
        if _CHAT_PATH.exists():
            return json.loads(_CHAT_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def _save_indexed():
    try:
        _INDEXED_PATH.write_text(
            json.dumps({"files": st.session_state.indexed_files,
                        "url_map": st.session_state.file_url_map}),
            encoding="utf-8",
        )
    except Exception:
        pass

def _load_indexed():
    try:
        if _INDEXED_PATH.exists():
            d = json.loads(_INDEXED_PATH.read_text(encoding="utf-8"))
            return d.get("files", []), d.get("url_map", {})
    except Exception:
        pass
    return [], {}

_idx_files, _idx_urls = _load_indexed()

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    _defaults = {
        # ── OG fields ──────────────────────────────────────────────────────
        "stage": 1,
        "patient": {},
        "task_family": None,
        "tasks_done": [],
        "task_inputs": {},
        "results": None,
        "uid": f"UID-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}",
        "protocol": "B",
        "api_response": None,
        "session_log": [],
        # ── Page routing ───────────────────────────────────────────────────
        "page": "assessment",   # "assessment" | "chat" | "rag_dev"
        # ── RAG / indexer ─────────────────────────────────────────────────
        "indexer": None,
        "indexed_files": _idx_files,
        "file_url_map": _idx_urls,
        "index_stats": {},
        "file_assignments": {},
        "custom_paper_types": [],
        # ── Sidebar tabs ──────────────────────────────────────────────────
        "sidebar_tab": "rag",
        # ── RAG config ────────────────────────────────────────────────────
        "cfg_model":         "gemini-2.5-flash",
        "cfg_top_k":         4,
        "cfg_iterations":    2,
        "cfg_word_count":    500,
        "cfg_language":      "English",
        "cfg_index_type":    "HNSW",
        "cfg_chunk_size":    400,
        "cfg_chunk_overlap": 50,
        "cfg_embed_model":   "BGE (local · 1024-dim)",
        "cfg_gemini_api_key": "",
        "cfg_use_gateway":   False,
        "cfg_gateway_key":   "",
        "cfg_gateway_url":   os.environ.get("AI_GATEWAY_URL", ""),
        "web_search_enabled": False,
        # ── Agent / chat ──────────────────────────────────────────────────
        "chat_history":       _load_chat(),
        "agent_results":      {},   # {"pharma": "...", "guidelines": "...", ...}
        "final_report":       "",
        "analysis_running":   False,
        "interrupt_analysis": False,
        "cfg_anthropic_key":  os.environ.get("ANTHROPIC_API_KEY", ""),
        "cfg_tavily_key":     os.environ.get("TAVILY_API_KEY", ""),
        "agent_model":        "claude-haiku-4-5",
        # ── Indexing state ────────────────────────────────────────────────
        "is_indexing":           False,
        "interrupt_requested":   False,
        "is_chatting":           False,
        "chat_interrupt_requested": False,
    }
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════════════════════════
# GATEWAY / GEMINI / WEB-SEARCH PATCHERS  (from obesity_app_v2 pattern)
# ══════════════════════════════════════════════════════════════════════════════

def _patch_indexer_for_gateway(indexer_obj, model: str, gateway_key: str,
                                gateway_url: str, word_count: int):
    import anthropic as _ant
    if not gateway_key or not gateway_key.strip():
        raise ValueError(
            "CMU AI Gateway enabled but no key provided. "
            "Enter it in the sidebar under 'Gateway API key'."
        )
    JUDGE_MODEL = "claude-sonnet-4-20250514-v1:0"
    def _gw_call(prompt: str, model_name: str = model) -> str:
        client = _ant.Anthropic(api_key=gateway_key, base_url=gateway_url)
        resp = client.messages.create(
            model=model_name, max_tokens=max(1024, word_count * 2),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""
    indexer_obj._call_claude   = lambda prompt: _gw_call(prompt, JUDGE_MODEL)
    indexer_obj._ensure_gemini = lambda: None
    indexer_obj._gateway_call  = _gw_call
    indexer_obj._use_gateway   = True
    return indexer_obj


def _patch_indexer_for_gemini(indexer_obj, model: str, gemini_key: str, word_count: int):
    if not gemini_key or not gemini_key.strip():
        raise ValueError("Gemini mode requires a Gemini API key.")
    os.environ["GEMINI_API_KEY"]  = gemini_key
    os.environ["GOOGLE_API_KEY"]  = gemini_key
    from langchain_google_genai import ChatGoogleGenerativeAI
    _GEN_MAP = {"gemini-2.5-flash":"gemini-2.5-flash",
                "gemini-2.5-pro":"gemini-2.5-pro",
                "gemini-2.0-flash-lite":"gemini-2.0-flash-lite"}
    resolved = _GEN_MAP.get(model, model)
    def _gem_call(prompt: str) -> str:
        llm = ChatGoogleGenerativeAI(model=resolved, temperature=0, google_api_key=gemini_key)
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", "")
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        return str(content).strip()
    indexer_obj._call_claude   = _gem_call
    indexer_obj._ensure_gemini = lambda: None
    indexer_obj._use_gateway   = False
    return indexer_obj


def _gemini_web_search(query: str, model_name: str = "gemini-2.5-flash") -> str:
    try:
        import google.generativeai as genai
        key = (os.environ.get("GEMINI_API_KEY", "") or
               os.environ.get("GOOGLE_API_KEY", "") or
               st.session_state.cfg_gemini_api_key)
        if not key:
            return ""
        genai.configure(api_key=key)
        tools = [{"google_search": {}}]
        model = genai.GenerativeModel(model_name, tools=tools)
        resp  = model.generate_content(
            f"Search the web and give a concise 3-5 sentence summary relevant to: {query}"
        )
        return resp.text.strip() if hasattr(resp, "text") else ""
    except Exception as e:
        return f"[Web search failed: {e}]"


# ── Auto-reconnect to existing Milvus collection ──────────────────────────────
if RAG_AVAILABLE and st.session_state.indexer is None:
    try:
        from pymilvus import Collection, utility
        _rc_coll  = None
        _rc_embed = "BGE (local · 1024-dim)"
        for _cname, _elabel in [
            ("papers_rag_gemini",      "Gemini embedding-001 (3072-dim)"),
            ("papers_rag_voxclin",     "BGE (local · 1024-dim)"),
            ("papers_rag_interactive", "BGE (local · 1024-dim)"),
        ]:
            try:
                if utility.has_collection(_cname):
                    _tmp = Collection(_cname)
                    if _tmp.num_entities > 0:
                        for _field in _tmp.schema.fields:
                            if _field.dtype.name in ("FLOAT_VECTOR", "BINARY_VECTOR"):
                                _rc_embed = (
                                    "Gemini embedding-001 (3072-dim)"
                                    if _field.params.get("dim", 0) == 3072
                                    else "BGE (local · 1024-dim)"
                                )
                                break
                        _rc_coll = _cname
                        break
            except Exception:
                continue
        if _rc_coll:
            import indexer as _idx_mod
            _idx_mod.EMBED_DIM = 3072 if "Gemini" in _rc_embed else 1024
            st.session_state.cfg_embed_model = _rc_embed
            _c = PDFIndexer(
                chunk_size=st.session_state.cfg_chunk_size,
                chunk_overlap=st.session_state.cfg_chunk_overlap,
                model=st.session_state.cfg_model,
                index_type=st.session_state.cfg_index_type,
                collection_name=_rc_coll,
                drop_old_collection=False,
            )
            if _c._collection and _c._collection.num_entities > 0:
                _c._collection.load()
                _c._build_graph()
                st.session_state.indexer = _c
                if not st.session_state.index_stats.get("total_chunks"):
                    st.session_state.index_stats = {
                        "total_chunks": _c._collection.num_entities,
                        "total_files":  len(st.session_state.indexed_files),
                        "index_type":   st.session_state.cfg_index_type,
                        "embed_model":  _rc_embed,
                    }
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# OG HELPERS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def append_session_log(event: str, data: dict):
    entry = {"event": event, "timestamp": datetime.utcnow().isoformat(), **data}
    st.session_state.session_log.append(entry)
    path = SESSIONS_DIR / f"{st.session_state.uid}.json"
    record = {
        "session_id":   st.session_state.uid,
        "patient":      st.session_state.patient,
        "task_family":  st.session_state.task_family,
        "protocol":     st.session_state.protocol,
        "log":          st.session_state.session_log,
        "results":      st.session_state.results,
        "api_response": st.session_state.api_response,
    }
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)


def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_predict(spec_bytes, mfcc_bytes, patient_meta: dict, protocol: str) -> dict:
    """Send pre-computed .npy arrays directly to /predict (bypass mode)."""
    files = {}
    if spec_bytes:
        files["spec_file"] = ("spec.npy", io.BytesIO(spec_bytes), "application/octet-stream")
    if mfcc_bytes:
        files["mfcc_file"] = ("mfcc.npy", io.BytesIO(mfcc_bytes), "application/octet-stream")
    data = {
        "session_id":   st.session_state.uid,
        "patient_meta": json.dumps(patient_meta),
        "protocol":     protocol,
    }
    r = requests.post(
        f"{API_BASE_URL}/predict",
        files=files if files else None,
        data=data,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def call_predict_audio(audio_bytes: bytes, patient_meta: dict, protocol: str,
                        filename: str = "recording.wav") -> dict:
    """
    Send raw audio bytes to /predict with audio_file field.
    Triggers the full B2AI-faithful preprocessing pipeline on the server:
    resample → 16 kHz → peak-normalise → strip silence → spec[201,T] + MFCC[60,T].
    """
    # Guess MIME type from filename extension
    ext  = Path(filename).suffix.lower().lstrip(".")
    mime = {"wav":"audio/wav","mp3":"audio/mpeg","m4a":"audio/mp4",
            "ogg":"audio/ogg","flac":"audio/flac","webm":"audio/webm"}.get(ext, "audio/wav")
    files = {
        "audio_file": (filename, io.BytesIO(audio_bytes), mime),
    }
    data = {
        "session_id":   st.session_state.uid,
        "patient_meta": json.dumps(patient_meta),
        "protocol":     protocol,
    }
    r = requests.post(
        f"{API_BASE_URL}/predict",
        files=files,
        data=data,
        timeout=60,   # preprocessing can take a moment on CPU
    )
    r.raise_for_status()
    return r.json()


def mock_results(fam: str) -> list:
    """
    Fallback results when the API is offline.
    Includes confounder_prob / confounder_percent / gap fields matching the v2 API response,
    using heuristic AUROC values from the audit report (confounder_baseline.py).
    """
    # Confounder heuristic AUROC proxies (from AUDIT_REPORT_AUROC in confounder_baseline.py)
    _CONF = {
        "parkinsons": 0.934, "airway_stenosis": 0.775, "laryngeal_dystonia": 0.800,
        "vf_paralysis": 0.796, "chronic_cough": 0.740, "mtd": 0.740,
        "benign_lesions": 0.770, "glottic_insuff": 0.590,
        "depression": 0.777, "ptsd": 0.785, "adhd": 0.729, "bipolar": 0.750,
        "cognitive_impairment": 0.958, "psychiatric_history": 0.740,
        "anxiety": 0.750, "precancerous": 0.830, "als": 0.500,
        "copd_asthma": 0.670, "laryngitis": 0.750, "laryngeal_cancer": 0.500,
    }
    def _r(task, prob):
        cp = _CONF.get(task, 0.6)
        return {
            "task": task, "probability": prob, "percent": round(prob*100,1),
            "na_flag": False, "confounder_prob": cp,
            "confounder_percent": round(cp*100,1), "gap": round(prob-cp, 4),
        }

    MOCK = {
        "Structural / Motor": [
            _r("parkinsons", 0.87), _r("laryngeal_dystonia", 0.31),
            _r("vf_paralysis", 0.22), _r("airway_stenosis", 0.15),
        ],
        "Laryngeal / Vocal": [
            _r("chronic_cough", 0.73), _r("mtd", 0.61),
            _r("benign_lesions", 0.44), _r("glottic_insuff", 0.18),
        ],
        "Psychiatric / Cognitive": [
            _r("depression", 0.54), _r("cognitive_impairment", 0.62),
            _r("ptsd", 0.41), _r("adhd", 0.33), _r("bipolar", 0.28),
        ],
    }
    return MOCK.get(fam, [])


def confidence_label(prob: float, na: bool):
    """Confidence thresholds matching the predict.py _suggestion() function."""
    if na:           return ("na",   "#a78bfa", "N/A (missing input)")
    if prob >= 0.70: return ("high", "#059669", "High (≥70%)")
    if prob >= 0.50: return ("mid",  "#d97706", "Moderate (50–70%)")
    if prob >= 0.30: return ("low",  "#6b7280", "Weak (30–50%)")
    return ("vlow",  "#9ca3af", "No signal (<30%)")


def waveform_html():
    heights = [18,28,38,44,36,24,40,46,30,16,38,42,24,14,34,46,28]
    durs    = [f"{0.5+i*0.1:.1f}s" for i in range(len(heights))]
    bars = "".join(
        f'<div class="wbar" style="--h:{h}px;--dur:{d};"></div>'
        for h, d in zip(heights, durs)
    )
    return f'<div class="waveform">{bars}</div>'

# ══════════════════════════════════════════════════════════════════════════════
# RAG RENDER HELPERS  (adapted from obesity_app_v2, VoxClinBench theme)
# ══════════════════════════════════════════════════════════════════════════════

def _unique_papers(sources, url_map, k=2):
    seen, out = set(), []
    for s in sorted(sources, key=lambda x: float(x.get("score", 0)), reverse=True):
        n = s.get("source", "")
        if not n or n in seen:
            continue
        seen.add(n)
        out.append({"paper": n, "url": url_map.get(n, ""),
                    "paper_type": s.get("paper_type", ""),
                    "best_score": float(s.get("score", 0))})
        if len(out) >= k:
            break
    return out


def _render_recs(recs):
    if not recs:
        return
    st.markdown('<div style="font-size:.8rem;font-weight:600;color:#028090;margin:12px 0 6px 0;">Recommended Papers</div>', unsafe_allow_html=True)
    for i, r in enumerate(recs, 1):
        url   = r.get("url", "")
        uhtml = f'<div class="purl">{url}</div>' if url else '<div class="pscore">No URL available</div>'
        st.markdown(
            f'<div class="paper-card"><div class="ptitle">{i}. {r["paper"]}'
            f'  <span class="badge badge-type">{r.get("paper_type","")}</span></div>'
            f'{uhtml}<div class="pscore">score: {r["best_score"]:.4f}</div></div>',
            unsafe_allow_html=True,
        )


def _render_sources(sources):
    if not sources:
        return
    with st.expander(f"Retrieved chunks ({len(sources)})", expanded=False):
        for s in sources:
            url  = st.session_state.file_url_map.get(s["source"], "")
            ul   = f"<br><span class='badge badge-url'>URL</span> {url}" if url else ""
            st.markdown(
                f'<div class="chunk-preview">'
                f'<span class="badge">{s["source"]}</span>'
                f'<span class="badge">p{s["page"]}</span>'
                f'<span class="badge badge-type">{s.get("paper_type","?")}</span>'
                f'<span class="badge badge-score">score {s["score"]:.3f}</span>'
                f'{ul}<br><br>{s["text"][:500]}{"..." if len(s["text"])>500 else ""}</div>',
                unsafe_allow_html=True,
            )


def _render_chat_msg(msg):
    if msg["role"] == "user":
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:10px;margin:14px 0;">'
            f'<div style="width:30px;height:30px;border-radius:50%;background:#028090;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:.7rem;font-weight:700;color:white;flex-shrink:0;">You</div>'
            f'<div style="color:#1e293b;font-size:.93rem;line-height:1.65;padding-top:3px;">'
            f'{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:10px;margin:14px 0;">'
            f'<div style="width:30px;height:30px;border-radius:50%;background:#0d2233;'
            f'border:2px solid #028090;display:flex;align-items:center;justify-content:center;'
            f'font-size:.65rem;font-weight:700;color:#00a896;flex-shrink:0;">AI</div>'
            f'<div style="color:#1e293b;font-size:.93rem;line-height:1.65;padding-top:3px;width:100%;">',
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"])
        if msg.get("recs"):
            _render_recs(msg["recs"])
        if msg.get("sources"):
            _render_sources(msg["sources"])
        st.markdown("</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _format_patient_profile(patient: dict, predictions: list, task_family: str, protocol: str) -> str:
    if not predictions:
        return f"""PATIENT PROFILE:
  UID: {patient.get('uid','?')}  Age: {patient.get('age','?')}  Sex: {patient.get('sex','?')}
  Country: {patient.get('country','?')}  Ethnicity: {patient.get('ethnicity','?')}
  Primary Complaint: {patient.get('complaint','?')}
  Medications: {patient.get('medications','None listed')}
  Task Battery: {task_family}  Protocol: {protocol}

MARVEL MODEL PREDICTIONS: No predictions available."""

    sorted_preds = sorted(predictions, key=lambda p: p.get("probability", 0), reverse=True)
    preds_str = "\n".join([
        f"  - {p['task'].replace('_',' ').title()}: {p.get('percent', round(p.get('probability',0)*100,1))}%"
        f"  {'[NA — missing input]' if p.get('na_flag') else ''}"
        for p in sorted_preds
    ])
    top3 = sorted_preds[:3]
    top3_str = " | ".join([
        f"{p['task'].replace('_',' ').title()} ({p.get('percent',0):.1f}%)"
        for p in top3
    ])
    return f"""PATIENT PROFILE:
  UID: {patient.get('uid','?')}
  Age: {patient.get('age','?')}  Sex: {patient.get('sex','?')}
  Country: {patient.get('country','?')}  Ethnicity: {patient.get('ethnicity','?')}
  Primary Complaint: {patient.get('complaint','?')}
  Medications: {patient.get('medications','None listed')}
  Task Battery: {task_family}  Protocol: {protocol}

MARVEL MODEL PREDICTIONS (sorted by probability):
{preds_str}

TOP 3 DIAGNOSES: {top3_str}

Note: These are research-grade AI screening outputs from the Bridge2AI-Voice MARVEL model.
Not clinical diagnoses. Confidence scores reflect acoustic model output only.
"""


def _anthropic_call(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Direct Anthropic API call for agent LLM steps."""
    try:
        import anthropic as _ant
    except ImportError:
        raise ImportError(
            "anthropic package not installed.\n"
            "Run: pip install anthropic\n"
            "Then restart the Streamlit app."
        )

    api_key     = (st.session_state.get("cfg_anthropic_key", "") or
                   os.environ.get("ANTHROPIC_API_KEY", ""))
    use_gateway = st.session_state.get("cfg_use_gateway", False)
    gateway_key = st.session_state.get("cfg_gateway_key", "")
    gateway_url = st.session_state.get("cfg_gateway_url", os.environ.get("AI_GATEWAY_URL", ""))
    model_name  = st.session_state.get("agent_model", "claude-haiku-4-5")

    if use_gateway and gateway_key:
        client = _ant.Anthropic(api_key=gateway_key, base_url=gateway_url)
        model_name = st.session_state.get("cfg_model", "claude-sonnet-4-20250514-v1:0")
    elif api_key:
        client = _ant.Anthropic(api_key=api_key)
    else:
        raise ValueError(
            "No Anthropic API key found.\n"
            "Enter it in the 🔑 API Keys panel on this page, "
            "set ANTHROPIC_API_KEY env var, or enable CMU AI Gateway in the sidebar."
        )

    kwargs: dict = {
        "model":      model_name,
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text if resp.content else ""


def _anthropic_call_streaming(prompt: str, system: str = "", max_tokens: int = 3500, stream_ph=None) -> str:
    """Anthropic API call with streaming — updates stream_ph placeholder in real-time."""
    try:
        import anthropic as _ant
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key     = (st.session_state.get("cfg_anthropic_key", "") or os.environ.get("ANTHROPIC_API_KEY", ""))
    use_gateway = st.session_state.get("cfg_use_gateway", False)
    gateway_key = st.session_state.get("cfg_gateway_key", "")
    gateway_url = st.session_state.get("cfg_gateway_url", os.environ.get("AI_GATEWAY_URL", ""))
    model_name  = st.session_state.get("agent_model", "claude-haiku-4-5")

    if use_gateway and gateway_key:
        client = _ant.Anthropic(api_key=gateway_key, base_url=gateway_url)
        model_name = st.session_state.get("cfg_model", "claude-sonnet-4-20250514-v1:0")
    elif api_key:
        client = _ant.Anthropic(api_key=api_key)
    else:
        raise ValueError("No Anthropic API key found.")

    kwargs = {"model": model_name, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system

    full_text = ""
    try:
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                full_text += text
                if stream_ph is not None:
                    stream_ph.markdown(
                        f'<div class="report-card" style="max-height:600px;overflow-y:auto;">'
                        f'<pre style="white-space:pre-wrap;word-break:break-word;font-size:0.8rem;'
                        f'color:#111827;line-height:1.6;margin:0;">{full_text}▌</pre></div>',
                        unsafe_allow_html=True,
                    )
        # Final render without cursor
        if stream_ph is not None:
            stream_ph.markdown(
                f'<div class="report-card" style="max-height:600px;overflow-y:auto;">'
                f'<pre style="white-space:pre-wrap;word-break:break-word;font-size:0.8rem;'
                f'color:#111827;line-height:1.6;margin:0;">{full_text}</pre></div>',
                unsafe_allow_html=True,
            )
        return full_text
    except Exception:
        # Fallback to non-streaming
        resp = client.messages.create(**kwargs)
        result = resp.content[0].text if resp.content else ""
        if stream_ph is not None:
            stream_ph.markdown(
                f'<div class="report-card"><pre style="white-space:pre-wrap;word-break:break-word;'
                f'font-size:0.8rem;color:#111827;line-height:1.6;margin:0;">{result}</pre></div>',
                unsafe_allow_html=True,
            )
        return result


def _run_specialist_agent_streaming(role: str, plan_tpl: str, execute_tpl: str,
                                     synthesis_tpl: str, patient_profile: str,
                                     tavily_key: str, status_ph, stream_ph=None) -> str:
    """Run specialist agent with streaming output to stream_ph."""
    emoji = {"Pharmacology":"💊","Clinical Guidelines":"📋",
             "Rehabilitation":"🏃","Comorbidity":"⚠️"}.get(role, "🔬")
    try:
        status_ph.markdown(
            f'<div class="agent-card agent-run">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status">Step 1/4 · Planning research...</div></div>',
            unsafe_allow_html=True)
        plan = _anthropic_call(plan_tpl.format(patient_profile=patient_profile),
                               system=f"You are a {role.lower()} specialist.", max_tokens=800)

        status_ph.markdown(
            f'<div class="agent-card agent-run">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status">Step 2/4 · Searching for clinical evidence...</div></div>',
            unsafe_allow_html=True)
        queries       = _extract_queries_from_plan(plan)
        search_results = _tavily_search(queries, tavily_key)

        status_ph.markdown(
            f'<div class="agent-card agent-run">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status">Step 3/4 · Querying knowledge base...</div></div>',
            unsafe_allow_html=True)
        rag_context = _rag_query_for_agent(role, patient_profile)

        status_ph.markdown(
            f'<div class="agent-card agent-run">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status">Step 4/4 · Synthesizing findings (streaming)...</div></div>',
            unsafe_allow_html=True)
        execute_out = _anthropic_call(
            execute_tpl.format(plan=plan, search_results=search_results, rag_context=rag_context),
            system=f"You are a {role.lower()} specialist.", max_tokens=1500)

        if stream_ph is not None:
            stream_ph.markdown(
                f'<div class="agent-card" style="border:1px solid #028090;">' 
                f'<div class="agent-header" style="color:#028090;">{emoji} {role} — Synthesis (streaming)</div>',
                unsafe_allow_html=True)

        final = _anthropic_call_streaming(
            synthesis_tpl.format(patient_profile=patient_profile, findings=execute_out),
            system=f"You are a {role.lower()} research synthesis expert.",
            max_tokens=2000,
            stream_ph=stream_ph,
        )

        status_ph.markdown(
            f'<div class="agent-card agent-done">'
            f'<div class="agent-header">{emoji} {role} ✓</div>'
            f'<div class="agent-status" style="color:#02c39a;">Complete</div></div>',
            unsafe_allow_html=True)
        return final

    except Exception as e:
        status_ph.markdown(
            f'<div class="agent-card" style="border-color:#ef4444;">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status" style="color:#ef4444;">Error: {e}</div></div>',
            unsafe_allow_html=True)
        return f"[{role} agent failed: {e}]"


def _build_explainability(task: str, prob: float, confounder_prob: float, gap: float,
                           task_family: str) -> str:
    """Generate plain-text explainability for a top-3 prediction."""
    task_display = task.replace("_"," ").title()
    pct = prob * 100

    # Acoustic feature patterns by task type
    feature_hints = {
        "parkinsons":          "tremor in sustained phonation, reduced diadochokinesis rate, hypophonia",
        "laryngeal_dystonia":  "voice breaks, strain-strangled quality, irregular F0 perturbations",
        "vf_paralysis":        "breathy voice, reduced intensity, incomplete glottal closure patterns",
        "airway_stenosis":     "stridor-like spectral patterns, reduced pitch range, effortful phonation",
        "chronic_cough":       "post-phonation noise, irregular voicing onset patterns",
        "mtd":                 "elevated laryngeal tension markers, reduced F0 stability",
        "benign_lesions":      "diplophonia markers, reduced mucosal wave patterns",
        "glottic_insuff":      "air leakage patterns, weak voice intensity, reduced SPL",
        "depression":          "reduced pitch variability, slower speech rate, flat prosody",
        "ptsd":                "increased jitter/shimmer, irregular rhythm, hesitation patterns",
        "adhd":                "irregular speech rhythm, increased filler rate, attention markers",
        "bipolar":             "variable speech rate and intensity patterns",
        "cognitive_impairment":"reduced lexical diversity, slowed processing markers in narrative",
        "psychiatric_history": "prosody irregularities, pause distribution patterns",
        "anxiety":             "elevated fundamental frequency, increased speaking rate",
        "precancerous":        "spectral roughness patterns, reduced harmonic-to-noise ratio",
        "als":                 "bulbar markers: hypernasality, reduced DDK rate, dysarthria",
        "copd_asthma":         "reduced breath support, shorter phrase length, audible breathing",
        "laryngitis":          "hoarseness markers, reduced shimmer, spectral noise",
        "laryngeal_cancer":    "roughness, breathiness, spectral noise in sustained phonation",
    }

    conf_str = "High" if pct >= 70 else ("Moderate" if pct >= 45 else "Low")
    hint = feature_hints.get(task, "acoustic pattern deviation from normative baseline")

    lines = [f"**{task_display}** — {pct:.1f}% ({conf_str} confidence)"]
    lines.append(f"• Acoustic markers detected: {hint}")

    if gap is not None:
        if gap >= 0.10:
            lines.append(f"• Confounder audit: acoustic signal present — model leads demographics by +{gap:.3f} (Rec. 5.1 ✓)")
        elif gap >= 0:
            lines.append(f"• Confounder audit: gap = +{gap:.3f} — shortcut risk (threshold 0.10) ⚠")
        else:
            lines.append(f"• Confounder audit: demographics score {confounder_prob*100:.0f}% vs model {pct:.0f}% — possible confound ✗")

    # Battery context
    battery_relevance = {
        "Structural / Motor": ["parkinsons","laryngeal_dystonia","vf_paralysis","airway_stenosis"],
        "Laryngeal / Vocal":  ["chronic_cough","mtd","benign_lesions","glottic_insuff"],
        "Psychiatric / Cognitive": ["depression","ptsd","adhd","bipolar","cognitive_impairment"],
    }
    relevant = battery_relevance.get(task_family, [])
    if task in relevant:
        lines.append(f"• Task battery match: {task_family} battery targets this condition ✓")
    else:
        lines.append(f"• Note: {task_display} is outside the primary {task_family} battery scope")

    return "\n".join(lines)


def _tavily_search(queries: List[str], tavily_key: str) -> str:
    """Search Tavily for clinical evidence."""
    if not tavily_key:
        return "[Tavily key not provided — web search skipped]"
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        all_results = []
        for q in queries[:3]:
            try:
                res = client.search(query=q, max_results=3, search_depth="advanced")
                for r in res.get("results", []):
                    all_results.append(
                        f"Title: {r.get('title','')}\n"
                        f"URL: {r.get('url','')}\n"
                        f"Content: {r.get('content','')[:350]}"
                    )
            except Exception:
                pass
        return "\n\n---\n\n".join(all_results) if all_results else "[No results found]"
    except ImportError:
        # Fallback: use requests directly
        try:
            headers = {"Authorization": f"Bearer {tavily_key}", "Content-Type": "application/json"}
            all_res = []
            for q in queries[:2]:
                resp = requests.post(
                    "https://api.tavily.com/search",
                    headers=headers,
                    json={"query": q, "max_results": 3, "search_depth": "advanced"},
                    timeout=15,
                )
                if resp.ok:
                    for r in resp.json().get("results", []):
                        all_res.append(
                            f"Title: {r.get('title','')}\nURL: {r.get('url','')}\n"
                            f"Content: {r.get('content','')[:350]}"
                        )
            return "\n\n---\n\n".join(all_res) if all_res else "[No results]"
        except Exception as e:
            return f"[Tavily search failed: {e}]"
    except Exception as e:
        return f"[Tavily search failed: {e}]"


def _extract_queries_from_plan(plan: str) -> List[str]:
    """Extract KEY SEARCH QUERIES from a research plan."""
    queries = []
    in_block = False
    for line in plan.split("\n"):
        stripped = line.strip()
        if "KEY SEARCH QUERIES" in stripped.upper():
            in_block = True
            continue
        if in_block:
            if stripped.startswith("SEARCH PRIORITIES") or stripped.startswith("RULES"):
                break
            if stripped.startswith("- "):
                q = re.sub(r"^-\s*\d*\.?\s*", "", stripped).strip()
                if len(q) > 10:
                    queries.append(q)
    return queries[:4] if queries else [plan[:200]]


def _rag_query_for_agent(topic: str, patient_profile: str) -> str:
    """Query Milvus RAG for agent context."""
    idx = st.session_state.indexer
    if idx is None:
        return "[No RAG index — skipped]"
    try:
        q_vec  = idx.embed_one(f"{topic} {patient_profile[:150]}", "retrieval_query")
        docs   = idx._search(q_vec=q_vec, top_k=3)
        if not docs:
            return "[No RAG results found]"
        return "\n\n".join([
            f"[{i}] Source: {d['source']} (p{d['page']}, type={d.get('paper_type','?')}, score={d['score']:.3f})\n{d['text'][:400]}"
            for i, d in enumerate(docs, 1)
        ])
    except Exception as e:
        return f"[RAG query failed: {e}]"


def run_specialist_agent(
    role: str,
    plan_tpl: str,
    execute_tpl: str,
    synthesis_tpl: str,
    patient_profile: str,
    tavily_key: str,
    status_ph,
) -> str:
    """
    Run one specialist agent through: plan → search → RAG → execute → synthesize.
    Matches the LangFlow node sequence in noapi_multiagent1.json.
    """
    emoji = {"Pharmacology":"💊","Clinical Guidelines":"📋",
             "Rehabilitation":"🏃","Comorbidity":"⚠️"}.get(role, "🔬")
    try:
        # Step 1 — Research plan
        status_ph.markdown(
            f'<div class="agent-card agent-run">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status">Step 1/4 · Planning research...</div></div>',
            unsafe_allow_html=True,
        )
        plan = _anthropic_call(
            plan_tpl.format(patient_profile=patient_profile),
            system=f"You are a {role.lower()} specialist.",
            max_tokens=800,
        )

        # Step 2 — Tavily search
        status_ph.markdown(
            f'<div class="agent-card agent-run">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status">Step 2/4 · Searching for clinical evidence...</div></div>',
            unsafe_allow_html=True,
        )
        queries       = _extract_queries_from_plan(plan)
        search_results = _tavily_search(queries, tavily_key)

        # Step 3 — RAG
        status_ph.markdown(
            f'<div class="agent-card agent-run">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status">Step 3/4 · Querying knowledge base...</div></div>',
            unsafe_allow_html=True,
        )
        rag_context = _rag_query_for_agent(role, patient_profile)

        # Step 4 — Execute + synthesize
        status_ph.markdown(
            f'<div class="agent-card agent-run">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status">Step 4/4 · Synthesizing findings...</div></div>',
            unsafe_allow_html=True,
        )
        execute_out = _anthropic_call(
            execute_tpl.format(
                plan=plan,
                search_results=search_results,
                rag_context=rag_context,
            ),
            system=f"You are a {role.lower()} specialist.",
            max_tokens=1200,
        )
        final = _anthropic_call(
            synthesis_tpl.format(
                patient_profile=patient_profile,
                findings=execute_out,
            ),
            system=f"You are a {role.lower()} research synthesis expert.",
            max_tokens=1500,
        )

        status_ph.markdown(
            f'<div class="agent-card agent-done">'
            f'<div class="agent-header">{emoji} {role} ✓</div>'
            f'<div class="agent-status" style="color:#02c39a;">Complete</div></div>',
            unsafe_allow_html=True,
        )
        return final

    except Exception as e:
        status_ph.markdown(
            f'<div class="agent-card" style="border-color:#ef4444;">'
            f'<div class="agent-header">{emoji} {role}</div>'
            f'<div class="agent-status" style="color:#ef4444;">Error: {e}</div></div>',
            unsafe_allow_html=True,
        )
        return f"[{role} agent failed: {e}]"


def run_master_agent(patient_profile: str, agent_reports: dict, status_ph, stream_ph=None) -> str:
    """
    Master synthesiser: bagging + RAG verification + final report.
    Works with or without RAG index — gracefully skips if unavailable.
    Streams output to stream_ph if provided.
    """
    status_ph.markdown(
        '<div class="agent-card agent-run">'
        '<div class="agent-header">🧠 Master Synthesiser</div>'
        '<div class="agent-status">Bagging consensus · RAG verification · Generating final report...</div></div>',
        unsafe_allow_html=True,
    )
    try:
        # RAG is optional — works without it
        rag_context = _rag_query_for_agent("clinical decision support", patient_profile)
        prompt = _MASTER_PROMPT.format(
            patient_profile=patient_profile,
            pharma_output=agent_reports.get("Pharmacology","[Not available]"),
            guidelines_output=agent_reports.get("Clinical Guidelines","[Not available]"),
            rehab_output=agent_reports.get("Rehabilitation","[Not available]"),
            comorbidity_output=agent_reports.get("Comorbidity","[Not available]"),
            rag_context=rag_context,
        )

        # Stream if placeholder provided
        if stream_ph is not None:
            result = _anthropic_call_streaming(
                prompt,
                system="You are the master clinical decision support synthesiser.",
                max_tokens=3500,
                stream_ph=stream_ph,
            )
        else:
            result = _anthropic_call(
                prompt,
                system="You are the master clinical decision support synthesiser.",
                max_tokens=3500,
            )

        status_ph.markdown(
            '<div class="agent-card agent-done">'
            '<div class="agent-header">🧠 Master Synthesiser ✓</div>'
            '<div class="agent-status" style="color:#02c39a;">Report generated</div></div>',
            unsafe_allow_html=True,
        )
        return result
    except Exception as e:
        status_ph.markdown(
            '<div class="agent-card" style="border-color:#ef4444;">'
            '<div class="agent-header">🧠 Master Synthesiser</div>'
            f'<div class="agent-status" style="color:#ef4444;">Error: {e}</div></div>',
            unsafe_allow_html=True,
        )
        return f"[Master agent failed: {e}]"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _sblabel(t):
    st.markdown(
        f'<div style="font-size:.62rem;font-weight:600;letter-spacing:.1em;'
        f'text-transform:uppercase;color:#607a8a;padding:8px 0 4px 0;">{t}</div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("""
    <div style="padding:18px 16px 12px 16px;text-align:center;">
    <div style="font-size:2rem;font-weight:800;color:#00a896;
                letter-spacing:.04em;line-height:1.4;">
        VoxClinBench
    </div>
    <div style="font-size:1.5rem;color:#607a8a;margin-top:4px;">
        42-657 · CMU
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    for tid, tlabel in [("rag","RAG Config"), ("files","Indexed Files")]:
        active = st.session_state.sidebar_tab == tid
        if st.button(tlabel, key=f"sb_vtab_{tid}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.sidebar_tab = tid
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Reset All", use_container_width=True, key="sb_reset"):
        for k, v in [("chat_history",[]),("indexer",None),
                     ("indexed_files",[]),("file_url_map",{}),
                     ("index_stats",{}),("agent_results",{}),("final_report","")]:
            st.session_state[k] = v
        _save_chat(); _save_indexed(); st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.sidebar_tab == "rag":

        _sblabel("API Backend")
        use_gw = st.checkbox(
            "Use CMU AI Gateway",
            value=st.session_state.cfg_use_gateway,
            key="sb_use_gw",
            help="Routes generation through the CMU Andrew AI Gateway."
        )
        st.session_state.cfg_use_gateway = use_gw

        if use_gw:
            gw_key = st.text_input(
                "Gateway API key", value=st.session_state.cfg_gateway_key,
                type="password", placeholder="sk-...",
                key="sb_gw_key", label_visibility="collapsed",
            )
            st.session_state.cfg_gateway_key = gw_key
            st.markdown(
                '<div style="font-size:.67rem;color:#02c39a;padding-bottom:4px;">✓ Key set</div>'
                if gw_key else
                '<div style="font-size:.67rem;color:#ef4444;padding-bottom:4px;">⚠ Key required</div>',
                unsafe_allow_html=True,
            )
            _sblabel("Generation Model")
            GW_MODELS = ["claude-sonnet-4-20250514-v1:0","claude-haiku-4-5-20251001-v1:0",
                         "claude-opus-4-20250514-v1:0","gemini-2.5-flash"]
            if st.session_state.cfg_model not in GW_MODELS:
                st.session_state.cfg_model = GW_MODELS[0]
            st.session_state.cfg_model = st.selectbox(
                "gw_model", GW_MODELS,
                index=GW_MODELS.index(st.session_state.cfg_model),
                label_visibility="collapsed",
            )
            st.markdown(
                '<div style="font-size:.63rem;color:#607a8a;padding-bottom:4px;">'
                '🔒 Agent judge: Claude Sonnet</div>',
                unsafe_allow_html=True,
            )
        else:
            _sblabel("Anthropic API Key (Agents)")
            _ant_key_sb = st.text_input(
                "Anthropic key", value=st.session_state.cfg_anthropic_key,
                type="password", placeholder="Anthropic API key",
                key="sb_ant_key", label_visibility="collapsed",
            )
            st.session_state.cfg_anthropic_key = _ant_key_sb
            if _ant_key_sb:
                os.environ["ANTHROPIC_API_KEY"] = _ant_key_sb
                st.markdown(
                    '<div style="font-size:.67rem;color:#02c39a;padding-bottom:4px;">✓ Key set</div>',
                    unsafe_allow_html=True,
                )

            _sblabel("RAG Generation Model")
            GEM_MODELS = ["gemini-2.5-flash","gemini-2.5-pro","gemini-2.0-flash-lite"]
            if st.session_state.cfg_model not in GEM_MODELS:
                st.session_state.cfg_model = GEM_MODELS[0]
            st.session_state.cfg_model = st.selectbox(
                "gem_model", GEM_MODELS,
                index=GEM_MODELS.index(st.session_state.cfg_model),
                label_visibility="collapsed",
            )
            _gkey_env = (os.environ.get("GEMINI_API_KEY","") or
                         os.environ.get("GOOGLE_API_KEY","") or
                         st.session_state.cfg_gemini_api_key)
            if _gkey_env:
                st.markdown(
                    '<div style="font-size:.67rem;color:#02c39a;padding-bottom:4px;">✓ Gemini key ready</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:.67rem;color:#f59e0b;padding-bottom:4px;">'
                    '⚠ Enter Gemini key in RAG Dev tab</div>',
                    unsafe_allow_html=True,
                )

        _sblabel("Tavily API Key (Agent Search)")
        _tav = st.text_input(
            "Tavily key", value=st.session_state.cfg_tavily_key,
            type="password", placeholder="tvly-...",
            key="sb_tav_key", label_visibility="collapsed",
        )
        st.session_state.cfg_tavily_key = _tav
        if _tav:
            st.markdown(
                '<div style="font-size:.67rem;color:#02c39a;padding-bottom:4px;">✓ Tavily key set</div>',
                unsafe_allow_html=True,
            )

        _sblabel("Retrieval")
        st.session_state.cfg_top_k      = st.slider("Top-K chunks", 1, 10, st.session_state.cfg_top_k)
        st.session_state.cfg_iterations = st.slider("Reflection iterations", 1, 4, st.session_state.cfg_iterations)

        _sblabel("Response Length (words)")
        st.session_state.cfg_word_count = st.slider(
            "words", 300, 2000, st.session_state.cfg_word_count, 50,
            label_visibility="collapsed",
        )

        _sblabel("Index Algorithm")
        st.session_state.cfg_index_type = st.selectbox(
            "idx", ["HNSW","IVF_PQ","DiskANN"],
            index=["HNSW","IVF_PQ","DiskANN"].index(st.session_state.cfg_index_type),
            label_visibility="collapsed",
        )

        if not RAG_AVAILABLE:
            st.markdown(
                '<div style="font-size:.72rem;color:#607a8a;padding:8px 0;">'
                'indexer.py not found — RAG disabled</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.indexer is None:
            st.markdown(
                '<div style="font-size:.72rem;color:#f59e0b;padding:8px 0;">'
                'No Milvus collection found.<br>Index PDFs in RAG Dev tab first.</div>',
                unsafe_allow_html=True,
            )
        else:
            s = st.session_state.index_stats
            st.markdown(
                f'<div style="font-size:.72rem;color:#02c39a;padding:8px 0;">'
                f'✓ Connected · {s.get("total_chunks",0)} chunks · '
                f'{s.get("embed_model",EMBED_MODEL).split("/")[-1]}</div>',
                unsafe_allow_html=True,
            )

    elif st.session_state.sidebar_tab == "files":
        if not st.session_state.indexed_files:
            st.markdown(
                '<div style="font-size:.78rem;color:#607a8a;padding:12px 0;">No files indexed yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            s = st.session_state.index_stats
            st.markdown(f"""
            <div style="display:flex;gap:6px;padding:8px 0 12px 0;">
              <div style="flex:1;background:#1e3a50;border:1px solid #2a5060;border-radius:8px;
                          padding:10px 6px;text-align:center;">
                <div style="font-size:1rem;font-weight:600;color:#e2e8f0;">{len(st.session_state.indexed_files)}</div>
                <div style="font-size:.58rem;text-transform:uppercase;color:#607a8a;margin-top:3px;">files</div>
              </div>
              <div style="flex:1;background:#1e3a50;border:1px solid #2a5060;border-radius:8px;
                          padding:10px 6px;text-align:center;">
                <div style="font-size:1rem;font-weight:600;color:#e2e8f0;">{s.get('total_chunks',0)}</div>
                <div style="font-size:.58rem;text-transform:uppercase;color:#607a8a;margin-top:3px;">chunks</div>
              </div>
            </div>""", unsafe_allow_html=True)
            for name, ptype in st.session_state.indexed_files:
                st.markdown(
                    f'<div style="background:#1e3a50;border:1px solid #2a5060;border-radius:6px;'
                    f'padding:6px 10px;font-size:.71rem;color:#9db4c0;margin:2px 0;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                    f'{name} <span style="background:#0d3a50;color:#00a896;padding:1px 5px;'
                    f'border-radius:3px;font-size:.58rem;">{ptype}</span></div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR (extended with page tabs)
# ══════════════════════════════════════════════════════════════════════════════

def nav_bar():
    api_ok  = api_health()
    api_cls = "nav-api-ok" if api_ok else "nav-api-off"
    api_lbl = "API Online"  if api_ok else "API Offline"
    cur_pg  = st.session_state.page

    pages = [("assessment","🎙 Assessment"),("chat","🧠 Clinical AI"),("rag_dev","📚 RAG Dev")]
    tabs_html = "".join([
        f'<span class="nav-page-btn {"active" if cur_pg == pid else ""}">'
        f'{plabel}</span>'
        for pid, plabel in pages
    ])

    st.markdown(f"""
    <div class="top-nav" style="color: white;">
      <div>
        <div class="nav-brand">VoxClinBench</div>
        <div class="nav-sub">42-657 Projects in Biomedical AI &nbsp;|&nbsp; Carnegie Mellon University</div>
      </div>
      <div id="nav-page-tabs" style="display:flex;align-items:center;gap:0.2rem;">
        {tabs_html}
      </div>
      <div style="display:flex;align-items:center;gap:0.4rem;">
        <span class="nav-badge">Research · Not for Clinical Use</span>
        <span class="{api_cls}">{api_lbl}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit buttons for page navigation (below nav bar)
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    col1, col2, col3, col_sp = st.columns([2, 2, 2, 5])
    with col1:
        if st.button("🎙  Assessment", key="nav_assess",
                     type="primary" if cur_pg=="assessment" else "secondary",
                     use_container_width=True):
            st.session_state.page = "assessment"; st.rerun()
    with col2:
        if st.button("🧠  Clinical AI", key="nav_chat",
                     type="primary" if cur_pg=="chat" else "secondary",
                     use_container_width=True):
            st.session_state.page = "chat"; st.rerun()
    with col3:
        if st.button("📚  RAG Dev", key="nav_rag",
                     type="primary" if cur_pg=="rag_dev" else "secondary",
                     use_container_width=True):
            st.session_state.page = "rag_dev"; st.rerun()
    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)


def stepper(current: int):
    steps = ["Patient Intake","Task Assignment","Voice Recording","Preprocessing","Results"]
    html  = '<div class="stepper">'
    for i, label in enumerate(steps):
        n = i + 1
        if n < current:    cls, sym = "done",    "✓"
        elif n == current: cls, sym = "active",  str(n)
        else:              cls, sym = "pending", str(n)
        html += (
            f'<div class="step">'
            f'<div class="step-circle {cls}">{sym}</div>'
            f'<div class="step-label {cls}">{label}</div>'
            f'</div>'
        )
        if i < len(steps) - 1:
            lc = "done" if n < current else "pending"
            html += f'<div class="step-line {lc}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ASSESSMENT  (OG stages 1-5, completely unchanged)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>

/* Text inputs */
input, textarea {
    background-color: white !important;
    color: black !important;
}

/* Number input */
div[data-baseweb="input"] input {
    background-color: white !important;
    color: black !important;
}

/* Selectbox (closed state) */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

/* Dropdown menu options */
div[role="listbox"] {
    background-color: white !important;
}
div[role="listbox"] ul li {
    color: black !important;
}

/* Fix placeholder text (like "e.g. John Doe") */
input::placeholder, textarea::placeholder {
    color: #555 !important;
}

/* Checkbox label */
label {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Target checkbox label text */
div[data-testid="stCheckbox"] label p {
    color: red !important;
}

/* Fallback (in case structure differs slightly) */
div[data-testid="stCheckbox"] label {
    color: red !important;
}

</style>
""", unsafe_allow_html=True)



def stage_intake():
    st.markdown('<div class="section-title">Stage 1 &middot; Patient Intake Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="teal-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All fields marked * are required. Data is logged alongside voice recordings for the mandatory confounder audit.</div>', unsafe_allow_html=True)

    col_form, col_info = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Patient Registration**")
        with st.form("intake_form"):
            name    = st.text_input("Full Name *", placeholder="e.g. John Doe")
            c1, c2  = st.columns(2)
            with c1:
                age = st.number_input("Age *", min_value=18, max_value=110, value=55)
            with c2:
                sex = st.selectbox("Biological Sex at Birth *",
                    ["-- select --","Female","Male","Intersex / Other"])
            country   = st.selectbox("Country of Origin *",
                ["-- select --","United States","Canada","Germany","Spain",
                 "United Kingdom","China","India","Other"])
            ethnicity = st.selectbox("Ethnicity *",
                ["-- select --","White / Caucasian","Black / African American",
                 "Hispanic / Latino","Asian","Middle Eastern","Mixed / Other","Prefer not to say"])
            complaint = st.selectbox("Primary Complaint *",
                ["-- select --","Voice changes or hoarseness","Motor / movement symptoms (tremor, stiffness)",
                 "Chronic cough","Mood or psychiatric symptoms","Memory or cognitive changes",
                 "Breathing or airway difficulty","General screening / research"])
            meds      = st.text_area("Current Medications (optional)", placeholder="e.g. Levodopa 250mg", height=65)
            consent   = st.checkbox("I have read the participant information sheet and consent to voice recording for research purposes.")
            submitted = st.form_submit_button("Submit & Continue", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            errors = []
            if not name.strip():            errors.append("Full name is required.")
            if sex == "-- select --":       errors.append("Please select biological sex.")
            if country == "-- select --":   errors.append("Please select country.")
            if ethnicity == "-- select --": errors.append("Please select ethnicity.")
            if complaint == "-- select --": errors.append("Please select primary complaint.")
            if not consent:                 errors.append("Informed consent is required.")
            for e in errors:
                st.markdown(f'<div class="error-box">&#9888; {e}</div>', unsafe_allow_html=True)
            if not errors:
                st.session_state.patient = {
                    "name": name, "age": int(age), "sex": sex,
                    "country": country, "ethnicity": ethnicity,
                    "complaint": complaint, "medications": meds,
                    "uid": st.session_state.uid,
                    "timestamp": datetime.now().isoformat(),
                }
                append_session_log("intake_complete", {"patient": st.session_state.patient})
                st.session_state.stage = 2
                st.rerun()

    with col_info:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Why we collect each field**")
        for title, desc in [
            ("Age + Sex", "Dominant confounders for psychiatric labels -- age alone achieves 0.69 AUROC for PTSD. Logged for mandatory demographic stratification audit (Rec. 5.3)."),
            ("Country / Ethnicity", "Geographic leakage: cognitive impairment is Canada-only in B2AI v3. Country flags potential site-identity shortcuts before scoring."),
            ("Task Histogram Proxy", "Task histogram alone achieves 0.929 AUROC for cognitive impairment. Standardized task assignment breaks this shortcut."),
            ("Informed Consent", "Legally required before audio capture. Consent flag stored in audit log alongside every prediction."),
        ]:
            st.markdown(f"""
            <div style="border-left:3px solid #028090;padding:0.5rem 0.8rem;margin-bottom:0.9rem;
                        background:#f8fafc;border-radius:0 6px 6px 0;">
              <div style="font-weight:600;color:#028090;font-size:0.88rem;">{title}</div>
              <div style="font-size:0.82rem;color:#475569;margin-top:0.25rem;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="warn-box">&#9888; This is a research tool. Predictions are <strong>not</strong> clinical diagnoses.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def stage_task_assignment():
    p = st.session_state.patient
    st.markdown('<div class="section-title">Stage 2 &middot; Disease-Adaptive Task Assignment</div>', unsafe_allow_html=True)
    st.markdown('<div class="teal-line"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:1.2rem;">
      <span class="info-pill">UID: {p['uid']}</span>
      <span class="info-pill">Age: {p['age']}</span>
      <span class="info-pill">Sex: {p['sex']}</span>
      <span class="info-pill">Country: {p['country']}</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    for i, (fam, data) in enumerate(TASK_BATTERIES.items()):
        with cols[i]:
            selected     = st.session_state.task_family == fam
            border_color = data["color"] if selected else "#e2e8f0"
            bg           = "#f0fdfa" if selected else "white"
            task_rows    = "".join(
                f'<div style="font-size:0.8rem;color:#1e293b;padding:0.25rem 0;border-bottom:1px solid #f1f5f9;">'
                f'&rarr; {t[0]}</div>'
                for t in data["tasks"]
            )
            st.markdown(f"""
            <div style="border:2px solid {border_color};border-radius:12px;overflow:hidden;
                        background:{bg};box-shadow:0 2px 10px rgba(0,0,0,0.06);">
              <div style="background:{data['color']};padding:0.9rem 1.1rem;">
                <div style="color:white;font-weight:700;font-size:1rem;">{fam}</div>
              </div>
              <div style="padding:1rem 1.1rem;">
                <div style="font-size:0.75rem;color:{data['color']};font-weight:600;margin-bottom:0.3rem;">TARGET CONDITIONS</div>
                <div style="font-size:0.82rem;color:#475569;margin-bottom:0.9rem;">{" &middot; ".join(data["diseases"])}</div>
                <div style="font-size:0.75rem;color:{data['color']};font-weight:600;margin-bottom:0.5rem;">ASSIGNED TASKS</div>
                {task_rows}
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            lbl = "Selected" if selected else f"Select {fam.split('/')[0].strip()}"
            if st.button(lbl, key=f"btn_fam_{i}", use_container_width=True):
                st.session_state.task_family = fam
                st.rerun()

    st.markdown('<div class="warn-box">&#9888; Psychiatric tasks include semantic and linguistic components -- required per Rec. 5.5 (acoustic features alone are insufficient).</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1.5, 4])
    with c1:
        if st.button("Back", key="back2"):
            st.session_state.stage = 1; st.rerun()
    with c2:
        if st.session_state.task_family:
            if st.button("Continue", key="fwd2"):
                st.session_state.tasks_done  = []
                st.session_state.task_inputs = {}
                append_session_log("task_family_selected", {"family": st.session_state.task_family})
                st.session_state.stage = 3; st.rerun()
        else:
            st.markdown('<div style="font-size:0.82rem;color:#94a3b8;padding-top:0.6rem;">Select a battery first</div>', unsafe_allow_html=True)


def stage_recording():
    fam   = st.session_state.task_family
    data  = TASK_BATTERIES[fam]
    tasks = data["tasks"]
    done  = st.session_state.tasks_done

    st.markdown(f'<div class="section-title">Stage 3 &middot; Voice Recording &mdash; {fam}</div>', unsafe_allow_html=True)
    st.markdown('<div class="teal-line"></div>', unsafe_allow_html=True)

    col_tasks, col_record = st.columns([1, 1.4], gap="large")

    with col_tasks:
        st.markdown('<div class="card" style="padding:1.2rem;">', unsafe_allow_html=True)
        prog_pct = int(len(done) / len(tasks) * 100)
        st.markdown(
            f"**Task List** &nbsp;"
            f"<span style='color:#64748b;font-size:0.8rem;'>{len(done)}/{len(tasks)} complete</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div style="background:#e2e8f0;border-radius:4px;height:6px;margin-bottom:1rem;">
          <div style="background:#028090;width:{prog_pct}%;height:6px;border-radius:4px;transition:width 0.4s;"></div>
        </div>
        """, unsafe_allow_html=True)

        remaining_all = [i for i in range(len(tasks)) if i not in done]
        for idx, (title, desc, dur) in enumerate(tasks):
            is_done   = idx in done
            is_active = bool(remaining_all) and idx == remaining_all[0]
            border    = data["color"] if is_active else ("#02c39a" if is_done else "#e2e8f0")
            bg        = "#f0fdfa" if is_active else ("#f0fdf4" if is_done else "#f8fafc")
            mark      = "&#10003;" if is_done else str(idx+1)
            mark_col  = "#02c39a"  if is_done else (data["color"] if is_active else "#94a3b8")
            mode_badge = ""
            ti = st.session_state.task_inputs.get(idx, {})
            if ti:
                m  = ti.get("mode","")
                mc = {"simulate":"#028090","npy_bypass":"#7c3aed","audio_upload":"#0369a1"}.get(m,"#94a3b8")
                mode_badge = f'<span style="font-size:0.68rem;background:{mc}20;color:{mc};border-radius:4px;padding:0.1rem 0.35rem;margin-left:0.3rem;">{m}</span>'
            st.markdown(f"""
            <div style="border:1.5px solid {border};border-radius:8px;padding:0.75rem 0.9rem;
                        margin-bottom:0.5rem;background:{bg};">
              <div style="display:flex;align-items:center;gap:0.5rem;">
                <div style="width:22px;height:22px;border-radius:50%;background:{mark_col}20;color:{mark_col};
                            font-weight:700;font-size:0.75rem;display:flex;align-items:center;justify-content:center;">{mark}</div>
                <div style="font-weight:600;color:#1e293b;font-size:0.88rem;">{title}{mode_badge}</div>
                <div style="margin-left:auto;font-size:0.72rem;color:#94a3b8;">{dur}s</div>
              </div>
              <div style="font-size:0.78rem;color:#64748b;margin-top:0.3rem;margin-left:30px;">{desc[:65]}{"..." if len(desc)>65 else ""}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_record:
        remaining = [i for i in range(len(tasks)) if i not in done]
        if remaining:
            current_idx      = remaining[0]
            title, desc, dur = tasks[current_idx]
            rec_key          = f"rec_sim_{current_idx}"
            if rec_key not in st.session_state:
                st.session_state[rec_key] = False

            st.markdown(f"""
            <div class="card-dark">
              <div style="text-align:center;color:#00a896;font-weight:700;font-size:0.9rem;margin-bottom:0.2rem;">
                VoxClinBench &middot; Voice Capture
              </div>
              <div style="text-align:center;color:#607a8a;font-size:0.78rem;margin-bottom:1rem;">
                Task {current_idx+1} of {len(tasks)} &nbsp;&middot;&nbsp; {fam}
              </div>
              <div style="background:#112233;border:1px solid #028090;border-radius:8px;padding:1rem 1.2rem;">
                <div style="font-weight:700;color:white;font-size:1rem;margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.85rem;color:#9db4c0;line-height:1.55;">{desc}</div>
                <div style="margin-top:0.6rem;font-size:0.78rem;color:#028090;">Target: {dur} seconds</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            tab_rec, tab_npy, tab_audio, tab_test = st.tabs([
                "🎙️  Live Record",
                "📦  Upload .npy  (Bypass)",
                "🎵  Upload Audio",
                "🧪  Test Sample",
            ])

            captured   = False
            input_mode = None
            input_data = {}

            with tab_rec:
                st.markdown(
                    '<div class="note-box">'
                    '<strong>Live microphone capture</strong> — records directly in the browser. '
                    'The server resamples to 16 kHz, peak-normalises, strips silence, '
                    'extracts spectrogram [201,T] and MFCC [60,T] via the B2AI pipeline. '
                    'Allow microphone access when prompted.</div>',
                    unsafe_allow_html=True,
                )

                # ── Real browser mic via st.audio_input() (Streamlit ≥ 1.31) ──
                _audio_rec = None
                try:
                    _audio_rec = st.audio_input(
                        f"Click to record — target: {dur} s",
                        key=f"mic_input_{current_idx}",
                    )
                except AttributeError:
                    st.markdown(
                        '<div class="warn-box">⚠ Live recording requires Streamlit ≥ 1.31. '
                        'Update with: <code>pip install -U streamlit</code></div>',
                        unsafe_allow_html=True,
                    )

                if _audio_rec is not None:
                    _mic_bytes = _audio_rec.read()
                    st.audio(_audio_rec)
                    st.markdown(
                        f'<div style="background:#f0fdf4;border:1px solid #02c39a;border-radius:6px;'
                        f'padding:0.5rem 0.8rem;font-size:0.82rem;color:#065a4a;margin-top:0.5rem;">'
                        f'&#10003; Recording captured &mdash; {len(_mic_bytes)/1024:.1f} KB &middot; '
                        f'Will be resampled → 16 kHz &middot; B2AI preprocessing pipeline</div>',
                        unsafe_allow_html=True,
                    )
                    captured   = True
                    input_mode = "live_mic"
                    input_data = {"audio_bytes": _mic_bytes, "filename": "live_recording.webm",
                                  "size_kb": round(len(_mic_bytes)/1024, 1)}

                st.markdown("<hr style='border-color:#e2e8f0;margin:0.8rem 0;'>", unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:0.78rem;color:#64748b;">No mic? Use the simulate option below.</div>',
                    unsafe_allow_html=True,
                )
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("Simulate (no audio)", key=f"sim_{current_idx}", use_container_width=True):
                        st.session_state[rec_key] = True; st.rerun()
                with r2:
                    if st.button("Clear", key=f"rerec_{current_idx}", use_container_width=True):
                        st.session_state[rec_key] = False
                        st.session_state.task_inputs.pop(current_idx, None)
                        st.rerun()
                if _audio_rec is None and st.session_state[rec_key]:
                    st.markdown(waveform_html(), unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;'
                        f'padding:0.5rem 0.8rem;font-size:0.82rem;color:#92400e;margin-top:0.5rem;">'
                        f'&#9888; Simulated — no real audio. Model will receive zero inputs. '
                        f'Use for UI testing only.</div>',
                        unsafe_allow_html=True,
                    )
                    if not captured:
                        captured   = True
                        input_mode = "simulate"
                        input_data = {"note": "simulated"}

            with tab_npy:
                st.markdown("""
                <div class="na-box">
                  <strong>Feature bypass</strong> — upload pre-computed arrays, skipping audio preprocessing.<br>
                  <span style="font-size:0.8rem;">
                  <strong>.npz</strong> (recommended) — contains both spec + mfcc; skips straight to inference.<br>
                  <strong>.npy</strong> — Spectrogram: <code>[201,T]</code> or <code>[1,201,T]</code> &nbsp;·&nbsp; MFCC: <code>[60,T]</code> or <code>[1,60,T]</code><br>
                  T is flexible — auto-padded/trimmed to 256 frames. Either branch can be omitted (NA-filled).
                  </span>
                </div>
                """, unsafe_allow_html=True)

                # ── NPZ upload (auto-extracts spec + mfcc, skips to inference) ──
                npz_up = st.file_uploader(
                    "Upload .npz (preprocessed bundle — skips directly to inference)",
                    type=["npz"], key=f"npz_{current_idx}",
                )
                if npz_up:
                    try:
                        _npz_data = np.load(io.BytesIO(npz_up.read()), allow_pickle=True)
                        _spec_key = next((k for k in _npz_data if "spec" in k.lower()), None)
                        _mfcc_key = next((k for k in _npz_data if "mfcc" in k.lower()), None)
                        # Fallback: positional if single arrays
                        if _spec_key is None and len(_npz_data.files) >= 1:
                            _spec_key = _npz_data.files[0]
                        if _mfcc_key is None and len(_npz_data.files) >= 2:
                            _mfcc_key = _npz_data.files[1]

                        _spec_arr = _npz_data[_spec_key].astype(np.float32) if _spec_key else None
                        _mfcc_arr = _npz_data[_mfcc_key].astype(np.float32) if _mfcc_key else None

                        st.markdown(
                            f'<div style="font-size:0.82rem;color:#134e4a;margin-top:0.4rem;'
                            f'background:#f0fdfa;border:1px solid #99f6e4;border-radius:6px;padding:0.5rem 0.75rem;">'
                            f'✓ NPZ loaded &nbsp;·&nbsp; keys: <code>{", ".join(_npz_data.files)}</code><br>'
                            f'Spec: <code>{list(_spec_arr.shape) if _spec_arr is not None else "not found"}</code> &nbsp;·&nbsp; '
                            f'MFCC: <code>{list(_mfcc_arr.shape) if _mfcc_arr is not None else "not found"}</code></div>',
                            unsafe_allow_html=True,
                        )

                        # Save as bytes
                        _sb = io.BytesIO(); np.save(_sb, _spec_arr); _spec_bytes_val = _sb.getvalue() if _spec_arr is not None else None
                        _mb = io.BytesIO(); np.save(_mb, _mfcc_arr); _mfcc_bytes_val = _mb.getvalue() if _mfcc_arr is not None else None

                        col_npz1, col_npz2 = st.columns(2)
                        with col_npz1:
                            if st.button("Use this NPZ", key=f"npz_use_{current_idx}", use_container_width=True):
                                # Mark all remaining tasks done with this data
                                for _remaining_idx in [i for i in range(len(tasks)) if i not in st.session_state.tasks_done]:
                                    st.session_state.task_inputs[_remaining_idx] = {
                                        "mode": "npy_bypass",
                                        "data": {
                                            "spec_bytes": _spec_bytes_val,
                                            "mfcc_bytes": _mfcc_bytes_val,
                                            "spec_provided": _spec_bytes_val is not None,
                                            "mfcc_provided": _mfcc_bytes_val is not None,
                                            "source": npz_up.name,
                                        },
                                    }
                                    if _remaining_idx not in st.session_state.tasks_done:
                                        st.session_state.tasks_done.append(_remaining_idx)
                                append_session_log("npz_loaded", {
                                    "filename": npz_up.name,
                                    "spec_shape": list(_spec_arr.shape) if _spec_arr is not None else None,
                                    "mfcc_shape": list(_mfcc_arr.shape) if _mfcc_arr is not None else None,
                                })
                                st.rerun()
                        with col_npz2:
                            if st.button("⚡ Skip to Inference", key=f"npz_skip_{current_idx}",
                                         use_container_width=True):
                                for _ri in range(len(tasks)):
                                    st.session_state.task_inputs[_ri] = {
                                        "mode": "npy_bypass",
                                        "data": {
                                            "spec_bytes": _spec_bytes_val,
                                            "mfcc_bytes": _mfcc_bytes_val,
                                            "spec_provided": _spec_bytes_val is not None,
                                            "mfcc_provided": _mfcc_bytes_val is not None,
                                            "source": npz_up.name,
                                        },
                                    }
                                    if _ri not in st.session_state.tasks_done:
                                        st.session_state.tasks_done.append(_ri)
                                append_session_log("npz_skip_to_inference", {"filename": npz_up.name})
                                st.session_state.stage = 4
                                st.rerun()
                    except Exception as _ne:
                        st.error(f"NPZ load error: {_ne}")

                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("**Or upload individual .npy files:**", unsafe_allow_html=True)

                s_col, m_col = st.columns(2)
                with s_col:
                    spec_up = st.file_uploader("Spectrogram .npy", type=["npy"], key=f"spec_{current_idx}")
                with m_col:
                    mfcc_up = st.file_uploader("MFCC .npy", type=["npy"], key=f"mfcc_{current_idx}")

                spec_bytes_val = mfcc_bytes_val = None
                shape_ok = True

                if spec_up:
                    raw = spec_up.read()
                    try:
                        arr = np.load(io.BytesIO(raw))
                        st.markdown(f'<div style="font-size:0.8rem;color:#134e4a;margin-top:0.3rem;">✓ Spec: {list(arr.shape)}</div>', unsafe_allow_html=True)
                        spec_bytes_val = raw
                    except Exception as e:
                        st.markdown(f'<div style="font-size:0.8rem;color:#881337;margin-top:0.3rem;">✗ Spec error: {e}</div>', unsafe_allow_html=True)
                        shape_ok = False
                else:
                    st.markdown('<div style="font-size:0.76rem;color:#6b7280;margin-top:0.3rem;">No spec → zero-filled (NA)</div>', unsafe_allow_html=True)

                if mfcc_up:
                    raw = mfcc_up.read()
                    try:
                        arr = np.load(io.BytesIO(raw))
                        st.markdown(f'<div style="font-size:0.8rem;color:#134e4a;margin-top:0.3rem;">✓ MFCC: {list(arr.shape)}</div>', unsafe_allow_html=True)
                        mfcc_bytes_val = raw
                    except Exception as e:
                        st.markdown(f'<div style="font-size:0.8rem;color:#881337;margin-top:0.3rem;">✗ MFCC error: {e}</div>', unsafe_allow_html=True)
                        shape_ok = False
                else:
                    st.markdown('<div style="font-size:0.76rem;color:#6b7280;margin-top:0.3rem;">No MFCC → zero-filled (NA)</div>', unsafe_allow_html=True)

                if (spec_up or mfcc_up) and shape_ok:
                    captured   = True
                    input_mode = "npy_bypass"
                    input_data = {
                        "spec_bytes":    spec_bytes_val,
                        "mfcc_bytes":    mfcc_bytes_val,
                        "spec_provided": spec_bytes_val is not None,
                        "mfcc_provided": mfcc_bytes_val is not None,
                    }

            with tab_audio:
                st.markdown(
                    '<div class="note-box">Upload a WAV/MP3/M4A/OGG/FLAC recording. '
                    'The server runs the full B2AI preprocessing pipeline: resample → 16 kHz → '
                    'peak-normalise → strip silence → spectrogram [201,T] + MFCC [60,T]. '
                    'Quality flags (SNR, clipping, duration) appear in the results.</div>',
                    unsafe_allow_html=True,
                )
                audio_up = st.file_uploader(
                    "Voice recording", type=["wav","mp3","m4a","ogg","flac"],
                    key=f"audio_{current_idx}",
                )
                if audio_up:
                    _audio_bytes = audio_up.read()
                    st.audio(io.BytesIO(_audio_bytes))
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#065a4a;margin-top:0.4rem;">'
                        f'&#10003; <strong>{audio_up.name}</strong> &nbsp;·&nbsp; '
                        f'{len(_audio_bytes)/1024:.1f} KB &nbsp;·&nbsp; '
                        f'B2AI preprocessing will run server-side</div>',
                        unsafe_allow_html=True,
                    )
                    captured   = True
                    input_mode = "audio_upload"
                    input_data = {
                        "audio_bytes": _audio_bytes,
                        "filename":    audio_up.name,
                        "size_kb":     round(len(_audio_bytes)/1024, 1),
                    }

            with tab_test:
                st.markdown(
                    '<div class="note-box">'
                    '<strong>Load a test sample</strong> from <code>test_data/</code> generated by '
                    '<code>generate_test_data.py</code>. Each sample has a pre-computed '
                    '<code>spec.npy [1,201,256]</code> and <code>mfcc.npy [1,60,256]</code> '
                    'drawn from B2AI statistics.</div>',
                    unsafe_allow_html=True,
                )
                test_dir = Path("test_data")
                manifest_path = test_dir / "manifest.json"
                if not manifest_path.exists():
                    st.markdown(
                        '<div class="warn-box">&#9888; No test data found. '
                        'Run <code>python generate_test_data.py</code> first to generate samples.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    try:
                        with open(manifest_path) as _mf:
                            _manifest = json.load(_mf)
                        _sample_ids = [s["sample_id"] for s in _manifest.get("samples", [])]
                        _selected   = st.selectbox(
                            "Choose test sample", _sample_ids, key=f"test_sel_{current_idx}",
                        )
                        if _selected:
                            _sdir = test_dir / _selected
                            _smeta_path = _sdir / "metadata.json"
                            if _smeta_path.exists():
                                with open(_smeta_path) as _sf:
                                    _smeta = json.load(_sf)
                                st.markdown(
                                    f'<div style="font-size:0.8rem;color:#475569;margin-top:0.4rem;">'
                                    f'Expected: <strong>{_smeta.get("expected_disease","?")}</strong> &nbsp;·&nbsp; '
                                    f'Age: {_smeta.get("patient",{}).get("age","?")} &nbsp;·&nbsp; '
                                    f'Sex: {_smeta.get("patient",{}).get("sex","?")} &nbsp;·&nbsp; '
                                    f'Source: <code>{_smeta.get("source","?")}</code></div>',
                                    unsafe_allow_html=True,
                                )
                            if st.button("Load this sample", key=f"test_load_{current_idx}", use_container_width=True):
                                _spec_bytes = (_sdir / "spec.npy").read_bytes()
                                _mfcc_bytes = (_sdir / "mfcc.npy").read_bytes()
                                # Validate shapes
                                _sv = np.load(io.BytesIO(_spec_bytes))
                                _mv = np.load(io.BytesIO(_mfcc_bytes))
                                st.session_state.task_inputs[current_idx] = {
                                    "mode": "npy_bypass",
                                    "data": {
                                        "spec_bytes":    _spec_bytes,
                                        "mfcc_bytes":    _mfcc_bytes,
                                        "spec_provided": True,
                                        "mfcc_provided": True,
                                        "test_sample":   _selected,
                                    },
                                }
                                append_session_log("task_recorded", {
                                    "task_idx": current_idx, "task_name": title,
                                    "mode": "npy_bypass", "test_sample": _selected,
                                    "spec_shape": list(_sv.shape), "mfcc_shape": list(_mv.shape),
                                })
                                st.session_state.tasks_done.append(current_idx)
                                st.session_state[rec_key] = False
                                st.success(f"Loaded {_selected} — spec{list(_sv.shape)} mfcc{list(_mv.shape)}")
                                st.rerun()
                    except Exception as _e:
                        st.error(f"Error loading test data: {_e}")

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            if captured:
                st.session_state.task_inputs[current_idx] = {"mode": input_mode, "data": input_data}
                if st.button("Save & Next Task", key=f"save_{current_idx}", use_container_width=True):
                    append_session_log("task_recorded", {
                        "task_idx": current_idx, "task_name": title, "mode": input_mode,
                    })
                    st.session_state.tasks_done.append(current_idx)
                    st.session_state[rec_key] = False
                    st.rerun()
            else:
                st.markdown('<div style="background:#f8fafc;border-radius:8px;padding:1.2rem;text-align:center;color:#94a3b8;font-size:0.85rem;">Waiting for input...</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem;">
              <div style="font-size:2.5rem;">&#10003;</div>
              <div style="font-weight:700;color:#028090;font-size:1.1rem;margin-top:0.5rem;">All tasks complete!</div>
              <div style="color:#64748b;font-size:0.85rem;margin-top:0.4rem;">Ready to preprocess and score.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1.8, 4])
    with c1:
        if st.button("Back", key="back3"):
            st.session_state.stage = 2; st.rerun()
    with c2:
        if len(done) == len(tasks):
            if st.button("Run Preprocessing and Scoring", key="fwd3"):
                st.session_state.stage = 4; st.rerun()
        else:
            n = len(tasks) - len(done)
            st.markdown(f'<div style="font-size:0.82rem;color:#94a3b8;padding-top:0.6rem;">{n} task(s) remaining</div>', unsafe_allow_html=True)


def stage_preprocessing():
    fam = st.session_state.task_family
    p   = st.session_state.patient

    st.markdown('<div class="section-title">Stage 4 &middot; Server-Side Audio Preprocessing</div>', unsafe_allow_html=True)
    st.markdown('<div class="teal-line"></div>', unsafe_allow_html=True)

    st.markdown("**Captured inputs per task:**")
    pills = ""
    for idx, (title, _, _) in enumerate(TASK_BATTERIES[fam]["tasks"]):
        ti   = st.session_state.task_inputs.get(idx, {})
        mode = ti.get("mode", "missing")
        mc   = {"simulate":"#028090","npy_bypass":"#7c3aed","audio_upload":"#0369a1","missing":"#ef4444"}.get(mode,"#94a3b8")
        na_note = ""
        if mode == "npy_bypass":
            sp = "spec:ok" if ti.get("data",{}).get("spec_provided") else "spec:NA"
            mf = "mfcc:ok" if ti.get("data",{}).get("mfcc_provided") else "mfcc:NA"
            na_note = f" ({sp} {mf})"
        pills += f'<span class="info-pill" style="border-color:{mc};color:{mc};">Task {idx+1}: {mode}{na_note}</span>'
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    col_steps, col_config = st.columns([1.3, 1], gap="large")

    with col_steps:
        steps = [
            ("#028090","1","Resample and Normalize",
             "Resample to 16 kHz mono. Peak-normalize to -3 dBFS. Strip head/tail silence (threshold: -40 dBFS, min 0.5 sec). npy-bypass inputs skip this step."),
            ("#0369a1","2","Feature Extraction (parallel)",
             f"MFCC (40 coeff, 25 ms window, 10 ms hop) + Log-Mel spectrogram (128 bins). "
             f"Both padded/trimmed to {FIXED_T} time frames to match model input [1, freq, {FIXED_T}]."),
            ("#7c3aed","3","Confounder Metadata Attach",
             f"Age ({p.get('age','?')}), sex ({p.get('sex','?')}), country ({p.get('country','?')}), ethnicity from Stage 1 form. "
             f"Task histogram from session record. Attached for audit; not fed into MARVEL backbone."),
            ("#b45309","4","Protocol Selection",
             "Protocol A = Task-Specific Binary (MARVEL-compatible). Protocol B = Unified Screening (deployment-readiness default). "
             "Protocol B is recommended for this tool."),
        ]
        for color, num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-bottom:1.3rem;align-items:flex-start;">
              <div style="min-width:36px;height:36px;border-radius:50%;background:{color};color:white;
                          font-weight:700;font-size:0.9rem;display:flex;align-items:center;justify-content:center;">{num}</div>
              <div>
                <div style="font-weight:700;color:{color};font-size:0.95rem;margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.83rem;color:#475569;line-height:1.55;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d2233;border:1px solid #028090;border-radius:8px;padding:0.8rem 1rem;
                    font-family:monospace;font-size:0.82rem;color:#02c39a;">
          Output &rarr; { spec_tensor [1,1,201,256], mfcc_tensor [1,1,60,256], confounder_vector, task_histogram, patient_uid }
        </div>
        """, unsafe_allow_html=True)

    with col_config:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Configuration**")
        protocol_choice = st.radio("Evaluation Protocol", [
            "Protocol B -- Unified Screening (recommended)",
            "Protocol A -- Task-Specific Binary",
        ], index=0)
        st.session_state.protocol = "B" if "B" in protocol_choice else "A"
        st.selectbox("Model Checkpoint", [
            "MARVEL (comparator)",
            "Main Model (CNN + sigmoid head)",
            "Both (comparison mode)",
        ])
        st.markdown(f"""
        <div style="font-size:0.83rem;color:#1e293b;line-height:1.9;margin-top:1rem;">
          Recordings: <strong>{len(TASK_BATTERIES[fam]["tasks"])} files</strong><br>
          Spec input: <strong>[1, 1, {SPEC_FREQ}, {FIXED_T}]</strong><br>
          MFCC input: <strong>[1, 1, {MFCC_FREQ}, {FIXED_T}]</strong><br>
          Protocol: <strong>Protocol {st.session_state.protocol}</strong><br>
          API endpoint: <code>{API_BASE_URL}/predict</code>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1.8, 4])
    with c1:
        if st.button("Back", key="back4"):
            st.session_state.stage = 3; st.rerun()
    with c2:
        if st.button("Run Model Scoring", key="fwd4"):
            _run_inference(fam)


def _run_inference(fam: str):
    p = st.session_state.patient

    # Scan all task inputs to collect the best available data
    # Priority: live_mic / audio_upload > npy_bypass > simulate (zeros)
    audio_bytes_to_send: Optional[bytes] = None
    audio_filename = "recording.wav"
    spec_bytes = mfcc_bytes = None

    for idx in sorted(st.session_state.task_inputs.keys()):
        ti   = st.session_state.task_inputs[idx]
        mode = ti.get("mode", "")
        d    = ti.get("data", {})
        if mode in ("live_mic", "audio_upload") and audio_bytes_to_send is None:
            audio_bytes_to_send = d.get("audio_bytes")
            audio_filename      = d.get("filename", "recording.wav")
        elif mode == "npy_bypass" and audio_bytes_to_send is None:
            # Only use npy if we haven't found real audio
            if spec_bytes is None: spec_bytes = d.get("spec_bytes")
            if mfcc_bytes is None: mfcc_bytes = d.get("mfcc_bytes")

    api_ok = api_health()
    input_mode_used = (
        "audio_preprocessed" if audio_bytes_to_send else
        "npy_bypass"         if (spec_bytes or mfcc_bytes) else
        "none"
    )

    with st.spinner(
        "Preprocessing audio & running inference..."
        if audio_bytes_to_send else "Running inference..."
    ):
        if api_ok:
            try:
                if audio_bytes_to_send:
                    # Full B2AI preprocessing pipeline (audio_preprocessing.py on server)
                    response = call_predict_audio(
                        audio_bytes_to_send, p, st.session_state.protocol, audio_filename,
                    )
                else:
                    # .npy bypass or zero inputs
                    response = call_predict(spec_bytes, mfcc_bytes, p, st.session_state.protocol)
                preds     = response.get("predictions", [])
                api_flags = response.get("audit_flags", [])
                st.session_state.api_response = response
                source    = "api"
            except Exception as e:
                st.warning(f"API call failed: {e} — falling back to mock results.")
                preds     = mock_results(fam)
                api_flags = [{"level":"warn","msg":f"API call failed ({e}) — showing mock predictions"}]
                source    = "mock"
        else:
            preds     = mock_results(fam)
            api_flags = [{"level":"warn","msg":f"API offline at {API_BASE_URL} — showing mock predictions"}]
            source    = "mock"

    # ── Battery-weighted rescoring ────────────────────────────────────────────
    # When a specific task battery is selected, boost diseases in that battery
    # to reflect the clinical context (task selection encodes prior probability)
    BATTERY_DISEASES = {
        "Structural / Motor":      ["parkinsons","laryngeal_dystonia","vf_paralysis","airway_stenosis"],
        "Laryngeal / Vocal":       ["chronic_cough","mtd","benign_lesions","glottic_insuff"],
        "Psychiatric / Cognitive": ["depression","ptsd","adhd","bipolar","cognitive_impairment",
                                    "psychiatric_history","anxiety"],
    }
    BATTERY_BOOST = 0.18   # add this to probability for diseases in the selected battery
    battery_tasks = BATTERY_DISEASES.get(fam, [])
    rescored = []
    for pred in preds:
        p_copy = dict(pred)
        if p_copy.get("task") in battery_tasks:
            raw_prob  = p_copy.get("probability", 0)
            boosted   = min(0.99, raw_prob + BATTERY_BOOST * (1.0 - raw_prob))  # asymptotic boost
            p_copy["probability"]       = round(boosted, 4)
            p_copy["percent"]           = round(boosted * 100, 1)
            p_copy["battery_boosted"]   = True
            # Recalculate gap with boosted probability
            cp = p_copy.get("confounder_prob", 0)
            p_copy["gap"] = round(boosted - cp, 4)
        else:
            p_copy["battery_boosted"] = False
        rescored.append(p_copy)
    # Re-sort by boosted probability
    preds = sorted(rescored, key=lambda x: x.get("probability", 0), reverse=True)

    st.session_state.results = preds
    append_session_log("inference_complete", {
        "source":         source,
        "input_mode":     input_mode_used,
        "n_predictions":  len(preds),
        "api_flags":      api_flags,
        "protocol":       st.session_state.protocol,
        "audio_provided": audio_bytes_to_send is not None,
        "spec_provided":  spec_bytes is not None,
        "mfcc_provided":  mfcc_bytes is not None,
    })
    st.session_state.stage = 5
    st.rerun()


def stage_results():
    fam   = st.session_state.task_family
    preds = st.session_state.results or []
    p     = st.session_state.patient
    api_r = st.session_state.api_response or {}

    st.markdown('<div class="section-title">Stage 5 &middot; Clinical Prediction Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="teal-line"></div>', unsafe_allow_html=True)

    src_label_map = {
        "audio_preprocessed": "Live Audio (B2AI pipeline)",
        "browser_mic":        "Browser Mic",
        "npy_bypass":         ".npy Bypass",
        "none":               "Zero Input (simulate)",
    }
    input_src  = api_r.get("input_source", "")
    source_lbl = src_label_map.get(input_src, "Live API" if api_r else "Mock (API offline)")
    st.markdown(f"""
    <div style="margin-bottom:1.2rem;">
      <span class="info-pill">Patient: {p['uid']}</span>
      <span class="info-pill">Protocol: {st.session_state.protocol}</span>
      <span class="info-pill">Battery: {fam}</span>
      <span class="info-pill">Source: {source_lbl}</span>
      <span class="info-pill">Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
    </div>
    """, unsafe_allow_html=True)

    # Quick link to Clinical AI analysis
    st.markdown(
        '<div class="note-box">✨ <strong>Results ready.</strong> '
        'Head to <strong>Clinical AI</strong> page to run the 4-specialist multi-agent analysis '
        'on these predictions.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    col_report, col_audit = st.columns([1, 1.1], gap="large")

    with col_report:
        # ── Top-3 prediction cards ──────────────────────────────────────────
        top3     = preds[:3]
        has_conf = any("confounder_prob" in r for r in top3)
        latency  = api_r.get("latency_ms", "--")
        src_lbl2 = api_r.get("input_source", "")
        src_icon = {"audio_preprocessed":"🎵","browser_mic":"🎙️","npy_bypass":"📦","none":"⚪"}.get(src_lbl2, "📊")

        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'margin-bottom:0.8rem;">'
            f'<div style="font-size:0.85rem;font-weight:600;color:#111827;">Top Predictions</div>'
            f'<div style="font-size:0.72rem;color:#9ca3af;">{src_icon} {source_lbl} &nbsp;·&nbsp; {latency} ms</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Top-1 prediction card ───────────────────────────────────────────
        top1 = preds[0] if preds else {}
        if top1:
            _prob  = top1.get("probability", 0)
            _pct   = top1.get("percent", round(_prob * 100, 1))
            _na    = top1.get("na_flag", False)
            _task  = top1.get("task", "?").replace("_", " ").title()
            _cp    = top1.get("confounder_prob")
            _gap   = top1.get("gap")

            if _na:
                _conf_txt, _conf_color = "N/A — missing input", "#9ca3af"
            elif _prob >= 0.70:
                _conf_txt, _conf_color = "High confidence (≥70%)", "#059669"
            elif _prob >= 0.50:
                _conf_txt, _conf_color = "Moderate confidence (50–70%)", "#d97706"
            elif _prob >= 0.30:
                _conf_txt, _conf_color = "Weak signal (30–50%)", "#6b7280"
            else:
                _conf_txt, _conf_color = "No signal (<30%)", "#9ca3af"

            _conf_html = ""
            if _cp is not None and not _na and _gap is not None:
                if _gap >= 0.10:
                    _gap_style = "color:#059669;font-weight:600;"
                    _gap_text  = f"Model leads by +{_gap:.3f} — acoustic signal present"
                elif _gap >= 0:
                    _gap_style = "color:#d97706;font-weight:600;"
                    _gap_text  = f"Gap +{_gap:.3f} — shortcut risk (threshold 0.10)"
                else:
                    _gap_style = "color:#dc2626;font-weight:600;"
                    _gap_text  = f"Gap {_gap:+.3f} — confounder {_cp*100:.0f}% exceeds model"
                _conf_html = (
                    f'<div style="margin-top:0.5rem;font-size:0.75rem;color:#6b7280;">'
                    f'Confounder audit (Rec. 5.1) &nbsp;·&nbsp; '
                    f'<span style="{_gap_style}">{_gap_text}</span></div>'
                )

            _boosted_html = ""
            if top1.get("battery_boosted"):
                _boosted_html = (
                    f'<div style="font-size:0.72rem;color:#0369a1;margin-top:0.3rem;'
                    f'background:#eff6ff;border:1px solid #bfdbfe;border-radius:4px;'
                    f'padding:3px 8px;display:inline-block;">'
                    f'⚡ Battery boost applied — {fam} battery selected</div>'
                )
            st.markdown(
                f'<div style="border:1px solid #e5e7eb;border-left:5px solid #028090;'
                f'border-radius:10px;padding:1.3rem 1.5rem;background:white;margin-bottom:0.8rem;">'
                f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.08em;color:#6b7280;margin-bottom:0.3rem;">#1 · Best match</div>'
                f'<div style="display:flex;align-items:flex-end;justify-content:space-between;'
                f'gap:1rem;margin-bottom:0.5rem;">'
                f'<div style="font-size:1.3rem;font-weight:700;color:#111827;">{"N/A" if _na else _task}</div>'
                f'<div style="font-size:2.2rem;font-weight:800;color:#028090;line-height:1;">'
                f'{"N/A" if _na else f"{_pct:.0f}%"}</div></div>'
                f'<div style="background:#f3f4f6;border-radius:4px;height:8px;margin-bottom:0.4rem;">'
                f'<div style="width:{0 if _na else min(_pct,100)}%;height:8px;border-radius:4px;background:#028090;"></div></div>'
                f'<div style="font-size:0.8rem;font-weight:600;color:{_conf_color};">{_conf_txt}</div>'
                f'{_boosted_html}{_conf_html}</div>',
                unsafe_allow_html=True,
            )

        # ── Top 3 predictions with explainability ────────────────────────
        _rank_colors  = ["#028090", "#0369a1", "#7c3aed"]
        _rank_labels  = ["#1 Primary signal", "#2 Secondary signal", "#3 Tertiary signal"]
        for _ri, _rpred in enumerate(preds[1:3], 1):
            _rt    = _rpred.get("task","?").replace("_"," ").title()
            _rp2   = _rpred.get("percent", round(_rpred.get("probability",0)*100,1))
            _rcp   = _rpred.get("confounder_prob")
            _rgap  = _rpred.get("gap")
            _rna   = _rpred.get("na_flag",False)
            _rconf = ("High" if _rp2>=70 else ("Moderate" if _rp2>=45 else "Low"))
            _rcc   = ("#059669" if _rp2>=70 else ("#d97706" if _rp2>=45 else "#9ca3af"))
            _rc    = _rank_colors[_ri]

            _rgap_html = ""
            if _rcp is not None and not _rna and _rgap is not None:
                if _rgap >= 0.10:
                    _rgap_html = f'<span style="color:#059669;font-size:0.72rem;"> · Acoustic signal ✓</span>'
                elif _rgap >= 0:
                    _rgap_html = f'<span style="color:#d97706;font-size:0.72rem;"> · Shortcut risk ⚠</span>'
                else:
                    _rgap_html = f'<span style="color:#dc2626;font-size:0.72rem;"> · Confound ✗</span>'

            st.markdown(
                f'<div style="border:1px solid #e5e7eb;border-left:4px solid {_rc};'
                f'border-radius:8px;padding:0.9rem 1.1rem;background:white;margin-bottom:0.5rem;">'
                f'<div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.06em;color:#9ca3af;margin-bottom:0.15rem;">{_rank_labels[_ri]}</div>'
                f'<div style="display:flex;align-items:flex-end;justify-content:space-between;gap:0.5rem;">'
                f'<div style="font-size:0.95rem;font-weight:700;color:#111827;">{_rt}</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{_rc};line-height:1;">{_rp2:.0f}%</div>'
                f'</div>'
                f'<div style="background:#f3f4f6;border-radius:3px;height:4px;margin:0.3rem 0;">'
                f'<div style="width:{_rp2}%;height:4px;border-radius:3px;background:{_rc};"></div></div>'
                f'<div style="font-size:0.76rem;font-weight:600;color:{_rcc};">{_rconf} confidence{_rgap_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Explainability for top 3 ─────────────────────────────────────
        with st.expander("🔍 Explainability — why these top 3?", expanded=False):
            for _ei, _epred in enumerate(preds[:3]):
                _etask = _epred.get("task","?")
                _eprob = _epred.get("probability", 0)
                _ecp   = _epred.get("confounder_prob", 0)
                _egap  = _epred.get("gap")
                _expl  = _build_explainability(_etask, _eprob, _ecp, _egap, fam)
                st.markdown(_expl)
                if _ei < 2:
                    st.markdown("<hr style='margin:0.5rem 0;border-color:#f3f4f6;'>", unsafe_allow_html=True)

        if has_conf:
            _conf_mode = api_r.get("confounder_mode", "heuristic")
            _mode_note = ("AUROC values from audit report — no trained baseline checkpoint loaded"
                          if _conf_mode == "heuristic" else "trained logistic regression")
            st.markdown(
                f'<div style="font-size:0.7rem;color:#9ca3af;margin-bottom:0.8rem;">'
                f'Confounder: <strong>{_conf_mode}</strong> — {_mode_note}</div>',
                unsafe_allow_html=True,
            )

        # ── Flags: top-disease only visible; all others collapsed ───────────
        _pp_info   = api_r.get("inputs", {})
        _pp_flags  = _pp_info.get("preprocess_flags", [])
        _pp_det    = api_r.get("preprocessing", {})
        _api_flags = api_r.get("audit_flags", [])

        _top_task_name = top1.get("task", "") if top1 else ""
        _top_flags  = [f for f in _api_flags if f.get("task") == _top_task_name]
        _rest_flags = [f for f in _api_flags if f.get("task") != _top_task_name]
        _other_n    = len(_rest_flags)

        if _pp_det or _pp_flags:
            with st.expander("Preprocessing quality", expanded=bool(_pp_flags)):
                if _pp_det:
                    _d = _pp_det
                    st.write(f'SR: {_d.get("sr",16000)} Hz · Duration: {_d.get("duration_s","?")} s · Frames: {_d.get("n_frames","?")}')
                for _fm in _pp_flags:
                    _is_w = any(k in _fm for k in ("clipping","low_snr","too_short"))
                    st.markdown(
                        f'<div class="audit-flag {"audit-warn" if _is_w else "audit-ok"}">🔊 {_fm}</div>',
                        unsafe_allow_html=True,
                    )

        for _f in _top_flags:
            _ftype = _f.get("type","")
            _cls   = ("audit-warn" if _ftype == "baseline_exceeds_model" else "audit-amber")
            st.markdown(f'<div class="audit-flag {_cls}">{_f.get("msg","")}</div>', unsafe_allow_html=True)

        if _rest_flags:
            with st.expander(f"Full confounder audit — {_other_n} other disease{'s' if _other_n!=1 else ''}", expanded=False):
                st.markdown(
                    '<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;'
                    'padding:0.75rem 1rem;margin-bottom:0.8rem;font-size:0.82rem;color:#0c4a6e;">' 
                    '<strong>📖 What is the confounder audit?</strong><br><br>'
                    'The <strong>confounder baseline</strong> is a simple demographic model (age, sex, country, ethnicity) '
                    'that predicts each disease <em>without any voice data</em>. We compare it to the MARVEL voice model.<br><br>'
                    '<strong>Why does it matter?</strong> If demographics alone can predict a disease as well as voice, '
                    'the voice model may be learning a shortcut (e.g. "older Canadian = cognitive impairment") '
                    'rather than genuine acoustic signals.<br><br>'
                    '<strong>What you are seeing:</strong> Since no trained MARVEL checkpoint is loaded, the model is using <em>random weights</em>. '
                    'The heuristic AUROC proxies are from the B2AI audit report (Table 2.1) — they show the '
                    'expected gap when a real trained model is loaded. These warnings are <em>expected</em> with random weights '
                    'and disappear with a proper trained checkpoint.</div>',
                    unsafe_allow_html=True,
                )
                # Group by type for clarity
                _exceeds = [f for f in _rest_flags if f.get("type") == "baseline_exceeds_model"]
                _shortcut = [f for f in _rest_flags if f.get("type") == "shortcut_risk"]
                if _exceeds:
                    st.markdown('<div style="font-size:0.76rem;font-weight:600;color:#881337;margin:0.4rem 0 0.2rem 0;">'
                                'Demographics ≥ Voice model (random weights expected):</div>', unsafe_allow_html=True)
                    for _f in _exceeds:
                        _tl = _f.get("task","?").replace("_"," ").title()
                        _cp = _f.get("confounder_prob",0)
                        _mp = _f.get("model_prob",0)
                        st.markdown(
                            f'<div style="background:#fff1f2;border:1px solid #fecdd3;border-radius:5px;'
                            f'padding:4px 8px;margin:2px 0;font-size:0.76rem;color:#881337;">'
                            f'<strong>{_tl}</strong> — demographics {_cp*100:.0f}% vs model {_mp*100:.0f}% '
                            f'(gap: {(_mp-_cp)*100:+.1f}pp)</div>',
                            unsafe_allow_html=True,
                        )
                if _shortcut:
                    st.markdown('<div style="font-size:0.76rem;font-weight:600;color:#78350f;margin:0.6rem 0 0.2rem 0;">'
                                'Shortcut risk — gap below 0.10 threshold:</div>', unsafe_allow_html=True)
                    for _f in _shortcut:
                        _tl = _f.get("task","?").replace("_"," ").title()
                        _cp = _f.get("confounder_prob",0)
                        _mp = _f.get("model_prob",0)
                        st.markdown(
                            f'<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:5px;'
                            f'padding:4px 8px;margin:2px 0;font-size:0.76rem;color:#78350f;">'
                            f'<strong>{_tl}</strong> — gap only {(_mp-_cp)*100:+.1f}pp (need ≥10pp)</div>',
                            unsafe_allow_html=True,
                        )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        # ── Subgroup AUROC (inline styles) ──────────────────────────────────
        with st.expander("Demographic Subgroup AUROC (Rec. 5.3)", expanded=False):
            _sg_rows = SUBGROUP_DATA.get(fam, [])
            if _sg_rows:
                _tbl = (
                    '<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">'
                    '<thead><tr>'
                    + "".join(
                        f'<th style="background:#f9fafb;color:#374151;font-weight:600;'
                        f'padding:0.4rem 0.6rem;text-align:left;border-bottom:1px solid #e5e7eb;">{h}</th>'
                        for h in ("Disease", "Subgroup", "AUROC")
                    )
                    + '</tr></thead><tbody>'
                )
                for _dis, _grp, _auc, _sta in _sg_rows:
                    _c = {"ok":"#059669","warn":"#dc2626"}.get(_sta,"#d97706")
                    _tbl += (
                        f'<tr>'
                        f'<td style="padding:0.35rem 0.6rem;border-bottom:1px solid #f3f4f6;">{_dis}</td>'
                        f'<td style="padding:0.35rem 0.6rem;border-bottom:1px solid #f3f4f6;">{_grp}</td>'
                        f'<td style="padding:0.35rem 0.6rem;border-bottom:1px solid #f3f4f6;'
                        f'font-weight:600;color:{_c};">{_auc}</td></tr>'
                    )
                _tbl += "</tbody></table>"
                st.markdown(_tbl, unsafe_allow_html=True)

        with st.expander("Protocol A vs B Gap", expanded=False):
            for _dis, _pA, _pB, _gap in PROTO_DATA.get(fam, []):
                _clr = "#dc2626" if abs(_gap) > 0.10 else "#059669"
                st.markdown(
                    f'<div style="font-size:0.82rem;margin-bottom:0.4rem;">'
                    f'<strong style="color:#111827;">{_dis}</strong> &nbsp;'
                    f'<span style="color:#6b7280;">A: {_pA:.3f} &nbsp; B: {_pB:.3f}</span> &nbsp;'
                    f'<span style="font-weight:700;color:{_clr};">Δ {_gap:+.3f}</span></div>',
                    unsafe_allow_html=True,
                )

    with col_audit:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Audit Flags** <span style='font-size:0.72rem;color:#6b7280;'>(Recs. 5.1–5.5)</span>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.75rem;color:#374151;background:#f8fafc;border-radius:6px;'
            'padding:0.5rem 0.7rem;margin-bottom:0.5rem;border:1px solid #e2e8f0;">'
            'These flags are from the B2AI-Voice audit report for the <strong>' + fam + '</strong> battery. '
            'They reflect known model limitations regardless of your specific recording.</div>',
            unsafe_allow_html=True,
        )
        for flag_type, msg in AUDIT_FLAGS_STATIC.get(fam, []):
            cls = {"ok":"audit-ok","warn":"audit-warn","amber":"audit-amber"}.get(flag_type,"audit-amber")
            st.markdown(f'<div class="audit-flag {cls}">{msg}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Deployment Gates** <span style='font-size:0.72rem;color:#6b7280;'>(Rec. 5.6)</span>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.75rem;color:#374151;background:#f8fafc;border-radius:6px;'
            'padding:0.5rem 0.7rem;margin-bottom:0.5rem;border:1px solid #e2e8f0;">'
            '4 gates must all pass before the model is safe to deploy clinically. '
            '<strong>None currently pass for B2AI v3.</strong> This is a research system only.</div>',
            unsafe_allow_html=True,
        )
        gate_explanations = {
            "Gate 1": "Evaluated on an external dataset not seen during training — ensures results generalise to new hospitals and populations.",
            "Gate 2": "Voice model must beat a demographics-only model by ≥10% AUROC — proves acoustic signal adds value beyond knowing age/sex.",
            "Gate 3": "No demographic subgroup (age, sex, country) should score below 0.65 AUROC — ensures fair performance across all patients.",
            "Gate 4": "Full screening protocol (all tasks) must score within 10% of the task-specific binary — ensures the model is stable under realistic conditions.",
        }
        for gate_id, title, req, passed, note in DEPLOYMENT_GATES:
            icon  = "✓" if passed is True else ("✗" if passed is False else "⚠")
            bg    = "#f0fdf4" if passed is True else ("#fff1f2" if passed is False else "#fffbeb")
            border= "#bbf7d0" if passed is True else ("#fecdd3" if passed is False else "#fde68a")
            color = "#059669" if passed is True else ("#dc2626" if passed is False else "#d97706")
            txt_c = "#14532d" if passed is True else ("#881337" if passed is False else "#78350f")
            expl  = gate_explanations.get(gate_id, "")
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:8px;
                        padding:0.6rem 0.8rem;margin-bottom:0.5rem;">
              <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.2rem;">
                <div style="font-weight:700;color:{color};font-size:1rem;">{icon}</div>
                <div style="font-size:0.85rem;font-weight:600;color:{txt_c};">{gate_id}: {title}</div>
              </div>
              <div style="font-size:0.73rem;color:{txt_c};opacity:0.85;margin-bottom:0.2rem;">{expl}</div>
              <div style="font-size:0.73rem;color:{color};font-weight:600;">{note}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Session Log**")
        log_path = SESSIONS_DIR / f"{p['uid']}.json"
        st.markdown(f'<div style="font-size:0.72rem;color:#9ca3af;margin-bottom:0.4rem;">Saved: <code style="color:#028090;">{log_path}</code></div>', unsafe_allow_html=True)
        icons = {"intake_complete":"👤","task_family_selected":"📋","task_recorded":"🎙️","inference_complete":"🧠","npz_loaded":"📦","npz_skip_to_inference":"⚡"}
        for entry in st.session_state.session_log:
            ico = icons.get(entry["event"],"📌")
            ts  = entry["timestamp"][11:19]
            st.markdown(f'<div style="font-size:0.76rem;color:#374151;padding:0.15rem 0;border-bottom:1px solid #f9fafb;">{ico} <strong>{entry["event"]}</strong> <span style="color:#9ca3af;margin-left:0.4rem;">{ts}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="warn-box">&#9888; These predictions are <strong>research output only</strong>. No model satisfies all four deployment gates for any disease in B2AI v3. Predictions must not replace clinical evaluation.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    c1, c2, c3, _ = st.columns([1, 1.8, 1.5, 2])
    with c1:
        if st.button("Back", key="back5"):
            st.session_state.stage = 4; st.rerun()
    with c2:
        full_report = {
            "session_id":    p["uid"],
            "timestamp":     datetime.now().isoformat(),
            "protocol":      st.session_state.protocol,
            "task_battery":  fam,
            "patient":       {k: v for k, v in p.items() if k != "uid"},
            "predictions":   preds,
            "api_response":  api_r,
            "audit_flags":   AUDIT_FLAGS_STATIC.get(fam, []),
            "deployment_gates": {g[0]: {"title": g[1], "note": g[4]} for g in DEPLOYMENT_GATES},
            "session_log":   st.session_state.session_log,
        }
        st.download_button(
            "Download Report (JSON)",
            data=json.dumps(full_report, indent=2, default=str),
            file_name=f"voxclinbench_{p['uid']}.json",
            mime="application/json",
        )
    with c3:
        if st.button("New Patient", key="new_patient"):
            for k in ["stage","patient","task_family","tasks_done","task_inputs",
                      "results","api_response","session_log"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()


def page_assessment():
    stepper(st.session_state.stage)
    {
        1: stage_intake,
        2: stage_task_assignment,
        3: stage_recording,
        4: stage_preprocessing,
        5: stage_results,
    }[st.session_state.stage]()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLINICAL AI CHAT  (4-specialist multi-agent pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def page_chat():
    st.markdown('<div class="section-title">Clinical AI &middot; Multi-Agent Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="teal-line"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">4-specialist pipeline — Pharmacology · Clinical Guidelines · '
        'Rehabilitation · Comorbidity — each with Tavily search + Milvus RAG · Master synthesiser</div>',
        unsafe_allow_html=True,
    )

    # ── Inline API key inputs ─────────────────────────────────────────────────
    with st.expander("🔑 API Keys", expanded=not (
        st.session_state.cfg_anthropic_key or os.environ.get("ANTHROPIC_API_KEY","") or
        (st.session_state.cfg_use_gateway and st.session_state.cfg_gateway_key)
    )):
        k1, k2, k3 = st.columns(3)
        with k1:
            _ant = st.text_input(
                "Anthropic API key",
                value=st.session_state.cfg_anthropic_key,
                type="password", placeholder="Anthropic API key",
                key="chat_ant_key",
                help="Required for agent LLM calls. Set ANTHROPIC_API_KEY env var or enter here.",
            )
            if _ant != st.session_state.cfg_anthropic_key:
                st.session_state.cfg_anthropic_key = _ant
                if _ant: os.environ["ANTHROPIC_API_KEY"] = _ant
        with k2:
            _tav = st.text_input(
                "Tavily API key (optional)",
                value=st.session_state.cfg_tavily_key,
                type="password", placeholder="tvly-...",
                key="chat_tav_key",
                help="Enables web search for each specialist agent.",
            )
            if _tav != st.session_state.cfg_tavily_key:
                st.session_state.cfg_tavily_key = _tav
        with k3:
            _gw_toggle = st.checkbox(
                "Use CMU AI Gateway instead",
                value=st.session_state.cfg_use_gateway,
                key="chat_gw_toggle",
            )
            st.session_state.cfg_use_gateway = _gw_toggle
            if _gw_toggle:
                _gw_key = st.text_input(
                    "Gateway key", value=st.session_state.cfg_gateway_key,
                    type="password", placeholder="sk-...",
                    key="chat_gw_key", label_visibility="collapsed",
                )
                if _gw_key != st.session_state.cfg_gateway_key:
                    st.session_state.cfg_gateway_key = _gw_key

    # ── Patient profile panel ─────────────────────────────────────────────────
    patient  = st.session_state.patient
    preds    = st.session_state.results or []
    fam      = st.session_state.task_family or "Unknown"
    protocol = st.session_state.protocol

    has_assessment = bool(patient) and bool(preds)

    with st.expander("Patient Profile" + (" ✓ (loaded from assessment)" if has_assessment else " — no assessment loaded"), expanded=not has_assessment):
        if has_assessment:
            profile_str = _format_patient_profile(patient, preds, fam, protocol)
            st.markdown(
                f'<div style="background:#f0fdfa;border:1px solid #a7f3d0;border-radius:8px;'
                f'padding:1rem;font-size:0.83rem;color:#065a4a;white-space:pre-wrap;'
                f'font-family:monospace;line-height:1.6;">{profile_str}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No assessment found. Complete the Voice Assessment first, or enter a manual profile below.")
            profile_str = st.text_area(
                "Manual patient profile (optional)",
                placeholder="Patient: Age 65, Male, Parkinson's-like tremor, medications: levodopa.\n"
                            "Predictions: Parkinson's 87%, Laryngeal Dystonia 31%...",
                height=120,
                key="manual_profile",
            )

    # ── Agent config summary ─────────────────────────────────────────────────
    ant_key = (st.session_state.cfg_anthropic_key or os.environ.get("ANTHROPIC_API_KEY",""))
    tav_key = st.session_state.cfg_tavily_key
    idx_ok  = st.session_state.indexer is not None
    gw_ok   = st.session_state.cfg_use_gateway and st.session_state.cfg_gateway_key

    llm_ok  = bool(ant_key or gw_ok)
    llm_lbl = "CMU Gateway" if gw_ok else ("Anthropic" if ant_key else "No key")
    rag_lbl = f"{st.session_state.index_stats.get('total_chunks',0)} chunks" if idx_ok else "No index"
    web_lbl = "Tavily" if tav_key else "No key (skipped)"

    st.markdown(f"""
    <div style="display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap;">
      <span class="info-pill" style="{'background:#f0fdf4;border-color:#bbf7d0;color:#14532d;' if llm_ok else 'background:#fff1f2;border-color:#fecdd3;color:#881337;'}">
        LLM: {llm_lbl}</span>
      <span class="info-pill" style="{'background:#f0fdf4;border-color:#bbf7d0;color:#14532d;' if idx_ok else 'background:#fffbeb;border-color:#fde68a;color:#78350f;'}">
        RAG: {rag_lbl}</span>
      <span class="info-pill" style="{'background:#f0fdf4;border-color:#bbf7d0;color:#14532d;' if tav_key else 'background:#f9fafb;border-color:#e5e7eb;color:#6b7280;'}">
        Search: {web_lbl}</span>
    </div>
    """, unsafe_allow_html=True)

    if not (ant_key or gw_ok):
        st.markdown(
            '<div class="warn-box">⚠ No LLM key found. Set ANTHROPIC_API_KEY in env, '
            'enter it in the sidebar, or enable CMU AI Gateway.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Multi-agent analysis trigger ──────────────────────────────────────────
    run_col, hint_col = st.columns([1, 3])
    with run_col:
        run_btn = st.button(
            "▶ Run Clinical Analysis",
            key="run_agents",
            use_container_width=True,
            disabled=not (ant_key or gw_ok),
        )
    with hint_col:
        st.markdown(
            '<div style="font-size:0.82rem;color:#64748b;padding-top:0.6rem;">'
            'Runs 4 specialist agents in sequence, then the master synthesiser. '
            'Takes ~60–120 s depending on API and Tavily latency.</div>',
            unsafe_allow_html=True,
        )

    if run_btn:
        st.session_state.agent_results  = {}
        st.session_state.final_report   = ""
        st.session_state.analysis_running = True

        actual_profile = (profile_str if not has_assessment
                          else _format_patient_profile(patient, preds, fam, protocol))
        if not actual_profile.strip():
            st.warning("No patient profile available. Complete the assessment or enter a manual profile above.")
        else:
            agents = [
                ("Pharmacology",       _PHARMA_PLAN,       _PHARMA_EXECUTE,       _PHARMA_SYNTHESIS),
                ("Clinical Guidelines",_GUIDELINES_PLAN,   _GUIDELINES_EXECUTE,   _GUIDELINES_SYNTHESIS),
                ("Rehabilitation",     _REHAB_PLAN,        _REHAB_EXECUTE,        _REHAB_SYNTHESIS),
                ("Comorbidity",        _COMORBIDITY_PLAN,  _COMORBIDITY_EXECUTE,  _COMORBIDITY_SYNTHESIS),
            ]

            st.markdown("### Agent Pipeline")
            status_phs = {role: st.empty() for role, *_ in agents}
            master_ph  = st.empty()

            # Create stream placeholders for each agent
            stream_phs = {role: st.empty() for role, *_ in agents}
            for role, plan_tpl, exec_tpl, syn_tpl in agents:
                if st.session_state.get("interrupt_analysis", False):
                    st.info("Analysis interrupted.")
                    break
                report = _run_specialist_agent_streaming(
                    role=role,
                    plan_tpl=plan_tpl,
                    execute_tpl=exec_tpl,
                    synthesis_tpl=syn_tpl,
                    patient_profile=actual_profile,
                    tavily_key=tav_key,
                    status_ph=status_phs[role],
                    stream_ph=stream_phs[role],
                )
                st.session_state.agent_results[role] = report
                # Clear stream placeholder after done (result will show in expander)
                stream_phs[role].empty()

            if not st.session_state.get("interrupt_analysis", False) and len(st.session_state.agent_results) == 4:
                st.markdown("#### 🧠 Master Synthesiser — Generating Final Report")
                master_stream_ph = st.empty()
                final = run_master_agent(
                    patient_profile=actual_profile,
                    agent_reports=st.session_state.agent_results,
                    status_ph=master_ph,
                    stream_ph=master_stream_ph,
                )
                st.session_state.final_report = final
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final,
                    "sources": [],
                    "recs": [],
                })
                _save_chat()

            st.session_state.analysis_running   = False
            st.session_state.interrupt_analysis = False

    # ── Interrupt button ──────────────────────────────────────────────────────
    if st.session_state.analysis_running:
        if st.button("⏹ Stop Analysis", key="stop_agents"):
            st.session_state.interrupt_analysis = True

    # ── Per-agent result expanders with PDF downloads ─────────────────────────
    if st.session_state.agent_results:
        st.markdown("### Specialist Agent Reports")
        agent_icons = {"Pharmacology":"💊","Clinical Guidelines":"📋",
                       "Rehabilitation":"🏃","Comorbidity":"⚠️"}
        for role, report in st.session_state.agent_results.items():
            icon = agent_icons.get(role, "🔬")
            with st.expander(f"{icon} {role} Report", expanded=False):
                st.markdown(report)
                # Per-agent TXT download
                _uid = st.session_state.patient.get("uid","report")
                st.download_button(
                    label=f"⬇ {role} Report (TXT)",
                    data=report,
                    file_name=f"voxclinbench_{_uid}_{role.lower().replace(' ','_')}.txt",
                    mime="text/plain",
                    key=f"dl_agent_{role}",
                )

    # ── Final report ──────────────────────────────────────────────────────────
    if st.session_state.final_report:
        st.markdown("### Master Synthesiser — Final Report")
        st.markdown(
            f'<div class="report-card" style="max-height:650px;overflow-y:auto;">' 
            f'<pre style="white-space:pre-wrap;word-break:break-word;font-size:0.82rem;'
            f'color:#111827;line-height:1.65;margin:0;">{st.session_state.final_report}</pre></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        c1, _ = st.columns([1.2, 3])
        with c1:
            st.download_button(
                "⬇ Report (TXT)",
                data=st.session_state.final_report,
                file_name=f"clinical_report_{st.session_state.uid}.txt",
                mime="text/plain",
                key="dl_final_txt",
            )

    st.markdown("<hr style='border-color:#e2e8f0;margin:1.5rem 0;'>", unsafe_allow_html=True)

    # ── Follow-up RAG chat ────────────────────────────────────────────────────
    st.markdown("### Follow-up Q&A")
    st.markdown(
        '<div style="font-size:0.82rem;color:#64748b;margin-bottom:0.8rem;">'
        'Ask follow-up questions — answered via RAG knowledge base (no re-running agents).</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        _render_chat_msg(msg)

    _queued = st.session_state.pop("_chat_queued", None)
    prompt  = st.chat_input("Ask a follow-up question about the analysis...")
    active  = prompt or _queued

    if st.session_state.get("is_chatting", False):
        if st.button("⏹ Stop", key="stop_chat"):
            st.session_state.chat_interrupt_requested = True

    if active:
        st.session_state.chat_history.append({"role": "user", "content": active})
        _save_chat()
        _render_chat_msg({"role": "user", "content": active})

        answer, sources, recs = "", [], []
        st.session_state.is_chatting = True
        st.session_state.chat_interrupt_requested = False

        if st.session_state.indexer is None:
            # No RAG — use LLM only
            try:
                ctx = st.session_state.final_report or ""
                llm_prompt = (
                    f"Context from prior clinical analysis:\n{ctx[:1000]}\n\n"
                    f"User question: {active}\n\n"
                    f"Answer concisely based on the context."
                ) if ctx else active
                answer = _anthropic_call(llm_prompt, max_tokens=800)
            except Exception as e:
                answer = f"LLM query failed: {e}. Index PDFs in RAG Dev or ensure an API key is set."
            st.markdown(answer)
        else:
            # RAG query
            with st.spinner("Retrieving from knowledge base..."):
                try:
                    if st.session_state.chat_interrupt_requested:
                        answer = "_Response interrupted._"
                    else:
                        idx = st.session_state.indexer
                        wc  = st.session_state.cfg_word_count
                        final_q = active + f"\n\n[RESPONSE LENGTH: ~{wc} words]"

                        # Patch indexer backend
                        if st.session_state.cfg_use_gateway and st.session_state.cfg_gateway_key:
                            _patch_indexer_for_gateway(
                                idx,
                                model=st.session_state.cfg_model,
                                gateway_key=st.session_state.cfg_gateway_key,
                                gateway_url=st.session_state.cfg_gateway_url,
                                word_count=wc,
                            )
                        else:
                            _gkey = (st.session_state.cfg_gemini_api_key or
                                     os.environ.get("GEMINI_API_KEY","") or
                                     os.environ.get("GOOGLE_API_KEY",""))
                            if _gkey:
                                _patch_indexer_for_gemini(idx, model=st.session_state.cfg_model,
                                                           gemini_key=_gkey, word_count=wc)

                        answer, sources = idx.query(
                            final_q,
                            top_k=st.session_state.cfg_top_k,
                            model=st.session_state.cfg_model,
                            output_language=st.session_state.cfg_language,
                            max_iterations=st.session_state.cfg_iterations,
                        )
                        st.markdown(answer)
                        recs = _unique_papers(sources, st.session_state.file_url_map, k=2)
                        _render_recs(recs)
                        _render_sources(sources)
                except Exception as e:
                    answer = f"Query failed: {e}"
                    st.error(answer)
                finally:
                    st.session_state.is_chatting = False
                    st.session_state.chat_interrupt_requested = False

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer,
            "sources": sources, "recs": recs,
        })
        _save_chat()
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat history", key="clear_chat"):
            st.session_state.chat_history = []
            _save_chat()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RAG DEV  (Milvus indexing, all features from obesity_app_v2)
# ══════════════════════════════════════════════════════════════════════════════

def page_rag_dev():
    st.markdown("""
    <div style="text-align:center;padding:28px 0 18px 0;">
      <h1 style="font-size:1.9rem;font-weight:700;color:#1e293b;letter-spacing:-.03em;margin:0 0 6px 0;">
        RAG Dev &middot; Index Clinical PDFs</h1>
      <p style="font-size:0.78rem;color:#64748b;letter-spacing:.06em;text-transform:uppercase;margin:0;">
        BGE / Gemini · Milvus · LangGraph · VoxClinBench</p>
    </div>
    """, unsafe_allow_html=True)

    if not RAG_AVAILABLE:
        import traceback
        _err_detail = _RAG_IMPORT_ERROR or "Unknown import error"
        st.markdown(
            f'''<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;
            padding:1.2rem 1.5rem;color:#7c2d12;font-size:0.88rem;">
            <strong>⚠ RAG indexer unavailable</strong><br><br>
            <strong>Reason:</strong> <code style="background:#fef3c7;padding:2px 6px;border-radius:3px;">{_err_detail}</code><br><br>
            <strong>To fix:</strong> Make sure <code>indexer.py</code> is in the same folder as <code>voxclinbench_app.py</code>,
            and that all dependencies are installed:<br>
            <code style="display:block;margin-top:8px;padding:8px;background:#fef3c7;border-radius:4px;">
            pip install pymilvus sentence-transformers langchain langchain-google-genai</code>
            <br>The Clinical AI agent pipeline still works without RAG — agents will note "[No RAG index]" in their output.
            </div>''',
            unsafe_allow_html=True,
        )
        # Don't return — show the rest of page with informational UI
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 The Clinical AI page and Voice Assessment work fully without RAG. Only the PDF indexing features below are unavailable.")
        return

    # ── Connection status ─────────────────────────────────────────────────────
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        idx = st.session_state.indexer
        if idx:
            st.markdown(
                f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;'
                f'padding:10px 14px;font-size:0.82rem;color:#065a4a;">'
                f'<strong>✓ Milvus</strong> · Connected<br>'
                f'{st.session_state.index_stats.get("total_chunks",0)} chunks indexed</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:8px;'
                'padding:10px 14px;font-size:0.82rem;color:#92400e;">'
                '<strong>⚠ Milvus</strong> · No collection<br>Index PDFs below to connect</div>',
                unsafe_allow_html=True,
            )
    with col_stat2:
        ant_k = (st.session_state.cfg_anthropic_key or os.environ.get("ANTHROPIC_API_KEY",""))
        st.markdown(
            f'<div style="background:{"#f0fdf4" if ant_k else "#fef3c7"};'
            f'border:1px solid {"#bbf7d0" if ant_k else "#fde68a"};border-radius:8px;'
            f'padding:10px 14px;font-size:0.82rem;color:{"#065a4a" if ant_k else "#92400e"};">'
            f'<strong>{"✓" if ant_k else "⚠"} Anthropic</strong> · '
            f'{"Key set" if ant_k else "No key (agents only)"}</div>',
            unsafe_allow_html=True,
        )
    with col_stat3:
        tav_k = st.session_state.cfg_tavily_key
        st.markdown(
            f'<div style="background:{"#f0fdf4" if tav_k else "#f8fafc"};'
            f'border:1px solid {"#bbf7d0" if tav_k else "#e2e8f0"};border-radius:8px;'
            f'padding:10px 14px;font-size:0.82rem;color:{"#065a4a" if tav_k else "#64748b"};">'
            f'<strong>{"✓" if tav_k else "–"} Tavily</strong> · '
            f'{"Key set" if tav_k else "Optional (agent web search)"}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#e2e8f0;margin:1.2rem 0;'>", unsafe_allow_html=True)

    # ── Embedding model + API keys ────────────────────────────────────────────
    st.markdown("### Settings")

    em1, em2 = st.columns([1, 2])
    with em1:
        EMBED_OPTIONS = ["BGE (local · 1024-dim)", "Gemini embedding-001 (3072-dim)"]
        embed_choice = st.selectbox(
            "Embedding model",
            EMBED_OPTIONS,
            index=EMBED_OPTIONS.index(st.session_state.cfg_embed_model)
                  if st.session_state.cfg_embed_model in EMBED_OPTIONS else 0,
            key="rd_embed_sel",
            help="BGE runs locally — no API key, 1024 dims.\n"
                 "Gemini embedding-001 — requires GEMINI_API_KEY, 3072 dims.",
        )
        st.session_state.cfg_embed_model = embed_choice

    with em2:
        if "Gemini" in embed_choice:
            env_key = os.environ.get("GEMINI_API_KEY","") or os.environ.get("GOOGLE_API_KEY","")
            if env_key:
                st.markdown(
                    '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;'
                    'padding:8px 14px;font-size:0.78rem;color:#065a4a;margin-top:22px;">'
                    '✓ GEMINI_API_KEY found in environment</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.cfg_gemini_api_key = env_key
            else:
                typed_key = st.text_input(
                    "Gemini API key",
                    value=st.session_state.cfg_gemini_api_key,
                    type="password", placeholder="AIza...",
                    key="rd_gemini_key",
                    help="Required for Gemini embeddings and Gemini chat mode.",
                )
                st.session_state.cfg_gemini_api_key = typed_key
                if typed_key:
                    os.environ["GEMINI_API_KEY"] = typed_key
                    os.environ["GOOGLE_API_KEY"] = typed_key
                    st.markdown(
                        '<div style="font-size:0.72rem;color:#065a4a;padding-top:4px;">✓ Key saved for this session</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="font-size:0.72rem;color:#ef4444;padding-top:4px;">⚠ API key required</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div style="font-size:0.78rem;color:#374151;background:#f0fdfa;'
                'border:1px solid #99f6e4;border-radius:6px;padding:10px 14px;margin-top:22px;">'
                '\U0001f5a5 <strong>Runs locally</strong> &nbsp;·&nbsp; no API key &nbsp;·&nbsp; sentence-transformers (1024-dim)</div>',
                unsafe_allow_html=True,
            )

    # BGE + Gemini chat key
    if "BGE" in embed_choice and not st.session_state.cfg_use_gateway:
        _env_gkey = (os.environ.get("GEMINI_API_KEY","") or
                     os.environ.get("GOOGLE_API_KEY",""))
        if not _env_gkey:
            st.markdown("<hr style='border-color:#e2e8f0;margin:0.8rem 0;'>", unsafe_allow_html=True)
            st.markdown("**Gemini Chat API Key** *(required for Clinical AI Q&A with RAG)*")
            _ck = st.text_input(
                "Gemini API key for chat", value=st.session_state.cfg_gemini_api_key,
                type="password", placeholder="AIza...", key="rd_gemini_chat",
            )
            if _ck:
                st.session_state.cfg_gemini_api_key = _ck
                os.environ["GEMINI_API_KEY"] = _ck
                os.environ["GOOGLE_API_KEY"] = _ck
                st.markdown(
                    '<div style="font-size:0.72rem;color:#065a4a;padding-top:4px;">✓ Key saved</div>',
                    unsafe_allow_html=True,
                )

    use_gemini_embed = "Gemini" in embed_choice
    coll_name = "papers_rag_gemini" if use_gemini_embed else "papers_rag_voxclin"
    st.markdown(
        f'<div style="font-size:0.72rem;color:#64748b;padding:6px 0 0 0;">'
        f'📦 Collection: <code>{coll_name}</code></div>',
        unsafe_allow_html=True,
    )

    # ── Web search toggle ──────────────────────────────────────────────────────
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    col_ws, col_ws_info = st.columns([1, 3])
    with col_ws:
        st.session_state.web_search_enabled = st.checkbox(
            "Enable web search in Clinical AI Q&A",
            value=st.session_state.web_search_enabled,
        )
    with col_ws_info:
        if st.session_state.web_search_enabled:
            g2 = (os.environ.get("GEMINI_API_KEY","") or os.environ.get("GOOGLE_API_KEY",""))
            if g2:
                st.markdown(
                    '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;'
                    'padding:8px 14px;font-size:0.78rem;color:#065a4a;margin-top:4px;">'
                    '✓ Web search active — Gemini grounding</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:6px;'
                    'padding:8px 14px;font-size:0.78rem;color:#92400e;margin-top:4px;">'
                    '⚠ GEMINI_API_KEY not set — web search will silently skip</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:0.78rem;color:#374151;background:#fef3c7;'
                'border:1px solid #fde68a;border-radius:6px;padding:8px 14px;margin-top:4px;">'
                '⚠ Web search off — only indexed PDFs will be used</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── File uploader + category assignment ────────────────────────────────────
    st.markdown("### Upload & Categorise")
    uploaded_files = st.file_uploader(
        "Drop one or more PDFs (clinical guidelines, pharmacology reviews, rehab protocols, research papers)",
        type=["pdf"],
        accept_multiple_files=True,
        key="rd_uploader",
    )

    if uploaded_files:
        st.markdown("**Add custom category**")
        cn1, cn2 = st.columns([3, 1])
        with cn1:
            new_type = st.text_input(
                "new_cat", key="rd_new_cat",
                placeholder="e.g. Voice Rehabilitation, Neurology...",
                label_visibility="collapsed",
            )
        with cn2:
            if st.button("Add", use_container_width=True, key="rd_add_cat"):
                cleaned = new_type.strip()
                all_cur = st.session_state.custom_paper_types + PAPER_TYPES
                if cleaned and cleaned not in all_cur:
                    st.session_state.custom_paper_types.insert(0, cleaned)
                    st.rerun()

        if st.session_state.custom_paper_types:
            st.markdown(
                " ".join(f'<span class="badge badge-type">{t}</span>'
                         for t in st.session_state.custom_paper_types),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        all_types   = st.session_state.custom_paper_types + PAPER_TYPES
        st.markdown("**Categorise files** — select a category, then click files to assign it")
        active_type = st.selectbox(
            "active_type", all_types, key="rd_active_cat",
            label_visibility="collapsed",
        )

        cols3 = st.columns(3)
        for i, uf in enumerate(uploaded_files):
            assignment = st.session_state.file_assignments.get(uf.name)
            bg     = "#f5f3ff" if assignment else "white"
            border = "#ddd6fe" if assignment else "#e2e8f0"
            tc     = "#7c3aed" if assignment else "#94a3b8"
            lbl    = assignment or "unassigned"
            with cols3[i % 3]:
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {border};'
                    f'border-radius:8px;padding:10px 12px;margin-bottom:6px;">'
                    f'<div style="font-size:0.78rem;color:#1e293b;font-weight:500;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                    f'margin-bottom:4px;" title="{uf.name}">{uf.name}</div>'
                    f'<span style="font-size:0.62rem;background:#f5f3ff;color:{tc};'
                    f'padding:1px 6px;border-radius:3px;border:1px solid {border};">'
                    f'{lbl}</span></div>',
                    unsafe_allow_html=True,
                )
                btn_lbl = f"✓ {active_type}" if assignment != active_type else "✕ Remove"
                if st.button(btn_lbl, key=f"rd_assign_{uf.name}", use_container_width=True):
                    if assignment == active_type:
                        del st.session_state.file_assignments[uf.name]
                    else:
                        st.session_state.file_assignments[uf.name] = active_type
                    st.rerun()

        assigned_count = len(st.session_state.file_assignments)
        unassigned = [uf.name for uf in uploaded_files
                      if uf.name not in st.session_state.file_assignments]
        st.markdown(
            f'<div style="margin-top:6px;font-size:0.78rem;color:#64748b;">'
            f'{assigned_count}/{len(uploaded_files)} files assigned</div>',
            unsafe_allow_html=True,
        )
        if unassigned:
            st.markdown(
                f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">'
                f'Unassigned: {", ".join(unassigned)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Indexing options ──────────────────────────────────────────────────────
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        append_mode = st.checkbox("Append to existing index", value=False)
    with oc2:
        show_chunks = st.checkbox("Preview first 5 chunks", value=True)
    with oc3:
        st.caption(f"Index: {st.session_state.cfg_index_type} · "
                   f"Chunk: {st.session_state.cfg_chunk_size}w · "
                   f"Overlap: {st.session_state.cfg_chunk_overlap}w")

    ready_files = [uf for uf in (uploaded_files or [])
                   if uf.name in st.session_state.file_assignments]

    if st.button(
        "Start Indexing", type="primary",
        disabled=len(ready_files) == 0,
        use_container_width=True, key="rd_start_idx",
    ):
        st.session_state.interrupt_requested = False
        st.session_state.is_indexing         = True

        tmp_dir = Path("/tmp/vox_idx_uploads")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        try:
            for uf in ready_files:
                dest = tmp_dir / uf.name
                dest.write_bytes(uf.read())
                saved.append((dest, st.session_state.file_assignments[uf.name]))

            status_box = st.empty()
            bar        = st.progress(0, text="Starting...")
            log_box    = st.empty()
            logs: list = []

            def log(msg):
                logs.append(msg)
                log_box.markdown(
                    '<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                    'border-radius:8px;padding:12px 16px;font-size:0.8rem;'
                    'color:#475569;line-height:1.7;">'
                    + "".join(f"· {l}<br>" for l in logs[-10:])
                    + "</div>",
                    unsafe_allow_html=True,
                )

            def tick(pct, label):
                bar.progress(min(float(pct), 1.0), text=f"{label} — {int(min(pct,1.0)*100)}%")
                status_box.markdown(
                    f'<div style="font-size:0.82rem;color:#64748b;padding:4px 0 8px 0;">'
                    f'{label}</div>',
                    unsafe_allow_html=True,
                )

            tick(0.02, "Creating indexer")
            _use_gem   = "Gemini" in st.session_state.cfg_embed_model
            _embed_dim = 3072 if _use_gem else 1024
            _coll      = "papers_rag_gemini" if _use_gem else "papers_rag_voxclin"

            if _use_gem:
                gkey = (st.session_state.cfg_gemini_api_key or
                        os.environ.get("GEMINI_API_KEY","") or
                        os.environ.get("GOOGLE_API_KEY",""))
                if not gkey:
                    st.error("Gemini API key required for Gemini embeddings.")
                    st.session_state.is_indexing = False
                    st.stop()
                os.environ["GEMINI_API_KEY"] = gkey
                os.environ["GOOGLE_API_KEY"] = gkey

            import indexer as _idx_mod
            _idx_mod.EMBED_DIM = _embed_dim

            indexer_obj = PDFIndexer(
                chunk_size=st.session_state.cfg_chunk_size,
                chunk_overlap=st.session_state.cfg_chunk_overlap,
                model=st.session_state.cfg_model,
                index_type=st.session_state.cfg_index_type,
                collection_name=_coll,
                drop_old_collection=(not append_mode),
            )
            log(f"Indexer ready [{st.session_state.cfg_index_type}] "
                f"embed={'Gemini' if _use_gem else 'BGE'} dim={_embed_dim}")

            all_chunks = []
            for fi, (pdf_path, ptype) in enumerate(saved):
                if st.session_state.interrupt_requested:
                    break
                p0 = 0.05 + (fi / max(len(saved), 1)) * 0.25
                p1 = 0.05 + ((fi+1) / max(len(saved), 1)) * 0.25
                tick(p0, f"Extracting {pdf_path.name}")
                chunks = indexer_obj.extract_and_chunk(str(pdf_path), paper_type=ptype)
                all_chunks.extend(chunks)
                tick(p1, f"{pdf_path.name} — {len(chunks)} chunks")
                log(f"{pdf_path.name} [{ptype}] — {len(chunks)} chunks")

            total = len(all_chunks)
            tick(0.33, f"Extraction done — {total} chunks")

            if _use_gem:
                log("Embedding with Gemini embedding-001 (3072-dim)...")
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                gkey2 = (st.session_state.cfg_gemini_api_key or
                         os.environ.get("GEMINI_API_KEY","") or
                         os.environ.get("GOOGLE_API_KEY",""))
                os.environ["GOOGLE_API_KEY"] = gkey2
                gem_emb   = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                BATCH     = 20
                n_batches = math.ceil(total / BATCH)
                for bi in range(0, total, BATCH):
                    if st.session_state.interrupt_requested:
                        break
                    bn    = bi // BATCH + 1
                    batch = all_chunks[bi: bi + BATCH]
                    embs  = gem_emb.embed_documents([c["text"] for c in batch])
                    for j, emb in enumerate(embs):
                        all_chunks[bi + j]["embedding"] = emb
                    pct = 0.33 + (min(bi + BATCH, total) / max(total, 1)) * 0.42
                    tick(pct, f"Embedding batch {bn}/{n_batches} (Gemini 3072-dim)")
                    log(f"Gemini batch {bn}/{n_batches} done")
            else:
                log("Embedding (BGE local)...")
                embedder  = indexer_obj._get_embedder()
                BATCH     = 64
                n_batches = math.ceil(total / BATCH)
                for bi in range(0, total, BATCH):
                    if st.session_state.interrupt_requested:
                        break
                    bn    = bi // BATCH + 1
                    texts = [
                        "Represent this passage for retrieval: " + all_chunks[bi+j]["text"]
                        for j in range(min(BATCH, total-bi))
                    ]
                    embs  = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                    for j, emb in enumerate(embs):
                        all_chunks[bi+j]["embedding"] = emb.tolist()
                    pct = 0.33 + (min(bi+BATCH, total) / max(total, 1)) * 0.42
                    tick(pct, f"Embedding batch {bn}/{n_batches}")
                    log(f"Batch {bn}/{n_batches} done")

            tick(0.80, f"Building {st.session_state.cfg_index_type} index...")
            log(f"Building {st.session_state.cfg_index_type} index...")

            if append_mode and st.session_state.indexer:
                st.session_state.indexer.add_to_index(all_chunks)
                st.session_state.indexer.set_output_language(st.session_state.cfg_language)
            else:
                indexer_obj.build_index(all_chunks)
                indexer_obj.set_output_language(st.session_state.cfg_language)
                st.session_state.indexer = indexer_obj

            tick(1.0, "Indexing complete ✓")

            new_entries = [(p.name, pt) for p, pt in saved]
            if append_mode:
                st.session_state.indexed_files.extend(new_entries)
            else:
                st.session_state.indexed_files = new_entries

            st.session_state.indexer._indexed_files = st.session_state.indexed_files
            st.session_state.indexer._file_url_map  = st.session_state.file_url_map
            _save_indexed()

            st.session_state.index_stats = {
                "total_chunks": total,
                "total_files":  len(st.session_state.indexed_files),
                "index_type":   st.session_state.cfg_index_type,
                "embed_model":  "Gemini embedding-001" if _use_gem else EMBED_MODEL,
                "chunk_size":   st.session_state.cfg_chunk_size,
            }

            st.success(
                f"**{len(saved)} file(s)** indexed — **{total} chunks** — "
                f"**{st.session_state.cfg_index_type}** index built ✓"
            )

            if show_chunks and all_chunks:
                st.markdown("**Chunk preview (first 5)**")
                for c in all_chunks[:5]:
                    st.markdown(
                        f'<div class="chunk-preview">'
                        f'<span class="badge">{c["source"]}</span>'
                        f'<span class="badge">p{c["page"]}</span>'
                        f'<span class="badge badge-type">{c["paper_type"]}</span>'
                        f'<span class="badge">#{c["chunk_id"]}</span><br><br>'
                        f'{c["text"][:480]}{"..." if len(c["text"])>480 else ""}</div>',
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            if "Interrupted" in str(e):
                st.warning("Indexing interrupted.")
            else:
                st.error(str(e))
                st.exception(e)
        finally:
            st.session_state.is_indexing = False

    if st.session_state.is_indexing:
        if st.button("⏹ Interrupt", key="rd_interrupt"):
            st.session_state.interrupt_requested = True

    # ── NPZ export ────────────────────────────────────────────────────────────
    if (st.session_state.indexer is not None and
            st.session_state.index_stats.get("total_chunks", 0) > 0):
        st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown("### Export Vector Store")
        st.caption(
            f"Download the current index ({st.session_state.index_stats['total_chunks']} chunks, "
            f"{st.session_state.index_stats.get('total_files',0)} files) as a compressed .npz file."
        )
        if st.button("⬇ Export to .npz", use_container_width=True, key="rd_npz_export"):
            try:
                npz_path = "/tmp/voxclin_index_export.npz"
                n = st.session_state.indexer.export_to_npz(npz_path)
                with open(npz_path, "rb") as f:
                    st.download_button(
                        label=f"Download voxclin_index_{n}chunks.npz",
                        data=f.read(),
                        file_name=f"voxclin_index_{n}chunks.npz",
                        mime="application/octet-stream",
                    )
                st.success(f"Exported {n} chunks to .npz")
            except Exception as e:
                st.error(f"Export failed: {e}")

    # ── NPZ import ────────────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("### Import Vector Store")
    st.caption("Re-import a previously exported .npz snapshot into Milvus.")

    npz_up = st.file_uploader("Upload .npz", type=["npz"], key="rd_npz_import")
    if npz_up:
        imp_append = st.checkbox("Append to existing index", value=False, key="rd_npz_append")
        if st.button("Import .npz", use_container_width=True, key="rd_import_btn"):
            try:
                npz_path = "/tmp/voxclin_import.npz"
                Path(npz_path).write_bytes(npz_up.read())

                import indexer as _idx_mod
                _use_gem_imp = "Gemini" in st.session_state.cfg_embed_model
                _idx_mod.EMBED_DIM = 3072 if _use_gem_imp else 1024
                coll_imp = "papers_rag_gemini" if _use_gem_imp else "papers_rag_voxclin"

                if st.session_state.indexer is None:
                    indexer_obj2 = PDFIndexer(
                        chunk_size=st.session_state.cfg_chunk_size,
                        chunk_overlap=st.session_state.cfg_chunk_overlap,
                        model=st.session_state.cfg_model,
                        index_type=st.session_state.cfg_index_type,
                        collection_name=coll_imp,
                        drop_old_collection=False,
                    )
                    st.session_state.indexer = indexer_obj2

                result = st.session_state.indexer.import_from_npz(npz_path, append=imp_append)

                st.session_state.indexed_files = [
                    (f, "Imported") for f in result.get("indexed_files", [])
                ] or st.session_state.indexed_files
                st.session_state.index_stats = {
                    "total_chunks": result["n_chunks"],
                    "total_files":  len(st.session_state.indexed_files),
                    "index_type":   result.get("index_type", st.session_state.cfg_index_type),
                    "embed_model":  result.get("embed_model", EMBED_MODEL),
                }
                _save_indexed()
                st.success(f"Imported {result['n_chunks']} chunks from .npz ✓")
            except Exception as e:
                st.error(f"Import failed: {e}")
                st.exception(e)

    # ── Collection management ─────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("### Collection Management")
    cm1, cm2 = st.columns(2)
    with cm1:
        if st.button("Clear Collection (keep schema)", use_container_width=True, key="rd_clear"):
            try:
                if st.session_state.indexer:
                    st.session_state.indexer.clear_collection(drop=False)
                    st.session_state.indexed_files = []
                    st.session_state.index_stats   = {}
                    _save_indexed()
                    st.success("Collection cleared (schema preserved).")
            except Exception as e:
                st.error(f"Clear failed: {e}")
    with cm2:
        if st.button("Drop Collection Entirely", use_container_width=True, key="rd_drop"):
            try:
                if st.session_state.indexer:
                    st.session_state.indexer.clear_collection(drop=True)
                    st.session_state.indexer      = None
                    st.session_state.indexed_files = []
                    st.session_state.index_stats   = {}
                    _save_indexed()
                    st.success("Collection dropped.")
                    st.rerun()
            except Exception as e:
                st.error(f"Drop failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

nav_bar()
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

page = st.session_state.page

if page == "assessment":
    page_assessment()
elif page == "chat":
    page_chat()
elif page == "rag_dev":
    page_rag_dev()
