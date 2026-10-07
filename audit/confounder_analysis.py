"""
Member 2: Confounder & Shortcut Analysis
Bridge2AI-Voice v3.0.0

Produces:
  1. confounder_vulnerability_table.csv  — per-disease AUROC from demographics only
  2. shortcut_ablation_table.csv         — per-disease AUROC broken down by feature group
  3. figures/vulnerability_heatmap.png   — heatmap for the paper
  4. figures/ablation_barplot.png        — which shortcut dominates per disease

Run:
    conda activate syst
    B2AI_DATA_ROOT=/path/to/b2ai-voice/3.0.0 python audit/confounder_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = Path(os.environ["B2AI_DATA_ROOT"]) / "phenotype"
OUT  = Path(os.environ.get("AUDIT_OUT_DIR", "outputs/audit"))
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

DIAG_DIR  = BASE / "diagnosis"
DEMO_FILE = BASE / "demographics" / "demographics.tsv"

TASK_FILES = {
    "acoustic_task":          BASE / "task" / "acoustic_task.tsv",
    "harvard_sentences":      BASE / "task" / "harvard_sentences.tsv",
    "random_item_generation": BASE / "task" / "random_item_generation.tsv",
    "stroop":                 BASE / "task" / "stroop.tsv",
    "voice_perception":       BASE / "task" / "voice_perception.tsv",
    "voice_problem_severity": BASE / "task" / "voice_problem_severity.tsv",
    "winograd":               BASE / "task" / "winograd.tsv",
}

DISEASE_FILES = {
    "adhd":                "adhd_adult.tsv",
    "airway_stenosis":     "airway_stenosis.tsv",
    "als":                 "amyotrophic_lateral_sclerosis.tsv",
    "anxiety":             "anxiety.tsv",
    "benign_lesions":      "benign_lesions.tsv",
    "bipolar":             "bipolar_disorder.tsv",
    "cognitive_impairment":"cognitive_impairment.tsv",
    "copd_asthma":         "copd_and_asthma.tsv",
    "depression":          "depression.tsv",
    "glottic_insuff":      "glottic_insufficiency.tsv",
    "laryngeal_cancer":    "laryngeal_cancer.tsv",
    "laryngeal_dystonia":  "laryngeal_dystonia.tsv",
    "laryngitis":          "laryngitis.tsv",
    "mtd":                 "muscle_tension_dysphonia.tsv",
    "parkinsons":          "parkinsons_disease.tsv",
    "precancerous":        "precancerous_lesions.tsv",
    "psychiatric_history": "psychiatric_history.tsv",
    "ptsd":                "ptsd_adult.tsv",
    "chronic_cough":       "unexplained_chronic_cough.tsv",
    "vf_paralysis":        "unilateral_vocal_fold_paralysis.tsv",
}

# Psychiatric / cognitive diseases — expected to show strongest shortcut signal
PSYCHIATRIC = {"adhd", "ptsd", "depression", "anxiety", "bipolar",
               "psychiatric_history", "cognitive_impairment"}

# ── Step 1: Load demographics ─────────────────────────────────────────────────

print("Loading demographics...")
demo = pd.read_csv(DEMO_FILE, sep="\t", low_memory=False)
demo["participant_id"] = demo["participant_id"].astype(str).str.lstrip("0")

DEMO_COLS = ["participant_id", "age", "sex_at_birth", "country", "ethnicity"]
available = [c for c in DEMO_COLS if c in demo.columns]
demo = demo[available].drop_duplicates("participant_id")
print(f"  Demographics: {len(demo)} participants, columns: {available}")

# ── Step 2: Build task histogram ──────────────────────────────────────────────

print("\nBuilding task histogram...")
task_counts = {}
for task_name, task_path in TASK_FILES.items():
    if not task_path.exists():
        print(f"  MISSING: {task_path.name}")
        continue
    df = pd.read_csv(task_path, sep="\t", low_memory=False, usecols=["participant_id"])
    df["participant_id"] = df["participant_id"].astype(str).str.lstrip("0")
    counts = df["participant_id"].value_counts().rename(f"task_{task_name}")
    task_counts[task_name] = counts

if task_counts:
    task_df = pd.concat(task_counts.values(), axis=1).reset_index()
    task_df.columns = ["participant_id"] + [f"task_{t}" for t in task_counts.keys()]
    task_df = task_df.fillna(0)
    print(f"  Task histogram: {len(task_df)} participants, {len(task_df.columns)-1} tasks")
else:
    task_df = pd.DataFrame(columns=["participant_id"])

# ── Step 3: Merge into master feature table ───────────────────────────────────

print("\nBuilding master feature table...")
master = demo.copy()
if not task_df.empty:
    master = master.merge(task_df, on="participant_id", how="left")
    task_cols = [c for c in task_df.columns if c != "participant_id"]
    master[task_cols] = master[task_cols].fillna(0)

print(f"  Master table: {len(master)} participants, {master.shape[1]} columns")

# ── Step 4: Feature encoding ──────────────────────────────────────────────────

def encode_features(df, feature_set="all"):
    """
    feature_set:
        'all'   — age + sex + country + ethnicity + task histogram
        'age'   — age only
        'sex'   — sex only
        'demo'  — country + ethnicity only
        'task'  — task histogram only
    """
    result = pd.DataFrame(index=df.index)

    if feature_set in ("all", "age") and "age" in df.columns:
        result["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(
            pd.to_numeric(df["age"], errors="coerce").median()
        )

    if feature_set in ("all", "sex") and "sex_at_birth" in df.columns:
        le = LabelEncoder()
        result["sex"] = le.fit_transform(df["sex_at_birth"].fillna("Unknown").astype(str))

    if feature_set in ("all", "demo"):
        for col in ["country", "ethnicity"]:
            if col in df.columns:
                le = LabelEncoder()
                result[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))

    if feature_set in ("all", "task"):
        task_cols = [c for c in df.columns if c.startswith("task_")]
        for c in task_cols:
            result[c] = df[c].fillna(0).astype(float)

    return result

# ── Step 5: Cross-validated AUROC ─────────────────────────────────────────────

def compute_auroc(X, y, n_splits=5, random_state=42):
    if X.shape[0] < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        return np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aurocs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_te)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        aurocs.append(roc_auc_score(y_te, prob))

    return np.mean(aurocs) if aurocs else np.nan

# ── Step 6: Load disease labels ───────────────────────────────────────────────

print("\nLoading disease labels...")
disease_participants = {}
for disease, fname in DISEASE_FILES.items():
    fpath = DIAG_DIR / fname
    if not fpath.exists():
        print(f"  MISSING: {fname}")
        continue
    df = pd.read_csv(fpath, sep="\t", low_memory=False, usecols=["participant_id"])
    df["participant_id"] = df["participant_id"].astype(str).str.lstrip("0")
    pids = df["participant_id"].unique().tolist()
    disease_participants[disease] = set(pids)
    print(f"  {disease:25s}: {len(pids)} participants")

# ── Step 7: Run analysis ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("RUNNING CONFOUNDER VULNERABILITY ANALYSIS")
print("="*60)

FEATURE_SETS = {
    "full_demographics": "all",
    "age_only":          "age",
    "sex_only":          "sex",
    "country_ethnicity": "demo",
    "task_histogram":    "task",
}

results = []
all_pids = set(master["participant_id"].tolist())

for disease, pos_pids in disease_participants.items():
    neg_pids = all_pids - pos_pids
    pos_df = master[master["participant_id"].isin(pos_pids)].copy()
    neg_df = master[master["participant_id"].isin(neg_pids)].copy()
    pos_df["label"] = 1
    neg_df["label"] = 0
    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    y = combined["label"].values

    row = {"disease": disease, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}

    for feat_name, feat_set in FEATURE_SETS.items():
        X_enc = encode_features(combined, feature_set=feat_set)
        if X_enc.shape[1] == 0:
            row[feat_name] = np.nan
            continue
        auroc = compute_auroc(X_enc.values, y)
        row[feat_name] = round(auroc, 4) if not np.isnan(auroc) else np.nan
        tag = " ← PSYCHIATRIC" if disease in PSYCHIATRIC else ""
        val = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"
        print(f"  {disease:25s} | {feat_name:20s} | {val}{tag}")

    results.append(row)

# ── Step 8: Save tables ───────────────────────────────────────────────────────

df_results = pd.DataFrame(results).sort_values("full_demographics", ascending=False)

vuln_table = df_results[["disease", "n_pos", "n_neg", "full_demographics"]].copy()
vuln_table.columns = ["Disease", "N_positive", "N_negative", "Confounder_AUROC"]
vuln_table["Is_Psychiatric"] = vuln_table["Disease"].isin(PSYCHIATRIC)
vuln_table.to_csv(OUT / "confounder_vulnerability_table.csv", index=False)

ablation_table = df_results[["disease", "full_demographics", "age_only",
                               "sex_only", "country_ethnicity", "task_histogram"]].copy()
ablation_table.columns = ["Disease", "Full_Demographics", "Age_Only",
                           "Sex_Only", "Country_Ethnicity", "Task_Histogram"]
ablation_table.to_csv(OUT / "shortcut_ablation_table.csv", index=False)

# ── Step 9: Print summary ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("CONFOUNDER VULNERABILITY TABLE (sorted by AUROC)")
print("="*60)
print(f"{'Disease':<25} {'AUROC':>8}  {'Type'}")
print("-"*50)
for _, row in vuln_table.iterrows():
    tag = " ← PSYCHIATRIC" if row["Is_Psychiatric"] else ""
    val = f"{row['Confounder_AUROC']:.4f}" if not pd.isna(row["Confounder_AUROC"]) else "  N/A"
    print(f"{row['Disease']:<25} {val:>8}{tag}")

# ── Step 10: Heatmap figure ───────────────────────────────────────────────────

print("\nGenerating figures...")

plot_df = df_results[["disease", "full_demographics", "age_only",
                        "sex_only", "country_ethnicity", "task_histogram"]].copy()
plot_df = plot_df.set_index("disease")
plot_df.columns = ["Full Demo", "Age Only", "Sex Only", "Country/Eth", "Task Hist"]
plot_df = plot_df.sort_values("Full Demo", ascending=False)

fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.5)))
sns.heatmap(
    plot_df, annot=True, fmt=".2f", cmap="RdYlGn",
    vmin=0.45, vmax=1.0, linewidths=0.5, ax=ax,
    cbar_kws={"label": "AUROC"},
)
ax.set_title(
    "Confounder Vulnerability: AUROC from Demographics Alone\n"
    "(higher = more signal explained by non-acoustic shortcuts)",
    fontsize=11, pad=12
)
ax.set_xlabel("Feature Subset", fontsize=10)
ax.set_ylabel("Disease", fontsize=10)

for lbl in ax.get_yticklabels():
    if lbl.get_text() in PSYCHIATRIC:
        lbl.set_color("crimson")
        lbl.set_fontweight("bold")

plt.tight_layout()
fig.savefig(FIGS / "vulnerability_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/vulnerability_heatmap.png")

# ── Step 11: Ablation bar chart ───────────────────────────────────────────────

top_diseases = plot_df.sort_values("Full Demo", ascending=False).head(12)
x = np.arange(len(top_diseases))
width = 0.17
cols   = ["Full Demo", "Age Only", "Sex Only", "Country/Eth", "Task Hist"]
colors = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c", "#1a9641"]

fig2, ax2 = plt.subplots(figsize=(13, 6))
for i, (col, color) in enumerate(zip(cols, colors)):
    vals = top_diseases[col].values
    ax2.bar(x + (i - 2) * width, vals, width, label=col, color=color, alpha=0.85)

ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Chance (0.5)")
ax2.set_xticks(x)
ax2.set_xticklabels(top_diseases.index, rotation=40, ha="right", fontsize=9)
ax2.set_ylabel("AUROC", fontsize=10)
ax2.set_ylim(0.4, 1.05)
ax2.set_title(
    "Shortcut Ablation: Which Feature Group Drives the Confounder Signal?\n"
    "(Top 12 most vulnerable diseases — red labels = psychiatric)",
    fontsize=11
)
ax2.legend(loc="upper right", fontsize=8)

for tick, name in zip(ax2.get_xticklabels(), top_diseases.index):
    if name in PSYCHIATRIC:
        tick.set_color("crimson")
        tick.set_fontweight("bold")

plt.tight_layout()
fig2.savefig(FIGS / "ablation_barplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/ablation_barplot.png")

# ── Done ──────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ALL DONE")
print("="*60)
print(f"Output folder: {OUT}")
print("Files created:")
print("  confounder_vulnerability_table.csv")
print("  shortcut_ablation_table.csv")
print("  figures/vulnerability_heatmap.png")
print("  figures/ablation_barplot.png")
