"""
Demographic Stratification of MARVEL Errors
Bridge2AI-Voice v3.0.0

Uses the handoff prediction CSVs + demographics to compute:
  1. Per-disease AUROC stratified by age group, sex, country
  2. Protocol sensitivity gap table with bootstrap CIs
  3. Figures for the paper

Outputs (all saved to update2/):
  demographic_stratification_marvel.csv
  protocol_sensitivity_gap.csv
  figures/age_stratification_heatmap.png
  figures/sex_stratification_plot.png
  figures/protocol_sensitivity_gap.png

Run:
    conda activate syst
    B2AI_DATA_ROOT=/path/to/b2ai-voice/3.0.0 PREDICTIONS_DIR=/path/to/patient_level_predictions \
        python audit/demographic_stratification.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

# PREDICTIONS_DIR: patient-level prediction CSVs ({marvel,main}_tier{1,2}_{fair,unified}.csv)
# produced by separate benchmark models (not included in this repo).
HANDOFF = Path(os.environ["PREDICTIONS_DIR"])
PHENO   = Path(os.environ["B2AI_DATA_ROOT"]) / "phenotype"
OUT     = Path(os.environ.get("AUDIT_OUT_DIR", "outputs/audit"))
FIGS    = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Load demographics ─────────────────────────────────────────────────────────

print("Loading demographics...")
demo = pd.read_csv(PHENO / "demographics/demographics.tsv", sep="\t", low_memory=False)
demo["participant_id"] = demo["participant_id"].astype(str).str.lstrip("0")

# Age bins
demo["age_num"] = pd.to_numeric(demo["age"], errors="coerce")
demo["age_group"] = pd.cut(
    demo["age_num"],
    bins=[0, 40, 55, 70, 120],
    labels=["<=40", "41-55", "56-70", "71+"],
)

# Sex — keep Female / Male / Other
demo["sex"] = demo["sex_at_birth"].fillna("Unknown").astype(str)
demo["sex"] = demo["sex"].apply(
    lambda x: x if x in ("Female", "Male") else "Other/Unknown"
)

# Country — keep top 3, group rest as Other
top_countries = demo["country"].value_counts().head(3).index.tolist()
demo["country_grp"] = demo["country"].apply(
    lambda x: x if x in top_countries else "Other"
)

demo_slim = demo[["participant_id", "age_group", "sex", "country_grp"]].drop_duplicates("participant_id")
print(f"  {len(demo_slim)} participants with demographics")

# ── Helper: safe AUROC ────────────────────────────────────────────────────────

def safe_auroc(y_true, y_score):
    y_true  = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)
    mask    = np.isfinite(y_score) & np.isfinite(y_true)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) < 6 or y_true.sum() < 2 or (1 - y_true).sum() < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_auroc_ci(y_true, y_score, n_boot=1000, ci=95):
    y_true  = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)
    mask    = np.isfinite(y_score) & np.isfinite(y_true)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) < 10 or y_true.sum() < 2 or (1-y_true).sum() < 2:
        return np.nan, np.nan
    rng   = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() < 1 or (1 - yt).sum() < 1:
            continue
        boots.append(roc_auc_score(yt, ys))
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return round(lo, 4), round(hi, 4)

# ── Load prediction CSVs ──────────────────────────────────────────────────────

def load_preds(fname):
    df = pd.read_csv(HANDOFF / fname)
    df["participant_id"] = df["participant_id"].astype(str).str.lstrip("0")
    return df

print("Loading prediction CSVs...")
marvel_t1_fair    = load_preds("marvel_tier1_fair.csv")
marvel_t1_unified = load_preds("marvel_tier1_unified.csv")
marvel_t2_fair    = load_preds("marvel_tier2_fair.csv")
marvel_t2_unified = load_preds("marvel_tier2_unified.csv")
main_t1_fair      = load_preds("main_tier1_fair.csv")
main_t1_unified   = load_preds("main_tier1_unified.csv")
main_t2_fair      = load_preds("main_tier2_fair.csv")
main_t2_unified   = load_preds("main_tier2_unified.csv")

# ── Identify disease columns ──────────────────────────────────────────────────

def get_diseases(df):
    return [c.replace("prob_", "") for c in df.columns if c.startswith("prob_")]

# ── Section 1: Demographic stratification of MARVEL errors ───────────────────

print("\n" + "="*60)
print("SECTION 1: DEMOGRAPHIC STRATIFICATION OF MARVEL ERRORS")
print("="*60)

def stratify_by(pred_df, demo_df, stratum_col, model_label, protocol_label):
    """Compute per-disease AUROC broken down by a demographic stratum."""
    diseases = get_diseases(pred_df)
    merged   = pred_df.merge(demo_df, on="participant_id", how="left")
    rows = []
    for disease in diseases:
        prob_col  = f"prob_{disease}"
        label_col = f"label_{disease}"
        if prob_col not in merged.columns or label_col not in merged.columns:
            continue
        sub = merged[[prob_col, label_col, stratum_col]].dropna(subset=[prob_col, label_col])
        for stratum_val, grp in sub.groupby(stratum_col):
            auroc = safe_auroc(grp[label_col], grp[prob_col])
            rows.append({
                "model":    model_label,
                "protocol": protocol_label,
                "disease":  disease,
                "stratum":  stratum_col,
                "value":    str(stratum_val),
                "n_total":  len(grp),
                "n_pos":    int(grp[label_col].sum()),
                "auroc":    round(auroc, 4) if not np.isnan(auroc) else np.nan,
            })
    return pd.DataFrame(rows)

strat_frames = []
for pred_df, model_lbl, proto_lbl in [
    (marvel_t1_unified, "MARVEL", "unified"),
    (marvel_t2_unified, "MARVEL", "unified"),
    (main_t1_unified,   "Main",   "unified"),
    (main_t2_unified,   "Main",   "unified"),
]:
    for stratum in ["age_group", "sex", "country_grp"]:
        strat_frames.append(
            stratify_by(pred_df, demo_slim, stratum, model_lbl, proto_lbl)
        )

strat_df = pd.concat(strat_frames, ignore_index=True)
strat_df.to_csv(OUT / "demographic_stratification_marvel.csv", index=False)
print(f"  Saved: demographic_stratification_marvel.csv ({len(strat_df)} rows)")

# Print summary — most concerning findings
print("\n  Top age-group disparities (MARVEL, unified, |max-min| AUROC across age):")
age_marvel = strat_df[
    (strat_df["model"] == "MARVEL") &
    (strat_df["stratum"] == "age_group") &
    (strat_df["n_pos"] >= 2)
].copy()
if not age_marvel.empty:
    spread = age_marvel.groupby("disease")["auroc"].agg(lambda x: x.max() - x.min()).sort_values(ascending=False)
    for disease, gap in spread.head(8).items():
        print(f"    {disease:<25} gap={gap:.3f}")

# ── Section 2: Protocol sensitivity gap ──────────────────────────────────────

print("\n" + "="*60)
print("SECTION 2: PROTOCOL SENSITIVITY GAP (bootstrap CIs)")
print("="*60)

def compute_protocol_gap(fair_df, unified_df, model_label, tier_label):
    """For each disease, compute fair AUROC, unified AUROC, gap, and 95% CI on gap."""
    diseases = get_diseases(unified_df)
    rows = []
    for disease in diseases:
        prob_col  = f"prob_{disease}"
        label_col = f"label_{disease}"

        # Fair AUROC
        if prob_col in fair_df.columns and label_col in fair_df.columns:
            fair_sub = fair_df[[prob_col, label_col]].dropna()
            auroc_fair = safe_auroc(fair_sub[label_col], fair_sub[prob_col])
        else:
            auroc_fair = np.nan

        # Unified AUROC
        if prob_col in unified_df.columns and label_col in unified_df.columns:
            uni_sub = unified_df[[prob_col, label_col]].dropna()
            auroc_uni = safe_auroc(uni_sub[label_col], uni_sub[prob_col])
            ci_lo, ci_hi = bootstrap_auroc_ci(uni_sub[label_col], uni_sub[prob_col])
        else:
            auroc_uni = np.nan
            ci_lo, ci_hi = np.nan, np.nan

        gap = round(auroc_fair - auroc_uni, 4) if not np.isnan(auroc_fair) and not np.isnan(auroc_uni) else np.nan

        rows.append({
            "model":        model_label,
            "tier":         tier_label,
            "disease":      disease,
            "auroc_fair":   round(auroc_fair, 4) if not np.isnan(auroc_fair) else np.nan,
            "auroc_unified":round(auroc_uni, 4)  if not np.isnan(auroc_uni)  else np.nan,
            "gap_fair_minus_unified": gap,
            "unified_ci_lo": ci_lo,
            "unified_ci_hi": ci_hi,
        })

    return pd.DataFrame(rows)

gap_frames = [
    compute_protocol_gap(marvel_t1_fair, marvel_t1_unified, "MARVEL", "tier1"),
    compute_protocol_gap(marvel_t2_fair, marvel_t2_unified, "MARVEL", "tier2"),
    compute_protocol_gap(main_t1_fair,   main_t1_unified,   "Main",   "tier1"),
    compute_protocol_gap(main_t2_fair,   main_t2_unified,   "Main",   "tier2"),
]
gap_df = pd.concat(gap_frames, ignore_index=True).drop_duplicates(
    subset=["model", "disease"]
).sort_values(["model", "gap_fair_minus_unified"], ascending=[True, False])

gap_df.to_csv(OUT / "protocol_sensitivity_gap.csv", index=False)
print(f"  Saved: protocol_sensitivity_gap.csv")

print("\n  Protocol sensitivity gap — MARVEL tier2 (fair → unified drop):")
marvel_gap = gap_df[(gap_df["model"] == "MARVEL") & (gap_df["tier"] == "tier2")].copy()
for _, row in marvel_gap.iterrows():
    if not np.isnan(row["gap_fair_minus_unified"]):
        print(f"    {row['disease']:<25} fair={row['auroc_fair']:.4f}  unified={row['auroc_unified']:.4f}  gap={row['gap_fair_minus_unified']:+.4f}")

# ── Figure 1: Age stratification heatmap (MARVEL, unified) ───────────────────

print("\nGenerating figures...")

PSYCHIATRIC = {"adhd", "ptsd", "depression", "anxiety", "bipolar",
               "psychiatric_history", "cognitive_impairment"}

age_pivot_marvel = age_marvel.groupby(["disease", "value"])["auroc"].mean().unstack("value")
age_order = ["<=40", "41-55", "56-70", "71+"]
age_pivot_marvel = age_pivot_marvel.reindex(columns=[c for c in age_order if c in age_pivot_marvel.columns])
age_pivot_marvel = age_pivot_marvel.dropna(how="all").sort_index()

if not age_pivot_marvel.empty:
    fig, ax = plt.subplots(figsize=(8, max(5, len(age_pivot_marvel) * 0.5)))
    sns.heatmap(
        age_pivot_marvel, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0.4, vmax=1.0, linewidths=0.5, ax=ax,
        cbar_kws={"label": "AUROC"},
    )
    ax.set_title("MARVEL Error Concentration by Age Group\n(unified protocol)", fontsize=11, pad=10)
    ax.set_xlabel("Age Group", fontsize=10)
    ax.set_ylabel("Disease", fontsize=10)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in PSYCHIATRIC:
            lbl.set_color("crimson")
            lbl.set_fontweight("bold")
    plt.tight_layout()
    fig.savefig(FIGS / "age_stratification_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/age_stratification_heatmap.png")

# ── Figure 2: Sex stratification (MARVEL vs Main, unified) ───────────────────

sex_data = strat_df[
    (strat_df["stratum"] == "sex") &
    (strat_df["value"].isin(["Female", "Male"])) &
    (strat_df["n_pos"] >= 2)
].copy()

if not sex_data.empty:
    diseases_sex = sex_data["disease"].unique()
    x = np.arange(len(diseases_sex))
    width = 0.2

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    combos = [("MARVEL","Female","#2166ac"), ("MARVEL","Male","#92c5de"),
              ("Main","Female","#d6604d"),   ("Main","Male","#f4a582")]
    for i, (model, sex_val, color) in enumerate(combos):
        subset = sex_data[(sex_data["model"] == model) & (sex_data["value"] == sex_val)]
        vals = [subset[subset["disease"] == d]["auroc"].mean() if d in subset["disease"].values else np.nan
                for d in diseases_sex]
        ax2.bar(x + (i - 1.5) * width, vals, width, label=f"{model} {sex_val}", color=color, alpha=0.85)

    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(diseases_sex, rotation=40, ha="right", fontsize=9)
    ax2.set_ylabel("AUROC")
    ax2.set_ylim(0.3, 1.1)
    ax2.set_title("Sex Stratification: MARVEL vs Main Model (unified protocol)", fontsize=11)
    ax2.legend(fontsize=8, ncol=4)
    for tick, name in zip(ax2.get_xticklabels(), diseases_sex):
        if name in PSYCHIATRIC:
            tick.set_color("crimson")
    plt.tight_layout()
    fig2.savefig(FIGS / "sex_stratification_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/sex_stratification_plot.png")

# ── Figure 3: Protocol sensitivity gap bar chart ──────────────────────────────

plot_gap = gap_df[gap_df["tier"].isin(["tier2"])].copy()
plot_gap = plot_gap.dropna(subset=["gap_fair_minus_unified"])
plot_gap = plot_gap.sort_values("gap_fair_minus_unified", ascending=False)

if not plot_gap.empty:
    diseases_g = plot_gap["disease"].unique()
    x = np.arange(len(diseases_g))
    width = 0.35

    fig3, ax3 = plt.subplots(figsize=(13, 5))
    for i, (model, color) in enumerate([("MARVEL","#2166ac"), ("Main","#d6604d")]):
        sub = plot_gap[plot_gap["model"] == model]
        vals = [sub[sub["disease"]==d]["gap_fair_minus_unified"].values[0]
                if d in sub["disease"].values else np.nan for d in diseases_g]
        bars = ax3.bar(x + (i - 0.5) * width, vals, width, label=model, color=color, alpha=0.85)

    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(diseases_g, rotation=40, ha="right", fontsize=9)
    ax3.set_ylabel("AUROC drop (fair − unified)")
    ax3.set_title("Protocol Sensitivity Gap: Fair vs Unified Screening (tier2)\n"
                  "Positive = model scores higher under curated binary protocol", fontsize=10)
    ax3.legend(fontsize=9)
    for tick, name in zip(ax3.get_xticklabels(), diseases_g):
        if name in PSYCHIATRIC:
            tick.set_color("crimson")
    plt.tight_layout()
    fig3.savefig(FIGS / "protocol_sensitivity_gap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/protocol_sensitivity_gap.png")

# ── Done ──────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ALL DONE")
print("="*60)
print(f"Outputs in: {OUT}")
print("  demographic_stratification_marvel.csv")
print("  protocol_sensitivity_gap.csv")
print("  figures/age_stratification_heatmap.png")
print("  figures/sex_stratification_plot.png")
print("  figures/protocol_sensitivity_gap.png")
