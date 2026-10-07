"""
tc_dann_predict.py  —  TC-DANN inference wrapper
=================================================
Drop-in replacement for predict.py, adapted for the TC-DANN quad-model system.

Usage:
    from tc_dann_predict import TCDANNPredictor

    pred   = TCDANNPredictor("tc_dann_results/")
    result = pred.predict("recording.wav", task="prolonged-vowel")

    print(result["top3"])           # unified cross-model differential
    print(result["audit_flags"])    # confounder / quality flags

Or from command line:
    python tc_dann_predict.py recording.wav
    python tc_dann_predict.py recording.wav --task read-speech
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Union, Optional

import joblib
import numpy as np
import torch

from run_tc_dann import (
    TCDANN, MODEL_DISEASE_MAP, MODEL_DISPLAY,
    CS_PHYSICAL, CS_PSYCH, PSYCHIATRIC_NO_DEMO,
    TASKS, TASK_ENC, load_bundle, bundle_preprocessors,
)

SAMPLE_RATE = 16_000
MAX_AUDIO_S = 30

# Human-readable disease display names
DISEASE_DISPLAY = {
    "laryngeal_dystonia":             "Laryngeal Dystonia",
    "unilateral_vocal_fold_paralysis":"Unilateral Vocal Fold Paralysis",
    "benign_lesions":                 "Benign Laryngeal Lesions",
    "muscle_tension_dysphonia":       "Muscle Tension Dysphonia",
    "glottic_insufficiency":          "Glottic Insufficiency",
    "laryngeal_cancer":               "Laryngeal Cancer",
    "precancerous_lesions":           "Precancerous Lesions",
    "control":                        "Healthy Control",
    "parkinsons_disease":             "Parkinson's Disease",
    "cognitive_impairment":           "Cognitive Impairment",
    "amyotrophic_lateral_sclerosis":  "ALS",
    "airway_stenosis":                "Airway Stenosis",
    "copd_and_asthma":                "COPD / Asthma",
    "unexplained_chronic_cough":      "Unexplained Chronic Cough",
    "laryngitis":                     "Laryngitis",
    "adhd_adult":                     "ADHD (Adult)",
    "ptsd_adult":                     "PTSD (Adult)",
    "anxiety":                        "Anxiety",
    "depression":                     "Depression",
    "bipolar_disorder":               "Bipolar Disorder",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _suggestion(disease: str, confidence: float) -> str:
    display = DISEASE_DISPLAY.get(disease, disease.replace("_", " ").title())
    if confidence >= 0.65:
        return (f"High-confidence signal for {display}. Research-grade output only — "
                f"please consult a qualified clinician for evaluation.")
    if confidence >= 0.45:
        return (f"Moderate signal for {display}. Consider a follow-up recording in a "
                f"quieter environment and discuss findings with a clinician.")
    return (f"Weak signal for {display}. This confidence level is insufficient to act on — "
            f"retake the recording if concerned and consult a clinician.")


class _BundleLoader:
    """Loads and holds a single TC-DANN model bundle."""

    def __init__(self, stem: str, bundle_dir: Path):
        self.stem         = stem
        self.label        = MODEL_DISPLAY[stem]
        self.loaded       = False
        self.disease_cols: list[str] = []
        self.feat_cols:    list[str] = []
        self.model:        Optional[TCDANN]  = None
        self.imputer                         = None
        self.scaler                          = None

        path = bundle_dir / f"{stem}_model_best.joblib"
        if not path.exists():
            print(f"  WARNING: {path.name} not found — {self.label} model unavailable")
            return

        try:
            b = load_bundle(path)
            self.disease_cols = b["disease_cols"]
            self.feat_cols    = b["feat_cols"]

            self.model = TCDANN(
                n_features=b["n_features"],
                n_tasks=b["n_tasks"],
                n_sites=b.get("n_sites", 2),
                n_diseases=b["n_diseases"],
            ).to(DEVICE)

            state = {k: (torch.tensor(v).to(DEVICE) if isinstance(v, np.ndarray) else v.to(DEVICE))
                     for k, v in b["model_state"].items()}
            self.model.load_state_dict(state)
            self.model.eval()

            # Rebuild imputer + scaler from the stored arrays (sklearn-version independent)
            self.imputer, self.scaler = bundle_preprocessors(b)

            self.loaded = True
        except Exception as e:
            print(f"  ERROR loading {self.label}: {e}")

    @torch.no_grad()
    def infer(self, feat_row: dict, task_name: str) -> list[dict]:
        if not self.loaded:
            return []

        task_id = TASK_ENC.get(task_name, 0)
        X = np.zeros((1, len(self.feat_cols)), dtype=np.float32)
        for j, col in enumerate(self.feat_cols):
            if col in feat_row:
                X[0, j] = float(feat_row[col])

        X_imp = self.imputer.transform(X)
        X_sc  = self.scaler.transform(X_imp).astype(np.float32)
        X_t   = torch.tensor(X_sc).to(DEVICE)
        ids_t = torch.tensor([task_id], dtype=torch.long).to(DEVICE)

        logits, _ = self.model(X_t, ids_t, lam=0.0)
        probs     = torch.sigmoid(logits).cpu().numpy()[0]

        results = []
        for di, disease in enumerate(self.disease_cols):
            y_prob    = float(probs[di])
            is_psych  = disease in PSYCHIATRIC_NO_DEMO
            w         = CS_PSYCH if is_psych else CS_PHYSICAL
            others    = [float(probs[k]) for k in range(len(self.disease_cols)) if k != di]
            z_uncert  = max(others) if others else 0.0
            x_cosine  = 0.5
            m_demo    = 0.0 if is_psych else 0.5
            p_conf    = 0.5

            confidence = (
                w["a"] * x_cosine
                + w["b"] * y_prob
                - w["c"] * z_uncert
                + w["d"] * m_demo
                + w["e"] * p_conf
            )

            results.append({
                "disease":        disease,
                "display":        DISEASE_DISPLAY.get(disease, disease),
                "model":          self.label,
                "y_prob":         round(y_prob, 4),
                "confidence":     round(float(confidence), 4),
                "x_cosine":       x_cosine,
                "z_uncertainty":  round(z_uncert, 4),
                "m_demo":         m_demo,
                "p_confounder":   p_conf,
                "percent":        round(y_prob * 100, 1),
            })
        return results


def _extract_features_from_audio(audio_source, task_name: str) -> tuple[dict, list[str]]:
    """
    Load audio (file path, bytes, or numpy array), run preprocessing,
    return (feature_row_dict, quality_flags).
    """
    import torchaudio
    import torchaudio.transforms as T

    # Load waveform
    if isinstance(audio_source, (str, Path, bytes)):
        raw = Path(audio_source).read_bytes() if not isinstance(audio_source, bytes) else audio_source
        try:
            import io
            wav, sr = torchaudio.load(io.BytesIO(raw))
        except (ImportError, RuntimeError):
            # torchaudio >= 2.9 needs torchcodec to decode; use soundfile/librosa instead
            from audio_preprocessing import _load_bytes_fallback
            arr, sr = _load_bytes_fallback(raw)
            wav = torch.from_numpy(arr).unsqueeze(0)
    elif isinstance(audio_source, np.ndarray):
        wav = torch.from_numpy(audio_source.astype(np.float32)).unsqueeze(0)
        sr  = SAMPLE_RATE
    else:
        raise TypeError(f"Unsupported audio source: {type(audio_source)}")

    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    max_len = SAMPLE_RATE * MAX_AUDIO_S
    if wav.shape[-1] > max_len:
        wav = wav[..., :max_len]

    flags: list[str] = []
    if wav.shape[-1] < SAMPLE_RATE:
        flags.append(f"too_short: {wav.shape[-1]/SAMPLE_RATE:.2f}s (min 1s)")

    wav_np = wav.squeeze(0).numpy()

    # MFCC
    mfcc_tf = T.MFCC(
        sample_rate=SAMPLE_RATE, n_mfcc=60,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80,
                   "window_fn": torch.hann_window},
    )
    mfcc = mfcc_tf(wav).squeeze(0).numpy()   # [60, T]

    # Spectrogram
    spec_tf = T.Spectrogram(n_fft=400, hop_length=160, win_length=400, power=2.0)
    spec    = torch.log1p(spec_tf(wav)).squeeze(0).numpy()   # [201, T]

    feats: dict[str, float] = {}

    # MFCC scalars (mean, std, max, min + delta + deltadelta)
    for i in range(mfcc.shape[0]):
        row = mfcc[i]
        feats[f"mfcc{i:02d}_mean"] = float(row.mean())
        feats[f"mfcc{i:02d}_std"]  = float(row.std())
        feats[f"mfcc{i:02d}_max"]  = float(row.max())
        feats[f"mfcc{i:02d}_min"]  = float(row.min())
    if mfcc.shape[1] >= 2:
        d1 = np.diff(mfcc, axis=1)
        for i in range(d1.shape[0]):
            feats[f"mfcc{i:02d}_delta_mean"] = float(d1[i].mean())
            feats[f"mfcc{i:02d}_delta_std"]  = float(d1[i].std())
        if d1.shape[1] >= 2:
            d2 = np.diff(d1, axis=1)
            for i in range(d2.shape[0]):
                feats[f"mfcc{i:02d}_deltadelta_mean"] = float(d2[i].mean())
                feats[f"mfcc{i:02d}_deltadelta_std"]  = float(d2[i].std())

    # Spectrogram scalars
    feats["spectral_mean"]     = float(spec.mean())
    feats["spectral_std_dev"]  = float(spec.std())
    sk = spec.flatten(); mu = sk.mean(); sg = sk.std()
    feats["spectral_skewness"] = float(((sk - mu)**3).mean() / (sg**3 + 1e-8))
    feats["spectral_kurtosis"] = float(((sk - mu)**4).mean() / (sg**4 + 1e-8)) - 3.0
    feats["duration"]          = float(wav_np.shape[0] / SAMPLE_RATE)

    return feats, flags


class TCDANNPredictor:
    """
    High-level TC-DANN inference interface.

    Loads all four domain models from a tc_dann_results/ directory
    and runs unified cross-model top-3 prediction on new audio.
    """

    def __init__(self, bundle_dir: str | Path | None = None):
        self.bundle_dir = Path(bundle_dir or DEFAULT_BUNDLE_DIR).expanduser()
        self._loaders: dict[str, _BundleLoader] = {}

        print(f"Loading TC-DANN models from {self.bundle_dir} ...")
        for stem in ["voice_onco", "neurological", "respiratory", "psychiatric"]:
            loader = _BundleLoader(stem, self.bundle_dir)
            self._loaders[stem] = loader
            status = "✓" if loader.loaded else "✗"
            print(f"  {status} {loader.label:<16} ({len(loader.disease_cols)} diseases)")

        n_loaded = sum(1 for l in self._loaders.values() if l.loaded)
        print(f"\n{n_loaded}/4 models loaded on {DEVICE}\n")
        self.n_loaded = n_loaded

    def predict(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        task: str = "prolonged-vowel",
    ) -> dict:
        """
        Run TC-DANN inference on audio input.

        Args:
            audio: file path, raw bytes, or numpy waveform
            task:  one of prolonged-vowel, read-speech, free-speech, diadochokinesis

        Returns dict with:
            top3            list of top-3 differential diagnoses
            all_ranked      all diseases sorted by confidence
            audit_flags     quality + confounder flags
            top_disease     name of top-ranked disease
            confidence      confidence score of top disease
            suggestion      clinical interpretation string
            disclaimer      mandatory research disclaimer
        """
        feat_row, flags = _extract_features_from_audio(audio, task)

        all_results: list[dict] = []
        for loader in self._loaders.values():
            all_results.extend(loader.infer(feat_row, task))

        # De-duplicate by disease (keep highest confidence)
        seen: dict[str, dict] = {}
        for r in all_results:
            d = r["disease"]
            if d not in seen or r["confidence"] > seen[d]["confidence"]:
                seen[d] = r

        ranked = sorted(seen.values(), key=lambda x: x["confidence"], reverse=True)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1

        top3 = ranked[:3]
        top1 = top3[0] if top3 else {}

        audit_flags = []
        for f in flags:
            audit_flags.append({"level": "warn", "msg": f"audio_quality: {f}"})
        for r in top3:
            if r.get("y_prob", 1.0) < 0.20:
                audit_flags.append({
                    "level": "info",
                    "msg":   f"{r['disease']}: low raw probability ({r['y_prob']:.3f})"
                })

        return {
            "top_disease":   top1.get("disease", ""),
            "top_display":   top1.get("display", ""),
            "confidence":    top1.get("confidence", 0.0),
            "suggestion":    _suggestion(top1.get("disease", ""), top1.get("confidence", 0.0)),
            "top3":          top3,
            "all_ranked":    ranked,
            "audit_flags":   audit_flags,
            "task":          task,
            "disclaimer":    (
                "Research-grade screening output from the Bridge2AI-Voice TC-DANN model. "
                "Not a medical device. Not FDA-cleared. Not for diagnostic or clinical decision-making."
            ),
        }


DEFAULT_BUNDLE_DIR = Path(os.getenv(
    "TC_DANN_BUNDLE_DIR", str(Path(__file__).resolve().parent.parent / "models")))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TC-DANN voice inference")
    parser.add_argument("audio",       help="Path to audio file (WAV/MP3/etc.)")
    parser.add_argument("--bundle_dir", default=None,
                        help="Folder with *_model_best.joblib (default: $TC_DANN_BUNDLE_DIR or <repo>/models)")
    parser.add_argument("--task",      default="prolonged-vowel",
                        choices=TASKS)
    args = parser.parse_args()

    predictor = TCDANNPredictor(args.bundle_dir)
    if predictor.n_loaded == 0:
        sys.exit(
            f"No TC-DANN bundles found in {predictor.bundle_dir}. Trained weights are not "
            "distributed (Bridge2AI-Voice / PhysioNet DUA). Train them with "
            "`python tcdann/run_tc_dann.py --data_root $B2AI_DATA_ROOT --out_dir models` "
            "or pass --bundle_dir / set TC_DANN_BUNDLE_DIR.")
    result    = predictor.predict(args.audio, task=args.task)

    print(f"\nTop disease  : {result['top_display']}")
    print(f"Confidence   : {result['confidence']:.4f}")
    print(f"Suggestion   : {result['suggestion']}")
    print(f"\nTop-3 differential:")
    for r in result["top3"]:
        print(f"  #{r['rank']}  [{r['model']:<14}]  {r['display']:<40}  conf={r['confidence']:.3f}  prob={r['y_prob']:.3f}")
    if result["audit_flags"]:
        print(f"\nAudit flags:")
        for f in result["audit_flags"]:
            print(f"  [{f['level'].upper()}] {f['msg']}")
    print(f"\n{result['disclaimer']}")
