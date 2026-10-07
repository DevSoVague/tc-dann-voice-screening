"""
audio_preprocessing.py
=======================
B2AI-Voice faithful audio preprocessing pipeline.

Mirrors torchaudio_mfcc.parquet + torchaudio_mel_spectrogram.parquet
extraction from EDA_v3.ipynb:
  - MFCC:        N_MFCC=60 coefficients, hop=10ms, win=25ms → [60, T]
  - Log-Mel:     128 mel bins (B2AI used 128-bin mel for visual inspection)
                 Model spec branch = [201, T] (linear spectrogram)
  - Spectrogram: n_fft=400 → 201 frequency bins → [201, T]
  - Resampling:  16 kHz target (B2AI raw recordings were 44.1 kHz device capture)
  - Normalise:   peak-normalise to -3 dBFS, strip head/tail silence
  - Quality:     SNR check, clipping check, min-duration gate

Usage:
    from audio_preprocessing import preprocess_audio_bytes, preprocess_audio_file
    result = preprocess_audio_bytes(wav_bytes)
    spec_tensor  = result["spec"]     # np.float32 [1, 201, T]
    mfcc_tensor  = result["mfcc"]     # np.float32 [1, 60,  T]
    flags        = result["flags"]    # list of warning strings
"""

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("voxclinbench.preprocess")

# ── Try torchaudio first, fall back to librosa ───────────────────────────────
try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    _BACKEND = "torchaudio"
except ImportError:
    torch = None
    torchaudio = None
    T = None
    _BACKEND = None

try:
    import librosa
    if _BACKEND is None:
        _BACKEND = "librosa"
except ImportError:
    librosa = None

if _BACKEND is None:
    raise ImportError(
        "Either torchaudio or librosa must be installed.\n"
        "  pip install torchaudio   (preferred, matches B2AI pipeline)\n"
        "  pip install librosa      (fallback)"
    )

log.info("Audio backend: %s", _BACKEND)

# ── Constants — matched to B2AI torchaudio feature extraction ────────────────
TARGET_SR       = 16_000       # resample target (Hz)
N_FFT           = 400          # 25 ms @ 16 kHz  →  201 freq bins
HOP_LENGTH      = 160          # 10 ms @ 16 kHz
WIN_LENGTH      = 400          # 25 ms @ 16 kHz
N_MFCC          = 60           # matches EDA_v3 N_MFCC = 60
N_MELS          = 128          # mel bins for log-mel branch
SPEC_FREQ_BINS  = N_FFT // 2 + 1   # 201
PEAK_TARGET_DB  = -3.0         # peak normalise to -3 dBFS
SILENCE_DB      = -40.0        # head/tail silence threshold
MIN_SILENCE_SEC = 0.5          # minimum segment length to keep
MIN_DURATION_S  = 1.0          # reject recordings shorter than this
CLIP_THRESHOLD  = 0.99         # samples > 99% of max = clipping
CLIP_MAX_FRAC   = 0.005        # max 0.5% clipped samples before warning
MIN_SNR_DB      = 20.0         # minimum SNR (simple energy ratio estimate)


# ── Public API ───────────────────────────────────────────────────────────────

def preprocess_audio_bytes(
    audio_bytes: bytes,
    fmt: str = "wav",
) -> dict:
    """
    Full preprocessing pipeline from raw audio bytes.

    Returns dict:
        spec      : np.float32 [1, 201, T]  — linear spectrogram (model spec branch)
        mfcc      : np.float32 [1, 60,  T]  — MFCC array (model MFCC branch)
        log_mel   : np.float32 [1, 128, T]  — log-mel spectrogram (for visualisation)
        sr        : int                      — sample rate after resampling
        n_frames  : int                      — T (time frames)
        duration_s: float                    — duration in seconds
        flags     : list[str]                — quality warnings
    """
    waveform, sr = _load_bytes(audio_bytes)
    return _pipeline(waveform, sr)


def preprocess_audio_file(path: str | Path) -> dict:
    """Same as preprocess_audio_bytes but reads from a file path."""
    path = Path(path)
    with open(path, "rb") as fh:
        return preprocess_audio_bytes(fh.read(), fmt=path.suffix.lstrip("."))


# ── Internal pipeline ─────────────────────────────────────────────────────────

def _load_bytes_fallback(audio_bytes: bytes):
    """Decode with soundfile (WAV/FLAC/OGG), then librosa/audioread for the rest."""
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), int(sr)
    except Exception:
        import librosa as _lr
        waveform, sr = _lr.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        return waveform.astype(np.float32), int(sr)


def _load_bytes(audio_bytes: bytes):
    """Load audio bytes → (waveform float32 numpy [samples], sr)."""
    if _BACKEND == "torchaudio":
        try:
            buf = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(buf)          # [C, N] float32 tensor
            waveform = waveform.mean(dim=0).numpy()       # mono, [N]
            return waveform.astype(np.float32), int(sr)
        except (ImportError, RuntimeError):
            # torchaudio >= 2.9 needs the separate torchcodec package for decoding;
            # fall back to soundfile / librosa.
            return _load_bytes_fallback(audio_bytes)
    else:
        buf = io.BytesIO(audio_bytes)
        waveform, sr = librosa.load(buf, sr=None, mono=True)
        return waveform.astype(np.float32), int(sr)


def _pipeline(waveform: np.ndarray, sr: int) -> dict:
    flags = []

    # 1. Resample to TARGET_SR
    if sr != TARGET_SR:
        waveform = _resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR

    # 2. Clipping check (before normalise)
    clip_frac = float(np.mean(np.abs(waveform) > CLIP_THRESHOLD * np.max(np.abs(waveform) + 1e-8)))
    if clip_frac > CLIP_MAX_FRAC:
        flags.append(f"clipping: {clip_frac*100:.1f}% of samples near full-scale — re-record advised")

    # 3. Peak-normalise to PEAK_TARGET_DB
    peak = np.max(np.abs(waveform))
    if peak > 1e-8:
        target_linear = 10 ** (PEAK_TARGET_DB / 20.0)
        waveform = waveform * (target_linear / peak)

    # 4. Strip head/tail silence
    waveform = _strip_silence(waveform, sr)

    # 5. Duration check
    duration_s = len(waveform) / sr
    if duration_s < MIN_DURATION_S:
        flags.append(f"too_short: {duration_s:.2f}s — minimum is {MIN_DURATION_S}s")

    # 6. SNR estimate
    snr = _estimate_snr(waveform)
    if snr < MIN_SNR_DB:
        flags.append(f"low_snr: estimated SNR = {snr:.1f} dB (threshold {MIN_SNR_DB} dB)")

    # 7. Extract features
    spec    = _extract_spectrogram(waveform, sr)    # [201, T]
    mfcc    = _extract_mfcc(waveform, sr)           # [60,  T]
    log_mel = _extract_log_mel(waveform, sr)        # [128, T]

    n_frames = spec.shape[1]

    # Quality gate: flag very short feature sequences
    if n_frames < 10:
        flags.append(f"flag_short_recording: only {n_frames} frames — matches B2AI quality filter")

    return {
        "spec":       spec[np.newaxis].astype(np.float32),      # [1, 201, T]
        "mfcc":       mfcc[np.newaxis].astype(np.float32),      # [1, 60,  T]
        "log_mel":    log_mel[np.newaxis].astype(np.float32),   # [1, 128, T]
        "sr":         sr,
        "n_frames":   n_frames,
        "duration_s": round(duration_s, 3),
        "flags":      flags,
    }


# ── Feature extractors ────────────────────────────────────────────────────────

def _extract_spectrogram(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Linear power spectrogram → [201, T]
    Matches torchaudio_spectrogram.parquet shape from B2AI:
      n_fft=400 → 201 bins, hop=160 (10 ms), win=400 (25 ms)
    """
    if _BACKEND == "torchaudio":
        transform = T.Spectrogram(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            power=2.0,
            normalized=False,
        )
        t = torch.from_numpy(waveform).unsqueeze(0)   # [1, N]
        spec = transform(t).squeeze(0).numpy()         # [201, T]
    else:
        spec = np.abs(librosa.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )) ** 2                                        # [201, T]

    # Convert to dB-scale (log) — more stable for the model
    spec = np.log1p(spec)
    return spec.astype(np.float32)


def _extract_mfcc(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    MFCC [60, T] — exactly mirrors torchaudio_mfcc.parquet from B2AI:
      N_MFCC=60, n_fft=400 (25ms), hop=160 (10ms), n_mels=128
    """
    if _BACKEND == "torchaudio":
        transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=N_MFCC,
            melkwargs={
                "n_fft":      N_FFT,
                "hop_length": HOP_LENGTH,
                "win_length": WIN_LENGTH,
                "n_mels":     N_MELS,
                "f_min":      0.0,
                "f_max":      sr / 2.0,
            },
        )
        t    = torch.from_numpy(waveform).unsqueeze(0)
        mfcc = transform(t).squeeze(0).numpy()   # [60, T]
    else:
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
        )                                        # [60, T]

    return mfcc.astype(np.float32)


def _extract_log_mel(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Log-Mel spectrogram [128, T] — mirrors torchaudio_mel_spectrogram.parquet.
    Used for visualisation in the frontend; not fed into the model directly.
    """
    if _BACKEND == "torchaudio":
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            f_min=0.0,
            f_max=sr / 2.0,
        )
        t       = torch.from_numpy(waveform).unsqueeze(0)
        mel     = mel_transform(t).squeeze(0).numpy()    # [128, T]
        log_mel = np.log1p(mel)
    else:
        mel     = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            n_mels=N_MELS,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel.astype(np.float32)


# ── MFCC scalar aggregation (mirrors MFCCDataLoader from EDA_v3.ipynb) ────────

def aggregate_mfcc_scalars(mfcc: np.ndarray) -> dict:
    """
    [60, T] → 480 scalar features matching EDA_v3 MFCCDataLoader output:
      mfcc{i:02d}_mean/std/max/min           (240 features)
      mfcc{i:02d}_delta_mean/std             (120 features)
      mfcc{i:02d}_deltadelta_mean/std        (120 features)
    Used for the confounder baseline logistic regression.
    """
    feats = {}
    for i, row in enumerate(mfcc):
        feats[f"mfcc{i:02d}_mean"] = float(row.mean())
        feats[f"mfcc{i:02d}_std"]  = float(row.std())
        feats[f"mfcc{i:02d}_max"]  = float(row.max())
        feats[f"mfcc{i:02d}_min"]  = float(row.min())

    # Delta MFCC
    if mfcc.shape[1] >= 2:
        delta = np.diff(mfcc, axis=1)
        for i in range(delta.shape[0]):
            feats[f"mfcc{i:02d}_delta_mean"] = float(delta[i].mean())
            feats[f"mfcc{i:02d}_delta_std"]  = float(delta[i].std())

        # Delta-delta MFCC
        if delta.shape[1] >= 2:
            dd = np.diff(delta, axis=1)
            for i in range(dd.shape[0]):
                feats[f"mfcc{i:02d}_deltadelta_mean"] = float(dd[i].mean())
                feats[f"mfcc{i:02d}_deltadelta_std"]  = float(dd[i].std())

    return feats


# ── Signal utilities ──────────────────────────────────────────────────────────

def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if _BACKEND == "torchaudio":
        t        = torch.from_numpy(waveform).unsqueeze(0)
        resampled = torchaudio.functional.resample(t, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    else:
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def _strip_silence(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Strip head/tail silence using energy threshold (mirrors B2AI quality filter)."""
    threshold_linear = 10 ** (SILENCE_DB / 20.0)
    min_samples      = int(MIN_SILENCE_SEC * sr)
    energy           = np.abs(waveform)

    # Find first and last index above threshold
    above = np.where(energy > threshold_linear)[0]
    if len(above) == 0:
        return waveform  # all silence — return as-is, will be flagged

    start = max(0, above[0] - min_samples // 2)
    end   = min(len(waveform), above[-1] + min_samples // 2)
    trimmed = waveform[start:end]

    # Only return trimmed if it's long enough
    return trimmed if len(trimmed) >= min_samples else waveform


def _estimate_snr(waveform: np.ndarray) -> float:
    """
    Simple frame-energy SNR estimate.
    Sorts 10ms frames by energy; bottom 10% = noise floor, top 50% = signal.
    """
    frame_len = int(0.01 * TARGET_SR)   # 10ms
    if len(waveform) < frame_len * 4:
        return 99.0                      # too short to estimate — assume ok

    n_frames  = len(waveform) // frame_len
    frames    = waveform[:n_frames * frame_len].reshape(n_frames, frame_len)
    energies  = np.mean(frames ** 2, axis=1) + 1e-10
    energies_sorted = np.sort(energies)

    noise_energy  = np.mean(energies_sorted[:max(1, n_frames // 10)])
    signal_energy = np.mean(energies_sorted[n_frames // 2:])
    snr_db        = 10 * np.log10(signal_energy / noise_energy)
    return float(snr_db)
