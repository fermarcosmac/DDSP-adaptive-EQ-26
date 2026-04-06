"""
Audio quality metrics for experiment evaluation.

Each metric function has the signature:
    metric(reference, degraded, sr) -> float

Available metrics:
- rmse:               Root-mean-square error in the time domain.
- si_sdr:             Scale-invariant signal-to-distortion ratio (dB).
- lufs_difference:    Absolute difference in integrated loudness (LUFS).
- fd_mse_framewise:   Per-frame frequency-domain MSE over a log-smoothed ROI.

A convenience wrapper ``evaluate_pair`` runs all requested metrics and
returns a results dict.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def rmse(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """Root-mean-square error between reference and degraded signals.

    Args:
        reference: Ground-truth signal (float, any amplitude).
        degraded:  Processed/degraded signal (same length as reference).
        sr:        Sample rate in Hz (unused, kept for uniform signature).

    Returns:
        RMSE as a non-negative float.
    """
    ref = np.asarray(reference, dtype=np.float64)
    deg = np.asarray(degraded, dtype=np.float64)
    n = min(len(ref), len(deg))
    if n == 0:
        return float("nan")
    err = ref[:n] - deg[:n]
    return float(np.sqrt(np.mean(err ** 2)))


def si_sdr(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """Scale-invariant signal-to-distortion ratio in dB.

    Computes:
        SI-SDR = 10 * log10(||alpha * ref||^2 / ||alpha * ref - deg||^2)
    where alpha = <deg, ref> / <ref, ref>.

    Args:
        reference: Ground-truth signal.
        degraded:  Estimated/processed signal.
        sr:        Sample rate in Hz (unused).

    Returns:
        SI-SDR in dB.
    """
    ref = np.asarray(reference, dtype=np.float64)
    deg = np.asarray(degraded, dtype=np.float64)
    n = min(len(ref), len(deg))
    if n == 0:
        return float("nan")
    ref = ref[:n]
    deg = deg[:n]

    ref_energy = float(np.dot(ref, ref))
    if ref_energy < 1e-12:
        return float("nan")

    alpha = float(np.dot(deg, ref)) / ref_energy
    target = alpha * ref
    noise = deg - target

    target_power = float(np.dot(target, target))
    noise_power = float(np.dot(noise, noise))
    if noise_power < 1e-12:
        return float("inf")

    return float(10.0 * np.log10(target_power / noise_power))


def lufs_difference(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """Absolute difference in integrated loudness between reference and degraded (LUFS).

    Uses ``torchaudio.functional.loudness`` (ITU-R BS.1770-4).

    Args:
        reference: Ground-truth signal (mono float).
        degraded:  Processed signal (mono float).
        sr:        Sample rate in Hz.

    Returns:
        Absolute LUFS difference (non-negative float).
    """
    def _lufs(x: np.ndarray, sr: int) -> float:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        val = torchaudio.functional.loudness(t, sr)
        return float(val.item())

    ref = np.asarray(reference, dtype=np.float64)
    deg = np.asarray(degraded, dtype=np.float64)
    n = min(len(ref), len(deg))
    if n < int(sr * 0.4):  # BS.1770 needs at least ~400 ms
        return float("nan")

    try:
        lufs_ref = _lufs(ref[:n], sr)
        lufs_deg = _lufs(deg[:n], sr)
    except Exception:
        return float("nan")

    if not (np.isfinite(lufs_ref) and np.isfinite(lufs_deg)):
        return float("nan")
    return float(abs(lufs_ref - lufs_deg))


def fd_mse_framewise(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int,
    frame_len: int = 8192,
    hop_len: Optional[int] = None,
    roi: Tuple[float, float] = (50.0, 20000.0),
    n_log_points: int = 256,
) -> np.ndarray:
    """Per-frame frequency-domain MSE over a log-smoothed ROI.

    For each frame, computes the power spectrum of the reference and
    degraded signals, interpolates both to a log-frequency grid within
    the ROI, and returns the MSE between log-magnitude spectra.

    Args:
        reference:    Ground-truth audio (float).
        degraded:     Processed audio (float).
        sr:           Sample rate in Hz.
        frame_len:    Analysis frame length in samples.
        hop_len:      Hop size in samples (defaults to ``frame_len``).
        roi:          Frequency region of interest ``(f_low, f_high)`` in Hz.
        n_log_points: Number of log-frequency grid points.

    Returns:
        1-D float64 array of per-frame FD-MSE values.
    """
    if hop_len is None:
        hop_len = frame_len

    ref = np.asarray(reference, dtype=np.float64)
    deg = np.asarray(degraded, dtype=np.float64)
    n = min(len(ref), len(deg))
    if n < frame_len:
        return np.array([], dtype=np.float64)

    n_frames = (n - frame_len) // hop_len + 1
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    f_log = np.logspace(np.log10(max(roi[0], freqs[1])), np.log10(roi[1]), n_log_points)
    mask = (freqs >= roi[0]) & (freqs <= roi[1])

    out = np.zeros(n_frames, dtype=np.float64)
    for k in range(n_frames):
        s = k * hop_len
        e = s + frame_len
        R = np.abs(np.fft.rfft(ref[s:e]))
        D = np.abs(np.fft.rfft(deg[s:e]))
        r_db = 20.0 * np.log10(R[mask] + 1e-8)
        d_db = 20.0 * np.log10(D[mask] + 1e-8)
        f_roi = freqs[mask]
        r_log = np.interp(f_log, f_roi, r_db)
        d_log = np.interp(f_log, f_roi, d_db)
        out[k] = float(np.mean((r_log - d_log) ** 2))

    return out


# ---------------------------------------------------------------------------
# Convenience batch evaluator
# ---------------------------------------------------------------------------

_ALL_METRICS = {
    "RMSE": rmse,
    "SI-SDR": si_sdr,
    "LUFS-diff": lufs_difference,
}


def evaluate_pair(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Run a set of metrics on a reference/degraded pair.

    Args:
        reference: Ground-truth audio array (float).
        degraded:  Processed/equalized audio array (float).
        sr:        Sample rate in Hz.
        metrics:   List of metric names to compute. Defaults to all available
                   metrics: ``["RMSE", "SI-SDR", "LUFS-diff"]``.

    Returns:
        Dict mapping metric name to float score.

    Raises:
        ValueError: If an unknown metric name is requested.
    """
    if metrics is None:
        metrics = list(_ALL_METRICS.keys())

    results: Dict[str, float] = {}
    for name in metrics:
        if name not in _ALL_METRICS:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(_ALL_METRICS)}")
        try:
            results[name] = _ALL_METRICS[name](reference, degraded, sr)
        except Exception as exc:
            results[name] = float("nan")
            print(f"[metrics] Warning: {name} failed — {exc}")
    return results
