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

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import torch
import torchaudio

try:
    import auraloss
except ModuleNotFoundError:
    auraloss = None

try:
    from aquatk.metrics.PEAQ.peaq_basic import process_audio_files as peaq_process_files
except ModuleNotFoundError:
    peaq_process_files = None

try:
    from librosa.feature import spectral_centroid as librosa_spectral_centroid
except ModuleNotFoundError:
    librosa_spectral_centroid = None

try:
    from mel_cepstral_distance import compare_audio_files as melcd_compare_files
except ModuleNotFoundError:
    melcd_compare_files = None

try:
    import scipy.io.wavfile as wavfile
except ModuleNotFoundError:
    wavfile = None


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


_PEAQ_TMP_DIR = Path(tempfile.gettempdir()) / "ddsp_eq_metrics_peaq"
_MSD_TMP_DIR = Path(tempfile.gettempdir()) / "ddsp_eq_metrics_msd"
_MRSTFT_CACHE: dict[int, object] = {}


def _write_chunk_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write normalized float audio to int32 PCM WAV for file-based metrics."""
    if wavfile is None:
        raise ModuleNotFoundError("scipy is required for temporary WAV export")

    x = np.asarray(audio, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    x_i32 = np.round(x * 2147483647.0).astype(np.int32)
    wavfile.write(str(path), int(sr), x_i32)


def peaq(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """Compute PEAQ basic ODG using aquatk's file-based implementation."""
    if peaq_process_files is None:
        return float("nan")

    n = min(len(reference), len(degraded))
    if n == 0:
        return float("nan")

    _PEAQ_TMP_DIR.mkdir(parents=True, exist_ok=True)
    uid = uuid4().hex
    ref_path = _PEAQ_TMP_DIR / f"ref_{uid}.wav"
    deg_path = _PEAQ_TMP_DIR / f"deg_{uid}.wav"

    _write_chunk_wav(ref_path, np.asarray(reference[:n], dtype=np.float32), sr)
    _write_chunk_wav(deg_path, np.asarray(degraded[:n], dtype=np.float32), sr)

    try:
        result = peaq_process_files(str(ref_path), str(deg_path))
        odg_list = result.get("ODG_list", []) if isinstance(result, dict) else []
        if not odg_list:
            return float("nan")
        return float(np.mean(np.asarray(odg_list, dtype=float)))
    except Exception:
        return float("nan")
    finally:
        for p in (ref_path, deg_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def mrstft(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """Compute multi-resolution STFT loss using ``auraloss``."""
    if auraloss is None:
        return float("nan")

    n = min(len(reference), len(degraded))
    if n == 0:
        return float("nan")

    sr_key = int(sr)
    if sr_key not in _MRSTFT_CACHE:
        _MRSTFT_CACHE[sr_key] = auraloss.freq.MelSTFTLoss(sample_rate=sr_key)
    loss_fn = _MRSTFT_CACHE[sr_key]

    ref_t = torch.from_numpy(np.asarray(reference[:n], dtype=np.float32)).view(1, 1, -1)
    deg_t = torch.from_numpy(np.asarray(degraded[:n], dtype=np.float32)).view(1, 1, -1)
    loss_val = loss_fn(ref_t, deg_t)
    return float(loss_val.detach().cpu().item())


def msd(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """Compute Mel spectral distance via ``mel_cepstral_distance`` package."""
    if melcd_compare_files is None:
        return float("nan")

    n = min(len(reference), len(degraded))
    if n == 0:
        return float("nan")

    _MSD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    uid = uuid4().hex
    ref_path = _MSD_TMP_DIR / f"ref_{uid}.wav"
    deg_path = _MSD_TMP_DIR / f"deg_{uid}.wav"

    _write_chunk_wav(ref_path, np.asarray(reference[:n], dtype=np.float32), sr)
    _write_chunk_wav(deg_path, np.asarray(degraded[:n], dtype=np.float32), sr)

    try:
        result, _ = melcd_compare_files(
            str(ref_path),
            str(deg_path),
            sample_rate=16000,
            n_fft=32,
            win_len=32,
            hop_len=8,
            aligning="pad",
        )
        return float(result)
    except Exception:
        return float("nan")
    finally:
        for p in (ref_path, deg_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def spectral_centroid_delta(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """Compute mean absolute difference between spectral centroids."""
    if librosa_spectral_centroid is None:
        return float("nan")

    n = min(len(reference), len(degraded))
    if n == 0:
        return float("nan")

    ref = np.asarray(reference[:n], dtype=np.float32)
    deg = np.asarray(degraded[:n], dtype=np.float32)

    sc_ref = librosa_spectral_centroid(y=ref, sr=int(sr), n_fft=2048, hop_length=512)
    sc_deg = librosa_spectral_centroid(y=deg, sr=int(sr), n_fft=2048, hop_length=512)
    return float(np.mean(np.abs(sc_ref - sc_deg)))


# ---------------------------------------------------------------------------
# Metrics-evaluation utility helpers
# ---------------------------------------------------------------------------

def safe_token(text: str) -> str:
    """Normalize a token so it can be matched against filename fragments."""
    return str(text).replace("-", "_").replace(" ", "_")


def format_transition_label(transition_s: float) -> str:
    """Format a transition time in seconds for table/report display."""
    if float(transition_s).is_integer():
        return f"{int(transition_s)} s"
    return f"{transition_s:g} s"


def load_experiment_metadata(results_root: Path) -> tuple[list[str], list[float]]:
    """Load optimizer names and transition times from an experiment config.

    Args:
        results_root: Path to ``results/<experiment_name>``.

    Returns:
        Tuple ``(optimizers, transition_times)`` extracted from
        ``simulation_params`` in ``config.json``. Missing or invalid fields
        return empty lists.
    """
    config_path = results_root / "config.json"
    if not config_path.exists():
        return [], []

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    sim_cfg = cfg.get("simulation_params", {}) if isinstance(cfg, dict) else {}
    optimizers = [str(opt) for opt in sim_cfg.get("optim_type", [])]
    transition_times: list[float] = []
    for tt in sim_cfg.get("transition_time_s", []):
        try:
            transition_times.append(float(tt))
        except (TypeError, ValueError):
            continue
    return optimizers, transition_times


def parse_optimizer_from_stem(parts: list[str], known_optimizers: list[str]) -> str:
    """Infer optimizer name from an ``EQ_*.wav`` filename split into tokens."""
    post_eq_parts = parts[1:]
    candidate_tokens = sorted(
        [(safe_token(opt).split("_"), opt) for opt in known_optimizers],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for opt_tokens, opt_display in candidate_tokens:
        if post_eq_parts[: len(opt_tokens)] == opt_tokens:
            return opt_display

    if post_eq_parts:
        return post_eq_parts[0]
    return "unknown"


def parse_transition_time_from_stem(parts: list[str]) -> float | None:
    """Extract transition time from filename tokens encoded as ``tt<value>``."""
    for token in reversed(parts):
        if token.startswith("tt"):
            raw = token[2:].replace("p", ".")
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def summarize_rows(
    all_rows: list[dict],
    metrics: list[str],
    known_optimizers: list[str],
    known_transition_times: list[float],
) -> list[dict]:
    """Aggregate per-file metric rows by ``(optimizer, transition_s)``.

    For each metric, this computes mean and standard deviation across files
    in the group and returns table-ready rows.
    """
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in all_rows:
        optimizer = row.get("optimizer")
        transition_s = row.get("transition_s")
        if optimizer is None or transition_s is None:
            continue
        grouped[(str(optimizer), float(transition_s))].append(row)

    optimizer_order = {opt: idx for idx, opt in enumerate(known_optimizers)}
    transition_order = {tt: idx for idx, tt in enumerate(known_transition_times)}

    def _sort_key(item: tuple[str, float]) -> tuple[int, int, str, float]:
        optimizer, transition_s = item
        return (
            optimizer_order.get(optimizer, len(optimizer_order)),
            transition_order.get(transition_s, len(transition_order)),
            optimizer,
            transition_s,
        )

    summary_rows: list[dict] = []
    for optimizer, transition_s in sorted(grouped.keys(), key=_sort_key):
        rows = grouped[(optimizer, transition_s)]
        summary_row: dict = {
            "optimizer": optimizer,
            "transition_s": transition_s,
            "transition_label": format_transition_label(transition_s),
            "n_files": len(rows),
        }
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in rows if metric in row and np.isfinite(float(row[metric]))],
                dtype=float,
            )
            summary_row[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            summary_row[f"{metric}_std"] = float(np.std(values)) if values.size else float("nan")
        summary_rows.append(summary_row)

    return summary_rows


def load_mono_float(path: Path, sr_target: int) -> tuple[np.ndarray, int]:
    """Load a WAV file, optionally resample, and return mono float64 data.

    Args:
        path: Path to a WAV file.
        sr_target: If positive and different from file sample rate, resample
            to this rate. If ``0`` or negative, keep the original rate.

    Returns:
        Tuple ``(audio, sr)`` where ``audio`` is a mono ``float64`` array.
    """
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != sr_target and sr_target > 0:
        import torchaudio.transforms as T

        wav = T.Resample(orig_freq=int(sr), new_freq=sr_target)(wav)
        sr = sr_target
    return wav.squeeze(0).numpy().astype(np.float64), int(sr)


def apply_sample_delay(audio: np.ndarray, delay_samples: int) -> np.ndarray:
    """Delay a signal by adding leading zeros and preserving original length."""
    x = np.asarray(audio, dtype=np.float64)
    d = int(delay_samples)
    if d <= 0:
        return x
    y = np.zeros_like(x)
    if d < x.shape[0]:
        y[d:] = x[:-d]
    return y


def add_shared_dither(
    reference: np.ndarray,
    degraded: np.ndarray,
    std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add identical white-noise dither to reference and degraded signals."""
    n = min(len(reference), len(degraded))
    if n <= 0:
        return np.asarray(reference[:0], dtype=np.float64), np.asarray(degraded[:0], dtype=np.float64)

    ref = np.asarray(reference[:n], dtype=np.float64)
    deg = np.asarray(degraded[:n], dtype=np.float64)
    if std <= 0.0:
        return ref, deg

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(std), size=n)
    return ref + noise, deg + noise


# ---------------------------------------------------------------------------
# Convenience batch evaluator
# ---------------------------------------------------------------------------

_ALL_METRICS = {
    "PEAQ": peaq,
    "MRSTFT": mrstft,
    "STFT": mrstft,
    "MSD": msd,
    "Spectral-Centroid": spectral_centroid_delta,
    "Spectral Centroid": spectral_centroid_delta,
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
