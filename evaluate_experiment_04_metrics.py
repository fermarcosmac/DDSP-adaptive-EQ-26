from __future__ import annotations

import csv
from dataclasses import dataclass
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import os
from pathlib import Path
import re
import tempfile
from typing import Callable
from uuid import uuid4

from aquatk.metrics.PEAQ.peaq_basic import process_audio_files as peaq_process_files
from aquatk.metrics.errors import si_sdr, mean_squared_error
from torchaudio.functional import loudness
from mel_cepstral_distance import compare_audio_files as melcd_compare_files
import auraloss
try:
    from librosa.feature import spectral_centroid
except ModuleNotFoundError:
    spectral_centroid = None
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from tqdm.auto import tqdm

# Use LaTeX-style mathtext for figure text and numbers
mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    }
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXPERIMENT_NAME = "experiment_04_ALL_SONGS_MOVING_POSITION"
MODE = "ALL_SONGS"  # "ALL_SONGS" or "WHITE_NOISE"
WINDOW_SECONDS = 180.0  # no-overlap sliding window length
EVAL_LAST_S = 150.0  # if > 0, evaluate once on last EVAL_LAST_S seconds instead of windowing
REFERENCE_DELAY_SAMPLES = 300  # delay applied to reference before metric windowing
DITHER_STD = 1e-8  # std of shared white-noise dither added to both compared signals
DITHER_SEED = 12345
MAX_PLOTTED_ERRORBARS = 12
SHOW_TQDM_PROGRESS = True
SUPPRESS_INTERNAL_METRIC_PRINTS = True

# -----------------------------------------------------------------------------
# Metric placeholders (replace with your final implementations later)
# Each function must take (reference, degraded, sf) and return a float.
# -----------------------------------------------------------------------------


def _safe_rmse_proxy(reference: np.ndarray, degraded: np.ndarray) -> float:
    n = min(reference.shape[0], degraded.shape[0])
    if n == 0:
        return np.nan
    err = reference[:n] - degraded[:n]
    return float(np.sqrt(np.mean(err**2) + 1e-12))


_PEAQ_TMP_DIR = Path(tempfile.mkdtemp(prefix="peaq_windows_"))
_MCD_TMP_DIR = Path(tempfile.mkdtemp(prefix="mcd_windows_"))


def _write_chunk_wav(path: Path, x: np.ndarray, sr: int) -> None:
    # Write int32 PCM to maximize compatibility with file-based metric implementations.
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    x_i32 = np.round(x * 2147483647.0).astype(np.int32)
    wavfile.write(str(path), int(sr), x_i32)


def apply_sample_delay(x: np.ndarray, delay_samples: int) -> np.ndarray:
    """Delay signal by inserting leading zeros and truncating to original length."""
    x = np.asarray(x)
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
    std: float = DITHER_STD,
    seed: int = DITHER_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the same low-level white noise to both signals (after delay handling)."""
    n = min(reference.shape[0], degraded.shape[0])
    if n <= 0:
        return reference[:0], degraded[:0]
    if std <= 0.0:
        return reference[:n], degraded[:n]

    ref_n = np.asarray(reference[:n], dtype=np.float32)
    deg_n = np.asarray(degraded[:n], dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=n).astype(np.float32)
    return ref_n + noise, deg_n + noise


def metric_peaq(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    #return 0.0
    n = min(reference.shape[0], degraded.shape[0])
    if n == 0:
        return np.nan

    ref_chunk = reference[:n]
    deg_chunk = degraded[:n]
    sr = int(round(float(sf)))

    uid = uuid4().hex
    ref_path = _PEAQ_TMP_DIR / f"ref_{uid}.wav"
    deg_path = _PEAQ_TMP_DIR / f"deg_{uid}.wav"

    _write_chunk_wav(ref_path, ref_chunk, sr)
    _write_chunk_wav(deg_path, deg_chunk, sr)

    try:
        result = peaq_process_files(str(ref_path), str(deg_path))
        avg_ODG = np.mean(result["ODG_list"])
        return float(avg_ODG)
    except Exception as e:
        print(f"Error occurred while processing PEAQ: {e}")
        return float(-4.0)  # TODO Worst ODG fallback See why this is failing
    finally:
        for p in (ref_path, deg_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def metric_si_sdr(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    # Optional debug line for quick visual inspection when this metric is hit:
    # save_metric_temp_plot(reference, degraded, sf)
    #return 0.0
    return si_sdr(reference, degraded)


def metric_mrstft_error(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    mrstft = auraloss.freq.MelSTFTLoss(sf)
    return mrstft(torch.from_numpy(reference).float().view(1,1,-1), torch.from_numpy(degraded).float().view(1,1,-1)).numpy()


def metric_mel_spectral_distance(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    #return 0.0
    n = min(reference.shape[0], degraded.shape[0])
    if n == 0:
        return np.nan

    ref_chunk = reference[:n]
    deg_chunk = degraded[:n]
    sr = int(round(float(sf)))

    uid = uuid4().hex
    ref_path = _MCD_TMP_DIR / f"ref_{uid}.wav"
    deg_path = _MCD_TMP_DIR / f"deg_{uid}.wav"

    _write_chunk_wav(ref_path, ref_chunk, sr)
    _write_chunk_wav(deg_path, deg_chunk, sr)

    try:
        # Use pad alignment for significantly faster runtime during large evaluations.
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
    except:
        return float('inf') # TODO Worst MCD fallback
    finally:
        for p in (ref_path, deg_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def metric_spectral_centroid_delta(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    #return 0.0
    if spectral_centroid is None:
        raise ImportError(
            "librosa is required for spectral centroid metric. "
            "Activate the AudioEval environment before running this script."
        )
    sc_ref = spectral_centroid(
        y=reference,
        sr=int(round(float(sf))),
        n_fft=2048,
        hop_length=512,
    )
    sc_deg = spectral_centroid(
        y=degraded,
        sr=int(round(float(sf))),
        n_fft=2048,
        hop_length=512,
    )
    return float(np.mean(np.abs(sc_ref - sc_deg)))


def metric_spectral_bandwidth_delta(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    return 0.0
    return _safe_rmse_proxy(reference, degraded)


def metric_spectral_flatness_delta(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    return 0.0
    return _safe_rmse_proxy(reference, degraded)


def metric_rmse(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    #return 0.0
    n = min(reference.shape[0], degraded.shape[0])
    if n == 0:
        return np.nan
    err = reference[:n] - degraded[:n]
    return float(np.sqrt(np.mean(err**2) + 1e-12))


def metric_lufs_difference(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    #return 0.0
    loudness_ref = loudness(torch.from_numpy(reference).view(1,-1), sample_rate=int(sf)).item()
    loudness_deg = loudness(torch.from_numpy(degraded).view(1,-1), sample_rate=int(sf)).item()
    return np.abs(loudness_ref - loudness_deg)


def metric_crest_factor_difference(reference: np.ndarray, degraded: np.ndarray, sf: float) -> float:
    return 0.0
    return _safe_rmse_proxy(reference, degraded)


METRICS: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
    "PEAQ": metric_peaq,
    "SI-SDR": metric_si_sdr,
    "MR-STFT Error": metric_mrstft_error,
    "Mel Spectral Distance": metric_mel_spectral_distance,
    "Spectral Centroid Delta": metric_spectral_centroid_delta,
    "RMSE": metric_rmse,
    "LUFS Difference": metric_lufs_difference,
    "Crest Factor Difference": metric_crest_factor_difference,
}


@dataclass(frozen=True)
class EqCondition:
    optimizer: str
    loss: str
    transition_token: str
    transition_seconds: float


EQ_FILENAME_RE = re.compile(
    r"^EQ_(?P<optimizer>.+)_(?P<loss>FD_MSE|TD_MSE)_tt(?P<tt>[^_]+)_(?P<song>.+)$"
)


@dataclass
class ParsedFiles:
    desired: dict[str, Path]
    noeq: dict[str, Path]
    eq: dict[tuple[str, EqCondition], Path]


def parse_transition_seconds(tt_token: str) -> float:
    # Stored as e.g. "1p0" in filenames.
    try:
        return float(tt_token.replace("p", "."))
    except ValueError:
        return float("nan")


def normalize_loss_label(loss_token: str) -> str:
    if loss_token == "FD_MSE":
        return "FD-MSE"
    if loss_token == "TD_MSE":
        return "TD-MSE"
    return loss_token


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        peak = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / float(max(peak, 1))
    else:
        data = data.astype(np.float32)
    return data, int(sr)


def parse_output_directory(output_dir: Path) -> ParsedFiles:
    desired: dict[str, Path] = {}
    noeq: dict[str, Path] = {}
    eq: dict[tuple[str, EqCondition], Path] = {}

    for wav_path in sorted(output_dir.glob("*.wav")):
        stem = wav_path.stem

        if stem.startswith("desired_"):
            song = stem[len("desired_") :]
            desired[song] = wav_path
            continue

        if stem.startswith("noEQ_"):
            song = stem[len("noEQ_") :]
            noeq[song] = wav_path
            continue

        match = EQ_FILENAME_RE.match(stem)
        if match is None:
            continue

        optimizer = match.group("optimizer")
        loss = normalize_loss_label(match.group("loss"))
        tt_token = match.group("tt")
        transition_seconds = parse_transition_seconds(tt_token)
        song = match.group("song")

        cond = EqCondition(
            optimizer=optimizer,
            loss=loss,
            transition_token=tt_token,
            transition_seconds=transition_seconds,
        )
        eq[(song, cond)] = wav_path

    return ParsedFiles(desired=desired, noeq=noeq, eq=eq)


def mode_song_filter(song_name: str) -> bool:
    if MODE == "ALL_SONGS":
        return True
    if MODE == "WHITE_NOISE":
        return "white_noise" in song_name.lower()
    raise ValueError(f"Unsupported MODE={MODE}. Use ALL_SONGS or WHITE_NOISE.")


def compute_windowed_metric(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int,
    window_seconds: float,
    metric_fn: Callable[[np.ndarray, np.ndarray, float], float],
) -> np.ndarray:
    n = min(reference.shape[0], degraded.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    if EVAL_LAST_S > 0:
        eval_len = int(round(EVAL_LAST_S * sr))
        if eval_len <= 0:
            raise ValueError("EVAL_LAST_S must result in at least one sample.")
        start = max(0, n - eval_len)

        def _call_metric(ref_w: np.ndarray, deg_w: np.ndarray, sample_rate: float) -> float:
            if not SUPPRESS_INTERNAL_METRIC_PRINTS:
                return metric_fn(ref_w, deg_w, sample_rate)

            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    return metric_fn(ref_w, deg_w, sample_rate)

        value = _call_metric(reference[start:n], degraded[start:n], float(sr))
        return np.array([value], dtype=np.float32)

    window_len = int(round(window_seconds * sr))
    if window_len <= 0:
        raise ValueError("WINDOW_SECONDS must result in at least one sample per window.")

    n_windows = n // window_len
    if n_windows == 0:
        return np.zeros(0, dtype=np.float32)

    out = np.empty(n_windows, dtype=np.float32)

    def _call_metric(ref_w: np.ndarray, deg_w: np.ndarray, sample_rate: float) -> float:
        if not SUPPRESS_INTERNAL_METRIC_PRINTS:
            return metric_fn(ref_w, deg_w, sample_rate)

        # Silence internal metric prints (e.g., third-party metric functions)
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                return metric_fn(ref_w, deg_w, sample_rate)

    for i in range(n_windows):
        start = i * window_len
        end = start + window_len
        out[i] = _call_metric(reference[start:end], degraded[start:end], float(sr))
    return out


def align_and_stack(series_list: list[np.ndarray]) -> np.ndarray:
    valid = [s for s in series_list if s.size > 0 and np.isfinite(s).any()]
    if not valid:
        return np.zeros((0, 0), dtype=np.float32)
    min_len = min(s.shape[0] for s in valid)
    return np.stack([s[:min_len] for s in valid], axis=0)


def compute_all_metrics(parsed: ParsedFiles, output_dir: Path) -> dict:
    songs = sorted(set(parsed.desired.keys()) & set(parsed.noeq.keys()))
    songs = [s for s in songs if mode_song_filter(s)]

    if not songs:
        raise RuntimeError(
            f"No songs available in mode {MODE}. Check files in: {output_dir}"
        )

    # Metric values per song and condition
    # noeq_metrics[metric][song] -> np.ndarray(time)
    noeq_metrics: dict[str, dict[str, np.ndarray]] = {name: {} for name in METRICS}

    # eq_metrics[metric][(cond, song)] -> np.ndarray(time)
    eq_metrics: dict[str, dict[tuple[EqCondition, str], np.ndarray]] = {
        name: {} for name in METRICS
    }

    all_conditions = sorted(
        {cond for (song, cond) in parsed.eq.keys() if song in songs},
        key=lambda c: (c.transition_seconds, c.loss, c.optimizer),
    )

    if not all_conditions:
        raise RuntimeError(f"No EQ files found in: {output_dir}")

    # Add synthetic noEQ conditions as optimizer="No EQ" for each (loss, transition).
    none_conditions: list[EqCondition] = []
    seen_loss_tt: set[tuple[str, str]] = set()
    for cond in all_conditions:
        key = (cond.loss, cond.transition_token)
        if key in seen_loss_tt:
            continue
        seen_loss_tt.add(key)
        none_conditions.append(
            EqCondition(
                optimizer="No EQ",
                loss=cond.loss,
                transition_token=cond.transition_token,
                transition_seconds=cond.transition_seconds,
            )
        )

    all_conditions = sorted(
        all_conditions + none_conditions,
        key=lambda c: (c.transition_seconds, c.loss, c.optimizer),
    )

    print(f"Mode: {MODE}")
    print(f"Experiment output directory: {output_dir}")
    print(f"Songs selected: {len(songs)}")
    for s in songs:
        print(f"  - {s}")

    total_metric_evals = 0
    for song in songs:
        available_cond = sum(
            1
            for cond in all_conditions
            if cond.optimizer == "No EQ" or (song, cond) in parsed.eq
        )
        total_metric_evals += len(METRICS) * (1 + available_cond)

    pbar_ctx = (
        tqdm(total=total_metric_evals, desc="Evaluating metrics", unit="eval")
        if SHOW_TQDM_PROGRESS
        else nullcontext()
    )

    with pbar_ctx as pbar:
        for song in songs:
            desired, sr_ref = load_audio_mono(parsed.desired[song])
            noeq, sr_noeq = load_audio_mono(parsed.noeq[song])
            #noeq = noeq[REFERENCE_DELAY_SAMPLES:]
            if sr_ref != sr_noeq:
                raise RuntimeError(
                    f"Sample-rate mismatch for song {song}: desired={sr_ref}, noEQ={sr_noeq}"
                )

            desired_delayed = apply_sample_delay(desired, REFERENCE_DELAY_SAMPLES)
            desired_noeq, noeq_noisy = add_shared_dither(
                desired_delayed,
                noeq,
            )

            for metric_name, metric_fn in METRICS.items():
                noeq_metrics[metric_name][song] = compute_windowed_metric(
                    reference=desired_noeq,
                    degraded=noeq_noisy,
                    sr=sr_ref,
                    window_seconds=WINDOW_SECONDS,
                    metric_fn=metric_fn,
                )
                if pbar is not None:
                    pbar.update(1)

            for cond in all_conditions:
                if cond.optimizer == "No EQ":
                    for metric_name in METRICS:
                        eq_metrics[metric_name][(cond, song)] = noeq_metrics[metric_name][song]
                        if pbar is not None:
                            pbar.update(1)
                    continue

                eq_path = parsed.eq.get((song, cond), None)
                if eq_path is None:
                    continue

                eq_audio, sr_eq = load_audio_mono(eq_path)
                #eq_audio = eq_audio[REFERENCE_DELAY_SAMPLES:]
                if sr_eq != sr_ref:
                    raise RuntimeError(
                        f"Sample-rate mismatch for song {song}, condition {cond}: "
                        f"desired={sr_ref}, EQ={sr_eq}"
                    )

                desired_eq, eq_noisy = add_shared_dither(
                    desired_delayed,
                    eq_audio,
                )

                for metric_name, metric_fn in METRICS.items():
                    eq_metrics[metric_name][(cond, song)] = compute_windowed_metric(
                        reference=desired_eq,
                        degraded=eq_noisy,
                        sr=sr_ref,
                        window_seconds=WINDOW_SECONDS,
                        metric_fn=metric_fn,
                    )
                    if pbar is not None:
                        pbar.update(1)

    return {
        "songs": songs,
        "conditions": all_conditions,
        "noeq_metrics": noeq_metrics,
        "eq_metrics": eq_metrics,
    }


def aggregate_for_plot(results: dict) -> dict:
    songs: list[str] = results["songs"]
    conditions: list[EqCondition] = results["conditions"]
    noeq_metrics: dict[str, dict[str, np.ndarray]] = results["noeq_metrics"]
    eq_metrics: dict[str, dict[tuple[EqCondition, str], np.ndarray]] = results["eq_metrics"]

    unique_losses = sorted({c.loss for c in conditions})
    unique_tt = sorted({c.transition_token for c in conditions}, key=lambda t: parse_transition_seconds(t))
    unique_optimizers = sorted({c.optimizer for c in conditions})

    aggregate: dict = {
        "songs": songs,
        "losses": unique_losses,
        "transition_tokens": unique_tt,
        "optimizers": unique_optimizers,
        "metrics": {},
    }

    for metric_name in METRICS:
        metric_payload = {
            "noeq": {"mean": None, "std": None, "time_s": None},
            "eq": {},
        }

        noeq_stack = align_and_stack([noeq_metrics[metric_name][song] for song in songs if song in noeq_metrics[metric_name]])
        if noeq_stack.size > 0:
            noeq_mean = np.mean(noeq_stack, axis=0)
            noeq_std = np.std(noeq_stack, axis=0)
            noeq_time = np.arange(noeq_mean.shape[0], dtype=float) * WINDOW_SECONDS
            metric_payload["noeq"] = {
                "mean": noeq_mean,
                "std": noeq_std,
                "time_s": noeq_time,
            }

        for cond in conditions:
            if cond.optimizer == "No EQ":
                series = [
                    noeq_metrics[metric_name][song]
                    for song in songs
                    if song in noeq_metrics[metric_name]
                ]
            else:
                series = [
                    eq_metrics[metric_name][(cond, song)]
                    for song in songs
                    if (cond, song) in eq_metrics[metric_name]
                ]
            stack = align_and_stack(series)
            if stack.size == 0:
                continue
            mean_curve = np.mean(stack, axis=0)
            std_curve = np.std(stack, axis=0)
            time_axis = np.arange(mean_curve.shape[0], dtype=float) * WINDOW_SECONDS
            metric_payload["eq"][cond] = {
                "mean": mean_curve,
                "std": std_curve,
                "time_s": time_axis,
                "n_songs": stack.shape[0],
            }

        aggregate["metrics"][metric_name] = metric_payload

    return aggregate


def print_metric_summary_tables(results: dict) -> None:
    """Print one table per loss with mean/std across all evaluated windows.

    Rows: (transition_time, optimizer)
    Cols: metrics formatted as "mean +- std"
    """
    songs: list[str] = results["songs"]
    conditions: list[EqCondition] = results["conditions"]
    noeq_metrics: dict[str, dict[str, np.ndarray]] = results["noeq_metrics"]
    eq_metrics: dict[str, dict[tuple[EqCondition, str], np.ndarray]] = results["eq_metrics"]

    unique_losses = sorted({c.loss for c in conditions})
    metric_names = list(METRICS.keys())

    print("\nMetric summary tables (mean +- std across all windows):")

    for loss in unique_losses:
        loss_conditions = [c for c in conditions if c.loss == loss]
        loss_conditions = sorted(loss_conditions, key=lambda c: (c.transition_seconds, c.optimizer))

        # Build rows first so we can compute column widths for readable printing.
        rows = []
        for cond in loss_conditions:
            row = {
                "Transition[s]": f"{cond.transition_seconds:.3g}",
                "Optimizer": cond.optimizer,
            }

            for metric_name in metric_names:
                if cond.optimizer == "No EQ":
                    series_list = [
                        noeq_metrics[metric_name][song]
                        for song in songs
                        if song in noeq_metrics[metric_name]
                    ]
                else:
                    series_list = [
                        eq_metrics[metric_name][(cond, song)]
                        for song in songs
                        if (cond, song) in eq_metrics[metric_name]
                    ]

                valid = [s[np.isfinite(s)] for s in series_list if s.size > 0]
                if not valid:
                    row[metric_name] = "n/a"
                    continue

                flat = np.concatenate(valid, axis=0)
                if flat.size == 0:
                    row[metric_name] = "n/a"
                    continue

                m = float(np.mean(flat))
                s = float(np.std(flat))
                row[metric_name] = f"{m:.4g} +- {s:.4g}"

            rows.append(row)

        if not rows:
            continue

        headers = ["Transition[s]", "Optimizer"] + metric_names
        col_widths = {}
        for h in headers:
            col_widths[h] = max(len(h), max(len(r[h]) for r in rows))

        print(f"\nLoss: {loss}")
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        sep_line = "-+-".join("-" * col_widths[h] for h in headers)
        print(header_line)
        print(sep_line)
        for r in rows:
            print(" | ".join(r[h].ljust(col_widths[h]) for h in headers))


def export_metric_summary_tables_csv(results: dict, export_dir: Path) -> None:
    """Export one CSV per loss with mean/std across all evaluated windows.

    Rows: (transition_time, optimizer)
    Cols: metrics (split into <metric>_mean and <metric>_std)
    """
    songs: list[str] = results["songs"]
    conditions: list[EqCondition] = results["conditions"]
    noeq_metrics: dict[str, dict[str, np.ndarray]] = results["noeq_metrics"]
    eq_metrics: dict[str, dict[tuple[EqCondition, str], np.ndarray]] = results["eq_metrics"]

    export_dir.mkdir(parents=True, exist_ok=True)

    unique_losses = sorted({c.loss for c in conditions})
    metric_names = list(METRICS.keys())

    for loss in unique_losses:
        loss_conditions = [c for c in conditions if c.loss == loss]
        loss_conditions = sorted(loss_conditions, key=lambda c: (c.transition_seconds, c.optimizer))
        if not loss_conditions:
            continue

        safe_loss = re.sub(r"[^A-Za-z0-9._-]+", "_", loss)
        out_csv = export_dir / f"metric_summary_{safe_loss}.csv"

        headers = ["transition_s", "transition_token", "optimizer", "loss"]
        for metric_name in metric_names:
            metric_slug = re.sub(r"[^A-Za-z0-9]+", "_", metric_name).strip("_").lower()
            headers.extend([f"{metric_slug}_mean", f"{metric_slug}_std"])

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for cond in loss_conditions:
                row = {
                    "transition_s": f"{cond.transition_seconds:.10g}",
                    "transition_token": cond.transition_token,
                    "optimizer": cond.optimizer,
                    "loss": cond.loss,
                }

                for metric_name in metric_names:
                    metric_slug = re.sub(r"[^A-Za-z0-9]+", "_", metric_name).strip("_").lower()
                    mean_key = f"{metric_slug}_mean"
                    std_key = f"{metric_slug}_std"

                    if cond.optimizer == "No EQ":
                        series_list = [
                            noeq_metrics[metric_name][song]
                            for song in songs
                            if song in noeq_metrics[metric_name]
                        ]
                    else:
                        series_list = [
                            eq_metrics[metric_name][(cond, song)]
                            for song in songs
                            if (cond, song) in eq_metrics[metric_name]
                        ]
                    valid = [s[np.isfinite(s)] for s in series_list if s.size > 0]
                    if not valid:
                        row[mean_key] = ""
                        row[std_key] = ""
                        continue

                    flat = np.concatenate(valid, axis=0)
                    if flat.size == 0:
                        row[mean_key] = ""
                        row[std_key] = ""
                        continue

                    row[mean_key] = f"{float(np.mean(flat)):.10g}"
                    row[std_key] = f"{float(np.std(flat)):.10g}"

                writer.writerow(row)

        print(f"Exported CSV table: {out_csv}")


def plot_aggregated_metrics(aggregate: dict) -> None:
    losses: list[str] = aggregate["losses"]
    transition_tokens: list[str] = aggregate["transition_tokens"]
    optimizers: list[str] = aggregate["optimizers"]
    metrics_payload: dict = aggregate["metrics"]
    metric_names = list(metrics_payload.keys())

    color_map = plt.get_cmap("tab10")
    optimizer_colors = {opt: color_map(i % 10) for i, opt in enumerate(optimizers)}

    n_rows = max(1, len(transition_tokens))
    n_cols = max(1, len(metric_names))

    for loss in losses:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.4 * n_cols, 2.8 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=False,
        )

        for r, tt in enumerate(transition_tokens):
            tt_seconds = parse_transition_seconds(tt)

            for c, metric_name in enumerate(metric_names):
                payload = metrics_payload[metric_name]
                ax = axes[r, c]
                noeq_mean = payload["noeq"]["mean"]
                noeq_std = payload["noeq"]["std"]
                noeq_time = payload["noeq"]["time_s"]

                if r == 0:
                    ax.set_title(metric_name)

                if noeq_mean is not None and noeq_time is not None:
                    ax.plot(
                        noeq_time,
                        noeq_mean,
                        color="black",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.9,
                        label="noEQ" if (r == 0 and c == 0) else None,
                    )
                    if noeq_std is not None:
                        ax.fill_between(
                            noeq_time,
                            noeq_mean - noeq_std,
                            noeq_mean + noeq_std,
                            color="gray",
                            alpha=0.2,
                        )

                plotted_any = False
                for optimizer in optimizers:
                    if optimizer in ("No EQ", "None"):
                        continue

                    condition_match = [
                        cond
                        for cond in payload["eq"].keys()
                        if cond.optimizer == optimizer
                        and cond.loss == loss
                        and cond.transition_token == tt
                    ]
                    if not condition_match:
                        continue

                    cond = condition_match[0]
                    curve = payload["eq"][cond]
                    t = curve["time_s"]
                    m = curve["mean"]
                    s = curve["std"]

                    ax.plot(
                        t,
                        m,
                        color=optimizer_colors[optimizer],
                        linewidth=1.2,
                        alpha=0.95,
                        label=optimizer if (r == 0 and c == 0) else None,
                    )

                    n_markers = min(MAX_PLOTTED_ERRORBARS, t.shape[0])
                    if n_markers > 0:
                        idx = np.linspace(0, t.shape[0] - 1, num=n_markers, dtype=int)
                        ax.errorbar(
                            t[idx],
                            m[idx],
                            yerr=s[idx],
                            fmt="none",
                            ecolor=optimizer_colors[optimizer],
                            elinewidth=1.0,
                            capsize=2,
                            alpha=0.6,
                        )

                    plotted_any = True

                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
                if r == n_rows - 1:
                    ax.set_xlabel("Time [s]")
                if c == 0:
                    ax.set_ylabel(f"tt={tt_seconds:.1f} s")
                if not plotted_any and (noeq_mean is None):
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, fontsize=7, loc="best")

        mode_suffix = "(white noise)" if MODE == "WHITE_NOISE" else "(all songs avg)"
        fig.suptitle(f"Loss: {loss} {mode_suffix}", y=1.01)
        plt.tight_layout()

    plt.show()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "audio" / "output" / EXPERIMENT_NAME

    if not output_dir.exists():
        raise FileNotFoundError(f"Output folder not found: {output_dir}")

    parsed = parse_output_directory(output_dir)
    results = compute_all_metrics(parsed, output_dir)
    print_metric_summary_tables(results)
    csv_export_dir = root / "results" / EXPERIMENT_NAME / "metric_summary_tables"
    export_metric_summary_tables_csv(results, csv_export_dir)
    aggregate = aggregate_for_plot(results)

    print("\nSummary:")
    print(f"  Conditions found: {len(aggregate['optimizers'])} optimizers, "
          f"{len(aggregate['losses'])} losses, "
          f"{len(aggregate['transition_tokens'])} transition times")
    print(f"  Metrics computed: {len(METRICS)}")
    plot_aggregated_metrics(aggregate)


if __name__ == "__main__":
    main()
