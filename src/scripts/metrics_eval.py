"""
Metrics evaluation script.

Loads EQ-processed WAV files from a completed experiment and computes
per-file audio quality metrics against the desired (reference) output.

Results are printed to stdout and saved as a CSV.

Usage:
  python src/scripts/metrics_eval.py --experiment main_experiment
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torchaudio

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "src" / "external"))

from utils.metrics import evaluate_pair

# ---------------------------------------------------------------------------
# Configuration (edit these globals or pass CLI arguments)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "main_experiment"
METRICS = ["RMSE", "SI-SDR", "LUFS-diff"]   # subset of available metrics
EVAL_LAST_S: float = 0.0                     # if > 0, evaluate only the last N seconds
# ---------------------------------------------------------------------------


def _load_mono_float(path: Path, sr_target: int) -> tuple[np.ndarray, int]:
    """Load a WAV file, resample if needed, return (mono float64 array, sr)."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != sr_target and sr_target > 0:
        import torchaudio.transforms as T
        wav = T.Resample(orig_freq=int(sr), new_freq=sr_target)(wav)
        sr = sr_target
    return wav.squeeze(0).numpy().astype(np.float64), int(sr)


def evaluate_experiment(experiment_name: str, metrics: list[str], eval_last_s: float) -> None:
    audio_dir = root / "data" / "audio" / "output" / experiment_name
    results_root = root / "results" / experiment_name

    if not audio_dir.exists():
        print(f"Audio output directory not found: {audio_dir}")
        return

    # Discover all EQ output files (named EQ_*.wav)
    eq_files = sorted(audio_dir.glob("EQ_*.wav"))
    if not eq_files:
        print(f"No EQ_*.wav files found in: {audio_dir}")
        return

    all_rows: list[dict] = []

    for eq_path in eq_files:
        # Derive the song stem from the EQ filename.
        # Filename format: EQ_<optim>_<lt>_fl<fl>_tt<tt>_<song>.wav
        # or:              EQ_<algo>_NA_tt<tt>_<song>.wav
        stem = eq_path.stem  # e.g. EQ_SGD_FD_MSE_fl8192_tt1p0_AimeeNorwich
        parts = stem.split("_")
        # The song name is everything after the last "tt<N>" token.
        song_name = None
        for i, part in enumerate(parts):
            if part.startswith("tt") or part.startswith("fl"):
                # Everything from i+1 onward is the song stem
                if i + 1 < len(parts):
                    song_name = "_".join(parts[i + 1 :])
                break

        if song_name is None:
            # Fallback: use the last token
            song_name = parts[-1]

        desired_path = audio_dir / f"desired_{song_name}.wav"
        if not desired_path.exists():
            print(f"  [skip] No desired file for song '{song_name}': {desired_path}")
            continue

        try:
            eq_audio, sr = _load_mono_float(eq_path, sr_target=0)
            ref_audio, _ = _load_mono_float(desired_path, sr_target=sr)
        except Exception as exc:
            print(f"  [error] Loading {eq_path.name}: {exc}")
            continue

        if eval_last_s > 0:
            n_keep = int(eval_last_s * sr)
            eq_audio = eq_audio[-n_keep:]
            ref_audio = ref_audio[-n_keep:]

        scores = evaluate_pair(ref_audio, eq_audio, sr, metrics=metrics)
        row = {"file": eq_path.name, "song": song_name, **scores}
        all_rows.append(row)

        score_str = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"  {eq_path.name:<60s}  {score_str}")

    if not all_rows:
        print("No results computed.")
        return

    # Save CSV
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "metrics.csv"
    fieldnames = ["file", "song"] + metrics
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved metrics CSV: {csv_path}")

    # Print summary statistics
    print("\nSummary (mean ± std):")
    for m in metrics:
        vals = np.asarray([r[m] for r in all_rows if np.isfinite(r.get(m, float("nan")))], dtype=float)
        if vals.size:
            print(f"  {m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio quality metrics for a completed experiment.")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="Experiment name.")
    parser.add_argument("--metrics", nargs="+", default=METRICS, help="Metrics to compute.")
    parser.add_argument("--eval-last-s", type=float, default=EVAL_LAST_S,
                        help="If > 0, evaluate only the last N seconds of each file.")
    args = parser.parse_args()
    evaluate_experiment(args.experiment, args.metrics, args.eval_last_s)
