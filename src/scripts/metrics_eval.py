"""
Metrics evaluation script.

Loads EQ-processed WAV files from a completed experiment and computes
per-file audio quality metrics against the desired (reference) output.

Results are printed to stdout and saved as results/<experiment_name>/metrics.csv.

Usage:
  python src/scripts/metrics_eval.py --experiment main_experiment
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "src" / "external"))

from utils.metrics import (
    add_shared_dither,
    apply_sample_delay,
    evaluate_pair,
    format_transition_label,
    load_experiment_metadata,
    load_mono_float,
    parse_optimizer_from_stem,
    parse_transition_time_from_stem,
    summarize_rows,
)

# ---------------------------------------------------------------------------
# Configuration (edit these globals or pass CLI arguments)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "main_experiment"
METRICS = ["PEAQ", "MRSTFT", "MSD", "Spectral-Centroid", "RMSE", "SI-SDR", "LUFS-diff"]
EVAL_LAST_S: float = 0.0                     # if > 0, evaluate only the last N seconds
REFERENCE_DELAY_SAMPLES = 300                # apply delay to reference before scoring
DITHER_STD = 1e-8                            # shared noise floor to stabilize sensitive metrics
DITHER_SEED = 12345
# ---------------------------------------------------------------------------


def evaluate_experiment(experiment_name: str, metrics: list[str], eval_last_s: float) -> None:
    results_root = root / "results" / experiment_name
    audio_dir = results_root / "audio"
    known_optimizers, known_transition_times = load_experiment_metadata(results_root)

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
        optimizer = parse_optimizer_from_stem(parts, known_optimizers)
        transition_s = parse_transition_time_from_stem(parts)

        # The song name is everything after the last "tt<N>" token.
        song_name = None
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].startswith("tt"):
                # Everything after the transition-time token is the song stem.
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
            eq_audio, sr = load_mono_float(eq_path, sr_target=0)
            ref_audio, _ = load_mono_float(desired_path, sr_target=sr)
        except Exception as exc:
            print(f"  [error] Loading {eq_path.name}: {exc}")
            continue

        ref_audio = apply_sample_delay(ref_audio, REFERENCE_DELAY_SAMPLES)

        if eval_last_s > 0:
            n_keep = int(eval_last_s * sr)
            eq_audio = eq_audio[-n_keep:]
            ref_audio = ref_audio[-n_keep:]

        ref_audio, eq_audio = add_shared_dither(
            ref_audio,
            eq_audio,
            std=DITHER_STD,
            seed=DITHER_SEED,
        )

        scores = evaluate_pair(ref_audio, eq_audio, sr, metrics=metrics)
        row = {
            "file": eq_path.name,
            "song": song_name,
            "optimizer": optimizer,
            "transition_s": transition_s,
            "transition_label": format_transition_label(transition_s) if transition_s is not None else "NA",
            **scores,
        }
        all_rows.append(row)

        score_str = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"  {eq_path.name:<60s}  {score_str}")

    if not all_rows:
        print("No results computed.")
        return

    summary_rows = summarize_rows(all_rows, metrics, known_optimizers, known_transition_times)
    if not summary_rows:
        print("No grouped summary rows were computed.")
        return

    # Save grouped summary CSV at results/<experiment_name>/metrics.csv
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "metrics.csv"
    fieldnames = ["optimizer", "transition_s", "transition_label", "n_files"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved grouped metrics CSV: {csv_path}")

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
