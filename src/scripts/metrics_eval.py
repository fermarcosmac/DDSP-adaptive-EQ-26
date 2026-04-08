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
import json
import sys
from collections import defaultdict
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


def _safe_token(text: str) -> str:
    return str(text).replace("-", "_").replace(" ", "_")


def _format_transition_label(transition_s: float) -> str:
    if float(transition_s).is_integer():
        return f"{int(transition_s)} s"
    return f"{transition_s:g} s"


def _load_experiment_metadata(results_root: Path) -> tuple[list[str], list[float]]:
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


def _parse_optimizer_from_stem(parts: list[str], known_optimizers: list[str]) -> str:
    post_eq_parts = parts[1:]
    candidate_tokens = sorted(
        [(_safe_token(opt).split("_"), opt) for opt in known_optimizers],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for opt_tokens, opt_display in candidate_tokens:
        if post_eq_parts[: len(opt_tokens)] == opt_tokens:
            return opt_display

    if post_eq_parts:
        return post_eq_parts[0]
    return "unknown"


def _parse_transition_time_from_stem(parts: list[str]) -> float | None:
    for token in reversed(parts):
        if token.startswith("tt"):
            raw = token[2:].replace("p", ".")
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def _summarize_rows(all_rows: list[dict], metrics: list[str], known_optimizers: list[str], known_transition_times: list[float]) -> list[dict]:
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
            "transition_label": _format_transition_label(transition_s),
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


def _build_metric_report(summary_rows: list[dict], metrics: list[str], known_optimizers: list[str], known_transition_times: list[float]) -> str:
    lookup = {(row["optimizer"], float(row["transition_s"])): row for row in summary_rows}
    optimizer_order = {opt: idx for idx, opt in enumerate(known_optimizers)}
    transition_order = {tt: idx for idx, tt in enumerate(known_transition_times)}

    sorted_optimizers = sorted(
        {row["optimizer"] for row in summary_rows},
        key=lambda opt: (optimizer_order.get(opt, len(optimizer_order)), opt),
    )
    sorted_transitions = sorted(
        {float(row["transition_s"]) for row in summary_rows},
        key=lambda tt: (transition_order.get(tt, len(transition_order)), tt),
    )

    lines: list[str] = ["# Metrics Summary"]
    for metric in metrics:
        lines.append("")
        lines.append(f"## {metric}")
        header = ["Optimizer"] + [_format_transition_label(tt) for tt in sorted_transitions]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for optimizer in sorted_optimizers:
            row_cells = [optimizer]
            for transition_s in sorted_transitions:
                summary_row = lookup.get((optimizer, transition_s))
                if summary_row is None:
                    row_cells.append("NA")
                    continue
                mean = float(summary_row.get(f"{metric}_mean", float("nan")))
                std = float(summary_row.get(f"{metric}_std", float("nan")))
                if np.isfinite(mean) and np.isfinite(std):
                    row_cells.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row_cells.append("NA")
            lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


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
    results_root = root / "results" / experiment_name
    audio_dir = results_root / "audio"
    known_optimizers, known_transition_times = _load_experiment_metadata(results_root)

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
        optimizer = _parse_optimizer_from_stem(parts, known_optimizers)
        transition_s = _parse_transition_time_from_stem(parts)

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
        row = {
            "file": eq_path.name,
            "song": song_name,
            "optimizer": optimizer,
            "transition_s": transition_s,
            "transition_label": _format_transition_label(transition_s) if transition_s is not None else "NA",
            **scores,
        }
        all_rows.append(row)

        score_str = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"  {eq_path.name:<60s}  {score_str}")

    if not all_rows:
        print("No results computed.")
        return

    # Save CSV
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "metrics.csv"
    fieldnames = ["file", "song", "optimizer", "transition_s", "transition_label"] + metrics
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved metrics CSV: {csv_path}")

    summary_rows = _summarize_rows(all_rows, metrics, known_optimizers, known_transition_times)
    if summary_rows:
        summary_dir = results_root / "metric_summary_tables"
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary_csv_path = summary_dir / "metric_summary_by_optimizer_transition.csv"
        summary_fieldnames = ["optimizer", "transition_s", "transition_label", "n_files"]
        for metric in metrics:
            summary_fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        summary_md_path = summary_dir / "metric_summary_by_optimizer_transition.md"
        summary_md = _build_metric_report(summary_rows, metrics, known_optimizers, known_transition_times)
        summary_md_path.write_text(summary_md, encoding="utf-8")

        print(f"Saved grouped summary CSV: {summary_csv_path}")
        print(f"Saved grouped summary report: {summary_md_path}")
        print(summary_md)

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
