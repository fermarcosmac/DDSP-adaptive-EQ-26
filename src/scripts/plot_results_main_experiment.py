"""
Plot results from the main experiment.

Generates:
  results/<experiment_name>/<experiment_name>_curves.png          — validation curves grid
  results/<experiment_name>/<experiment_name>_compute_time.png    — compute-time table

Usage:
  python src/scripts/plot_results_main_experiment.py --experiment main_experiment
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "src" / "external"))

from utils.plotting import (
    add_panel_label,
    configure_text_rendering,
    format_algo_label,
    format_input_label,
    load_results,
    log_smooth_curve,
    plot_mean_std,
    plot_transition_markers,
    select_series_for_averaging,
)


# ---------------------------------------------------------------------------
# Key normalisation helpers (main experiment uses 4-tuple keys)
# ---------------------------------------------------------------------------

def _parse_curve_key(key):
    """Parse a curve key into (tt, frame_len, optim, loss_type)."""
    if not isinstance(key, tuple):
        raise ValueError(f"Invalid key type: {type(key)}")
    if len(key) == 3:
        tt, optim, lt = key
        return tt, None, optim, lt
    if len(key) == 4:
        tt, frame_len, optim, lt = key
        return tt, frame_len, optim, lt
    raise ValueError(f"Unsupported curve key: {key}")


def _normalize_curve_dict(curves_raw: dict) -> defaultdict:
    out = defaultdict(list)
    for key, series in curves_raw.items():
        try:
            tt, fl, optim, lt = _parse_curve_key(key)
        except ValueError:
            continue
        out[(tt, fl, optim, lt)].extend(series)
    return out


def _group_curves_ignore_frame(curves_norm: defaultdict) -> defaultdict:
    grouped = defaultdict(list)
    for (tt, _fl, optim, lt), series in curves_norm.items():
        grouped[(tt, optim, lt)].extend(series)
    return grouped


def _normalize_compute_time_stats(raw: dict) -> defaultdict:
    out = defaultdict(
        lambda: {
            "total_time_s": 0.0,
            "total_frames": 0,
            "num_runs": 0,
            "min_avg_time_per_frame_s": float("inf"),
            "max_avg_time_per_frame_s": float("-inf"),
        }
    )
    for key, stats in raw.items():
        if not isinstance(key, tuple):
            continue
        if len(key) == 2:
            tt, optim = key
            fl = None
        elif len(key) == 3:
            tt, fl, optim = key
        else:
            continue
        nk = (tt, fl, optim)
        out[nk]["total_time_s"] += float(stats.get("total_time_s", 0.0))
        out[nk]["total_frames"] += int(stats.get("total_frames", 0))
        out[nk]["num_runs"] += int(stats.get("num_runs", 0))
        out[nk]["min_avg_time_per_frame_s"] = min(
            float(out[nk]["min_avg_time_per_frame_s"]),
            float(stats.get("min_avg_time_per_frame_s", float("inf"))),
        )
        out[nk]["max_avg_time_per_frame_s"] = max(
            float(out[nk]["max_avg_time_per_frame_s"]),
            float(stats.get("max_avg_time_per_frame_s", float("-inf"))),
        )
    return out


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_results(experiment_name: str) -> None:
    configure_text_rendering()
    cfg, data, results_root = load_results(experiment_name)

    curves_raw = data.get("curves", {})
    loss_curves_raw = data.get("loss_curves", {})
    tt_transitions_raw = data.get("tt_transitions", {})
    input_signals = data.get("input_signals", None)
    run_labels = [format_input_label(s) for s in input_signals] if input_signals else None

    curves_norm = _normalize_curve_dict(curves_raw)
    loss_curves_norm = _normalize_curve_dict(loss_curves_raw)
    curves_grouped = _group_curves_ignore_frame(curves_norm)
    loss_curves_grouped = _group_curves_ignore_frame(loss_curves_norm)
    ct_norm = _normalize_compute_time_stats(data.get("compute_time_stats", {}))

    if not curves_grouped:
        print("No validation curves found in plot data.")
        return

    all_keys = list(curves_grouped.keys())
    transition_times = sorted({k[0] for k in all_keys})
    optim_types = sorted({k[1] for k in all_keys})
    loss_types = sorted({k[2] for k in all_keys})

    color_map = plt.get_cmap("tab10")
    optim_color = {opt: color_map(i % 10) for i, opt in enumerate(optim_types)}
    lt_linestyle = {lt: ls for lt, ls in zip(loss_types, ["-", "--", "-.", ":"])}

    # -----------------------------------------------------------------------
    # Compute-time table (console)
    # -----------------------------------------------------------------------
    print("\nCompute time per frame [s/frame]: avg [min, max]")
    for tt in transition_times:
        row = [f"tt={tt}s"]
        for opt in optim_types:
            matches = [v for (t, _fl, o), v in ct_norm.items() if t == tt and o == opt]
            if matches:
                total_time_s = float(sum(float(m.get("total_time_s", 0.0)) for m in matches))
                total_frames = int(sum(int(m.get("total_frames", 0)) for m in matches))
                min_pf = float(min(float(m.get("min_avg_time_per_frame_s", float("inf"))) for m in matches))
                max_pf = float(max(float(m.get("max_avg_time_per_frame_s", float("-inf"))) for m in matches))
            else:
                total_time_s = 0.0
                total_frames = 0
                min_pf = float("nan")
                max_pf = float("nan")

            if total_frames > 0:
                avg = total_time_s / float(total_frames)
                row.append(f"{avg:.6f} [{min_pf:.6f}, {max_pf:.6f}]")
            else:
                row.append("nan [nan, nan]")
        print("  ".join(row))

    # -----------------------------------------------------------------------
    # Figure: validation curves
    # -----------------------------------------------------------------------
    n_rows = len(transition_times)
    n_cols = len(loss_types)
    fig, axes = plt.subplots(
        n_rows, max(n_cols, 1),
        figsize=(4.5 * max(n_cols, 1), 3.5 * n_rows),
        squeeze=False,
    )

    for row, tt in enumerate(transition_times):
        for col, lt in enumerate(loss_types):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(lt)

            for opt in optim_types:
                key = (tt, opt, lt)
                if key not in curves_grouped:
                    continue
                plot_mean_std(
                    ax,
                    curves_grouped[key],
                    color=optim_color[opt],
                    label=format_algo_label(opt),
                    linestyle=lt_linestyle.get(lt, "-"),
                    run_labels=run_labels,
                    report_context=f"tt={tt}, opt={opt}, lt={lt}",
                )

            plot_transition_markers(ax, tt_transitions_raw.get(tt, None))
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_ylabel(r"$D_{\mathrm{rel}}$")
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
            if row == n_rows - 1:
                ax.set_xlabel("Time [s]")

            if row == 0 and col == 0:
                add_panel_label(ax, "A")

        if n_cols > 1:
            axes[row, -1].legend(loc="upper right", fontsize=8)
        else:
            axes[row, 0].legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{experiment_name} — Validation error")
    fig.tight_layout()
    out_png = results_root / f"{experiment_name}_curves.png"
    fig.savefig(out_png, dpi=180)
    print(f"Saved figure: {out_png}")

    # -----------------------------------------------------------------------
    # Figure: loss curves (separate)
    # -----------------------------------------------------------------------
    if loss_curves_grouped:
        fig_loss, axes_loss = plt.subplots(
            n_rows, max(n_cols, 1),
            figsize=(4.5 * max(n_cols, 1), 3.5 * n_rows),
            squeeze=False,
        )

        for row, tt in enumerate(transition_times):
            for col, lt in enumerate(loss_types):
                ax = axes_loss[row, col]
                if row == 0:
                    ax.set_title(lt)

                for opt in optim_types:
                    key = (tt, opt, lt)
                    if key not in loss_curves_grouped:
                        continue
                    plot_mean_std(
                        ax,
                        loss_curves_grouped[key],
                        color=optim_color[opt],
                        label=format_algo_label(opt),
                        linestyle=lt_linestyle.get(lt, "-"),
                        run_labels=run_labels,
                        report_context=f"loss tt={tt}, opt={opt}, lt={lt}",
                    )

                plot_transition_markers(ax, tt_transitions_raw.get(tt, None))
                ax.set_yscale("log")
                ax.set_ylabel("Loss")
                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
                if row == n_rows - 1:
                    ax.set_xlabel("Time [s]")

                if row == 0 and col == 0:
                    add_panel_label(ax, "B")

            if n_cols > 1:
                axes_loss[row, -1].legend(loc="upper right", fontsize=8)
            else:
                axes_loss[row, 0].legend(loc="upper right", fontsize=8)

        fig_loss.suptitle(f"{experiment_name} — Loss")
        fig_loss.tight_layout()
        out_loss_png = results_root / f"{experiment_name}_loss_curves.png"
        fig_loss.savefig(out_loss_png, dpi=180)
        print(f"Saved figure: {out_loss_png}")
    else:
        print("No loss curves found in plot data; skipping loss figure.")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot main experiment results.")
    parser.add_argument(
        "--experiment",
        default="main_experiment",
        help="Experiment name (subdirectory of results/).",
    )
    args = parser.parse_args()
    plot_results(args.experiment)
