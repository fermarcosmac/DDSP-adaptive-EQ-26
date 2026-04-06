"""
Plot results from the ablation study.

Generates a single figure with three panels per transition-time row:
  A) TD-MSE over time
  B) Relative system distance (validation error) over time
  C) Desired vs. final equalized frequency response

Saved as:
  results/<experiment_name>/<experiment_name>_curves.png

Usage:
  python src/scripts/plot_results_ablation_study.py --experiment ablation_study
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    plot_response_mean_std,
    plot_transition_markers,
)


def plot_results(experiment_name: str) -> None:
    configure_text_rendering()
    cfg, data, results_root = load_results(experiment_name)

    td_mse_curves = data.get("td_mse_curves", {})
    validation_curves = data.get("validation_curves", {})
    final_response_curves = data.get("final_response_curves", {})
    compute_time_stats = data.get("compute_time_stats", {})
    tt_transitions = data.get("tt_transitions", {})
    target_example = data.get("target_response_example", None)
    true_lem_example = data.get("true_lem_response_example", None)
    input_signals = data.get("input_signals", None)
    run_labels = [format_input_label(s) for s in input_signals] if input_signals else None

    if not td_mse_curves and not validation_curves:
        print("No curves found in plot data.")
        return

    all_keys = sorted(set(td_mse_curves.keys()) | set(validation_curves.keys()))
    transition_times = sorted({k[0] for k in all_keys})
    algorithms = sorted({k[1] for k in all_keys})

    # Console compute-time table
    print("\nCompute time per frame [s/frame]: avg [min, max]")
    print("key" + "".join(f"\t{format_algo_label(a)}" for a in algorithms))
    for tt in transition_times:
        row = [f"tt={tt}s"]
        for algo in algorithms:
            stats = compute_time_stats.get((tt, algo), None)
            if stats and int(stats.get("total_frames", 0)) > 0:
                avg = float(stats["total_time_s"]) / float(stats["total_frames"])
                v_min = float(stats.get("min_avg_time_per_frame_s", float("nan")))
                v_max = float(stats.get("max_avg_time_per_frame_s", float("nan")))
                if np.isfinite(v_min) and np.isfinite(v_max):
                    row.append(f"{avg:.6f} [{v_min:.6f}, {v_max:.6f}]")
                else:
                    row.append(f"{avg:.6f}")
            else:
                row.append("nan")
        print("\t".join(row))

    colors = plt.get_cmap("tab10")
    algo_color = {a: colors(i % 10) for i, a in enumerate(algorithms)}

    n_rows = len(transition_times)
    fig = plt.figure(figsize=(5.8, max(5.0, 2.6 * n_rows + 2.8)))
    gs = fig.add_gridspec(n_rows + 1, 2, height_ratios=[1.0] * n_rows + [1.2], hspace=0.48, wspace=0.34)
    axes = np.empty((n_rows, 2), dtype=object)
    for row in range(n_rows):
        axes[row, 0] = fig.add_subplot(gs[row, 0])
        axes[row, 1] = fig.add_subplot(gs[row, 1])
    ax_resp = fig.add_subplot(gs[n_rows, :])

    add_panel_label(axes[0, 0], "A")
    add_panel_label(axes[0, 1], "B")
    add_panel_label(ax_resp, "C")

    for row, tt in enumerate(transition_times):
        ax_td = axes[row, 0]
        ax_val = axes[row, 1]

        for algo in algorithms:
            key = (tt, algo)
            plot_mean_std(
                ax_td,
                td_mse_curves.get(key, []),
                algo_color[algo],
                format_algo_label(algo),
                run_labels=run_labels,
                report_context=f"tt={tt}, algo={algo}, metric=TD-MSE",
            )
            plot_mean_std(
                ax_val,
                validation_curves.get(key, []),
                algo_color[algo],
                format_algo_label(algo),
                run_labels=run_labels,
                report_context=f"tt={tt}, algo={algo}, metric=validation",
            )

        for ax in (ax_td, ax_val):
            plot_transition_markers(ax, tt_transitions.get(tt, None))
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
            if row == n_rows - 1:
                ax.set_xlabel(r"$\mathrm{Time\ [s]}$")

        ax_td.set_yscale("log")
        ax_td.set_ylabel(r"$\mathrm{TD\text{-}MSE}$")
        ax_td.set_title(r"$\mathrm{Time\text{-}domain\ MSE}$")

        ax_val.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_val.set_ylabel(r"$D_{\mathrm{rel}}$")
        ax_val.set_title(r"$\mathrm{Relative\ system\ distance}$")
        ax_val.set_ylim(0.0, 10.0)

    # Bottom panel: desired, true LEM, and final equalized response
    if target_example is not None and len(target_example.get("freq_axis", [])):
        f = np.asarray(target_example["freq_axis"], dtype=float)
        tdb = np.asarray(target_example["target_mag_db"], dtype=float)
        m = f > 0
        ax_resp.plot(f[m], tdb[m], color="black", linestyle="-", linewidth=1.3, label=r"$\mathrm{Desired}$")

    if true_lem_example is not None and len(true_lem_example.get("freq_axis", [])):
        f_lem = np.asarray(true_lem_example["freq_axis"], dtype=float)
        lem_db = np.asarray(true_lem_example["lem_mag_db"], dtype=float)
        m_lem = (f_lem > 0) & np.isfinite(lem_db)
        f_s, lem_s = log_smooth_curve(f_lem[m_lem], lem_db[m_lem], window_pts=121)
        ax_resp.plot(f_s, lem_s, color="black", linestyle="--", linewidth=1.1, label=r"$\mathrm{True\ LEM\ (unprocessed)}$")

    if final_response_curves:
        by_algo: dict = {a: [] for a in algorithms}
        for (tt, algo), series in final_response_curves.items():
            if algo not in by_algo:
                by_algo[algo] = []
            by_algo[algo].extend(series)
        for algo in sorted(by_algo.keys()):
            plot_response_mean_std(
                ax_resp,
                by_algo[algo],
                algo_color.get(algo, "C0"),
                format_algo_label(algo),
                run_labels=run_labels,
                report_context=f"algo={algo}, metric=final_response",
            )

    ax_resp.set_xscale("log")
    ax_resp.set_xlim(20, 20000)
    ax_resp.set_xlabel(r"$\mathrm{Frequency\ [Hz]}$")
    ax_resp.set_ylim(-40, 20)
    ax_resp.set_ylabel(r"$\mathrm{Magnitude\ [dB]}$")
    ax_resp.set_title(r"$\mathrm{Desired\ vs.\ Final\ Equalized\ Response}$")
    ax_resp.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    handles, labels = ax_resp.get_legend_handles_labels()
    if handles:
        ax_resp.legend(handles, labels, loc="lower center", ncol=min(2, len(handles)), fontsize=7.5, frameon=True)

    fig.tight_layout(h_pad=0.9, w_pad=0.9, rect=(0.08, 0.02, 0.98, 0.98))
    out_png = results_root / f"{experiment_name}_curves.png"
    fig.savefig(out_png, dpi=180)
    print(f"Saved figure: {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ablation study results.")
    parser.add_argument(
        "--experiment",
        default="ablation_study",
        help="Experiment name (subdirectory of results/).",
    )
    args = parser.parse_args()
    plot_results(args.experiment)
