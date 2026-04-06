from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent.parent


def _configure_text_rendering() -> None:
    """Use Computer Modern mathtext for a LaTeX-like look without requiring TeX."""
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def _log_smooth_curve(freq_hz, mag_db, window_pts: int = 61):
    """Return a moving-average smoothed curve over a log-frequency grid."""
    f = np.asarray(freq_hz, dtype=float)
    y = np.asarray(mag_db, dtype=float)
    m = np.isfinite(f) & np.isfinite(y) & (f > 0)
    f = f[m]
    y = y[m]
    if f.size < 3:
        return f, y

    n_log = max(256, f.size)
    f_log = np.logspace(np.log10(f[0]), np.log10(f[-1]), n_log)
    y_log = np.interp(f_log, f, y)

    w = max(5, int(window_pts))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    y_smooth = np.convolve(y_log, kernel, mode="same")
    return f_log, y_smooth


def load_results(experiment_name: str) -> tuple[dict, dict, Path]:
    results_root = root / "results" / experiment_name
    cfg_path = results_root / "config.json"
    pkl_path = results_root / "plot1_data.pkl"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing plot data file: {pkl_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    with pkl_path.open("rb") as f:
        plot_data = pickle.load(f)
    return cfg, plot_data, results_root


def _display_algo_label(algo: str) -> str:
    return str(algo).replace("GHAM", "iHAM")


def _format_input_label(input_signal) -> str:
    s = str(input_signal)
    if s.startswith("white_noise"):
        return s
    p = Path(s)
    if p.suffix:
        return p.stem
    return s


def _add_panel_label(ax, label: str) -> None:
    ax.text(
        0.95,
        0.90,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0, "boxstyle": "square,pad=0.3"},
        zorder=5,
    )


def _select_series_for_averaging(
    series,
    n_remove_highest_mean_curves: int = 0,
    run_labels=None,
    report_context: str = None,
):
    """Optionally remove n runs with the highest time-mean value."""
    if not series:
        return series

    n_remove = max(0, int(n_remove_highest_mean_curves))
    if n_remove == 0:
        return series

    min_len = min(len(v[1]) for v in series)
    if min_len == 0 or len(series) <= 1:
        return series

    # Keep at least one run to avoid empty aggregation.
    n_remove = min(n_remove, len(series) - 1)
    if n_remove == 0:
        return series

    mean_per_run = []
    for _, vals in series:
        run_vals = np.asarray(vals[:min_len], dtype=float)
        mean_per_run.append(np.nanmean(run_vals))

    means = np.asarray(mean_per_run, dtype=float)
    idx_sorted_desc = np.argsort(-means)
    idx_remove_list = [int(i) for i in idx_sorted_desc[:n_remove]]
    idx_remove = set(idx_remove_list)

    if report_context:
        removed_labels = []
        for i in idx_remove_list:
            if run_labels is not None and i < len(run_labels):
                removed_labels.append(str(run_labels[i]))
            else:
                removed_labels.append(f"run_{i}")
        if removed_labels:
            print(f"[curve-filter] {report_context} -> removed: {', '.join(removed_labels)}")

    return [s for i, s in enumerate(series) if i not in idx_remove]


def _plot_mean_std(
    ax,
    series,
    color,
    label,
    n_remove_highest_mean_curves: int = 0,
    run_labels=None,
    report_context: str = None,
):
    series = _select_series_for_averaging(
        series,
        n_remove_highest_mean_curves,
        run_labels=run_labels,
        report_context=report_context,
    )
    if not series:
        return
    min_len = min(len(vals) for _, vals in series)
    t = None
    stack = []
    for ta, vals in series:
        ta = np.asarray(ta, dtype=float)
        vals = np.asarray(vals, dtype=float)
        if vals.size < min_len or np.isnan(vals).any():
            continue
        stack.append(vals[:min_len])
        if t is None:
            t = ta[:min_len]
    if not stack or t is None:
        return

    y = np.stack(stack, axis=0)
    avg = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    ax.plot(t, avg, color=color, linewidth=1.1, label=label)
    n_mark = min(10, len(t))
    idx = np.linspace(0, len(t) - 1, num=n_mark, dtype=int)
    ax.errorbar(t[idx], avg[idx], yerr=std[idx], fmt="none", ecolor=color, elinewidth=0.9, capsize=3, alpha=0.75)


def _plot_response_mean_std(
    ax,
    series,
    color,
    label,
    n_remove_highest_mean_curves: int = 0,
    run_labels=None,
    report_context: str = None,
):
    series = _select_series_for_averaging(
        series,
        n_remove_highest_mean_curves,
        run_labels=run_labels,
        report_context=report_context,
    )
    if not series:
        return
    ref_f = np.asarray(series[0][0], dtype=float)
    if ref_f.size < 2:
        return

    stack = []
    for f, mag_db in series:
        f = np.asarray(f, dtype=float)
        mag_db = np.asarray(mag_db, dtype=float)
        if f.size < 2 or mag_db.size != f.size or np.isnan(mag_db).any():
            continue
        stack.append(np.interp(ref_f, f, mag_db))

    if not stack:
        return

    y = np.stack(stack, axis=0)
    avg = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    m = ref_f > 0
    f_s, avg_s = _log_smooth_curve(ref_f[m], avg[m], window_pts=121)
    _, std_s = _log_smooth_curve(ref_f[m], std[m], window_pts=121)
    ax.plot(f_s, avg_s, color=color, linewidth=1.25, label=label)
    ax.fill_between(f_s, avg_s - std_s, avg_s + std_s, color=color, alpha=0.16, linewidth=0)


def plot_results(experiment_name: str, n_remove_highest_mean_curves: int = 0) -> None:
    _configure_text_rendering()
    cfg, data, results_root = load_results(experiment_name)

    td_mse_curves = data.get("td_mse_curves", {})
    validation_curves = data.get("validation_curves", {})
    final_response_curves = data.get("final_response_curves", {})
    compute_time_stats = data.get("compute_time_stats", {})
    tt_transitions = data.get("tt_transitions", {})
    target_example = data.get("target_response_example", None)
    true_lem_example = data.get("true_lem_response_example", None)
    input_signals = data.get("input_signals", None)
    run_labels = [_format_input_label(sig) for sig in input_signals] if input_signals else None

    if not td_mse_curves or not validation_curves:
        print("No curves found in plot data.")
        return

    all_keys = sorted(set(td_mse_curves.keys()) | set(validation_curves.keys()))
    transition_times = sorted({k[0] for k in all_keys})
    algorithms = sorted({k[1] for k in all_keys})

    # Console compute-time table
    print("Compute time per frame [s/frame]: avg [min, max]")
    header = "transition_time_s" + "".join(f"\t{_display_algo_label(a)}" for a in algorithms)
    print(header)
    avg_table = np.full((len(transition_times), len(algorithms)), np.nan)
    min_table = np.full_like(avg_table, np.nan)
    max_table = np.full_like(avg_table, np.nan)
    for i, tt in enumerate(transition_times):
        row = [f"{tt}"]
        for j, algo in enumerate(algorithms):
            stats = compute_time_stats.get((tt, algo), None)
            if stats and int(stats.get("total_frames", 0)) > 0:
                v_avg = float(stats["total_time_s"]) / float(stats["total_frames"])
                v_min = float(stats.get("min_avg_time_per_frame_s", np.nan))
                v_max = float(stats.get("max_avg_time_per_frame_s", np.nan))
                avg_table[i, j] = v_avg
                min_table[i, j] = v_min
                max_table[i, j] = v_max
                if np.isfinite(v_min) and np.isfinite(v_max):
                    row.append(f"{v_avg:.6f} [{v_min:.6f}, {v_max:.6f}]")
                else:
                    row.append(f"{v_avg:.6f}")
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

    # Panel labels
    _add_panel_label(axes[0, 0], "A")
    _add_panel_label(axes[0, 1], "B")
    _add_panel_label(ax_resp, "C")

    for row, tt in enumerate(transition_times):
        ax_td = axes[row, 0]
        ax_val = axes[row, 1]

        for algo in algorithms:
            key = (tt, algo)
            _plot_mean_std(
                ax_td,
                td_mse_curves.get(key, []),
                algo_color[algo],
                _display_algo_label(algo),
                n_remove_highest_mean_curves=n_remove_highest_mean_curves,
                run_labels=run_labels,
                report_context=f"tt={tt}, algo={algo}, metric=TD-MSE",
            )
            _plot_mean_std(
                ax_val,
                validation_curves.get(key, []),
                algo_color[algo],
                _display_algo_label(algo),
                n_remove_highest_mean_curves=n_remove_highest_mean_curves,
                run_labels=run_labels,
                report_context=f"tt={tt}, algo={algo}, metric=validation",
            )

        for ax in (ax_td, ax_val):
            trans = tt_transitions.get(tt, None)
            if trans is not None:
                for t_start, t_end in trans:
                    ax.axvline(float(t_start), color="0.2", linestyle="--", linewidth=1.0, alpha=0.8)
                    if t_end != t_start:
                        ax.axvline(float(t_end), color="0.2", linestyle="--", linewidth=1.0, alpha=0.8)
                        ax.axvspan(float(t_start), float(t_end), color="0.85", alpha=0.35)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
            if row == n_rows - 1:
                ax.set_xlabel(r"$\mathrm{Time\ [s]}$")

        ax_td.set_yscale("log")
        ax_td.set_ylabel(r"$\mathrm{TD\text{-}MSE}$")
        ax_td.set_title(r"$\mathrm{Time\mathrm{-}domain\ MSE}$")

        ax_val.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_val.set_ylabel(r"$D_{\mathrm{rel}}$")
        ax_val.set_title(r"$\mathrm{Relative\ system\ distance}$")
        ax_val.set_ylim(0.0, 10.0)

    # Legend only in bottom subplot (response); skip legends on top subplots.

    # Bottom subplot: desired response, true LEM (unprocessed), and final equalized response.
    if target_example is not None and len(target_example.get("freq_axis", [])):
        f = np.asarray(target_example["freq_axis"], dtype=float)
        tdb = np.asarray(target_example["target_mag_db"], dtype=float)
        m = f > 0
        ax_resp.plot(f[m], tdb[m], color="black", linestyle="-", linewidth=1.3, label=r"$\mathrm{Desired}$")

    if true_lem_example is not None and len(true_lem_example.get("freq_axis", [])):
        f_lem = np.asarray(true_lem_example["freq_axis"], dtype=float)
        lem_db = np.asarray(true_lem_example["lem_mag_db"], dtype=float)
        m_lem = (f_lem > 0) & np.isfinite(lem_db)
        f_lem_s, lem_db_s = _log_smooth_curve(f_lem[m_lem], lem_db[m_lem], window_pts=121)
        ax_resp.plot(f_lem_s, lem_db_s, color="black", linestyle="--", linewidth=1.1, label=r"$\mathrm{True\ LEM\ (unprocessed)}$")
    else:
        print("No true_lem_response_example found in plot data. Re-run experiment_05.py after saving true LEM response to include dashed black reference.")

    if final_response_curves:
        by_algo = {algo: [] for algo in algorithms}
        for (tt, algo), series in final_response_curves.items():
            if algo not in by_algo:
                by_algo[algo] = []
            by_algo[algo].extend(series)
        for algo in sorted(by_algo.keys()):
            _plot_response_mean_std(
                ax_resp,
                by_algo[algo],
                algo_color.get(algo, "C0"),
                _display_algo_label(algo),
                n_remove_highest_mean_curves=n_remove_highest_mean_curves,
                run_labels=run_labels,
                report_context=f"algo={algo}, metric=final_response",
            )
    else:
        print("No final_response_curves found in plot data. Re-run experiment_05.py to populate final equalized response subplot.")

    ax_resp.set_xscale("log")
    ax_resp.set_xlim(20, 20000)
    ax_resp.set_xlabel(r"$\mathrm{Frequency\ [Hz]}$")
    ax_resp.set_ylim(-40, 20)
    ax_resp.set_ylabel(r"$\mathrm{Magnitude\ [dB]}$")
    ax_resp.set_title(r"$\mathrm{Desired\ vs\ Final\ Equalized\ Response}$")
    ax_resp.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    handles_resp, labels_resp = ax_resp.get_legend_handles_labels()
    if handles_resp:
        ax_resp.legend(
            handles_resp,
            labels_resp,
            loc="lower center",
            ncol=min(2, len(handles_resp)),
            fontsize=7.5,
            frameon=True,
            borderpad=0.18,
            borderaxespad=0.6,
            columnspacing=0.5,
            handlelength=1.0,
            handletextpad=0.30,
            labelspacing=0.28,
            fancybox=False,
            framealpha=0.95,
        )

    fig.tight_layout(h_pad=0.9, w_pad=0.9, rect=(0.08, 0.02, 0.98, 0.98))
    out_png = results_root / f"{experiment_name}_curves.png"
    fig.savefig(out_png, dpi=180)
    print(f"Saved figure: {out_png}")

    # Timing table figure
    fig_t, ax_t = plt.subplots(figsize=(max(6.0, 1.2 * len(algorithms) + 2.0), max(2.4, 0.6 * len(transition_times) + 1.8)))
    ax_t.axis("off")
    cell_text = []
    for i in range(len(transition_times)):
        row_cells = []
        for j in range(len(algorithms)):
            v_avg = avg_table[i, j]
            v_min = min_table[i, j]
            v_max = max_table[i, j]
            if np.isfinite(v_avg) and np.isfinite(v_min) and np.isfinite(v_max):
                row_cells.append(f"{v_avg:.6f}\n[{v_min:.6f}, {v_max:.6f}]")
            elif np.isfinite(v_avg):
                row_cells.append(f"{v_avg:.6f}")
            else:
                row_cells.append("-")
        cell_text.append(row_cells)
    row_labels = [f"tt={tt}s" for tt in transition_times]
    table_labels = [_display_algo_label(a) for a in algorithms]
    tbl = ax_t.table(cellText=cell_text, rowLabels=row_labels, colLabels=table_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    ax_t.set_title(r"$\mathrm{Per\text{-}frame\ compute\ time\ [s/frame]:\ avg\ [min,\ max]}$")
    fig_t.tight_layout()
    out_table = results_root / f"{experiment_name}_compute_time_table.png"
    fig_t.savefig(out_table, dpi=180)
    print(f"Saved timing table: {out_table}")

    plt.show()


def main() -> None:
    experiment_name = "experiment_05_ablation_debug_2"
    n_remove_highest_mean_curves = 10  # Set 0 to keep all curves, or n to remove the n highest-mean runs
    plot_results(
        experiment_name,
        n_remove_highest_mean_curves=n_remove_highest_mean_curves,
    )


if __name__ == "__main__":
    main()
