import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Use LaTeX-style mathtext for figure text and numbers
mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.titlesize": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 13,
    }
)

def load_results(experiment_name: str, root: Path) -> tuple[dict, dict]:
    results_root = root / "results" / experiment_name
    config_path = results_root / "config.json"
    plot_data_path = results_root / "plot1_data.pkl"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not plot_data_path.exists():
        raise FileNotFoundError(f"Plot data file not found: {plot_data_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    with plot_data_path.open("rb") as f:
        plot1_data = pickle.load(f)

    return cfg, plot1_data


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace(" ", r"\ ")


def _format_input_label(input_signal) -> str:
    """Map an input-signal identifier to a compact, readable label."""
    s = str(input_signal)
    if s.startswith("white_noise"):
        return s
    p = Path(s)
    if p.suffix:
        return p.stem
    return s


def _format_optim_label(optim: str) -> str:
    """Human-readable optimizer label for plots."""
    return str(optim).replace("_", " ").replace("GHAM", "iHAM")


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
    linestyle,
    label,
    n_remove_highest_mean_curves: int = 0,
    run_labels=None,
    report_context: str = None,
):
    """Plot mean +- std from list of (time_axis, values)."""
    series = _select_series_for_averaging(
        series,
        n_remove_highest_mean_curves,
        run_labels=run_labels,
        report_context=report_context,
    )
    if not series:
        return
    min_len = min(len(v[1]) for v in series)
    all_vals, common_time = [], None
    for time_axis, vals in series:
        if not np.isnan(vals).any():
            all_vals.append(vals[:min_len])
        if common_time is None:
            common_time = time_axis[:min_len]
    if not all_vals or common_time is None:
        return
    vals_stack = np.stack(all_vals, axis=0)
    avg = np.mean(vals_stack, axis=0)
    std = np.std(vals_stack, axis=0)
    ax.plot(common_time, avg, color=color, linestyle=linestyle,
            linewidth=1.0, alpha=0.95, label=label)
    num_m = min(10, len(common_time))
    idxs = np.linspace(0, len(common_time) - 1, num=num_m, dtype=int)
    ax.errorbar(common_time[idxs], avg[idxs], yerr=std[idxs],
                fmt="none", ecolor=color, elinewidth=1.0, capsize=3, alpha=0.7)


def _parse_curve_key(key):
    """Parse legacy/new curve keys into (tt, frame_len, optim, loss)."""
    if not isinstance(key, tuple):
        raise ValueError(f"Invalid key type: {type(key)}")
    if len(key) == 3:
        tt, optim, loss_type = key
        frame_len = None
        return tt, frame_len, optim, loss_type
    if len(key) == 4:
        tt, frame_len, optim, loss_type = key
        return tt, frame_len, optim, loss_type
    raise ValueError(f"Unsupported curve key format: {key}")


def _normalize_curve_dict(curves_raw):
    """Return normalized map keyed by (tt, frame_len, optim, loss)."""
    curves_norm = defaultdict(list)
    for key, series in curves_raw.items():
        try:
            tt, frame_len, optim, loss_type = _parse_curve_key(key)
        except ValueError:
            continue
        curves_norm[(tt, frame_len, optim, loss_type)].extend(series)
    return curves_norm


def _group_curves_ignore_frame(curves_norm):
    """Aggregate normalized curves across frame lengths."""
    grouped = defaultdict(list)
    for (tt, _frame_len, optim, loss_type), series in curves_norm.items():
        grouped[(tt, optim, loss_type)].extend(series)
    return grouped


def _normalize_compute_time_stats(compute_time_stats_raw):
    """Return compute-time stats keyed by (tt, frame_len, optim)."""
    out = defaultdict(
        lambda: {
            "total_time_s": 0.0,
            "total_frames": 0,
            "num_runs": 0,
            "min_avg_time_per_frame_s": float("inf"),
            "max_avg_time_per_frame_s": float("-inf"),
        }
    )
    for key, stats in compute_time_stats_raw.items():
        if not isinstance(key, tuple):
            continue
        if len(key) == 2:
            tt, optim = key
            frame_len = None
        elif len(key) == 3:
            tt, frame_len, optim = key
        else:
            continue
        norm_key = (tt, frame_len, optim)
        out[norm_key]["total_time_s"] += float(stats.get("total_time_s", 0.0))
        out[norm_key]["total_frames"] += int(stats.get("total_frames", 0))
        out[norm_key]["num_runs"] += int(stats.get("num_runs", 0))
        out[norm_key]["min_avg_time_per_frame_s"] = min(
            float(out[norm_key]["min_avg_time_per_frame_s"]),
            float(stats.get("min_avg_time_per_frame_s", float("inf"))),
        )
        out[norm_key]["max_avg_time_per_frame_s"] = max(
            float(out[norm_key]["max_avg_time_per_frame_s"]),
            float(stats.get("max_avg_time_per_frame_s", float("-inf"))),
        )
    return out


def _plot_dual_scenario_validation(root: Path, n_remove_highest_mean_curves: int = 0) -> None:
    scenario_to_experiment = {
        "Moving listener position (all songs)": "experiment_04_ALL_SONGS_MOVING_POSITION",
        "Moving person (all songs)": "experiment_04_ALL_SONGS_MOVING_PERSON",
        "Moving listener position (white noise)": "experiment_04_WHITE_NOISE_MOVING_POSITION",
        "Moving person (white noise)": "experiment_04_WHITE_NOISE_MOVING_PERSON",
    }

    loaded = {}
    for scenario_name, experiment_name in scenario_to_experiment.items():
        try:
            _, plot_data = load_results(experiment_name, root)
            curves_norm = _normalize_curve_dict(plot_data.get("curves", {}))
            loaded[scenario_name] = {
                "curves_grouped": _group_curves_ignore_frame(curves_norm),
                "tt_transitions": plot_data.get("tt_transitions", {}),
                "input_signals": plot_data.get("input_signals", None),
            }
        except FileNotFoundError as exc:
            print(f"Skipping dual-scenario plot: {exc}")
            return

    all_tt = sorted(
        {
            key[0]
            for plot_data in loaded.values()
            for key in plot_data.get("curves_grouped", {}).keys()
        }
    )
    if not all_tt:
        print("Skipping dual-scenario plot: no validation curves available.")
        return

    # Build unified optimizer/loss spaces for consistent colors/styles.
    all_optim = sorted(
        {
            key[1]
            for plot_data in loaded.values()
            for key in plot_data.get("curves_grouped", {}).keys()
        }
    )
    all_loss = sorted(
        {
            key[2]
            for plot_data in loaded.values()
            for key in plot_data.get("curves_grouped", {}).keys()
        }
    )

    color_map = plt.get_cmap("tab10")
    optim_to_color = {opt: color_map(i % 10) for i, opt in enumerate(all_optim)}
    lt_linestyle = {lt: ls for lt, ls in zip(all_loss, ["-", "--", "-.", ":"])}

    n_rows = len(all_tt)
    n_cols = len(scenario_to_experiment)

    # Dedicated left label column for transition-time annotations.
    fig = plt.figure(figsize=(2.5 * n_cols + 1.5, 1.5 * n_rows + 0.8))
    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
            width_ratios=[0.22] + [1.0] * n_cols,
    )
    label_axes = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
    for ax_label in label_axes:
        ax_label.set_axis_off()
    axes = np.array([[fig.add_subplot(gs[r, c + 1]) for c in range(n_cols)]
                     for r in range(n_rows)])

    scenario_items = list(scenario_to_experiment.items())
    for row_i, tt in enumerate(all_tt):
        for col_i, (scenario_name, _) in enumerate(scenario_items):
            ax = axes[row_i, col_i]
            plot_data = loaded[scenario_name]
            curves = plot_data.get("curves_grouped", {})
            tt_transitions = plot_data.get("tt_transitions", {})
            scenario_input_signals = plot_data.get("input_signals", None)
            run_labels = [
                _format_input_label(sig) for sig in scenario_input_signals
            ] if scenario_input_signals else None

            if row_i == 0:
                ax.set_title(r"$\mathrm{" + _latex_escape(scenario_name) + r"}$")

            ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

            trans = tt_transitions.get(tt, None)
            if trans is not None:
                for t_start, t_end in trans:
                    ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                    if t_start != t_end:
                        ax.axvline(x=t_end, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                        ax.axvspan(t_start, t_end, color="0.8", alpha=0.3)

            for lt in all_loss:
                for optim in all_optim:
                    key = (tt, optim, lt)
                    if key not in curves or not curves[key]:
                        continue

                    legend_label = r"$\mathrm{" + _latex_escape(_format_optim_label(optim)) + r"}$"
                    _plot_mean_std(
                        ax,
                        curves[key],
                        color=optim_to_color[optim],
                        linestyle=lt_linestyle.get(lt, "-"),
                        label=legend_label,
                        n_remove_highest_mean_curves=n_remove_highest_mean_curves,
                        run_labels=run_labels,
                        report_context=f"{scenario_name} | tt={tt}, optim={optim}, loss={lt}",
                    )

            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
            ax.set_ylim(-0.2, 1.5)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0", "1"])
            is_left_col = (col_i == 0)
            is_bottom_row = (row_i == n_rows - 1)
            ax.tick_params(
                axis="x",
                which="both",
                bottom=is_bottom_row,
                labelbottom=is_bottom_row,
                top=False,
                labeltop=False,
            )
            ax.tick_params(
                axis="y",
                which="both",
                left=is_left_col,
                labelleft=is_left_col,
                right=False,
                labelright=False,
            )
            if row_i == n_rows - 1:
                ax.set_xlabel(r"$\mathrm{Time\ [s]}$")

        label_axes[row_i].text(
            0.5, 0.5,
            f"Transition\ntime: {tt} s",
            transform=label_axes[row_i].transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        # De-duplicate labels while preserving order.
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l in unique_l:
                continue
            unique_h.append(h)
            unique_l.append(l)
        axes[0, -1].legend(
            unique_h,
            unique_l,
            loc="upper right",
            fontsize=7,
            frameon=True,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.2,
            handletextpad=0.3,
            columnspacing=0.6,
            borderaxespad=0.2,
        )

    #fig.suptitle(r"$\mathrm{Validation\ Error\ Comparison:\ Moving\ Position\ vs\ Moving\ Person}$")
    fig.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.9)


def _plot_validation_only_column_true_lem(
    cfg: dict,
    plot1_data: dict,
    n_remove_highest_mean_curves: int = 0,
) -> None:
    """Extra figure: one column with validation curves per transition time."""
    curves_norm = _normalize_curve_dict(plot1_data.get("curves", {}))
    curves = _group_curves_ignore_frame(curves_norm)
    tt_transitions = plot1_data.get("tt_transitions", {})
    input_signals = plot1_data.get("input_signals", None)
    run_labels = [_format_input_label(sig) for sig in input_signals] if input_signals else None

    if not curves:
        print("Skipping TRUE_LEM validation-only plot: no validation curves available.")
        return

    all_curve_keys = list(curves.keys())
    unique_tt = sorted({k[0] for k in all_curve_keys})
    # Only plot smallest and largest transition times for the TRUE_LEM figure.
    if len(unique_tt) > 2:
        unique_tt = [unique_tt[0], unique_tt[-1]]
    optim_types = sorted({k[1] for k in all_curve_keys})
    unique_loss_types = sorted({k[2] for k in all_curve_keys})

    color_map = plt.get_cmap("tab10")
    optim_to_color = {opt: color_map(i % 10) for i, opt in enumerate(optim_types)}
    lt_linestyle = {lt: ls for lt, ls in zip(unique_loss_types, ["-", "--", "-.", ":"])}

    n_rows = 1
    n_cols = len(unique_tt)

    # Match the same subplot styling/layout conventions as _plot_dual_scenario_validation.
    fig = plt.figure(figsize=(2.8 * n_cols + 1.2, 1.5 * n_rows + 0.8))
    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[0.12] + [1.0] * n_cols,
        wspace=0.02,
    )
    label_axes = [fig.add_subplot(gs[0, 0])]
    for ax_label in label_axes:
        ax_label.set_axis_off()
    axes = np.array([fig.add_subplot(gs[0, c + 1]) for c in range(n_cols)])

    for col_i, tt in enumerate(unique_tt):
        ax = axes[col_i]
        ax.set_title(f"Transition time = {tt} s", pad=14)

        # Keep per-axis geometry consistent with the dual-scenario subplot style.
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1.5 / 2.5)

        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

        trans = tt_transitions.get(tt, None)
        if trans is not None:
            for t_start, t_end in trans:
                ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                if t_start != t_end:
                    ax.axvline(x=t_end, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                    ax.axvspan(t_start, t_end, color="0.8", alpha=0.3)

        for lt in unique_loss_types:
            for optim in optim_types:
                key = (tt, optim, lt)
                if key not in curves or not curves[key]:
                    continue

                if len(unique_loss_types) > 1:
                    label = f"{_format_optim_label(optim)} ({lt})"
                else:
                    label = _format_optim_label(optim)

                _plot_mean_std(
                    ax,
                    curves[key],
                    color=optim_to_color[optim],
                    linestyle=lt_linestyle.get(lt, "-"),
                    label=label,
                    n_remove_highest_mean_curves=n_remove_highest_mean_curves,
                    run_labels=run_labels,
                    report_context=f"TRUE_LEM | tt={tt}, optim={optim}, loss={lt}",
                )

        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_ylim(-0.2, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])

        is_bottom_row = True
        ax.tick_params(
            axis="x",
            which="both",
            bottom=is_bottom_row,
            labelbottom=is_bottom_row,
            top=False,
            labeltop=False,
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=(col_i == 0),
            labelleft=(col_i == 0),
            right=False,
            labelright=False,
        )
        if is_bottom_row:
            ax.set_xlabel(r"$\mathrm{Time\ [s]}$")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l in unique_l:
                continue
            unique_h.append(h)
            unique_l.append(l)
        axes[0].legend(
            unique_h,
            unique_l,
            loc="upper right",
            fontsize=7,
            frameon=True,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.2,
            handletextpad=0.3,
            columnspacing=0.6,
            borderaxespad=0.2,
        )

    fig.tight_layout(
        pad=0.6,
        w_pad=0.05,
        h_pad=1.15,
        rect=(0.03, 0.12, 0.97, 0.90),
    )
    fig.subplots_adjust(top=0.9, bottom=0.12)

    output_dir = Path(__file__).resolve().parents[1] / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "experiment_04_true_lem_validation.png", dpi=200, bbox_inches="tight")


def _plot_frame_size_validation_overlay_grids(
    cfg: dict,
    plot1_data: dict,
    n_remove_highest_mean_curves: int = 0,
) -> None:
    """For each transition-time scenario, plot one optimizer-grid with frame-size overlays."""
    plotting_cfg = cfg.get("plotting", {}) if isinstance(cfg, dict) else {}
    if plotting_cfg.get("enable_frame_size_analysis_plot", True) is False:
        return

    curves_norm = _normalize_curve_dict(plot1_data.get("curves", {}))
    if not curves_norm:
        print("Skipping frame-size analysis plot: no validation curves available.")
        return

    tt_transitions = plot1_data.get("tt_transitions", {})
    input_signals = plot1_data.get("input_signals", None)
    run_labels = [_format_input_label(sig) for sig in input_signals] if input_signals else None

    unique_tt = sorted({k[0] for k in curves_norm.keys()})
    unique_loss_types = sorted({k[3] for k in curves_norm.keys()})
    optim_types = sorted({k[2] for k in curves_norm.keys()})
    frame_lengths = sorted({k[1] for k in curves_norm.keys() if k[1] is not None})

    if len(frame_lengths) <= 1:
        return

    frame_color_map = plt.get_cmap("tab10")
    frame_to_color = {fl: frame_color_map(i % 10) for i, fl in enumerate(frame_lengths)}

    for tt in unique_tt:
        n_rows = len(unique_loss_types)
        n_cols = max(1, len(optim_types))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(1.0 * (2.0 * n_cols + 1.0), 1.0 * (1.3 * n_rows + 0.8)),
            squeeze=False,
        )

        for row_i, lt in enumerate(unique_loss_types):
            for col_i, optim in enumerate(optim_types):
                ax = axes[row_i, col_i]

                if row_i == 0:
                    ax.set_title(r"$\mathrm{" + _latex_escape(_format_optim_label(optim)) + r"}$")

                ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

                trans = tt_transitions.get(tt, None)
                if trans is not None:
                    for t_start, t_end in trans:
                        ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                        if t_start != t_end:
                            ax.axvline(x=t_end, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                            ax.axvspan(t_start, t_end, color="0.8", alpha=0.3)

                for fl in frame_lengths:
                    key = (tt, fl, optim, lt)
                    if key not in curves_norm or not curves_norm[key]:
                        continue
                    _plot_mean_std(
                        ax,
                        curves_norm[key],
                        color=frame_to_color[fl],
                        linestyle="-",
                        label=f"FL={fl}",
                        n_remove_highest_mean_curves=n_remove_highest_mean_curves,
                        run_labels=run_labels,
                        report_context=f"frame-size grid | tt={tt}, optim={optim}, loss={lt}, fl={fl}",
                    )

                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
                if lt == "TD-MSE":
                    ax.set_ylim(-0.2, 3.5)
                else:
                    ax.set_ylim(-0.2, 1.5)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["0", "1"])

                is_left_col = (col_i == 0)
                is_bottom_row = (row_i == n_rows - 1)
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=is_bottom_row,
                    labelbottom=is_bottom_row,
                    top=False,
                    labeltop=False,
                )
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=is_left_col,
                    labelleft=is_left_col,
                    right=False,
                    labelright=False,
                )

                if is_left_col:
                    if lt == "FD-MSE":
                        ax.set_ylabel(r"$D_{\text{rel}}$")
                    else:
                        ax.set_ylabel(r"$\mathrm{" + _latex_escape(str(lt)) + r"}$")
                if is_bottom_row:
                    ax.set_xlabel(r"$\mathrm{Time\ [s]}$")

        # One legend per figure (frame-length mapping).
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            unique_h, unique_l = [], []
            for h, l in zip(handles, labels):
                if l in unique_l:
                    continue
                unique_h.append(h)
                unique_l.append(l)
            axes[0, 0].legend(
                unique_h,
                unique_l,
                loc="upper right",
                fontsize=7,
                frameon=True,
                borderpad=0.2,
                labelspacing=0.2,
                handlelength=1.2,
                handletextpad=0.3,
                columnspacing=0.6,
                borderaxespad=0.2,
            )

        fig.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.9)


def plot_results(cfg: dict, plot1_data: dict, n_remove_highest_mean_curves: int = 0) -> None:
    curves_norm = _normalize_curve_dict(plot1_data["curves"])
    loss_curves_norm = _normalize_curve_dict(plot1_data.get("loss_curves", {}))
    curves = _group_curves_ignore_frame(curves_norm)
    loss_curves = _group_curves_ignore_frame(loss_curves_norm)
    compute_time_stats_norm = _normalize_compute_time_stats(plot1_data.get("compute_time_stats", {}))
    tt_transitions = plot1_data.get("tt_transitions", {})
    input_signals = plot1_data.get("input_signals", None)
    checkpoint_examples = plot1_data.get("checkpoint_examples", {})

    # Derive dimensions from normalized grouped keys: (transition_time_s, optim_type, loss_type)
    all_curve_keys = list(curves.keys())
    if not all_curve_keys:
        print("No curves found in plot data.")
        return

    unique_tt   = sorted({k[0] for k in all_curve_keys})
    optim_types = sorted({k[1] for k in all_curve_keys})
    unique_loss_types = sorted({k[2] for k in all_curve_keys})
    run_labels = [_format_input_label(sig) for sig in input_signals] if input_signals else None

    # Pretty-print configuration
    print("Loaded configuration:\n")
    for top_key in sorted(cfg.keys()):
        section = cfg[top_key]
        print(f"{top_key}:")
        if isinstance(section, dict):
            for k, v in sorted(section.items()):
                print(f"  {k}: {v}")
        else:
            print(f"  {section}")
    print()

    if input_signals:
        print("Input signals used:")
        for sig in input_signals:
            print(f"  {sig}")
        print()

    # ------------------------------------------------------------------
    # Compute-time table: rows = transition_time_s, cols = optimizer
    # Each cell reports average compute time per frame [s/frame]
    # ------------------------------------------------------------------
    if compute_time_stats_norm:
        compute_time_table = np.full((len(unique_tt), len(optim_types)), np.nan, dtype=float)
        compute_time_min_table = np.full((len(unique_tt), len(optim_types)), np.nan, dtype=float)
        compute_time_max_table = np.full((len(unique_tt), len(optim_types)), np.nan, dtype=float)
        for tt_i, tt in enumerate(unique_tt):
            for opt_i, opt in enumerate(optim_types):
                matching_stats = [
                    v for (tt_k, _fl_k, opt_k), v in compute_time_stats_norm.items()
                    if tt_k == tt and opt_k == opt
                ]
                if not matching_stats:
                    continue
                total_frames = int(sum(s.get("total_frames", 0) for s in matching_stats))
                total_time_s = float(sum(s.get("total_time_s", 0.0) for s in matching_stats))
                if total_frames > 0:
                    compute_time_table[tt_i, opt_i] = total_time_s / total_frames
                min_vals = [float(s.get("min_avg_time_per_frame_s", float("inf"))) for s in matching_stats]
                max_vals = [float(s.get("max_avg_time_per_frame_s", float("-inf"))) for s in matching_stats]
                min_vals = [v for v in min_vals if np.isfinite(v)]
                max_vals = [v for v in max_vals if np.isfinite(v)]
                if min_vals:
                    compute_time_min_table[tt_i, opt_i] = float(min(min_vals))
                if max_vals:
                    compute_time_max_table[tt_i, opt_i] = float(max(max_vals))

        print("Compute time per frame [s/frame] as avg [min, max] (rows=transition_time_s, cols=optimizer):")
        header = "transition_time_s" + "".join(f"\t{opt}" for opt in optim_types)
        print(header)
        for tt_i, tt in enumerate(unique_tt):
            row_vals = []
            for opt_i in range(len(optim_types)):
                v_avg = compute_time_table[tt_i, opt_i]
                v_min = compute_time_min_table[tt_i, opt_i]
                v_max = compute_time_max_table[tt_i, opt_i]
                if np.isfinite(v_avg) and np.isfinite(v_min) and np.isfinite(v_max):
                    row_vals.append(f"{v_avg:.6f} [{v_min:.6f}, {v_max:.6f}]")
                else:
                    row_vals.append("nan")
            print(f"{tt}\t" + "\t".join(row_vals))
        print()

        # Plot timing table in a dedicated figure for quick visual comparison
        fig_time, ax_time = plt.subplots(
            figsize=(max(6.0, 1.2 * len(optim_types) + 2.0), max(2.5, 0.6 * len(unique_tt) + 1.8))
        )
        ax_time.axis("off")
        table_cell_text = []
        for tt_i in range(len(unique_tt)):
            row = []
            for opt_i in range(len(optim_types)):
                v_avg = compute_time_table[tt_i, opt_i]
                v_min = compute_time_min_table[tt_i, opt_i]
                v_max = compute_time_max_table[tt_i, opt_i]
                if np.isfinite(v_avg) and np.isfinite(v_min) and np.isfinite(v_max):
                    row.append(f"{v_avg:.6f}\n[{v_min:.6f}, {v_max:.6f}]")
                else:
                    row.append("-")
            table_cell_text.append(row)
        table_row_labels = [f"tt={tt}s" for tt in unique_tt]
        timing_table = ax_time.table(
            cellText=table_cell_text,
            rowLabels=table_row_labels,
            colLabels=optim_types,
            cellLoc="center",
            loc="center",
        )
        timing_table.auto_set_font_size(False)
        timing_table.set_fontsize(8)
        timing_table.scale(1.0, 1.2)
        ax_time.set_title("Compute time per frame [s/frame]: avg [min, max]")
        fig_time.tight_layout()

    # ------------------------------------------------------------------
    # Combined figure: validation error + training loss for every
    # (transition_time_s, loss_type) combination.
    # Layout: rows = unique_tt
    #         cols = 2 * len(unique_loss_types)
    #               [val_LT0 | loss_LT0 | val_LT1 | loss_LT1 | ...]
    # ------------------------------------------------------------------
    color_map = plt.get_cmap("tab10")
    optim_to_color = {opt: color_map(i % 10) for i, opt in enumerate(optim_types)}
    # Line style per loss_type (useful when overlaying)
    lt_linestyle = {lt: ls for lt, ls in zip(unique_loss_types, ["-", "--", "-.", ":"])}

    n_rows = len(unique_tt)
    n_cols = 2 * len(unique_loss_types)
    # Extra label column (index 0) with half the width of a data column
    fig = plt.figure(figsize=(2.5 * n_cols + 1.2, 2 * n_rows))
    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[0.5] + [1.0] * n_cols,
    )
    # Hidden axes in column 0 — used only for row labels
    label_axes = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
    for _ax in label_axes:
        _ax.set_axis_off()
    # Data axes occupy columns 1..n_cols
    axes = np.array([[fig.add_subplot(gs[r, c + 1]) for c in range(n_cols)]
                     for r in range(n_rows)])

    def _plot_curve_series(ax, series, color, linestyle, label, report_context: str = None):
        """Plot mean ± std of a list of (time_axis, values) pairs."""
        series = _select_series_for_averaging(
            series,
            n_remove_highest_mean_curves,
            run_labels=run_labels,
            report_context=report_context,
        )
        if not series:
            return
        min_len = min(len(v[1]) for v in series)
        all_vals, common_time = [], None
        for time_axis, vals in series:
            if not np.isnan(vals).any(): # Only append if no NaNs appear
                all_vals.append(vals[:min_len])
            if common_time is None:
                common_time = time_axis[:min_len]
        if not all_vals or common_time is None:
            return
        vals_stack = np.stack(all_vals, axis=0)
        avg = np.mean(vals_stack, axis=0)
        std = np.std(vals_stack, axis=0)
        ax.plot(common_time, avg, color=color, linestyle=linestyle,
                linewidth=1.0, alpha=0.95, label=label)
        num_m = min(10, len(common_time))
        idxs = np.linspace(0, len(common_time) - 1, num=num_m, dtype=int)
        ax.errorbar(common_time[idxs], avg[idxs], yerr=std[idxs],
                    fmt="none", ecolor=color, elinewidth=1.0, capsize=3, alpha=0.7)

    def _add_transitions(ax, tt):
        trans = tt_transitions.get(tt, None)
        if trans is None:
            return
        for t_start, t_end in trans:
            ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
            if t_start != t_end:
                ax.axvline(x=t_end, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                ax.axvspan(t_start, t_end, color="0.8", alpha=0.3)

    for row_i, tt in enumerate(unique_tt):
        for lt_i, lt in enumerate(unique_loss_types):
            # Column order: loss first, then validation error
            ax_loss = axes[row_i, lt_i * 2]
            ax_val  = axes[row_i, lt_i * 2 + 1]

            # Column header titles on top row only (no tt — shown via row annotation)
            if row_i == 0:
                ax_loss.set_title(f"{lt} — Training loss")
                ax_val.set_title(f"{lt} — Validation error")

            ax_val.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
            _add_transitions(ax_val, tt)
            _add_transitions(ax_loss, tt)

            for optim in optim_types:
                color = optim_to_color[optim]
                label = _format_optim_label(optim)

                val_key  = (tt, optim, lt)
                loss_key = (tt, optim, lt)

                if val_key in curves and curves[val_key]:
                    _plot_curve_series(
                        ax_val,
                        curves[val_key],
                        color,
                        "-",
                        label,
                        report_context=f"tt={tt}, optim={optim}, loss={lt}",
                    )

                if loss_key in loss_curves and loss_curves[loss_key]:
                    series = loss_curves[loss_key]
                    # clip for log scale
                    clipped = [(ta, np.clip(v, 1e-8, None)) for ta, v in series]
                    _plot_curve_series(ax_loss, clipped, color, "-", label, report_context=None)

                # Set y-limits based on loss type
                if lt == 'FD-MSE':
                    ax_val.set_ylim(-0.1, 1.5)
                    ax_loss.set_ylim(1, 1e4)
                elif lt == 'TD-MSE':
                    ax_val.set_ylim(-0.1, 3.5)
            ax_loss.set_yscale("log")

            if row_i == n_rows - 1:
                ax_val.set_xlabel("Time [s]")
                ax_loss.set_xlabel("Time [s]")


        # Row label in its own dedicated column — always inside the figure
        label_axes[row_i].text(
            0.5, 0.5,
            f"Transition\ntime: {tt} s",
            transform=label_axes[row_i].transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    # Legend on the first (loss) subplot of the first loss type, first row only
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper right", fontsize=7)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Condensed checkpoint figure
    # Layout: rows = unique_loss_types
    #         cols = n_checkpoints  (taken from representative optimizer)
    # All optimizers' EQ responses are overlaid per cell.
    # Shared responses (desired, LEM, total) from the first available optimizer.
    # ------------------------------------------------------------------
    if checkpoint_examples:
        # checkpoint_examples: {loss_type: {optim_type: [cp, ...]}}
        # Normalize: handle old format ({optim_type: [cp]}) transparently
        if not isinstance(next(iter(checkpoint_examples.values())), dict):
            # Old single-level format — wrap it in a dict keyed by a placeholder loss_type
            checkpoint_examples = {"(default)": checkpoint_examples}

        cp_loss_types = sorted(checkpoint_examples.keys())
        # Find max number of checkpoints across all (lt, optim) combos
        n_cp_cols = max(
            len(cps)
            for lt_cps in checkpoint_examples.values()
            for cps in lt_cps.values()
        )
        n_cp_rows = len(cp_loss_types)

        fig2, axes2 = plt.subplots(
            n_cp_rows, n_cp_cols,
            figsize=(1.8 * n_cp_cols, 2.4 * n_cp_rows), # adjust checkpoint figure size
            sharex=True, sharey=True, squeeze=False,
        )

        eq_colors = {opt: color_map(i % 10) for i, opt in enumerate(optim_types)}

        color_desired = "black"
        color_lem     = "black"

        for row_i, lt in enumerate(cp_loss_types):
            by_optim = checkpoint_examples[lt]
            all_optims_sorted = sorted(by_optim.keys())

            # reference (shared background responses) from the first optimizer
            ref_optim = all_optims_sorted[0]
            ref_cps   = by_optim[ref_optim]

            for col_i in range(n_cp_cols):
                ax = axes2[row_i, col_i]

                # Shared reference from first optimizer
                if col_i < len(ref_cps):
                    ref_cp = ref_cps[col_i]
                    freqs_log  = np.asarray(ref_cp["freqs_log"],    dtype=float)
                    H_desired  = np.asarray(ref_cp["H_desired_db"], dtype=float)
                    H_lem      = np.asarray(ref_cp["H_lem_db"],     dtype=float)
                    H_total_ref = np.asarray(ref_cp["H_total_db"],  dtype=float)
                    sr_cp = float(ref_cp.get("sr", cfg.get("sample_rate", 48000)))
                    t_s   = float(ref_cp.get("time_s", 0.0))

                    ax.plot(freqs_log, H_desired, color=color_desired,
                            linewidth=1.4, linestyle="-", alpha=1.0, label="Desired", zorder=5)
                    ax.plot(freqs_log, H_lem, color=color_lem,
                            linewidth=1.0, linestyle="--", alpha=0.8, label="LEM (no EQ)", zorder=4)

                    if row_i == 0:
                        ax.set_title(f"t = {t_s:.1f} s")

                # Per-optimizer compensated response (EQ + LEM) overlays
                for opt in all_optims_sorted:
                    opt_cps = by_optim[opt]
                    if col_i >= len(opt_cps):
                        continue
                    cp = opt_cps[col_i]
                    freqs_log_o = np.asarray(cp["freqs_log"], dtype=float)
                    H_total_opt = cp.get("H_total_db", None)
                    if H_total_opt is not None:
                        ax.plot(freqs_log_o, np.asarray(H_total_opt, dtype=float),
                                color=eq_colors.get(opt, "gray"),
                                linewidth=1.0, alpha=0.55,
                                label="_nolegend_", zorder=3)

                ax.set_xscale("log")
                ax.set_ylim(-20, 20)
                ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
                if col_i == 0:
                    ax.set_ylabel("Magnitude [dB]", fontsize=10)

        # Single legend in top-left cell
        handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
        if handles2:
            axes2[0, 0].legend(handles2, labels2, loc="best", fontsize=7)

        # Use manual figure text so x-label position is independent and tightly controlled.
        fig2.tight_layout()
        bottom_axes_y = min(ax.get_position().y0 for ax in axes2[-1, :])
        x_label_y = max(0.005, bottom_axes_y - 0.12)
        fig2.text(0.5, x_label_y, "Frequency [Hz]", ha="center", va="top", fontsize=10)


def main() -> None:
    # Select the experiment to plot here
    experiment_name = "experiment_04_ALL_SONGS_MOVING_POSITION_TRUE_LEM"
    n_remove_highest_mean_curves = 2  # Set 0 to keep all curves, or n to remove the n highest-mean runs

    # Project root (same convention as experiment_04.py)
    root = Path(__file__).resolve().parents[1]

    cfg, plot1_data = load_results(experiment_name, root)
    print(f"Experiment name: {experiment_name}")
    plot_results(cfg, plot1_data, n_remove_highest_mean_curves=n_remove_highest_mean_curves)

    if experiment_name in {
        "experiment_04_ALL_SONGS_MOVING_POSITION",
        "experiment_04_ALL_SONGS_MOVING_PERSON",
        "experiment_04_WHITE_NOISE_MOVING_POSITION",
        "experiment_04_WHITE_NOISE_MOVING_PERSON",
    }:
        _plot_dual_scenario_validation(root, n_remove_highest_mean_curves=n_remove_highest_mean_curves)

    if experiment_name == "experiment_04_ALL_SONGS_MOVING_POSITION_TRUE_LEM":
        _plot_validation_only_column_true_lem(
            cfg,
            plot1_data,
            n_remove_highest_mean_curves=n_remove_highest_mean_curves,
        )

    frame_lengths_meta = plot1_data.get("unique_frame_lengths", None)
    if frame_lengths_meta is None:
        curves_norm = _normalize_curve_dict(plot1_data.get("curves", {}))
        frame_lengths_meta = sorted({k[1] for k in curves_norm.keys() if k[1] is not None})
    if len(frame_lengths_meta) > 1:
        _plot_frame_size_validation_overlay_grids(
            cfg,
            plot1_data,
            n_remove_highest_mean_curves=n_remove_highest_mean_curves,
        )
    
    plt.show()


if __name__ == "__main__":
    main()
