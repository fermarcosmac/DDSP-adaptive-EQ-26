"""
Shared plotting utilities used by both plot_results_main_experiment.py
and plot_results_ablation_study.py.

Covers:
- Matplotlib text/font configuration
- Result file loading
- Axis annotation helpers (panel labels, transition markers)
- Log-frequency smoothing
- Mean ± std time-series plots
- Mean ± std frequency-response plots
- Series outlier filtering
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Repository root (two levels above this file: src/utils -> src -> repo root)
root = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def configure_text_rendering() -> None:
    """Configure Matplotlib to use Computer Modern mathtext (LaTeX-like without TeX)."""
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "axes.titlesize": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 13,
        }
    )


# ---------------------------------------------------------------------------
# Result file loading
# ---------------------------------------------------------------------------

def load_results(experiment_name: str) -> Tuple[dict, dict, Path]:
    """Load config and plot data for a completed experiment.

    Args:
        experiment_name: Subdirectory name under ``results/``.

    Returns:
        Tuple of ``(cfg, plot_data, results_root)`` where ``results_root``
        is the ``Path`` to the experiment's results directory.

    Raises:
        FileNotFoundError: If ``config.json`` or ``plot_data.pkl`` is missing.
    """
    results_root = root / "results" / experiment_name
    cfg_path = results_root / "config.json"
    pkl_path = results_root / "plot_data.pkl"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing plot data file: {pkl_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    with pkl_path.open("rb") as f:
        plot_data = pickle.load(f)
    return cfg, plot_data, results_root


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

def format_algo_label(algo: str) -> str:
    """Return a human-readable algorithm label (replaces GHAM with iHAM).

    Args:
        algo: Raw algorithm name string.

    Returns:
        Display-friendly label.
    """
    return str(algo).replace("GHAM", "iHAM").replace("_", " ")


def format_input_label(input_signal) -> str:
    """Map an input-signal identifier to a compact readable label.

    Args:
        input_signal: String path, ``"white_noise_NNN"``, or similar identifier.

    Returns:
        Short display label (filename stem or original string).
    """
    s = str(input_signal)
    if s.startswith("white_noise"):
        return s
    p = Path(s)
    if p.suffix:
        return p.stem
    return s


# ---------------------------------------------------------------------------
# Axis annotation helpers
# ---------------------------------------------------------------------------

def add_panel_label(ax, label: str) -> None:
    """Add a bold panel label (e.g. 'A', 'B') in the top-right corner of an axes.

    Args:
        ax:    Matplotlib axes object.
        label: Label string to display.
    """
    ax.text(
        0.95,
        0.90,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.0,
            "boxstyle": "square,pad=0.3",
        },
        zorder=5,
    )


def plot_transition_markers(ax, transitions) -> None:
    """Add vertical dashed lines and shaded regions for acoustic transitions.

    Args:
        ax:          Matplotlib axes object.
        transitions: List of ``(t_start, t_end)`` tuples in seconds, or ``None``.
    """
    if transitions is None:
        return
    for t_start, t_end in transitions:
        ax.axvline(float(t_start), color="0.2", linestyle="--", linewidth=1.0, alpha=0.8)
        if float(t_end) != float(t_start):
            ax.axvline(float(t_end), color="0.2", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axvspan(float(t_start), float(t_end), color="0.85", alpha=0.35)


# ---------------------------------------------------------------------------
# Frequency smoothing
# ---------------------------------------------------------------------------

def log_smooth_curve(
    freq_hz: np.ndarray,
    mag_db: np.ndarray,
    window_pts: int = 61,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate to a log-frequency grid and apply moving-average smoothing.

    Args:
        freq_hz:    Linear frequency axis (Hz), must have all-positive values.
        mag_db:     Corresponding magnitude values in dB.
        window_pts: Moving-average kernel size (forced odd, minimum 5).

    Returns:
        Tuple ``(f_log, mag_smooth)`` on the log-frequency grid.
    """
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


# ---------------------------------------------------------------------------
# Outlier-series filtering
# ---------------------------------------------------------------------------

def select_series_for_averaging(
    series: list,
    n_remove_highest_mean: int = 0,
    run_labels: Optional[List] = None,
    report_context: Optional[str] = None,
) -> list:
    """Optionally remove the ``n`` runs with the highest time-mean value.

    Args:
        series:                 List of ``(time_axis, values)`` tuples.
        n_remove_highest_mean:  Number of runs to drop (0 = keep all).
        run_labels:             Optional labels for logging which runs were removed.
        report_context:         If given, print a removal log line with this context.

    Returns:
        Filtered series list.
    """
    if not series:
        return series

    n_remove = max(0, int(n_remove_highest_mean))
    if n_remove == 0:
        return series

    min_len = min(len(v[1]) for v in series)
    if min_len == 0 or len(series) <= 1:
        return series

    n_remove = min(n_remove, len(series) - 1)
    if n_remove == 0:
        return series

    means = np.asarray(
        [np.nanmean(np.asarray(vals[:min_len], dtype=float)) for _, vals in series],
        dtype=float,
    )
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


# ---------------------------------------------------------------------------
# Time-series mean ± std plot
# ---------------------------------------------------------------------------

def plot_mean_std(
    ax,
    series: list,
    color,
    label: str,
    linestyle: str = "-",
    n_remove_highest_mean: int = 0,
    run_labels: Optional[List] = None,
    report_context: Optional[str] = None,
) -> None:
    """Plot the mean ± std of a collection of time-series curves.

    Args:
        ax:                    Matplotlib axes object.
        series:                List of ``(time_axis, values)`` tuples.
        color:                 Line colour passed to ``ax.plot``.
        label:                 Legend label.
        linestyle:             Line style (default ``"-"``).
        n_remove_highest_mean: Number of outlier runs to drop before averaging.
        run_labels:            Optional labels used when logging dropped runs.
        report_context:        Context string printed when runs are dropped.
    """
    series = select_series_for_averaging(
        series, n_remove_highest_mean, run_labels=run_labels, report_context=report_context
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

    ax.plot(t, avg, color=color, linestyle=linestyle, linewidth=1.1, alpha=0.95, label=label)
    n_mark = min(10, len(t))
    idx = np.linspace(0, len(t) - 1, num=n_mark, dtype=int)
    ax.errorbar(
        t[idx], avg[idx], yerr=std[idx],
        fmt="none", ecolor=color, elinewidth=0.9, capsize=3, alpha=0.75,
    )


# ---------------------------------------------------------------------------
# Frequency-response mean ± std plot
# ---------------------------------------------------------------------------

def plot_response_mean_std(
    ax,
    series: list,
    color,
    label: str,
    n_remove_highest_mean: int = 0,
    run_labels: Optional[List] = None,
    report_context: Optional[str] = None,
    smooth_window_pts: int = 121,
) -> None:
    """Plot the mean ± std of a collection of frequency-response curves.

    Curves are interpolated to a common log-frequency grid before averaging,
    then smoothed with a moving-average kernel.

    Args:
        ax:                    Matplotlib axes object.
        series:                List of ``(freq_hz, mag_db)`` tuples.
        color:                 Line colour.
        label:                 Legend label.
        n_remove_highest_mean: Number of outlier runs to drop.
        run_labels:            Optional labels for dropped-run logging.
        report_context:        Context string printed when runs are dropped.
        smooth_window_pts:     Moving-average kernel size for smoothing.
    """
    series = select_series_for_averaging(
        series, n_remove_highest_mean, run_labels=run_labels, report_context=report_context
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
    f_s, avg_s = log_smooth_curve(ref_f[m], avg[m], window_pts=smooth_window_pts)
    _, std_s = log_smooth_curve(ref_f[m], std[m], window_pts=smooth_window_pts)
    ax.plot(f_s, avg_s, color=color, linewidth=1.25, label=label)
    ax.fill_between(f_s, avg_s - std_s, avg_s + std_s, color=color, alpha=0.16, linewidth=0)
