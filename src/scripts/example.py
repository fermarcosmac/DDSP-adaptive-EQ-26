"""
Minimal end-to-end example of the adaptive EQ framework.

Runs a single experiment configuration on one input signal and displays:
  - The validation error curve over time
  - Optional per-frame debug plots of the EQ magnitude response

Intended as a quick sanity check and starting point for new users.

Usage:
  python src/scripts/example.py --config configs/example_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "src" / "external"))

from utils.common import (
    get_delay_from_ir,
    get_compensation_EQ_params,
    interp_to_log_freq,
    kirkeby_deconvolve,
    load_config,
    load_rirs,
    run_control_experiment,
    set_seed,
)
from utils.plotting import configure_text_rendering, log_smooth_curve

# ---------------------------------------------------------------------------
# RIR scenario map
# ---------------------------------------------------------------------------
_RIR_MAP = {
    "moving_position": root / "data" / "SoundCam" / "moving_listener",
    "moving_person":   root / "data" / "SoundCam" / "moving_person",
    "static":          root / "data" / "SoundCam" / "moving_listener",
}


def _plot_validation_curve(time_axis: np.ndarray, val_hist: np.ndarray, transitions=None) -> None:
    """Plot the validation error over time."""
    configure_text_rendering()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time_axis, val_hist, linewidth=1.2, color="C0", label=r"$D_{\mathrm{rel}}$")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Unprocessed reference")
    if transitions:
        for t_start, t_end in transitions:
            ax.axvline(float(t_start), color="0.3", linestyle="--", linewidth=0.9, alpha=0.8)
            if t_end != t_start:
                ax.axvspan(float(t_start), float(t_end), color="0.85", alpha=0.35)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$D_{\mathrm{rel}}$")
    ax.set_title("Adaptive EQ — Validation Error")
    ax.set_ylim(0.0, 5.0)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    plt.show()


def _plot_eq_response(result: dict, frame_len: int, sr: int) -> None:
    """Plot the final compensated response vs the target and unprocessed LEM."""
    configure_text_rendering()
    nfft = int(2 * frame_len - 1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Target response
    if "target_mag_db" in result and "target_freq_axis" in result:
        f_t = np.asarray(result["target_freq_axis"], dtype=float)
        d_t = np.asarray(result["target_mag_db"], dtype=float)
        m = f_t > 0
        ax.plot(f_t[m], d_t[m], color="black", linewidth=1.3, label="Desired")

    # True LEM (unprocessed)
    if "final_true_lem_ir" in result:
        lem_ir = np.asarray(result["final_true_lem_ir"], dtype=np.float64)
        H_lem = np.fft.rfft(lem_ir, n=nfft)
        lem_db = 20.0 * np.log10(np.abs(H_lem) + 1e-12)
        m = freqs > 0
        f_s, lem_s = log_smooth_curve(freqs[m], lem_db[m], window_pts=121)
        ax.plot(f_s, lem_s, color="black", linestyle="--", linewidth=1.0, label="True LEM (unprocessed)")

    ax.set_xscale("log")
    ax.set_xlim(50, 20000)
    ax.set_ylim(-40, 20)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Adaptive EQ — Final Response")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    plt.show()


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 124))
    set_seed(seed)

    sim_cfg = dict(cfg.get("simulation_params", {}))
    sim_cfg.setdefault("frame_len", 8192)
    sim_cfg.setdefault("hop_len", sim_cfg["frame_len"])
    sim_cfg.setdefault("loss_type", "FD-MSE")
    sim_cfg.setdefault("optim_type", "SGD")
    sim_cfg.setdefault("mu_opt", 0.005)
    sim_cfg.setdefault("lambda_newton", 1.0)
    sim_cfg.setdefault("eps_0", 0.0)
    sim_cfg.setdefault("n_rirs", 1)
    sim_cfg.setdefault("ROI", [50.0, 20000.0])
    sim_cfg.setdefault("forget_factor", 0.05)
    sim_cfg.setdefault("target_response_type", "delay_and_mag")
    sim_cfg.setdefault("transition_time_s", 1.0)
    sim_cfg.setdefault("use_true_LEM", False)
    sim_cfg.setdefault("n_checkpoints", 0)

    scenario = cfg.get("scenario", "moving_position")
    if scenario not in _RIR_MAP:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {list(_RIR_MAP)}.")
    sim_cfg["rir_dir"] = str(_RIR_MAP[scenario])
    if scenario == "static":
        sim_cfg["n_rirs"] = 1

    # Input signal: white noise by default, or a song file
    input_cfg = cfg.get("input", {})
    max_audio_len_s = float(input_cfg.get("max_audio_len_s", 30.0))
    use_white_noise = bool(input_cfg.get("use_white_noise", True))

    if use_white_noise:
        input_spec = ("white_noise", {"max_audio_len_s": max_audio_len_s})
    else:
        song_path = input_cfg.get("song_path", None)
        if song_path is None:
            raise ValueError("Input config must provide 'song_path' when 'use_white_noise' is false.")
        input_spec = ("file", {"path": song_path, "max_audio_len_s": max_audio_len_s})

    print("Running example experiment...")
    print(f"  Scenario:   {scenario}")
    print(f"  Optimizer:  {sim_cfg['optim_type']}")
    print(f"  Loss:       {sim_cfg['loss_type']}")
    print(f"  Frame len:  {sim_cfg['frame_len']} samples")
    print(f"  Input:      {input_spec[0]}")

    result = run_control_experiment(sim_cfg, input_spec)
    if result is None:
        print("Experiment returned no result.")
        return

    time_axis = np.asarray(result["time_axis"], dtype=float)
    val_hist = np.asarray(result["validation_error_history"], dtype=float)
    transitions = result.get("transition_times", None)
    frame_len = int(sim_cfg["frame_len"])
    sr = int(result.get("sr", 48000))

    print(f"\nFinal validation error: {val_hist[-1]:.4f}  (1.0 = unprocessed)")

    _plot_validation_curve(time_axis, val_hist, transitions=transitions)
    _plot_eq_response(result, frame_len=frame_len, sr=sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a minimal adaptive-EQ example.")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "example_config.json",
        help="Path to the example JSON config file.",
    )
    args = parser.parse_args()
    main(args.config)
