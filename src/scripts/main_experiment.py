"""
Main experiment script.

Runs a full grid search over simulation parameters (optimizers, frame sizes,
transition times, loss types) using the adaptive EQ framework.

Results are saved as:
  results/<experiment_name>/config.json       — copy of the run config
  results/<experiment_name>/plot_data.pkl     — serialised curve data for plotting
  results/<experiment_name>/audio/            — per-run EQ-processed WAV files

Usage:
  python src/scripts/main_experiment.py --config configs/main_experiment_config.json
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import json
import numpy as np

# Add repo root and external library directory to sys.path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "src" / "external"))

from utils.common import (
    discover_input_signals,
    iter_param_grid,
    load_config,
    run_control_experiment,
    set_seed,
)
from utils.main import (
    accumulate_compute_time,
    build_experiment_grid,
    eq_audio_filename,
    make_compute_time_stats,
    save_wav,
    song_stem,
)


# ---------------------------------------------------------------------------
# RIR scenario map
# ---------------------------------------------------------------------------
_RIR_SUBDIR_MAP = {
    "moving_position": root / "data" / "SoundCam" / "moving_listener",
    "moving_person":   root / "data" / "SoundCam" / "moving_person",
    "static":          root / "data" / "SoundCam" / "moving_listener",
}


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 124))
    set_seed(seed)

    experiment_name = cfg.get("experiment_name", "main_experiment")
    scenario = cfg.get("scenario", "moving_position")
    input_cfg = cfg.get("input", {})

    if scenario not in _RIR_SUBDIR_MAP:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {list(_RIR_SUBDIR_MAP)}.")
    rir_dir = _RIR_SUBDIR_MAP[scenario]
    print(f"Scenario: '{scenario}' -> RIR dir: {rir_dir}")

    # Build fully resolved simulation configs from the grid
    resolved_cfgs = build_experiment_grid(cfg)
    total_combos = len(resolved_cfgs)

    input_signals = discover_input_signals(input_cfg)
    if not input_signals:
        input_signals = [("white_noise", {"max_audio_len_s": None})]

    # -----------------------------------------------------------------------
    # Aggregation structures
    # -----------------------------------------------------------------------
    curves = defaultdict(list)         # (tt, frame_len, optim, loss_type) -> [(t, val_hist), ...]
    loss_curves = defaultdict(list)    # (tt, frame_len, optim, loss_type) -> [(t, loss_hist), ...]
    compute_time_stats = make_compute_time_stats()
    tt_transitions: dict = {}
    input_ids_used: set = set()
    checkpoint_examples = defaultdict(dict)  # loss_type -> {optim: [checkpoints]}

    audio_out_dir = root / "results" / experiment_name / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    audio_saved_keys: set = set()

    for combo_idx, sim_cfg in enumerate(resolved_cfgs, start=1):
        sim_cfg["rir_dir"] = str(rir_dir)
        if scenario == "static":
            sim_cfg["n_rirs"] = 1

        loss_type_cfg = sim_cfg.get("loss_type", "FD-MSE")
        optim_used = sim_cfg.get("optim_type", "SGD")
        frame_len_used = int(sim_cfg.get("frame_len", 0))

        print("\n############################################")
        print(f"Combination {combo_idx}/{total_combos}")
        print("Simulation config:")
        for k, v in sorted(sim_cfg.items()):
            print(f"  {k}: {v}")
        print("############################################")

        for input_spec in input_signals:
            mode = input_spec[0] if isinstance(input_spec, (list, tuple)) else str(input_spec)
            info = input_spec[1] if isinstance(input_spec, (list, tuple)) and len(input_spec) == 2 else {}
            run_seed = seed + int(info.get("seed_offset", 0)) if mode == "white_noise" and isinstance(info, dict) else seed
            set_seed(run_seed)

            result = run_control_experiment(sim_cfg, input_spec)
            if result is None:
                continue

            if mode == "white_noise":
                wn_idx = info.get("realization_idx", 0) if isinstance(info, dict) else 0
                input_id = f"white_noise_{wn_idx}"
            else:
                input_id = str(info.get("path", mode))
            input_ids_used.add(input_id)

            song = song_stem(input_spec)
            tt = result.get("transition_time_s", sim_cfg.get("transition_time_s"))
            time_axis = np.asarray(result["time_axis"], dtype=float)
            val_hist = np.asarray(result["validation_error_history"], dtype=float)
            loss_hist = np.asarray(result.get("loss_history", []), dtype=float)

            key = (tt, frame_len_used, optim_used, loss_type_cfg)
            curves[key].append((time_axis, val_hist))
            if loss_hist.size:
                loss_curves[key].append((time_axis, loss_hist))

            ct_key = (tt, frame_len_used, optim_used)
            accumulate_compute_time(
                compute_time_stats[ct_key],
                float(result.get("control_experiment_time_s", 0.0)),
                int(result.get("n_frames", 0)),
            )

            if tt not in tt_transitions:
                tt_transitions[tt] = result.get("transition_times", None)

            if "checkpoints" in result and optim_used not in checkpoint_examples[loss_type_cfg]:
                checkpoint_examples[loss_type_cfg][optim_used] = result["checkpoints"]

            if "input_audio" in result:
                sr_audio = result["sr"]

                common_key = ("common", song)
                if common_key not in audio_saved_keys:
                    save_wav(audio_out_dir / f"input_{song}.wav",   result["input_audio"],   sr_audio)
                    save_wav(audio_out_dir / f"desired_{song}.wav", result["desired_audio"], sr_audio)
                    save_wav(audio_out_dir / f"noEQ_{song}.wav",    result["y_noEQ"],        sr_audio)
                    audio_saved_keys.add(common_key)

                eq_fname = eq_audio_filename(optim_used, loss_type_cfg, frame_len_used, tt, song)
                eq_key = (optim_used, loss_type_cfg, frame_len_used, tt, song)
                if eq_key not in audio_saved_keys:
                    save_wav(audio_out_dir / eq_fname, result["y_control"], sr_audio)
                    audio_saved_keys.add(eq_key)

    # -----------------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------------
    results_root = root / "results" / experiment_name
    results_root.mkdir(parents=True, exist_ok=True)

    with (results_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    plot_data = {
        "curves":               dict(curves),
        "loss_curves":          dict(loss_curves),
        "compute_time_stats":   dict(compute_time_stats),
        "tt_transitions":       tt_transitions,
        "input_signals":        sorted(input_ids_used),
        "unique_frame_lengths": sorted({k[1] for k in curves.keys()}),
        "frame_size_unit":      "samples",
        "checkpoint_examples":  {lt: dict(by_opt) for lt, by_opt in checkpoint_examples.items()},
    }
    with (results_root / "plot_data.pkl").open("wb") as f:
        pickle.dump(plot_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved config:        {results_root / 'config.json'}")
    print(f"Saved plot data:     {results_root / 'plot_data.pkl'}")
    print(f"Saved audio to:      {audio_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main adaptive-EQ experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "main_experiment_config.json",
        help="Path to the experiment JSON config file.",
    )
    args = parser.parse_args()
    main(args.config)
