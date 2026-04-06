from pathlib import Path
from collections import defaultdict
import json
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

from utils_ex04 import (
    load_config,
    iter_param_grid,
    discover_input_signals,
    run_control_experiment,
)

root = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    if torch.cuda.is_available():
        #torch.cuda.manual_seed_all(seed)
        pass


def _song_stem(input_spec) -> str:
    """Derive a filesystem-safe stem from an input_spec tuple."""
    if isinstance(input_spec, (list, tuple)) and len(input_spec) == 2:
        mode, info = input_spec
        if mode == "white_noise":
            if isinstance(info, dict) and "realization_idx" in info:
                try:
                    idx = int(info["realization_idx"])
                    return f"white_noise_{idx:03d}"
                except (TypeError, ValueError):
                    pass
            return "white_noise"
        if isinstance(info, dict) and "path" in info:
            return Path(info["path"]).stem
        return str(mode)
    return str(input_spec)


def main() -> None:

    # Load configuration
    config_path = root / "configs" / "experiment_04_config_debug.json"
    cfg = load_config(config_path)

    # Global seeding for reproducibility (EQ init, white noise, song sampling)
    seed = int(cfg.get("seed", 124))
    set_seed(seed)

    experiment_name = cfg.get("experiment_name", "experiment_04")
    sim_param_grid = cfg.get("simulation_params", {})
    input_cfg = cfg.get("input", {})

    # Resolve RIR directory from scenario config
    scenario = cfg.get("scenario", "moving_position")
    _rir_subdir_map = {
        "moving_position": root / "data" / "rir" / "moving_position",
        "moving_person":   root / "data" / "rir" / "moving_person",
        "static":          root / "data" / "rir" / "moving_position",
    }
    if scenario not in _rir_subdir_map:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Must be one of: {list(_rir_subdir_map)}"
        )
    rir_dir = _rir_subdir_map[scenario]
    print(f"Scenario: '{scenario}' -> RIR dir: {rir_dir}")

    if not sim_param_grid:
        raise ValueError("No 'simulation_params' section found in config.")

    input_signals = discover_input_signals(input_cfg)
    if not input_signals:
        input_signals = [("white_noise", {"max_audio_len_s": None})]

    # Pair optim_type and mu_opt
    # mu_opt can be:
    #   - a flat list (same step sizes for all loss types)
    #   - a dict {loss_type: [mu, ...]} for per-loss-type step sizes
    optim_list = sim_param_grid.get("optim_type", [])
    mu_opt_raw = sim_param_grid.get("mu_opt", [])
    mu_per_loss: dict = {}   # loss_type -> list[float]
    mu_list: list = []

    if isinstance(mu_opt_raw, dict):
        mu_per_loss = mu_opt_raw          # dict format
        # Validate each loss-type list length matches optim_list
        for lt_key, lt_mus in mu_per_loss.items():
            if optim_list and len(optim_list) != len(lt_mus):
                raise ValueError(
                    f"Length mismatch for mu_opt['{lt_key}']: "
                    f"got {len(lt_mus)} values but {len(optim_list)} optimizers."
                )
    else:
        mu_list = mu_opt_raw              # flat list format
        if optim_list and mu_list and len(optim_list) != len(mu_list):
            raise ValueError(
                "Length mismatch between 'optim_type' and 'mu_opt' in config: "
                f"got {len(optim_list)} optimizers and {len(mu_list)} mu_opt values. "
                "They must have the same length so each optimizer pairs with one mu_opt."
            )

    # All other parameters form a full grid (includes loss_type, transition_time_s, ...)
    # lambda_newton and eps_0 are excluded from the grid if they are dicts (per-loss-type)
    # hop_len is intentionally excluded: it is always set equal to frame_len.
    _per_loss_scalar_keys = ("lambda_newton", "eps_0")

    lambda_newton_raw = sim_param_grid.get("lambda_newton", None)
    eps_0_raw         = sim_param_grid.get("eps_0", None)

    # Per-loss-type scalars: dict {loss_type: [val]} or flat list [val]
    lambda_newton_per_loss: dict = {}
    eps_0_per_loss: dict = {}
    if isinstance(lambda_newton_raw, dict):
        lambda_newton_per_loss = lambda_newton_raw
    if isinstance(eps_0_raw, dict):
        eps_0_per_loss = eps_0_raw

    base_param_grid = {
        k: v for k, v in sim_param_grid.items()
        if k not in ("optim_type", "mu_opt", "hop_len")
        and not (k in _per_loss_scalar_keys and isinstance(v, dict))
    }

    # Pre-compute total number of parameter combinations for logging
    base_cfgs = list(iter_param_grid(base_param_grid))
    num_opt_mu = len(optim_list) if optim_list else 0
    total_combos = len(base_cfgs) * max(1, num_opt_mu)

    # -----------------------------------------------------------------------
    # Aggregation structures
    # curves / loss_curves: keyed by (transition_time_s, frame_len, optim_type, loss_type)
    # checkpoint_examples:  {loss_type: {optim_type: [cp, ...]}}  (first occurrence)
    # audio saved for EVERY (input_signal, optim_type, loss_type) combination
    # -----------------------------------------------------------------------
    curves = defaultdict(list)        # (tt, frame_len, optim, lt) -> [(time_axis, val_hist), ...]
    loss_curves = defaultdict(list)   # (tt, frame_len, optim, lt) -> [(time_axis, loss_hist), ...]
    compute_time_stats = defaultdict(
        lambda: {
            "total_time_s": 0.0,
            "total_frames": 0,
            "num_runs": 0,
            "min_avg_time_per_frame_s": float("inf"),
            "max_avg_time_per_frame_s": float("-inf"),
        }
    )
    tt_transitions = {}               # tt -> list of (start_s, end_s)
    input_ids_used = set()
    checkpoint_examples = defaultdict(dict)  # lt -> {optim: [cp, ...]}

    audio_out_dir = root / "data" / "audio" / "output" / experiment_name
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    audio_saved_keys = set()          # (song_stem,) or (optim, lt, song_stem)

    def _save_wav(path: Path, arr: np.ndarray, sr: int) -> None:
        peak = np.abs(arr).max()
        if peak > 0:
            arr = arr / peak
        t = torch.from_numpy(arr).float().unsqueeze(0)
        torchaudio.save(str(path), t, sr)
        print(f"  Saved: {path}")

    def _unwrap(v):
        """Unwrap a single-element list to a scalar; pass scalars through unchanged."""
        return v[0] if isinstance(v, list) and len(v) == 1 else v

    combo_idx = 0
    for base_cfg in base_cfgs:

        # Resolve mu list for this base_cfg's loss_type
        current_lt = base_cfg.get("loss_type", "")
        if mu_per_loss:
            resolved_mu_list = mu_per_loss.get(current_lt, list(mu_per_loss.values())[0])
        else:
            resolved_mu_list = mu_list

        # Resolve per-loss-type scalars (unwrap single-element lists if present)
        resolved_lambda_newton = (
            _unwrap(lambda_newton_per_loss.get(current_lt, list(lambda_newton_per_loss.values())[0]))
            if lambda_newton_per_loss else None
        )
        resolved_eps_0 = (
            _unwrap(eps_0_per_loss.get(current_lt, list(eps_0_per_loss.values())[0]))
            if eps_0_per_loss else None
        )

        if optim_list and resolved_mu_list:
            opt_mu_pairs = list(zip(optim_list, resolved_mu_list))
        else:
            opt_mu_pairs = [(None, None)]

        for optim, mu in opt_mu_pairs:
            sim_cfg = dict(base_cfg)
            if "frame_len" not in sim_cfg:
                raise ValueError("'frame_len' must be provided in simulation_params.")
            sim_cfg["hop_len"] = sim_cfg["frame_len"]
            if optim is not None:
                sim_cfg["optim_type"] = optim
            if mu is not None:
                sim_cfg["mu_opt"] = mu
            if resolved_lambda_newton is not None:
                sim_cfg["lambda_newton"] = resolved_lambda_newton
            if resolved_eps_0 is not None:
                sim_cfg["eps_0"] = resolved_eps_0

            sim_cfg["rir_dir"] = str(rir_dir)
            if scenario == "static":
                sim_cfg["n_rirs"] = 1

            loss_type_cfg = sim_cfg.get("loss_type", "FD-MSE")

            combo_idx += 1
            print("\n############################################")
            if total_combos > 0:
                print(f"Combination {combo_idx}/{total_combos}")
            else:
                print(f"Combination {combo_idx}")
            print("Simulation config:")
            for k, v in sorted(sim_cfg.items()):
                print(f"  {k}: {v}")
            print("############################################")

            for input_spec in input_signals:
                # Keep deterministic reproducibility while allowing independent
                # white-noise realizations per run.
                run_seed = seed
                mode = input_spec[0] if isinstance(input_spec, (list, tuple)) else str(input_spec)
                info = input_spec[1] if isinstance(input_spec, (list, tuple)) and len(input_spec) == 2 else {}
                if mode == "white_noise" and isinstance(info, dict):
                    run_seed = seed + int(info.get("seed_offset", 0))
                set_seed(run_seed)
                result = run_control_experiment(sim_cfg, input_spec)
                if result is None:
                    continue

                # Derive stable identifiers
                if mode == "white_noise":
                    wn_idx = info.get("realization_idx", 0) if isinstance(info, dict) else 0
                    input_id = f"white_noise_{wn_idx}"
                else:
                    input_id = str(info.get("path", mode))
                song = _song_stem(input_spec)
                input_ids_used.add(input_id)

                tt = result.get("transition_time_s", sim_cfg.get("transition_time_s"))
                frame_len_used = int(sim_cfg.get("frame_len", 0))
                optim_used = result.get("optim_type", sim_cfg.get("optim_type"))
                loss_type_used = loss_type_cfg  # loss_type is not echoed in result dict
                time_axis = np.asarray(result["time_axis"], dtype=float)
                val_hist = np.asarray(result["validation_error_history"], dtype=float)
                loss_hist = np.asarray(result.get("loss_history", []), dtype=float)

                curves[(tt, frame_len_used, optim_used, loss_type_used)].append((time_axis, val_hist))
                if loss_hist.size:
                    loss_curves[(tt, frame_len_used, optim_used, loss_type_used)].append((time_axis, loss_hist))

                # Aggregate compute-time statistics per (transition_time_s, frame_len, optimizer)
                total_time_s = float(result.get("control_experiment_time_s", 0.0))
                n_frames = int(result.get("n_frames", 0))
                ct_key = (tt, frame_len_used, optim_used)
                compute_time_stats[ct_key]["total_time_s"] += total_time_s
                compute_time_stats[ct_key]["total_frames"] += n_frames
                compute_time_stats[ct_key]["num_runs"] += 1
                if n_frames > 0:
                    avg_time_per_frame_s = total_time_s / n_frames
                    compute_time_stats[ct_key]["min_avg_time_per_frame_s"] = min(
                        float(compute_time_stats[ct_key]["min_avg_time_per_frame_s"]),
                        float(avg_time_per_frame_s),
                    )
                    compute_time_stats[ct_key]["max_avg_time_per_frame_s"] = max(
                        float(compute_time_stats[ct_key]["max_avg_time_per_frame_s"]),
                        float(avg_time_per_frame_s),
                    )

                if tt not in tt_transitions:
                    tt_transitions[tt] = result.get("transition_times", None)

                # First checkpoint set per (loss_type, optim)
                if "checkpoints" in result and optim_used not in checkpoint_examples[loss_type_used]:
                    checkpoint_examples[loss_type_used][optim_used] = result["checkpoints"]

                # Save audio for every (song, optim, loss_type) combination
                if "input_audio" in result:
                    sr_audio = result["sr"]
                    safe_optim = optim_used.replace("-", "_").replace(" ", "_")
                    safe_lt = loss_type_used.replace("-", "_")

                    # Common signals: once per song (input, desired, noEQ are optimizer-independent)
                    common_key = ("common", song)
                    if common_key not in audio_saved_keys:
                        _save_wav(audio_out_dir / f"input_{song}.wav",       result["input_audio"],   sr_audio)
                        _save_wav(audio_out_dir / f"desired_{song}.wav",     result["desired_audio"], sr_audio)
                        _save_wav(audio_out_dir / f"noEQ_{song}.wav",        result["y_noEQ"],        sr_audio)
                        audio_saved_keys.add(common_key)

                    # Per-(optim, loss_type, frame_len, tt, song) EQ output
                    safe_tt = str(tt).replace(".", "p")
                    safe_fl = str(frame_len_used)
                    eq_key = (safe_optim, safe_lt, safe_fl, safe_tt, song)
                    if eq_key not in audio_saved_keys:
                        _save_wav(
                            audio_out_dir / f"EQ_{safe_optim}_{safe_lt}_fl{safe_fl}_tt{safe_tt}_{song}.wav",
                            result["y_control"], sr_audio,
                        )
                        audio_saved_keys.add(eq_key)
            # end for input_spec

    # -----------------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------------
    results_root = root / "results" / experiment_name
    results_root.mkdir(parents=True, exist_ok=True)

    config_out_path = results_root / "config.json"
    with config_out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    plot1_data = {
        "curves":             dict(curves),
        "loss_curves":        dict(loss_curves),
        "compute_time_stats": dict(compute_time_stats),
        "tt_transitions":     tt_transitions,
        "input_signals":      sorted(input_ids_used),
        "unique_frame_lengths": sorted({k[1] for k in curves.keys()}),
        "frame_size_unit": "samples",
        # checkpoint_examples: {loss_type: {optim_type: [cp, ...]}}
        "checkpoint_examples": {lt: dict(by_optim) for lt, by_optim in checkpoint_examples.items()},
    }
    plot_out_path = results_root / "plot1_data.pkl"
    with plot_out_path.open("wb") as f:
        pickle.dump(plot1_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved config to: {config_out_path}")
    print(f"Saved plotting data to: {plot_out_path}")
    print(f"Saved audio to: {audio_out_dir}")

if __name__ == "__main__":
    main()