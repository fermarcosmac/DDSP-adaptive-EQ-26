"""
Ablation study script.

Runs the proposed adaptive-EQ framework under multiple configurations and
compares them against FxLMS and FxFDAF FIR baselines.

Results are saved as:
  results/<experiment_name>/config.json      — copy of the run config
  results/<experiment_name>/plot_data.pkl    — serialised curve data for plotting
  results/<experiment_name>/audio/           — per-run EQ-processed WAV files

Usage:
  python src/scripts/ablation_study.py --config configs/ablation_study_config.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

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
from utils.main import make_compute_time_stats, save_wav
from utils.ablation import (
    build_proposed_sim_cfg,
    compute_exact_final_response_fir,
    compute_exact_final_response_proposed,
    estimate_final_equalized_response,
    framewise_td_mse,
    run_fir_baseline_experiment,
    safe_token,
)


# ---------------------------------------------------------------------------
# RIR scenario map
# ---------------------------------------------------------------------------
_RIR_MAP = {
    "moving_position": root / "data" / "SoundCam" / "moving_listener",
    "moving_person":   root / "data" / "SoundCam" / "moving_person",
    "static":          root / "data" / "SoundCam" / "moving_listener",
}


def _song_stem(input_spec) -> str:
    if isinstance(input_spec, (list, tuple)) and len(input_spec) == 2:
        mode, info = input_spec
        if mode == "white_noise":
            idx = int(info.get("realization_idx", 0)) if isinstance(info, dict) else 0
            return f"white_noise_{idx:03d}"
        if isinstance(info, dict) and "path" in info:
            return Path(info["path"]).stem
        return str(mode)
    return str(input_spec)


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 124))
    set_seed(seed)

    experiment_name = cfg.get("experiment_name", "ablation_study")
    scenario = cfg.get("scenario", "moving_position")
    input_cfg = cfg.get("input", {})
    shared_grid = cfg.get("shared_simulation_params", {})
    proposed_cfgs = cfg.get("proposed_configs", [])
    fir_baselines_cfg = cfg.get("fir_baselines", {})

    if not shared_grid:
        raise ValueError("Missing 'shared_simulation_params' in config.")
    if scenario not in _RIR_MAP:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {list(_RIR_MAP)}.")
    rir_dir = _RIR_MAP[scenario]

    input_signals = discover_input_signals(input_cfg)
    if not input_signals:
        input_signals = [("white_noise", {"max_audio_len_s": 60.0})]

    # -----------------------------------------------------------------------
    # Aggregation structures
    # -----------------------------------------------------------------------
    td_mse_curves = defaultdict(list)
    validation_curves = defaultdict(list)
    final_response_curves = defaultdict(list)
    compute_time_stats = make_compute_time_stats()
    tt_transitions: dict = {}
    input_ids_used: set = set()
    target_response_example = None
    true_lem_response_example = None

    audio_out_dir = root / "results" / experiment_name / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    audio_saved_keys: set = set()

    base_cfgs = list(iter_param_grid(shared_grid))
    combo_total = len(base_cfgs) * max(1, len(input_signals))
    combo_idx = 0

    for shared_cfg in base_cfgs:
        sim_shared = dict(shared_cfg)
        sim_shared["rir_dir"] = str(rir_dir)
        if scenario == "static":
            sim_shared["n_rirs"] = 1

        frame_len = int(sim_shared["frame_len"])
        hop_len = int(sim_shared["hop_len"])

        for input_spec in input_signals:
            combo_idx += 1
            mode, info = input_spec
            run_seed = seed + int(info.get("seed_offset", 0)) if mode == "white_noise" and isinstance(info, dict) else seed
            set_seed(run_seed)

            if mode == "white_noise":
                input_id = f"white_noise_{int(info.get('realization_idx', 0))}"
            else:
                input_id = str(info.get("path", mode))
            input_ids_used.add(input_id)

            song = _song_stem(input_spec)
            print("\n############################################")
            print(f"Combination {combo_idx}/{combo_total}")
            print(f"Scenario: {scenario}  |  Input: {song}")
            print("############################################")

            # ----------------------------------------------------------------
            # 1) Proposed framework configurations
            # ----------------------------------------------------------------
            for proposed_entry in proposed_cfgs:
                label = str(proposed_entry.get("label", "Proposed"))
                sim_cfg = build_proposed_sim_cfg(sim_shared, proposed_entry)
                result = run_control_experiment(sim_cfg, input_spec)
                if result is None:
                    continue

                td_curve = framewise_td_mse(
                    np.asarray(result["desired_audio"], dtype=np.float64),
                    np.asarray(result["y_control"], dtype=np.float64),
                    frame_len=frame_len,
                    hop_len=hop_len,
                )
                t_axis = np.asarray(result.get("time_axis", []), dtype=np.float64)
                v_curve = np.asarray(result.get("validation_error_history", []), dtype=np.float64)

                # Align lengths
                if td_curve.size and t_axis.size:
                    n = min(len(td_curve), len(t_axis))
                    td_curve, t_axis = td_curve[:n], t_axis[:n]
                if t_axis.size and v_curve.size:
                    n = min(len(t_axis), len(v_curve))
                    t_axis, v_curve = t_axis[:n], v_curve[:n]

                tt = sim_cfg["transition_time_s"]
                key = (tt, f"Proposed:{label}")
                td_mse_curves[key].append((t_axis, td_curve))
                validation_curves[key].append((t_axis, v_curve))

                if tt not in tt_transitions:
                    tt_transitions[tt] = result.get("transition_times", None)

                sr_audio = int(result.get("sr", 48000))
                common_key = ("common", song)
                if common_key not in audio_saved_keys:
                    save_wav(audio_out_dir / f"desired_{song}.wav", np.asarray(result.get("desired_audio", []), dtype=np.float32), sr_audio)
                    save_wav(audio_out_dir / f"noEQ_{song}.wav", np.asarray(result.get("y_noEQ", []), dtype=np.float32), sr_audio)
                    audio_saved_keys.add(common_key)

                safe_algo = safe_token(f"Proposed_{label}")
                safe_lt = safe_token(sim_cfg.get("loss_type", "NA")).replace("-", "_")
                safe_tt = safe_token(str(tt).replace(".", "p"))
                eq_key = (safe_algo, safe_lt, safe_tt, song)
                if eq_key not in audio_saved_keys:
                    save_wav(
                        audio_out_dir / f"EQ_{safe_algo}_{safe_lt}_tt{safe_tt}_{song}.wav",
                        np.asarray(result.get("y_control", []), dtype=np.float32),
                        sr_audio,
                    )
                    audio_saved_keys.add(eq_key)

                if all(k in result for k in ("final_eq_params_normalized", "final_gain_db", "final_true_lem_ir")):
                    resp_f, resp_db = compute_exact_final_response_proposed(
                        final_eq_params_normalized=np.asarray(result["final_eq_params_normalized"], dtype=np.float32),
                        final_gain_db=np.asarray(result["final_gain_db"], dtype=np.float32),
                        true_lem_ir=np.asarray(result["final_true_lem_ir"], dtype=np.float32),
                        sr=sr_audio,
                        nfft=int(2 * frame_len - 1),
                    )
                else:
                    resp_f, resp_db = estimate_final_equalized_response(
                        input_audio=np.asarray(result.get("input_audio", []), dtype=np.float64),
                        output_audio=np.asarray(result.get("y_control", []), dtype=np.float64),
                        sr=sr_audio,
                        frame_len=frame_len,
                    )
                if resp_f.size and resp_db.size:
                    final_response_curves[key].append((resp_f, resp_db))

                stats = compute_time_stats[key]
                stats["total_time_s"] += float(result.get("control_experiment_time_s", 0.0))
                stats["total_frames"] += int(result.get("n_frames", 0))
                stats["num_runs"] += 1
                avg_pf = float(result.get("avg_compute_time_per_frame_s", float("nan")))
                if np.isfinite(avg_pf):
                    stats["min_avg_time_per_frame_s"] = min(stats["min_avg_time_per_frame_s"], avg_pf)
                    stats["max_avg_time_per_frame_s"] = max(stats["max_avg_time_per_frame_s"], avg_pf)

            # ----------------------------------------------------------------
            # 2) FIR baselines
            # ----------------------------------------------------------------
            for algo_name in ("FxLMS", "FxFDAF"):
                algo_cfg = fir_baselines_cfg.get(algo_name, {})
                if not bool(algo_cfg.get("enabled", True)):
                    continue

                fir_result = run_fir_baseline_experiment(sim_shared, input_spec, algorithm=algo_name, algo_cfg=algo_cfg)

                tt = fir_result["transition_time_s"]
                key = (tt, algo_name)
                t_axis = np.asarray(fir_result["time_axis"], dtype=np.float64)

                td_curve = framewise_td_mse(
                    np.asarray(fir_result.get("desired_audio", []), dtype=np.float64),
                    np.asarray(fir_result.get("y_control", []), dtype=np.float64),
                    frame_len=frame_len,
                    hop_len=hop_len,
                )
                v_curve = np.asarray(fir_result["validation_error_history"], dtype=np.float64)

                if td_curve.size and t_axis.size:
                    n = min(len(td_curve), len(t_axis))
                    td_curve, t_axis = td_curve[:n], t_axis[:n]
                if t_axis.size and v_curve.size:
                    n = min(len(t_axis), len(v_curve))
                    t_axis, v_curve = t_axis[:n], v_curve[:n]

                td_mse_curves[key].append((t_axis, td_curve))
                validation_curves[key].append((t_axis, v_curve))

                if tt not in tt_transitions:
                    tt_transitions[tt] = fir_result.get("transition_times", None)

                sr_audio = int(fir_result.get("sr", 48000))
                common_key = ("common", song)
                if common_key not in audio_saved_keys:
                    save_wav(audio_out_dir / f"desired_{song}.wav", np.asarray(fir_result.get("desired_audio", []), dtype=np.float32), sr_audio)
                    if "final_true_lem_ir" in fir_result:
                        x_in = np.asarray(fir_result.get("input_audio", []), dtype=np.float64)
                        h_lem = np.asarray(fir_result["final_true_lem_ir"], dtype=np.float64)
                        y_noeq = np.convolve(x_in, h_lem, mode="full")[: len(x_in)]
                        save_wav(audio_out_dir / f"noEQ_{song}.wav", y_noeq.astype(np.float32), sr_audio)
                    audio_saved_keys.add(common_key)

                safe_algo = safe_token(algo_name)
                safe_tt = safe_token(str(tt).replace(".", "p"))
                eq_key = (safe_algo, "NA", safe_tt, song)
                if eq_key not in audio_saved_keys:
                    save_wav(
                        audio_out_dir / f"EQ_{safe_algo}_NA_tt{safe_tt}_{song}.wav",
                        np.asarray(fir_result.get("y_control", []), dtype=np.float32),
                        sr_audio,
                    )
                    audio_saved_keys.add(eq_key)

                if all(k in fir_result for k in ("final_ctrl_ir", "final_true_lem_ir")):
                    resp_f, resp_db = compute_exact_final_response_fir(
                        final_ctrl_ir=np.asarray(fir_result["final_ctrl_ir"], dtype=np.float32),
                        true_lem_ir=np.asarray(fir_result["final_true_lem_ir"], dtype=np.float32),
                        sr=sr_audio,
                        nfft=int(2 * frame_len - 1),
                    )
                else:
                    resp_f, resp_db = estimate_final_equalized_response(
                        input_audio=np.asarray(fir_result.get("input_audio", []), dtype=np.float64),
                        output_audio=np.asarray(fir_result.get("y_control", []), dtype=np.float64),
                        sr=sr_audio,
                        frame_len=frame_len,
                    )
                if resp_f.size and resp_db.size:
                    final_response_curves[key].append((resp_f, resp_db))

                stats = compute_time_stats[key]
                stats["total_time_s"] += float(fir_result.get("control_experiment_time_s", 0.0))
                stats["total_frames"] += int(fir_result.get("n_frames", 0))
                stats["num_runs"] += 1
                avg_pf = float(fir_result.get("avg_compute_time_per_frame_s", float("nan")))
                if np.isfinite(avg_pf):
                    stats["min_avg_time_per_frame_s"] = min(stats["min_avg_time_per_frame_s"], avg_pf)
                    stats["max_avg_time_per_frame_s"] = max(stats["max_avg_time_per_frame_s"], avg_pf)

                if target_response_example is None and fir_result.get("target_freq_axis") is not None:
                    target_response_example = {
                        "freq_axis": np.asarray(fir_result["target_freq_axis"], dtype=np.float32),
                        "target_mag_db": np.asarray(fir_result["target_mag_db"], dtype=np.float32),
                    }
                if true_lem_response_example is None and fir_result.get("true_lem_freq_axis") is not None:
                    true_lem_response_example = {
                        "freq_axis": np.asarray(fir_result["true_lem_freq_axis"], dtype=np.float32),
                        "lem_mag_db": np.asarray(fir_result["true_lem_mag_db"], dtype=np.float32),
                    }

    # -----------------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------------
    results_root = root / "results" / experiment_name
    results_root.mkdir(parents=True, exist_ok=True)

    with (results_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    plot_data = {
        "td_mse_curves":           dict(td_mse_curves),
        "validation_curves":       dict(validation_curves),
        "final_response_curves":   dict(final_response_curves),
        "compute_time_stats":      dict(compute_time_stats),
        "tt_transitions":          tt_transitions,
        "input_signals":           sorted(input_ids_used),
        "target_response_example": target_response_example,
        "true_lem_response_example": true_lem_response_example,
    }
    with (results_root / "plot_data.pkl").open("wb") as f:
        pickle.dump(plot_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved config:    {results_root / 'config.json'}")
    print(f"Saved plot data: {results_root / 'plot_data.pkl'}")
    print(f"Saved audio to:  {audio_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ablation study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "ablation_study_config.json",
        help="Path to the ablation study JSON config file.",
    )
    args = parser.parse_args()
    main(args.config)
