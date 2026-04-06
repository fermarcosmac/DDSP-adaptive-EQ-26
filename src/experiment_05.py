from __future__ import annotations

import sys
import json
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))

import numpy as np
import torch
import torchaudio

from lib.local_dasp_pytorch import signal as dasp_signal
from lib.local_dasp_pytorch.modules import ParametricEQ
from utils_ex04 import discover_input_signals, iter_param_grid, load_config, run_control_experiment
from utils_ex05 import build_proposed_sim_cfg, run_fir_baseline_experiment

root = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


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


def _framewise_mse(reference: np.ndarray, estimate: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
	n = min(len(reference), len(estimate))
	if n < frame_len:
		return np.array([], dtype=np.float64)
	n_frames = (n - frame_len) // hop_len + 1
	out = np.zeros(n_frames, dtype=np.float64)
	for k in range(n_frames):
		s = k * hop_len
		e = s + frame_len
		out[k] = float(np.mean((reference[s:e] - estimate[s:e]) ** 2))
	return out


def _estimate_final_equalized_response(
	input_audio: np.ndarray,
	output_audio: np.ndarray,
	sr: int,
	frame_len: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Estimate final equalized transfer magnitude (dB) from last analysis window."""
	x = np.asarray(input_audio, dtype=np.float64)
	y = np.asarray(output_audio, dtype=np.float64)
	x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
	y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
	n = min(len(x), len(y))
	if n < 8:
		return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

	win_len = min(n, int(frame_len))
	x_seg = x[n - win_len : n]
	y_seg = y[n - win_len : n]

	window = np.ones(win_len)
	xw = x_seg * window
	yw = y_seg * window

	x_peak = float(np.max(np.abs(xw))) if xw.size else 0.0
	y_peak = float(np.max(np.abs(yw))) if yw.size else 0.0
	if x_peak > 0:
		xw = xw / x_peak
	if y_peak > 0:
		yw = yw / y_peak

	nfft = int(2 ** np.ceil(np.log2(max(win_len, 512))))
	X = np.fft.rfft(xw, n=nfft)
	Y = np.fft.rfft(yw, n=nfft)
	H = Y / (X + 1e-12)
	freqs = np.fft.rfftfreq(nfft, d=1.0 / float(sr))
	mag_db = 20.0 * np.log10(np.abs(H) + 1e-12)
	mag_db = np.nan_to_num(mag_db, nan=-120.0, posinf=120.0, neginf=-120.0)
	mag_db = np.clip(mag_db, -120.0, 120.0)
	return freqs.astype(np.float64), mag_db.astype(np.float64)


def _compute_exact_final_response_proposed(
	final_eq_params_normalized: np.ndarray,
	final_gain_db: np.ndarray,
	true_lem_ir: np.ndarray,
	sr: int,
	nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute exact final compensated response: |H_eq * H_lem| in dB for proposed parametric EQ."""
	eq = ParametricEQ(sample_rate=int(sr))
	norm = np.asarray(final_eq_params_normalized, dtype=np.float32)
	if norm.ndim == 1:
		norm = norm[None, :]
	norm_t = torch.from_numpy(norm)

	with torch.no_grad():
		param_dict = eq.extract_param_dict(norm_t)
		p = eq.denormalize_param_dict(param_dict)

		sos = torch.zeros(1, 7, 6, dtype=torch.float32)
		b, a = dasp_signal.biquad(p["low_shelf_gain_db"], p["low_shelf_cutoff_freq"], p["low_shelf_q_factor"], float(sr), "low_shelf")
		sos[:, 0, :] = torch.cat((b, a), dim=-1)
		b, a = dasp_signal.biquad(p["band0_gain_db"], p["band0_cutoff_freq"], p["band0_q_factor"], float(sr), "peaking")
		sos[:, 1, :] = torch.cat((b, a), dim=-1)
		b, a = dasp_signal.biquad(p["band1_gain_db"], p["band1_cutoff_freq"], p["band1_q_factor"], float(sr), "peaking")
		sos[:, 2, :] = torch.cat((b, a), dim=-1)
		b, a = dasp_signal.biquad(p["band2_gain_db"], p["band2_cutoff_freq"], p["band2_q_factor"], float(sr), "peaking")
		sos[:, 3, :] = torch.cat((b, a), dim=-1)
		b, a = dasp_signal.biquad(p["band3_gain_db"], p["band3_cutoff_freq"], p["band3_q_factor"], float(sr), "peaking")
		sos[:, 4, :] = torch.cat((b, a), dim=-1)
		b, a = dasp_signal.biquad(p["band4_gain_db"], p["band4_cutoff_freq"], p["band4_q_factor"], float(sr), "peaking")
		sos[:, 5, :] = torch.cat((b, a), dim=-1)
		b, a = dasp_signal.biquad(p["high_shelf_gain_db"], p["high_shelf_cutoff_freq"], p["high_shelf_q_factor"], float(sr), "high_shelf")
		sos[:, 6, :] = torch.cat((b, a), dim=-1)

		h_eq = dasp_signal.fft_sosfreqz(sos, n_fft=int(nfft)).squeeze(0).detach().cpu().numpy()

	gain_db = float(np.asarray(final_gain_db, dtype=np.float64).reshape(-1)[0])
	h_eq = h_eq * (10.0 ** (gain_db / 20.0))

	lem = np.asarray(true_lem_ir, dtype=np.float64)
	h_lem = np.fft.rfft(lem, n=int(nfft))
	h_total = h_eq * h_lem
	freqs = np.fft.rfftfreq(int(nfft), d=1.0 / float(sr))
	mag_db = 20.0 * np.log10(np.abs(h_total) + 1e-12)
	mag_db = np.nan_to_num(mag_db, nan=-120.0, posinf=120.0, neginf=-120.0)
	mag_db = np.clip(mag_db, -120.0, 120.0)
	return freqs.astype(np.float64), mag_db.astype(np.float64)


def _compute_exact_final_response_fir(
	final_ctrl_ir: np.ndarray,
	true_lem_ir: np.ndarray,
	sr: int,
	nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute exact final compensated response: |H_fir * H_lem| in dB for FIR baselines."""
	ctrl = np.asarray(final_ctrl_ir, dtype=np.float64)
	lem = np.asarray(true_lem_ir, dtype=np.float64)
	h_ctrl = np.fft.rfft(ctrl, n=int(nfft))
	h_lem = np.fft.rfft(lem, n=int(nfft))
	h_total = h_ctrl * h_lem
	freqs = np.fft.rfftfreq(int(nfft), d=1.0 / float(sr))
	mag_db = 20.0 * np.log10(np.abs(h_total) + 1e-12)
	mag_db = np.nan_to_num(mag_db, nan=-120.0, posinf=120.0, neginf=-120.0)
	mag_db = np.clip(mag_db, -120.0, 120.0)
	return freqs.astype(np.float64), mag_db.astype(np.float64)


def _safe_token(s: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _save_wav(path: Path, arr: np.ndarray, sr: int) -> None:
	x = np.asarray(arr, dtype=np.float32)
	if x.size == 0:
		return
	peak = float(np.max(np.abs(x)))
	if peak > 0:
		x = x / peak
	t = torch.from_numpy(x).float().unsqueeze(0)
	torchaudio.save(str(path), t, int(sr))
	print(f"Saved audio: {path}")


def main() -> None:
	cfg = load_config(root / "configs" / "experiment_05_config.json")
	seed = int(cfg.get("seed", 124))
	set_seed(seed)

	experiment_name = cfg.get("experiment_name", "experiment_05")
	scenario = cfg.get("scenario", "moving_position")
	input_cfg = cfg.get("input", {})
	shared_grid = cfg.get("shared_simulation_params", {})
	proposed_cfgs = cfg.get("proposed_configs", [])
	fir_baselines_cfg = cfg.get("fir_baselines", {})

	if not shared_grid:
		raise ValueError("Missing 'shared_simulation_params' in experiment_05_config.json")

	rir_map = {
		"moving_position": root / "data" / "rir" / "moving_position",
		"moving_person": root / "data" / "rir" / "moving_person",
		"static": root / "data" / "rir" / "moving_position",
	}
	if scenario not in rir_map:
		raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {list(rir_map)}")
	rir_dir = rir_map[scenario]

	input_signals = discover_input_signals(input_cfg)
	if not input_signals:
		input_signals = [("white_noise", {"max_audio_len_s": 60.0})]

	td_mse_curves = defaultdict(list)
	validation_curves = defaultdict(list)
	compute_time_stats = defaultdict(
		lambda: {
			"total_time_s": 0.0,
			"total_frames": 0,
			"num_runs": 0,
			"min_avg_time_per_frame_s": float("inf"),
			"max_avg_time_per_frame_s": float("-inf"),
		}
	)
	tt_transitions = {}
	input_ids_used = set()
	final_response_curves = defaultdict(list)
	audio_out_dir = root / "data" / "audio" / "output" / experiment_name
	audio_out_dir.mkdir(parents=True, exist_ok=True)
	audio_saved_keys = set()

	# Optional metadata for plotting target response if available.
	target_response_example = None
	true_lem_response_example = None

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
			run_seed = seed
			if mode == "white_noise" and isinstance(info, dict):
				run_seed = seed + int(info.get("seed_offset", 0))
			set_seed(run_seed)

			if mode == "white_noise":
				input_id = f"white_noise_{int(info.get('realization_idx', 0))}"
			else:
				input_id = str(info.get("path", mode))
			input_ids_used.add(input_id)

			print("\n############################################")
			print(f"Combination {combo_idx}/{combo_total}")
			print(f"Scenario: {scenario}")
			print(f"Input: {_song_stem(input_spec)}")
			print("############################################")
			song = _song_stem(input_spec)

			# 1) Proposed framework configurations (experiment_04 controller)
			for proposed_entry in proposed_cfgs:
				label = str(proposed_entry.get("label", "Proposed"))
				sim_cfg = build_proposed_sim_cfg(sim_shared, proposed_entry)
				result = run_control_experiment(sim_cfg, input_spec)
				if result is None:
					continue

				td_curve = _framewise_mse(
					np.asarray(result["desired_audio"], dtype=np.float64),
					np.asarray(result["y_control"], dtype=np.float64),
					frame_len=frame_len,
					hop_len=hop_len,
				)

				t_axis = np.asarray(result.get("time_axis", []), dtype=np.float64)
				if td_curve.size and t_axis.size:
					min_len = min(len(td_curve), len(t_axis))
					td_curve = td_curve[:min_len]
					t_axis = t_axis[:min_len]

				v_curve = np.asarray(result.get("validation_error_history", []), dtype=np.float64)
				if t_axis.size and v_curve.size:
					min_len = min(len(t_axis), len(v_curve))
					t_axis = t_axis[:min_len]
					v_curve = v_curve[:min_len]

				key = (sim_cfg["transition_time_s"], f"Proposed:{label}")
				td_mse_curves[key].append((t_axis, td_curve))
				validation_curves[key].append((t_axis, v_curve))

				sr_audio = int(result.get("sr", sim_cfg.get("sr", 48000)))
				common_key = ("common", song)
				if common_key not in audio_saved_keys:
					_save_wav(audio_out_dir / f"desired_{song}.wav", np.asarray(result.get("desired_audio", []), dtype=np.float32), sr_audio)
					_save_wav(audio_out_dir / f"noEQ_{song}.wav", np.asarray(result.get("y_noEQ", []), dtype=np.float32), sr_audio)
					audio_saved_keys.add(common_key)

				safe_algo = _safe_token(f"Proposed_{label}")
				safe_lt = _safe_token(sim_cfg.get("loss_type", "NA")).replace("-", "_")
				safe_tt = _safe_token(str(sim_cfg["transition_time_s"]).replace(".", "p"))
				eq_key = (safe_algo, safe_lt, safe_tt, song)
				if eq_key not in audio_saved_keys:
					_save_wav(
						audio_out_dir / f"EQ_{safe_algo}_{safe_lt}_tt{safe_tt}_{song}.wav",
						np.asarray(result.get("y_control", []), dtype=np.float32),
						sr_audio,
					)
					audio_saved_keys.add(eq_key)

				if all(k in result for k in ("final_eq_params_normalized", "final_gain_db", "final_true_lem_ir")):
					resp_f, resp_db = _compute_exact_final_response_proposed(
						final_eq_params_normalized=np.asarray(result.get("final_eq_params_normalized"), dtype=np.float32),
						final_gain_db=np.asarray(result.get("final_gain_db"), dtype=np.float32),
						true_lem_ir=np.asarray(result.get("final_true_lem_ir"), dtype=np.float32),
						sr=int(result.get("sr", sim_cfg.get("sr", 48000))),
						nfft=int(2 * frame_len - 1),
					)
				else:
					resp_f, resp_db = _estimate_final_equalized_response(
						input_audio=np.asarray(result.get("input_audio", []), dtype=np.float64),
						output_audio=np.asarray(result.get("y_control", []), dtype=np.float64),
						sr=int(result.get("sr", sim_cfg.get("sr", 48000))),
						frame_len=frame_len,
					)
				if resp_f.size and resp_db.size:
					final_response_curves[key].append((resp_f, resp_db))

				ct_key = key
				stats = compute_time_stats[ct_key]
				stats["total_time_s"] += float(result.get("control_experiment_time_s", 0.0))
				stats["total_frames"] += int(result.get("n_frames", 0))
				stats["num_runs"] += 1
				avg_pf = float(result.get("avg_compute_time_per_frame_s", float("nan")))
				if np.isfinite(avg_pf):
					stats["min_avg_time_per_frame_s"] = min(stats["min_avg_time_per_frame_s"], avg_pf)
					stats["max_avg_time_per_frame_s"] = max(stats["max_avg_time_per_frame_s"], avg_pf)

				if key[0] not in tt_transitions:
					tt_transitions[key[0]] = result.get("transition_times", None)

			# 2) FIR baselines
			for algo_name in ("FxLMS", "FxFDAF"):
				algo_cfg = fir_baselines_cfg.get(algo_name, {})
				if not bool(algo_cfg.get("enabled", True)):
					continue

				fir_result = run_fir_baseline_experiment(sim_shared, input_spec, algorithm=algo_name, algo_cfg=algo_cfg)

				key = (fir_result["transition_time_s"], algo_name)
				t_axis = np.asarray(fir_result["time_axis"], dtype=np.float64)
				td_curve = _framewise_mse(
					np.asarray(fir_result.get("desired_audio", []), dtype=np.float64),
					np.asarray(fir_result.get("y_control", []), dtype=np.float64),
					frame_len=frame_len,
					hop_len=hop_len,
				)
				v_curve = np.asarray(fir_result["validation_error_history"], dtype=np.float64)

				if td_curve.size and t_axis.size:
					min_len = min(len(td_curve), len(t_axis))
					td_curve = td_curve[:min_len]
					t_axis = t_axis[:min_len]

				if t_axis.size and v_curve.size:
					min_len = min(len(t_axis), len(v_curve))
					t_axis = t_axis[:min_len]
					v_curve = v_curve[:min_len]

				td_mse_curves[key].append((t_axis, td_curve))
				validation_curves[key].append((t_axis, v_curve))

				sr_audio = int(fir_result.get("sr", sim_cfg.get("sr", 48000)))
				common_key = ("common", song)
				if common_key not in audio_saved_keys:
					_save_wav(audio_out_dir / f"desired_{song}.wav", np.asarray(fir_result.get("desired_audio", []), dtype=np.float32), sr_audio)
					if "final_true_lem_ir" in fir_result:
						x_in = np.asarray(fir_result.get("input_audio", []), dtype=np.float64)
						h_lem = np.asarray(fir_result.get("final_true_lem_ir", []), dtype=np.float64)
						y_noeq = np.convolve(x_in, h_lem, mode="full")[: len(x_in)]
						_save_wav(audio_out_dir / f"noEQ_{song}.wav", y_noeq.astype(np.float32), sr_audio)
					audio_saved_keys.add(common_key)

				safe_algo = _safe_token(algo_name)
				safe_lt = "NA"
				safe_tt = _safe_token(str(fir_result["transition_time_s"]).replace(".", "p"))
				eq_key = (safe_algo, safe_lt, safe_tt, song)
				if eq_key not in audio_saved_keys:
					_save_wav(
						audio_out_dir / f"EQ_{safe_algo}_{safe_lt}_tt{safe_tt}_{song}.wav",
						np.asarray(fir_result.get("y_control", []), dtype=np.float32),
						sr_audio,
					)
					audio_saved_keys.add(eq_key)

				if all(k in fir_result for k in ("final_ctrl_ir", "final_true_lem_ir")):
					resp_f, resp_db = _compute_exact_final_response_fir(
						final_ctrl_ir=np.asarray(fir_result.get("final_ctrl_ir"), dtype=np.float32),
						true_lem_ir=np.asarray(fir_result.get("final_true_lem_ir"), dtype=np.float32),
						sr=int(fir_result.get("sr", sim_cfg.get("sr", 48000))),
						nfft=int(2 * frame_len - 1),
					)
				else:
					resp_f, resp_db = _estimate_final_equalized_response(
						input_audio=np.asarray(fir_result.get("input_audio", []), dtype=np.float64),
						output_audio=np.asarray(fir_result.get("y_control", []), dtype=np.float64),
						sr=int(fir_result.get("sr", sim_cfg.get("sr", 48000))),
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

				if key[0] not in tt_transitions:
					tt_transitions[key[0]] = fir_result.get("transition_times", None)

				if target_response_example is None:
					target_response_example = {
						"freq_axis": np.asarray(fir_result.get("target_freq_axis", []), dtype=np.float32),
						"target_mag_db": np.asarray(fir_result.get("target_mag_db", []), dtype=np.float32),
					}

				if true_lem_response_example is None:
					true_lem_response_example = {
						"freq_axis": np.asarray(fir_result.get("true_lem_freq_axis", []), dtype=np.float32),
						"lem_mag_db": np.asarray(fir_result.get("true_lem_mag_db", []), dtype=np.float32),
					}

	results_root = root / "results" / experiment_name
	results_root.mkdir(parents=True, exist_ok=True)

	config_out = results_root / "config.json"
	with config_out.open("w", encoding="utf-8") as f:
		json.dump(cfg, f, indent=2)

	plot_data = {
		"td_mse_curves": dict(td_mse_curves),
		"validation_curves": dict(validation_curves),
		"final_response_curves": dict(final_response_curves),
		"compute_time_stats": dict(compute_time_stats),
		"tt_transitions": tt_transitions,
		"input_signals": sorted(input_ids_used),
		"target_response_example": target_response_example,
		"true_lem_response_example": true_lem_response_example,
	}
	out_path = results_root / "plot1_data.pkl"
	with out_path.open("wb") as f:
		pickle.dump(plot_data, f, protocol=pickle.HIGHEST_PROTOCOL)

	print(f"Saved config: {config_out}")
	print(f"Saved plot data: {out_path}")
	print(f"Saved audio directory: {audio_out_dir}")


if __name__ == "__main__":
	main()
