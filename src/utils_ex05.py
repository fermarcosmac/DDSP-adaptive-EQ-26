from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.fft import rfft
from numpy.fft import irfft
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import fftconvolve, lfilter, minimum_phase
from tqdm import tqdm

from lib.local_pyaec.time_domain_adaptive_filters import fxlms as lib_fxlms
from lib.local_pyaec.frequency_domain_adaptive_filters import fxfdaf as lib_fxfdaf
from utils_ex04 import (
    build_target_response_lin_phase,
    discover_input_signals,
    get_compensation_EQ_params,
    get_delay_from_ir,
    interp_to_log_freq,
    iter_param_grid,
    load_config,
    load_rirs,
    run_control_experiment,
    update_LEM,
)

root = Path(__file__).resolve().parent.parent


def prepare_rir_context(sim_cfg: Dict[str, Any], max_audio_len_s: float, device: torch.device) -> Dict[str, Any]:
    """Load and prepare RIR tensors and transition schedule using experiment_04 logic."""
    n_rirs = int(sim_cfg["n_rirs"])
    rir_dir = Path(sim_cfg["rir_dir"])
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs, normalize=False)
    if not rirs:
        raise ValueError(f"No RIRs found in: {rir_dir}")

    sr = int(rirs_srs[0])
    rirs_tensors = [torch.from_numpy(r).float().to(device) for r in rirs]
    max_rir_len = max(r.shape[-1] for r in rirs_tensors)
    rirs_tensors = [F.pad(r, (0, max_rir_len - r.shape[-1])) for r in rirs_tensors]

    transition_times_s = []
    if n_rirs > 1:
        segment_duration_s = max_audio_len_s / n_rirs
        tt = float(sim_cfg["transition_time_s"])
        for i in range(1, n_rirs):
            t_start = i * segment_duration_s
            t_end = min(t_start + tt, max_audio_len_s)
            transition_times_s.append((float(t_start), float(t_end)))

    return {
        "sr": sr,
        "rirs": rirs,
        "rirs_tensors": rirs_tensors,
        "transition_times_s": transition_times_s,
    }


def load_input_for_sr(input_spec: Tuple[str, Dict[str, Any]], sr: int, device: torch.device) -> np.ndarray:
    """Load one input signal as mono float32 np array at requested sample rate."""
    mode, info = input_spec
    max_audio_len_s = info.get("max_audio_len_s", None)

    if mode == "white_noise":
        if max_audio_len_s is None:
            raise ValueError("white_noise requires max_audio_len_s in input spec")
        n = int(float(max_audio_len_s) * sr)
        x = torch.randn(n, device=device)
    else:
        audio_path = Path(info["path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        wav, wav_sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(wav_sr) != sr:
            wav = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=sr)(wav)
        wav = wav.squeeze(0).to(device)

        if max_audio_len_s is not None:
            max_samples = int(float(max_audio_len_s) * sr)
            wav = wav[:max_samples]
        x = wav

    peak = torch.max(torch.abs(x)).item()
    if peak > 0:
        x = x / peak
    return x.detach().cpu().numpy().astype(np.float64)


def build_target_response_np(rir_init: np.ndarray, sr: int, sim_cfg: Dict[str, Any], device: torch.device) -> Dict[str, np.ndarray]:
    """Build target impulse response and target magnitude response using experiment_04 utilities."""
    roi = tuple(sim_cfg["ROI"])
    target_response_type = sim_cfg["target_response_type"]

    lem_delay = get_delay_from_ir(rir_init, sr)
    eq_comp = get_compensation_EQ_params(rir_init, sr, ROI=roi, num_sections=7)
    target_mag_resp = eq_comp["target_response_db"]
    target_mag_freqs = eq_comp["freq_axis_smoothed"]

    h_target = build_target_response_lin_phase(
        sr=sr,
        response_type=target_response_type,
        target_mag_resp=target_mag_resp,
        target_mag_freqs=target_mag_freqs,
        fir_len=2048,
        ROI=roi,
        rolloff_octaves=0.5,
        device=device,
    )

    h_linear_np = h_target.squeeze().detach().cpu().numpy()
    h_min_np = minimum_phase(h_linear_np, method="homomorphic", half=False)
    h_target_np = np.concatenate([np.zeros(int(lem_delay), dtype=np.float64), h_min_np.astype(np.float64)]) / 2.0

    nfft = int(2 * int(sim_cfg["frame_len"]) - 1)
    f = np.fft.rfftfreq(nfft, d=1.0 / sr)
    target_db = 20.0 * np.log10(np.abs(np.fft.rfft(h_target_np, n=nfft)) + 1e-8)

    return {
        "h_target": h_target_np,
        "freqs": f,
        "target_db": target_db,
    }


def _smooth_log_mag_numpy(mag_db: np.ndarray, freqs: np.ndarray, roi: Tuple[float, float], n_points: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate to log frequency and apply moving-average smoothing."""
    mask = (freqs >= roi[0]) & (freqs <= roi[1])
    f_roi = torch.from_numpy(freqs[mask].astype(np.float32))
    mag_roi = torch.from_numpy(mag_db[mask].astype(np.float32))

    mag_log, f_log = interp_to_log_freq(mag_roi, f_roi, n_points=n_points)

    k = 15
    pad = k // 2
    ker = torch.ones(1, 1, k, dtype=torch.float32) / float(k)
    x = mag_log.view(1, 1, -1)
    x = F.pad(x, (pad, pad), mode="reflect")
    smoothed = F.conv1d(x, ker, padding=0).squeeze(0).squeeze(0)

    return f_log.cpu().numpy().astype(np.float64), smoothed.cpu().numpy().astype(np.float64)


def _compute_validation_error(
    w_ctrl: np.ndarray,
    lem_ir: np.ndarray,
    target_db: np.ndarray,
    freqs: np.ndarray,
    roi: Tuple[float, float],
    prev_total_db: np.ndarray | None,
    forget_factor: float,
) -> Tuple[float, np.ndarray]:
    """Compute smoothed frequency-domain validation error following experiment_04 style."""
    nfft = int((len(freqs) - 1) * 2)
    h_ctrl = np.fft.rfft(w_ctrl, n=nfft)
    h_lem = np.fft.rfft(lem_ir, n=nfft)
    h_total = h_ctrl * h_lem

    total_db_current = 20.0 * np.log10(np.abs(h_total) + 1e-8)
    if prev_total_db is None:
        total_db = total_db_current
    else:
        total_db = float(forget_factor) * total_db_current + (1.0 - float(forget_factor)) * prev_total_db

    lem_db = 20.0 * np.log10(np.abs(h_lem) + 1e-8)

    _, total_s = _smooth_log_mag_numpy(total_db, freqs, roi)
    _, target_s = _smooth_log_mag_numpy(target_db, freqs, roi)
    _, lem_s = _smooth_log_mag_numpy(lem_db, freqs, roi)

    num = float(np.mean(np.abs(total_s - target_s)))
    den = float(np.mean(np.abs(lem_s - target_s))) + 1e-12
    return num / den, total_db


def _fxlms_frame(
    x_block: np.ndarray,
    d_block: np.ndarray,
    lem_ir: np.ndarray,
    h_hat: np.ndarray,
    w: np.ndarray,
    mu: float,
    u_state: np.ndarray,
    u_f_state: np.ndarray,
    x_state: np.ndarray,
    sec_state: np.ndarray,
    primary_path_ir_len: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper around local_pyaec FxLMS for one frame-sized block."""
    lem_ir = np.asarray(lem_ir, dtype=np.float64).reshape(-1)
    # Truncate true path used in forward simulation to reduce per-frame cost.
    keep_len = max(1, int(primary_path_ir_len))
    lem_ir_sim = lem_ir[:keep_len]
    prev_sec_state = np.asarray(sec_state, dtype=np.float64).reshape(-1)

    e_model, w_new, u_state, u_f_state, x_state, sec_state, y_ctrl = lib_fxlms.fxlms(
        x_block,
        d_block,
        h_hat=h_hat,
        h_sec=lem_ir_sim,
        N=len(w),
        mu=mu,
        w_init=w,
        u_state=u_state,
        u_f_state=u_f_state,
        x_state=x_state,
        y_state=sec_state,
    )

    # Forward simulation through true secondary path via FFT convolution.
    l_sec = int(len(lem_ir_sim))
    if l_sec > 1:
        if prev_sec_state.size != l_sec:
            prev_sec_state = np.zeros(l_sec, dtype=np.float64)
        prev_tail = prev_sec_state[l_sec - 2 :: -1]
        y_in = np.concatenate([prev_tail, np.asarray(y_ctrl, dtype=np.float64)], axis=0)
        y_full = fftconvolve(y_in, lem_ir_sim, mode="full")
        start = l_sec - 1
        y_out = y_full[start : start + len(y_ctrl)]
    else:
        y_out = np.asarray(y_ctrl, dtype=np.float64).copy()

    e = np.asarray(d_block[: len(y_out)], dtype=np.float64) - np.asarray(y_out, dtype=np.float64)
    return e, y_ctrl, y_out, w_new, u_state, u_f_state, x_state, sec_state


def _fxfdaf_frame(
    x_block: np.ndarray,
    d_block: np.ndarray,
    h_hat: np.ndarray,
    w: np.ndarray,
    x_state: np.ndarray,
    mu: float,
    beta: float,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper around local_pyaec FxFDAF for one frame-sized block."""
    m = int(block_size)
    n = min(len(x_block), len(d_block))
    if m <= 0:
        raise ValueError("FxFDAF block_size must be > 0")
    if n < m or (n % m) != 0:
        # local_pyaec.fxfdaf processes floor(n/M) full blocks only.
        # Fallback to a single full-frame block to avoid tail-length shape mismatches.
        m = n

    # local_pyaec.fxfdaf uses rfft on 2*M buffers, so W must be length M+1.
    W = rfft(w, n=2 * m)
    e, W_new, x_state = lib_fxfdaf.fxfdaf(
        x_block,
        d_block,
        h_hat=h_hat,
        M=m,
        mu=mu,
        beta=beta,
        W_init=W,
        x_state=x_state,
    )
    w_new = irfft(W_new, n=2 * m)[:m]

    # Ensure error vector matches current frame length for stable downstream bookkeeping.
    if len(e) != len(d_block):
        e_aligned = np.zeros(len(d_block), dtype=np.float64)
        c = min(len(e), len(d_block))
        e_aligned[:c] = np.asarray(e[:c], dtype=np.float64)
        e = e_aligned

    y_out = d_block - e
    y_ctrl = y_out.copy()
    return e, y_ctrl, y_out, w_new, x_state

def run_fir_baseline_experiment(
    sim_cfg: Dict[str, Any],
    input_spec: Tuple[str, Dict[str, Any]],
    algorithm: str,
    algo_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one FIR baseline (FxLMS/FxFDAF) and return per-frame curves."""
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_len = int(sim_cfg["frame_len"])
    hop_len = int(sim_cfg["hop_len"])
    if frame_len != hop_len:
        raise ValueError("For current FIR baseline implementation, require frame_len == hop_len")

    rir_ctx = prepare_rir_context(sim_cfg, float(input_spec[1]["max_audio_len_s"]), device)
    sr = int(rir_ctx["sr"])

    x = load_input_for_sr(input_spec, sr=sr, device=device)
    if x.size < frame_len:
        raise ValueError("Input shorter than frame_len")

    target_ctx = build_target_response_np(rir_ctx["rirs"][0], sr, sim_cfg, device)
    h_target = target_ctx["h_target"]
    freqs = target_ctx["freqs"]
    target_db = target_ctx["target_db"]
    nfft = int((len(freqs) - 1) * 2)

    # Unprocessed (true) LEM response used as reference in result plots.
    true_lem_ir = np.asarray(rir_ctx["rirs"][0], dtype=np.float64)
    true_lem_mag_db = 20.0 * np.log10(np.abs(np.fft.rfft(true_lem_ir, n=nfft)) + 1e-8)

    d_full = np.convolve(x, h_target, mode="full")[: len(x)]

    mu = float(algo_cfg.get("mu", 0.01))
    beta = float(algo_cfg.get("beta", 0.9))
    primary_path_ir_len = int(algo_cfg.get("primary_path_ir_len", 20000))
    n_ctrl = int(algo_cfg.get("filter_len", frame_len))
    n_ctrl = max(8, min(n_ctrl, frame_len * 2))
    fdaf_block_size = int(algo_cfg.get("block_size", n_ctrl))
    fdaf_block_size = max(8, min(fdaf_block_size, frame_len))

    # TODO: estimate h_hat from input/output data
    h_hat = np.asarray(rir_ctx["rirs"][0], dtype=np.float64)
    h_hat = h_hat[:48000] # TODO: hardcoded!
    # max_h_hat_len = int(algo_cfg.get("h_hat_len", 0))
    # if max_h_hat_len > 0:
    #     h_hat = h_hat[:max_h_hat_len]
    # if h_hat.ndim != 1:
    #     h_hat = h_hat.reshape(-1)

    # Initialize control filter and state buffers for adaptive algorithm.
    w = np.zeros(n_ctrl, dtype=np.float64) # control filter coefficients
    u_state = np.zeros(n_ctrl, dtype=np.float64)
    u_f_state = np.zeros(n_ctrl, dtype=np.float64)
    x_state = np.zeros(len(h_hat), dtype=np.float64)
    sec_state = np.zeros(len(h_hat), dtype=np.float64)

    x_state_fdaf = None # separate state for FxFDAF since it processes blocks instead of samples

    n_frames = (len(x) - frame_len) // hop_len + 1
    td_mse_history: List[float] = []
    val_history: List[float] = []
    prev_total_db = None

    y_control = np.zeros(len(x), dtype=np.float64)
    y_out = np.zeros(len(x), dtype=np.float64)

    roi = tuple(sim_cfg["ROI"])
    forget_factor = float(sim_cfg["forget_factor"])

    transition_times = rir_ctx["transition_times_s"] if int(sim_cfg["n_rirs"]) > 1 else None

    for k in tqdm(range(n_frames), desc=f"{algorithm} baseline", unit="frame"):
        start = k * hop_len
        stop = start + frame_len
        now_s = float(start / sr)

        lem_t = update_LEM(now_s, int(sim_cfg["n_rirs"]), transition_times or [], rir_ctx["rirs_tensors"])
        lem_ir = lem_t.squeeze().detach().cpu().numpy().astype(np.float64)

        x_fr = x[start:stop]
        d_fr = d_full[start:stop]

        if algorithm == "FxLMS":
            e_fr, y_ctrl_fr, y_out_fr, w, u_state, u_f_state, x_state, sec_state = _fxlms_frame(
                x_fr,
                d_fr,
                lem_ir,
                h_hat,
                w,
                mu,
                u_state,
                u_f_state,
                x_state,
                sec_state,
                primary_path_ir_len,
            )
        elif algorithm == "FxFDAF":
            e_fr, y_ctrl_fr, y_out_fr, w_fr, x_state_fdaf = _fxfdaf_frame(
                x_block=x_fr,
                d_block=d_fr,
                h_hat=h_hat,
                w=w,
                x_state=x_state_fdaf,
                mu=mu,
                beta=beta,
                block_size=fdaf_block_size,
            )
            w = np.asarray(w, dtype=np.float64)
            w[: min(len(w), len(w_fr))] = np.asarray(w_fr[: min(len(w), len(w_fr))], dtype=np.float64)
        else:
            raise ValueError(f"Unknown FIR algorithm: {algorithm}")

        y_control[start:stop] = y_ctrl_fr
        y_out[start:stop] = y_out_fr

        td_mse = float(np.mean(np.square(e_fr)))
        val_err, prev_total_db = _compute_validation_error(
            w_ctrl=w,
            lem_ir=lem_ir,
            target_db=target_db,
            freqs=freqs,
            roi=roi,
            prev_total_db=prev_total_db,
            forget_factor=forget_factor,
        )

        td_mse_history.append(td_mse)
        val_history.append(val_err)

    elapsed = float(time.perf_counter() - t0)
    time_axis = np.arange(n_frames, dtype=np.float64) * (hop_len / sr)

    return {
        "algorithm": algorithm,
        "time_axis": time_axis,
        "td_mse_history": np.asarray(td_mse_history, dtype=np.float64),
        "validation_error_history": np.asarray(val_history, dtype=np.float64),
        "transition_times": transition_times,
        "transition_time_s": float(sim_cfg["transition_time_s"]),
        "input_audio": x.astype(np.float32),
        "desired_audio": d_full.astype(np.float32),
        "y_control": y_control.astype(np.float64),
        "y_out": y_out.astype(np.float64),
        "target_freq_axis": freqs.astype(np.float32),
        "target_mag_db": target_db.astype(np.float32),
        "true_lem_freq_axis": freqs.astype(np.float32),
        "true_lem_mag_db": true_lem_mag_db.astype(np.float32),
        "sr": sr,
        "n_frames": int(n_frames),
        "control_experiment_time_s": elapsed,
        "avg_compute_time_per_frame_s": elapsed / n_frames if n_frames > 0 else float("nan"),
        "final_ctrl_ir": np.asarray(w, dtype=np.float32),
        "final_true_lem_ir": np.asarray(lem_ir, dtype=np.float32),
    }


def build_proposed_sim_cfg(shared_cfg: Dict[str, Any], proposed_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Compose a run_control_experiment-compatible cfg from shared + entry config."""
    sim = dict(shared_cfg)
    sim.update(proposed_entry)

    sim.setdefault("loss_type", "FD-MSE")
    sim.setdefault("optim_type", "SGD")
    sim.setdefault("mu_opt", 0.005)
    sim.setdefault("lambda_newton", 1.0)
    sim.setdefault("eps_0", 0.0)
    sim.setdefault("n_checkpoints", 0)
    sim.setdefault("use_true_LEM", False)
    sim.setdefault("debug_plot", False)

    return sim
