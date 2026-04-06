"""
Utility functions used exclusively by the ablation study script.

Covers:
- FIR baseline frame wrappers: FxLMS and FxFDAF
- Full FIR baseline experiment runner (run_fir_baseline_experiment)
- Simulation config builder for proposed framework configs (build_proposed_sim_cfg)
- RIR context preparation and input loading helpers
- Target response builder (numpy)
- Framewise TD-MSE computation
- Exact final equalized response computation (proposed EQ and FIR)
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from numpy.fft import irfft, rfft
from scipy.signal import fftconvolve, minimum_phase
from tqdm import tqdm

from local_pyaec.frequency_domain_adaptive_filters import fxfdaf as lib_fxfdaf
from local_pyaec.time_domain_adaptive_filters import fxlms as lib_fxlms
from local_dasp_pytorch import signal as dasp_signal
from local_dasp_pytorch.modules import ParametricEQ

from utils.common import (
    build_target_response_lin_phase,
    get_compensation_EQ_params,
    get_delay_from_ir,
    interp_to_log_freq,
    load_rirs,
    update_LEM,
)

# Repository root (three levels up: src/utils -> src -> repo root)
root = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def build_proposed_sim_cfg(
    shared_cfg: Dict[str, Any],
    proposed_entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Compose a run_control_experiment-compatible cfg from shared and per-entry config.

    Merges ``shared_cfg`` (from ``shared_simulation_params``) with the per-entry
    overrides in ``proposed_entry``, and fills in safe defaults for any required
    fields not present in either.

    Args:
        shared_cfg:     Shared simulation parameters (frame_len, hop_len, ROI, …).
        proposed_entry: Per-config overrides (optim_type, loss_type, mu_opt, …).

    Returns:
        A flat dict ready for ``run_control_experiment``.
    """
    sim = dict(shared_cfg)
    sim.update(proposed_entry)

    sim.setdefault("loss_type", "FD-MSE")
    sim.setdefault("optim_type", "SGD")
    sim.setdefault("mu_opt", 0.005)
    sim.setdefault("lambda_newton", 1.0)
    sim.setdefault("eps_0", 0.0)
    sim.setdefault("n_checkpoints", 0)
    sim.setdefault("use_true_LEM", False)

    return sim


# ---------------------------------------------------------------------------
# RIR context and input loading
# ---------------------------------------------------------------------------

def prepare_rir_context(
    sim_cfg: Dict[str, Any],
    max_audio_len_s: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Load and prepare RIR tensors and transition schedule.

    Args:
        sim_cfg:          Simulation config dict with ``n_rirs``, ``rir_dir``,
                          and ``transition_time_s``.
        max_audio_len_s:  Total audio duration in seconds (used to place transitions).
        device:           PyTorch device to place tensors on.

    Returns:
        Dict with keys ``sr``, ``rirs``, ``rirs_tensors``, ``transition_times_s``.

    Raises:
        ValueError: If no RIRs are found in ``rir_dir``.
    """
    n_rirs = int(sim_cfg["n_rirs"])
    rir_dir = Path(sim_cfg["rir_dir"])
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs, normalize=False)
    if not rirs:
        raise ValueError(f"No RIRs found in: {rir_dir}")

    sr = int(rirs_srs[0])
    rirs_tensors = [torch.from_numpy(r).float().to(device) for r in rirs]
    max_rir_len = max(r.shape[-1] for r in rirs_tensors)
    rirs_tensors = [F.pad(r, (0, max_rir_len - r.shape[-1])) for r in rirs_tensors]

    transition_times_s: List[Tuple[float, float]] = []
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


def load_input_for_sr(
    input_spec: Tuple[str, Dict[str, Any]],
    sr: int,
    device: torch.device,
) -> np.ndarray:
    """Load one input signal as a mono float64 numpy array at the requested sample rate.

    Args:
        input_spec: ``(mode, info)`` tuple. ``mode`` is ``"white_noise"`` or a
                    file-based mode. ``info`` dict must contain ``max_audio_len_s``
                    for white noise and ``path`` for file-based modes.
        sr:         Target sample rate in Hz.
        device:     PyTorch device used for resampling.

    Returns:
        1-D float64 numpy array, peak-normalised to ±1.

    Raises:
        ValueError: If white_noise mode is missing ``max_audio_len_s``.
        FileNotFoundError: If the audio file does not exist.
    """
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
            wav = wav[: int(float(max_audio_len_s) * sr)]
        x = wav

    peak = torch.max(torch.abs(x)).item()
    if peak > 0:
        x = x / peak
    return x.detach().cpu().numpy().astype(np.float64)


def build_target_response_np(
    rir_init: np.ndarray,
    sr: int,
    sim_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Build the target impulse response and magnitude response for FIR baselines.

    Mirrors the target construction in ``run_control_experiment`` but returns
    numpy arrays for use in the time-domain baseline loops.

    Args:
        rir_init:  Initial (first) RIR as a 1-D numpy array.
        sr:        Sample rate in Hz.
        sim_cfg:   Simulation config dict (needs ``ROI``, ``target_response_type``,
                   ``frame_len``).
        device:    PyTorch device used internally.

    Returns:
        Dict with keys ``h_target`` (numpy IR), ``freqs`` (frequency axis),
        and ``target_db`` (target magnitude in dB).
    """
    roi = tuple(sim_cfg["ROI"])
    target_response_type = sim_cfg["target_response_type"]

    lem_delay = get_delay_from_ir(rir_init, sr)
    eq_comp = get_compensation_EQ_params(rir_init, sr, ROI=roi, num_sections=7)
    target_mag_resp = eq_comp["target_response_db"]
    target_mag_freqs = eq_comp["freq_axis_smoothed"]

    h_target_t = build_target_response_lin_phase(
        sr=sr,
        response_type=target_response_type,
        target_mag_resp=target_mag_resp,
        target_mag_freqs=target_mag_freqs,
        fir_len=2048,
        ROI=roi,
        rolloff_octaves=0.5,
        device=device,
    )

    h_linear_np = h_target_t.squeeze().detach().cpu().numpy()
    h_min_np = minimum_phase(h_linear_np, method="homomorphic", half=False)
    h_target_np = (
        np.concatenate(
            [np.zeros(int(lem_delay), dtype=np.float64), h_min_np.astype(np.float64)]
        )
        / 2.0
    )

    nfft = int(2 * int(sim_cfg["frame_len"]) - 1)
    f = np.fft.rfftfreq(nfft, d=1.0 / sr)
    target_db = 20.0 * np.log10(np.abs(np.fft.rfft(h_target_np, n=nfft)) + 1e-8)

    return {"h_target": h_target_np, "freqs": f, "target_db": target_db}


# ---------------------------------------------------------------------------
# Validation error helper
# ---------------------------------------------------------------------------

def _smooth_log_mag(
    mag_db: np.ndarray,
    freqs: np.ndarray,
    roi: Tuple[float, float],
    n_points: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate magnitude to log frequency and apply moving-average smoothing."""
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


def compute_validation_error(
    w_ctrl: np.ndarray,
    lem_ir: np.ndarray,
    target_db: np.ndarray,
    freqs: np.ndarray,
    roi: Tuple[float, float],
    prev_total_db: Optional[np.ndarray],
    forget_factor: float,
) -> Tuple[float, np.ndarray]:
    """Compute the smoothed frequency-domain validation error for FIR baselines.

    Args:
        w_ctrl:        Current control filter coefficients.
        lem_ir:        Current estimated LEM impulse response.
        target_db:     Target magnitude response in dB (linear frequency axis).
        freqs:         Frequency axis matching ``target_db``.
        roi:           Region of interest ``(f_low, f_high)`` in Hz.
        prev_total_db: Previous frame's total response in dB (for EMA), or None.
        forget_factor: EMA weight for the current frame's contribution.

    Returns:
        Tuple of ``(normalised_validation_error, updated_total_db)``.
    """
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

    _, total_s = _smooth_log_mag(total_db, freqs, roi)
    _, target_s = _smooth_log_mag(target_db, freqs, roi)
    _, lem_s = _smooth_log_mag(lem_db, freqs, roi)

    num = float(np.mean(np.abs(total_s - target_s)))
    den = float(np.mean(np.abs(lem_s - target_s))) + 1e-12
    return num / den, total_db


# ---------------------------------------------------------------------------
# FIR baseline frame wrappers
# ---------------------------------------------------------------------------

def fxlms_frame(
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
    """Wrapper around local_pyaec FxLMS for one frame-sized block.

    Args:
        x_block:              Input frame (reference signal).
        d_block:              Desired frame (target response output).
        lem_ir:               True secondary path impulse response.
        h_hat:                Estimated secondary path (used for gradient).
        w:                    Current control filter coefficients.
        mu:                   Step size.
        u_state:              FxLMS internal buffer for filtered reference.
        u_f_state:            FxLMS internal buffer for filtered reference (freq).
        x_state:              FxLMS internal buffer for input.
        sec_state:            FxLMS internal buffer for secondary path output.
        primary_path_ir_len:  Max length of true path used for forward simulation.

    Returns:
        Tuple of ``(e, y_ctrl, y_out, w_new, u_state, u_f_state, x_state, sec_state)``.
    """
    lem_ir = np.asarray(lem_ir, dtype=np.float64).reshape(-1)
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


def fxfdaf_frame(
    x_block: np.ndarray,
    d_block: np.ndarray,
    h_hat: np.ndarray,
    w: np.ndarray,
    x_state: Optional[np.ndarray],
    mu: float,
    beta: float,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Wrapper around local_pyaec FxFDAF for one frame-sized block.

    Args:
        x_block:    Input frame.
        d_block:    Desired frame.
        h_hat:      Estimated secondary path impulse response.
        w:          Current control filter coefficients (time domain, length M).
        x_state:    FxFDAF internal state buffer (or None on first call).
        mu:         Step size.
        beta:       Power normalisation factor.
        block_size: Internal FxFDAF processing block size (≤ frame_len).

    Returns:
        Tuple of ``(e, y_ctrl, y_out, w_new, x_state)``.
    """
    m = int(block_size)
    n = min(len(x_block), len(d_block))
    if m <= 0:
        raise ValueError("FxFDAF block_size must be > 0")
    if n < m or (n % m) != 0:
        m = n

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

    if len(e) != len(d_block):
        e_aligned = np.zeros(len(d_block), dtype=np.float64)
        c = min(len(e), len(d_block))
        e_aligned[:c] = np.asarray(e[:c], dtype=np.float64)
        e = e_aligned

    y_out = d_block - e
    y_ctrl = y_out.copy()
    return e, y_ctrl, y_out, w_new, x_state


# ---------------------------------------------------------------------------
# Full FIR baseline experiment runner
# ---------------------------------------------------------------------------

def run_fir_baseline_experiment(
    sim_cfg: Dict[str, Any],
    input_spec: Tuple[str, Dict[str, Any]],
    algorithm: str,
    algo_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one FIR baseline (FxLMS or FxFDAF) and return per-frame result curves.

    Args:
        sim_cfg:    Shared simulation config (frame_len, hop_len, ROI, …).
        input_spec: ``(mode, info)`` input signal descriptor.
        algorithm:  ``"FxLMS"`` or ``"FxFDAF"``.
        algo_cfg:   Algorithm-specific hyperparameters (mu, filter_len, …).

    Returns:
        Dict containing time-series curves, audio arrays, and timing stats.

    Raises:
        ValueError: For unsupported algorithm names or config mismatches.
    """
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_len = int(sim_cfg["frame_len"])
    hop_len = int(sim_cfg["hop_len"])
    if frame_len != hop_len:
        raise ValueError("FIR baselines require frame_len == hop_len.")

    rir_ctx = prepare_rir_context(sim_cfg, float(input_spec[1]["max_audio_len_s"]), device)
    sr = int(rir_ctx["sr"])

    x = load_input_for_sr(input_spec, sr=sr, device=device)
    if x.size < frame_len:
        raise ValueError("Input signal shorter than frame_len.")

    target_ctx = build_target_response_np(rir_ctx["rirs"][0], sr, sim_cfg, device)
    h_target = target_ctx["h_target"]
    freqs = target_ctx["freqs"]
    target_db = target_ctx["target_db"]
    nfft = int((len(freqs) - 1) * 2)

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

    h_hat = np.asarray(rir_ctx["rirs"][0], dtype=np.float64)
    h_hat = h_hat[:48000]  # NOTE: hardcoded max h_hat length

    w = np.zeros(n_ctrl, dtype=np.float64)
    u_state = np.zeros(n_ctrl, dtype=np.float64)
    u_f_state = np.zeros(n_ctrl, dtype=np.float64)
    x_state_lms = np.zeros(len(h_hat), dtype=np.float64)
    sec_state = np.zeros(len(h_hat), dtype=np.float64)
    x_state_fdaf: Optional[np.ndarray] = None

    n_frames = (len(x) - frame_len) // hop_len + 1
    td_mse_history: List[float] = []
    val_history: List[float] = []
    prev_total_db: Optional[np.ndarray] = None

    y_control = np.zeros(len(x), dtype=np.float64)
    y_out_full = np.zeros(len(x), dtype=np.float64)

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
            e_fr, y_ctrl_fr, y_out_fr, w, u_state, u_f_state, x_state_lms, sec_state = fxlms_frame(
                x_fr, d_fr, lem_ir, h_hat, w, mu,
                u_state, u_f_state, x_state_lms, sec_state,
                primary_path_ir_len,
            )
        elif algorithm == "FxFDAF":
            e_fr, y_ctrl_fr, y_out_fr, w_fr, x_state_fdaf = fxfdaf_frame(
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
            raise ValueError(f"Unknown FIR algorithm: '{algorithm}'. Expected 'FxLMS' or 'FxFDAF'.")

        y_control[start:stop] = y_ctrl_fr
        y_out_full[start:stop] = y_out_fr

        td_mse = float(np.mean(np.square(e_fr)))
        val_err, prev_total_db = compute_validation_error(
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
        "y_out": y_out_full.astype(np.float64),
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


# ---------------------------------------------------------------------------
# TD-MSE and final-response helpers
# ---------------------------------------------------------------------------

def framewise_td_mse(
    reference: np.ndarray,
    estimate: np.ndarray,
    frame_len: int,
    hop_len: int,
) -> np.ndarray:
    """Compute per-frame time-domain MSE between reference and estimate arrays.

    Args:
        reference:  Ground-truth (desired) audio array.
        estimate:   Controlled output audio array.
        frame_len:  Analysis window length in samples.
        hop_len:    Hop between analysis windows in samples.

    Returns:
        1-D float64 array of per-frame MSE values.
    """
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


def compute_exact_final_response_proposed(
    final_eq_params_normalized: np.ndarray,
    final_gain_db: np.ndarray,
    true_lem_ir: np.ndarray,
    sr: int,
    nfft: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the exact final compensated response for the proposed parametric EQ.

    Evaluates ``|H_eq(ω) · H_lem(ω)|`` in dB using the biquad cascade
    corresponding to the final normalized EQ parameters.

    Args:
        final_eq_params_normalized: Normalized EQ parameter vector (shape ``(P,)``).
        final_gain_db:              Final gain value in dB (scalar or 1-element array).
        true_lem_ir:                True LEM impulse response at the last frame.
        sr:                         Sample rate in Hz.
        nfft:                       FFT size.

    Returns:
        Tuple ``(freqs, mag_db)`` — frequency axis and total response in dB.
    """
    eq = ParametricEQ(sample_rate=int(sr))
    norm = np.asarray(final_eq_params_normalized, dtype=np.float32)
    if norm.ndim == 1:
        norm = norm[None, :]
    norm_t = torch.from_numpy(norm)

    with torch.no_grad():
        param_dict = eq.extract_param_dict(norm_t)
        p = eq.denormalize_param_dict(param_dict)

        sos = torch.zeros(1, 7, 6, dtype=torch.float32)
        band_specs = [
            ("low_shelf", "low_shelf"),
            ("band0", "peaking"),
            ("band1", "peaking"),
            ("band2", "peaking"),
            ("band3", "peaking"),
            ("band4", "peaking"),
            ("high_shelf", "high_shelf"),
        ]
        for idx, (prefix, ftype) in enumerate(band_specs):
            b, a = dasp_signal.biquad(
                p[f"{prefix}_gain_db"],
                p[f"{prefix}_cutoff_freq"],
                p[f"{prefix}_q_factor"],
                float(sr),
                ftype,
            )
            sos[:, idx, :] = torch.cat((b, a), dim=-1)

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


def compute_exact_final_response_fir(
    final_ctrl_ir: np.ndarray,
    true_lem_ir: np.ndarray,
    sr: int,
    nfft: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the exact final compensated response for FIR baselines.

    Evaluates ``|H_fir(ω) · H_lem(ω)|`` in dB.

    Args:
        final_ctrl_ir:  Final FIR control filter coefficients.
        true_lem_ir:    True LEM impulse response at the last frame.
        sr:             Sample rate in Hz.
        nfft:           FFT size.

    Returns:
        Tuple ``(freqs, mag_db)`` — frequency axis and total response in dB.
    """
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


def estimate_final_equalized_response(
    input_audio: np.ndarray,
    output_audio: np.ndarray,
    sr: int,
    frame_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the final equalized transfer magnitude from the last analysis window.

    Used as a fallback when exact EQ parameters are not available.

    Args:
        input_audio:  Dry input signal.
        output_audio: Equalized output signal.
        sr:           Sample rate in Hz.
        frame_len:    Analysis window length in samples.

    Returns:
        Tuple ``(freqs, mag_db)`` or ``(empty, empty)`` if signals are too short.
    """
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

    x_peak = float(np.max(np.abs(x_seg))) if x_seg.size else 0.0
    y_peak = float(np.max(np.abs(y_seg))) if y_seg.size else 0.0
    if x_peak > 0:
        x_seg = x_seg / x_peak
    if y_peak > 0:
        y_seg = y_seg / y_peak

    nfft = int(2 ** np.ceil(np.log2(max(win_len, 512))))
    X = np.fft.rfft(x_seg, n=nfft)
    Y = np.fft.rfft(y_seg, n=nfft)
    H = Y / (X + 1e-12)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(sr))
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    mag_db = np.nan_to_num(mag_db, nan=-120.0, posinf=120.0, neginf=-120.0)
    mag_db = np.clip(mag_db, -120.0, 120.0)
    return freqs.astype(np.float64), mag_db.astype(np.float64)


# ---------------------------------------------------------------------------
# Filename helper
# ---------------------------------------------------------------------------

def safe_token(s: str) -> str:
    """Replace any non-alphanumeric characters (except ``._-``) with underscores."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
