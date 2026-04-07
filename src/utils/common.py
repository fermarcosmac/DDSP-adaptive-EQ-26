"""
Common utilities shared across all experiment scripts and notebooks.

Covers:
- Global seeding for reproducibility
- Signal processing: RIR analysis, EQ optimisation, target response building,
  frequency-domain smoothing, Kirkeby deconvolution, LEM interpolation
- I/O helpers: load RIRs, load config, discover input signals
- Core experiment: parameter-to-loss mapping, per-frame buffer processing,
  and the full adaptive-EQ control loop (run_control_experiment)
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _configure_windows_cuda_dll_search_paths() -> None:
    """Ensure CUDA/NVRTC DLL directories are discoverable on Windows.

    VS Code debug launches can use the selected interpreter without a fully
    activated conda PATH, which may hide nvrtc-builtins64_121.dll from the
    loader. Registering likely env DLL folders makes the behavior deterministic.
    """
    if os.name != "nt":
        return

    env_roots = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env_roots.append(Path(conda_prefix))
    env_roots.append(Path(sys.prefix))

    candidates: List[Path] = []
    for root_dir in env_roots:
        candidates.extend([
            root_dir / "bin",
            root_dir / "Library" / "bin",
            root_dir / "Scripts",
            root_dir,
        ])

    seen: set[str] = set()
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(";") if current_path else []

    for p in candidates:
        p_resolved = str(p)
        p_key = p_resolved.lower()
        if p_key in seen or not p.is_dir():
            continue
        seen.add(p_key)

        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(p_resolved)
            except OSError:
                pass

        # Keep PATH in sync for subprocesses and loaders that still consult PATH.
        if all(existing.lower() != p_key for existing in path_entries):
            path_entries.insert(0, p_resolved)

    os.environ["PATH"] = ";".join(path_entries)


_configure_windows_cuda_dll_search_paths()

import torch
import torch.nn.functional as F
import torchaudio
from scipy.optimize import least_squares
from scipy.signal import minimum_phase
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    sf = None

from torch.func import jacfwd, jacrev
from torch.linalg import lstsq

from modules.modules import LEMConv, Ridge
from local_dasp_pytorch.modules import Gain, ParametricEQ

# Repository root (two levels above this file: src/utils -> src -> repo root)
root = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Signal processing utilities
# ---------------------------------------------------------------------------

def next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 that is >= x.

    Args:
        x: A positive integer.

    Returns:
        Smallest power of 2 >= x.
    """
    return 1 << (x - 1).bit_length()


def get_delay_from_ir(rir: np.ndarray, sr: int) -> int:
    """Estimate the direct-sound delay from an RIR by locating the peak.

    Args:
        rir: 1-D numpy array containing the room impulse response.
        sr:  Sample rate in Hz (not used numerically, kept for interface consistency).

    Returns:
        Estimated delay in samples (index of the absolute maximum).
    """
    return int(np.argmax(np.abs(rir)))


def biquad_coefficients(
    gain_db: float,
    fc: float,
    Q: float,
    Fs: float,
    filter_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RBJ cookbook biquad coefficients.

    Args:
        gain_db:     Filter gain in dB (used by peaking and shelf types).
        fc:          Centre / corner frequency in Hz.
        Q:           Quality factor (or shelf slope for shelf types).
        Fs:          Sample rate in Hz.
        filter_type: One of "peaking", "low_shelf", "high_shelf".

    Returns:
        (b, a): Numerator and denominator coefficient arrays, normalised so a[0] = 1.
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / Fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * Q)

    if filter_type == "peaking":
        b0 = 1 + alpha * A;  b1 = -2 * cos_w0;  b2 = 1 - alpha * A
        a0 = 1 + alpha / A;  a1 = -2 * cos_w0;  a2 = 1 - alpha / A

    elif filter_type == "low_shelf":
        sqrtA = np.sqrt(A)
        two_sqrtA_alpha = 2 * sqrtA * alpha
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha

    elif filter_type == "high_shelf":
        sqrtA = np.sqrt(A)
        two_sqrtA_alpha = 2 * sqrtA * alpha
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha

    else:
        raise ValueError(f"Unknown filter_type: '{filter_type}'")

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a


def sosfreqz_np(sos: np.ndarray, worN: np.ndarray) -> np.ndarray:
    """Complex frequency response of a second-order section (SOS) cascade.

    Args:
        sos:  (n_sections, 6) array — [b0 b1 b2 a0 a1 a2] per row.
        worN: Digital radian frequencies at which to evaluate.

    Returns:
        Complex frequency response array of shape (len(worN),).
    """
    z = np.exp(1j * worN)
    H = np.ones_like(z, dtype=np.complex128)
    for b0, b1, b2, a0, a1, a2 in sos:
        H *= (b0 + b1 / z + b2 / z**2) / (a0 + a1 / z + a2 / z**2)
    return H


def compute_parametric_eq_response(params: np.ndarray, f: np.ndarray, Fs: float) -> np.ndarray:
    """Compute the total EQ magnitude response (dB) for a biquad SOS cascade.

    Args:
        params: (num_filters, 3) array with columns [gain_dB, Q_or_slope, Fc].
                Rows 0 and -1 are treated as low- and high-shelf; all others as peaking.
        f:      Frequency vector in Hz.
        Fs:     Sample rate in Hz.

    Returns:
        Magnitude response in dB, shape (len(f),).
    """
    num_filters = params.shape[0]
    sos = np.zeros((num_filters, 6))
    for i, (gain_db, Q, Fc) in enumerate(params):
        ftype = "low_shelf" if i == 0 else ("high_shelf" if i == num_filters - 1 else "peaking")
        b, a = biquad_coefficients(gain_db=gain_db, fc=Fc, Q=Q, Fs=Fs, filter_type=ftype)
        sos[i, :] = np.hstack([b, a])

    worN = 2 * np.pi * f / Fs
    H = sosfreqz_np(sos, worN)
    return 20.0 * np.log10(np.abs(H) + 1e-12)


def _octave_average(
    f: np.ndarray,
    resp: np.ndarray,
    bpo: int,
    freq_range: Tuple[float, float] = None,
    b_smooth: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average a magnitude response over fractional-octave bands.

    Uses Bark-scale constant bandwidth below ~300 Hz and octave spacing above.

    Args:
        f:          Frequency vector in Hz.
        resp:       Magnitude response (linear, not dB).
        bpo:        Bands per octave (e.g. 24 for 1/24-octave).
        freq_range: Optional (fmin, fmax) Hz tuple to restrict analysis.
        b_smooth:   Apply a light 3- and 5-point smoothing pass.

    Returns:
        (oa, cf): Octave-averaged magnitudes and corresponding centre frequencies.
    """
    f = np.atleast_1d(f).astype(float)
    resp = np.atleast_1d(resp).astype(float)

    if freq_range is not None:
        idx = (f >= freq_range[0]) & (f <= freq_range[1])
        f, resp = f[idx], resp[idx]
    if len(f) == 0:
        return np.array([]), np.array([])

    G = 10 ** (3 / 10)
    ref_freq = f[0]
    lgbg = bpo / np.log(G)
    octave_cutoff = 300

    f_min_idx = int(np.round(lgbg * np.log(octave_cutoff / ref_freq)))
    f_max_idx = int(np.floor(lgbg * np.log(f[-1] / ref_freq)))
    cf = ref_freq * (G ** (np.arange(f_min_idx, f_max_idx + 1) / bpo))

    last_cst_cf = ref_freq * (G ** (f_min_idx / bpo))
    lf_bw = ref_freq * (G ** (f_min_idx / bpo) - G ** ((f_min_idx - 1) / bpo))
    nb_cst = 1 + int(np.ceil((last_cst_cf - f[0]) / lf_bw))
    if nb_cst > 1:
        lf_cf = np.linspace(f[0], last_cst_cf, nb_cst)
        cf = np.concatenate([lf_cf[:-1], cf])

    bef = np.concatenate([cf[:-1] * (G ** (1 / (2 * bpo))), [f[-1]]])
    oa = np.full(len(cf), np.nan)

    f_beg = 0
    for ii in range(len(cf)):
        f_end_idx = np.where(f <= bef[ii])[0]
        if len(f_end_idx) > 0:
            f_end = f_end_idx[-1] + 1
            oa[ii] = np.nanmean(resp[f_beg:f_end])
            f_beg = f_end

    # Fill NaN gaps (octave bandwidth wider than available frequency points)
    for ii in np.where(~np.isnan(oa[:-1]) & np.isnan(oa[1:]))[0]:
        f_end_idx = np.where(~np.isnan(oa[ii + 1 :]))[0]
        if len(f_end_idx) > 0:
            f_end = f_end_idx[0]
            cf[ii] = np.prod(cf[ii : ii + f_end]) ** (1 / f_end)

    mask = ~np.isnan(oa)
    oa, cf = oa[mask], cf[mask]

    if b_smooth and len(oa) > 4:
        oa[1:-1] = (oa[1:-1] + oa[:-2] + oa[2:]) / 3
        oa[2:-2] = (2 * oa[2:-2] + oa[:-4] + oa[4:]) / 4

    return oa, cf


def octave_average_torch(
    f: torch.Tensor,
    resp: torch.Tensor,
    bpo: int,
    freq_range: Tuple[float, float] = None,
    b_smooth: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch version of fractional-octave averaging (preserves complex values).

    Args:
        f:          Frequency vector (Hz) as a 1-D tensor.
        resp:       Response tensor (real or complex), same length as f.
        bpo:        Bands per octave.
        freq_range: Optional (fmin, fmax) Hz tuple.
        b_smooth:   Apply smoothing passes to the output.

    Returns:
        (resp_averaged, cf): Averaged response and centre-frequency tensors.
    """
    device = f.device
    dtype = resp.dtype

    if freq_range is not None:
        mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f, resp = f[mask], resp[mask]
    if len(f) == 0:
        return torch.tensor([], device=device, dtype=dtype), torch.tensor([], device=device)

    G = 10 ** (3 / 10)
    ref_freq = f[0].item() + 1e-6
    lgbg = bpo / np.log(G)

    f_min_idx = int(np.round(lgbg * np.log(300 / ref_freq)))
    f_max_idx = int(np.floor(lgbg * np.log(f[-1].item() / ref_freq)))
    cf = torch.tensor(ref_freq * (G ** (np.arange(f_min_idx, f_max_idx + 1) / bpo)), device=device)

    last_cst_cf = ref_freq * (G ** (f_min_idx / bpo))
    lf_bw = ref_freq * (G ** (f_min_idx / bpo) - G ** ((f_min_idx - 1) / bpo))
    nb_cst = 1 + int(np.ceil((last_cst_cf - f[0].item()) / lf_bw))
    if nb_cst > 1:
        lf_cf = torch.linspace(f[0].item(), last_cst_cf, nb_cst, device=device)
        cf = torch.cat([lf_cf[:-1], cf])

    bef = torch.cat([cf[:-1] * (G ** (1 / (2 * bpo))), torch.tensor([f[-1].item()], device=device)])
    resp_averaged = torch.full((len(cf),), torch.nan, device=device, dtype=dtype)

    f_beg = 0
    for ii in range(len(cf)):
        f_end_idx = torch.where(f <= bef[ii])[0]
        if len(f_end_idx) > 0:
            f_end = f_end_idx[-1] + 1
            resp_averaged[ii] = torch.nanmean(resp[f_beg:f_end])
            f_beg = f_end

    mask = ~torch.isnan(resp_averaged)
    resp_averaged, cf = resp_averaged[mask], cf[mask]

    if b_smooth and len(resp_averaged) > 4:
        resp_averaged[1:-1] = (resp_averaged[1:-1] + resp_averaged[:-2] + resp_averaged[2:]) / 3
        resp_averaged[2:-2] = (2 * resp_averaged[2:-2] + resp_averaged[:-4] + resp_averaged[4:]) / 4

    return resp_averaged, cf


def get_compensation_EQ_params(
    rir: np.ndarray,
    sr: int,
    ROI: Tuple[float, float] = (20.0, 20000.0),
    num_sections: int = 7,
) -> dict:
    """Estimate parametric EQ parameters that compensate for the given RIR.

    Octave-averages the RIR magnitude response, fits a linear target in
    log-frequency space, then runs a least-squares EQ optimisation.

    Args:
        rir:          1-D numpy array with the room impulse response.
        sr:           Sample rate in Hz.
        ROI:          Region of interest (fmin, fmax) in Hz.
        num_sections: Number of biquad EQ sections (default 7).

    Returns:
        Dictionary with keys: "eq_params" (named dict), "EQ_matrix" (7×3 array),
        "measured_response_db", "freq_axis_smoothed", "target_response_db",
        "equalized_response_db", "filter_response_db", "frq_axis_full".
    """
    nfft = len(rir)
    freq_response = np.abs(np.fft.rfft(rir, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1 / sr)

    oa, cf = _octave_average(freqs, freq_response, bpo=24, freq_range=ROI, b_smooth=False)
    target_resp, _, pdb = _get_target_response_comp_EQ(cf, oa, ROI)

    print(f"Starting EQ optimisation: {num_sections} filters, ROI {ROI[0]}–{ROI[1]} Hz …")
    EQ_params, out_resp_db, filt_resp_db = _eq_optimizer(num_sections, cf, pdb, target_resp, ROI, sr)

    return {
        "eq_params": _seven_biquad_eq_params_to_dict(EQ_params),
        "EQ_matrix": EQ_params,
        "measured_response_db": pdb,
        "frq_axis_full": freqs,
        "target_response_db": target_resp,
        "equalized_response_db": out_resp_db,
        "filter_response_db": filt_resp_db,
        "freq_axis_smoothed": cf,
    }


def build_target_response_lin_phase(
    sr: int,
    response_type: str = "delay_only",
    target_mag_resp: np.ndarray = None,
    target_mag_freqs: np.ndarray = None,
    fir_len: int = 1024,
    ROI: tuple = None,
    rolloff_octaves: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Build a linear-phase target FIR with optional magnitude shaping.

    Args:
        sr:               Sample rate in Hz.
        response_type:    "delay_only" (Kronecker delta) or "delay_and_mag".
        target_mag_resp:  Target magnitude response in dB (required for "delay_and_mag").
        target_mag_freqs: Frequency axis for target_mag_resp in Hz.
        fir_len:          FIR filter length.
        ROI:              Region of interest (fmin, fmax) Hz; outside is attenuated.
        rolloff_octaves:  Cosine rolloff width in octaves at ROI boundaries.
        device:           Torch device for output tensor.

    Returns:
        Target impulse response tensor of shape (1, 1, fir_len).
    """
    if device is None:
        device = torch.device("cpu")

    if response_type == "delay_only":
        h = torch.zeros(1, 1, fir_len, device=device)
        h[:, :, 0] = 1.0
        return h

    if response_type != "delay_and_mag":
        raise ValueError(f"Unknown response_type: '{response_type}'. Use 'delay_only' or 'delay_and_mag'.")

    if target_mag_resp is None or target_mag_freqs is None:
        raise ValueError("target_mag_resp and target_mag_freqs are required for 'delay_and_mag'.")

    fft_freqs = np.fft.rfftfreq(fir_len, d=1.0 / sr)
    n_fft = len(fft_freqs)
    log_freqs = np.logspace(np.log10(fft_freqs[1]), np.log10(fft_freqs[-1]), n_fft)
    mag_db = np.interp(log_freqs, target_mag_freqs, target_mag_resp)

    if ROI is not None:
        f_low, f_high = ROI
        f_low_end = f_low / (2 ** rolloff_octaves)
        f_high_end = f_high * (2 ** rolloff_octaves)
        rolloff = np.zeros_like(log_freqs)
        for i, f in enumerate(log_freqs):
            if f < f_low_end:
                rolloff[i] = -120.0
            elif f < f_low:
                t = np.log2(f_low / f) / rolloff_octaves
                rolloff[i] = -120.0 * 0.5 * (1 - np.cos(np.pi * t))
            elif f > f_high_end:
                rolloff[i] = -120.0
            elif f > f_high:
                t = np.log2(f / f_high) / rolloff_octaves
                rolloff[i] = -120.0 * 0.5 * (1 - np.cos(np.pi * t))
        mag_db = mag_db + rolloff

    mag_db = np.interp(fft_freqs, log_freqs, mag_db)
    H_mag = torch.from_numpy(10 ** (mag_db / 20.0)).float().to(device)

    h_zerophase = torch.fft.irfft(H_mag, n=fir_len)
    h_causal = torch.roll(h_zerophase, fir_len // 2)
    h_windowed = h_causal * torch.hann_window(fir_len, device=device)
    h_windowed = h_windowed / h_windowed.abs().max()
    return h_windowed.view(1, 1, -1)


def interp_to_log_freq(
    mag_db: torch.Tensor,
    freqs_lin: torch.Tensor,
    n_points: int = None,
    f_min: float = None,
    f_max: float = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate a magnitude response from linear to log-spaced frequency bins.

    Enables perceptually uniform processing: lower frequencies receive more
    detail, higher frequencies less, matching human hearing.

    Args:
        mag_db:    Magnitude response in dB on linear frequency bins.
        freqs_lin: Corresponding linear frequency axis as a 1-D tensor.
        n_points:  Number of log-spaced output points (default: same as input).
        f_min:     Minimum output frequency (default: first non-zero bin).
        f_max:     Maximum output frequency (default: last bin).

    Returns:
        (mag_db_log, freqs_log): Interpolated response and log frequency axis.
    """
    if n_points is None:
        n_points = len(freqs_lin)
    if f_min is None:
        f_min = freqs_lin[freqs_lin > 0][0].item()
    if f_max is None:
        f_max = freqs_lin[-1].item()

    freqs_log = torch.logspace(
        torch.log10(torch.tensor(f_min, device=mag_db.device)),
        torch.log10(torch.tensor(f_max, device=mag_db.device)),
        n_points,
        device=mag_db.device,
    )

    indices = torch.searchsorted(freqs_lin.contiguous(), freqs_log.contiguous())
    indices = torch.clamp(indices, 1, len(freqs_lin) - 1)
    idx_low = indices - 1
    idx_high = indices

    weights = (freqs_log - freqs_lin[idx_low]) / (freqs_lin[idx_high] - freqs_lin[idx_low] + 1e-10)
    weights = torch.clamp(weights, 0, 1)
    mag_db_log = mag_db[idx_low] + weights * (mag_db[idx_high] - mag_db[idx_low])
    return mag_db_log, freqs_log


def _movmean_1d(x: torch.Tensor, L_before: int, L_after: int) -> torch.Tensor:
    """1-D moving average with an asymmetric window and edge-aware normalisation.

    Args:
        x:        1-D tensor.
        L_before: Number of samples to look back.
        L_after:  Number of samples to look ahead.

    Returns:
        Smoothed tensor of the same length.
    """
    x = x.view(-1)
    N = x.numel()
    if N == 0:
        return x.clone()
    L_before = max(int(L_before), 0)
    L_after = max(int(L_after), 0)

    cumsum = torch.cumsum(F.pad(x, (1, 0)), dim=0)
    idx = torch.arange(N, device=x.device)
    left = torch.clamp(idx - L_before, 0, N - 1)
    right = torch.clamp(idx + L_after, 0, N - 1)
    seg_sum = cumsum[right + 1] - cumsum[left]
    count = (right - left + 1).to(x.dtype)
    return seg_sum / count


def kirkeby_deconvolve(
    x: torch.Tensor,
    y: torch.Tensor,
    nfft: int,
    sr: int,
    ROI: Tuple[float, float],
) -> torch.Tensor:
    """Kirkeby deconvolution: stable inverse filter in the frequency domain.

    Estimates H = Y / X with a frequency-dependent regularisation term that
    limits amplification outside the region of interest.

    Args:
        x:    Input (excitation) signal, 1-D tensor.
        y:    Output signal, 1-D tensor.
        nfft: FFT size (should be >= len(x) + len(y) - 1).
        sr:   Sample rate in Hz.
        ROI:  (f_low, f_high) region of interest in Hz.

    Returns:
        Complex frequency response H of length nfft//2 + 1.
    """
    X = torch.fft.rfft(x, n=nfft)
    Y = torch.fft.rfft(y, n=nfft)
    N = X.shape[-1]
    device, dtype_real = X.device, X.real.dtype

    Xflat = torch.zeros(N, device=device, dtype=dtype_real)
    if ROI is not None:
        f0_bin = max(0, min(int(math.ceil((ROI[0] / sr) * nfft)), N - 1))
        f1_bin = max(0, min(int(math.floor((ROI[1] / sr) * nfft)), N - 1))
        if f1_bin >= f0_bin:
            Xflat[f0_bin : f1_bin + 1] = 1.0
    else:
        Xflat[:] = 1.0

    LL = max(1, int(round(N * 1e-1)))
    SL = max(1, int(round(N * 1e-3)))
    Xflat_smooth = torch.minimum(_movmean_1d(Xflat, LL, SL), _movmean_1d(Xflat, SL, LL))
    mflat = torch.max(Xflat_smooth)

    epsilon = (
        torch.full_like(Xflat_smooth, 1e-8)
        if mflat <= 0
        else torch.maximum(1e-4 * mflat, 0.38 * mflat - Xflat_smooth)
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*batching rule.*", category=UserWarning)
        denom = X * torch.conj(X) + epsilon.to(X.dtype)
        H = (Y * torch.conj(X)) / denom
    return H


def squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Element-wise squared error.

    Args:
        y_pred: Predicted signal tensor.
        y_true: Ground-truth signal tensor.

    Returns:
        Element-wise squared difference.
    """
    return (y_pred - y_true) ** 2


def _safe_complex_abs(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Magnitude of a (possibly complex) tensor, avoiding CUDA JIT issues.

    Args:
        x:   Input tensor (real or complex).
        eps: Small value added under the square root for numerical stability.

    Returns:
        Magnitude tensor of the same shape as x.
    """
    if torch.is_complex(x):
        return torch.sqrt(x.real ** 2 + x.imag ** 2 + eps)
    return torch.abs(x)


def _unwrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Unwrap a 1-D phase tensor (radians) along its only axis.

    Args:
        phase: 1-D tensor of phase values in radians.

    Returns:
        Unwrapped phase tensor of the same length.
    """
    if phase.numel() <= 1:
        return phase.clone()
    diff = phase[1:] - phase[:-1]
    two_pi = 2.0 * math.pi
    wrapped = (diff + math.pi) % two_pi - math.pi
    corr = torch.cat([
        torch.tensor([0.0], device=phase.device, dtype=phase.dtype),
        torch.cumsum(wrapped - diff, dim=0),
    ])
    return phase + corr


def interpolate_IRs(
    alpha: float,
    prev_rir: torch.Tensor,
    curr_rir: torch.Tensor,
) -> torch.Tensor:
    """Interpolate between two RIRs in the frequency domain (magnitude in dB + phase).

    Args:
        alpha:    Interpolation factor in [0, 1]  (0 → prev_rir, 1 → curr_rir).
        prev_rir: Previous RIR tensor (any shape — will be squeezed to 1-D).
        curr_rir: Next RIR tensor (same conventions).

    Returns:
        Interpolated IR as a (1, 1, N) tensor.
    """
    alpha_val = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
    device = prev_rir.device if isinstance(prev_rir, torch.Tensor) else torch.device("cpu")
    x1 = prev_rir.detach().squeeze().to(torch.get_default_dtype()).to(device)
    x2 = curr_rir.detach().squeeze().to(torch.get_default_dtype()).to(device)

    n = max(x1.numel(), x2.numel())
    x1 = F.pad(x1, (0, n - x1.numel()))
    x2 = F.pad(x2, (0, n - x2.numel()))

    H1 = torch.fft.rfft(x1, n=n)
    H2 = torch.fft.rfft(x2, n=n)
    eps = 1e-12

    mag_db1 = 20.0 * torch.log10(_safe_complex_abs(H1) + eps)
    mag_db2 = 20.0 * torch.log10(_safe_complex_abs(H2) + eps)
    phase1_un = _unwrap_phase(torch.atan2(H1.imag, H1.real))
    phase2_un = _unwrap_phase(torch.atan2(H2.imag, H2.real))

    mag_db_i = (1.0 - alpha_val) * mag_db1 + alpha_val * mag_db2
    phase_i = (1.0 - alpha_val) * phase1_un + alpha_val * phase2_un
    mag_lin = 10.0 ** (mag_db_i / 20.0)

    H_interp = torch.complex(mag_lin * torch.cos(phase_i), mag_lin * torch.sin(phase_i))
    h_interp = torch.fft.irfft(H_interp, n=n)
    return h_interp.view(1, 1, -1).to(device)


def update_LEM(
    current_time_s: float,
    n_rirs: int,
    transition_times_s: List[Tuple[float, float]],
    rirs_tensors: List[torch.Tensor],
) -> torch.Tensor:
    """Return the current linear echo model (LEM) tensor at a given simulation time.

    In static scenarios (n_rirs == 1) the single RIR is returned directly.
    In time-varying scenarios the function interpolates between consecutive RIRs
    during transition windows and returns the appropriate RIR outside them.

    Args:
        current_time_s:   Current simulation time in seconds.
        n_rirs:           Total number of RIRs in the scenario.
        transition_times_s: List of (t_start, t_end) tuples, one per RIR transition.
        rirs_tensors:     List of RIR tensors, padded to the same length.

    Returns:
        LEM tensor of shape (1, 1, M).
    """
    if n_rirs <= 1:
        return rirs_tensors[0].view(1, 1, -1)

    current_rir_idx = 0
    for i, (t_start, t_end) in enumerate(transition_times_s):
        if current_time_s >= t_end:
            current_rir_idx = i + 1
        elif current_time_s >= t_start:
            alpha = 1.0 if t_end <= t_start else (current_time_s - t_start) / (t_end - t_start)
            return interpolate_IRs(alpha, rirs_tensors[i], rirs_tensors[i + 1])

    return rirs_tensors[current_rir_idx].view(1, 1, -1)


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def load_rirs(
    rir_dir: Path,
    max_n: int = None,
    normalize: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
    """Load WAV files from a directory as float32 mono RIRs.

    Args:
        rir_dir:   Directory containing .wav files.
        max_n:     If set, only load the first max_n files (sorted order).
        normalize: If True, divide all RIRs by sum(|rir_0|).

    Returns:
        (rirs, srs): Lists of 1-D float32 arrays and their sample rates.
    """
    files = sorted(rir_dir.glob("*.wav"))
    if max_n is not None:
        files = files[:max_n]

    rirs, srs = [], []
    for f in files:
        if sf is not None:
            data, sr = sf.read(str(f))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
        else:
            wav, sr = torchaudio.load(str(f))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            data = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        rirs.append(data)
        srs.append(sr)

    if normalize and rirs:
        norm = np.sum(np.abs(rirs[0]))
        if norm > 0:
            rirs = [r / norm for r in rirs]

    return rirs, srs


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a JSON experiment configuration file.

    Args:
        config_path: Path to the .json configuration file.

    Returns:
        Parsed configuration as a dictionary.
    """
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_param_grid(param_grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    """Yield all Cartesian-product combinations from a parameter grid.

    Args:
        param_grid: Mapping from parameter name to an iterable of candidate values.

    Yields:
        Dictionaries with one value per parameter key.
    """
    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))


def discover_input_signals(input_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Build the list of (mode, info) input-signal specifications from config.

    Supports two modes:
      - "white_noise": generates synthetic noise; info contains max_audio_len_s,
        realization_idx, and seed_offset.
      - "song": loads a WAV file; info contains path and max_audio_len_s.

    Args:
        input_cfg: Dict with keys "use_white_noise", "use_songs_folder",
                   "max_audio_len_s", and optionally "max_num_songs".

    Returns:
        List of (mode, info) tuples.
    """
    modes: List[Tuple[str, Dict[str, Any]]] = []
    use_white_noise = bool(input_cfg.get("use_white_noise", False))
    use_songs_folder = bool(input_cfg.get("use_songs_folder", False))
    max_audio_len_s = (input_cfg.get("max_audio_len_s", [None]) or [None])[0]
    max_num_songs = input_cfg.get("max_num_songs", None)

    if use_white_noise:
        n_realizations = max(1, int(max_num_songs)) if max_num_songs is not None else 1
        for i in range(n_realizations):
            modes.append((
                "white_noise",
                {"max_audio_len_s": max_audio_len_s, "realization_idx": i, "seed_offset": i},
            ))

    if use_songs_folder:
        songs_dir = root / "data" / "MedleyDB"
        if songs_dir.is_dir():
            all_songs = sorted(p for p in songs_dir.iterdir() if p.is_file())
            if max_num_songs is not None:
                try:
                    n = int(max_num_songs)
                    if 0 < n < len(all_songs):
                        all_songs = random.sample(all_songs, n)
                except (TypeError, ValueError):
                    pass
            for p in all_songs:
                modes.append(("song", {"path": p, "max_audio_len_s": max_audio_len_s}))

    return modes


# ---------------------------------------------------------------------------
# Adaptive EQ experiment core
# ---------------------------------------------------------------------------

def build_step_sizes(
    mu_opt: float,
    shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    """Build per-parameter step-size tensor with a larger step for the gain parameter.

    The last column (gain in dB) receives a step 100× larger than the EQ parameters.

    Args:
        mu_opt: Base learning rate.
        shape:  Shape of the EQG_params tensor (1, num_params+1).
        device: Target device.

    Returns:
        Step-size tensor of the same shape as EQG_params.
    """
    step_sizes = mu_opt * torch.ones(shape, device=device)
    step_sizes[:, -1] = step_sizes[:, -1] * 1e2 # NOTE: This is a heuristic scaling factor to make gain updates more effective; it may require tuning.
    return step_sizes


def params_to_loss(
    EQG_params: torch.Tensor,
    in_buffer: torch.Tensor,
    EQ_out_buffer: torch.Tensor,
    LEM_out_buffer: torch.Tensor,
    est_mag_response_buffer: torch.Tensor,
    est_cpx_response_buffer: torch.Tensor,
    EQ: ParametricEQ,
    G: Gain,
    LEM: torch.Tensor,
    frame_len: int,
    hop_len: int,
    target_frame: torch.Tensor,
    target_response: torch.Tensor,
    forget_factor: float,
    loss_fcn,
    loss_type: str,
    sr: int = None,
    ROI: Tuple[float, float] = None,
    use_true_LEM: bool = False,
) -> torch.Tensor:
    """Compute the loss for the current EQ parameter vector (used by Jacobian functions).

    This is the differentiable forward pass through EQ → LEM → loss. It is kept
    separate from process_buffers so that jacrev / jacfwd can differentiate through it.

    Args:
        EQG_params:              Combined EQ + gain parameter tensor (1, num_params+1).
        in_buffer, EQ_out_buffer, LEM_out_buffer, est_mag_response_buffer,
        est_cpx_response_buffer: State buffers from the previous frame.
        EQ:                      Differentiable parametric EQ module.
        G:                       Learnable gain module.
        LEM:                     Current LEM tensor (1, 1, M).
        frame_len, hop_len:      Frame and hop lengths in samples.
        target_frame:            Desired output for the current frame.
        target_response:         Full desired impulse response (1, 1, L).
        forget_factor:           Exponential smoothing factor for response estimates.
        loss_fcn:                Loss function (e.g. F.mse_loss or squared_error).
        loss_type:               "FD-MSE" / "FD-SE" (frequency domain) or time domain.
        sr:                      Sample rate (required for FD losses).
        ROI:                     Region of interest (fmin, fmax) Hz.
        use_true_LEM:            If True, use the true LEM IR for gradient computation.

    Returns:
        Scalar (or per-element) loss value.
    """
    EQ_params = EQG_params[:, :-1]
    G_param = EQG_params[:, -1:]

    EQ_out = EQ.process_normalized(in_buffer, EQ_params)
    EQ_out = G.process(EQ_out, sr, G_param)
    EQ_out_buffer = F.pad(EQ_out_buffer[..., hop_len:], (0, hop_len)) + EQ_out

    if use_true_LEM:
        LEM_est = LEM.view(1, 1, -1).detach()
    else:
        LEM_H_est = torch.view_as_complex(est_cpx_response_buffer.squeeze())
        LEM_est = torch.fft.irfft(LEM_H_est, n=2 * frame_len - 1).view(1, 1, -1).detach()

    LEM_out = LEMConv.apply(EQ_out_buffer[:, :, :frame_len], LEM.view(1, 1, -1), LEM_est)
    LEM_out_buffer = F.pad(LEM_out_buffer[..., hop_len:], (0, hop_len)) + LEM_out

    nfft = 2 * frame_len - 1
    freqs = torch.fft.rfftfreq(nfft, d=1.0 / sr, device=LEM_out_buffer.device)

    if loss_type in ("FD-MSE", "FD-SE"):
        H_SS = kirkeby_deconvolve(in_buffer.squeeze(), LEM_out_buffer[:, :, :frame_len].squeeze(), nfft, sr, ROI)
        H_mag_db_current = 20 * torch.log10(_safe_complex_abs(H_SS, eps=1e-8))

        if torch.sum(torch.abs(est_mag_response_buffer)) == 0:
            H_mag_db = H_mag_db_current
        else:
            H_mag_db = forget_factor * H_mag_db_current + (1 - forget_factor) * est_mag_response_buffer.squeeze()

        desired_mag_db = 20 * torch.log10(_safe_complex_abs(torch.fft.rfft(target_response.squeeze(), n=nfft), eps=1e-8))
        roi_mask = (freqs >= ROI[0]) & (freqs <= ROI[1]) if ROI is not None else torch.ones_like(H_mag_db, dtype=torch.bool)

        H_log, _ = interp_to_log_freq(H_mag_db[roi_mask], freqs[roi_mask], n_points=256)
        D_log, _ = interp_to_log_freq(desired_mag_db[roi_mask], freqs[roi_mask], n_points=256)

        smooth_kernel = torch.ones(1, 1, 15, device=H_log.device) / 15
        pad = 7
        H_s = F.conv1d(F.pad(H_log.view(1, 1, -1), (pad, pad), mode="reflect"), smooth_kernel, padding=0).squeeze()
        D_s = F.conv1d(F.pad(D_log.view(1, 1, -1), (pad, pad), mode="reflect"), smooth_kernel, padding=0).squeeze()
        return loss_fcn(H_s, D_s)

    return loss_fcn(LEM_out_buffer[:, :, :frame_len], target_frame)


def process_buffers(
    EQG_params: torch.Tensor,
    in_buffer: torch.Tensor,
    EQ_out_buffer: torch.Tensor,
    LEM_out_buffer: torch.Tensor,
    est_mag_response_buffer: torch.Tensor,
    est_cpx_response_buffer: torch.Tensor,
    noEQ_in_buffer: torch.Tensor,
    noEQ_out_buffer: torch.Tensor,
    EQ: ParametricEQ,
    G: Gain,
    LEM: torch.Tensor,
    frame_len: int,
    hop_len: int,
    target_frame: torch.Tensor,
    target_response: torch.Tensor,
    forget_factor: float,
    loss_type: str,
    loss_fcn,
    sr: int = None,
    ROI: Tuple[float, float] = None,
    use_true_LEM: bool = False,
    checkpoint_state: dict = None,
) -> Tuple[torch.Tensor, tuple]:
    """Run one frame of the adaptive-EQ forward pass and compute loss and validation error.

    Processes the current input frame through the EQ, the LEM, and optionally a
    non-EQ reference path. Updates state buffers and computes the per-frame loss
    and frequency-domain validation error.

    Args:
        EQG_params:              Combined EQ + gain parameter tensor (1, num_params+1).
        in_buffer:               Current input frame (1, 1, frame_len).
        EQ_out_buffer:           Running EQ output buffer.
        LEM_out_buffer:          Running LEM output buffer.
        est_mag_response_buffer: Running estimated magnitude response buffer.
        est_cpx_response_buffer: Running estimated complex response buffer.
        noEQ_in_buffer:          Running input buffer for the no-EQ path.
        noEQ_out_buffer:         Running LEM output buffer for the no-EQ path.
        EQ:                      Differentiable parametric EQ module.
        G:                       Learnable gain module.
        LEM:                     Current LEM tensor (1, 1, M).
        frame_len, hop_len:      Frame and hop lengths in samples.
        target_frame:            Desired output for the current frame (1, 1, frame_len).
        target_response:         Full desired impulse response (1, 1, L).
        forget_factor:           Exponential smoothing factor for response estimates.
        loss_type:               Loss type string (e.g. "FD-MSE").
        loss_fcn:                Loss function callable.
        sr:                      Sample rate in Hz.
        ROI:                     Region of interest (fmin, fmax) Hz.
        use_true_LEM:            Use the true LEM IR as the gradient path estimate.
        checkpoint_state:        If not None, populated with frequency-domain snapshot data.

    Returns:
        (loss, updated_buffers): Loss tensor and a tuple of updated buffer tensors.
    """
    EQ_params = EQG_params[:, :-1]
    G_param = EQG_params[:, -1:]

    EQ_out = EQ.process_normalized(in_buffer, EQ_params)
    EQ_out = G.process(EQ_out, sr, G_param)
    EQ_out_buffer = F.pad(EQ_out_buffer[..., hop_len:], (0, hop_len)) + EQ_out

    if use_true_LEM:
        LEM_est = LEM.view(1, 1, -1).detach()
    else:
        LEM_H_est = torch.view_as_complex(est_cpx_response_buffer.squeeze())
        LEM_est = torch.fft.irfft(LEM_H_est, n=2 * frame_len - 1).view(1, 1, -1).detach()

    LEM_out = LEMConv.apply(EQ_out_buffer[:, :, :frame_len], LEM.view(1, 1, -1), LEM_est)
    LEM_out_buffer = F.pad(LEM_out_buffer[..., hop_len:], (0, hop_len)) + LEM_out

    # Non-EQ reference path (input → LEM, no equaliser applied)
    with torch.no_grad():
        noEQ_in_pad = F.pad(in_buffer, (0, noEQ_in_buffer.shape[-1] - in_buffer.shape[-1]))
        noEQ_in_buffer = F.pad(noEQ_in_buffer[..., hop_len:], (0, hop_len)) + noEQ_in_pad
        noEQ_out = LEMConv.apply(noEQ_in_buffer[:, :, :frame_len], LEM.view(1, 1, -1), LEM_est)
        noEQ_out_buffer = F.pad(noEQ_out_buffer[..., hop_len:], (0, hop_len)) + noEQ_out

    nfft = 2 * frame_len - 1
    freqs = torch.fft.rfftfreq(nfft, d=1.0 / sr, device=LEM_out_buffer.device)
    eps = 1e-8

    # Update running complex LEM estimate via Kirkeby deconvolution
    LEM_H_est_new = kirkeby_deconvolve(EQ_out_buffer.squeeze(), LEM_out_buffer[:, :, :frame_len].squeeze(), nfft, sr, ROI)
    roi_mask = (freqs >= ROI[0]) & (freqs <= ROI[1]) if ROI is not None else torch.ones(len(freqs), dtype=torch.bool, device=freqs.device)

    forget_factor_cpx = 1.0 if torch.sum(torch.abs(est_mag_response_buffer)) == 0 else forget_factor
    LEM_H_est_ri = torch.view_as_real(LEM_H_est_new).view(1, 1, -1, 2)
    est_cpx_response_buffer = (1 - forget_factor_cpx) * est_cpx_response_buffer + forget_factor_cpx * LEM_H_est_ri.detach()

    # Compute smoothed frequency-domain magnitude responses
    H_SS = kirkeby_deconvolve(in_buffer.squeeze(), LEM_out_buffer[:, :, :frame_len].squeeze(), nfft, sr, ROI)
    H_mag_db_current = 20 * torch.log10(_safe_complex_abs(H_SS, eps=eps))

    forget_factor_loss = 1.0 if torch.sum(torch.abs(est_mag_response_buffer)) == 0 else forget_factor
    H_mag_db = forget_factor_loss * H_mag_db_current + (1 - forget_factor_loss) * est_mag_response_buffer.squeeze()
    est_mag_response_buffer = H_mag_db.view(1, 1, -1).detach()

    desired_mag_db = 20 * torch.log10(_safe_complex_abs(torch.fft.rfft(target_response.squeeze(), n=nfft), eps=eps))
    LEM_mag_db = 20 * torch.log10(_safe_complex_abs(torch.fft.rfft(LEM.squeeze(), n=nfft), eps=eps))

    # Restrict to ROI, then interpolate to log frequency and smooth
    freqs_roi = freqs[roi_mask]
    n_log = 256
    smooth_kernel = torch.ones(1, 1, 15, device=H_mag_db.device) / 15
    pad = 7

    def _smooth(mag: torch.Tensor) -> torch.Tensor:
        m_roi = mag[roi_mask]
        m_log, _ = interp_to_log_freq(m_roi, freqs_roi, n_points=n_log)
        return F.conv1d(F.pad(m_log.view(1, 1, -1), (pad, pad), mode="reflect"), smooth_kernel, padding=0).squeeze()

    H_s = _smooth(H_mag_db)
    D_s = _smooth(desired_mag_db)
    LEM_s = _smooth(LEM_mag_db)
    _, freqs_log = interp_to_log_freq(H_mag_db[roi_mask], freqs_roi, n_points=n_log)

    # Loss
    if loss_type in ("FD-MSE", "FD-SE"):
        loss = loss_fcn(H_s, D_s)
    else:
        loss = loss_fcn(LEM_out_buffer[:, :, :frame_len], target_frame)

    # Validation error: relative L1 distance vs. un-equalised LEM response
    validation_error = F.l1_loss(H_s, D_s) / (F.l1_loss(LEM_s, D_s) + 1e-12)

    if checkpoint_state is not None:
        checkpoint_state["freqs_log"] = freqs_log.detach().cpu().numpy().astype(np.float32)
        checkpoint_state["H_total_db"] = H_s.detach().cpu().numpy().astype(np.float32)
        checkpoint_state["H_desired_db"] = D_s.detach().cpu().numpy().astype(np.float32)
        checkpoint_state["H_lem_db"] = LEM_s.detach().cpu().numpy().astype(np.float32)

    updated_buffers = (EQ_out_buffer, LEM_out_buffer, est_mag_response_buffer, est_cpx_response_buffer, validation_error, noEQ_in_buffer, noEQ_out_buffer)
    return loss, updated_buffers


def run_control_experiment(
    sim_cfg: Dict[str, Any],
    input_spec: Tuple[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one full adaptive room-equalisation (ARE) experiment.

    Loads the RIRs and input signal, builds the target response, and runs the
    frame-based control loop that adapts a differentiable parametric EQ to match
    the target frequency response as the acoustic scenario evolves.

    Args:
        sim_cfg:    Dictionary with all simulation parameters (see configs/).
        input_spec: (mode, info) tuple — mode is "white_noise" or "song".

    Returns:
        Result dictionary containing loss and validation histories, audio outputs,
        timing statistics, and the final EQ state. Returns None on failure.
    """
    exp_t_start = time.perf_counter()
    mode, info = input_spec
    print(f"\n=== Running control experiment | mode={mode} ===")

    # Load configuration parameters
    max_audio_len_s = info["max_audio_len_s"]
    ROI = sim_cfg["ROI"]
    n_rirs = sim_cfg["n_rirs"]
    loss_type = sim_cfg["loss_type"]
    optim_type = sim_cfg["optim_type"]
    mu_opt = sim_cfg["mu_opt"]
    lambda_newton = sim_cfg.get("lambda_newton", 0.0)
    target_response_type = sim_cfg["target_response_type"]
    frame_len = sim_cfg["frame_len"]
    hop_len = sim_cfg["hop_len"]
    forget_factor = sim_cfg["forget_factor"]
    eps_0 = sim_cfg["eps_0"]
    use_true_LEM = sim_cfg["use_true_LEM"]
    n_checkpoints = sim_cfg.get("n_checkpoints", 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load RIRs
    rir_dir = Path(sim_cfg.get("rir_dir", root / "data" / "SoundCam" / "moving_listener"))
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs, normalize=False)
    rir_init = rirs[0]
    sr = rirs_srs[0]

    # Prepare time-varying scenario
    rirs_tensors, transition_times_s = [], []
    if n_rirs > 1:
        rirs_tensors = [torch.from_numpy(r).float().to(device) for r in rirs]
        max_len = max(r.shape[0] for r in rirs_tensors)
        rirs_tensors = [F.pad(r, (0, max_len - r.shape[0])) for r in rirs_tensors]
        seg_s = max_audio_len_s / n_rirs
        for i in range(1, n_rirs):
            t_start = i * seg_s
            t_end = min(t_start + sim_cfg["transition_time_s"], max_audio_len_s)
            transition_times_s.append((t_start, t_end))

    # Compute target response from the initial RIR
    lem_delay = get_delay_from_ir(rir_init, sr)
    eq_comp = get_compensation_EQ_params(rir_init, sr, ROI, num_sections=7)
    target_response = build_target_response_lin_phase(
        sr=sr,
        response_type=target_response_type,
        target_mag_resp=eq_comp["target_response_db"],
        target_mag_freqs=eq_comp["freq_axis_smoothed"],
        fir_len=2048,
        ROI=ROI,
        rolloff_octaves=0.5,
        device=device,
    )
    h_linear_np = target_response.squeeze().cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        h_min_np = minimum_phase(h_linear_np, method="homomorphic", half=False)
    h_min = torch.from_numpy(h_min_np).float().to(device)
    delay_zeros = torch.zeros(lem_delay, device=device)
    target_response = torch.cat([delay_zeros, h_min]).view(1, 1, -1) / 2  # NOTE: hardcoded /2 factor

    # Initialise LEM estimate (assume perfect knowledge at t=0)
    LEM = torch.from_numpy(rir_init).view(1, 1, -1).to(device)

    # Initialise differentiable EQ + gain
    EQ = ParametricEQ(sample_rate=sr)
    G = Gain(sample_rate=sr)
    EQG_params = torch.nn.Parameter(
        torch.cat([torch.rand(1, EQ.num_params), torch.zeros(1, 1)], dim=-1).to(device)
    )
    EQ_memory = 128  # NOTE: hardcoded; must be > 0

    # Load / synthesise input signal
    if mode == "white_noise":
        T = int(max_audio_len_s * sr)
        inp = torch.randn(T, device=device)
    else:
        audio_path = info["path"]
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        wav, audio_sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        if audio_sr != sr:
            wav = torchaudio.transforms.Resample(orig_freq=audio_sr, new_freq=sr)(wav)
        inp = wav.to(device)
        if max_audio_len_s is not None:
            inp = inp[: int(max_audio_len_s * sr)]
        T = len(inp)

    inp = inp / inp.abs().max()
    inp = inp.view(1, 1, -1)

    # Optimiser setup
    match optim_type:
        case "SGD":
            optimizer = torch.optim.SGD([EQG_params], lr=mu_opt)
        case "Adam":
            optimizer = torch.optim.Adam([EQG_params], lr=mu_opt, betas=(0.9, 0.99999))
        case "GHAM-1" | "GHAM-2":
            ridge_regressor = Ridge(alpha=5e0, fit_intercept=False)
            jac_fcn = (jacrev if loss_type in ("TD-MSE", "FD-MSE") else jacfwd)(params_to_loss, argnums=0)
        case "Newton" | "GHAM-3" | "GHAM-4":
            ridge_regressor = Ridge(alpha=5e0, fit_intercept=False)
            jac_fcn = (jacrev if loss_type in ("TD-MSE", "FD-MSE") else jacfwd)(params_to_loss, argnums=0)
            hess_fcn = jacfwd(jac_fcn, argnums=0)
            if optim_type == "GHAM-4":
                raise NotImplementedError("GHAM-4 is not implemented yet due to PyTorch implementation.")
                jac3_fcn = jacfwd(hess_fcn, argnums=0)
        case _:
            raise ValueError(f"Unknown optim_type: '{optim_type}'")

    # Loss function
    match loss_type:
        case "TD-MSE" | "FD-MSE":
            loss_fcn = F.mse_loss
        case "TD-SE" | "FD-SE":
            loss_fcn = squared_error
        case _:
            raise NotImplementedError(f"Unsupported loss_type: '{loss_type}'")

    # Pre-compute desired output signal
    desired_output = torchaudio.functional.fftconvolve(inp, target_response, mode="full")

    # Allocate buffers
    y_control = torch.zeros(1, 1, T, device=device)
    y_noEQ = torch.zeros(1, 1, T, device=device)
    in_buffer = torch.zeros(1, 1, frame_len, device=device)
    EQ_out_len = next_power_of_2(2 * frame_len - 1)
    EQ_out_buffer = torch.zeros(1, 1, EQ_out_len, device=device)
    LEM_out_len = frame_len + LEM.shape[-1] - 1
    LEM_out_buffer = torch.zeros(1, 1, LEM_out_len, device=device)
    est_mag_response_buffer = torch.zeros(1, 1, frame_len, device=device)
    init_cpx = torch.fft.rfft(target_response, n=2 * frame_len - 1)
    est_cpx_response_buffer = torch.view_as_real(init_cpx).view(1, 1, -1, 2)
    noEQ_in_buffer = torch.zeros(1, 1, EQ_out_len, device=device)
    noEQ_out_buffer = torch.zeros(1, 1, LEM_out_len, device=device)

    n_frames = (T - frame_len) // hop_len + 1
    loss_history: List[float] = []
    validation_error_history: List[float] = []
    checkpoint_states: List[dict] = []

    # Determine checkpoint frame indices
    if n_checkpoints and n_frames > 1:
        raw_idxs = np.linspace(0, n_frames - 1, n_checkpoints + 2)
        checkpoint_indices = sorted({int(round(i)) for i in raw_idxs})
    else:
        checkpoint_indices = [0, n_frames - 1] if n_frames > 0 else []

    # Shared arguments for params_to_loss / process_buffers (excluding EQG_params and buffers)
    _jac_args = (EQ, G, LEM, frame_len, hop_len, None, target_response, forget_factor, loss_fcn, loss_type, sr, ROI, use_true_LEM)

    ###########################################################################
    # Main frame loop
    ###########################################################################
    for k in tqdm(range(n_frames), desc="ARE Simulation", unit="frame"):
        start_idx = k * hop_len
        current_time_s = start_idx / sr

        if n_rirs > 1:
            LEM = update_LEM(current_time_s, n_rirs, transition_times_s, rirs_tensors)

        in_buffer = inp[:, :, start_idx : start_idx + frame_len]
        target_frame = desired_output[:, :, start_idx : start_idx + frame_len]

        do_checkpoint = k in checkpoint_indices
        checkpoint_state = {} if do_checkpoint else None

        loss, buffers = process_buffers(
            EQG_params, in_buffer, EQ_out_buffer, LEM_out_buffer,
            est_mag_response_buffer, est_cpx_response_buffer,
            noEQ_in_buffer, noEQ_out_buffer,
            EQ, G, LEM, frame_len, hop_len,
            target_frame, target_response, forget_factor,
            loss_type, loss_fcn, sr=sr, ROI=ROI,
            use_true_LEM=use_true_LEM, checkpoint_state=checkpoint_state,
        )
        (EQ_out_buffer, LEM_out_buffer, est_mag_response_buffer,
         est_cpx_response_buffer, validation_error, noEQ_in_buffer, noEQ_out_buffer) = buffers

        loss_history.append(torch.mean(loss).item())
        validation_error_history.append(validation_error.item())

        # Collect checkpoint snapshot
        if checkpoint_state is not None:
            with torch.no_grad():
                checkpoint_state["EQ_params"] = EQG_params[:, :-1].detach().cpu().numpy().astype(np.float32)
                checkpoint_state["G_param_db"] = float(EQG_params[:, -1:].item())
                checkpoint_state["time_s"] = float(current_time_s)
                checkpoint_state["frame_idx"] = int(k)
                checkpoint_state["sr"] = int(sr)
                try:
                    eq_t = EQG_params[:, :-1].detach().view(1, -1)
                    p = EQ.denormalize_param_dict(EQ.extract_param_dict(eq_t))
                    names = [
                        ("low_shelf_gain_db", "low_shelf_q_factor", "low_shelf_cutoff_freq"),
                        ("band0_gain_db", "band0_q_factor", "band0_cutoff_freq"),
                        ("band1_gain_db", "band1_q_factor", "band1_cutoff_freq"),
                        ("band2_gain_db", "band2_q_factor", "band2_cutoff_freq"),
                        ("band3_gain_db", "band3_q_factor", "band3_cutoff_freq"),
                        ("band4_gain_db", "band4_q_factor", "band4_cutoff_freq"),
                        ("high_shelf_gain_db", "high_shelf_q_factor", "high_shelf_cutoff_freq"),
                    ]
                    eq_matrix = np.array([[float(p[g].view(-1)[0]), float(p[q].view(-1)[0]), float(p[f].view(-1)[0])] for g, q, f in names], dtype=np.float32)
                    checkpoint_state["EQ_matrix"] = eq_matrix
                except Exception:
                    pass
            checkpoint_states.append(checkpoint_state)

        # Parameter update
        _jac_buf_args = (in_buffer, EQ_out_buffer, LEM_out_buffer, est_mag_response_buffer, est_cpx_response_buffer)

        match optim_type:
            case "GHAM-1" | "GHAM-2":
                if loss_type in ("TD-MSE", "FD-MSE"):
                    loss.backward()
                    jac = EQG_params.grad.clone().view(1, -1)
                else:
                    jac = jac_fcn(EQG_params, *_jac_buf_args, EQ, G, LEM, frame_len, hop_len, target_frame, target_response, forget_factor, loss_fcn, loss_type, sr, ROI, use_true_LEM).squeeze()

                loss_val = torch.maximum(loss.detach() - torch.tensor(eps_0, device=device), torch.tensor(0.0, device=device))
                with torch.no_grad():
                    ridge_regressor.fit(jac, loss_val.view(-1, 1))
                    update = ridge_regressor.w
                    scale = mu_opt if optim_type == "GHAM-1" else mu_opt * (2 - mu_opt)
                    step_sizes = build_step_sizes(scale, EQG_params.shape, device)
                    EQG_params -= step_sizes * update.view_as(EQG_params)
                EQG_params.grad = None

            case "Newton":
                jac = jac_fcn(EQG_params, *_jac_buf_args, EQ, G, LEM, frame_len, hop_len, target_frame, target_response, forget_factor, loss_fcn, loss_type, sr, ROI, use_true_LEM).squeeze()
                hess = hess_fcn(EQG_params, *_jac_buf_args, EQ, G, LEM, frame_len, hop_len, target_frame, target_response, forget_factor, loss_fcn, loss_type, sr, ROI, use_true_LEM).squeeze()
                hess_reg = hess + lambda_newton * torch.eye(hess.shape[-1], device=device, dtype=hess.dtype)
                with torch.no_grad():
                    update = lstsq(hess_reg, jac.view(-1, 1)).solution
                    step_sizes = build_step_sizes(mu_opt, EQG_params.shape, device)
                    EQG_params -= step_sizes * update.view_as(EQG_params)

            case "GHAM-3" | "GHAM-4":
                jac = jac_fcn(EQG_params, *_jac_buf_args, EQ, G, LEM, frame_len, hop_len, target_frame, target_response, forget_factor, loss_fcn, loss_type, sr, ROI, use_true_LEM)
                hess = hess_fcn(EQG_params, *_jac_buf_args, EQ, G, LEM, frame_len, hop_len, target_frame, target_response, forget_factor, loss_fcn, loss_type, sr, ROI, use_true_LEM).squeeze()
                loss_val = torch.maximum(loss.detach() - torch.tensor(eps_0, device=device), torch.tensor(0.0, device=device)).view(-1, 1)
                with torch.no_grad():
                    step_sizes = build_step_sizes(mu_opt, EQG_params.shape, device).T
                    ridge_regressor.fit(jac, loss_val)
                    theta_1 = -step_sizes * ridge_regressor.w
                    theta_2 = (1 - step_sizes) * theta_1
                    residual_3 = theta_1.T @ hess @ theta_1 + jac @ theta_2
                    ridge_regressor.fit(jac, residual_3)
                    theta_3 = theta_2 + -step_sizes * ridge_regressor.w
                    if optim_type == "GHAM-3":
                        correction = theta_1 + theta_2 + theta_3
                    else:
                        raise NotImplementedError("GHAM-4 is not implemented yet due to PyTorch implementation.") # NOTE: to be implemented
                        jac3 = jac3_fcn(EQG_params, *_jac_buf_args, EQ, G, LEM, frame_len, hop_len, target_frame, target_response, forget_factor, loss_fcn, loss_type, sr, ROI, use_true_LEM).squeeze()
                        residual_4 = -step_sizes * (torch.einsum("ijk,i,j,k->", jac3, theta_1.squeeze(), theta_2.squeeze(), theta_3.squeeze()) / 6 + theta_2.T @ hess @ theta_1 + jac @ theta_3)
                        ridge_regressor.fit(jac, residual_4)
                        correction = theta_1 + theta_2 + theta_3 + ridge_regressor.w
                    EQG_params += correction.view_as(EQG_params)

            case _:
                optimizer.zero_grad()
                loss.backward()
                EQG_params.grad[:, -1:] *= 1e2
                optimizer.step()

        with torch.no_grad():
            EQG_params.data[:, :-1].clamp_(0.0, 1.0)
            EQ_out_buffer = EQ_out_buffer.detach()
            LEM_out_buffer = LEM_out_buffer.detach()
            est_mag_response_buffer = est_mag_response_buffer.detach()
            est_cpx_response_buffer = est_cpx_response_buffer.detach()
            noEQ_in_buffer = noEQ_in_buffer.detach()
            noEQ_out_buffer = noEQ_out_buffer.detach()

        end_idx = min(start_idx + frame_len, T)
        y_control[:, :, start_idx:end_idx] += LEM_out_buffer[:, :, : end_idx - start_idx]
        y_noEQ[:, :, start_idx:end_idx] += noEQ_out_buffer[:, :, : end_idx - start_idx]

    total_time_s = time.perf_counter() - exp_t_start
    time_axis = np.arange(len(validation_error_history), dtype=float) * (hop_len / sr)
    nfft_plot = 2 * frame_len - 1
    target_freq_axis = np.fft.rfftfreq(nfft_plot, d=1.0 / sr)
    target_mag_db = 20.0 * np.log10(
        np.abs(np.fft.rfft(target_response.squeeze().detach().cpu().numpy(), n=nfft_plot)) + 1e-12
    )

    with torch.no_grad():
        result = {
            "loss_history": np.array(loss_history, dtype=float),
            "validation_error_history": np.array(validation_error_history, dtype=float),
            "time_axis": time_axis,
            "transition_times": transition_times_s if n_rirs > 1 else None,
            "optim_type": optim_type,
            "transition_time_s": sim_cfg["transition_time_s"],
            "input_audio": inp.squeeze().cpu().numpy().astype(np.float32),
            "desired_audio": desired_output.squeeze().cpu().numpy()[:T].astype(np.float32),
            "y_control": y_control.squeeze().cpu().numpy().astype(np.float32),
            "y_noEQ": y_noEQ.squeeze().cpu().numpy().astype(np.float32),
            "sr": int(sr),
            "n_frames": int(n_frames),
            "control_experiment_time_s": float(total_time_s),
            "avg_compute_time_per_frame_s": total_time_s / n_frames if n_frames > 0 else float("nan"),
            "final_eq_params_normalized": EQG_params[:, :-1].detach().cpu().numpy().astype(np.float32),
            "final_gain_db": EQG_params[:, -1:].detach().cpu().numpy().astype(np.float32),
            "final_true_lem_ir": LEM.squeeze().detach().cpu().numpy().astype(np.float32),
            "target_freq_axis": target_freq_axis.astype(np.float32),
            "target_mag_db": target_mag_db.astype(np.float32),
        }
    if checkpoint_states:
        result["checkpoints"] = checkpoint_states
    return result


# ---------------------------------------------------------------------------
# Private helpers for get_compensation_EQ_params
# ---------------------------------------------------------------------------

def _seven_biquad_eq_params_to_dict(EQ_params: np.ndarray) -> dict:
    """Convert a (7, 3) EQ parameter matrix into a named parameter dictionary."""
    assert EQ_params.shape == (7, 3), "Expected EQ_params shape (7, 3)"
    labels = ["low_shelf", "band0", "band1", "band2", "band3", "band4", "high_shelf"]
    keys = ["gain_db", "q_factor", "cutoff_freq"]
    return {f"{lbl}_{k}": float(EQ_params[i, j]) for i, lbl in enumerate(labels) for j, k in enumerate(keys)}


def _get_target_response_comp_EQ(
    cf: np.ndarray,
    oa: np.ndarray,
    ROI: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linearised target response for compensation EQ (log-frequency fit)."""
    cf = np.atleast_1d(cf).astype(float)
    oa = np.atleast_1d(oa).astype(float)
    pdb = 20.0 * np.log10(oa + 1e-12)

    lfcutoff = 2.5 * ROI[0]
    hfmax = 0.6 * ROI[1]
    linIdx = (cf > 0) & (cf >= lfcutoff) & (cf <= hfmax)

    target_resp = np.copy(pdb)
    pfit = np.array([0.0, 0.0])
    if np.any(linIdx):
        slope, intercept = np.polyfit(np.log(cf[linIdx]), pdb[linIdx], 1)
        pfit = np.array([intercept, slope])
        target_resp = pfit[0] + pfit[1] * np.log(cf)
        lfcutoff_inner = 1.05 * lfcutoff
        idx_low = cf < lfcutoff_inner
        if np.any(idx_low):
            max_atten = min(30.0, ROI[0] / 2.0)
            frac = (lfcutoff_inner - cf[idx_low]) / lfcutoff_inner
            target_resp[idx_low] -= max_atten * frac ** 2

    return target_resp, pfit, pdb


def _get_initial_gains(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def _init_eq_parameters(
    num_filters: int,
    ROI: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialise EQ parameter matrix and bounds for least-squares optimisation."""
    fmin = max(20.0, ROI[0])
    fmax = min(20e3, ROI[1])
    Fc_peaks = np.logspace(np.log10(fmin), np.log10(fmax), num_filters - 2)
    Fc_shelves = np.array([(2 * ROI[0] + ROI[1]) / 3, (ROI[0] + 2 * ROI[1]) / 3])

    init_params = np.zeros((num_filters, 3))
    init_params[:, 0] = _get_initial_gains(num_filters)
    init_params[:-2, 1] = 1.5
    init_params[-2:, 1] = 1.0
    init_params[:-2, 2] = Fc_peaks
    init_params[-2:, 2] = Fc_shelves

    lb = np.zeros_like(init_params)
    lb[:, 0] = -24.0; lb[:-2, 1] = 0.2; lb[-2:, 1] = 0.1; lb[:, 2] = fmin - 1e-6

    ub = np.zeros_like(init_params)
    ub[:, 0] = 20.0; ub[:-2, 1] = 17.3; ub[-2:, 1] = 5.0; ub[:, 2] = fmax + 1e-6

    return init_params, lb, ub


def _eq_optimizer(
    num_filters: int,
    f: np.ndarray,
    meas_resp_db: np.ndarray,
    target_resp_db: np.ndarray,
    ROI: Tuple[float, float],
    Fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimise parametric EQ parameters to match a target response."""
    init_params, lb, ub = _init_eq_parameters(num_filters, ROI)

    def _residual(params_flat):
        params = params_flat.reshape(num_filters, 3)
        filt_resp = compute_parametric_eq_response(params, f, Fs)
        idx = (f >= ROI[0]) & (f <= ROI[1])
        return (meas_resp_db + filt_resp - target_resp_db)[idx]

    res = least_squares(_residual, init_params.ravel(), bounds=(lb.ravel(), ub.ravel()),
                        ftol=1e-8, max_nfev=min(500 * num_filters, 1000), verbose=1)
    EQ_params = res.x.reshape(num_filters, 3)
    filt_resp = compute_parametric_eq_response(EQ_params, f, Fs)
    return EQ_params, meas_resp_db + filt_resp, filt_resp
