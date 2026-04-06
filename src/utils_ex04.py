import json, itertools, random, sys, math, time, torch, torchaudio, warnings, os
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from typing import Dict, Any, Iterable, List, Tuple
import numpy as np
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import least_squares
from scipy.signal import minimum_phase
try:
    import soundfile as sf
except Exception:
    sf = None
import torch.nn.functional as F
from torch.func import jacrev, jacfwd
from torch.linalg import lstsq
from modules_ex04 import LEMConv, Ridge
from lib.local_dasp_pytorch.modules import ParametricEQ, Gain
try:
    from experiment_03 import frame_analysis_plot
except Exception:
    def frame_analysis_plot(*args, **kwargs):
        return None
from utils import (
    load_audio,
    load_rirs,
    ensure_rirs_sample_rate,
    get_delay_from_ir,
    next_power_of_2,
    save_audio,
    get_compensation_EQ_params,
)

root = Path(__file__).resolve().parent.parent

##############################
# Signal processing utiities #
##############################

def get_delay_from_ir(rir: np.ndarray, sr: int) -> int:
    """Estimate delay (in samples) from an RIR by finding the position of the first maximum.

    Finds the sample index where the first maximum (peak) of the RIR occurs.
    This is a robust estimate of the direct sound arrival time.

    Args:
        rir: 1-D numpy array containing the RIR
        sr: sample rate (Hz)
    Returns:
        delay_samples: estimated delay in samples (int) - index of the maximum value
    """
    # Find the index of the maximum absolute value
    delay_samples = np.argmax(np.abs(rir))
    return delay_samples


def _seven_biquad_eq_params_to_dict(EQ_params: np.ndarray) -> dict:
    """
    Convert (7, 3) EQ parameter matrix into a named parameter dictionary.

    EQ_params: np.ndarray with shape (7, 3)
               columns: [gain_dB, Q_or_slope, Fc]
    """
    assert EQ_params.shape == (7, 3), "Expected EQ_params shape (7, 3)"

    return {
        # ---- Low shelf ----
        "low_shelf_gain_db": float(EQ_params[0, 0]),
        "low_shelf_q_factor": float(EQ_params[0, 1]),
        "low_shelf_cutoff_freq": float(EQ_params[0, 2]),

        # ---- Band 0 ----
        "band0_gain_db": float(EQ_params[1, 0]),
        "band0_q_factor": float(EQ_params[1, 1]),
        "band0_cutoff_freq": float(EQ_params[1, 2]),

        # ---- Band 1 ----
        "band1_gain_db": float(EQ_params[2, 0]),
        "band1_q_factor": float(EQ_params[2, 1]),
        "band1_cutoff_freq": float(EQ_params[2, 2]),

        # ---- Band 2 ----
        "band2_gain_db": float(EQ_params[3, 0]),
        "band2_q_factor": float(EQ_params[3, 1]),
        "band2_cutoff_freq": float(EQ_params[3, 2]),

        # ---- Band 3 ----
        "band3_gain_db": float(EQ_params[4, 0]),
        "band3_q_factor": float(EQ_params[4, 1]),
        "band3_cutoff_freq": float(EQ_params[4, 2]),

        # ---- Band 4 ----
        "band4_gain_db": float(EQ_params[5, 0]),
        "band4_q_factor": float(EQ_params[5, 1]),
        "band4_cutoff_freq": float(EQ_params[5, 2]),

        # ---- High shelf ----
        "high_shelf_gain_db": float(EQ_params[6, 0]),
        "high_shelf_q_factor": float(EQ_params[6, 1]),
        "high_shelf_cutoff_freq": float(EQ_params[6, 2]),
    }


def biquad_coefficients(
    gain_db: float,
    fc: float,
    Q: float,
    Fs: float,
    filter_type: str,
):
    """
    RBJ cookbook biquad coefficients.
    Returns (b, a) normalized so a[0] = 1.
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / Fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * Q)

    if filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

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
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Normalize
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return b, a


def compute_parametric_eq_response(
    params: np.ndarray,
    f: np.ndarray,
    Fs: float,
) -> np.ndarray:
    """
    Compute total EQ filter response in dB using SOS biquad cascade.

    params: (num_filters, 3) -> [gain_dB, Q_or_slope, Fc]
    f: frequency vector in Hz
    """
    num_filters = params.shape[0]

    # Build SOS array: (num_filters, 6)
    sos = np.zeros((num_filters, 6))

    for i, (gain_db, Q, Fc) in enumerate(params):
        if i == 0:
            ftype = "low_shelf"
        elif i == num_filters - 1:
            ftype = "high_shelf"
        else:
            ftype = "peaking"

        b, a = biquad_coefficients(
            gain_db=gain_db,
            fc=Fc,
            Q=Q,
            Fs=Fs,
            filter_type=ftype,
        )

        sos[i, :] = np.hstack([b, a])

    # Convert frequency vector to digital radian frequency
    worN = 2 * np.pi * f / Fs

    # Complex frequency response
    H = sosfreqz_np(sos, worN)

    # Magnitude in dB
    filt_resp_db = 20.0 * np.log10(np.abs(H) + 1e-12)

    return filt_resp_db


def sosfreqz_np(sos: np.ndarray, worN: np.ndarray) -> np.ndarray:
    """
    Compute complex frequency response of SOS cascade.

    sos: (n_sections, 6) → [b0 b1 b2 a0 a1 a2]
    worN: digital radian frequencies
    """
    z = np.exp(1j * worN)
    H = np.ones_like(z, dtype=np.complex128)

    for sec in sos:
        b0, b1, b2, a0, a1, a2 = sec
        num = b0 + b1 / z + b2 / z**2
        den = a0 + a1 / z + a2 / z**2
        H *= num / den

    return H


def _eq_objective_function(
    params_flat: np.ndarray,
    num_filters: int,
    f: np.ndarray,
    meas_resp_db: np.ndarray,
    target_resp_db: np.ndarray,
    ROI: Tuple[float, float],
    Fs: float,
) -> np.ndarray:
    """
    Residual vector for least-squares EQ optimization.
    """
    params = params_flat.reshape(num_filters, 3)

    filt_resp_db = compute_parametric_eq_response(params, f, Fs)
    output_resp_db = meas_resp_db + filt_resp_db

    # Restrict error to ROI
    idx = (f >= ROI[0]) & (f <= ROI[1])
    error = output_resp_db[idx] - target_resp_db[idx]

    return error


def _get_initial_gains(n: int, seed: int = 0) -> np.ndarray:
    """
    Generate initial gain values for EQ filters.
    Uses a fixed RNG seed so results are repeatable.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def _init_eq_parameters(
    num_filters: int,
    ROI: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize EQ parameter matrix init_params and bounds lb / ub.

    Parameter format per filter:
        [gain_dB, Q_or_slope, Fc]
    """
    fmin = max(20.0, ROI[0])
    fmax = min(20e3, ROI[1])

    # Center frequencies
    Fc_peaks = np.logspace(                 # All but last two are peaking filters
        np.log10(fmin),
        np.log10(fmax),
        num_filters - 2
    )
    Fc_shelves = np.array([                 # Last two filters are shelves (high and low)
        (2 * ROI[0] + ROI[1]) / 3,
        (ROI[0] + 2 * ROI[1]) / 3
    ])

    # Initial parameter matrix
    init_params = np.zeros((num_filters, 3))
    init_params[:, 0] = _get_initial_gains(num_filters)
    init_params[:-2, 1] = 1.5               # Q for peak filters
    init_params[-2:, 1] = 1.0               # slope for shelf filters
    init_params[:-2, 2] = Fc_peaks
    init_params[-2:, 2] = Fc_shelves

    # Lower bounds
    lb = np.zeros_like(init_params)
    lb[:, 0] = -24.0                        # gain lower bound
    lb[:-2, 1] = 0.2                        # Q for peak filters lower bound
    lb[-2:, 1] = 0.1                        # slope for shelf filters lower bound
    lb[:, 2] = fmin-1e-6                    # Fc lower bound

    # Upper bounds
    ub = np.zeros_like(init_params)
    ub[:, 0] = 20.0                         # gain upper bound
    ub[:-2, 1] = 17.3                       # Q for peak filters upper bound
    ub[-2:, 1] = 5.0                        # slope for shelf filters upper bound
    ub[:, 2] = fmax+1e-6                    # Fc upper bound

    return init_params, lb, ub


def _eq_optimizer(
    num_filters: int,
    f: np.ndarray,
    meas_resp_db: np.ndarray,
    target_resp_db: np.ndarray,
    ROI: Tuple[float, float],
    Fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize parametric EQ parameters.

    Returns:
        EQ         -> optimized parameter matrix (num_filters, 3)
        outputResp -> final equalized response (dB)
        filtResp   -> EQ filter response (dB)
    """
    init_params, lb, ub = _init_eq_parameters(num_filters, ROI)

    res = least_squares(
        _eq_objective_function,                                         # loss
        init_params.ravel(),                                            # initial parameters (flattened)
        bounds=(lb.ravel(), ub.ravel()),
        args=(num_filters, f, meas_resp_db, target_resp_db, ROI, Fs),   # additional args to compute loss function
        ftol=1e-8,
        max_nfev=min(500 * num_filters, 1000),
        verbose=1,
    )

    EQ_params = res.x.reshape(num_filters, 3)                           # Extract optimized (LS) EQ parameters
    filtResp = compute_parametric_eq_response(EQ_params, f, Fs)
    outputResp = meas_resp_db + filtResp

    return EQ_params, outputResp, filtResp


def _get_target_response_comp_EQ(cf: np.ndarray, oa: np.ndarray, ROI: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the linearized target response (in dB) for compensation EQ based on octave-averaged response.

    Args:
        cf: center frequencies (Hz) for octave bands (1D array)
        oa: octave-averaged magnitudes (linear) corresponding to cf (1D array)
        ROI: region of interest [fmin, fmax] (Hz)

    Returns:
        target_resp: target response in dB (same length as cf)
        pfit: array([intercept, slope]) of the linear fit in log-frequency domain (dB = intercept + slope * log(freq))
        pdb: octave-averaged response in dB (same length as cf)
    """
    # Defensive copy / ensure arrays
    cf = np.atleast_1d(cf).astype(float)
    oa = np.atleast_1d(oa).astype(float)

    # Convert to dB (avoid log(0))
    pdb = 20.0 * np.log10(oa + 1e-12)

    # Set the range of the linear fit (translated from MATLAB)
    lfCutOff = 2.5 * ROI[0]   # lowest frequency to linearize
    hfMaxFit = 0.6 * ROI[1]   # max frequency to fit for

    # Only consider positive center freqs for log
    valid_cf = cf > 0.0
    linIdx = valid_cf & (cf >= lfCutOff) & (cf <= hfMaxFit)

    # Initialize outputs
    pfit = np.array([0.0, 0.0], dtype=float)
    target_resp = np.copy(pdb)

    if np.any(linIdx):
        # Fit straight line in log-frequency domain: pdb = intercept + slope * log(cf)
        logx = np.log(cf[linIdx])
        y = pdb[linIdx]

        # Simple linear regression via polyfit (slope first, intercept second)
        slope, intercept = np.polyfit(logx, y, 1)
        pfit = np.array([intercept, slope])  # MATLAB ordering: pfit(1)=intercept, pfit(2)=slope

        # Build target response across all cf
        target_resp = pfit[0] + pfit[1] * np.log(cf)

        # Roll off the low frequencies, starting slightly above the linear range:
        lfcutoff = 1.05 * lfCutOff
        idx_low = cf < lfcutoff
        if np.any(idx_low):
            # attenuation amount: min(30, ROI[0]/2) * ((lfcutoff - cf)/lfcutoff)^2
            max_atten = min(30.0, ROI[0] / 2.0)
            frac = (lfcutoff - cf[idx_low]) / lfcutoff
            rolloff = max_atten * (frac ** 2)
            target_resp[idx_low] = target_resp[idx_low] - rolloff
    else:
        # No suitable linear region: keep pdb as target and leave pfit zeros
        target_resp = np.copy(pdb)
        pfit = np.array([0.0, 0.0])

    return target_resp, pfit, pdb


def _octave_average(f: np.ndarray, resp: np.ndarray, bpo: int, 
                   freq_range: Tuple[float, float] = None, b_smooth: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute average power per fractional octave band.
    
    Based on MATLAB's octaveAverage function. Uses constant bandwidth below 
    300-400 Hz (Bark scale approximation) and octave spacing above.
    
    Args:
        f: Frequency vector (Hz)
        resp: Magnitude response (linear, not dB)
        bpo: Bands per octave (e.g., 1 for 1-octave, 3 for 1/3-octave, 6 for 1/6-octave)
        freq_range: Frequency range [fmin, fmax]. If None, uses full range
        b_smooth: Apply smoothing to output (recommended for noisy responses)
    
    Returns:
        oa: Averaged octave magnitudes
        cf: Center frequencies (Hz)
    """
    # Ensure row vectors
    f = np.atleast_1d(f).astype(float)
    resp = np.atleast_1d(resp).astype(float)
    
    # Apply frequency range filter
    if freq_range is not None:
        idx = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[idx]
        resp = resp[idx]
    
    # If range is empty
    if len(f) == 0:
        return np.array([]), np.array([])
    
    # Compute center frequencies (cf) and band-edge frequencies (bef)
    G = 10 ** (3/10)  # ~2.0 (octave ratio)
    ref_freq = f[0]
    f_range = np.array([f[0], f[-1]])
    lgbg = bpo / np.log(G)
    
    # Use a constant bandwidth below 400 Hz (based on Zwicker's Bark scale resolution)
    octave_cutoff = 300  # Hz, set between 120 and 400
    f_min = int(np.round(lgbg * np.log(octave_cutoff / ref_freq)))
    f_max = int(np.floor(lgbg * np.log(f_range[1] / ref_freq)))
    
    # Octave-spaced center frequencies above cutoff
    cf = ref_freq * (G ** (np.arange(f_min, f_max + 1) / bpo))
    
    # Constant bandwidth section below cutoff
    last_cst_cf = ref_freq * (G ** (f_min / bpo))
    lf_bw = ref_freq * (G ** (f_min / bpo) - G ** ((f_min - 1) / bpo))
    nb_cst = 1 + int(np.ceil((last_cst_cf - f_range[0]) / lf_bw))
    
    if nb_cst > 1:
        lf_cf = np.linspace(f_range[0], last_cst_cf, nb_cst)
        cf = np.concatenate([lf_cf[:-1], cf])
    
    # Band-edge frequencies
    bef = np.concatenate([cf[:-1] * (G ** (1 / (2 * bpo))), [f[-1]]])
    
    oa = np.full(len(cf), np.nan)
    
    # Average each octave band
    f_beg = 0
    for ii in range(len(cf)):
        f_end_indices = np.where(f <= bef[ii])[0]
        if len(f_end_indices) > 0:
            f_end = f_end_indices[-1] + 1
            oa[ii] = np.nanmean(resp[f_beg:f_end])
            f_beg = f_end
    
    # Handle empty bins (when octave bandwidth > available frequency points)
    idx = np.where(~np.isnan(oa[:-1]) & np.isnan(oa[1:]))[0]
    for ii in idx:
        oa_idx = ii
        f_end_indices = np.where(~np.isnan(oa[oa_idx + 1:]))[0]
        if len(f_end_indices) > 0:
            f_end = f_end_indices[0]
            # Geometric mean of center frequencies
            cf[oa_idx] = np.prod(cf[oa_idx:oa_idx + f_end]) ** (1 / f_end)
        else:
            break
    
    # Remove empty bins
    idx = ~np.isnan(oa)
    oa = oa[idx]
    cf = cf[idx]
    
    # Apply smoothing if requested (for measured/noisy responses)
    if b_smooth:
        # First smoothing pass (3-point average)
        oa[1:-1] = (1*oa[1:-1] + oa[:-2] + oa[2:]) / 3
        # Second smoothing pass (5-point average)
        if len(oa) > 4:
            oa[2:-2] = (2*oa[2:-2] + oa[:-4] + oa[4:]) / 4
    
    return oa, cf


def octave_average_torch(f: torch.Tensor, resp: torch.Tensor, bpo: int, 
                         freq_range: Tuple[float, float] = None, b_smooth: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute average power per fractional octave band (PyTorch version).
    
    Preserves complex values (e.g., for phase information in frequency responses).
    Works with PyTorch tensors on GPU/CPU.
    
    Based on MATLAB's octaveAverage function. Uses constant bandwidth below 
    300-400 Hz (Bark scale approximation) and octave spacing above.
    
    Args:
        f: Frequency vector (Hz) as torch tensor
        resp: Response (complex or real) as torch tensor
        bpo: Bands per octave (e.g., 1 for 1-octave, 3 for 1/3-octave, 6 for 1/6-octave)
        freq_range: Frequency range [fmin, fmax]. If None, uses full range
        b_smooth: Apply smoothing to output (recommended for noisy responses)
    
    Returns:
        resp_averaged: Averaged octave response (complex or real, depending on input)
        cf: Center frequencies (Hz)
    """
    device = f.device
    dtype = resp.dtype
    
    # Apply frequency range filter
    if freq_range is not None:
        mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[mask]
        resp = resp[mask]
    
    # If range is empty
    if len(f) == 0:
        return torch.tensor([], device=device, dtype=dtype), torch.tensor([], device=device)
    
    # Compute center frequencies (cf) and band-edge frequencies (bef)
    G = 10 ** (3/10)  # ~2.0 (octave ratio)
    ref_freq = f[0].item()+1e-6
    f_min_val = f[0].item()
    f_max_val = f[-1].item()
    lgbg = bpo / np.log(G)
    
    # Use a constant bandwidth below 400 Hz (based on Zwicker's Bark scale resolution)
    octave_cutoff = 300  # Hz, set between 120 and 400
    f_min = int(np.round(lgbg * np.log(octave_cutoff / ref_freq)))
    f_max = int(np.floor(lgbg * np.log(f_max_val / ref_freq)))
    
    # Octave-spaced center frequencies above cutoff
    cf = ref_freq * (G ** (np.arange(f_min, f_max + 1) / bpo))
    cf = torch.tensor(cf, device=device, dtype=torch.float32)
    
    # Constant bandwidth section below cutoff
    last_cst_cf = ref_freq * (G ** (f_min / bpo))
    lf_bw = ref_freq * (G ** (f_min / bpo) - G ** ((f_min - 1) / bpo))
    nb_cst = 1 + int(np.ceil((last_cst_cf - f_min_val) / lf_bw))
    
    if nb_cst > 1:
        lf_cf = torch.linspace(f_min_val, last_cst_cf, nb_cst, device=device)
        cf = torch.cat([lf_cf[:-1], cf])
    
    # Band-edge frequencies
    bef = torch.cat([cf[:-1] * (G ** (1 / (2 * bpo))), torch.tensor([f_max_val], device=device)])
    
    resp_averaged = torch.full((len(cf),), torch.nan, device=device, dtype=dtype)
    
    # Average each octave band
    f_beg = 0
    for ii in range(len(cf)):
        f_end_indices = torch.where(f <= bef[ii])[0]
        if len(f_end_indices) > 0:
            f_end = f_end_indices[-1] + 1
            resp_averaged[ii] = torch.nanmean(resp[f_beg:f_end])
            f_beg = f_end
    
    # Remove empty bins
    idx = ~torch.isnan(resp_averaged)
    resp_averaged = resp_averaged[idx]
    cf = cf[idx]
    
    # Apply smoothing if requested
    if b_smooth and len(resp_averaged) > 4:
        # First smoothing pass (3-point average)
        resp_averaged[1:-1] = (1*resp_averaged[1:-1] + resp_averaged[:-2] + resp_averaged[2:]) / 3
        # Second smoothing pass (5-point average)
        resp_averaged[2:-2] = (2*resp_averaged[2:-2] + resp_averaged[:-4] + resp_averaged[4:]) / 4
    
    return resp_averaged, cf


def get_compensation_EQ_params(rir: np.ndarray, sr: int, ROI: Tuple[float, float]=(20.0, 20000.0), num_sections: int=6) -> dict:
    """Estimate Parametric EQ parameters to compensate for the given RIR.

    This function analyzes the RIR
    and compute suitable EQ parameters to flatten its frequency response.
    """
    nfft = len(rir)
    freq_response = np.abs(np.fft.rfft(rir, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1/sr)

    # Apply octave averaging (no smoothing for filter responses)
    oa, cf = _octave_average(freqs, freq_response, bpo=24, freq_range=ROI, b_smooth=False)

    # Compute target response for compensation EQ
    target_resp, pfit, pdb = _get_target_response_comp_EQ(cf, oa, ROI)

    # Log the optimization start
    print(f"Starting initial EQ optimization with {num_sections} filters over ROI {ROI[0]}-{ROI[1]} Hz...")

    # Optimize parametric EQ to match target response
    EQ_params, out_resp_db, filt_resp_db = _eq_optimizer(
    num_filters=num_sections,                               # 6 filters for dasp_pytorch parametric eq
    f=cf,
    meas_resp_db=pdb,
    target_resp_db=target_resp,
    ROI=ROI,
    Fs=sr,
    )

    # Optional: plot results and save figure
    #plt.figure()
    #plt.semilogx(freqs, 20*np.log10(freq_response + 1e-12), label="RIR Frequency Response")
    #plt.semilogx(cf, 20*np.log10(oa + 1e-12), label="Octave-Averaged Response")
    #plt.semilogx(cf, target_resp, label="Target Compensation Response", linestyle='--')
    #plt.semilogx(cf, out_resp_db, label="Equalized Response", linestyle='-.')
    #plt.semilogx(cf, filt_resp_db, label="EQ Filter Response", linestyle=':')
    #plt.axvline(ROI[0], color="red", linestyle="--", label="ROI Limits" if ROI else "")
    #plt.axvline(ROI[1], color="red", linestyle="--" if ROI else "")
    #plt.legend(), plt.xlabel("Frequency (Hz)"), plt.ylabel("Magnitude (dB)")
    #plt.title("Room Frequency Response and Target Compensation EQ")
    #plt.savefig("RFR_compensation.png",dpi=150, bbox_inches='tight')

    # Build parameter dictionary for downstream use (e.g., PyTorch EQ)
    eq_param_dict = _seven_biquad_eq_params_to_dict(EQ_params)

    return {
        "eq_params": eq_param_dict,
        "EQ_matrix": EQ_params,
        "measured_response_db": pdb,
        "frq_axis_full": freqs,
        "target_response_db": target_resp,
        "equalized_response_db": out_resp_db,
        "filter_response_db": filt_resp_db,
        "freq_axis_smoothed": cf,
    }


def build_target_response_lin_phase(sr: int, response_type: str = "delay_only",
                           target_mag_resp: np.ndarray = None, target_mag_freqs: np.ndarray = None, 
                           fir_len: int = 1024, ROI: tuple = None, rolloff_octaves: float = 1.0,
                           device: torch.device = None):
    """Build a desired response FIR filter with optional target magnitude response.
    
    Args:
        sr: Sample rate in Hz
        response_type: Type of response - "delay_only" or "delay_and_mag"
        target_mag_resp: Target magnitude response in dB (1D array), required if response_type="delay_and_mag"
        target_mag_freqs: Frequency axis for target_mag_resp in Hz (1D array), required if response_type="delay_and_mag"
        fir_len: Length of the FIR filter for magnitude shaping (default 1024)
        ROI: Region of interest as (low_freq, high_freq) in Hz. Frequencies outside will be attenuated.
        rolloff_octaves: Width of the rolloff transition in octaves (default 1.0)
        device: Torch device for output tensor
        
    Returns:
        target_response: Tensor of shape (1, 1, N) representing the desired impulse response
    """
    if device is None:
        device = torch.device("cpu")
    
    if response_type == "delay_only":
        # Simple Kronecker delta (no delay - add delay externally if needed)
        target_response = torch.zeros(1, 1, fir_len, device=device)
        target_response[:, :, 0] = 1.0
        
    elif response_type == "delay_and_mag":
        if target_mag_resp is None or target_mag_freqs is None:
            raise ValueError("target_mag_resp and target_mag_freqs are required for response_type='delay_and_mag'")
        
        # Interpolate target magnitude response to FFT bins
        fft_freqs = np.fft.rfftfreq(fir_len, d=1.0/sr)

        # Build a log-spaced frequency axis with the same number of points as fft_freqs.
        # DC (0 Hz) cannot be represented on a log scale, so log_freqs spans
        # [fft_freqs[1], fft_freqs[-1]] — the first non-zero bin to Nyquist.
        n_fft = len(fft_freqs)
        f_min_log = fft_freqs[1]          # first non-zero FFT bin
        f_max_log = fft_freqs[-1]         # Nyquist
        log_freqs = np.logspace(np.log10(f_min_log), np.log10(f_max_log), n_fft)

        # Interpolate target magnitude response (in dB) to log-spaced frequency axis
        target_mag_interp_db = np.interp(log_freqs, target_mag_freqs, target_mag_resp)

        # Apply ROI rolloff on the log-spaced frequency axis.
        # The target_mag_resp is kept as-is within ROI; decay starts exactly at ROI boundaries.
        if ROI is not None:
            f_low, f_high = ROI
            rolloff_mask_db = np.zeros_like(log_freqs)

            # Low-frequency decay (below f_low)
            # Decay starts at f_low (0 dB) and reaches -120 dB at f_low_end
            f_low_end = f_low / (2 ** rolloff_octaves)
            for i, f in enumerate(log_freqs):
                if f < f_low_end:
                    rolloff_mask_db[i] = -120  # Full attenuation below rolloff end
                elif f < f_low:
                    # Smooth cosine transition from 0 dB (at f_low) to -120 dB (at f_low_end)
                    # t=0 at f_low, t=1 at f_low_end
                    t = np.log2(f_low / f) / rolloff_octaves
                    rolloff_mask_db[i] = -120 * 0.5 * (1 - np.cos(np.pi * t))

            # High-frequency decay (above f_high)
            # Decay starts at f_high (0 dB) and reaches -120 dB at f_high_end
            f_high_end = f_high * (2 ** rolloff_octaves)
            for i, f in enumerate(log_freqs):
                if f > f_high_end:
                    rolloff_mask_db[i] = -120  # Full attenuation above rolloff end
                elif f > f_high:
                    # Smooth cosine transition from 0 dB (at f_high) to -120 dB (at f_high_end)
                    # t=0 at f_high, t=1 at f_high_end
                    t = np.log2(f / f_high) / rolloff_octaves
                    rolloff_mask_db[i] = -120 * 0.5 * (1 - np.cos(np.pi * t))

            target_mag_interp_db = target_mag_interp_db + rolloff_mask_db

        # Re-interpolate from log-spaced back to the linear FFT frequency axis (ready for IFFT).
        # DC bin (0 Hz) is out of range of log_freqs, so np.interp clamps it to the
        # value at log_freqs[0] (which is already heavily attenuated by the rolloff).
        target_mag_interp_db = np.interp(fft_freqs, log_freqs, target_mag_interp_db)

        # Convert from dB to linear magnitude
        target_mag_interp_linear = 10 ** (target_mag_interp_db / 20.0)
        
        # Create linear-phase FIR filter via IFFT (zero-phase, then shift for causality)
        H_mag = torch.from_numpy(target_mag_interp_linear).float().to(device)
        
        # Create zero-phase impulse response via IFFT
        h_zerophase = torch.fft.irfft(H_mag, n=fir_len)
        
        # Shift to make causal (linear phase) by circular shift
        h_causal = torch.roll(h_zerophase, fir_len // 2)
        
        # Apply a window to smooth the filter
        fir_window = torch.hann_window(fir_len, device=device)
        h_windowed = h_causal * fir_window
        
        # Normalize filter to unity gain at passband
        h_windowed = h_windowed / h_windowed.abs().max()
        
        target_response = h_windowed.view(1, 1, -1)
        
    else:
        raise ValueError(f"Unknown response_type: {response_type}. Use 'delay_only' or 'delay_and_mag'")
    
    return target_response


def interp_to_log_freq(mag_db, freqs_lin, n_points=None, f_min=None, f_max=None):
    """Interpolate magnitude response from linear to log-spaced frequency bins.
    
    This enables perceptually-uniform smoothing: lower frequencies get more detail,
    higher frequencies get less, matching human hearing characteristics.
    
    Args:
        mag_db: Magnitude response in dB (1D tensor on linear freq bins)
        freqs_lin: Linear frequency axis (1D tensor)
        n_points: Number of log-spaced points (default: same as input)
        f_min: Minimum frequency (default: first non-zero freq)
        f_max: Maximum frequency (default: last freq)
    
    Returns:
        mag_db_log: Magnitude response at log-spaced frequencies
        freqs_log: Log-spaced frequency axis
    """
    if n_points is None:
        n_points = len(freqs_lin)
    if f_min is None:
        f_min = freqs_lin[freqs_lin > 0][0].item()  # Skip DC
    if f_max is None:
        f_max = freqs_lin[-1].item()
    
    # Create log-spaced frequency points
    freqs_log = torch.logspace(
        torch.log10(torch.tensor(f_min, device=mag_db.device)),
        torch.log10(torch.tensor(f_max, device=mag_db.device)),
        n_points,
        device=mag_db.device
    )
    
    # Linear interpolation: find indices and weights
    indices = torch.searchsorted(freqs_lin.contiguous(), freqs_log.contiguous())
    indices = torch.clamp(indices, 1, len(freqs_lin) - 1)  # Ensure valid indices
    
    # Get surrounding values and frequencies
    idx_low = indices - 1
    idx_high = indices
    
    f_low = freqs_lin[idx_low]
    f_high = freqs_lin[idx_high]
    mag_low = mag_db[idx_low]
    mag_high = mag_db[idx_high]
    
    # Linear interpolation weights
    weights = (freqs_log - f_low) / (f_high - f_low + 1e-10)
    weights = torch.clamp(weights, 0, 1)
    
    # Interpolate
    mag_db_log = mag_low + weights * (mag_high - mag_low)
    
    return mag_db_log, freqs_log


def _movmean_1d(x: torch.Tensor, L_before: int, L_after: int) -> torch.Tensor:
    """1D moving average with asymmetric window.

    Uses edge-aware normalization so that near the boundaries the average
    is taken over the valid samples only (no zero-padding bias).
    """
    if x.ndim != 1:
        x = x.view(-1)
    N = x.numel()
    if N == 0:
        return x.clone()

    L_before = max(int(L_before), 0)
    L_after = max(int(L_after), 0)

    # Cumulative sum with a leading zero for convenient segment sums
    cumsum = torch.cumsum(F.pad(x, (1, 0)), dim=0)  # (N+1,)
    idx = torch.arange(N, device=x.device)

    left = torch.clamp(idx - L_before, 0, N - 1)
    right = torch.clamp(idx + L_after, 0, N - 1)

    seg_sum = cumsum[right + 1] - cumsum[left]
    count = (right - left + 1).to(x.dtype)
    return seg_sum / count


def kirkeby_deconvolve(x, y, nfft, sr, ROI):
    """Kirkeby deconvolution to compute stable inverse filter in frequency domain.
    Args:
        x:   input signal (1D tensor), excitation
        y:   output signal (1D tensor)
        nfft: FFT size for deconvolution (should be >= len(x) + len(y) - 1)
        sr:  sample rate in Hz
        ROI: (f_low, f_high) in Hz, frequency band of interest

    Returns:
        H: Estimated complex frequency response (1D tensor of length nfft//2 + 1)
    """
    X = torch.fft.rfft(x, n=nfft)
    Y = torch.fft.rfft(y, n=nfft)

    N = X.shape[-1]  # number of rfft bins = floor(nfft/2) + 1
    device = X.device
    dtype_real = X.real.dtype

    # Build Xflat, flat mask with smooth rolloff outside ROI)
    Xflat = torch.zeros(N, device=device, dtype=dtype_real)
    if ROI is not None:
        f_low, f_high = float(ROI[0]), float(ROI[1])
        # Map frequency to bin index using full-FFT convention (same as MATLAB): k * sr / nfft
        f0_bin = int(math.ceil((f_low / sr) * nfft))
        f1_bin = int(math.floor((f_high / sr) * nfft))
    else:
        # If no ROI is given, consider the full band
        f0_bin, f1_bin = 0, N - 1
    # Clamp to valid rfft bin range
    f0_bin = max(0, min(f0_bin, N - 1))
    f1_bin = max(0, min(f1_bin, N - 1))
    if f1_bin >= f0_bin:
        Xflat[f0_bin:f1_bin + 1] = 1.0

    # Smoothing windows, following LL and SL logic from MATLAB
    LL = max(1, int(round(N * 1e-1)))  # long window
    SL = max(1, int(round(N * 1e-3)))  # short window

    Xflat_LL_SL = _movmean_1d(Xflat, LL, SL)
    Xflat_SL_LL = _movmean_1d(Xflat, SL, LL)
    Xflat_smooth = torch.minimum(Xflat_LL_SL, Xflat_SL_LL)

    mflat = torch.max(Xflat_smooth)

    if mflat <= 0:
        # Degenerate case: ROI does not overlap the rfft band; fall back to small constant epsilon
        epsilon = torch.full_like(Xflat_smooth, 1e-8)
    else:
        # Frequency-dependent epsilon (from MATLAB implementation)
        epsilon = torch.maximum(1e-4 * mflat, 0.38 * mflat - Xflat_smooth)

    # Apply Kirkeby-style deconvolution filter with frequency-dependent epsilon.
    # Suppress functorch's performance warning about missing batching rule for aten::_conj_physical,
    # which otherwise clutters the console.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="There is a performance drop because we have not yet implemented the batching rule for aten::_conj_physical",
            category=UserWarning,
        )
        denom = X * torch.conj(X) + epsilon.to(X.dtype)
        H = (Y * torch.conj(X)) / denom

    return H


def squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute element-wise squared error between predicted and true signals.
    
    Args:
        y_pred: Predicted signal tensor
        y_true: True signal tensor
    Returns:
        Element-wise squared error tensor
    """
    return (y_pred - y_true) ** 2


def _safe_complex_abs(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Compute magnitude without relying on complex torch.abs CUDA JIT kernels."""
    if torch.is_complex(x):
        re = torch.real(x)
        im = torch.imag(x)
        return torch.sqrt(re * re + im * im + eps)
    return torch.abs(x)


def update_LEM(current_time_s, n_rirs, transition_times_s, rirs_tensors):
    """Update LEM based on current time and transition schedule.
    
    Args:
        current_time_s: Current time in seconds
        n_rirs: Number of RIRs
        transition_times_s: List of (start_time, end_time) tuples for each transition
        rirs_tensors: List of RIR tensors
    
    Returns:
        LEM: Updated LEM tensor of shape (1, 1, -1)
    """
    if n_rirs <= 1:
        return rirs_tensors[0].view(1, 1, -1)
    
    # Find which RIR segment we're in and compute interpolation
    current_rir_idx = 0
    in_transition = False
    alpha = 0.0  # interpolation factor (0 = previous RIR, 1 = current RIR)
    
    for i, (t_start, t_end) in enumerate(transition_times_s):
        if current_time_s >= t_end:
            # Past this transition entirely - use the new RIR
            current_rir_idx = i + 1
            in_transition = False
        elif current_time_s >= t_start:
            # In the middle of this transition
            current_rir_idx = i + 1
            in_transition = True
            if t_end > t_start:  # smooth transition
                alpha = (current_time_s - t_start) / (t_end - t_start)
            else:  # abrupt (t_start == t_end)
                alpha = 1.0  # instant jump to new RIR
            break
    
    # Interpolate between RIRs if in transition
    if in_transition and current_rir_idx > 0:
        prev_rir = rirs_tensors[current_rir_idx - 1]
        curr_rir = rirs_tensors[current_rir_idx]
        return interpolate_IRs(alpha, prev_rir, curr_rir)
    else:
        return rirs_tensors[current_rir_idx].view(1, 1, -1)


def interpolate_IRs(alpha, prev_rir: torch.Tensor, curr_rir: torch.Tensor) -> torch.Tensor:
    """
    Interpolate between two RIRs by interpolating magnitude (in dB) and unwrapped phase.

    Args:
        alpha: scalar in [0,1] (0 -> prev_rir, 1 -> curr_rir). Can be Python float or 0-dim tensor.
        prev_rir: 1-D torch tensor (num_samples,) or 2/3-D variant (we'll squeeze). On device.
        curr_rir: same shape as prev_rir or will be padded to match.

    Returns:
        interpolated_ir: torch.Tensor shaped (1, 1, N) where N == max(len(prev_rir), len(curr_rir))
    """
    # Convert alpha to float on CPU (safe) or to tensor on device
    if isinstance(alpha, torch.Tensor):
        alpha_val = float(alpha.item())
    else:
        alpha_val = float(alpha)

    # Squeeze input RIRs to 1-D
    x1 = prev_rir.detach().squeeze().to(torch.get_default_dtype())
    x2 = curr_rir.detach().squeeze().to(torch.get_default_dtype())

    # Move to same device
    device = prev_rir.device if isinstance(prev_rir, torch.Tensor) else torch.device("cpu")
    x1 = x1.to(device)
    x2 = x2.to(device)

    # Pad to same length if necessary
    n1 = x1.numel()
    n2 = x2.numel()
    n = max(n1, n2)
    if n1 < n:
        x1 = F.pad(x1, (0, n - n1))
    if n2 < n:
        x2 = F.pad(x2, (0, n - n2))

    # FFT size (use same length as time-domain signals)
    nfft = n

    # Compute complex frequency responses (rfft)
    H1 = torch.fft.rfft(x1, n=nfft)
    H2 = torch.fft.rfft(x2, n=nfft)

    eps = 1e-12

    # Magnitudes and phases
    mag1 = _safe_complex_abs(H1)
    mag2 = _safe_complex_abs(H2)

    # Convert magnitude to dB for interpolation (more perceptually linear)
    mag_db1 = 20.0 * torch.log10(mag1 + eps)
    mag_db2 = 20.0 * torch.log10(mag2 + eps)

    # Phases (atan2)
    phase1 = torch.atan2(H1.imag, H1.real)
    phase2 = torch.atan2(H2.imag, H2.real)

    # Unwrap phases along frequency axis
    # rfft returns length floor(nfft/2)+1 bins — unwrap works on that 1D array
    phase1_un = _unwrap_phase(phase1)
    phase2_un = _unwrap_phase(phase2)

    # Interpolate magnitude-in-dB and unwrapped phase
    mag_db_interp = (1.0 - alpha_val) * mag_db1 + alpha_val * mag_db2
    phase_interp = (1.0 - alpha_val) * phase1_un + alpha_val * phase2_un

    # Reconstruct complex spectrum from mag+phase
    mag_lin = 10.0 ** (mag_db_interp / 20.0)

    real = mag_lin * torch.cos(phase_interp)
    imag = mag_lin * torch.sin(phase_interp)
    H_interp = torch.complex(real, imag)

    # IFFT back to time domain (real signal)
    h_interp = torch.fft.irfft(H_interp, n=nfft)

    # Return as shape (1,1,-1) to match how LEM is used elsewhere
    return h_interp.view(1, 1, -1).to(device)


def _unwrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """
    Unwrap a 1-D phase tensor (radians) along its only axis.
    Implements the same idea as numpy.unwrap: replace jumps > pi by their 2*pi complement.
    """
    # phase: 1-D tensor (N,)
    if phase.numel() <= 1:
        return phase.clone()
    diff = phase[1:] - phase[:-1]                       # (N-1,)
    # wrap diffs into (-pi, pi]
    two_pi = 2.0 * math.pi
    wrapped = (diff + math.pi) % two_pi - math.pi      # (N-1,)
    # correction to apply to each subsequent element: wrapped - diff
    corr = torch.cat([torch.tensor([0.], device=phase.device, dtype=phase.dtype),
                      torch.cumsum(wrapped - diff, dim=0)])
    return phase + corr


#########################
# Main experiment logic #
#########################

def load_rirs(rir_dir: Path, max_n: int = None, normalize: bool = False) -> Tuple[List[np.ndarray], List[int]]:
    """Load all wav files in a directory as float32 RIRs. Returns (rirs, srs).

    RIRs are returned as lists of 1-D numpy arrays. Sample rates are returned
    for each file so the caller can check/resample if needed.

    Args:
        rir_dir: Directory containing .wav RIR files.
        max_n: If set, only the first max_n files (sorted) are loaded.
        normalize: If True, all RIRs are divided by sum(|rir_0|), where rir_0
                   is the first RIR in the returned list.
    """
    files = sorted([p for p in rir_dir.glob("*.wav")])
    if max_n is not None:
        files = files[:max_n]
    rirs = []
    srs = []
    for f in files:
        if sf is not None:
            data, sr = sf.read(str(f))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
        else:
            wav, sr = torchaudio.load(str(f))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=False)
            else:
                wav = wav.squeeze(0)
            data = wav.detach().cpu().numpy().astype(np.float32)
        rirs.append(data)
        srs.append(sr)
    if normalize and len(rirs) > 0:
        norm_factor = np.sum(np.abs(rirs[0]))
        if norm_factor > 0:
            rirs = [rir / norm_factor for rir in rirs]
    return rirs, srs


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON configuration for experiment_04.

    The JSON is expected to contain:
      - "simulation_params": {param_name: [values, ...]}
      - "input": {"use_white_noise": bool, "use_songs_folder": bool, "max_audio_len_s": [values, ...]}
    """
    with config_path.open("r") as f:
        cfg = json.load(f)
    return cfg


def iter_param_grid(param_grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    """Yield all combinations from a simple parameter grid.

    param_grid is a dict mapping param name -> iterable of candidate values.
    """
    keys = list(param_grid.keys())
    values_product = itertools.product(*(param_grid[k] for k in keys))
    for values in values_product:
        yield {k: v for k, v in zip(keys, values)}


def discover_input_signals(input_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Discover which input signals should be used.

        Returns a list of (mode, info) tuples where mode is one of:
            - "white_noise": info contains {"max_audio_len_s": float, "realization_idx": int, "seed_offset": int}
      - "song": info contains {"path": Path, "max_audio_len_s": float}
    """
    modes: List[Tuple[str, Dict[str, Any]]] = []

    use_white_noise = bool(input_cfg.get("use_white_noise", False))
    use_songs_folder = bool(input_cfg.get("use_songs_folder", False))
    max_len_list = input_cfg.get("max_audio_len_s", [None])
    max_num_songs = input_cfg.get("max_num_songs", None)

    # For now, take the first max_audio_len_s if multiple are given; it will
    # be combined with the simulation params grid separately.
    max_audio_len_s = max_len_list[0] if max_len_list else None

    if use_white_noise:
        # Run multiple independent white-noise realizations, analogous to per-song runs.
        # Reuse max_num_songs as the number of realizations (fallback to 1).
        n_white_noise = 1
        if max_num_songs is not None:
            try:
                n_white_noise = max(1, int(max_num_songs))
            except (TypeError, ValueError):
                n_white_noise = 1

        for i in range(n_white_noise):
            modes.append((
                "white_noise",
                {
                    "max_audio_len_s": max_audio_len_s,
                    "realization_idx": i,
                    "seed_offset": i,
                },
            ))

    if use_songs_folder:
        songs_dir = root / "data" / "audio" / "input" / "songs"
        if songs_dir.is_dir():
            all_songs = [p for p in sorted(songs_dir.iterdir()) if p.is_file()]

            # Randomly sample up to max_num_songs from the available tracks
            if max_num_songs is not None:
                try:
                    n = int(max_num_songs)
                except (TypeError, ValueError):
                    n = None
            else:
                n = None

            if n is not None and n > 0 and n < len(all_songs):
                selected_songs = random.sample(all_songs, n)
            else:
                selected_songs = all_songs

            for p in selected_songs:
                modes.append(("song", {"path": p, "max_audio_len_s": max_audio_len_s}))

    return modes


def build_step_sizes(mu_opt: float, EQG_params_shape: torch.Size, device: torch.device) -> torch.Tensor:
    step_sizes = mu_opt * torch.ones(EQG_params_shape, device=device)
    step_sizes[:,-1] = step_sizes[:,-1] * 1e2  # Scale step size for gain parameter
    return step_sizes


def run_control_experiment(sim_cfg: Dict[str, Any], input_spec: Tuple[str, Dict[str, Any]]) -> None:
    """ Code to run the ARE experiment given the configuration and inpput specifications
    """
    exp_t_start = time.perf_counter()
    mode, info = input_spec
    print("\n=== Running control experiment ===")
    print(f"Mode: {mode}")
    print(f"Input info: {info}")

    # Load config variables
    input_type = input_spec[0]
    max_audio_len_s = input_spec[1]["max_audio_len_s"]
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
    debug_plot_flag = sim_cfg["debug_plot"]
    n_checkpoints = sim_cfg.get("n_checkpoints", 0)
    debug_plot_state = {} if debug_plot_flag else None

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Acoustic path from actuator (speaker) to sensor (microphone)
    rir_dir = Path(sim_cfg.get("rir_dir", root / "data" / "rir"))
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs, normalize=False)
    rir_init = rirs[0]
    sr = rirs_srs[0]
    
    # Precompute RIR tensors and transition times for time-varying scenarios
    if n_rirs > 1:
        rirs_tensors = [torch.from_numpy(rir).float().to(device) for rir in rirs]
        max_rir_len = max(rir.shape[0] for rir in rirs_tensors)
        rirs_tensors = [F.pad(rir, (0, max_rir_len - rir.shape[0])) for rir in rirs_tensors]
        
        # Compute transition start/end times (in seconds)
        segment_duration_s = max_audio_len_s / n_rirs
        transition_times_s = [] 
        for i in range(1, n_rirs):
            transition_start_s = i * segment_duration_s
            assert (i==1) or (transition_start_s >= transition_times_s[-1][-1])
            transition_end_s = transition_start_s + sim_cfg["transition_time_s"]
            transition_times_s.append((transition_start_s, min(transition_end_s, max_audio_len_s)))

    # Compute target response
    lem_delay = get_delay_from_ir(rir_init, sr)
    EQ_comp_dict = get_compensation_EQ_params(rir_init, sr, ROI, num_sections=7)
    target_mag_resp = EQ_comp_dict["target_response_db"]
    target_mag_freqs = EQ_comp_dict["freq_axis_smoothed"]

    # Initialize the LEM estimate (assume LEM is well-identified)
    LEM = torch.from_numpy(rir_init).view(1,1,-1).to(device)

    # Initialize differentiable EQ
    EQ = ParametricEQ(sample_rate=sr)
    init_params_tensor = torch.rand(1,EQ.num_params)
    #init_params_tensor = torch.ones(1,EQ.num_params)*0.5
    #init_params_tensor[0, [3, 6, 9, 12, 15]] = 1.0
    #dasp_param_dict = { k: torch.as_tensor(v, dtype=torch.float32).view(1) for k, v in EQ_comp_dict["eq_params"].items() }
    #_, init_params_tensor = EQ.clip_normalize_param_dict(dasp_param_dict) # initial normalized parameter vector
    EQ_memory = 128 # TODO: hardcoded for now (should be greater than 0)

    # Initialize learnable gain
    G = Gain(sample_rate=sr)
    # Combine EQ params and gain into one parameter vector: EQG_params = [EQ_params | G_param]
    # EQ_params occupies EQG_params[:, :-1], G_param (dB) occupies EQG_params[:, -1:]
    EQG_params = torch.nn.Parameter(
        torch.cat([init_params_tensor.clone(), torch.zeros(1, 1)], dim=-1).to(device)
    )

    # Load/synthesise the input audio (as torch tensors)
    if input_type == "white_noise":
        T = int(max_audio_len_s * sr)
        input = torch.randn(T, device=device)

    else:
        audio_path = input_spec[1]["path"]
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        input, audio_sr = torchaudio.load(audio_path)
        
        if input.shape[0] > 1:
            input = input.mean(dim=0) # convert to mono
        else:
            input = input.squeeze(0)
        
        if audio_sr != sr:  # match sample rates
            resampler = torchaudio.transforms.Resample(orig_freq=audio_sr, new_freq=sr)
            input = resampler(input)
            print(f"Resampled audio from {audio_sr} Hz to {sr} Hz")
        
        input = input.to(device)
        
        if max_audio_len_s is not None:
            max_samples = int(max_audio_len_s * sr)
            if len(input) > max_samples:
                input = input[:max_samples]     # truncate to max length
                print(f"Truncated audio to first {max_audio_len_s} seconds")

        T = len(input)
        T_seconds = T / sr
        print(f"Input signal: {input_type} ({T_seconds:.2f} s, {T} samples)")
    input = input / input.abs().max()   # normalization
    input = input.view(1,1,-1).to(device)

    # Set optimization (adaptive filtering)
    match optim_type:
        case "SGD":
            optimizer = torch.optim.SGD([EQG_params], lr=mu_opt)
        case "Adam":
            optimizer = torch.optim.Adam([EQG_params], lr=mu_opt, betas=(0.9, 0.99999)) # TODO: debug
        case "Muon":
            raise ValueError("Muon optimizer requires newer PyTorch version.")
            optimizer = torch.optim.Muon([EQG_params], lr=mu_opt)
        case "GHAM-1" | "GHAM-2":
            alpha_ridge = 5e0
            ridge_regressor = Ridge(alpha = alpha_ridge, fit_intercept = False)
            match loss_type:
                case "TD-MSE" | "FD-MSE":
                    jac_fcn = jacrev(params_to_loss, argnums=0, has_aux=False)
                case "TD-SE" | "FD-SE":
                    jac_fcn = jacfwd(params_to_loss, argnums=0, has_aux=False)
        case "Newton" | "GHAM-3" | "GHAM-4":
            alpha_ridge = 5e0
            ridge_regressor = Ridge(alpha = alpha_ridge, fit_intercept = False)
            match loss_type:
                case "TD-MSE" | "FD-MSE":
                    jac_fcn = jacrev(params_to_loss, argnums=0, has_aux=False)
                case "TD-SE" | "FD-SE":
                    jac_fcn = jacfwd(params_to_loss, argnums=0, has_aux=False)
            hess_fcn = jacfwd(jac_fcn, argnums=0, has_aux=False)
            if optim_type == "GHAM-4":
                jac3_fcn = jacfwd(hess_fcn, argnums=0, has_aux=False)
        case "LBFGS":
            raise ValueError("LBFGS optimizer requires multiple function evaluations per optimization step. Not suitable for adaptive filtering scenario.")

    # Build desired response: delay + optional target magnitude response
    total_delay = lem_delay  # TODO: add EQ group delay if necessary. Hardcoded for now!
    target_response = build_target_response_lin_phase(
        sr=sr,
        response_type=target_response_type,
        target_mag_resp=target_mag_resp,
        target_mag_freqs=target_mag_freqs,
        fir_len=2048,
        ROI=ROI,
        rolloff_octaves=.5,
        device=device
    )

    # Convert linear-phase desired response to minimum phase (minimize group delay) and add desired delay
    h_linear_np = target_response.squeeze().cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        h_minphase_np = minimum_phase(h_linear_np, method="homomorphic", half=False)
    delay_zeros = torch.zeros(total_delay, device=device)
    h_minphase = torch.from_numpy(h_minphase_np).float().to(device)
    target_response = torch.cat([delay_zeros, h_minphase]).view(1, 1, -1)/2 # TODO 

    # Normalize desired response
    #target_response = target_response / target_response.abs().sum()  # match overall gain of initial RIR (since we're not optimizing gain in this experiment)

    # Precompute desired output
    desired_output = torchaudio.functional.fftconvolve(input, target_response, mode="full")
    
    # Initialize loss & loss history
    match loss_type:
        case "TD-MSE" | "FD-MSE":
            loss_fcn = F.mse_loss
        case "TD-SE" | "FD-SE":
            loss_fcn = squared_error
        case _:
            raise NotImplementedError(f"Not yet implemented loss_type: {loss_type}. Use 'TD-MSE'")
    loss_history = []
    validation_error_history = []
    jac_norm_history = []
    jac_cond_history = []
    hess_cond_history = []
    irreducible_loss_history = []

    # Initialize buffers
    y_control = torch.zeros(1,1,T, device=device)                       # output audio allocation
    in_buffer = torch.zeros(1,1,frame_len, device=device)               # input audio buffer
    EQ_out_len = next_power_of_2(2*frame_len - 1) # match dasp_pytorch sosfilt_via_fms implementation
    EQ_out_buffer = torch.zeros(1,1,EQ_out_len, device=device)          # buffer for EQ output (input to LEM)
    LEM_out_len = frame_len + LEM.shape[-1] - 1
    LEM_out_buffer = torch.zeros(1,1,LEM_out_len, device=device)        # buffer for soundsystem output
    est_mag_response_buffer = torch.zeros(1,1,frame_len, device=device) # buffer for estimated soundsystem response
    # buffer for estimated complex soundsystem response (store real/imag parts as real tensor)
    init_cpx = torch.fft.rfft(target_response, n=2*frame_len-1)
    est_cpx_response_buffer = torch.view_as_real(init_cpx).view(1, 1, -1, 2)
    rir_idx = 0                                                         # response index (adaptive scenarios)
    # Non-EQ path buffers: raw input history and non-equalized LEM output
    noEQ_in_buffer = torch.zeros(1, 1, EQ_out_len, device=device)       # mirror of EQ_out_buffer but without EQ
    noEQ_out_buffer = torch.zeros(1, 1, LEM_out_len, device=device)     # LEM output for non-EQ path
    y_noEQ = torch.zeros(1, 1, T, device=device)                        # accumulated non-EQ output audio
    

    #####################
    ##### Main loop #####
    #####################

    n_frames = (T - frame_len) // hop_len + 1

    # Determine frame indices at which to record EQ/LEM/target state
    checkpoint_states = []
    if n_checkpoints and n_frames > 1:
        # n_checkpoints + 2 points including first and last frame
        raw_idxs = np.linspace(0, n_frames - 1, n_checkpoints + 2)
        checkpoint_indices = sorted({int(round(i)) for i in raw_idxs})
    else:
        checkpoint_indices = [0, n_frames - 1] if n_frames > 0 else []

    for k in tqdm(range(n_frames), desc="ARE Simulation", unit="frame"):

        start_idx = k * hop_len
        current_time_s = start_idx / sr

        # Update LEM based on current time and transition schedule
        if n_rirs > 1:
            LEM = update_LEM(current_time_s, n_rirs, transition_times_s, rirs_tensors)
        
        # Update input buffer and apply window
        in_buffer = input[:,:,start_idx:start_idx+frame_len]

        # Get target frame
        target_frame = desired_output[:, :, start_idx:start_idx + frame_len]

        do_checkpoint = k in checkpoint_indices
        checkpoint_state = {} if do_checkpoint else None

        loss, buffers = process_buffers(EQG_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_mag_response_buffer,
            est_cpx_response_buffer,
            noEQ_in_buffer,
            noEQ_out_buffer,
            EQ,
            G,
            LEM,
            frame_len,
            hop_len,
            target_frame,
            target_response,
            forget_factor,
            loss_type,
            loss_fcn,
            sr=sr,
            ROI=ROI,
            use_true_LEM=use_true_LEM,
            debug_plot_state=debug_plot_state,
            checkpoint_state=checkpoint_state)
        EQ_out_buffer, LEM_out_buffer, est_mag_response_buffer, est_cpx_response_buffer, validation_error, noEQ_in_buffer, noEQ_out_buffer = buffers

        loss_history.append(torch.mean(loss).item())
        validation_error_history.append(validation_error.item())

        # Collect checkpoint data for later visualization
        if checkpoint_state is not None:
            with torch.no_grad():
                # Store flattened EQ parameters (normalized vector), time, frame index, and sample rate
                checkpoint_state["EQ_params"] = EQG_params[:, :-1].detach().cpu().numpy().astype(np.float32)
                checkpoint_state["G_param_db"] = float(EQG_params[:, -1:].detach().cpu().item())
                checkpoint_state["time_s"] = float(current_time_s)
                checkpoint_state["frame_idx"] = int(k)
                checkpoint_state["sr"] = int(sr)

                # Additionally store a denormalized (7, 3) EQ parameter matrix for
                # direct use with compute_parametric_eq_response. This keeps the
                # heavier denormalization work inside the experiment loop, so the
                # plotting script can operate on real-world units (gain/Q/Fc).
                try:
                    # EQ_params is shape (num_params,) -> make it (1, num_params)
                    eq_param_tensor = EQG_params[:, :-1].detach().view(1, -1)
                    param_dict_norm = EQ.extract_param_dict(eq_param_tensor)
                    param_dict_denorm = EQ.denormalize_param_dict(param_dict_norm)

                    def _scalar(name: str) -> float:
                        return float(param_dict_denorm[name].view(-1)[0].cpu().item())

                    eq_matrix = np.zeros((7, 3), dtype=np.float32)

                    # Low shelf
                    eq_matrix[0, 0] = _scalar("low_shelf_gain_db")
                    eq_matrix[0, 1] = _scalar("low_shelf_q_factor")
                    eq_matrix[0, 2] = _scalar("low_shelf_cutoff_freq")

                    # Bands 0-4
                    eq_matrix[1, 0] = _scalar("band0_gain_db")
                    eq_matrix[1, 1] = _scalar("band0_q_factor")
                    eq_matrix[1, 2] = _scalar("band0_cutoff_freq")

                    eq_matrix[2, 0] = _scalar("band1_gain_db")
                    eq_matrix[2, 1] = _scalar("band1_q_factor")
                    eq_matrix[2, 2] = _scalar("band1_cutoff_freq")

                    eq_matrix[3, 0] = _scalar("band2_gain_db")
                    eq_matrix[3, 1] = _scalar("band2_q_factor")
                    eq_matrix[3, 2] = _scalar("band2_cutoff_freq")

                    eq_matrix[4, 0] = _scalar("band3_gain_db")
                    eq_matrix[4, 1] = _scalar("band3_q_factor")
                    eq_matrix[4, 2] = _scalar("band3_cutoff_freq")

                    eq_matrix[5, 0] = _scalar("band4_gain_db")
                    eq_matrix[5, 1] = _scalar("band4_q_factor")
                    eq_matrix[5, 2] = _scalar("band4_cutoff_freq")

                    # High shelf
                    eq_matrix[6, 0] = _scalar("high_shelf_gain_db")
                    eq_matrix[6, 1] = _scalar("high_shelf_q_factor")
                    eq_matrix[6, 2] = _scalar("high_shelf_cutoff_freq")

                    checkpoint_state["EQ_matrix"] = eq_matrix
                except Exception:
                    # If anything goes wrong, skip EQ_matrix for this checkpoint
                    pass

            checkpoint_states.append(checkpoint_state)

        #frame_analysis_plot(in_buffer, EQ_out_buffer[:,:,:frame_len], LEM_out_buffer[:, :, :frame_len], target_frame, frame_idx=k)

        # Backpropagate and update EQ parameters
        match optim_type:
            case "GHAM-1" | "GHAM-2":
                match loss_type:
                    case "TD-MSE" | "FD-MSE":
                        loss.backward()
                        jac = EQG_params.grad.clone().view(1,-1)
                    case "TD-SE" | "FD-SE":
                        jac = jac_fcn(EQG_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,G,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                
                # TODO: check if this nonnegativity really prevents oscilatory behaviour
                loss_val = torch.maximum(loss.detach() - torch.tensor(eps_0, device=device), torch.tensor(0.0, device=device))
                
                # TODO: handle possible ill_conditioning of Jacobian/Hessian

                # Log irreducible loss and jacobian condition number
                irreducible_loss_history.append(loss_val.mean().item())
                #jac_cond_history.append(torch.linalg.cond(jac.detach().cpu().float()).item())
                
                with torch.no_grad():
                    b = loss_val.view(-1,1)                # (loss_dims, 1)
                    #update = lstsq(jac, b).solution        # (num_params, 1)
                    ridge_regressor.fit(jac,b)
                    update = ridge_regressor.w
                    if optim_type == "GHAM-1":
                        step_sizes = build_step_sizes(mu_opt, EQG_params.shape, device)
                        EQG_params -= step_sizes * update.view_as(EQG_params)
                    elif optim_type == "GHAM-2":
                        step_sizes = build_step_sizes(mu_opt*(2-mu_opt), EQG_params.shape, device)
                        EQG_params -= step_sizes * update.view_as(EQG_params)
                EQG_params.grad = None
            case "Newton":
                jac = jac_fcn(EQG_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,G,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                hess = hess_fcn(EQG_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,G,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                
                # Regularize Hessian: H_reg = H + lambda_newton * I
                dim = hess.shape[-1]
                I = torch.eye(dim, device=hess.device, dtype=hess.dtype)
                hess_reg = hess + lambda_newton * I

                with torch.no_grad():
                    jac = jac.view(-1,1)
                    update = lstsq(hess_reg, jac).solution        # (num_params, 1)
                    step_sizes = build_step_sizes(mu_opt, EQG_params.shape, device)
                    EQG_params -= step_sizes * update.view_as(EQG_params)

                # Backtrack
                #direction = -update.view_as(EQG_params)  # update computed from solve Hx = g
                #alpha = 1.0
                #c = 1e-4
                #rho = 0.5
                #loss0 = loss
                #grad_dot_dir = (jac.view_as(EQG_params) * direction).sum().item()

                #while alpha > 1e-6:
                #    step_sizes = build_step_sizes(alpha, EQG_params.shape, device)
                #    cand = EQG_params + step_sizes * direction
                #    cand_loss = params_to_loss(EQG_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,G,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM)
                #    if cand_loss <= loss0 + c * alpha * grad_dot_dir:
                #        EQG_params = cand
                #        break
                #    alpha *= rho


                # Log condition number of the regularized Hessian
                #hess_cond_history.append(torch.linalg.cond(hess_reg.detach().cpu().float()).item())

            case "GHAM-3" | "GHAM-4":
                jac = jac_fcn(EQG_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,G,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM)
                hess = hess_fcn(EQG_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,G,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                
                # TODO: check if this nonnegativity really prevents oscilatory behaviour
                loss_val = torch.maximum(loss.detach() - torch.tensor(eps_0, device=device), torch.tensor(0.0, device=device))
                loss_val = loss_val.view(-1,1)

                # TODO: handle possible ill_conditioning of Jacobian/Hessian

                # Log irreducible loss and jacobian condition number
                irreducible_loss_history.append(loss_val.mean().item())
                #jac_cond_history.append(torch.linalg.cond(jac.detach().cpu().float()).item())

                with torch.no_grad():
                    step_sizes = build_step_sizes(mu_opt, EQG_params.shape, device).T
                    #theta_1 = -step_sizes * lstsq(jac, loss_val).solution
                    ridge_regressor.fit(jac, loss_val)
                    theta_1 = -step_sizes * ridge_regressor.w
                    theta_2 = (1-step_sizes)*theta_1
                    residual_3 = theta_1.T@hess@theta_1 + jac@theta_2
                    #theta_3 = theta_2 + -step_sizes * lstsq(jac,residual_3).solution
                    ridge_regressor.fit(jac, residual_3)
                    theta_3 = theta_2 + -step_sizes * ridge_regressor.w
                    if optim_type == "GHAM-3":
                        correction = theta_1 + theta_2 + theta_3
                    elif optim_type == "GHAM-4":
                        jac3 = jac3_fcn(EQG_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,G,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                        residual_4 = -step_sizes * (torch.einsum("ijk,i,j,k->", jac3, theta_1.squeeze(), theta_2.squeeze(), theta_3.squeeze())/6 + theta_2.T@hess@theta_1 + jac@theta_3)
                        #theta_4 = theta_3 + lstsq(jac,residual_4).solution
                        ridge_regressor.fit(jac, residual_4)
                        theta_4 = theta_3 + ridge_regressor.w
                        correction = theta_1 + theta_2 + theta_3 + theta_4
                    EQG_params += correction.view_as(EQG_params)
            case _:
                optimizer.zero_grad()
                loss.backward()
                EQG_params.grad[:, -1:] = EQG_params.grad[:, -1:] * 1e2  # Scale gain gradient
                optimizer.step()


        with torch.no_grad():
            EQG_params.data[:, :-1].clamp_(0.0, 1.0)
            EQ_out_buffer = EQ_out_buffer.detach() # prevent graph accumulation across iterations
            LEM_out_buffer = LEM_out_buffer.detach()
            est_mag_response_buffer = est_mag_response_buffer.detach()
            est_cpx_response_buffer = est_cpx_response_buffer.detach()
            noEQ_in_buffer = noEQ_in_buffer.detach()
            noEQ_out_buffer = noEQ_out_buffer.detach()
        # Store output frame (only store hop_len new samples to handle overlap-add)
        end_idx = min(start_idx + frame_len, T)
        samples_to_store = end_idx - start_idx
        y_control[:, :, start_idx:end_idx] += LEM_out_buffer[:, :, :samples_to_store]
        y_noEQ[:, :, start_idx:end_idx] += noEQ_out_buffer[:, :, :samples_to_store]

    # Build results dictionary and return it to main experiment
    # Time axis in seconds for loss/validation samples
    time_axis_val = np.arange(len(validation_error_history)) * (hop_len / sr)

    # Persist audio outputs as numpy float32 arrays for WAV export
    with torch.no_grad():
        input_audio_np = input.squeeze().cpu().numpy().astype(np.float32)
        desired_audio_np = desired_output.squeeze().cpu().numpy()[:T].astype(np.float32)
        y_control_np = y_control.squeeze().cpu().numpy().astype(np.float32)
        y_noEQ_np = y_noEQ.squeeze().cpu().numpy().astype(np.float32)

    total_compute_time_s = time.perf_counter() - exp_t_start
    avg_compute_time_per_frame_s = total_compute_time_s / n_frames if n_frames > 0 else float("nan")

    result = {
        "loss_history": np.array(loss_history, dtype=float),
        "validation_error_history": np.array(validation_error_history, dtype=float),
        "time_axis": time_axis_val,
        "transition_times": transition_times_s if n_rirs > 1 else None,
        "optim_type": optim_type,
        "transition_time_s": sim_cfg["transition_time_s"],
        "input_audio": input_audio_np,
        "desired_audio": desired_audio_np,
        "y_control": y_control_np,
        "y_noEQ": y_noEQ_np,
        "sr": int(sr),
        "n_frames": int(n_frames),
        "control_experiment_time_s": float(total_compute_time_s),
        "avg_compute_time_per_frame_s": float(avg_compute_time_per_frame_s),
        "final_eq_params_normalized": EQG_params[:, :-1].detach().cpu().numpy().astype(np.float32),
        "final_gain_db": EQG_params[:, -1:].detach().cpu().numpy().astype(np.float32),
        "final_true_lem_ir": LEM.squeeze().detach().cpu().numpy().astype(np.float32),
    }
    if checkpoint_states:
        result["checkpoints"] = checkpoint_states
    return result


def params_to_loss(EQG_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_mag_response_buffer,
            est_cpx_response_buffer,
            EQ,
            G,
            LEM,
            frame_len,
            hop_len,
            target_frame,
            target_response,
            forget_factor,
            loss_fcn,
            loss_type,
            sr=None,
            ROI=None,
            use_true_LEM=False):
    # Split combined parameter vector into EQ params and gain
    EQ_params = EQG_params[:, :-1]
    G_param = EQG_params[:, -1:]
    # Process through EQ + gain
    EQ_out = EQ.process_normalized(in_buffer, EQ_params)
    EQ_out = G.process(EQ_out, sr, G_param)

    # Update EQ output buffer (shift left by hop_len and add new samples)
    EQ_out_buffer = F.pad(EQ_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
    EQ_out_buffer += EQ_out

    # Process through LEM
    #LEM_out = torchaudio.functional.fftconvolve(EQ_out_buffer[:,:,:frame_len], LEM.view(1,1,-1), mode="full")
    if use_true_LEM:
        LEM_est = LEM.view(1, 1, -1).detach() # True response for backward pass
    else:
        # est_cpx_response_buffer stores real/imag parts as a real tensor; reconstruct complex spectrum
        LEM_H_est = torch.view_as_complex(est_cpx_response_buffer.squeeze())
        LEM_est = torch.fft.irfft(LEM_H_est, n=2*frame_len-1).view(1, 1, -1).detach()
    LEM_out = LEMConv.apply(
        EQ_out_buffer[:, :, :frame_len],
        LEM.view(1, 1, -1),
        LEM_est
    )

    # Update LEM output buffer (shift left by hop_len)
    LEM_out_buffer = F.pad(LEM_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
    LEM_out_buffer += LEM_out

    # Deconvolve actual response within ROI limits
    nfft = 2*frame_len-1
    freqs = torch.fft.rfftfreq(nfft, d=1.0/sr, device=LEM_out_buffer.device)
    eps = 1e-8
    LEM_H_est = kirkeby_deconvolve(EQ_out_buffer.squeeze(), LEM_out_buffer[:, :, :frame_len].squeeze(), nfft, sr, ROI)
    if ROI is not None:
        roi_mask = (freqs >= ROI[0]) & (freqs <= ROI[1])
        # Set H_complex to 1 (no amplification) outside ROI
        #H = torch.where(roi_mask, H, torch.zeros_like(H) + eps)
    else:
        roi_mask = torch.ones_like(LEM_H_est, dtype=torch.bool)
    
    if torch.sum(torch.abs(est_mag_response_buffer)) == 0:
        forget_factor_loss = 1.0
    else:
        forget_factor_loss = forget_factor
    
    # Use LEM output to compute loss and update EQ parameters
    match loss_type:
        case "FD-MSE" | "FD-SE":

            # Compute full soundsystem (EQ+LEM) frequency response
            H_SS = kirkeby_deconvolve(in_buffer.squeeze(), LEM_out_buffer[:, :, :frame_len].squeeze(), nfft, sr, ROI)

            H_mag_db_current = 20 * torch.log10(_safe_complex_abs(H_SS, eps=eps))
            H_mag_db = (forget_factor_loss)*H_mag_db_current + (1-forget_factor_loss)*est_mag_response_buffer.squeeze()
            est_mag_response_buffer = H_mag_db.view(1,1,-1).detach()
            desired_mag_db = 20 * torch.log10(_safe_complex_abs(torch.fft.rfft(target_response.squeeze(), n=nfft), eps=eps))

            # ROI-limited responses
            freqs_roi = freqs[roi_mask]
            H_mag_db_roi = H_mag_db[roi_mask]
            desired_mag_db_roi = desired_mag_db[roi_mask]

            # Resample to log-frequency axis for perceptually-uniform processing
            n_log_points = 256  # Number of log-spaced frequency points
            H_mag_db_log, _ = interp_to_log_freq(H_mag_db_roi, freqs_roi, n_points=n_log_points)
            desired_mag_db_log, _ = interp_to_log_freq(desired_mag_db_roi, freqs_roi, n_points=n_log_points)

            # Smooth on log-frequency axis using PyTorch conv1d (moving average)
            smooth_window = 15  # Kernel size (odd)
            smooth_kernel = torch.ones(1, 1, smooth_window, device=H_mag_db.device) / smooth_window
            padding = smooth_window // 2
            H_mag_db_log_smoothed = F.conv1d(H_mag_db_log.view(1, 1, -1), smooth_kernel, padding=padding).squeeze()
            desired_mag_db_log_smoothed = F.conv1d(desired_mag_db_log.view(1, 1, -1), smooth_kernel, padding=padding).squeeze()
            
            loss = loss_fcn(H_mag_db_log_smoothed, desired_mag_db_log_smoothed)

        case _:
            loss = loss_fcn(LEM_out_buffer[:, :, :frame_len], target_frame)

    return loss


def process_buffers(EQG_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_mag_response_buffer,
            est_cpx_response_buffer,
            noEQ_in_buffer,
            noEQ_out_buffer,
            EQ,
            G,
            LEM,
            frame_len,
            hop_len,
            target_frame,
            target_response,
            forget_factor,
            loss_type,
            loss_fcn,
            sr=None,
            ROI=None,
            use_true_LEM=False,
            debug_plot_state=None,
            checkpoint_state=None):
    # Split combined parameter vector into EQ params and gain
    EQ_params = EQG_params[:, :-1]
    G_param = EQG_params[:, -1:]
    # Process through EQ + gain
    EQ_out = EQ.process_normalized(in_buffer, EQ_params)
    EQ_out = G.process(EQ_out, sr, G_param)

    # Update EQ output buffer (shift left by hop_len and add new samples)
    EQ_out_buffer = F.pad(EQ_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
    EQ_out_buffer += EQ_out

    # Process through LEM
    #LEM_out = torchaudio.functional.fftconvolve(EQ_out_buffer[:,:,:frame_len], LEM.view(1,1,-1), mode="full")
    if use_true_LEM:
        LEM_est = LEM.view(1, 1, -1).detach() # True response for backward pass
    else:
        # est_cpx_response_buffer stores real/imag parts as a real tensor; reconstruct complex spectrum
        LEM_H_est = torch.view_as_complex(est_cpx_response_buffer.squeeze())
        LEM_est = torch.fft.irfft(LEM_H_est, n=2*frame_len-1).view(1, 1, -1).detach()
    LEM_out = LEMConv.apply(
        EQ_out_buffer[:, :, :frame_len],
        LEM.view(1, 1, -1),
        LEM_est
    )

    # Update LEM output buffer (shift left by hop_len)
    LEM_out_buffer = F.pad(LEM_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
    LEM_out_buffer += LEM_out

    # Non-EQ path: convolve raw input directly with LEM (no EQ or gain applied).
    # This gives a reference output to compare against the adapted EQ output.
    with torch.no_grad():
        noEQ_in_pad = F.pad(in_buffer, (0, noEQ_in_buffer.shape[-1] - in_buffer.shape[-1]))
        noEQ_in_buffer = F.pad(noEQ_in_buffer[..., hop_len:], (0, hop_len))
        noEQ_in_buffer = noEQ_in_buffer + noEQ_in_pad
        noEQ_out = LEMConv.apply(
            noEQ_in_buffer[:, :, :frame_len],
            LEM.view(1, 1, -1),
            LEM_est
        )
        noEQ_out_buffer = F.pad(noEQ_out_buffer[..., hop_len:], (0, hop_len))
        noEQ_out_buffer = noEQ_out_buffer + noEQ_out

    # Deconvolve actual response within ROI limits
    nfft = 2*frame_len-1
    freqs = torch.fft.rfftfreq(nfft, d=1.0/sr, device=LEM_out_buffer.device)
    eps = 1e-8
    LEM_H_est = kirkeby_deconvolve(EQ_out_buffer.squeeze(), LEM_out_buffer[:, :, :frame_len].squeeze(), nfft, sr, ROI)
    if ROI is not None:
        roi_mask = (freqs >= ROI[0]) & (freqs <= ROI[1])
        # Set H_complex to 1 (no amplification) outside ROI
        #H = torch.where(roi_mask, H, torch.zeros_like(H) + eps)
    else:
        roi_mask = torch.ones_like(LEM_H_est, dtype=torch.bool)
    
    if torch.sum(torch.abs(est_mag_response_buffer)) == 0:
        forget_factor_loss = 1.0
        forget_factor_cpx = 1.0
    else:
        forget_factor_loss = forget_factor
        forget_factor_cpx = forget_factor
    
    # Update buffered complex response (stored as real/imag pairs)
    LEM_H_est_ri = torch.view_as_real(LEM_H_est).view(1, 1, -1, 2)
    est_cpx_response_buffer = (1-forget_factor_cpx)*est_cpx_response_buffer + forget_factor_cpx*LEM_H_est_ri.detach()

    # Use LEM output to compute loss and update EQ parameters
    # Compute full soundsystem (EQ+LEM) frequency response
    H_SS = kirkeby_deconvolve(in_buffer.squeeze(), LEM_out_buffer[:, :, :frame_len].squeeze(), nfft, sr, ROI)

    # Compute magnitude responses
    H_mag_db_current = 20 * torch.log10(_safe_complex_abs(H_SS, eps=eps))
    H_mag_db = (forget_factor_loss)*H_mag_db_current + (1-forget_factor_loss)*est_mag_response_buffer.squeeze()
    est_mag_response_buffer = H_mag_db.view(1,1,-1).detach()
    desired_mag_db = 20 * torch.log10(_safe_complex_abs(torch.fft.rfft(target_response.squeeze(), n=nfft), eps=eps))
    LEM_H = torch.fft.rfft(LEM.squeeze(), n=nfft)
    LEM_mag_db = 20 * torch.log10(_safe_complex_abs(LEM_H, eps=eps))

    # ROI-limited responses (keep as tensors)
    freqs_roi = freqs[roi_mask]
    H_mag_db_roi = H_mag_db[roi_mask]
    H_mag_db_current_roi = H_mag_db_current[roi_mask]
    LEM_mag_db_roi = LEM_mag_db[roi_mask]
    desired_mag_db_roi = desired_mag_db[roi_mask]

    # ROI-limited running complex estimate magnitude (from est_cpx_response_buffer)
    H_est_cpx_complex = torch.view_as_complex(est_cpx_response_buffer.squeeze())
    H_est_cpx_mag_db = 20 * torch.log10(_safe_complex_abs(H_est_cpx_complex, eps=eps))
    H_est_cpx_mag_db_roi = H_est_cpx_mag_db[roi_mask]

    # Resample to log-frequency axis for perceptually-uniform processing
    n_log_points = 256  # Number of log-spaced frequency points
    H_mag_db_log, freqs_log = interp_to_log_freq(H_mag_db_roi, freqs_roi, n_points=n_log_points)
    H_mag_db_current_log, _ = interp_to_log_freq(H_mag_db_current_roi, freqs_roi, n_points=n_log_points)
    LEM_mag_db_log, _ = interp_to_log_freq(LEM_mag_db_roi, freqs_roi, n_points=n_log_points)
    desired_mag_db_log, _ = interp_to_log_freq(desired_mag_db_roi, freqs_roi, n_points=n_log_points)
    H_est_cpx_mag_db_log, _ = interp_to_log_freq(H_est_cpx_mag_db_roi, freqs_roi, n_points=n_log_points)

    # Smooth on log-frequency axis using PyTorch conv1d (moving average)
    # This gives more detail at low frequencies, less at high frequencies (perceptually uniform)
    smooth_window = 15
    smooth_kernel = torch.ones(1, 1, smooth_window, device=H_mag_db.device) / smooth_window
    padding = smooth_window // 2

    # Pad with reflection instead of zeros
    padded_mag = F.pad(H_mag_db_log.view(1, 1, -1), (padding, padding), mode='reflect')
    padded_current = F.pad(H_mag_db_current_log.view(1, 1, -1), (padding, padding), mode='reflect')
    padded_lem = F.pad(LEM_mag_db_log.view(1, 1, -1), (padding, padding), mode='reflect')
    padded_desired = F.pad(desired_mag_db_log.view(1, 1, -1), (padding, padding), mode='reflect')

    # Apply convolution with no padding (data is already padded)
    H_mag_db_log_smoothed = F.conv1d(padded_mag, smooth_kernel, padding=0).squeeze()
    H_mag_db_current_log_smoothed = F.conv1d(padded_current, smooth_kernel, padding=0).squeeze()
    LEM_mag_db_log_smoothed = F.conv1d(padded_lem, smooth_kernel, padding=0).squeeze()
    desired_mag_db_log_smoothed = F.conv1d(padded_desired, smooth_kernel, padding=0).squeeze()

    # Compute loss and validation error
    match loss_type:
        case "FD-MSE" | "FD-SE":
            loss = loss_fcn(H_mag_db_log_smoothed, desired_mag_db_log_smoothed)
        case _:
            loss = loss_fcn(LEM_out_buffer[:, :, :frame_len], target_frame)
    validation_error = F.l1_loss(H_mag_db_log_smoothed, desired_mag_db_log_smoothed) / F.l1_loss(LEM_mag_db_log_smoothed, desired_mag_db_log_smoothed)

    # Optionally capture checkpoint state for later visualization
    if checkpoint_state is not None:
        checkpoint_state["freqs_log"] = freqs_log.detach().cpu().numpy().astype(np.float32)
        checkpoint_state["H_total_db"] = H_mag_db_log_smoothed.detach().cpu().numpy().astype(np.float32)
        checkpoint_state["H_desired_db"] = desired_mag_db_log_smoothed.detach().cpu().numpy().astype(np.float32)
        checkpoint_state["H_lem_db"] = LEM_mag_db_log_smoothed.detach().cpu().numpy().astype(np.float32)

    if debug_plot_state is not None:
        # Convert frequency-domain data to numpy for plotting (log-frequency axis)
        freqs_log_np = freqs_log.detach().cpu().numpy()
        H_mag_db_log_np = H_mag_db_log.detach().cpu().numpy()
        H_mag_db_log_smoothed_np = H_mag_db_log_smoothed.detach().cpu().numpy()
        LEM_mag_db_log_np = LEM_mag_db_log.detach().cpu().numpy()
        LEM_mag_db_log_smoothed_np = LEM_mag_db_log_smoothed.detach().cpu().numpy()
        desired_mag_db_log_np = desired_mag_db_log.detach().cpu().numpy()
        H_est_cpx_mag_db_log_np = H_est_cpx_mag_db_log.detach().cpu().numpy()

        # Time-domain true vs estimated LEM responses
        lem_true_td = LEM.squeeze()
        lem_est_td = LEM_est.squeeze()
        min_len_td = min(lem_true_td.numel(), lem_est_td.numel())
        if min_len_td > 0:
            lem_true_td_np = lem_true_td[:min_len_td].detach().cpu().numpy()
            lem_est_td_np = lem_est_td[:min_len_td].detach().cpu().numpy()
            t_td_np = np.arange(min_len_td)
        else:
            lem_true_td_np = np.array([])
            lem_est_td_np = np.array([])
            t_td_np = np.array([])

                # Initialize plot on first call
        if debug_plot_state.get('fig') is None:
            plt.ion()  # Enable interactive mode
            fig, (ax_mag, ax_td) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

            # Top subplot: magnitude responses (log-frequency)
            line_raw, = ax_mag.plot(freqs_log_np, H_mag_db_log_np, linewidth=0.5, alpha=0.4, label='Actual H(f)', color='tab:blue')
            line_smooth, = ax_mag.plot(freqs_log_np, H_mag_db_log_smoothed_np, linewidth=1.5, label='Actual H(f) (smoothed)', color='tab:blue')
            line_desired, = ax_mag.plot(freqs_log_np, desired_mag_db_log_np, linewidth=1, label='Desired H(f)', color='tab:orange')
            line_lem_raw, = ax_mag.plot(freqs_log_np, LEM_mag_db_log_np, linewidth=0.5, alpha=0.4, label='LEM H(f)', color='tab:green')
            line_lem_smooth, = ax_mag.plot(freqs_log_np, LEM_mag_db_log_smoothed_np, linewidth=1.5, label='LEM H(f) (smoothed)', color='tab:green')
            line_est_cpx, = ax_mag.plot(freqs_log_np, H_est_cpx_mag_db_log_np, linewidth=1.0, label='Estimated H_est(f) (from est_cpx)', color='tab:purple', alpha=0.8)
            ax_mag.set_xlabel("Frequency (Hz)")
            ax_mag.set_ylabel("Magnitude (dB)")
            ax_mag.set_xscale('log')
            ax_mag.set_title("FD-MSE: Actual vs Desired Magnitude Response + LEM (log-freq smoothing)")
            ax_mag.legend(loc='lower left')
            ax_mag.grid(True, alpha=0.3)
            ax_mag.set_ylim(-40, 30)  # Extended y-axis range for LEM

            # Bottom subplot: time-domain LEM IRs
            line_lem_true_td, = ax_td.plot(t_td_np, lem_true_td_np, linewidth=1.0, label='True LEM IR', color='tab:red')
            line_lem_est_td, = ax_td.plot(t_td_np, lem_est_td_np, linewidth=1.0, label='Estimated LEM_est IR', color='tab:purple', alpha=0.8)
            ax_td.set_xlabel("Sample")
            ax_td.set_ylabel("Amplitude")
            ax_td.set_title("Time-domain LEM Impulse Responses")
            ax_td.legend(loc='upper right')
            ax_td.grid(True, alpha=0.3)

            plt.tight_layout()
            debug_plot_state['fig'] = fig
            debug_plot_state['ax_mag'] = ax_mag
            debug_plot_state['ax_td'] = ax_td
            debug_plot_state['line_raw'] = line_raw
            debug_plot_state['line_smooth'] = line_smooth
            debug_plot_state['line_desired'] = line_desired
            debug_plot_state['line_lem_raw'] = line_lem_raw
            debug_plot_state['line_lem_smooth'] = line_lem_smooth
            debug_plot_state['line_est_cpx'] = line_est_cpx
            debug_plot_state['line_lem_true_td'] = line_lem_true_td
            debug_plot_state['line_lem_est_td'] = line_lem_est_td
        else:
            # Update existing magnitude-response lines
            debug_plot_state['line_raw'].set_ydata(H_mag_db_log_np)
            debug_plot_state['line_smooth'].set_ydata(H_mag_db_log_smoothed_np)
            debug_plot_state['line_desired'].set_ydata(desired_mag_db_log_np)
            debug_plot_state['line_lem_raw'].set_ydata(LEM_mag_db_log_np)
            debug_plot_state['line_lem_smooth'].set_ydata(LEM_mag_db_log_smoothed_np)
            debug_plot_state['line_est_cpx'].set_ydata(H_est_cpx_mag_db_log_np)

            # Update time-domain IR comparison
            debug_plot_state['line_lem_true_td'].set_data(t_td_np, lem_true_td_np)
            debug_plot_state['line_lem_est_td'].set_data(t_td_np, lem_est_td_np)
            debug_plot_state['ax_td'].relim()
            debug_plot_state['ax_td'].autoscale_view()

            debug_plot_state['fig'].canvas.draw()
            debug_plot_state['fig'].canvas.flush_events()

    buffers = (EQ_out_buffer, LEM_out_buffer, est_mag_response_buffer, est_cpx_response_buffer, validation_error, noEQ_in_buffer, noEQ_out_buffer)

    return loss, buffers
