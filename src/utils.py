import sys
from pathlib import Path
from typing import List, Callable, Tuple, Optional
import numpy as np
try:
    import soundfile as sf
except Exception:
    sf = None
from scipy.signal import fftconvolve, resample_poly, correlate, correlation_lags
from scipy.optimize import least_squares
import bisect
import warnings
import os
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
# Ensure the workspace root is first on sys.path so the local package is imported
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from lib.local_dasp_pytorch.modules import ParametricEQ
from modules import EQController_dasp, EQLogger





# GENERAL AUDIO/RIR FILE
def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio as float32 mono.

    Returns:
        audio (np.ndarray, shape=(N,)): floating point audio in range [-1, 1]
        sr (int): sample rate
    """
    if sf is not None:
        data, sr = sf.read(str(path))
        if data.ndim > 1:
            # Convert to mono by averaging channels (simple default)
            data = data.mean(axis=1)
        # ensure float32
        data = data.astype(np.float32)
    else:
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=False)
        else:
            wav = wav.squeeze(0)
        data = wav.detach().cpu().numpy().astype(np.float32)
    return data, sr





def save_audio(path: Path, data: np.ndarray, sr: int) -> None:
    """Save audio as 32-bit float WAV (or default by soundfile).
    
    Scales audio if peak value exceeds [-1, 1] range to prevent clipping.
    """
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        # Scale audio to prevent clipping
        scaled_data = data / max_val
        warnings.warn(f"Audio exceeded [-1, 1] range (peak={max_val:.3f}). Scaled by {1/max_val:.3f}.")
    else:
        scaled_data = data
    
    if sf is not None:
        sf.write(str(path), scaled_data, sr)
    else:
        t = torch.from_numpy(np.asarray(scaled_data, dtype=np.float32)).unsqueeze(0)
        torchaudio.save(str(path), t, int(sr))





def load_rirs(rir_dir: Path, max_n: int = None) -> Tuple[List[np.ndarray], List[int]]:
    """Load all wav files in a directory as float32 RIRs. Returns (rirs, srs).

    RIRs are returned as lists of 1-D numpy arrays. Sample rates are returned
    for each file so the caller can check/resample if needed.
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
        # Normalize so that maximum absolute value is one
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak
        rirs.append(data)
        srs.append(sr)
    return rirs, srs





def ensure_rirs_sample_rate(rirs: List[np.ndarray], rirs_srs: List[int], target_sr: int) -> List[np.ndarray]:
    """Resample RIRs to match the target sample-rate if needed.

    Uses a simple resample_poly approach (from scipy.signal) for reasonable quality.
    """
    out = []
    for rir, sr in zip(rirs, rirs_srs):
        if sr == target_sr:
            out.append(rir)
        else:
            # Compute integer factors for resample_poly if possible
            # fallback to float ratio using resample_poly with nearest ints
            gcd = np.gcd(sr, target_sr)
            up = target_sr // gcd
            down = sr // gcd
            rir_rs = resample_poly(rir, up, down).astype(np.float32)
            warnings.warn(f"Resampled RIR from {sr} Hz -> {target_sr} Hz; length {len(rir)} -> {len(rir_rs)}")
            out.append(rir_rs)
    return out





# TIME-VARYING FILTERING
def _active_rir_index_for_time(start_times_s: List[float], t: float) -> int:
    """Given a sorted list of start times (seconds) and a time t (seconds),
    return the index of the active RIR. The active RIR is the last index i
    such that start_times_s[i] <= t. If t < start_times_s[0], returns 0.

    Its a private functionality associated with simulate_time_varying_rir().
    """
    # bisect_right returns insertion point; subtract 1 to get index <= t
    if len(start_times_s) == 0:
        raise ValueError("start_times must be non-empty")
    idx = bisect.bisect_right(start_times_s, t) - 1
    if idx < 0:
        return 0
    if idx >= len(start_times_s):
        return len(start_times_s) - 1
    return idx






def simulate_time_varying_process(
    audio: torch.Tensor,
    sr: int,
    rirs: List[torch.Tensor],
    rir_indices: List[int],
    switch_times_s: List[float],
    process_fn: Optional[Callable[[torch.Tensor, int, torch.Tensor, int, int], torch.Tensor]] = None,
    EQ: ParametricEQ = None,
    controller: EQController_dasp = None,   # TODO Controller class to be defined. It holds state information and methods for updating parameters
    logger: EQLogger = None,                # TODO Logger class to be defined. It handles logging of parameters and performance metrics. Maybe it should be an attribute of the Controller!
    window: float = 100.0,
    hop: float = 50.0,
    win_hop_units: str = "ms",
) -> Tuple[torch.Tensor, int]:
    """
    TODO: redo documentation for this function.
    Torch version of simulate_time_varying_process: operates entirely on torch.Tensors.

    Arguments (same semantics as the NumPy version):
        audio: 3-D torch.Tensor with shape (1, 1, N). Expected dtype float32.
        sr: sample rate (int)
        rirs: list of 1-D torch.Tensors (float32) containing RIRs (resampled to sr)
        rir_indices: list of indices into `rirs` indicating the sequence of RIRs
        start_times_s: list of start times in seconds for each corresponding index.
                       Must be same length as rir_indices, sorted ascending, non-negative.
        process_fn: callable(frame, sr, rir, frame_start, frame_idx) -> processed_frame (torch.Tensor).
                    If None, defaults to FFT-based convolution using torch.fft (frame * rir).
        window_ms, hop_ms: analysis window/hop in milliseconds.

    Returns:
        y: processed audio as torch.Tensor (1-D, float32)
        sr: sample rate (int)
    """
    # --- input validation & conversion ---
    if isinstance(audio, torch.Tensor):
        audio_t = audio  # Keep gradients enabled (don't detach)
    else:
        audio_t = torch.as_tensor(audio)

    # Require mono audio with shape (1, 1, N)
    if audio_t.ndim != 3 or audio_t.shape[0] != 1 or audio_t.shape[1] != 1:
        raise ValueError(
            f"simulate_time_varying_process expects mono audio with shape (1, 1, N), "
            f"but got shape {tuple(audio_t.shape)}"
        )
    
    # Ensure float32
    audio_t = audio_t.to(dtype=torch.float32)

    # Convert rirs to torch tensors on same device/dtype
    device = audio_t.device
    rirs_t: List[torch.Tensor] = []
    for r in rirs:
        if not isinstance(r, torch.Tensor):
            r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
        else:
            r_t = r.to(device=device, dtype=torch.float32)
        rirs_t.append(r_t)

    # Validate start times / indices
    assert len(rir_indices) == len(switch_times_s), "rir_indices and switch_times_s must have same length"
    if any(t < 0 for t in switch_times_s):
        raise ValueError("switch_times_s must be non-negative")
    if any(switch_times_s[i] > switch_times_s[i + 1] for i in range(len(switch_times_s) - 1)):
        raise ValueError("switch_times_s must be sorted ascending")
    
    # Check that, if a controller has been provided, controller.EQ and EQ are references to the same instance
    if controller is not None and EQ is not None:
        if controller.EQ is not EQ:
            raise ValueError("The EQ instance provided to simulate_time_varying_process() must be the same as controller.EQ")
    
    # Default processing function: EQ + LEM + controller update
    if process_fn is None:
        def default_process(frame: torch.Tensor, sr_: int, rir_: torch.Tensor, frame_start: int, frame_idx: int, EQ: ParametricEQ, controller: EQController_dasp) -> torch.Tensor:
            
            # Apply compensation EQ if provided
            if  EQ is not None:
                EQed_frame = EQ.process_normalized(frame, controller.params)
            else:
                EQed_frame = frame
            
            # Pass audio through LEM system (represented by RIR)
            mic_signal = torchaudio.functional.fftconvolve(EQed_frame, rir_, mode="full")  # (1, 1, N + RIR_len - 1)
            # RIR is unknown! Here pytorch cheats by storing for gradient computation the real RIR!

            # Update controller with LEM estimation and parameter adaptation
            if controller is not None:
                controller.update(in_frame=frame, EQed_frame=EQed_frame, out_frame=mic_signal, sr=sr_, frame_start_sample=frame_start)
            
            return mic_signal
        process_fn = default_process

    # Convert ms to samples (ints)
    if win_hop_units == "ms":
        win_len = int(round(window * sr / 1000.0))
        hop_len = int(round(hop * sr / 1000.0))
    elif win_hop_units == "samples":
        win_len = int(window) 
        hop_len = int(hop)
    else:
        raise ValueError("win_hop_units must be 'ms' or 'samples'")
    if win_len <= 0 or hop_len <= 0:
        raise ValueError("window_ms and hop_ms must be positive and produce non-zero lengths")

    n = int(audio_t.shape[-1])
    max_rir_len = max((r.shape[-1] for r in rirs_t), default=0)

    # Output buffer: original length + max tail + window length (safe)
    out_len = n + max_rir_len + win_len
    y = torch.zeros(1, 1, out_len, dtype=torch.float32, device=device)

    # Frame start indices
    frame_starts = list(range(0, n, hop_len))
    
    # Create Hann window for COLA (Constant Overlap-Add) with 50% overlap
    # This ensures amplitude consistency across frame boundaries
    hann_window = torch.hann_window(win_len, periodic=False, device=device, dtype=torch.float32)
    
    print(f"[simulate_time_varying_process] Starting simulation:")
    print(f"  Audio length: {n} samples ({n/sr:.2f} sec)")
    print(f"  Frame length: {win_len} samples, Hop: {hop_len} samples")
    print(f"  Total frames: {len(frame_starts)}")
    print(f"  Max RIR length: {max_rir_len} samples")
    print(f"  Using Hann window for COLA overlap-add (50% overlap)")

    for frame_idx, s in enumerate(frame_starts):
        e = s + win_len
        if e <= n:
            frame = audio_t[:, :, s:e]
        else:
            # partial last frame, pad to win_len
            valid = audio_t[:, :, s:n]
            pad_len = win_len - valid.shape[-1]
            frame = torch.nn.functional.pad(valid, (0, pad_len), mode='constant', value=0.0)

        # Apply Hann window to frame for proper overlap-add reconstruction
        windowed_frame = frame * hann_window.unsqueeze(0).unsqueeze(0)
        windowed_frame = frame # Rectangular window TODO check if it is better

        # choose active RIR based on midpoint of the frame in seconds
        midpoint_s = (s + win_len // 2) / float(sr)
        seq_idx = _active_rir_index_for_time(switch_times_s, midpoint_s)
        rir_idx = rir_indices[seq_idx]
        rir = rirs_t[rir_idx]
        
        # Log progress every 1 frames or for first/last frames
        print(f"  Frame {frame_idx+1}/{len(frame_starts)} | Time: {midpoint_s:.2f}s | RIR: {rir_idx}")

        # call user-provided process function (expects torch tensors)
        processed = process_fn(windowed_frame, sr, rir, s, frame_idx, EQ, controller)

        if processed is None:
            continue

        # ensure 3-D tensor float32 on correct device
        if not isinstance(processed, torch.Tensor):
            processed = torch.as_tensor(processed, device=device, dtype=torch.float32)
        else:
            processed = processed.to(device=device, dtype=torch.float32)

        len_proc = int(processed.shape[-1])
        end_out = s + len_proc

        if s >= out_len:
            # frame starts beyond buffer - skip
            continue

        if end_out > out_len:
            keep_len = out_len - s
            if keep_len > 0:
                y[:, :, s:out_len] += processed[:, :, :keep_len]
        else:
            y[:, :, s:end_out] += processed

    # Trim final output to original length + max_rir_len (you may adjust)
    final_len = n + max_rir_len
    y = y[:, :, :final_len]
    
    print(f"[simulate_time_varying_process] Simulation complete. Output shape: {y.shape}")

    return y, sr





def simulate_time_varying_rir(
    audio: np.ndarray,
    sr: int,
    rirs: List[np.ndarray],
    rir_indices: List[int],
    start_times_s: List[float],
    window_ms: float = 100.0,
    hop_ms: float = 50.0,
) -> Tuple[np.ndarray, int]:
    """Simulate a time-varying RIR by processing the input audio in frames and
    convolving each frame with the active RIR. Overlap-add is used to reconstruct
    the final signal.

    Args:
        audio: 1-D numpy array (float32)
        sr: sample rate
        rirs: list of RIR arrays (already resampled to sr)
        rir_indices: list of indices into `rirs` indicating the sequence of RIRs
        start_times_s: list of start times in seconds for each corresponding index.
                       Must be same length as rir_indices, sorted ascending, and
                       first element normally 0.0 (but not required).
        window_ms: analysis window length in milliseconds
        hop_ms: hop size in milliseconds

    Returns:
        y: processed audio (float32)
        sr: sample rate
    """
    assert len(rir_indices) == len(start_times_s), "rir_indices and start_times_s must have same length"
    # sort/check consistency
    # create a mapping of absolute start_times for the sequence
    seq_start_times = list(start_times_s)
    if any(t < 0 for t in seq_start_times):
        raise ValueError("start_times_s must be non-negative")
    if any(seq_start_times[i] > seq_start_times[i + 1] for i in range(len(seq_start_times) - 1)):
        raise ValueError("start_times_s must be sorted ascending")

    # Convert to sample-based window/hop
    win_len = int(round(window_ms * sr / 1000.0))
    hop_len = int(round(hop_ms * sr / 1000.0))
    if win_len <= 0 or hop_len <= 0:
        raise ValueError("window_ms and hop_ms must be positive and produce non-zero lengths")

    n = len(audio)
    max_rir_len = max(len(r) for r in rirs)
    conv_len = win_len + max_rir_len - 1
    # Output length must accommodate convolution tail
    out_len = n + win_len +max_rir_len
    y = np.zeros(out_len, dtype=np.float32)

    # Frame starts
    frame_starts = list(range(0, n, hop_len))

    for s in frame_starts:              # start index
        e = s + win_len                 # end index  
        frame = audio[s:e]
        # If last frame is shorter, pad with zeros
        if len(frame) < win_len:
            frame = np.pad(frame, (0, win_len - len(frame)))

        # choose active RIR based on the midpoint of the frame
        midpoint_s = (s + win_len // 2) / sr        # I THINK I CAN DO IT WITH START INDEX!
        # Determine which index in rir_indices is active
        seq_idx = _active_rir_index_for_time(seq_start_times, midpoint_s)
        rir_idx = rir_indices[seq_idx]      # I THINK THIS IS INNEFICIENT
        rir = rirs[rir_idx]

        # convolve frame with the selected RIR. Here I should have the process() function!
        conv = fftconvolve(frame, rir, mode="full").astype(np.float32)

        # add to output at the correct place
        try:
            y[s:s+conv_len] += conv
        except:
            hey=0

    # Trim the output to reasonable length (you might prefer to keep the full tail)
    # Here we trim to original audio length + max_rir_len
    y = y[: n + max_rir_len]

    return y, sr





# MATH
def next_power_of_2(x):
    return 1 << (x - 1).bit_length()






def rms(x: np.ndarray) -> float:
    """Compute root-mean-square of a 1-D numpy array."""
    return np.sqrt(np.mean(x**2))





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





def get_delay_xcorr(input_signal: np.ndarray, output_signal: np.ndarray, sr: int, max_delay_s: float = None) -> int:
    """Estimate delay (in samples) between input and output signals using cross-correlation.

    Searches the entire cross-correlation to find the global maximum, without limiting 
    the search to a maximum delay window.

    Args:
        input_signal: 1-D numpy array containing the input signal
        output_signal: 1-D numpy array containing the output signal
        sr: sample rate (Hz)
        max_delay_s: deprecated parameter, ignored (kept for backwards compatibility)
    Returns:
        delay_samples: estimated delay in samples (int)
    """
    # Compute cross-correlation
    corr = correlate(output_signal, input_signal, mode='full', method='fft')
    lags = correlation_lags(len(output_signal), len(input_signal), mode='full')
    mid = np.where(lags == 0)[0][0]
    #mid = len(corr) // 2

    # Find global maximum in entire cross-correlation
    delay_index = np.argmax(np.abs(corr))
    delay_samples = delay_index - mid

    return delay_samples



# COMPENSATION EQ STATIC ESTIMATION
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
        _eq_objective_function,                                          # loss
        init_params.ravel(),                                            # initial parameters (flattened)
        bounds=(lb.ravel(), ub.ravel()),
        args=(num_filters, f, meas_resp_db, target_resp_db, ROI, Fs),   # additional args to compute loss function
        ftol=1e-8,
        max_nfev=max(500 * num_filters, 3600),
        verbose=2,
    )

    EQ_params = res.x.reshape(num_filters, 3)                           # Extract optimized (LS) EQ parameters
    filtResp = compute_parametric_eq_response(EQ_params, f, Fs)
    outputResp = meas_resp_db + filtResp

    return EQ_params, outputResp, filtResp





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
    Fc_peaks = np.logspace(         # All but last two are peaking filters
        np.log10(fmin),
        np.log10(fmax),
        num_filters - 2
    )
    Fc_shelves = np.array([         # Last two filters are shelves (high and low)
        (2 * ROI[0] + ROI[1]) / 3,
        (ROI[0] + 2 * ROI[1]) / 3
    ])

    # Initial parameter matrix
    init_params = np.zeros((num_filters, 3))
    init_params[:, 0] = _get_initial_gains(num_filters)
    init_params[:-2, 1] = 1.5     # Q for peak filters
    init_params[-2:, 1] = 1.0     # slope for shelf filters
    init_params[:-2, 2] = Fc_peaks
    init_params[-2:, 2] = Fc_shelves

    # Lower bounds
    lb = np.zeros_like(init_params)
    lb[:, 0] = -24.0        # gain lower bound
    lb[:-2, 1] = 0.2        # Q for peak filters lower bound
    lb[-2:, 1] = 0.1        # slope for shelf filters lower bound
    lb[:, 2] = fmin-1e-6    # Fc lower bound

    # Upper bounds
    ub = np.zeros_like(init_params)
    ub[:, 0] = 20.0         # gain upper bound
    ub[:-2, 1] = 17.3       # Q for peak filters upper bound
    ub[-2:, 1] = 5.0        # slope for shelf filters upper bound
    ub[:, 2] = fmax+1e-6    # Fc upper bound

    return init_params, lb, ub





def _get_initial_gains(n: int, seed: int = 0) -> np.ndarray:
    """
    Generate initial gain values for EQ filters.
    Uses a fixed RNG seed so results are repeatable.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)





def _six_biquad_eq_params_to_dict(EQ_params: np.ndarray) -> dict:
    """
    Convert (6, 3) EQ parameter matrix into a named parameter dictionary.

    EQ_params: np.ndarray with shape (6, 3)
               columns: [gain_dB, Q_or_slope, Fc]
    """
    assert EQ_params.shape == (6, 3), "Expected EQ_params shape (6, 3)"

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

        # ---- High shelf ----
        "high_shelf_gain_db": float(EQ_params[5, 0]),
        "high_shelf_q_factor": float(EQ_params[5, 1]),
        "high_shelf_cutoff_freq": float(EQ_params[5, 2]),
    }





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
    plt.figure()
    plt.semilogx(freqs, 20*np.log10(freq_response + 1e-12), label="RIR Frequency Response")
    plt.semilogx(cf, 20*np.log10(oa + 1e-12), label="Octave-Averaged Response")
    plt.semilogx(cf, target_resp, label="Target Compensation Response", linestyle='--')
    plt.semilogx(cf, out_resp_db, label="Equalized Response", linestyle='-.')
    plt.semilogx(cf, filt_resp_db, label="EQ Filter Response", linestyle=':')
    plt.axvline(ROI[0], color="red", linestyle="--", label="ROI Limits" if ROI else "")
    plt.axvline(ROI[1], color="red", linestyle="--" if ROI else "")
    plt.legend(), plt.xlabel("Frequency (Hz)"), plt.ylabel("Magnitude (dB)")
    plt.title("RIR Frequency Response and Target Compensation EQ")
    plt.savefig("rir_response.png",dpi=150, bbox_inches='tight')

    # Build parameter dictionary for downstream use (e.g., PyTorch EQ)
    eq_param_dict = _six_biquad_eq_params_to_dict(EQ_params)

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





# GENERAL PARAMETRIC EQ
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





