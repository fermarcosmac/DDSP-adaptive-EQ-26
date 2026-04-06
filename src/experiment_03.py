import sys, math, warnings
from pathlib import Path
from unittest import case
import numpy as np
import matplotlib.pyplot as plt
import torch, torchaudio, torchmin
import torch.nn.functional as F
from torch.linalg import lstsq
from torch.func import jacrev, jacfwd
from scipy.signal import firls, freqz, minimum_phase
from tqdm import tqdm
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from lib.local_dasp_pytorch.modules import ParametricEQ
from modules import Ridge
from utils import (
    load_audio,
    load_rirs,
    ensure_rirs_sample_rate,
    get_delay_from_ir,
    next_power_of_2,
    save_audio,
    get_compensation_EQ_params,
)

# NOTE 1: careful with the gradients you compute. If different samples are generated using different EQ parameters
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
    mag1 = torch.abs(H1)
    mag2 = torch.abs(H2)

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



def frame_analysis_plot(in_buffer, EQ_out, LEM_out, target_frame, frame_idx=None):
    """Plot 4 signals in a 4x1 subplot grid for frame-by-frame analysis.
    
    Args:
        in_buffer: Input buffer tensor (1, 1, frame_len)
        EQ_out: EQ output buffer tensor (1, 1, frame_len)
        LEM_out: LEM output buffer tensor (1, 1, frame_len)
        target_frame: Target frame tensor (1, 1, frame_len)
        frame_idx: Optional frame index for title
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    title = f"Frame Analysis (Frame {frame_idx})" if frame_idx is not None else "Frame Analysis"
    fig.suptitle(title)
    
    # Convert tensors to numpy for plotting
    in_np = in_buffer.squeeze().cpu().detach().numpy()
    eq_np = EQ_out.squeeze().cpu().detach().numpy()
    lem_np = LEM_out.squeeze().cpu().detach().numpy()
    target_np = target_frame.squeeze().cpu().detach().numpy()
    
    axes[0].plot(in_np)
    axes[0].set_ylabel("Input Buffer")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(eq_np)
    axes[1].set_ylabel("EQ Output")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(lem_np)
    axes[2].set_ylabel("LEM Output")
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(target_np)
    axes[3].set_ylabel("Target Frame")
    axes[3].set_xlabel("Sample")
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



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
        print(f"Desired response: delay only (1 sample)")
        
    elif response_type == "delay_and_mag":
        if target_mag_resp is None or target_mag_freqs is None:
            raise ValueError("target_mag_resp and target_mag_freqs are required for response_type='delay_and_mag'")
        
        # Interpolate target magnitude response to FFT bins
        fft_freqs = np.fft.rfftfreq(fir_len, d=1.0/sr)
        
        # Interpolate target magnitude response (in dB) to FFT frequency bins
        target_mag_interp_db = np.interp(fft_freqs, target_mag_freqs, target_mag_resp)
        
        # Apply ROI rolloff: attenuate frequencies outside the region of interest
        # The target_mag_resp is kept as-is within ROI; decay starts exactly at ROI boundaries
        if ROI is not None:
            f_low, f_high = ROI
            rolloff_mask_db = np.zeros_like(fft_freqs)
            
            # Low-frequency decay (below f_low)
            # Decay starts at f_low (0 dB) and reaches -120 dB at f_low_end
            f_low_end = f_low / (2 ** rolloff_octaves)
            for i, f in enumerate(fft_freqs):
                if f <= 0:
                    rolloff_mask_db[i] = -120  # DC component heavily attenuated
                elif f < f_low_end:
                    rolloff_mask_db[i] = -120  # Full attenuation below rolloff end
                elif f < f_low:
                    # Smooth cosine transition from 0 dB (at f_low) to -120 dB (at f_low_end)
                    # t=0 at f_low, t=1 at f_low_end
                    t = np.log2(f_low / f) / rolloff_octaves  # 0 at f_low, 1 at f_low_end
                    rolloff_mask_db[i] = -120 * 0.5 * (1 - np.cos(np.pi * t))
            
            # High-frequency decay (above f_high)
            # Decay starts at f_high (0 dB) and reaches -120 dB at f_high_end
            f_high_end = f_high * (2 ** rolloff_octaves)
            for i, f in enumerate(fft_freqs):
                if f > f_high_end:
                    rolloff_mask_db[i] = -120  # Full attenuation above rolloff end
                elif f > f_high:
                    # Smooth cosine transition from 0 dB (at f_high) to -120 dB (at f_high_end)
                    # t=0 at f_high, t=1 at f_high_end
                    t = np.log2(f / f_high) / rolloff_octaves  # 0 at f_high, 1 at f_high_end
                    rolloff_mask_db[i] = -120 * 0.5 * (1 - np.cos(np.pi * t))
            
            # Apply rolloff mask to target magnitude
            target_mag_interp_db = target_mag_interp_db + rolloff_mask_db
            print(f"Applied ROI decay: {f_low:.0f}-{f_high:.0f} Hz, transition: {rolloff_octaves} octave(s)")
        
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
        
        print(f"Desired response: {target_response.shape[-1]} samples (FIR len={fir_len})")
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



class LEMConv(torch.autograd.Function):
    """
    Custom LEM convolution:
    - forward: uses true LEM impulse response (h_true)
    - backward: uses estimated LEM impulse response (h_est) via FFT-based convolution
    """

    # Let PyTorch generate a vmap rule for forward-mode AD
    generate_vmap_rule = True

    @staticmethod
    def forward(x, h_true, h_est):
        """Forward pass using true LEM impulse response.

        Args:
            x:      (B, 1, N) input signal segment
            h_true: (1, 1, M) true LEM IR (no grad)
            h_est:  (1, 1, M) estimated LEM IR used only for gradients
        """
        # Do not touch ctx here; functorch will call setup_context separately.
        y = torchaudio.functional.fftconvolve(x, h_true, mode="full")
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save context for backward in a functorch-compatible way."""
        x, h_true, h_est = inputs
        ctx.save_for_backward(h_est)
        ctx.input_len = x.shape[-1]
        ctx.h_est_len = h_est.shape[-1]
        # Store true IR and input shape for jvp (forward-mode AD)
        ctx.h_true = h_true
        ctx.x_shape = x.shape

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using estimated LEM impulse response via FFT-based convolution."""
        (h_est,) = ctx.saved_tensors
        N = ctx.input_len
        M = ctx.h_est_len

        grad_full = torchaudio.functional.fftconvolve(grad_output, torch.flip(h_est, dims=[-1]), mode="full")
        grad_x = grad_full[..., M-1:M-1+N]

        return grad_x, None, None

    @staticmethod
    def jvp(ctx, x_t, h_true_t, h_est_t):
        """Jacobian-vector product for forward-mode AD.

        Signature follows PyTorch docs: jvp(ctx, *grad_inputs) -> *grad_outputs,
        where grad_inputs are tangents for each primal input.

        We treat only x as differentiable; h_true and h_est are constants.
        """
        # Use stored primals from setup_context
        h_true = ctx.h_true

        # Materialize missing tangent as zeros if needed
        if x_t is None:
            x_t = torch.zeros(ctx.x_shape, device=h_true.device, dtype=h_true.dtype)

        # Tangent output: d/dx conv(x, h_true)[x_t] = conv(x_t, h_true)
        y_t = torchaudio.functional.fftconvolve(x_t, h_true, mode="full")

        # No gradients w.r.t. h_true or h_est
        return y_t
    


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

    # Apply Kirkeby-style deconvolution filter with frequency-dependent epsilon
    denom = X * torch.conj(X) + epsilon.to(X.dtype)
    H = (Y * torch.conj(X)) / denom

    return H



def params_to_loss(EQ_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_mag_response_buffer,
            est_cpx_response_buffer,
            EQ,
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
    # Process through EQ
    EQ_out = EQ.process_normalized(in_buffer, EQ_params)

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

    # # Deconvolve actual response within ROI limits
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

            H_mag_db_current = 20*torch.log10(torch.abs(H_SS) + eps)
            H_mag_db = (forget_factor_loss)*H_mag_db_current + (1-forget_factor_loss)*est_mag_response_buffer.squeeze()
            est_mag_response_buffer = H_mag_db.view(1,1,-1).detach()
            desired_mag_db = 20*torch.log10(torch.abs(torch.fft.rfft(target_response.squeeze(), n=nfft)) + eps)

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



def process_buffers(EQ_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_mag_response_buffer,
            est_cpx_response_buffer,
            EQ,
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
            debug_plot_state=None):
    # Process through EQ
    EQ_out = EQ.process_normalized(in_buffer, EQ_params)

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
    H_mag_db_current = 20*torch.log10(torch.abs(H_SS) + eps)
    H_mag_db = (forget_factor_loss)*H_mag_db_current + (1-forget_factor_loss)*est_mag_response_buffer.squeeze()
    est_mag_response_buffer = H_mag_db.view(1,1,-1).detach()
    desired_mag_db = 20*torch.log10(torch.abs(torch.fft.rfft(target_response.squeeze(), n=nfft)) + eps)
    LEM_H = torch.fft.rfft(LEM.squeeze(), n=nfft)
    LEM_mag_db = 20 * torch.log10(torch.abs(LEM_H) + eps)

    # ROI-limited responses (keep as tensors)
    freqs_roi = freqs[roi_mask]
    H_mag_db_roi = H_mag_db[roi_mask]
    H_mag_db_current_roi = H_mag_db_current[roi_mask]
    LEM_mag_db_roi = LEM_mag_db[roi_mask]
    desired_mag_db_roi = desired_mag_db[roi_mask]

    # ROI-limited running complex estimate magnitude (from est_cpx_response_buffer)
    H_est_cpx_complex = torch.view_as_complex(est_cpx_response_buffer.squeeze())
    H_est_cpx_mag_db = 20 * torch.log10(torch.abs(H_est_cpx_complex) + eps)
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
    smooth_window = 15  # Kernel size (odd), smaller than before since we have fewer points
    smooth_kernel = torch.ones(1, 1, smooth_window, device=H_mag_db.device) / smooth_window
    padding = smooth_window // 2
    H_mag_db_log_smoothed = F.conv1d(H_mag_db_log.view(1, 1, -1), smooth_kernel, padding=padding).squeeze()
    H_mag_db_current_log_smoothed = F.conv1d(H_mag_db_current_log.view(1, 1, -1), smooth_kernel, padding=padding).squeeze()
    LEM_mag_db_log_smoothed = F.conv1d(LEM_mag_db_log.view(1, 1, -1), smooth_kernel, padding=padding).squeeze()
    desired_mag_db_log_smoothed = F.conv1d(desired_mag_db_log.view(1, 1, -1), smooth_kernel, padding=padding).squeeze()

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
    
    # Compute loss and validation error
    match loss_type:
        case "FD-MSE" | "FD-SE":
            loss = loss_fcn(H_mag_db_log_smoothed, desired_mag_db_log_smoothed)
        case _:
            loss = loss_fcn(LEM_out_buffer[:, :, :frame_len], target_frame)
    validation_error = F.l1_loss(H_mag_db_log_smoothed, desired_mag_db_log_smoothed) / F.l1_loss(LEM_mag_db_log_smoothed, desired_mag_db_log_smoothed)        

    buffers = (EQ_out_buffer, LEM_out_buffer, est_mag_response_buffer, est_cpx_response_buffer, validation_error)

    return loss, buffers



def squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute element-wise squared error between predicted and true signals.
    
    Args:
        y_pred: Predicted signal tensor
        y_true: True signal tensor
    Returns:
        Element-wise squared error tensor
    """
    return (y_pred - y_true) ** 2



def torchmin_closure():
    optimizer.zero_grad()

    EQ_out_buffer_local = EQ_out_buffer.clone()
    LEM_out_buffer_local = LEM_out_buffer.clone()
    est_mag_response_buffer_local = est_mag_response_buffer.clone()

    loss = params_to_loss(
        EQ_params,
        in_buffer,
        EQ_out_buffer_local,     # pass clones
        LEM_out_buffer_local,    # pass clones
        est_mag_response_buffer_local,
        EQ,
        LEM,
        frame_len,
        hop_len,
        target_frame,
        target_response,
        forget_factor,
        loss_fcn,
        loss_type,
        sr,
        ROI
    )
    return loss



#%% MAIN SCRIPT

if __name__ == "__main__":

    torch.manual_seed(123)                  # Seed for reproducibility

    # Set paths
    base = Path(".")
    audio_input_dir = base / "data" / "audio" / "input" / "songs"

    # Input configuration
    input_type = "white_noise"                 # "white_noise" or filename in data/audio/input/
    max_audio_len_s = 120.0                     # None = full length

    # Simulation configuration
    ROI = [50.0, 20000.0]                      # region of interest for EQ compensation (Hz)
    frame_len = 2048*4                          # Length (samples) of processing buffers
    hop_len = frame_len                         # Stride between frames
    window_type = None                          # "hann" or None
    forget_factor = 0.05                        # Forgetting factor for FD loss estimation (0=no memory, 1=full memory)
    optim_type = "SGD"                       # "SGD", "Adam", "GHAM-1", "GHAM-2", "Newton", "GHAM-3", "GHAM-4"
    mu_opt = 1e-2                                  # Learning rate for controller (*1e4 Adam) (*1e-2 SGD) (*1e0 GHAM-1)
    lambda_newton = 1e4                        # Tikhonov regularization for Newton Hessian (H + lambda_newton * I)
    loss_type = "FD-MSE"                        # "TD-MSE", "FD-MSE", "TD-SE", "FD-SE"
    eps_0 = 2.0                                 # Irreducible error floor
    target_response_type = "delay_and_mag"      # "delay_and_mag", "delay_only"
    scenario = "moving_position"                 # "moving_position", "moving_person", or "static"
    n_rirs = 2                                  # Number of RIRs to simulate (1 = constant response)
    transition_time_s = 1.0                     # Transition duration in seconds (0 = abrupt switch)
    use_true_LEM = False                         # Use true LEM for backward pass (oracle, not practical but serves as a reference)
    debug_plot_state = None                       # Debug plot state (set to None to disable, or {} to enable)

    # Resolve RIR directory from scenario
    _rir_subdir_map = {
        "moving_position": root / "data" / "rir" / "moving_position",
        "moving_person":   root / "data" / "rir" / "moving_person",
        "static":          root / "data" / "rir" / "moving_position",
    }
    if scenario not in _rir_subdir_map:
        raise ValueError(f"Unknown scenario '{scenario}'. Must be one of: {list(_rir_subdir_map)}")
    rir_dir = _rir_subdir_map[scenario]
    if scenario == "static":
        n_rirs = 1
    print(f"Scenario: '{scenario}' -> RIR dir: {rir_dir}")

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Acoustic path from actuator (speaker) to sensor (microphone)
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs)
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
            transition_end_s = transition_start_s + transition_time_s
            transition_times_s.append((transition_start_s, min(transition_end_s, max_audio_len_s)))

    # Compute target response
    lem_delay = get_delay_from_ir(rir_init, sr)
    EQ_comp_dict = get_compensation_EQ_params(rir_init, sr, ROI, num_sections=6)
    target_mag_resp = EQ_comp_dict["target_response_db"]
    target_mag_freqs = EQ_comp_dict["freq_axis_smoothed"]

    # Initialize the LEM estimate (assume LEM is well-identified)
    LEM = torch.from_numpy(rir_init).view(1,1,-1).to(device)

    # Initialize differentiable EQ
    EQ = ParametricEQ(sample_rate=sr)
    init_params_tensor = torch.rand(1,EQ.num_params) # random initialization: sensitive to initial parameters
    #init_params_tensor = torch.zeros(1,EQ.num_params) # zeros initialization
    #dasp_param_dict = { k: torch.as_tensor(v, dtype=torch.float32).view(1) for k, v in EQ_comp_dict["eq_params"].items() }
    #_, init_params_tensor = EQ.clip_normalize_param_dict(dasp_param_dict) # initial normalized parameter vector
    EQ_params = torch.nn.Parameter(init_params_tensor.clone().to(device))
    EQ_memory = 128 # TODO: hardcoded for now (should be greater than 0)

    # Load/synthesise the input audio (as torch tensors)
    if input_type == "white_noise":
        T = int(max_audio_len_s * sr)
        input = torch.randn(T, device=device)
        print(f"Input signal: White noise ({max_audio_len_s} s, {T} samples)")

    else:
        audio_path = audio_input_dir / input_type
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        input, audio_sr = torchaudio.load(audio_path)
        input = input.to(device)
        
        if input.shape[0] > 1:
            input = input.mean(dim=0) # convert to mono
        else:
            input = input.squeeze(0)
        
        if audio_sr != sr:  # match sample rates
            resampler = torchaudio.transforms.Resample(orig_freq=audio_sr, new_freq=sr)
            input = resampler(input)
            print(f"Resampled audio from {audio_sr} Hz to {sr} Hz")
        
        if max_audio_len_s is not None:
            max_samples = int(max_audio_len_s * sr)
            if len(input) > max_samples:
                input = input[:max_samples]     # trunctae to max length
                print(f"Truncated audio to first {max_audio_len_s} seconds")

        T = len(input)
        T_seconds = T / sr
        print(f"Input signal: {input_type} ({T_seconds:.2f} s, {T} samples)")
    input = input / input.abs().max()   # normalization
    input = input.view(1,1,-1).to(device)

    # Hanning window for overlap-add (use 50% overlap for perfect reconstruction)
    match window_type:
        case "hann":
            cond = frame_len / (2*hop_len) 
            assert cond.is_integer() and int(cond) > 0, "Invalid hop_length for Hann window. Must have frame_len / (2*hop_len) = integer > 0."
            window = torch.hann_window(frame_len, periodic=True, device=device).view(1, 1, -1)
        case _:
            cond = frame_len / hop_len
            assert cond.is_integer() and (cond % 2 == 0 or cond == 1), "Invalid hop_length for rectangular window."
            window = torch.ones(1, 1, frame_len, device=device)

    # Set optimization (adaptive filtering)
    match optim_type:
        case "SGD":
            optimizer = torch.optim.SGD([EQ_params], lr=mu_opt)
        case "Adam":
            optimizer = torch.optim.Adam([EQ_params], lr=mu_opt)
        case "Muon":
            raise ValueError("Muon optimizer requires newer PyTorch version.")
            optimizer = torch.optim.Muon([EQ_params], lr=mu_opt)
        case "GHAM-1" | "GHAM-2":
            alpha_ridge = 1e-3
            ridge_regressor = Ridge(alpha = alpha_ridge, fit_intercept = False)
            match loss_type:
                case "TD-MSE" | "FD-MSE":
                    jac_fcn = jacrev(params_to_loss, argnums=0, has_aux=False)
                case "TD-SE" | "FD-SE":
                    jac_fcn = jacfwd(params_to_loss, argnums=0, has_aux=False)
        case "Newton" | "GHAM-3" | "GHAM-4":
            alpha_ridge = 1e-3
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
    total_delay = lem_delay + 7  # TODO: add EQ group delay if necessary. Hardcoded for now!
    target_response = build_target_response_lin_phase(
        sr=sr,
        response_type=target_response_type,
        target_mag_resp=target_mag_resp,
        target_mag_freqs=target_mag_freqs,
        fir_len=1024,
        ROI=ROI,
        rolloff_octaves=1.0,
        device=device
    )
    
    # Convert linear-phase desired response to minimum phase (minimize group delay) and add desired delay
    h_linear_np = target_response.squeeze().cpu().numpy()
    h_minphase_np = minimum_phase(h_linear_np, method="homomorphic", half=False)
    delay_zeros = torch.zeros(total_delay, device=device)
    h_minphase = torch.from_numpy(h_minphase_np).float().to(device)
    target_response = torch.cat([delay_zeros, h_minphase]).view(1, 1, -1)

    # Precompute desired output
    desired_output = torchaudio.functional.fftconvolve(input, target_response, mode="full")
    print(f"Precomputed desired output (type: {target_response_type})")
    
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
    EQ_params_buffer = EQ_params.clone().detach()                       # buffer for EQ parameters
    rir_idx = 0                                                         # response index (adaptive scenarios)
    

    #####################
    ##### Main loop #####
    #####################

    n_frames = (T - frame_len) // hop_len + 1
    for k in tqdm(range(n_frames), desc="Processing", unit="frame"):

        start_idx = k * hop_len
        current_time_s = start_idx / sr

        # Update LEM based on current time and transition schedule
        if n_rirs > 1:
            LEM = update_LEM(current_time_s, n_rirs, transition_times_s, rirs_tensors)
        
        # Update input buffer and apply window
        in_buffer = input[:,:,start_idx:start_idx+frame_len] * window

        # Get target frame
        target_frame = desired_output[:, :, start_idx:start_idx + frame_len]

        loss, buffers = process_buffers(EQ_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_mag_response_buffer,
            est_cpx_response_buffer,
            EQ,
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
            debug_plot_state=debug_plot_state)
        EQ_out_buffer, LEM_out_buffer, est_mag_response_buffer, est_cpx_response_buffer, validation_error = buffers

        loss_history.append(torch.mean(loss).item())
        validation_error_history.append(validation_error.item())

        #frame_analysis_plot(in_buffer, EQ_out_buffer[:,:,:frame_len], LEM_out_buffer[:, :, :frame_len], target_frame, frame_idx=k)

        # Backpropagate and update EQ parameters
        match optim_type:
            case "GHAM-1" | "GHAM-2":
                match loss_type:
                    case "TD-MSE" | "FD-MSE":
                        loss.backward()
                        jac = EQ_params.grad.clone().view(1,-1)
                    case "TD-SE" | "FD-SE":
                        jac = jac_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                
                # TODO: check if this nonnegativity really prevents oscilatory behaviour
                loss_val = torch.maximum(loss.detach() - torch.tensor(eps_0, device=device), torch.tensor(0.0, device=device))
                
                # Log irreducible loss and jacobian condition number
                irreducible_loss_history.append(loss_val.mean().item())
                jac_cond_history.append(torch.linalg.cond(jac.detach().cpu().float()).item())
                
                with torch.no_grad():
                    b = loss_val.view(-1,1)                # (loss_dims, 1)
                    update = lstsq(jac, b).solution        # (num_params, 1)
                    #ridge_regressor.fit(jac,b)
                    #update_ridge = ridge_regressor.w      # (num_params, 1)
                    if optim_type == "GHAM-1":
                        EQ_params -= mu_opt * update.view_as(EQ_params)
                    elif optim_type == "GHAM-2":
                        EQ_params -= mu_opt*(mu_opt) * update.view_as(EQ_params)
                EQ_params.grad = None
            case "Newton":
                jac = jac_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                hess = hess_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                
                # Regularize Hessian with Tikhonov term: H_reg = H + lambda_newton * I
                dim = hess.shape[-1]
                I = torch.eye(dim, device=hess.device, dtype=hess.dtype)
                hess_reg = hess + lambda_newton * I

                # Log condition number of the regularized Hessian
                hess_cond_history.append(torch.linalg.cond(hess_reg.detach().cpu().float()).item())
                
                with torch.no_grad():
                    jac = jac.view(-1,1)
                    # Solve (H + lambda_newton * I) * update = jac
                    update = lstsq(hess_reg, jac).solution        # (num_params, 1)
                    EQ_params -= mu_opt * update.view_as(EQ_params)

            case "GHAM-3" | "GHAM-4":
                jac = jac_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM)
                hess = hess_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                
                # TODO: check if this nonnegativity really prevents oscilatory behaviour
                loss_val = torch.maximum(loss.detach() - torch.tensor(eps_0, device=device), torch.tensor(0.0, device=device))
                loss_val = loss_val.view(-1,1)

                # Log irreducible loss and jacobian condition number
                irreducible_loss_history.append(loss_val.mean().item())
                jac_cond_history.append(torch.linalg.cond(jac.detach().cpu().float()).item())

                with torch.no_grad():
                    theta_1 = -mu_opt * lstsq(jac, loss_val).solution
                    theta_2 = (1-mu_opt)*theta_1
                    residual_3 = -mu_opt * theta_1.T@hess@theta_1 + jac@theta_2
                    theta_3 = theta_2 + lstsq(jac,residual_3).solution
                    if optim_type == "GHAM-3":
                        correction = theta_1 + theta_2 + theta_3
                    elif optim_type == "GHAM-4":
                        jac3 = jac3_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_mag_response_buffer,est_cpx_response_buffer,EQ,LEM,frame_len,hop_len,target_frame,target_response,forget_factor,loss_fcn,loss_type,sr,ROI,use_true_LEM).squeeze()
                        residual_4 = -mu_opt * (torch.einsum("ijk,i,j,k->", jac3, theta_1.squeeze(), theta_2.squeeze(), theta_3.squeeze())/6 + theta_2.T@hess@theta_1 + jac@theta_3)
                        theta_4 = theta_3 + lstsq(jac,residual_4).solution
                        correction = theta_1 + theta_2 + theta_3 + theta_4
                    EQ_params += correction.view_as(EQ_params)
            case _:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        with torch.no_grad():
            EQ_params.clamp_(0.0, 1.0)
            EQ_out_buffer = EQ_out_buffer.detach() # prevent graph accumulation across iterations
            LEM_out_buffer = LEM_out_buffer.detach()
            est_mag_response_buffer = est_mag_response_buffer.detach()
            est_cpx_response_buffer = est_cpx_response_buffer.detach()
        # Store output frame (only store hop_len new samples to handle overlap-add)
        end_idx = min(start_idx + frame_len, T)
        samples_to_store = end_idx - start_idx
        y_control[:, :, start_idx:end_idx] += LEM_out_buffer[:, :, :samples_to_store]




    #%% SAVE AUDIO FILES
    
    # Create output directory
    output_dir = base / "data" / "audio" / "ex_03"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare signals for saving (squeeze to 1D and normalize)
    input_save = input.squeeze()
    input_save = input_save / input_save.abs().max() * 0.9
    
    y_control_save = y_control.squeeze()
    y_control_save = y_control_save / (y_control_save.abs().max() + 1e-10) * 0.9
    
    desired_output_save = desired_output.squeeze()[:T]  # Trim to input length
    desired_output_save = desired_output_save / (desired_output_save.abs().max() + 1e-10) * 0.9
    
    # Save audio files
    torchaudio.save(output_dir / "input.wav", input_save.unsqueeze(0).cpu(), sr)
    torchaudio.save(output_dir / "output_controlled.wav", y_control_save.unsqueeze(0).cpu().detach(), sr)
    torchaudio.save(output_dir / "desired_output.wav", desired_output_save.unsqueeze(0).cpu(), sr)
    
    print(f"\nAudio files saved to {output_dir}:")
    print(f"  - input.wav: Input signal")
    print(f"  - output_controlled.wav: Output after EQ and LEM")
    print(f"  - desired_output.wav: Target/desired output signal")

    # Plot loss and validation error progression (3x1 subplot grid)
    fig, (ax_log, ax_lin, ax_val) = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    time_axis = np.arange(len(loss_history)) * hop_len / sr
    
    # ---- Top subplot: Loss (Log scale) ----
    # Left axis: Loss and Irreducible Loss
    ax_log.semilogy(time_axis, loss_history, linewidth=1, label='Loss', color='tab:blue')
    if irreducible_loss_history:
        time_axis_irr = np.arange(len(irreducible_loss_history)) * hop_len / sr
        ax_log.semilogy(time_axis_irr, irreducible_loss_history, linewidth=1, label='Irreducible Loss', color='tab:orange')
    ax_log.set_ylabel("Loss (log)")
    ax_log.grid(True, alpha=0.3)
    ax_log.legend(loc='upper left')
    ax_log.set_title("Loss (Log Scale)")
    
    # Right axis: Jacobian/Hessian condition number (log)
    if jac_cond_history or hess_cond_history:
        ax_log_r = ax_log.twinx()
        if jac_cond_history:
            time_axis_cond = np.arange(len(jac_cond_history)) * hop_len / sr
            ax_log_r.semilogy(time_axis_cond, jac_cond_history, linewidth=1, label='Jacobian Cond. Number', color='tab:green')
        if hess_cond_history:
            time_axis_hess = np.arange(len(hess_cond_history)) * hop_len / sr
            ax_log_r.semilogy(time_axis_hess, hess_cond_history, linewidth=1, label='Hessian Cond. Number', color='tab:purple')
        ax_log_r.set_ylabel("Condition Number", color='tab:green')
        ax_log_r.tick_params(axis='y', labelcolor='tab:green')
        ax_log_r.legend(loc='upper right')
    
    # Add vertical lines at RIR transition times
    if n_rirs > 1:
        for t_start, t_end in transition_times_s:
            if t_start == t_end:
                ax_log.axvline(x=t_start, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            else:
                ax_log.axvline(x=t_start, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                ax_log.axvline(x=t_end, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # ---- Middle subplot: Loss (Linear scale) ----
    # Left axis: Loss and Irreducible Loss
    ax_lin.plot(time_axis, loss_history, linewidth=1, label='Loss', color='tab:blue')
    if irreducible_loss_history:
        time_axis_irr = np.arange(len(irreducible_loss_history)) * hop_len / sr
        ax_lin.plot(time_axis_irr, irreducible_loss_history, linewidth=1, label='Irreducible Loss', color='tab:orange')
    ax_lin.set_ylabel("Loss (linear)")
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(loc='upper left')
    ax_lin.set_title("Loss (Linear Scale)")
    
    # Right axis: Jacobian/Hessian condition number (linear)
    if jac_cond_history or hess_cond_history:
        ax_lin_r = ax_lin.twinx()
        if jac_cond_history:
            time_axis_cond = np.arange(len(jac_cond_history)) * hop_len / sr
            ax_lin_r.plot(time_axis_cond, jac_cond_history, linewidth=1, label='Jacobian Cond. Number', color='tab:green')
        if hess_cond_history:
            time_axis_hess = np.arange(len(hess_cond_history)) * hop_len / sr
            ax_lin_r.plot(time_axis_hess, hess_cond_history, linewidth=1, label='Hessian Cond. Number', color='tab:purple')
        ax_lin_r.set_ylabel("Condition Number", color='tab:green')
        ax_lin_r.tick_params(axis='y', labelcolor='tab:green')
        ax_lin_r.legend(loc='upper right')
    
    # Add vertical lines at RIR transition times
    if n_rirs > 1:
        for t_start, t_end in transition_times_s:
            if t_start == t_end:
                ax_lin.axvline(x=t_start, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            else:
                ax_lin.axvline(x=t_start, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                ax_lin.axvline(x=t_end, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # ---- Bottom subplot: Validation Error (Linear scale) ----
    time_axis_val = np.arange(len(validation_error_history)) * hop_len / sr
    ax_val.plot(time_axis_val, validation_error_history, linewidth=1, label='Validation Error', color='tab:red')
    ax_val.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.8, label='Baseline (no EQ)')
    ax_val.set_xlabel("Time (s)")
    ax_val.set_ylabel("Validation Error")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc='upper left')
    ax_val.set_title("Validation Error (Linear Scale)")
    
    # Add vertical lines at RIR transition times
    if n_rirs > 1:
        for t_start, t_end in transition_times_s:
            if t_start == t_end:
                ax_val.axvline(x=t_start, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            else:
                ax_val.axvline(x=t_start, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                ax_val.axvline(x=t_end, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    plt.tight_layout()
    
    # Plot magnitude responses of all RIRs
    fig_rirs, ax_rirs = plt.subplots(figsize=(6, 5))
    nfft_rir = 4096  # FFT size for RIR frequency response
    freqs_rir = np.fft.rfftfreq(nfft_rir, d=1.0/sr)
    smooth_window_rir = 31  # Smoothing window size (must be odd)
    
    # Get default color cycle
    colors = plt.cm.tab10.colors
    
    for i, rir in enumerate(rirs):
        H_rir = np.fft.rfft(rir, n=nfft_rir)
        H_rir_mag_db = 20 * np.log10(np.abs(H_rir) + 1e-10)
        
        # Smoothed version (moving average)
        H_rir_mag_db_smoothed = np.convolve(H_rir_mag_db, np.ones(smooth_window_rir)/smooth_window_rir, mode='same')
        
        color = colors[i % len(colors)]
        # Original (dim, in background)
        ax_rirs.semilogx(freqs_rir, H_rir_mag_db, linewidth=0.5, alpha=0.3, color=color)
        # Smoothed (prominent, with label)
        ax_rirs.semilogx(freqs_rir, H_rir_mag_db_smoothed, linewidth=1.5, label=f'RIR {i+1}', alpha=0.9, color=color)
    
    ax_rirs.set_xlabel("Frequency (Hz)")
    ax_rirs.set_ylabel("Magnitude (dB)")
    ax_rirs.set_title("Magnitude Responses of All RIRs")
    ax_rirs.set_xlim([20, sr/2])
    ax_rirs.legend()
    ax_rirs.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.ioff()  # Turn off interactive mode so plt.show() blocks
    plt.show()  # This will block until all figures are closed

    pass