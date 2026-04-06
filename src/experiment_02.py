"""
FxLMS Algorithm Example - Python Adaptation
============================================

This is a Python adaptation of a MATLAB FxLMS example for a single-channel 
feed-forward active noise control system. Reference: "Active Noise Control 
Systems - Algorithms and DSP Implementations," by S. M. Kuo and D. R. Morgan (1996).

System Diagram:
              +-----------+                       +   
 x(k) ---+--->|   P(z)    |--yp(k)----------------> sum --+---> e(k)
         |    +-----------+                          ^-   |
         |                                           |    |
         |        \                                ys(k)  |     
         |    +-----------+          +-----------+   |    |
         +--->|   C(z)    |--yw(k)-->|   S(z)    |---+    |
         |    +-----------+          +-----------+        |
         |            \                                   |
         |             \----------------\                 |
         |                               \                |
         |    +-----------+          +-----------+        |
         +--->|   Sh(z)   |--xs(k)-->|    LMS    |<-------+
              +-----------+          +-----------+        

Where:
- P(z): Primary path (noise propagation from source to sensor)
- S(z): Secondary path (actuator to sensor)
- C(z): Controller (adaptive filter)
- Sh(z): Estimate of S(z)

Original MATLAB code by Agustinus Oey <oeyaugust@gmail.com>
Center of Noise and Vibration Control (NoViC)
Department of Mechanical Engineering
Korea Advanced Institute of Science and Technology (KAIST)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firls, freqz

# Ensure the workspace root is first on sys.path so the local package is imported
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from utils import (
    load_audio,
    load_rirs,
    ensure_rirs_sample_rate,
    get_delay_from_ir,
    save_audio,
)


#%% MAIN SCRIPT

if __name__ == "__main__":

    #%% CONFIGURATION AND SETUP
    
    # Set paths (same as experiment_01)
    base = Path(".")
    rir_dir = base / "data" / "rir" / "moving_position"
    audio_input_dir = base / "data" / "audio" / "input" / "songs"
    
    # === INPUT SIGNAL CONFIGURATION ===
    # Choose input signal type: "white_noise" or audio filename from data/audio/input/
    # Available audio files: guitar-riff.wav, guitar-riff_short.wav, onde_day_funk.wav
    input_type = "white_noise"  # Options: "white_noise", or filename like "guitar-riff.wav"
    # input_type = "guitar-riff_short.wav"  # Uncomment to use guitar audio
    
    # Maximum duration for audio files (None = use full audio)
    max_audio_duration = 20.0  # Limit audio to first 10 seconds
    
    # Set simulation parameters
    sr = 48000  # Sample rate (will be updated if RIR has different rate)
    T_seconds = 10.0  # Simulation duration in seconds (used for white noise and identification)
    
    # Load RIR for secondary path S(z)
    # The RIR represents the acoustic path from actuator (speaker) to sensor (microphone)
    rirs, rirs_srs = load_rirs(rir_dir, max_n=1)
    
    # Use the first RIR's sample rate as reference
    sr = rirs_srs[0]
    rirs = ensure_rirs_sample_rate(rirs, rirs_srs, sr)
    rir = rirs[0]  # Use first RIR
    rir = rir[900:1100]

    # S(z): Secondary path coefficients from loaded RIR
    # Truncate or use full RIR based on desired filter length
    Sw_max_len = min(len(rir), 8192*2)  # Limit secondary path length for computational efficiency
    Sw = rir[:Sw_max_len].astype(np.float64)  # Secondary path coefficients
    Sw = Sw / np.max(np.abs(Sw))  # Normalize to prevent numerical issues
    
    # === PRIMARY PATH FILTER CONFIGURATION ===
    # Design a lowpass FIR filter for the primary path P(z)
    Pw_len = 64  # Filter length for primary path (number of taps)
    Pw_cutoff = 18000.0  # Cutoff frequency in Hz
    
    # Design lowpass filter using firwin
    from scipy.signal import firwin
    normalized_cutoff = Pw_cutoff / (sr / 2)  # Normalize to [0, 1] (Nyquist = 1)
    
    if normalized_cutoff >= 1.0:
        # Cutoff at or above Nyquist: use allpass (impulse)
        Pw_core = np.zeros(Pw_len)
        Pw_core[0] = 1.0
        print(f"Warning: Cutoff {Pw_cutoff} Hz >= Nyquist, using allpass")
    else:
        Pw_core = firwin(Pw_len, normalized_cutoff, window='hamming')
        Pw_core = Pw_core / np.max(np.abs(Pw_core))  # Normalize
    
    print(f"Primary path lowpass filter: {Pw_len} taps, cutoff = {Pw_cutoff} Hz")
    
    # Estimate delay from the RIR (position of first peak = direct sound arrival)
    est_delay = get_delay_from_ir(rir, sr)
    #est_delay += 256 # Hardcoded! Half the compensating filter length
    print(f"Estimated delay from RIR: {est_delay} samples ({est_delay/sr*1000:.2f} ms)")
    
    # Zero-pad Pw at the beginning to incorporate the delay into the filter itself
    # This way the delay is reflected in the filter's phase response
    primary_delay = est_delay
    Pw = np.concatenate([np.zeros(primary_delay), Pw_core])
    
    print(f"Primary path filter Pw length: {len(Pw)} samples ({len(Pw)/sr*1000:.2f} ms)")
    print(f"  - Core filter: {len(Pw_core)} taps, Zero-padding: {primary_delay} samples")
    
    print("=" * 60)
    print("FxLMS Algorithm with Real RIR")
    print("=" * 60)
    print(f"Sample rate: {sr} Hz")
    print(f"RIR length: {len(rir)} samples ({len(rir)/sr*1000:.2f} ms)")
    print(f"Secondary path S(z) length: {len(Sw)} samples ({len(Sw)/sr*1000:.2f} ms)")
    print(f"Primary path P(z): {primary_delay} samples delay + {len(Pw)}-tap FIR (RIR spectral match)")

    #%% PART 1: SECONDARY PATH IDENTIFICATION
    # The first task is to estimate S(z) using the LMS algorithm
    
    print("\n" + "=" * 60)
    print("PART 1: Secondary Path Identification")
    print("=" * 60)
    
    # Use fixed duration for identification (always white noise)
    T_iden = int(T_seconds * sr)  # Use T_seconds from config for identification
    
    # Generate white noise for identification
    np.random.seed(42)  # For reproducibility
    x_iden = np.random.randn(T_iden)
    
    # Send the noise through the actual secondary path S(z)
    y_iden = np.convolve(x_iden, Sw, mode='full')[:T_iden]  # Truncate to T_iden samples
    
    # Initialize the secondary path estimate Sh(z)
    # Use a filter length that can capture the main features of the RIR
    Sh_len = min(1024*4, len(Sw))  # Filter length for Sh(z) - balance between accuracy and computation
    Shx = np.zeros(Sh_len)  # State buffer (input samples)
    Shw = np.zeros(Sh_len)  # Filter weights/coefficients
    e_iden = np.zeros(T_iden)    # Identification error buffer
    
    # Apply vanilla LMS algorithm for identification
    # Learning rate needs to be smaller for longer filters and real RIRs
    mu_iden = 0.01 / Sh_len  # Normalized learning rate
    
    for k in range(T_iden):
        # Update the state buffer (shift and insert new sample)
        Shx = np.roll(Shx, 1)
        Shx[0] = x_iden[k]
        
        # Calculate output of Sh(z)
        Shy = np.dot(Shx, Shw)
        
        # Calculate identification error
        e_iden[k] = y_iden[k] - Shy
        
        # Update weights using LMS rule
        Shw = Shw + mu_iden * e_iden[k] * Shx
    
    print(f"Final identification error (last sample): {e_iden[-1]:.6f}")
    print(f"Mean squared error (last 100 samples): {np.mean(e_iden[-100:]**2):.6f}")

    #%% PART 2: ACTIVE NOISE CONTROL USING FxLMS
    
    print("\n" + "=" * 60)
    print("PART 2: Active Noise Control using FxLMS")
    print("=" * 60)
    
    # Load or generate the input signal x(k) for control
    if input_type == "white_noise":
        # Generate white noise excitation
        T = int(T_seconds * sr)
        np.random.seed(123)  # Different seed from identification
        input = np.random.randn(T)
        print(f"Input signal: White noise ({T_seconds} s, {T} samples)")
    else:
        # Load audio file from data/audio/input/
        audio_path = audio_input_dir / input_type
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        input, audio_sr = load_audio(audio_path)
        
        # Resample if necessary to match RIR sample rate
        if audio_sr != sr:
            from scipy.signal import resample
            new_len = int(len(input) * sr / audio_sr)
            input = resample(input, new_len)
            print(f"Resampled audio from {audio_sr} Hz to {sr} Hz")
        
        # Convert to mono if stereo
        if input.ndim > 1:
            input = np.mean(input, axis=1)
        
        # Truncate to max_audio_duration if specified
        if max_audio_duration is not None:
            max_samples = int(max_audio_duration * sr)
            if len(input) > max_samples:
                input = input[:max_samples]
                print(f"Truncated audio to first {max_audio_duration} seconds")
        
        T = len(input)
        T_seconds = T / sr
        print(f"Input signal: {input_type} ({T_seconds:.2f} s, {T} samples)")
    
    # Normalize input signal
    input = input / np.max(np.abs(input))
    
    # Measure the arriving noise at the sensor position (through primary path)
    # Primary path Pw already includes the delay (zero-padded at the beginning)
    # so the delay is reflected in its phase response
    y_primary = np.convolve(input, Pw, mode='full')[:T]  # Convolve with Pw and truncate
    
    # Initialize the controller C(z)
    # Controller length should be long enough to model the inverse of secondary path
    C_len = min(256, Sh_len)  # Controller filter length
    Cx = np.zeros(C_len)   # Controller state buffer (input samples)
    Cw = np.zeros(C_len)   # Controller weights/coefficients
    
    # Initialize secondary path simulation buffer
    Sx = np.zeros(len(Sw))  # Buffer for secondary path input
    
    # Initialize filtered-x buffer for FxLMS
    Xhx = np.zeros(C_len)   # Filtered reference signal buffer
    
    # Reinitialize Shx for the control phase (reuse the identified Shw)
    Shx = np.zeros(Sh_len)
    
    # Output buffers
    e_cont = np.zeros(T)    # Control error (residual noise)
    y_control = np.zeros(T) # Control signal (anti-noise)
    
    # Snapshot logging for magnitude MSE tracking
    snapshot_interval = int(0.05 * sr)  # Save controller weights every 50ms
    Cw_snapshots = []  # List of (sample_index, Cw_copy) tuples
    
    # Apply the FxLMS algorithm
    # Learning rate needs to be smaller for real RIRs
    mu_cont = 0.01 / C_len  # Normalized learning rate for controller
    
    for k in range(T):
        # Update the controller state buffer
        Cx = np.roll(Cx, 1)
        Cx[0] = input[k]
        
        # Calculate the controller output (anti-noise signal before secondary path)
        Cy = np.dot(Cx, Cw)
        
        # Propagate through secondary path (shift buffer and insert)
        Sx = np.roll(Sx, 1)
        Sx[0] = Cy
        
        # Calculate the actual anti-noise arriving at sensor (through S(z))
        y_secondary = np.dot(Sx, Sw)
        
        # Measure the residue (error signal at microphone)
        # e(k) = yp(k) - ys(k) where yp is primary noise, ys is anti-noise
        e_cont[k] = y_primary[k] - y_secondary
        
        # Store control signal for visualization
        y_control[k] = y_secondary
        
        # Update Sh(z) state buffer for filtered-x computation
        Shx = np.roll(Shx, 1)
        Shx[0] = input[k]
        
        # Calculate filtered reference signal x'(k) = Sh(z) * x(k)
        filtered_x = np.dot(Shx, Shw)
        
        # Update filtered-x buffer
        Xhx = np.roll(Xhx, 1)
        Xhx[0] = filtered_x
        
        # Update controller weights using FxLMS rule
        Cw = Cw + mu_cont * e_cont[k] * Xhx
        
        # Save snapshot of controller weights at regular intervals
        if k % snapshot_interval == 0:
            Cw_snapshots.append((k, Cw.copy()))
    
    print(f"Final residual noise (last sample): {e_cont[-1]:.6f}")
    print(f"Mean squared residual (last 100 samples): {np.mean(e_cont[-100:]**2):.6f}")
    print(f"Mean squared primary noise: {np.mean(y_primary**2):.6f}")
    print(f"Noise reduction: {10*np.log10(np.mean(y_primary**2)/np.mean(e_cont[-100:]**2)):.2f} dB")

    #%% PLOTS
    
    # Create time axis in seconds for better visualization
    time_axis = np.arange(T) / sr
    time_axis_samples = np.arange(T)
    time_axis_iden = np.arange(T_iden) / sr  # Time axis for identification (Part 1)
    
    # Plot 1: Secondary Path Identification Results
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Identification error over time
    axes1[0].plot(time_axis_iden, e_iden, label='Identification error', alpha=0.8)
    axes1[0].set_ylabel('Amplitude')
    axes1[0].set_xlabel('Time (s)')
    axes1[0].set_title('Secondary Path Identification Error')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # Subplot 2: Comparison of S(z) and Sh(z) coefficients (first Sh_len samples)
    # Only compare the portion that Sh(z) can model
    tap_time_Sw = np.arange(min(len(Sw), Sh_len)) / sr * 1000  # Time in ms
    tap_time_Shw = np.arange(len(Shw)) / sr * 1000  # Time in ms
    axes1[1].plot(tap_time_Sw, Sw[:len(tap_time_Sw)], 'b-', label='S(z) - Actual RIR', alpha=0.7)
    axes1[1].plot(tap_time_Shw, Shw, 'r--', label='Sh(z) - Estimated', alpha=0.9)
    axes1[1].axvline(x=est_delay/sr*1000, color='g', linestyle=':', label=f'Estimated delay ({est_delay/sr*1000:.2f} ms)')
    axes1[1].set_ylabel('Amplitude')
    axes1[1].set_xlabel('Time (ms)')
    axes1[1].set_title(f'Secondary Path: Actual RIR vs Estimated Sh(z) (first {Sh_len} taps)')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 2: Active Noise Control Results
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Noise residue over time
    axes2[0].plot(time_axis, e_cont, label='Noise residue e(k)', alpha=0.8)
    axes2[0].set_ylabel('Amplitude')
    axes2[0].set_xlabel('Time (s)')
    axes2[0].set_title('Active Noise Control - Residual Noise')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Subplot 2: Noise signal vs Control signal
    axes2[1].plot(time_axis, y_primary, label='Noise signal yp(k) (delayed input)', alpha=0.7)
    axes2[1].plot(time_axis, y_control, 'r:', label='Control signal ys(k)', alpha=0.9)
    axes2[1].set_ylabel('Amplitude')
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_title('Primary Noise (RIR-filtered input) vs Anti-Noise Control Signal')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 3: Additional analysis - Convergence and performance
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 10))
    
    # Subplot 1: Learning curves (MSE over time using moving average)
    window_size = int(0.05 * sr)  # 10ms moving average window
    if window_size < 10:
        window_size = 10
    mse_iden = np.convolve(e_iden**2, np.ones(window_size)/window_size, mode='valid')
    mse_cont = np.convolve(e_cont**2, np.ones(window_size)/window_size, mode='valid')
    mse_time_axis = np.arange(len(mse_iden)) / sr
    
    axes3[0].semilogy(mse_time_axis, mse_iden, label='Identification MSE', alpha=0.8)
    axes3[0].semilogy(np.arange(len(mse_cont)) / sr, mse_cont, label='Control MSE', alpha=0.8)
    axes3[0].set_ylabel('Mean Squared Error')
    axes3[0].set_xlabel('Time (s)')
    axes3[0].set_title(f'Learning Curves (Time-domain Moving Average MSE, window={window_size/sr*1000:.1f}ms)')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)
    
    # Subplot 2: Magnitude MSE of compensated response vs reference response over time
    # Compute reference magnitude response (P(z))
    n_fft_track = 1024
    _, H_Pw_ref = freqz(Pw, worN=n_fft_track, fs=sr)
    mag_Pw_ref = np.abs(H_Pw_ref)
    _, H_Sw_ref = freqz(Sw, worN=n_fft_track, fs=sr)
    
    # Compute magnitude MSE for each snapshot
    mag_mse_times = []
    mag_mse_values = []
    for sample_idx, Cw_snap in Cw_snapshots:
        _, H_Cw_snap = freqz(Cw_snap, worN=n_fft_track, fs=sr)
        H_compensated_snap = H_Cw_snap * H_Sw_ref
        mag_compensated = np.abs(H_compensated_snap)
        # Compute MSE between magnitudes (in dB scale for perceptual relevance)
        mag_Pw_dB = 20 * np.log10(mag_Pw_ref + 1e-10)
        mag_comp_dB = 20 * np.log10(mag_compensated + 1e-10)
        mag_mse = np.mean((mag_Pw_dB - mag_comp_dB) ** 2)
        mag_mse_times.append(sample_idx / sr)
        mag_mse_values.append(mag_mse)
    
    axes3[1].semilogy(mag_mse_times, mag_mse_values, 'b-o', markersize=3, label='Magnitude MSE (dB²)', alpha=0.8)
    axes3[1].set_ylabel('Magnitude MSE (dB²)')
    axes3[1].set_xlabel('Time (s)')
    axes3[1].set_title('Compensated C(z)*S(z) vs Reference P(z) - Magnitude Response MSE')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    
    # Subplot 3: Controller weight evolution (show final weights)
    tap_time_Cw = np.arange(len(Cw)) / sr * 1000  # Time in ms
    axes3[2].plot(tap_time_Cw, Cw, 'b-', label='Controller C(z) weights')
    axes3[2].set_ylabel('Amplitude')
    axes3[2].set_xlabel('Time (ms)')
    axes3[2].set_title(f'Final Controller Weights C(z) ({C_len} taps)')
    axes3[2].legend()
    axes3[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 4: Impulse and Frequency Response Comparison - True RIR vs Primary Path FIR vs Compensated
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Compute frequency responses
    n_fft_plot = 4096
    w_Sw, H_Sw = freqz(Sw, worN=n_fft_plot, fs=sr)
    w_Pw, H_Pw = freqz(Pw, worN=n_fft_plot, fs=sr)
    w_Cw, H_Cw = freqz(Cw, worN=n_fft_plot, fs=sr)
    
    # Compute compensated response: C(z) * S(z) - the actual control signal at the sensor
    H_compensated = H_Cw * H_Sw
    
    # Compute compensated impulse response by convolving C(z) with S(z)
    ir_compensated = np.convolve(Cw, Sw)
    
    # Subplot (0,0): Impulse responses comparison
    time_Sw = np.arange(len(Sw)) / sr * 1000  # Time in ms
    time_Pw = np.arange(len(Pw)) / sr * 1000  # Time in ms
    time_comp = np.arange(len(ir_compensated)) / sr * 1000  # Time in ms
    axes4[0, 0].plot(time_Sw, Sw, 'b-', label=f'S(z) - True RIR ({len(Sw)} taps)', alpha=0.7)
    axes4[0, 0].plot(time_Pw, Pw, 'r--', label=f'P(z) - Primary FIR ({len(Pw)} taps)', alpha=0.9, linewidth=1.5)
    axes4[0, 0].plot(time_comp, ir_compensated, 'g:', label=f'C(z)*S(z) - Compensated ({len(ir_compensated)} taps)', alpha=0.9, linewidth=1.5)
    axes4[0, 0].set_ylabel('Amplitude')
    axes4[0, 0].set_xlabel('Time (ms)')
    axes4[0, 0].set_title('Impulse Response Comparison')
    axes4[0, 0].legend()
    axes4[0, 0].grid(True, alpha=0.3)
    
    # Subplot (0,1): Impulse responses zoomed (first 10ms or Pw length)
    zoom_ms = max(len(Pw) / sr * 1000 * 1.5, 10)  # Zoom to 1.5x Pw length or 10ms
    axes4[0, 1].plot(time_Sw, Sw, 'b-', label=f'S(z) - True RIR', alpha=0.7)
    axes4[0, 1].plot(time_Pw, Pw, 'r--', label=f'P(z) - Primary FIR ({len(Pw)} taps)', alpha=0.9, linewidth=1.5)
    axes4[0, 1].plot(time_comp, ir_compensated, 'g:', label=f'C(z)*S(z) - Compensated', alpha=0.9, linewidth=1.5)
    axes4[0, 1].set_ylabel('Amplitude')
    axes4[0, 1].set_xlabel('Time (ms)')
    axes4[0, 1].set_title(f'Impulse Response Comparison (zoomed to {zoom_ms:.1f} ms)')
    axes4[0, 1].set_xlim([0, zoom_ms])
    axes4[0, 1].legend()
    axes4[0, 1].grid(True, alpha=0.3)
    
    # Subplot (1,0): Magnitude response comparison
    axes4[1, 0].semilogx(w_Sw, 20 * np.log10(np.abs(H_Sw) + 1e-10), 'b-', label='S(z) - True RIR', alpha=0.7)
    axes4[1, 0].semilogx(w_Pw, 20 * np.log10(np.abs(H_Pw) + 1e-10), 'r--', label=f'P(z) - Primary FIR ({len(Pw)} taps)', alpha=0.9)
    axes4[1, 0].semilogx(w_Cw, 20 * np.log10(np.abs(H_compensated) + 1e-10), 'g:', label='C(z)*S(z) - Compensated', alpha=0.9, linewidth=1.5)
    axes4[1, 0].set_ylabel('Magnitude (dB)')
    axes4[1, 0].set_xlabel('Frequency (Hz)')
    axes4[1, 0].set_title('Magnitude Response Comparison')
    axes4[1, 0].set_xlim([20, sr/2])
    axes4[1, 0].legend()
    axes4[1, 0].grid(True, alpha=0.3, which='both')
    
    # Subplot (1,1): Phase response comparison (unwrapped)
    axes4[1, 1].semilogx(w_Sw, np.unwrap(np.angle(H_Sw)), 'b-', label='S(z) - True RIR', alpha=0.7)
    axes4[1, 1].semilogx(w_Pw, np.unwrap(np.angle(H_Pw)), 'r--', label=f'P(z) - Primary FIR ({len(Pw)} taps)', alpha=0.9)
    axes4[1, 1].semilogx(w_Cw, np.unwrap(np.angle(H_compensated)), 'g:', label='C(z)*S(z) - Compensated', alpha=0.9, linewidth=1.5)
    axes4[1, 1].set_ylabel('Phase (radians)')
    axes4[1, 1].set_xlabel('Frequency (Hz)')
    axes4[1, 1].set_title('Phase Response Comparison (Unwrapped)')
    axes4[1, 1].set_xlim([20, sr/2])
    axes4[1, 1].legend()
    axes4[1, 1].grid(True, alpha=0.3, which='both')
    
    fig4.suptitle('True RIR S(z) vs Primary Path P(z) vs Compensated C(z)*S(z)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    #%% SAVE AUDIO FILES
    
    # Create output directory
    output_dir = base / "data" / "audio" / "ex_02"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize signals to prevent clipping
    input_norm = input / np.max(np.abs(input)) * 0.9
    y_primary_norm = y_primary / np.max(np.abs(y_primary)) * 0.9
    y_control_norm = y_control / np.max(np.abs(y_control) + 1e-10) * 0.9
    e_cont_norm = e_cont / np.max(np.abs(e_cont) + 1e-10) * 0.9
    
    # Compute input filtered through true RIR (Sw)
    input_through_Sw = np.convolve(input, Sw, mode='full')[:T]
    input_through_Sw_norm = input_through_Sw / np.max(np.abs(input_through_Sw)) * 0.9
    
    # Save audio files
    save_audio(output_dir / "input_noise.wav", input_norm, sr)
    save_audio(output_dir / "input_through_ref_fir.wav", y_primary_norm, sr)
    save_audio(output_dir / "controlled_signal.wav", y_control_norm, sr)
    save_audio(output_dir / "input_through_rir.wav", input_through_Sw_norm, sr)
    save_audio(output_dir / "noise_residue.wav", e_cont_norm, sr)
    
    print(f"\nAudio files saved to {output_dir}:")
    print(f"  - input_noise.wav: Input noise signal")
    print(f"  - input_through_ref_fir.wav: Input filtered through reference FIR P(z)")
    print(f"  - controlled_signal.wav: Control signal ys(k) from C(z)*S(z)")
    print(f"  - input_through_rir.wav: Input filtered through true RIR S(z)")
    print(f"  - noise_residue.wav: Noise residue e(k) = yp(k) - ys(k)")
    
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
