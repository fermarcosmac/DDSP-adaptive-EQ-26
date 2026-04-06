import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
# Ensure the workspace root is first on sys.path so the local package is imported
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from lib.local_dasp_pytorch.modules import ParametricEQ
from modules import EQController_dasp, EQLogger
from utils import (
    load_audio,
    get_compensation_EQ_params,
    load_rirs,
    ensure_rirs_sample_rate,
    simulate_time_varying_process,
    rms,
    save_audio,
    get_delay_from_ir,
    get_delay_xcorr,
)


# Define configuration for EQController_dasp here for now (better to pass it from file)
# TODO delete unnecessary fields
EQController_config = {
    "method": "TD-FxLMS",           # adaptation method
    "estLEM_desired_length": 4096,  # memory length for LEM estimate
    "estLEM_sustain_ms": 1000,       # [ms] time to sustain previous LEM estimate
    "estLEM_ridge_lambda": 10.0,    # Ridge L2 regularization parameter for LEM estimation
    "n_iter_opt": 5,                 # number of optimization iterations for EQ parameter update
    # TODO: check theory, I THINK I CAN ONLY DO ONE STEP PER FRAME!
}


#%% MAIN SCRIPT

if __name__ == "__main__":

    #%% CONFIGURATION AND SETUP

    # Set paths
    base = Path(".")
    audio_path = base / "data" / "audio" / "input" / "guitar-riff_short.wav"
    rir_dir = base / "data" / "rir"

    # Set experiment parameters
    n_rirs = 1  # number of RIRs to use
    #switch_times_norm = [t/(n_rirs-1)*0.6 for t in range(0,n_rirs)] # times to switch RIRs (normalized)
    switch_times_norm = [1.0]  # NORMALIZED RANGE
    ROI = [100.0, 14000.0]  # region of interest for EQ compensation (Hz)

    # Load probe and ground-truth RIR
    input, sr = load_audio(audio_path) # input audio signal (nsures mono)
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs)
    rirs = ensure_rirs_sample_rate(rirs, rirs_srs, sr)

    # Parametric EQ estimation for virtual room compensation and delay estimation
    rir_init = rirs[0]
    EQ_comp_dict = get_compensation_EQ_params(rir_init, sr, ROI, num_sections=6)

    # Prepare differentiable EQ module with initial compensation parameters
    EQ = ParametricEQ(sample_rate=sr)
    dasp_param_dict = { k: torch.as_tensor(v, dtype=torch.float32).view(1) for k, v in EQ_comp_dict["eq_params"].items() }
    _, init_params_tensor = EQ.clip_normalize_param_dict(dasp_param_dict) # initial normalized parameter vector


    # Estimate delay in LEM path from RIR
    est_LEM_delay = get_delay_from_ir(rir_init, sr)

    # Estimate delay introduced by the EQ compensation
    # Use the input-output crosscorrelation using noise input
    noise_input = torch.randn(1,1,sr*2)  # 2 seconds of white noise
    with torch.no_grad():
        EQed_noise = EQ.process_normalized(noise_input, init_params_tensor)
    est_EQ_delay = get_delay_xcorr(noise_input.squeeze().cpu().numpy(), EQed_noise.squeeze().cpu().numpy(), sr)
    est_EQ_delay = est_EQ_delay + 105 # HARD-CODED CORRECTION FOR THE EQ DELAY ESTIMATION BUG
    # TODO: I think I'm going to need an adative delay estimation during the process... I'm not sure, estimations seem too volatile

    # Transform audio and RIRs to Torch tensors on appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.as_tensor(input, dtype=torch.float32, device=device).view(1,1,-1)
    rirs = [torch.as_tensor(rir, dtype=torch.float32, device=device).view(1,1,-1) for rir in rirs]

    # Prepare controller and logger for adaptive EQ (to be defined)
    # TODO
    logger = EQLogger()  
    logger = None
    # EQController_dasp will take EQ module with its initial parameters and implement adaptation logic
    EQController_dasp = EQController_dasp(
        EQ = EQ,
        init_params_tensor = init_params_tensor,
        est_LEM_delay = est_LEM_delay,
        est_EQ_delay = est_EQ_delay,
        config = EQController_config,
        logger = logger,
        roi = ROI,
    )
                     # Maybe it should be an attribute of the EQController_dasp class...

    #%% PLAYBACK

    # Playback simulation:
    switch_times_s = [t * (input.shape[-1]/ sr) for t in switch_times_norm]
    rir_indices = list(range(n_rirs))  # use RIRS in order
    y, sr = simulate_time_varying_process(
        audio=input,
        sr=sr,
        rirs=rirs,
        rir_indices=rir_indices,
        switch_times_s=switch_times_s,
        EQ=EQ,                              # It is very important that this EQ instance is the same as in the controller
        controller=EQController_dasp,
        logger=logger,
        window=124,
        hop=1,
        win_hop_units="samples",)
    
    # For comparison purposes, also simulate withoput compensation EQ
    y_noEQ, sr = simulate_time_varying_process(
        audio=input,
        sr=sr,
        rirs=rirs,
        rir_indices=rir_indices,
        switch_times_s=switch_times_s,
        EQ=None,
        controller=None,
        logger=logger,
        window=124,
        hop=1,
        win_hop_units="samples",)

    #%% PLOTS
    # plot and show input, output signals and switch times
    # Extract audio data from 3D tensors (1, 1, N) -> 1D
    input_1d = input.squeeze().detach().cpu().numpy()
    y_1d = y.squeeze().detach().cpu().numpy()
    y_noEQ_1d = y_noEQ.squeeze().detach().cpu().numpy()
    
    # Normalize signals for comparison using RMS normalization
    rms_in = rms(input_1d)
    rms_y = rms(y_1d)
    rms_y_noEQ = rms(y_noEQ_1d)
    input_1d_norm = input_1d / rms_in
    y_1d_norm = y_1d / rms_y
    y_noEQ_1d_norm = y_noEQ_1d / rms_y_noEQ
    
    time_axis = np.arange(0, len(y_1d_norm)) / sr
    
    # Plot 1: Audio signals with RIR switch times
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(0, len(input_1d_norm)) / sr, input_1d_norm, label="Input", alpha=0.7)
    plt.plot(time_axis, y_noEQ_1d_norm, label="Variable Room Output (No EQ)", alpha=0.7)
    plt.plot(time_axis, y_1d_norm, label="Variable Room Output (With EQ)", alpha=0.7)
    for i, xt in enumerate(switch_times_s):
        plt.axvline(x=xt, color='k', linestyle='--', label="RIR Switch" if i==0 else None)
    plt.title("Input and Variable Room Output Signals (RMS Normalized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    
    
    # Plot 2: Loss progression by frames
    if logger is not None and len(logger.frames_start_samples) > 0:
        loss_time_axis = np.array(logger.frames_start_samples) / sr  # Convert frame start samples to time
        plt.figure(figsize=(12, 4))
        plt.semilogy(loss_time_axis, logger.loss_by_frames, marker='', markersize=4, linewidth=1, label="MSE Loss")
        for i, xt in enumerate(switch_times_s):
            plt.axvline(x=xt, color='r', linestyle='--', alpha=0.7, label="RIR Switch" if i==0 else None)
        plt.ylim([1e-3, 1e1])
        plt.title("Adaptation Loss Over Time (TD-FxLMS)")
        plt.xlabel("Time (s)")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 3: LEM and EQ delay estimates over time
    if logger is not None and len(logger.LEM_delay_by_frames) > 0:
        LEM_time_axis = logger.LEM_delay_log[:, 1] / sr
        LEM_delays = logger.LEM_delay_log[:, 0]
        EQ_time_axis = logger.EQ_delay_log[:, 1] / sr
        EQ_delays = logger.EQ_delay_log[:, 0]
        
        plt.figure(figsize=(12, 4))
        plt.plot(LEM_time_axis, LEM_delays, marker='o', markersize=4, linewidth=1, label="LEM Delay", color='tab:blue')
        plt.plot(EQ_time_axis, EQ_delays, marker='s', markersize=4, linewidth=1, label="EQ Delay", color='tab:orange')
        for i, xt in enumerate(switch_times_s):
            plt.axvline(x=xt, color='r', linestyle='--', alpha=0.7, label="RIR Switch" if i==0 else None)
        plt.title("Estimated Delays Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Delay (samples)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 4: Measured output vs delayed input (delay alignment check)
    total_delay = est_LEM_delay + est_EQ_delay
    # Create delayed input signal
    delayed_input = np.zeros_like(y_1d_norm)
    if total_delay > 0 and total_delay < len(input_1d_norm):
        copy_len = min(len(input_1d_norm) - total_delay, len(delayed_input) - total_delay)
        delayed_input[total_delay:total_delay + copy_len] = input_1d_norm[:copy_len]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, delayed_input, label=f"Input (delayed by {total_delay} samples)", alpha=0.7)
    plt.plot(time_axis, y_1d_norm, label="Measured Output (With EQ)", alpha=0.7)
    for i, xt in enumerate(switch_times_s):
        plt.axvline(x=xt, color='k', linestyle='--', label="RIR Switch" if i==0 else None)
    plt.title(f"Delay Alignment Check: Input delayed by {total_delay} samples vs Measured Output")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()


    # Save audio files to output directory
    output_dir = base / "data" / "audio" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_audio(output_dir / "input.wav", input_1d_norm, sr)
    save_audio(output_dir / "output_no_eq.wav", y_noEQ_1d_norm, sr)
    save_audio(output_dir / "output_with_eq.wav", y_1d_norm, sr)
    print(f"Audio files saved to {output_dir}")



    #    - Static room - no EQ
    #    - Static room + initial EQ
    #    - Dynamic room - no EQ
    #    - Dynamic room + adaptive EQ

    # Common interface for different adaptive EQ settings (my class, torch.optim?)
    #    - Pytorch optimizers (including Newton library)
    #    - GHAM adaptive filter
    #    - Neural network-based adaptive EQ (MLP/RNN)

    # They should take in the error signal and output EQ parameters at each correction step
    # Make sure each correctioin step fits in real-time constraints (e.g., 10 ms processing time budget)

    pass