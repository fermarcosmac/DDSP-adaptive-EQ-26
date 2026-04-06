"""
adaptive_equalization_skeleton.py

Skeleton code to simulate a time-varying room impulse response (RIR)
applied to an input audio file using an overlap-add scheme.

Behavior:
- Load an audio file from data/audio
- Load RIRs from data/rir (we'll pick two by default)
- The user supplies a sequence of RIR indices and corresponding start times
  (in seconds) describing when each RIR becomes active.
- The audio is processed in short analysis frames (window/hop). For each
  frame we pick the active RIR (by frame midpoint) and convolve the frame
  with that RIR. The convolved chunks are overlap-added into the output
  buffer.

Dependencies:
- numpy
- scipy
- soundfile (pysoundfile)

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_audio,
    save_audio,
    load_rirs,
    ensure_rirs_sample_rate,
    simulate_time_varying_rir,
    rms,
)


if __name__ == "__main__":
    # Set paths
    base = Path(".")
    audio_path = base / "data" / "audio" / "input" / "guitar-riff.wav"
    rir_dir = base / "data" / "rir"
    out_path = base / "data" / "audio" / "output" / "output.wav"

    # Load audio
    audio, sr = load_audio(audio_path)
    duration_s = len(audio) / sr
    print(f"Loaded audio: {audio_path}, {len(audio)} samples, {sr} Hz, {duration_s:.2f} s")

    # Load RIRs (pick 2 for now)
    max_n_rirs = 3
    rirs, rirs_srs = load_rirs(rir_dir, max_n=max_n_rirs)
    if len(rirs) < max_n_rirs:
        raise RuntimeError("Need at least two RIR files in data/rir for this demo")

    # Ensure RIRs have same sample rate as audio
    rirs = ensure_rirs_sample_rate(rirs, rirs_srs, sr)

    # Define RIR sequence and start times (seconds)
    # Example: use RIR 0 from 0s, switch to RIR 1 at halfway through the audio
    rir_indices = [int(k) for k in np.arange(max_n_rirs)]
    start_times_s = [0.0, duration_s / 3.0, duration_s / 3.0 * 2]

    # Simulate
    y, _ = simulate_time_varying_rir(audio, sr, rirs, rir_indices, start_times_s, window_ms=100, hop_ms=50)
    y = y*rms(audio)/rms(y)  # normalize output to input RMS

    # Plot results
    time_axis = np.arange(0, len(y)) / sr
    plt.figure()
    plt.plot(time_axis[:len(audio)],audio, label="Input")
    plt.plot(time_axis, y, label="Variable Room Output")
    # draw vertical change-point lines, add label only once
    for i, xt in enumerate(start_times_s):
        if i == 0:
            pass # First RIR is not a "change"
        else:
            plt.axvline(xt, color="red", linestyle="--")
    plt.title("Input and Time-Varying RIR Output")
    plt.legend()
    plt.show()

    # Save result
    save_audio(out_path, y, sr)
    print(f"Saved simulated time-varying output to: {out_path}")