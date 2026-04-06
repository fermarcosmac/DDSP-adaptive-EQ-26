"""
crop_audio.py — Interactively crop leading (or any) zero/silent samples from a WAV file.

Usage
-----
1. Set SONG_FILENAME to the name of the file you want to inspect.
2. Run the script.  Two things happen:
   a. A waveform plot opens.  Inspect it to decide the crop point.
   b. You are prompted to type a sample index (the first sample to KEEP).
      - Press Enter without typing anything to accept the auto-detected
        first non-zero sample.
      - Type  0  to keep the entire file unchanged.
3. The cropped audio is saved to data/audio/input/songs/ with the suffix
   "_cropped" appended before the extension, preserving the original
   sample rate and bit depth exactly.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
SONG_FILENAME = "QuantumChromos_Circuits_MIX_cropped.wav"   # ← change this to your file
# ──────────────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "audio" / "input" / "songs"
INPUT_PATH = INPUT_DIR / SONG_FILENAME

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
sr, data = wavfile.read(INPUT_PATH)
original_dtype = data.dtype
n_channels = 1 if data.ndim == 1 else data.shape[1]
n_samples  = data.shape[0]
duration_s = n_samples / sr

print(f"Loaded : {INPUT_PATH.name}")
print(f"  Sample rate : {sr} Hz")
print(f"  Bit depth   : {original_dtype}")
print(f"  Channels    : {n_channels}")
print(f"  Samples     : {n_samples}  ({duration_s:.3f} s)")

# ---------------------------------------------------------------------------
# Auto-detect first non-zero sample
# ---------------------------------------------------------------------------
mono = data if data.ndim == 1 else data.mean(axis=1)
nonzero_indices = np.where(mono != 0)[0]
auto_crop = int(nonzero_indices[0]) if len(nonzero_indices) > 0 else 0
print(f"\nAuto-detected first non-zero sample: {auto_crop}  ({auto_crop / sr * 1000:.2f} ms)")

# ---------------------------------------------------------------------------
# Plot waveform
# ---------------------------------------------------------------------------
sample_axis = np.arange(n_samples)

fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3 * n_channels),
                          squeeze=False, sharex=True)
fig.suptitle(SONG_FILENAME)

channel_data = [data] if data.ndim == 1 else [data[:, c] for c in range(n_channels)]
ch_labels    = ["Mono"] if n_channels == 1 else [f"Ch {c+1}" for c in range(n_channels)]

for ax, ch, lbl in zip(axes.flatten(), channel_data, ch_labels):
    ax.plot(sample_axis, ch, linewidth=0.3, color="steelblue")
    ax.axvline(auto_crop, color="red", linewidth=1.2, linestyle="--",
               label=f"Auto crop @ sample {auto_crop}")
    ax.set_ylabel(f"{lbl}\nAmplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

axes[-1, 0].set_xlabel("Sample index")
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ---------------------------------------------------------------------------
# Ask user for crop point
# ---------------------------------------------------------------------------
prompt = (
    f"\nEnter the first sample index to KEEP "
    f"[default = auto-detected {auto_crop}, 0 = keep all]: "
)
try:
    user_input = input(prompt).strip()
    crop_sample = auto_crop if user_input == "" else int(user_input)
except (EOFError, ValueError):
    crop_sample = auto_crop

crop_sample = max(0, min(crop_sample, n_samples))
print(f"Cropping at sample {crop_sample}  ({crop_sample / sr * 1000:.2f} ms)")

# ---------------------------------------------------------------------------
# Crop and save
# ---------------------------------------------------------------------------
cropped = data[crop_sample:]

stem   = INPUT_PATH.stem
suffix = INPUT_PATH.suffix
output_path = INPUT_DIR / f"{stem}_cropped{suffix}"

wavfile.write(output_path, sr, cropped.astype(original_dtype))

out_duration = cropped.shape[0] / sr
print(f"\nSaved  : {output_path.name}")
print(f"  Samples kept : {cropped.shape[0]}  ({out_duration:.3f} s)")
print(f"  Removed      : {crop_sample} sample(s) from the beginning")

# Keep the plot open until the user closes it
plt.show()
