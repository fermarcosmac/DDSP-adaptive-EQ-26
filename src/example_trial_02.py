import os
import sys
from pathlib import Path
import torch
import torchaudio
# Ensure the workspace root is first on sys.path so the local package is imported
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from local_dasp_pytorch.modules import ParametricEQ
import matplotlib.pyplot as plt
import numpy as np


#%% SETUP: LOAD AUDIO AND DEFINE PROCESSOR WITH GROUND-TRUTH PARAMETERS

# Load audio
x, sr = torchaudio.load("data/audio/input/guitar-riff.wav")

# create batch dim
# (batch_size, n_channels, n_samples)
x = x.unsqueeze(0)

# Define desired EQ parameters (migght be clipped later to allowed EQ ranges)
low_shelf_gain_db       = torch.tensor([-10.0])
low_shelf_cutoff_freq   = torch.tensor([100.0])
low_shelf_q_factor      = torch.tensor([1.0])
band0_gain_db           = torch.tensor([10.0])
band0_cutoff_freq       = torch.tensor([500.0])
band0_q_factor          = torch.tensor([1.0])
band1_gain_db           = torch.tensor([-10.0])
band1_cutoff_freq       = torch.tensor([2000.0])
band1_q_factor          = torch.tensor([1.0])
band2_gain_db           = torch.tensor([10.0])
band2_cutoff_freq       = torch.tensor([4000.0])
band2_q_factor          = torch.tensor([1.0])
band3_gain_db           = torch.tensor([-10.0])
band3_cutoff_freq       = torch.tensor([8000.0])
band3_q_factor          = torch.tensor([1.0])
high_shelf_gain_db      = torch.tensor([10.0])
high_shelf_cutoff_freq  = torch.tensor([7000.0])
high_shelf_q_factor     = torch.tensor([1.0])


# create processor for given sample rate
EQ = ParametricEQ(sample_rate=sr)

# assume bs = batch size (e.g. 1)
bs = 1

# Denormalized parameters (actual ranges), build a dict:
denorm = {
    "low_shelf_gain_db": low_shelf_gain_db.view(bs),     # tensor shape (bs,)
    "low_shelf_cutoff_freq": low_shelf_cutoff_freq.view(bs),
    "low_shelf_q_factor": low_shelf_q_factor.view(bs),
    "band0_gain_db": band0_gain_db.view(bs),
    "band0_cutoff_freq": band0_cutoff_freq.view(bs),
    "band0_q_factor": band0_q_factor.view(bs),
    "band1_gain_db": band1_gain_db.view(bs),
    "band1_cutoff_freq": band1_cutoff_freq.view(bs),
    "band1_q_factor": band1_q_factor.view(bs),
    "band2_gain_db": band2_gain_db.view(bs),
    "band2_cutoff_freq": band2_cutoff_freq.view(bs),
    "band2_q_factor": band2_q_factor.view(bs),
    "band3_gain_db": band3_gain_db.view(bs),
    "band3_cutoff_freq": band3_cutoff_freq.view(bs),
    "band3_q_factor": band3_q_factor.view(bs),
    "high_shelf_gain_db": high_shelf_gain_db.view(bs),
    "high_shelf_cutoff_freq": high_shelf_cutoff_freq.view(bs),
    "high_shelf_q_factor": high_shelf_q_factor.view(bs),
}


# Build clipped (denormalized but clipped to allowed ranges) dict and normalized tensor
clipped = {}
norm_list = []
for name in EQ.param_ranges.keys():
    minv, maxv = EQ.param_ranges[name]
    v = denorm[name]
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)

    min_t = torch.tensor(minv).type_as(v)
    max_t = torch.tensor(maxv).type_as(v)

    # clipped: keep values in real units (not normalized)
    clipped_val = v.clamp(min_t, max_t)
    clipped[name] = clipped_val

    # normalized value in [0,1]
    norm = (clipped_val - min_t) / (max_t - min_t)
    norm_list.append(norm)

param_tensor = torch.stack(norm_list, dim=1)   # shape (bs, num_params)

# process: proc.process_normalized expects normalized params (0..1)
y = EQ.process_normalized(x, param_tensor)

# create optimizable normalized parameters by sampling in [0,1]
optim_params = []
for name in EQ.param_ranges.keys():
    # sample a starting normalized value uniformly in [0,1]
    init_val = torch.rand(1)
    p = torch.nn.Parameter(init_val)
    optim_params.append(p)

# Adam optimizer over the iterable of parameters
optimizer = torch.optim.Adam(optim_params, lr=0.01)


#%% PARAMETER OPTIMIZATION LOOP
n_iters = 400
# prepare tracking structures
losses = []
param_names = list(EQ.param_ranges.keys())
param_errors = {name: [] for name in param_names}

for n in range(n_iters):
    # build normalized parameter tensor from the current Parameters (shape: bs, num_params)
    param_tensor = torch.stack([p.view(1).to(x.device) for p in optim_params], dim=1)
    #param_tensor = torch.clip(param_tensor, 0.0, 1.0)

    # apply EQ with the estimated parameter tensor
    y_hat = EQ.process_normalized(x, param_tensor)

    # compute distance between estimate and target
    loss = torch.nn.functional.mse_loss(y_hat, y)

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Hard clip parameters to [0,1] after each step
    with torch.no_grad():
        for p in optim_params:
            p.clamp_(0.0, 1.0)

    # record loss
    losses.append(float(loss.item()))

    # record per-parameter normalized absolute error vs ground truth (clipped)
    cur_norm = param_tensor.detach().cpu().view(-1).numpy()
    for i, name in enumerate(param_names):
        minv, maxv = EQ.param_ranges[name]
        fv = float(cur_norm[i])
        gt = float(clipped[name].cpu().view(-1).numpy()[0])
        # ground-truth normalized to [0,1]
        gt_norm = (gt - minv) / (maxv - minv)
        param_errors[name].append(abs(fv - gt_norm))

    print(f"step: {n+1}/{n_iters}, loss: {loss.item():.3e}\r")


#%% PLOTTING RESULTS

# Plot original, EQ-ed, and optimized signals on the same axis for comparison
plt.figure()
plt.plot(x.squeeze().cpu().numpy(), label="Input")
plt.plot(y.squeeze().cpu().numpy(), label="Output (ground truth)")
plt.plot(y_hat.squeeze().detach().cpu().numpy(), label="Output (optimized)")
plt.title("Signals Comparison")
plt.legend()

# Plot per-parameter absolute error and overall loss (2x1) with log y-scales
eps = 1e-12
fig, (ax1, ax2) = plt.subplots(2, 1)
# top: per-parameter absolute error over iterations (log scale)
for name in param_names:
    vals = np.maximum(np.array(param_errors[name], dtype=float), eps)
    ax1.plot(vals, label=name)
ax1.set_ylabel('Absolute error (denormalized units)')
ax1.set_title('Per-parameter absolute error during optimization')
ax1.set_yscale('log')
ax1.grid(which='both', axis='y', linestyle='--', linewidth=0.5)
ax1.legend(loc='upper right', ncol=2, fontsize='small')
# bottom: overall loss (log scale)
losses_arr = np.maximum(np.array(losses, dtype=float), eps)
ax2.plot(losses_arr)
ax2.set_ylabel('MSE loss')
ax2.set_xlabel('Iteration')
ax2.set_title('Loss over optimization')
ax2.set_yscale('log')
ax2.grid(which='both', axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# After optimization: compare ground-truth (clipped) parameters vs optimized final values
param_names = list(EQ.param_ranges.keys())
final_norm = torch.stack([p.detach().cpu().view(-1) for p in optim_params], dim=0).view(-1).numpy()

final_denorm = []
ground_truth = []
for i, name in enumerate(param_names):
    minv, maxv = EQ.param_ranges[name]
    fv = float(final_norm[i])
    denorm_val = minv + fv * (maxv - minv)
    final_denorm.append(denorm_val)

    gt = float(clipped[name].cpu().view(-1).numpy()[0])
    ground_truth.append(gt)


# Show normalized parameter values (0..1) for ground-truth and optimized final
x = np.arange(len(param_names))
width = 0.35
ground_truth_norm = []
for i, name in enumerate(param_names):
    minv, maxv = EQ.param_ranges[name]
    gt = float(clipped[name].cpu().view(-1).numpy()[0])
    gt_norm = (gt - minv) / (maxv - minv)
    ground_truth_norm.append(gt_norm)

final_norm_list = list(final_norm)

plt.figure()
plt.bar(x - width/2, ground_truth_norm, width, label='ground truth (normalized)')
plt.bar(x + width/2, final_norm_list, width, label='optimized (normalized)')
plt.xticks(x, param_names, rotation=45, ha='right')
plt.ylabel('Normalized value (0..1)')
plt.title('Parametric EQ: normalized ground truth vs optimized')
plt.legend()
plt.tight_layout()
plt.show()

hey = 0