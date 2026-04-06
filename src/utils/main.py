"""
Utility functions used exclusively by the main experiment script.

Covers:
- Input signal stem derivation (song_stem)
- Experiment parameter grid construction (build_experiment_grid)
- Audio file saving (save_wav)
- Per-run result accumulation into curves and compute_time_stats dicts
- EQ audio output filename construction
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchaudio

from utils.common import iter_param_grid


# ---------------------------------------------------------------------------
# Input signal helpers
# ---------------------------------------------------------------------------

def song_stem(input_spec) -> str:
    """Derive a filesystem-safe stem from an input_spec tuple.

    Args:
        input_spec: Either a (mode, info) tuple or a plain string.

    Returns:
        A short, filesystem-safe string identifying the input signal.
    """
    if isinstance(input_spec, (list, tuple)) and len(input_spec) == 2:
        mode, info = input_spec
        if mode == "white_noise":
            if isinstance(info, dict) and "realization_idx" in info:
                try:
                    idx = int(info["realization_idx"])
                    return f"white_noise_{idx:03d}"
                except (TypeError, ValueError):
                    pass
            return "white_noise"
        if isinstance(info, dict) and "path" in info:
            return Path(info["path"]).stem
        return str(mode)
    return str(input_spec)


# ---------------------------------------------------------------------------
# Experiment grid construction
# ---------------------------------------------------------------------------

def build_experiment_grid(cfg: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str, float]]:
    """Parse the main experiment config and return a flat list of run descriptors.

    Each element is a ``(sim_cfg, loss_type, mu)`` triple ready to be passed
    directly to ``run_control_experiment``.

    Handles:
    - Pairing of ``optim_type`` and ``mu_opt`` (flat list or per-loss-type dict).
    - Per-loss-type scalar overrides for ``lambda_newton`` and ``eps_0``.
    - ``hop_len`` always set equal to ``frame_len``.

    Args:
        cfg: Parsed JSON config dict (as returned by ``load_config``).

    Returns:
        List of fully resolved ``sim_cfg`` dicts.

    Raises:
        ValueError: On length mismatches or missing required fields.
    """
    sim_param_grid = cfg.get("simulation_params", {})
    if not sim_param_grid:
        raise ValueError("No 'simulation_params' section found in config.")

    optim_list: List[str] = sim_param_grid.get("optim_type", [])
    mu_opt_raw = sim_param_grid.get("mu_opt", [])

    mu_per_loss: Dict[str, List[float]] = {}
    mu_list: List[float] = []

    if isinstance(mu_opt_raw, dict):
        mu_per_loss = mu_opt_raw
        for lt_key, lt_mus in mu_per_loss.items():
            if optim_list and len(optim_list) != len(lt_mus):
                raise ValueError(
                    f"Length mismatch for mu_opt['{lt_key}']: "
                    f"got {len(lt_mus)} values but {len(optim_list)} optimizers."
                )
    else:
        mu_list = mu_opt_raw
        if optim_list and mu_list and len(optim_list) != len(mu_list):
            raise ValueError(
                "Length mismatch between 'optim_type' and 'mu_opt' in config: "
                f"got {len(optim_list)} optimizers and {len(mu_list)} mu_opt values. "
                "They must have the same length so each optimizer pairs with one mu_opt."
            )

    _per_loss_scalar_keys = ("lambda_newton", "eps_0")
    lambda_newton_raw = sim_param_grid.get("lambda_newton", None)
    eps_0_raw = sim_param_grid.get("eps_0", None)

    lambda_newton_per_loss: Dict[str, Any] = {}
    eps_0_per_loss: Dict[str, Any] = {}
    if isinstance(lambda_newton_raw, dict):
        lambda_newton_per_loss = lambda_newton_raw
    if isinstance(eps_0_raw, dict):
        eps_0_per_loss = eps_0_raw

    base_param_grid = {
        k: v
        for k, v in sim_param_grid.items()
        if k not in ("optim_type", "mu_opt", "hop_len")
        and not (k in _per_loss_scalar_keys and isinstance(v, dict))
    }

    resolved_cfgs: List[Dict[str, Any]] = []
    for base_cfg in iter_param_grid(base_param_grid):
        current_lt: str = base_cfg.get("loss_type", "")

        if mu_per_loss:
            resolved_mu_list = mu_per_loss.get(current_lt, list(mu_per_loss.values())[0])
        else:
            resolved_mu_list = mu_list

        resolved_lambda_newton = (
            _unwrap(lambda_newton_per_loss.get(current_lt, list(lambda_newton_per_loss.values())[0]))
            if lambda_newton_per_loss else None
        )
        resolved_eps_0 = (
            _unwrap(eps_0_per_loss.get(current_lt, list(eps_0_per_loss.values())[0]))
            if eps_0_per_loss else None
        )

        if optim_list and resolved_mu_list:
            opt_mu_pairs = list(zip(optim_list, resolved_mu_list))
        else:
            opt_mu_pairs = [(None, None)]

        for optim, mu in opt_mu_pairs:
            sim_cfg = dict(base_cfg)
            if "frame_len" not in sim_cfg:
                raise ValueError("'frame_len' must be provided in simulation_params.")
            sim_cfg["hop_len"] = sim_cfg["frame_len"]
            if optim is not None:
                sim_cfg["optim_type"] = optim
            if mu is not None:
                sim_cfg["mu_opt"] = mu
            if resolved_lambda_newton is not None:
                sim_cfg["lambda_newton"] = resolved_lambda_newton
            if resolved_eps_0 is not None:
                sim_cfg["eps_0"] = resolved_eps_0
            resolved_cfgs.append(sim_cfg)

    return resolved_cfgs


def _unwrap(v: Any) -> Any:
    """Unwrap a single-element list to its scalar; pass other values through."""
    return v[0] if isinstance(v, list) and len(v) == 1 else v


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def save_wav(path: Path, arr: np.ndarray, sr: int) -> None:
    """Save a numpy audio array as a peak-normalised WAV file.

    Args:
        path: Output file path (created if missing).
        arr:  1-D float numpy array.
        sr:   Sample rate in Hz.
    """
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak
    t = torch.from_numpy(x).float().unsqueeze(0)
    torchaudio.save(str(path), t, int(sr))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Result accumulation
# ---------------------------------------------------------------------------

def make_compute_time_stats() -> defaultdict:
    """Return the defaultdict used to accumulate per-key compute-time statistics."""
    return defaultdict(
        lambda: {
            "total_time_s": 0.0,
            "total_frames": 0,
            "num_runs": 0,
            "min_avg_time_per_frame_s": float("inf"),
            "max_avg_time_per_frame_s": float("-inf"),
        }
    )


def accumulate_compute_time(
    stats: dict,
    total_time_s: float,
    n_frames: int,
) -> None:
    """Update a compute-time stats bucket in-place with one run's results.

    Args:
        stats:        The dict bucket to update (from ``make_compute_time_stats()``).
        total_time_s: Wall-clock seconds for the run.
        n_frames:     Number of processed frames in the run.
    """
    stats["total_time_s"] += float(total_time_s)
    stats["total_frames"] += int(n_frames)
    stats["num_runs"] += 1
    if n_frames > 0:
        avg = float(total_time_s) / int(n_frames)
        stats["min_avg_time_per_frame_s"] = min(float(stats["min_avg_time_per_frame_s"]), avg)
        stats["max_avg_time_per_frame_s"] = max(float(stats["max_avg_time_per_frame_s"]), avg)


# ---------------------------------------------------------------------------
# Audio output filename helpers
# ---------------------------------------------------------------------------

def eq_audio_filename(
    optim: str,
    loss_type: str,
    frame_len: int,
    transition_time_s: float,
    stem: str,
) -> str:
    """Build the per-run EQ output WAV filename.

    Args:
        optim:            Optimizer name string (e.g. ``"SGD"``).
        loss_type:        Loss type string (e.g. ``"FD-MSE"``).
        frame_len:        Frame length in samples.
        transition_time_s: Transition duration in seconds.
        stem:             Song/input stem (from ``song_stem``).

    Returns:
        Filename string (no directory component).
    """
    safe_optim = optim.replace("-", "_").replace(" ", "_")
    safe_lt = loss_type.replace("-", "_")
    safe_tt = str(transition_time_s).replace(".", "p")
    safe_fl = str(int(frame_len))
    return f"EQ_{safe_optim}_{safe_lt}_fl{safe_fl}_tt{safe_tt}_{stem}.wav"
