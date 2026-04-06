import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
import matplotlib.pyplot as plt
# Don't import anything from utils (circular import issues)

#TODO
class EQController_dasp:
    def __init__(self, EQ, init_params_tensor, est_LEM_delay: int = 0, est_EQ_delay: int = 0, config: dict = None, logger=None, roi=None):
        self.EQ = EQ                                                            # ParametricEQ object (dasp_pytorch)
        self.params = torch.nn.Parameter(init_params_tensor.clone().detach().requires_grad_(True))  # (1 x n_params) torch Parameter
        self.prev_params: torch.Tensor = torch.zeros_like(self.params)   # (1 x n_params) torch tensor
        self.est_LEM_delay = est_LEM_delay                                      # estimated delay of LEM system (samples)
        self.est_EQ_delay = est_EQ_delay                                        # estimated delay introduced by EQ (samples)
        self.config: dict = config if config is not None else {}               # configuration dictionary
        self.logger = logger                                                    # logger object
        self.roi = roi                                                          # region of interest (Hz) for LEM estimation
        # The optimizer should depend on the chosen method TODO
        self.optimizer = torch.optim.SGD([self.params], lr=1e-6)        # optimizer for parameter updates
        self.state: dict = {}                                                   # dictionary to hold any state information
        
        # Initialize state for adaptive methods
        method = self.config.get("method", "")
        if method in ["TD-FxLMS", "TD-FxNLMS"]:
            self.state["current_estLEM"] = None  # LEM estimate will be initialized during first frame
            self.state["last_LEM_update_sample"] = 0  # Track sample index of last LEM update
            # TODO: keep completing the state as you implement the methods

    # Dont use getters or setters! Just make sure properties are public.
    
    def update(self, in_frame: torch.Tensor, EQed_frame: torch.Tensor, out_frame: torch.Tensor, sr: int, frame_start_sample: int):
        if self.config.get("method") == "TD-FxLMS":
            self._update_TD_FxLMS(in_frame, EQed_frame, out_frame, sr, frame_start_sample)
        else:   # placeholder (no adaptation logic)
            new_params = self.params
            self.prev_params = self.params
            self.params = new_params


    def _update_TD_FxLMS(self, in_frame: torch.Tensor, EQed_frame: torch.Tensor, out_frame: torch.Tensor, sr: int, frame_start_sample: int):
        # Placeholder adaptation logic for TD-FxLMS

        # Track parameters prior to step
        self.prev_params = self.params.clone()

        # Estimate LEM if needed (robust deconvolution)
        self._estimate_and_update_LEM(EQed_frame, out_frame, sr, frame_start_sample)

        # Use LEM estimate to produce filtered-x version of input frame
        if self.state["current_estLEM"] is not None:
            estLEM = self.state["current_estLEM"]  # (1, 1, LEM_len)
            filtered_x = torchaudio.functional.fftconvolve(in_frame, estLEM, mode="same")  # (1, 1, N + LEM_len - 1)
            filtered_x = filtered_x[..., :in_frame.shape[-1]]  # Match length to input frame (1, 1, N)
        else:
            filtered_x = in_frame  # keep frame as is (1, 1, N)

        # Reestimate delay introduced by EQ and total DELAY (EQ + LEM)
        #from utils import get_delay_xcorr
        #self.est_EQ_delay = get_delay_xcorr(in_frame.squeeze().cpu().detach().numpy(), EQed_frame.squeeze().cpu().detach().numpy(), sr)
        frame_delay = self.est_LEM_delay + self.est_EQ_delay

        # Delay in_frame to build desired output frame
        desired_out_frame = torch.zeros_like(in_frame)
        in_len = in_frame.shape[-1]
        if frame_delay > 0 and in_len - frame_delay > 0:
            desired_out_frame[..., frame_delay:in_len] =  in_frame[..., :in_len - frame_delay]
        
        # Match length of EQed_frame to out_frame if needed (only for loss evaluation)
        min_len = min(desired_out_frame.shape[-1], out_frame.shape[-1])

        # Compute Time-Domain loss
        loss = torch.nn.functional.mse_loss(desired_out_frame[..., :min_len], out_frame[..., :min_len])

        if self.logger is not None:
            # Log current loss value
            self.logger.log_loss(loss.item(), frame_start_sample)
            self.logger.log_delays(self.est_LEM_delay, self.est_EQ_delay, frame_start_sample)

        # Step the pytorch optimizer (updates self.params)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clip parameters to [0, 1] range to ensure normalized values (preserves gradient flow)
        with torch.no_grad():
            self.params.clamp_(0.0, 1.0)


    def _estimate_and_update_LEM(self, EQed_frame: torch.Tensor, out_frame: torch.Tensor, sr: int, frame_start_sample: int):
        """Estimate the LEM (Loudspeaker-Enclosure-Microphone) system response via frequency-domain division.
        
        Computes LEM response as: H(f) = FFT(out_frame) / FFT(EQed_frame)
        Stores impulse response with "approximately" estLEM_desired_length points.
        Much faster than time-domain least-squares deconvolution.
        
        Only re-estimates if enough time has passed since last estimation.
        If no previous estimate exists, estimates immediately.
        
        Args:
            EQed_frame: EQ'd input signal (1, 1, N)
            out_frame: Microphone output signal (1, 1, N)
            sr: Sample rate in Hz
            frame_start_sample: Sample index where this frame starts in the full signal
        """
        # Local import to avoid circular import issues
        from utils import octave_average_torch
        # Get configuration parameters
        estLEM_desired_length = self.config.get("estLEM_desired_length", 1024)
        estLEM_sustain_ms = self.config.get("estLEM_sustain_ms", 100)
        
        # Convert sustain time to samples
        sustain_samples = int(estLEM_sustain_ms * sr / 1000.0)
        
        # Check if we need to update the LEM estimate
        should_estimate = False
        if self.state["current_estLEM"] is None:
            # First estimation
            should_estimate = True
        elif frame_start_sample - self.state["last_LEM_update_sample"] >= sustain_samples:
            # Enough time has passed
            should_estimate = True
        
        if should_estimate:
            # Frequency-domain deconvolution via FFT division (preserving phase)
            # H(f) = FFT(out_frame) / FFT(EQed_frame)
            
            batch_size, n_channels, n_samples = EQed_frame.shape
            EQed_flat = EQed_frame.squeeze()  # (N,)
            out_flat = out_frame.squeeze()    # (M,)
            
            # Match lengths for FFT division
            min_len = min(EQed_flat.shape[0], out_flat.shape[0])
            EQed_flat = EQed_flat[:min_len]
            out_flat = out_flat[:min_len]
            
            # Compute FFTs (full complex spectrum)
            EQed_fft = torch.fft.rfft(EQed_flat)  # (min_len//2 + 1,) complex
            out_fft = torch.fft.rfft(out_flat)    # (min_len//2 + 1,) complex
            
            # Frequency-domain division with regularization (avoid division by zero)
            eps = 1e-8
            H_complex = out_fft / (EQed_fft + eps)  # Complex frequency response
            
            # Get frequency axis
            freqs = torch.fft.rfftfreq(min_len, d=1.0/sr, device=EQed_flat.device, dtype=torch.float32)
            
            # Limit amplification to ROI by zeroing out frequencies outside the region of interest
            if self.roi is not None:
                roi_mask = (freqs >= self.roi[0]) & (freqs <= self.roi[1])
                # Set H_complex to 1 (no amplification) outside ROI
                H_complex = torch.where(roi_mask, H_complex, torch.zeros_like(H_complex)+ eps)
            
            # Compute optimal bands per octave (bpo) based on desired output length
            # bpo * num_octaves ≈ estLEM_len_samples
            f_min = freqs[1].item() if len(freqs) > 1 else 1.0  # Avoid DC component
            f_max = freqs[-1].item()
            num_octaves = torch.log2(torch.tensor(f_max / f_min, device=freqs.device)).item() if f_max > f_min else 1.0
            bpo = max(1, int(torch.round(torch.tensor(estLEM_desired_length / num_octaves, device=freqs.device)).item()))  # Compute adaptive bpo
            
            # Apply octave averaging to smooth and reduce dimensionality (keeps phase!)
            # Smoothing applies to full frequency range [0, fs/2]
            #H_complex_smoothed, freqs_smoothed = octave_average_torch(
            #    freqs, H_complex, bpo=bpo, freq_range=None, b_smooth=True
            #)
            
            # Store as (1, 1, estLEM_len_samples) - time-domain impulse response of estimated LEM
            estLEM_time = torch.fft.irfft(H_complex)
            self.state["current_estLEM"] = estLEM_time.unsqueeze(0).unsqueeze(0).detach()
            self.state["last_LEM_update_sample"] = frame_start_sample
            
            # Update delay estimate from the estimated LEM impulse response
            # The peak of the IR indicates the system delay
            self.est_LEM_delay = int(torch.argmax(torch.abs(estLEM_time)).item())


# TODO
class EQLogger:
    """Logger for EQ adaptation metrics. Each variable is stored as Nx2 numpy array: [value, frame_start_sample]."""
    
    def __init__(self): 
        # Each array has shape (N, 2): column 0 = value, column 1 = frame_start_sample
        self.loss_log = np.empty((0, 2), dtype=np.float64)
        self.LEM_delay_log = np.empty((0, 2), dtype=np.float64)
        self.EQ_delay_log = np.empty((0, 2), dtype=np.float64)
        self.params_log = []  # List of (params_array, frame_start_sample) tuples for variable-length params

    def log_loss(self, loss: float, frame_start_sample: int):
        # Check if this frame already exists
        existing_idx = np.where(self.loss_log[:, 1] == frame_start_sample)[0]
        if len(existing_idx) > 0:
            self.loss_log[existing_idx[0], 0] = loss
        else:
            self.loss_log = np.vstack([self.loss_log, [loss, frame_start_sample]])

    def log_delays(self, LEM_delay: int, EQ_delay: int, frame_start_sample: int):
        # Log LEM delay
        existing_idx = np.where(self.LEM_delay_log[:, 1] == frame_start_sample)[0]
        if len(existing_idx) > 0:
            self.LEM_delay_log[existing_idx[0], 0] = LEM_delay
        else:
            self.LEM_delay_log = np.vstack([self.LEM_delay_log, [LEM_delay, frame_start_sample]])
        
        # Log EQ delay
        existing_idx = np.where(self.EQ_delay_log[:, 1] == frame_start_sample)[0]
        if len(existing_idx) > 0:
            self.EQ_delay_log[existing_idx[0], 0] = EQ_delay
        else:
            self.EQ_delay_log = np.vstack([self.EQ_delay_log, [EQ_delay, frame_start_sample]])

    # Convenience properties for backwards compatibility
    @property
    def frames_start_samples(self):
        """Return sorted unique frame start samples from loss log."""
        if len(self.loss_log) == 0:
            return []
        return self.loss_log[:, 1].astype(int).tolist()
    
    @property
    def loss_by_frames(self):
        """Return loss values aligned with frames_start_samples."""
        if len(self.loss_log) == 0:
            return []
        return self.loss_log[:, 0].tolist()
    
    @property
    def LEM_delay_by_frames(self):
        """Return LEM delay values."""
        if len(self.LEM_delay_log) == 0:
            return []
        return self.LEM_delay_log[:, 0].astype(int).tolist()
    
    @property
    def EQ_delay_by_frames(self):
        """Return EQ delay values."""
        if len(self.EQ_delay_log) == 0:
            return []
        return self.EQ_delay_log[:, 0].astype(int).tolist()
    



class Ridge:
    def __init__(self, alpha = 0, fit_intercept = True,):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        X = X.rename(None)
        y = y.rename(None).view(-1,1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim = 1)
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y 
        lhs = X.T @ X 
        rhs = X.T @ y
        if self.alpha == 0:
            self.w = torch.linalg.lstsq(lhs, rhs).solution
        else:
            ridge = self.alpha*torch.eye(lhs.shape[0],device=X.device)
            self.w = torch.linalg.lstsq(lhs + ridge, rhs).solution
            
    def predict(self, X: torch.tensor) -> None:
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim = 1)
        return X @ self.w