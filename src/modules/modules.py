"""
Custom PyTorch module definitions used throughout the DDSP-ARE experiments.

LEMConv: Differentiable convolution with the linear echo model (LEM), using the
         true impulse response in the forward pass and an estimated one for gradients.
Ridge:   Closed-form ridge regression solver built on torch.linalg.lstsq.
"""

import torch
import torchaudio


class LEMConv(torch.autograd.Function):
    """Differentiable LEM convolution with separate forward and gradient paths.

    Forward pass uses the true LEM impulse response (h_true) while the
    backward pass uses an estimated LEM impulse response (h_est). This
    allows gradient-based adaptation of the equalizer even when the true
    room response is not perfectly known.

    generate_vmap_rule = True enables functorch's vmap over this function.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x, h_true, h_est):
        """Convolve input with the true LEM impulse response.

        Args:
            x:      (B, 1, N) input signal segment.
            h_true: (1, 1, M) true LEM IR; not differentiated.
            h_est:  (1, 1, M) estimated LEM IR; used only for gradients.

        Returns:
            y: (B, 1, N+M-1) full convolution output.
        """
        return torchaudio.functional.fftconvolve(x, h_true, mode="full")

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors and metadata needed for backward/jvp."""
        x, h_true, h_est = inputs
        ctx.save_for_backward(h_est)
        ctx.input_len = x.shape[-1]
        ctx.h_est_len = h_est.shape[-1]
        ctx.h_true = h_true
        ctx.x_shape = x.shape

    @staticmethod
    def backward(ctx, grad_output):
        """Backpropagate through estimated LEM via FFT correlation.

        Args:
            grad_output: gradient of the loss w.r.t. the convolution output.

        Returns:
            Gradients for (x, h_true, h_est); only x receives a gradient.
        """
        (h_est,) = ctx.saved_tensors
        N = ctx.input_len
        M = ctx.h_est_len

        grad_full = torchaudio.functional.fftconvolve(
            grad_output, torch.flip(h_est, dims=[-1]), mode="full"
        )
        grad_x = grad_full[..., M - 1 : M - 1 + N]
        return grad_x, None, None

    @staticmethod
    def jvp(ctx, x_t, h_true_t, h_est_t):
        """Forward-mode Jacobian-vector product (for functorch jacfwd compatibility).

        Only x is treated as differentiable; h_true and h_est are constants.
        """
        h_true = ctx.h_true
        if x_t is None:
            x_t = torch.zeros(ctx.x_shape, device=h_true.device, dtype=h_true.dtype)
        return torchaudio.functional.fftconvolve(x_t, h_true, mode="full")


class Ridge:
    """Closed-form ridge (L2-regularized) regression using normal equations.

    Solves  argmin_w ||Xw - y||^2 + alpha * ||w||^2  via the equation
    (X^T X + alpha I) w = X^T y, using torch.linalg.lstsq for numerical
    stability.

    Args:
        alpha: L2 regularization strength (0 = ordinary least squares).
        fit_intercept: If True, prepends a column of ones to X before fitting.
    """

    def __init__(self, alpha: float = 0.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.w: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the ridge regressor on (X, y).

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target tensor of shape (n_samples,) or (n_samples, 1).
        """
        X = X.rename(None)
        y = y.rename(None).view(-1, 1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows must match."

        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)

        lhs = X.T @ X
        rhs = X.T @ y

        if self.alpha == 0:
            self.w = torch.linalg.lstsq(lhs, rhs).solution
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0], device=X.device)
            self.w = torch.linalg.lstsq(lhs + ridge, rhs).solution

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict targets for X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples, 1).
        """
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)
        return X @ self.w
