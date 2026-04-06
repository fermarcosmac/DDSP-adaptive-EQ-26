# Copyright 2020 ewan xu<ewan_xu@outlook.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

""" Frequency Domain Adaptive Filter """

import numpy as np
from scipy.signal import fftconvolve
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

def fxfdaf(x, d, h_hat, M, mu=0.05, beta=0.9, W_init: np.ndarray = None, x_state: np.ndarray = None):

  W = np.zeros(M+1,dtype=np.complex128) if W_init is None else W_init
  norm = np.full(M+1,1e-8)

  window =  np.hanning(M)
  x_old = np.zeros(M) if x_state is None else x_state

  num_block = min(len(x),len(d)) // M
  e = np.zeros(num_block*M)

  for n in range(num_block):
    x_n = np.concatenate([x_old,x[n*M:(n+1)*M]])
    x_n_f = fftconvolve(x_n, h_hat, mode='full')[:2*M] # Causal filtered-x

    d_n = d[n*M:(n+1)*M]
    x_old = x[n*M:(n+1)*M]

    X_n_f = fft(x_n_f)
    y_n = ifft(W*X_n_f)[M:]
    e_n = d_n-y_n
    e[n*M:(n+1)*M] = e_n

    e_fft = np.concatenate([np.zeros(M),e_n*window])
    E_n = fft(e_fft)

    # Stable normalized FDAF update.
    norm = beta*norm + (1-beta)*np.abs(X_n_f)**2
    G = mu*E_n/(norm+1e-3)
    W = W + X_n_f.conj()*G

    w = ifft(W)
    w[M:] = 0
    W = fft(w)

  return e, W, x_old