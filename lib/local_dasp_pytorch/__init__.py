# Expose local submodules and common symbols for convenient imports.
from . import functional, modules, signal

# Re-export commonly used callables and classes
from .functional import (
    gain,
    stereo_bus,
    stereo_panner,
    stereo_widener,
    noise_shaped_reverberation,
    compressor,
    distortion,
    parametric_eq,
)

from .modules import (
    Processor,
    Compressor,
    ParametricEQ,
    NoiseShapedReverb,
    Gain,
    Distortion,
)

__all__ = [
    'functional', 'modules', 'signal',
    'gain', 'stereo_bus', 'stereo_panner', 'stereo_widener',
    'noise_shaped_reverberation', 'compressor', 'distortion', 'parametric_eq',
    'Processor', 'Compressor', 'ParametricEQ', 'NoiseShapedReverb', 'Gain', 'Distortion',
]
