"""Portable dynamic mode decomposition. See docs for numerical conventions."""
from .core import DMD, FbDMD, RandomizedDMD, TLSqDMD
from .variants import DMDc, HankelDMD, RobustDMD, delay_embedding
from .streaming import StreamingDMD

__version__ = "0.1.2"
__all__ = ["DMD", "TLSqDMD", "FbDMD", "RandomizedDMD", "HankelDMD", "DMDc",
           "RobustDMD", "StreamingDMD", "delay_embedding"]
