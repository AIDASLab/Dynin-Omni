"""Public exports for model components used by training/inference scripts."""

from __future__ import annotations

import sys as _sys

from . import lr_scheduler as _lr_scheduler
from .modeling_magvitv2 import LFQuantizer, MAGVITv2, VQGANDecoder, VQGANEncoder
from .modeling_dynin_omni import DyninOmniConfig, DyninOmniModelLM, VideoTokenMerger
from .sampling import get_mask_schedule

# Backward compatibility for legacy class names used in prior MMaDA scripts.
MMadaModelLM = DyninOmniModelLM
MMadaConfig = DyninOmniConfig

# Backward compatibility for legacy import path:
# from models.lr_schedulers import get_scheduler
_sys.modules.setdefault(f"{__name__}.lr_schedulers", _lr_scheduler)

__all__ = [
    "VQGANEncoder",
    "VQGANDecoder",
    "LFQuantizer",
    "MAGVITv2",
    "DyninOmniModelLM",
    "DyninOmniConfig",
    "MMadaModelLM",
    "MMadaConfig",
    "VideoTokenMerger",
    "get_mask_schedule",
]
