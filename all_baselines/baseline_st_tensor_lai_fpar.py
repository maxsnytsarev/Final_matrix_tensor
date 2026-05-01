from __future__ import annotations

from dataclasses import dataclass

from all_baselines.baseline_st_tensor_ndvi import STTensorNDVI


@dataclass
class STTensorLAIFPAR(STTensorNDVI):
    """ST-Tensor-style baseline for Sensor-Independent LAI/FPAR CDR.

    The algorithmic core is shared with `STTensorNDVI`: low-rank iterative
    reconstruction + seasonal projection + temporal smoothing, with observed
    entries clamped at every iteration.
    """

