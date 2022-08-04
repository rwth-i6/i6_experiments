__all__ = ["GammatoneOptions"]

from dataclasses import dataclass
from typing import Any, Dict, Optional

import i6_core.features as features
import i6_core.rasr as rasr


@dataclass()
class GammatoneOptions:
    minfreq: int = 100
    maxfreq: int = 7500
    channels: int = 68
    warp_freqbreak: Optional[int] = None
    tempint_type: str = "hanning"
    tempint_shift: float = 0.01
    tempint_length: float = 0.025
    flush_before_gap: bool = True
    do_specint: bool = True
    specint_type: str = "hanning"
    specint_shift: int = 4
    specint_length: int = 9
    normalize: bool = True
    preemphasis: bool = True
    legacy_scaling: bool = False
    without_samples: bool = False
    samples_options: Optional[Dict[str, Any]] = None
    normalization_options: Optional[Dict[str, Any]] = None
    add_features_output: bool = False

    def get_flow(self) -> rasr.FlowNetwork:
        return features.gammatone_flow(
            minfreq=self.minfreq,
            maxfreq=self.maxfreq,
            channels=self.channels,
            warp_freqbreak=self.warp_freqbreak,
            tempint_type=self.tempint_type,
            tempint_shift=self.tempint_shift,
            tempint_length=self.tempint_length,
            flush_before_gap=self.flush_before_gap,
            do_specint=self.do_specint,
            specint_type=self.specint_type,
            specint_shift=self.specint_shift,
            specint_length=self.specint_length,
            normalize=self.normalize,
            preemphasis=self.preemphasis,
            legacy_scaling=self.legacy_scaling,
            without_samples=self.without_samples,
            samples_options=self.samples_options,
            normalization_options=self.normalization_options,
            add_features_output=self.add_features_output,
        )
