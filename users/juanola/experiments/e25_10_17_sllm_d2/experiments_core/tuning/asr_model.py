from dataclasses import dataclass
from typing import Any, Dict, Optional

from sisyphus import tk

from i6_core.returnn import PtCheckpoint


@dataclass
class ASRModel:
    """
    Contains information about how to call a trained model.
    """
    checkpoint: PtCheckpoint
    net_args: Dict[str, Any]
    network_module: str
    prior_file: Optional[tk.Path]
    prefix_name: Optional[str]
