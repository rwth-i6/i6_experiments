from dataclasses import dataclass
from typing import Any, Dict, Optional

from sisyphus import tk


@dataclass
class ASRModel:
    checkpoint: tk.Path
    net_args: Dict[str, Any]
    network_module: str
    prior_file: Optional[tk.Path]
    prefix_name: Optional[str]
