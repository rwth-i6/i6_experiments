import copy
from dataclasses import dataclass
from enum import Enum
import pprint
from textwrap import dedent
from typing import Dict, List, Optional

from i6_core.returnn.training import Checkpoint
from i6_core.util import instanciate_delayed


class CopyParamMode(Enum):
    subset = "subset"
    ifpossible = "ifpossible"
    reset = "reset"

    def get(self):
        return self.value


@dataclass(eq=True, frozen=True)
class StagingInfo:
    stage_epochs: List
    checkpoint: Optional[Checkpoint] = None
    copy_param_mode: CopyParamMode = CopyParamMode.subset

    @classmethod
    def default(cls):
        return StagingInfo(
            stage_epochs=[1, 2],
        )

    def get_load_from_file_dict(self) -> Dict:
        assert self.checkpoint is not None, "Please set the checkpoint first"
        return {
            "preload_from_files": {
                "existing-model": {
                    "init_for_train": True,
                    "ignore_missing": True,
                    "filename": self.checkpoint,
                }
            }
        }

    def get_net_string_with_prolog(self, network: Dict, python_prolog: str = "", python_epilog: str = "") -> str:
        staged_network = copy.deepcopy(network)
        pp = pprint.PrettyPrinter(indent=2, width=150)
        staged_network["#copy_param_mode"] = self.copy_param_mode.get()
        return python_prolog + "\nnetwork = %s" % pp.pformat(instanciate_delayed(staged_network)) + python_epilog
