import copy
from typing import Dict


def delete_right_context_branch(network: Dict):
    for k in ["currentState", "linear1-triphone", "linear2-triphone", "right-output"]:
        network.pop(k, None)
    return network