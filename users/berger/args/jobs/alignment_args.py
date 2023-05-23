from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from typing import Any, Union, Optional, Dict, List


def get_alignment_args(search_type: SearchTypes, **kwargs) -> Dict:
    return {SearchTypes.GenericSeq2SeqSearch: get_label_alignment_args,}.get(
        search_type, lambda **_: {}
    )(**kwargs)


def get_label_alignment_args(
    *,
    epochs: Optional[List[int]] = None,
    prior_scales: Union[float, List[float]] = 0.3,
    add_eow: bool = True,
    add_sow: bool = False,
    allow_blank: bool = True,
    allow_loop: bool = False,
    label_unit: str = "phoneme",
    label_scorer_type: str = "precomputed-log-posterior",
    label_scorer_args: Dict = {},
    use_gpu: bool = False,
) -> Dict[str, Any]:

    if isinstance(prior_scales, float):
        prior_scales = [prior_scales]
    if epochs is None:
        epochs = []

    return {
        "epochs": epochs,
        "prior_scales": prior_scales,
        "alignment_options": {
            "label-pruning": 50,
            "label-pruning-limit": 100000,
        },
        "align_node_options": {
            "allow-label-loop": allow_loop,
        },
        "label_unit": label_unit,
        "label_file_blank": allow_blank,
        "add_eow": add_eow,
        "add_sow": add_sow,
        "label_scorer_type": label_scorer_type,
        "label_scorer_args": label_scorer_args,
        "use_gpu": use_gpu,
        "rtf": 5,
    }
