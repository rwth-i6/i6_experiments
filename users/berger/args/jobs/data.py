from typing import Optional, Union, Dict
from sisyphus import tk

from i6_core.corpus import SegmentCorpusJob
from i6_core.util import MultiPath
from i6_core.rasr import FlagDependentFlowAttribute, FlowNetwork

from i6_experiments.common.setups.rasr.util import RasrDataInput, ReturnnRasrDataInput
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_am_config_args


def get_returnn_rasr_data_input(
    rasr_data: RasrDataInput,
    name: str = "init",
    segment_path: Optional[str] = None,
    am_args: Optional[Dict] = None,
    allophone_file: Optional[tk.Path] = None,
    concurrent: Optional[int] = None,
    **kwargs
):

    data_input = ReturnnRasrDataInput(name=name, **kwargs)

    if concurrent is None:
        concurrent = rasr_data.concurrent

    if segment_path is None:
        segment_path = SegmentCorpusJob(
            rasr_data.corpus_object.corpus_file, concurrent
        ).out_segment_path

    if am_args is None:
        am_args = {}

    data_input.get_crp(
        am_args=get_am_config_args(am_args),
        corpus_object=rasr_data.corpus_object,
        concurrent=concurrent,
        segment_path=segment_path,
        lexicon_args=rasr_data.lexicon,
        lm_args=rasr_data.lm,
        allophone_file=allophone_file,
    )

    return data_input
