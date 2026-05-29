"""
Perplexity evaluation for the phoneme Transformer LM.

Runs a `ReturnnForwardJobV2` with the LM model + ppl_forward_v1 module against
the phoneme-tokenized dev/cv LmDataset and writes a JSON `ppl.json` (per
checkpoint, per dev set).

Uses the standard RETURNN at `recipe/returnn` (RETURNN_ROOT) and the same
config conventions as the AM baseline.
"""

import copy
import os
from typing import Dict, Optional

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2

from ...config import get_forward_config
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...pipeline import NeuralLM


def _run_ppl(
    *,
    prefix_name: str,
    lm_model: NeuralLM,
    label_datastream_opts: Dict,
    dataset_returnn_opts: Dict,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    forward_config_overrides: Optional[Dict] = None,
):
    forward_config = {
        "behavior_version": 21,
        "extern_data": {
            "data": {**label_datastream_opts, "available_for_inference": True},
            "delayed": {**label_datastream_opts, "available_for_inference": True},
        },
        "batch_size": 4000,
        "max_seqs": 64,
        "torch_amp": {"dtype": "bfloat16"},
        "torch_dataloader_opts": {"num_workers": 1},
    }
    if forward_config_overrides:
        forward_config.update(forward_config_overrides)

    returnn_config: ReturnnConfig = get_forward_config(
        network_module=lm_model.network_module,
        config=forward_config,
        net_args=lm_model.net_args,
        decoder="lm.trafo.ppl_forward_v1",
        decoder_args={},
        debug=False,
    )
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["forward"] = dataset_returnn_opts

    ppl_job = ReturnnForwardJobV2(
        model_checkpoint=lm_model.checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=16,
        time_rqmt=4,
        device="gpu",
        cpu_rqmt=2,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["ppl.json"],
    )
    ppl_job.add_alias(prefix_name + "/ppl_job")
    tk.register_output(prefix_name + "/ppl.json", ppl_job.out_files["ppl.json"])
    return ppl_job


def evaluate_phon_lm_ppl(prefix_name: str, lm_model: NeuralLM, cv_dataset, label_datastream):
    """
    Run PPL on the held-out CV dataset that was used during LM training.
    """
    label_datastream_opts = label_datastream.as_returnn_extern_data_opts(available_for_inference=True)
    _run_ppl(
        prefix_name=os.path.join(prefix_name, "cv"),
        lm_model=lm_model,
        label_datastream_opts=label_datastream_opts,
        dataset_returnn_opts=cv_dataset.as_returnn_opts(),
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    )
