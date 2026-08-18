import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, get_keep_epochs, test_data_dict, base_num_epochs
from sisyphus import tk

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering=None,
)

train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)

base_config = copy.deepcopy(base_config)
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_mlm.train_step"


def py():
    print("Calling GPU test py()")
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    nodes = [227, 230, 231, 234, 238, 240, 241, 242, 243, 245, 246, 247, 248, 251, 257, 258, 263, 264, 265, 267, 268, 269, 271, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 285, 286]

    ablations = []
    for node in nodes:
        for amp_dtype in ["float16", "bfloat16"]:
            ablations.append(
                (
                    f"test_node_{node}_{amp_dtype}",
                    {
                        "num_enc_layers": 3,
                        "num_text_dec_layers": 3,
                        "num_audio_dec_layers": 3,
                    },
                    {
                        "codebook_diversity_loss_scale": 0.0,
                        "mlm_pretrain_steps": 10000,
                    },
                    {
                        "batch_size": 4000,
                        "torch_amp": amp_dtype,
                    },
                    node
                )
            )

    jobs = []
    for train_name, model_args, train_args, training_args, node in ablations:
        config = copy.deepcopy(base_config)
        
        dict_update_deep(config["model_args"], model_args)
        dict_update_deep(config["train_args"], train_args)
        dict_update_deep(config["training"], training_args)
        
        config["training"]["__dummy_hash"] = f"{train_name}_v2"
        
        train_job = run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=True,
        )
        train_job.rqmt["sbatch_args"] = ["-w", f"cn-{node}", "--time=00:06:00"]
    
    print(f"Finished defining GPU test jobs")
