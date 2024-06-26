from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_experiments.users.jxu.experiments.fairseq_multilingual.finetuning.util import (
    cache_hdf_files_locally,
)
from i6_experiments.common.setups.serialization import Import, ExplicitHash
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    PyTorchModel,
    Collection,
)

from ..default_tools import PACKAGE, FAIRSEQ
 

def get_returnn_finetune_configs_pytorch(
    data_train, data_dev, model_kwargs, base_config_args, model_type="wav2vec2", recog=False, forward=False,
):
    wav2vec2_base_config = {}
    wav2vec2_base_config["train"] = data_train
    wav2vec2_base_config["dev"] = data_dev
    wav2vec2_base_config["extern_data"] = {
        "classes": {"dim": 9001, "dtype": "int16", "sparse": True},
        "data": {"dim": 1, "dtype": "int16"},
    }

    base_post_config = {
        "backend": "torch",
        "use_tensorflow": False,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "cache_size": "0",
    }

    wav2vec2_base_config.update(
        {
            "batching": "random",
            "batch_size": 2080000,  # raw audio input with 16kHz
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
        }
    )

    wav2vec2_base_config.update(base_config_args)

    pytorch_model_import = Import(
        PACKAGE + ".pytorch_networks.%s.Wav2Vec2HybridModel" % model_type
    )
    pytorch_train_step = Import(
        PACKAGE + ".pytorch_networks.%s.train_step" % model_type
    )

    pytorch_model = PyTorchModel(
        model_class_name=pytorch_model_import.object_name,
        model_kwargs=model_kwargs,
    )
    serializer_objects = [
        pytorch_model_import,
        pytorch_train_step,
        pytorch_model,
    ]

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=True,
        packages={
            PACKAGE
        },
    )

    python_prolog = [
            "import sys",
            f"sys.path.insert(0, '{FAIRSEQ}')",
            "import torch",
            "from torch import nn",
            "import returnn.frontend as rf",
            "import torch.nn.functional as F",
            cache_hdf_files_locally,
        ]
    
    python_epilog = [serializer]
    if recog:
        pytorch_export_model = Import(
            PACKAGE + ".pytorch_networks.%s.export" % model_type
        )
        serializer_objects.append(pytorch_export_model)
        wav2vec2_base_config["model_outputs"] = {"output": {"dim": 9001},}
    elif forward:
        pytorch_forward_step = Import(
            PACKAGE + ".pytorch_networks.%s.forward_step" % model_type
        )
        serializer_objects.append(pytorch_forward_step)
        pytorch_forward_callback = Import(PACKAGE+f".util.ComputePriorCallback", import_as="forward_callback")
        serializer_objects.append(pytorch_forward_callback)
        python_prolog.append(f"sys.path.insert(0, \"/nas/models/asr/jxu/setups/2023-07-20--wav2vev2-fine-tune/recipe\")")
        python_prolog.append(f"sys.path.insert(0, \"/nas/models/asr/jxu/setups/2023-07-20--wav2vev2-fine-tune/tools/sisyphus\")")
    else:
        python_prolog.append("torch_distributed = {\"class\": torch.nn.parallel.DistributedDataParallel,\"options\": {\"find_unused_parameters\":True}}")
        python_epilog += ['cache_hdf_files_locally(dev["datasets"]["alignments"])\n'
                          'cache_hdf_files_locally(dev["datasets"]["features"])\n'
                          'cache_hdf_files_locally(train["datasets"]["alignments"])\n'
                          'cache_hdf_files_locally(train["datasets"]["features"])\n']
    returnn_config = ReturnnConfig(
        config=wav2vec2_base_config,
        post_config=base_post_config,
        pprint_kwargs={"sort_dicts": False},
        python_prolog=python_prolog,
        python_epilog=python_epilog,
    )

    return {
        "wav2vec_base": returnn_config,
    }
