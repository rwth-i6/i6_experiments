import copy
import os
import numpy as np

from torch import nn

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.returnn.oggzip import BlissToOggZipJob
import i6_experiments.users.engler.fairseq.training as runFairseq
from returnn_common.datasets import OggZipDataset
from i6_experiments.common.setups.returnn_common.serialization import (
    Collection,
    Import
)

def prepare_hdf_dataset(hdf_files, partition_epoch, num_workers=1, buffer_size=100):
    dataset = {
            'class': "HDFDataset",
            'files': hdf_files,
            'partition_epoch': partition_epoch,
            "seq_ordering": "random"
            }
    if num_workers == 1:
        return dataset
    else:
        multi_proc_dataset = {
                "class": "MultiProcDataset",
                "dataset": dataset,
                "num_workers": num_workers,
                "buffer_size": buffer_size,
                }
        return multi_proc_dataset


def prepare_zip_dataset(bliss_corpus, segment_file, returnn_python_exe, returnn_root):
    ogg_zip_job = BlissToOggZipJob(
            bliss_corpus,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            )
    tk.register_output("ogg_corpora", ogg_zip_job.out_ogg_zip)

    zip_dataset = OggZipDatase(
            files = [ogg_zip_job.out_ogg_zip],
            segment_file=segment_file,
            partition_epoch=2,
            seq_ordering="laplace:.1000"
            )
    return zip_dataset

def returnn_config_get_extern_data():
    from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim
    time_data_dim = SpatialDim('time:data')
    extern_data = {'data': {
        'dim_tags': [
            batch_dim,
            time_data_dim,
            FeatureDim('feat', 1),
        ],
        'dtype': 'float32',
        }}
    return extern_data


def get_returnn_configs_pytorch(data_train, data_dev, wav2vec2_args, base_config_args):
    wav2vec2_base_config = {}
    wav2vec2_base_config["train"] = data_train
    wav2vec2_base_config["dev"] = data_dev
    wav2vec2_base_config["wav2vec2_args"] = wav2vec2_args
    wav2vec2_base_config["extern_data"] = CodeWrapper("returnn_config_get_extern_data()")

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
            "batch_size": 2080000, # raw audio input with 16kHz
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "learning_rate_file": "learning_rates",
            "learning_rates": list(np.linspace(2.5e-5, 2.5e-4, 10)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.707,
            "newbob_multi_num_epochs": 40,
            "newbob_multi_update_interval": 1,
        }
    )

    wav2vec2_base_config.update(base_config_args)
    
    returnn_config = ReturnnConfig(
        config=wav2vec2_base_config,
        post_config=base_post_config,
        pprint_kwargs={"sort_dicts": False},
        python_prolog = [returnn_config_get_extern_data,
                         "import fairseq.models.wav2vec as wav2vec",
                         "import torch",
                         "from torch import nn",
                         "import torch.nn.functional as F"],
        python_epilog=[
            "def get_model(**_kwargs):\n"
            "    fairseq_config = wav2vec.Wav2Vec2Config(**wav2vec2_args)\n"
            "    model =  wav2vec.Wav2Vec2Model(fairseq_config)\n"
            "    return model\n",

            "def train_step(*, model:wav2vec.Wav2Vec2Model, data, train_ctx, **_kwargs):\n"
            "   waveforms = torch.squeeze(data[\"data\"], dim=-1)\n"
            "   waveforms = waveforms.type(torch.float32)\n"
            "   net_output = model(waveforms)\n"
            "   logits = model.get_logits(net_output).float()\n"
            "   x = net_output[\"x\"]\n"
            "   target = x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)\n"
            "   loss = F.cross_entropy(logits, target, reduction=\"sum\")\n"
            "   train_ctx.mark_as_loss(name=\"ce\", loss=loss)\n"
            "   extra_losses = model.get_extra_losses(net_output)\n"
            "   sample_size = model.get_targets(data, net_output).numel()\n"
            "   prob_perplexity_loss = extra_losses[0] * sample_size\n"
            "   features_pen_loss = extra_losses[1] * sample_size\n"
            "   train_ctx.mark_as_loss(name=\"prob_perplexity\", loss=prob_perplexity_loss, scale=loss_weights[0])\n"
            "   train_ctx.mark_as_loss(name=\"features_pen\", loss=features_pen_loss, scale=loss_weights[1])\n"],
    )

    return {
        "wav2vec_base": returnn_config,
    }
