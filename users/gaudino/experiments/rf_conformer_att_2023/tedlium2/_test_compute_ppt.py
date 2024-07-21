from __future__ import annotations

from typing import Dict

import numpy

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim, TensorDict

from i6_experiments.users.gaudino.models.asr.rf.trafo_lm.lm_import_2023_11_09 import (
    MakeModel,
)
from i6_experiments.users.gaudino.models.asr.rf.trafo_lm.model_forward_lm import (
    model_forward_lm,
)

from i6_experiments.users.gaudino.datasets.search_data_opts import (
    search_data_opts_ted2,
    search_data_opts_librispeech960,
)


# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config"
# # E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
_load_existing_ckpt_in_test = True
_lm_path = "/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/re_i128_m2048_m2048_m2048_m2048.sgd_b32_lr0_cl2.newbobabs.d0.0.1350/bk-net-model/network.035"

ted2_trafo_lm_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_only_24_02_05/network.020.pt"
librispeech960_trafo_lm_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt"

_ParamMapping = {}  # type: Dict[str,str]


def test_compute_perplexity():
    from returnn.util import debug

    debug.install_lib_sig_segfault()
    debug.install_native_signal_handler()
    debug.init_faulthandler()

    from returnn.datasets.util.vocabulary import Vocabulary

    dataset = "tedlium2"
    # dataset = "librispeech960"

    if dataset == "tedlium2":
        target_dimension = 1057
        search_data_opts = search_data_opts_ted2
        seq_list = [f"TED-LIUM-realease2/AlGore_2009/{i}" for i in range(1, 34)]
        model_args = {
            "ff_activation": "gelu",
            "use_pos_enc": False,
        }
        pt_checkpoint_path = ted2_trafo_lm_path
    elif dataset == "librispeech960":
        target_dimension = 10025
        search_data_opts = search_data_opts_librispeech960
        seq_list = [
            f"dev-other/116-288045-{i:04d}/116-288045-{i:04d}" for i in range(33)
        ]
        model_args = {
            "num_layers": 24,
            "layer_out_dim": 1024,
            "att_num_heads": 8,
        }
        pt_checkpoint_path = librispeech960_trafo_lm_path

    in_dim = Dim(name="in", dimension=target_dimension, kind=Dim.Types.Feature)
    time_dim = Dim(
        name="time",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("time_size", dims=[batch_dim], dtype="int32"),
    )
    target_dim = Dim(name="target", dimension=target_dimension, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
        [str(i) for i in range(target_dim.dimension)], eos_label=0
    )
    data = Tensor(
        "data",
        dim_tags=[batch_dim, time_dim, Dim(1, name="dummy-feature")],
        feature_dim_axis=-1,
    )
    target_spatial_dim = Dim(
        name="target_spatial",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("target_spatial_size", dims=[batch_dim], dtype="int32"),
    )
    target = Tensor(
        "target", dim_tags=[batch_dim, target_spatial_dim], sparse_dim=target_dim
    )

    num_layers = 12

    from returnn.config import Config

    config = Config(
        dict(
            log_verbositiy=5,
            extern_data={
                "audio_features": {"dim_tags": data.dims, "feature_dim_axis": -1},
                "bpe_labels": {
                    "dim_tags": target.dims,
                    "sparse_dim": target.sparse_dim,
                },
            },
        )
    )

    print("*** Construct input minibatch")
    extern_data = TensorDict()
    extern_data.update(config.typed_dict["extern_data"], auto_convert=True)

    from returnn.datasets.basic import init_dataset, Batch
    from returnn.tf.data_pipeline import FeedDictDataProvider, BatchSetGenerator

    dataset = init_dataset(search_data_opts)
    dataset.init_seq_order(epoch=1, seq_list=seq_list)
    # batch_num_seqs = 2
    batch_num_seqs = 10
    dataset.load_seqs(0, batch_num_seqs)
    batch = Batch()
    for seq_idx in range(batch_num_seqs):
        batch.add_sequence_as_slice(
            seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx)
        )
    batches = BatchSetGenerator(dataset, generator=iter([batch]))
    data_provider = FeedDictDataProvider(
        extern_data=extern_data,
        data_keys=list(extern_data.data.keys()),
        dataset=dataset,
        batches=batches,
    )
    batch_data = data_provider.get_next_batch()

    for key, data in extern_data.data.items():
        data.placeholder = batch_data[key]
        key_seq_lens = f"{key}_seq_lens"
        if key_seq_lens in batch_data:
            seq_lens = data.dims[1]
            if not seq_lens.dyn_size_ext:
                seq_lens.dyn_size_ext = Tensor(
                    key_seq_lens, dims=[batch_dim], dtype="int32"
                )
            seq_lens.dyn_size_ext.placeholder = batch_data[key_seq_lens]
    if not batch_dim.dyn_size_ext:
        batch_dim.dyn_size_ext = Tensor("batch_dim", dims=[], dtype="int32")
    batch_dim.dyn_size_ext.placeholder = numpy.array(
        batch_data["batch_dim"], dtype="int32"
    )
    extern_data_numpy_raw_dict = extern_data.as_raw_tensor_dict()
    extern_data.reset_content()

    rf.select_backend_torch()

    print("*** Model path")
    print(pt_checkpoint_path)

    print("*** Create new model")

    # returnn/torch/fonrtend/_backend 1635: sizes_raw = torch.reshape(sizes.raw_tensor, [batch_dim]).to('cpu')
    new_model = MakeModel.make_model(target_dim, target_dim, model_args=model_args)

    from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_

    rf.init_train_step_run_ctx(train_flag=False, step=0)
    extern_data.reset_content()
    extern_data.assign_from_raw_tensor_dict_(extern_data_numpy_raw_dict)
    tensor_dict_numpy_to_torch_(extern_data)

    import torch
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    print("*** Load new model params from disk")
    pt_module = rf_module_to_pt_module(new_model)
    checkpoint_state = torch.load(pt_checkpoint_path)
    # checkpoint_state = torch.load(pt_checkpoint_path.get_path())
    pt_module.load_state_dict(checkpoint_state["model"])

    cuda = torch.device("cuda")
    pt_module.to(cuda)
    pt_module.eval()
    extern_data["audio_features"].raw_tensor = extern_data[
        "audio_features"
    ].raw_tensor.to(cuda)

    print("*** Search ...")

    with torch.no_grad():
        with rf.set_default_device_ctx("cuda"):
            seq_targets, seq_log_prob = model_forward_lm(
                model=new_model,
                data=extern_data["audio_features"],
                data_spatial_dim=time_dim,
                ground_truth=extern_data["bpe_labels"],
            )
    print(seq_targets, seq_targets.raw_tensor)  # seq_targets [T,Batch,Beam]
    print(
        "Out spatial dim:",
        seq_targets.dims[1],
        seq_targets.dims[1].get_size_tensor().raw_tensor,
    )

    # Compute batch perplexity
    print("*** Compute perplexity")
    N = rf.reduce_sum(seq_targets.dims[1].get_size_tensor(), axis=batch_dim)
    log_ppt = -1 / N * rf.reduce_sum(seq_log_prob, axis=seq_log_prob.dims[0])
    ppt = rf.exp(log_ppt)
    print(ppt, ppt.raw_tensor)


# `py` is the default sis config function name. so when running this directly, run the import test.
# So you can just run:
# `sis m recipe/i6_experiments/users/zeyer/experiments/....py`
py = test_compute_perplexity

if __name__ == "__main__":
    mod_name = __package__
    # if mod_name.startswith("recipe."):
    #     mod_name = mod_name[len("recipe.") :]
    # mod_name += "." + os.path.basename(__file__)[: -len(".py")]
    test_compute_perplexity()
