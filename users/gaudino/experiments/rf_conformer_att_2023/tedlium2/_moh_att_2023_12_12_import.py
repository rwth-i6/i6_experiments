from __future__ import annotations

from typing import Dict

import os
import sys
import numpy

from sisyphus import tk

from i6_core.returnn.training import Checkpoint
from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim, TensorDict

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import Model, MakeModel, from_scratch_training, model_recog
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ctc_greedy import model_recog_ctc
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_time_sync import model_recog_time_sync
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output

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


_ParamMapping = {}  # type: Dict[str,str]


def _get_tf_checkpoint_path() -> tk.Path:
    """
    :return: Sisyphus tk.Path to the checkpoint file
    """
    return generic_job_output(_returnn_tf_ckpt_filename)


def _get_pt_checkpoint_path() -> tk.Path:
    old_tf_ckpt_path = _get_tf_checkpoint_path()
    old_tf_ckpt = Checkpoint(index_path=old_tf_ckpt_path)
    make_model_func = MakeModel(80, 10_025, eos_label=0, num_enc_layers=12)
    # TODO: problems with hash:
    #  make_model_func, map_func: uses full module name (including "zeyer"), should use sth like unhashed_package_root
    #  https://github.com/rwth-i6/sisyphus/issues/144
    converter = ConvertTfCheckpointToRfPtJob(
        checkpoint=old_tf_ckpt,
        make_model_func=make_model_func,
        map_func=map_param_func_v3,
        epoch=1,
        step=0,
    )
    # converter.run()
    return converter.out_checkpoint


def test_import_search():
    from returnn.util import debug

    debug.install_lib_sig_segfault()
    debug.install_native_signal_handler()
    debug.init_faulthandler()

    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=80, kind=Dim.Types.Feature)
    time_dim = Dim(
        name="time",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("time_size", dims=[batch_dim], dtype="int32"),
    )
    target_dim = Dim(name="target", dimension=1_057, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels([str(i) for i in range(target_dim.dimension)], eos_label=0)
    data = Tensor("data", dim_tags=[batch_dim, time_dim, Dim(1, name="dummy-feature")], feature_dim_axis=-1)
    target_spatial_dim = Dim(
        name="target_spatial",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("target_spatial_size", dims=[batch_dim], dtype="int32"),
    )
    target = Tensor("target", dim_tags=[batch_dim, target_spatial_dim], sparse_dim=target_dim)

    num_layers = 12

    from returnn.config import Config

    config = Config(
        dict(
            log_verbositiy=5,
            extern_data={
                "audio_features": {"dim_tags": data.dims, "feature_dim_axis": -1},
                "bpe_labels": {"dim_tags": target.dims, "sparse_dim": target.sparse_dim},
            },
        )
    )

    # data e.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
    search_data_opts = {
        "class": "MetaDataset",
        "data_map": {
            "audio_features": ("zip_dataset", "data"),
            "bpe_labels": ("zip_dataset", "classes"),
        },
        "datasets": {
            "zip_dataset": {
                "class": "OggZipDataset",
                "path": generic_job_output(
                    "i6_core/returnn/oggzip/BlissToOggZipJob.Wgp6724p1XD2/output/out.ogg.zip"
                ).get_path(), # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.Wgp6724p1XD2/output/out.ogg.zip",
                "use_cache_manager": True,
                "audio": {
                    "features": "raw",
                    "peak_normalization": True,
                    "preemphasis": None,
                },
                "targets": {
                    "class": "BytePairEncoding",
                    "bpe_file": generic_job_output(
                        "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes"
                    ).get_path(), # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes",
                    "vocab_file": generic_job_output(
                        "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab"
                    ).get_path(), # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab",
                    "unknown_label": "<unk>",
                    "seq_postfix": [0],
                },
                "segment_file": None,
                "partition_epoch": 1,
                # "seq_ordering": "sorted_reverse",
            }
        },
        "seq_order_control_dataset": "zip_dataset",
    }

    print("*** Construct input minibatch")
    extern_data = TensorDict()
    extern_data.update(config.typed_dict["extern_data"], auto_convert=True)

    from returnn.datasets.basic import init_dataset, Batch
    from returnn.tf.data_pipeline import FeedDictDataProvider, BatchSetGenerator

    dataset = init_dataset(search_data_opts)
    dataset.init_seq_order(
        epoch=1,
        seq_list=["TED-LIUM-realease2/BlaiseAguerayArcas_2007/22"],
        # seq_list=[f"TED-LIUM-realease2/AlGore_2009/{i}" for i in range(1,34)],
    )
    batch_num_seqs = 1
    # batch_num_seqs = 10
    dataset.load_seqs(0, batch_num_seqs)
    batch = Batch()
    for seq_idx in range(batch_num_seqs):
        batch.add_sequence_as_slice(seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx))
    batches = BatchSetGenerator(dataset, generator=iter([batch]))
    data_provider = FeedDictDataProvider(
        extern_data=extern_data, data_keys=list(extern_data.data.keys()), dataset=dataset, batches=batches
    )
    batch_data = data_provider.get_next_batch()

    for key, data in extern_data.data.items():
        data.placeholder = batch_data[key]
        key_seq_lens = f"{key}_seq_lens"
        if key_seq_lens in batch_data:
            seq_lens = data.dims[1]
            if not seq_lens.dyn_size_ext:
                seq_lens.dyn_size_ext = Tensor(key_seq_lens, dims=[batch_dim], dtype="int32")
            seq_lens.dyn_size_ext.placeholder = batch_data[key_seq_lens]
    if not batch_dim.dyn_size_ext:
        batch_dim.dyn_size_ext = Tensor("batch_dim", dims=[], dtype="int32")
    batch_dim.dyn_size_ext.placeholder = numpy.array(batch_data["batch_dim"], dtype="int32")
    extern_data_numpy_raw_dict = extern_data.as_raw_tensor_dict()
    extern_data.reset_content()

    rf.select_backend_torch()

    print("*** Convert old model to new model")
    # pt_checkpoint_path = _get_pt_checkpoint_path()
    pt_checkpoint_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/baseline_w_trafo_lm_23_12_28/average.pt"
    print(pt_checkpoint_path)

    print("*** Create new model")
    ctc_only = False
    time_sync = True
    time_sync_search_args = {
        "beam_size": 12,
        "att_scale": 0.0,
        "ctc_scale": 1.0,
        "add_trafo_lm": True,
        "lm_scale": 0.5,
    }
    search_args = {
        "beam_size": 1,
        "add_trafo_lm": True,
        "att_scale": 1.0,
        "lm_scale": 1.0,
    }
    if time_sync:
        search_args = time_sync_search_args
    model_args = {
        "add_ted2_trafo_lm": True,
        "target_embed_dim": 256,
    }
    # returnn/torch/fonrtend/_backend 1635: sizes_raw = torch.reshape(sizes.raw_tensor, [batch_dim]).to('cpu')
    new_model = MakeModel.make_model(in_dim, target_dim, num_enc_layers=num_layers, search_args=search_args, model_args=model_args)

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
    extern_data["audio_features"].raw_tensor = extern_data["audio_features"].raw_tensor.to(cuda)

    print("*** Search ...")

    with torch.no_grad():
        with rf.set_default_device_ctx("cuda"):
            # manually switch between different decoders
            if ctc_only:
                seq_targets, seq_log_prob, out_spatial_dim, beam_dim = model_recog_ctc(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                )
            elif time_sync:
                seq_targets, seq_log_prob, out_spatial_dim, beam_dim = model_recog_time_sync(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                )
            else:
                seq_targets, seq_log_prob, out_spatial_dim, beam_dim = model_recog(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                )
    print(seq_targets, seq_targets.raw_tensor) # seq_targets [T,Batch,Beam]
    print("Out spatial dim:", out_spatial_dim)


    # serialize output
    # vocab_1 = Vocabulary("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab", eos_label=0) # librispeech
    # does not work due to encoding of environment
    # vocab_1 = Vocabulary("/u/michel/setups/2023-08-09--librispeech_mohammad/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab", eos_label=0) # tedlium2
    vocab_1 = dataset.datasets['zip_dataset'].targets
    for batch_idx in range(batch_dim.get_dim_value()):
        # process seq
        hyps = seq_targets.raw_tensor[:, batch_idx, :]
        scores = seq_log_prob.raw_tensor[batch_idx, :]
        hyps_len = seq_targets.dims[0].dyn_size_ext.raw_tensor[:, batch_idx]
        num_beam = hyps.shape[1]
        only_max = True
        if only_max:
            max_score_idx = torch.argmax(scores)
            score = float(scores[max_score_idx])
            hyp_ids = hyps[: hyps_len[max_score_idx], max_score_idx]
            hyp_serialized = vocab_1.get_seq_labels(hyp_ids)
            print(f"  ({score!r}, {hyp_serialized!r}),\n")
            continue
        for i in range(num_beam):
            score = float(scores[i])
            hyp_ids = hyps[: hyps_len[i], i]
            hyp_serialized = vocab_1.get_seq_labels(hyp_ids)
            print(f"  ({score!r}, {hyp_serialized!r}),\n")



# `py` is the default sis config function name. so when running this directly, run the import test.
# So you can just run:
# `sis m recipe/i6_experiments/users/zeyer/experiments/....py`
py = test_import_search

if __name__ == "__main__":
    mod_name = __package__
    # if mod_name.startswith("recipe."):
    #     mod_name = mod_name[len("recipe.") :]
    # mod_name += "." + os.path.basename(__file__)[: -len(".py")]
    test_import_search()
