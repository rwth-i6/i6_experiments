from __future__ import annotations

from typing import Dict

import numpy
import math

from sisyphus import tk

from i6_core.returnn.training import Checkpoint
from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
    ConvertTfCheckpointToRfPtJob,
)

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim, TensorDict

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import (
    MakeModel,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_forward_ctc_sum import (
    model_forward_ctc_sum,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_forward_ctc_max import (
    model_forward_ctc_max,
)
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output

from i6_experiments.users.gaudino.datasets.search_data_opts import (
    # search_data_opts_ted2,
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

_tedlium2_ckpt_path_w_trafo_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/baseline_w_trafo_lm_23_12_28/average.pt"
_librispeech960_ckpt_path_w_trafo_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/full_w_trafo_lm_import_2024_02_05/average.pt"
_tedlium2_ckpt_path_lay8_ctc = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/without_lm/model_ctc1.0_att1.0_lay8/average.pt"
_tedlium2_ckpt_path_att_only_currL = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/without_lm/model_att_only_currL/average.pt"
_tedlium2_ckpt_path_ctc_only = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/without_lm/model_ctc_only/average.pt"

_tedlium2_ckpt_path_baseline__ctc_only = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/without_lm/model_baseline__ctc_only/average.pt"

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


def test_compute_search_errors_optsr():
    from returnn.util import debug

    debug.install_lib_sig_segfault()
    debug.install_native_signal_handler()
    debug.init_faulthandler()

    dataset = "tedlium2"
    # dataset = "librispeech960"

    if dataset == "tedlium2":
        search_data_opts_ted2 = {
            "class": "MetaDataset",
            "data_map": {
                "audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes"),
                "search_bpe_labels": ("hdf_dataset", "data")
            },
            "datasets": {
                "zip_dataset": {
                    "class": "OggZipDataset",
                    "path": generic_job_output(
                        "i6_core/returnn/oggzip/BlissToOggZipJob.Wgp6724p1XD2/output/out.ogg.zip"
                    ).get_path(),
                    # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.Wgp6724p1XD2/output/out.ogg.zip",
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
                        ).get_path(),
                        # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes",
                        "vocab_file": generic_job_output(
                            "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab"
                        ).get_path(),
                        # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab",
                        "unknown_label": "<unk>",
                        "seq_postfix": [0],
                    },
                    "segment_file": None,
                    "partition_epoch": 1,
                    # "seq_ordering": "sorted_reverse",
                },
                "hdf_dataset": {
                    "class": "HDFDataset",
                    "files": ["/u/luca.gaudino/debug/recombine/lenNorm/out_best_wo_blank.hdf"],
                }
            },
            "seq_order_control_dataset": "zip_dataset",
        }

        target_dimension = 1057
        search_data_opts = search_data_opts_ted2
        seq_list = [f"TED-LIUM-realease2/AlGore_2009/{i}" for i in range(1, 34)]
        # seq_list = ["TED-LIUM-realease2/BlaiseAguerayArcas_2007/22"] # fixed_random_subset = 1
        seq_list = ["TED-LIUM-realease2/AlGore_2009/36"]
        # seq_list = ['TED-LIUM-realease2/CraigVenter_2008/67', 'TED-LIUM-realease2/CraigVenter_2008/55', 'TED-LIUM-realease2/BarrySchwartz_2005G/78', 'TED-LIUM-realease2/CraigVenter_2008/22']

        batch_num_seqs = 1
        model_args = {
            "add_trafo_lm": True,
            "target_embed_dim": 256,
            "mel_normalization": True,
        }
        pt_checkpoint_path = _tedlium2_ckpt_path_w_trafo_lm

    elif dataset == "librispeech960":
        target_dimension = 10025
        search_data_opts = search_data_opts_librispeech960
        seq_list = [
            f"dev-other/116-288045-{i:04d}/116-288045-{i:04d}" for i in range(33)
        ]
        # seq_list=["dev-other/1585-131718-0004/1585-131718-0004", "dev-other/1255-90413-0020/1255-90413-0020"]

        batch_num_seqs = 10
        # batch_num_seqs = max(10, len(seq_list))
        model_args = {
            "add_trafo_lm": True,
            "trafo_lm_args": {
                "num_layers": 24,
                "layer_out_dim": 1024,
                "att_num_heads": 8,
                "use_pos_enc": True,
                "ff_activation": "relu",
            },
        }
        pt_checkpoint_path = _librispeech960_ckpt_path_w_trafo_lm

    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=80, kind=Dim.Types.Feature)
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
    search_target_dim = target_dim.copy()
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
    search_target_spatial_dim = target_spatial_dim.copy()
    target = Tensor(
        "target", dim_tags=[batch_dim, target_spatial_dim], sparse_dim=target_dim
    )
    search_target = Tensor(
        "search_target",
        dim_tags=[batch_dim, search_target_spatial_dim],
        sparse_dim=search_target_dim,
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
                "search_bpe_labels": {
                    "dim_tags": search_target.dims,
                    "sparse_dim": search_target.sparse_dim,
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
    dataset.init_seq_order(
        epoch=1,
        seq_list=seq_list,
    )
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

    print("*** Convert old model to new model")
    print(pt_checkpoint_path)

    print("*** Create new model")
    ctc_only = False
    time_sync = False
    time_sync_search_args = {
        "beam_size": 12,
        # "att_scale": 0.0,
        "ctc_scale": 1.0,
        "add_trafo_lm": True,
        "lm_scale": 0.5,
        "remove_trafo_lm_eos": True,  # ?
        # "add_eos_to_end": True,
    }
    ctc_only_search_args = {}
    search_args = {
        "beam_size": 12,
        "att_scale": 0.8,
        "ctc_scale": 0.2,
        "use_ctc": True,
        # "encoder_ctc": True,
        # "add_trafo_lm": True,
        # "lm_scale": 0.5,
        # "ctc_scale": 1.0,
        # "use_ctc": True,
        # "prior_corr": True,
        # "prior_scale": 0.5,
        # "ctc_prior_file": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt",  # ted2 baseline
    }
    if time_sync:
        search_args = time_sync_search_args
    if ctc_only:
        search_args = ctc_only_search_args

    # returnn/torch/fonrtend/_backend 1635: sizes_raw = torch.reshape(sizes.raw_tensor, [batch_dim]).to('cpu')
    new_model = MakeModel.make_model(
        in_dim,
        target_dim,
        num_enc_layers=num_layers,
        search_args=search_args,
        model_args=model_args,
    )

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
    pt_module.load_state_dict(checkpoint_state["model"])

    cuda = torch.device("cuda")
    pt_module.to(cuda)
    extern_data["audio_features"].raw_tensor = extern_data[
        "audio_features"
    ].raw_tensor.to(cuda)

    print("*** Search ...")

    compare_to_sum = False

    # forward ground truth
    with torch.no_grad():
        with rf.set_default_device_ctx("cuda"):
            if compare_to_sum:
                (
                    seq_targets_gt,
                    seq_log_prob_gt,
                    out_spatial_dim_gt,
                    beam_dim_gt,
                ) = model_forward_ctc_sum(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                    ground_truth=extern_data["bpe_labels"],
                )
            else:
                (
                    seq_targets_gt,
                    seq_log_prob_gt,
                    out_spatial_dim_gt,
                    beam_dim_gt,
                ) = model_forward_ctc_max(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                    ground_truth=extern_data["bpe_labels"],
                )
    print(seq_targets_gt, seq_targets_gt.raw_tensor)  # seq_targets [Batch,Beam,T]
    print("Out spatial dim:", out_spatial_dim_gt)

    # forward search_out
    with torch.no_grad():
        with rf.set_default_device_ctx("cuda"):
            if compare_to_sum:
                (
                    seq_targets,
                    seq_log_prob,
                    out_spatial_dim,
                    beam_dim,
                ) = model_forward_ctc_sum(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                    ground_truth=extern_data["search_bpe_labels"], # TODO
                )
            else:
                (
                    seq_targets,
                    seq_log_prob,
                    out_spatial_dim,
                    beam_dim,
                ) = model_forward_ctc_max(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                    ground_truth=extern_data["search_bpe_labels"], # TODO
                )
    print(seq_targets, seq_targets.raw_tensor)  # seq_targets [Batch,Beam,T]
    print("Out spatial dim:", out_spatial_dim)

    num_search_errors = 0
    num_num_search_errors = 0

    # serialize output
    vocab_1 = Vocabulary(
        "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab",
        eos_label=0,
    )
    for batch_idx in range(batch_dim.get_dim_value()):

        hyp_len = seq_targets.dims[2].dyn_size_ext.raw_tensor[batch_idx] - 1
        hyp_len_gt = seq_targets_gt.dims[2].dyn_size_ext.raw_tensor[batch_idx] - 1

        score = float(seq_log_prob.raw_tensor[batch_idx])
        score_gt = float(seq_log_prob_gt.raw_tensor[batch_idx])

        hyp_ids = seq_targets.raw_tensor[batch_idx, :, :hyp_len].squeeze()
        hyp_gt_ids = seq_targets_gt.raw_tensor[batch_idx, :, :hyp_len_gt].squeeze()

        hyp_serialized = vocab_1.get_seq_labels(hyp_ids)
        hyp_gt_serialized = vocab_1.get_seq_labels(hyp_gt_ids)

        print(f"  ({score!r}, {hyp_serialized!r}),")
        print(f"  ({score_gt!r}, {hyp_gt_serialized!r}),\n")
        if score_gt > score and (
            len(hyp_ids) != len(hyp_gt_ids) or not torch.all(hyp_ids == hyp_gt_ids)
        ):
            print(f"  {score_gt!r} > {score!r} Search Error!\n")
            num_search_errors += 1
            if math.isclose(score_gt, score):
                num_num_search_errors += 1

            continue

    print("Total number of search errors: ", num_search_errors)
    print("Total number of numerical search errors: ", num_num_search_errors)


# `py` is the default sis config function name. so when running this directly, run the import test.
# So you can just run:
# `sis m recipe/i6_experiments/users/zeyer/experiments/....py`
py = test_compute_search_errors_optsr

if __name__ == "__main__":
    mod_name = __package__
    # if mod_name.startswith("recipe."):
    #     mod_name = mod_name[len("recipe.") :]
    # mod_name += "." + os.path.basename(__file__)[: -len(".py")]
    test_compute_search_errors_optsr()
