"""
Config for experiments with AED on Librispeech 100h.

Based on
i6_experiments/users/rossenbach/experiments/librispeech/librispeech_100_attention/conformer_2022/conformer_tf_feature.py
"""

import copy
import os.path
from typing import List, Tuple, Dict, Optional, Union

from sisyphus import tk, gs
from i6_core.tools import CloneGitRepositoryJob
from i6_core.report import Report
from i6_core.returnn import CodeWrapper
from i6_experiments.users.vieting.experiments.librispeech.librispeech_100_attention.stoch_feat.pipeline import (
    build_training_datasets, build_test_dataset, training, search, search_single, get_average_checkpoint_v2
)
from i6_experiments.users.vieting.experiments.librispeech.librispeech_100_attention.stoch_feat.\
  attention_asr_config import create_config, ConformerEncoderArgs, RNNDecoderArgs
from i6_experiments.users.vieting.experiments.librispeech.librispeech_100_attention.stoch_feat.\
  base_config import get_lm_opts, apply_fairseq_init_to_conformer_encoder
from i6_experiments.users.vieting.experiments.librispeech.librispeech_100_attention.stoch_feat.\
  feature_extraction_net import log10_net_10ms_ref, log10_net_10ms, dim_tags, pre_emphasis


def conformer_tf_features():
  returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh",
                        hash_overwrite="GENERIC_RETURNN_LAUNCHER")
  returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
  prefix_name = "experiments/librispeech/librispeech_100_attention/stoch_feat/"
  prefix_name +=  os.path.splitext(os.path.basename(__file__))[0]

  # build the training datasets object containing train, cv, dev-train and the extern_data dict
  training_datasets_speedperturbed = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name,
                                                             bpe_size=2000, use_raw_features=True,
                                                             link_speed_perturbation=True)
  training_datasets = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name,
                                              bpe_size=2000, use_raw_features=True)
  # retrain dataset has curriculum options disabled
  training_datasets_speedperturbed_retrain = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name,
                                                                     bpe_size=2000, use_raw_features=True,
                                                                     link_speed_perturbation=True, use_curicculum=False)

  # build testing datasets
  test_dataset_tuples = {}
  for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe,
                                                      returnn_root=returnn_root_datasets, output_path=prefix_name,
                                                      use_raw_features=True)

  # ------------------------------------------------------------------------------------------------------------------ #

  conformer_enc_args = ConformerEncoderArgs(
    num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
    pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

  conformer_enc_fixed_bn_args = ConformerEncoderArgs(
    num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
    pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

  apply_fairseq_init_to_conformer_encoder(conformer_enc_args)
  conformer_enc_args.ctc_loss_scale = 1.0

  apply_fairseq_init_to_conformer_encoder(conformer_enc_fixed_bn_args)
  conformer_enc_fixed_bn_args.ctc_loss_scale = 1.0

  rnn_dec_args = RNNDecoderArgs()
  training_args = {}

  # LR scheduling
  training_args['const_lr'] = [50, 20]  # use const LR during pretraining
  training_args['wup_start_lr'] = 0.0002
  training_args['wup'] = 30
  training_args['with_staged_network'] = True
  training_args['speed_pert'] = True

  # overwrite BN params
  conformer_enc_args.batch_norm_opts = {
    'momentum': 0.1,
    'epsilon': 1e-3,
    'update_sample_only_in_training': False,
    'delay_sample_update': False
  }

  conformer_enc_fixed_bn_args.batch_norm_opts = {
    'momentum': 0.1,
    'epsilon': 1e-3,
    'update_sample_only_in_training': True,
    'delay_sample_update': True
  }

  transf_lm_opts = get_lm_opts()

  # conformer round 2
  name = 'tf_feature_conformer_12l_lstm_1l_normal_v2'
  local_conformer_enc_args = copy.deepcopy(conformer_enc_args)
  local_training_args = copy.deepcopy(training_args)

  # pretraining
  local_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 18000 * 200}
  local_training_args['pretrain_reps'] = 5
  local_training_args['batch_size'] = 12000 * 200  # frames * samples per frame

  exp_prefix = prefix_name + "/" + name
  args = copy.deepcopy(
    {**local_training_args, "encoder_args": conformer_enc_fixed_bn_args, "decoder_args": rnn_dec_args})
  args['name'] = name
  returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                       commit="3f62155a08722310f51276792819b3c7c64ad356").out_repository

  def run_exp_v2(
    ft_name, feature_extraction_net, datasets, train_args, search_args=None, search_extraction_net=None,
    report_args=None,
  ):
    report = Report(columns_start=["name"], columns_end=["dev-clean", "dev-other", "test-clean", "test-other"])
    report_args = {
      "specaug": train_args["encoder_args"].specaug,
      "speed": "U(0.9-1.1)" if train_args["speed_pert"] else "1.0",
      **(report_args or {})}
    search_args = search_args if search_args is not None else train_args
    search_extraction_net = search_extraction_net if search_extraction_net is not None else feature_extraction_net
    returnn_config = create_config(training_datasets=datasets, **train_args,
                                   feature_extraction_net=feature_extraction_net)
    returnn_search_config = create_config(training_datasets=datasets, **search_args,
                                          feature_extraction_net=search_extraction_net, is_recog=True)
    train_job = training(ft_name, returnn_config, returnn_exe, returnn_root, num_epochs=250)
    average = get_average_checkpoint_v2(train_job, returnn_exe=returnn_exe, returnn_root=returnn_root, num_average=4)
    from i6_core.returnn.training import GetBestTFCheckpointJob
    best_checkpoint_job = GetBestTFCheckpointJob(
      train_job.out_model_dir,
      train_job.out_learning_rates,
      key="dev_score_output/output_prob",
      index=0)

    exp_name = ft_name.split("/")[-1]
    # best checkpoint
    _, report_values = search(
      ft_name + "/default_best", returnn_search_config,
      best_checkpoint_job.out_checkpoint, test_dataset_tuples, returnn_exe, returnn_root, mail=False)
    rep_args = {k.replace(ft_name + "/default_best", "")[:-4]: v for k, v in report_values.items()}
    rep_args.update({"name": exp_name, "checkpoint": "best", **report_args})
    report.add(rep_args)
    # last checkpoint
    _, report_values = search(
      ft_name + "/default_last", returnn_search_config, train_job.out_checkpoints[250],
      test_dataset_tuples, returnn_exe, returnn_root, mail=False)
    rep_args = {k.replace(ft_name + "/default_last", "")[:-4]: v for k, v in report_values.items()}
    rep_args.update({"name": exp_name, "checkpoint": "last", **report_args})
    report.add(rep_args)
    # averaged checkpoint
    _, report_values = search(
      ft_name + "/average_4", returnn_search_config, average, test_dataset_tuples,
      returnn_exe, returnn_root, mail=False)
    rep_args = {k.replace(ft_name + "/average_4", "")[:-4]: v for k, v in report_values.items()}
    rep_args.update({"name": exp_name, "checkpoint": "average_4", **report_args})
    report.add(rep_args)

    ext_lm_search_args = copy.deepcopy(search_args)
    ext_lm_search_args["ext_lm_opts"] = transf_lm_opts

    for lm_scale in [0.36, 0.38, 0.4, 0.42, 0.44]:
      search_args = copy.deepcopy(ext_lm_search_args)
      search_args['ext_lm_opts']['lm_scale'] = lm_scale
      returnn_config = create_config(training_datasets=datasets, **search_args,
                                     feature_extraction_net=search_extraction_net, is_recog=True)
      returnn_config.config["batch_size"] = 8000 * 200  # smaller size for recognition
      wer = search_single(ft_name + "/default_last_ext_lm_%.2f" % lm_scale,
                    returnn_config,
                    train_job.out_checkpoints[250],
                    test_dataset_tuples["dev-other"][0],
                    test_dataset_tuples["dev-other"][1],
                    returnn_exe,
                    returnn_root)
      rep_args = {"name": exp_name, "checkpoint": "last", "dev-other": wer, "ext_lm": lm_scale, **report_args}
      report.add(rep_args)
    return report

  report_list = []
  # ref
  report_list.append(run_exp_v2(
    exp_prefix + "/" + "raw_log10_ref", log10_net_10ms_ref, datasets=training_datasets_speedperturbed,
    train_args=args))
  # baseline with custom log mel network
  args_base = copy.deepcopy(args)
  args_base["network_prolog"] = dim_tags
  report_list.append(run_exp_v2(
    exp_prefix + "/" + "raw_log10", log10_net_10ms, datasets=training_datasets_speedperturbed,
    train_args=args_base))
  # no SpecAugment
  args_no_spec = copy.deepcopy(args_base)
  args_no_spec["encoder_args"].specaug = False
  report_list.append(run_exp_v2(
    exp_prefix + "/" + "raw_log10_nospec", log10_net_10ms, datasets=training_datasets_speedperturbed,
    train_args=args_no_spec))
  # no speed perturbation
  args_no_speed = copy.deepcopy(args_base)
  args_no_speed["speed_pert"] = False
  report_list.append(run_exp_v2(
    exp_prefix + "/" + "raw_log10_nospeed", log10_net_10ms, datasets=training_datasets,
    train_args=args_no_speed))
  # no Specaugment and no speed perturbation
  args_no_speed_no_spec = copy.deepcopy(args_no_speed)
  args_no_speed_no_spec["encoder_args"].specaug = False
  report_list.append(run_exp_v2(
    exp_prefix + "/" + "raw_log10_nospec_nospeed", log10_net_10ms, datasets=training_datasets,
    train_args=args_no_speed_no_spec))
  # perturbation of center frequencies
  for pert_cf, stddev, specaug, speed in [
    ("mul", 0.01, False, False), ("mul", 0.02, False, False), ("mul", 0.04, False, False),  # a bit better on other
    ("add", 1.0, False, False), ("add", 2.0, False, False), ("add", 4.0, False, False),  # helps on other, 1.0 is best
    ("add", 0.5, True, False), ("add", 1.0, True, False), ("add", 1.5, True, False),
    ("add", 1.0, True, True),
    ("add_nnd", 1.0, True, False), #("add_nnd", 2.0, True, False),
  ]:
    args_tmp = copy.deepcopy(args_base)
    args_tmp["encoder_args"].specaug = specaug
    args_tmp["speed_pert"] = speed
    feat_net = copy.deepcopy(log10_net_10ms)
    subnet = feat_net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]
    subnet["center_freqs_clean"] = copy.deepcopy(subnet["center_freqs"])
    eval_str = f"tf.random.normal((82,), mean=0.0, stddev={stddev}, name='center_freqs_noise')"
    if pert_cf.startswith("mul"):
      eval_str = f"source(0) * (1 + {eval_str})"
    elif pert_cf.startswith("add"):
      eval_str = f"source(0) + {eval_str}"
    else:
      raise NotImplementedError(f"Unknown center frequency perturbation type: {pert_cf}")
    subnet["center_freqs"] = {"class": "eval", "eval": eval_str, "from": "center_freqs_clean"}
    if "nnd" in pert_cf:
      # avoid negative differences because they lead to filters which are not limited on one side.
      subnet["center_freqs_diff_raw"] = copy.deepcopy(subnet["center_freqs_diff"])
      subnet["threshold"] = {"class": "constant", "value": 1e-5}
      subnet["center_freqs_diff"] = {
        "class": "combine",
        "kind": "maximum",
        "from": ["center_freqs_diff_raw", "threshold"]}
    feat_net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"] = subnet
    name_tmp = (
        exp_prefix + "/" +
        f"raw_log10{'' if specaug else '_nospec'}{'' if speed else '_nospeed'}_pert_cf{pert_cf}{stddev}")
    report_list.append(run_exp_v2(
      name_tmp, feat_net, datasets=training_datasets,
      train_args=args_tmp, report_args={"pert_cf": f"{pert_cf}{stddev}"}))

  # filter width perturbation
  args_tmp = copy.deepcopy(args_base)
  for stddev in [0.1, 0.5, 1.0]:  # 2.0 diverges
    name_tmp = exp_prefix + "/" + f"raw_log10_pert_fw{stddev}"
    feat_net = copy.deepcopy(log10_net_10ms)
    subnet = feat_net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"]
    subnet["center_freqs_c"] = {
      "class": "slice",
      "axis": "F",
      "slice_start": 1,
      "slice_end": -1,
      "from": "center_freqs",
      "out_dim": CodeWrapper("center_freqs_dim"),
    }
    subnet["filter_offset_raw"] = {"class": "combine", "kind": "sub", "from": ["fft_bins", "center_freqs_c"]}
    subnet["filter_offset"] = {
      "class": "eval", "eval": f"source(0) * tf.random.normal((80,), stddev={stddev})", "from": "filter_offset_raw"}
    subnet["mel_filterbank_l_noisy"] = {
      "class": "combine", "kind": "add", "from": ["mel_filterbank_l", "filter_offset"]}
    subnet["mel_filterbank_r_noisy"] = {
      "class": "combine", "kind": "sub", "from": ["mel_filterbank_r", "filter_offset"]}
    subnet["mel_filterbank_lr"]["from"] = ["mel_filterbank_l_noisy", "mel_filterbank_r_noisy"]
    feat_net["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"] = subnet
    report_list.append(run_exp_v2(
      name_tmp, feat_net, datasets=training_datasets,
      train_args=args_tmp, report_args={"pert_fw": f"{stddev}"}))

  # pre-emphasis and perturbation of it
  args_tmp = copy.deepcopy(args_base)
  for pe in [0.9, 0.95, 1.0, (0.9, 1.0)]:  # (0.95, 1.0) diverges after 30 sub-epochs
    if isinstance(pe, (float, int)):
      eval_str = f"source(0) * {pe}"
      pe_str = str(pe)
    elif isinstance(pe, tuple):
      assert len(pe) == 2
      eval_str = f"source(0) * tf.random.uniform((1,), {pe[0]}, {pe[1]}, name='preemphasis_factor')"
      pe_str = f"U({pe[0]}-{pe[1]})"
    else:
      raise NotImplementedError(f"Unknown pre-emphasis type: {pe}")
    name_tmp = exp_prefix + "/" + f"raw_log10_pe_{pe_str}"
    feat_net = copy.deepcopy(log10_net_10ms)
    feat_net["log_mel_features"]["subnetwork"]["pre_emphasis"] = copy.deepcopy(pre_emphasis)
    feat_net["log_mel_features"]["subnetwork"]["stft"]["from"] = "pre_emphasis"
    feat_net["log_mel_features"]["subnetwork"]["pre_emphasis"]["subnetwork"]["shift_0_scale"]["eval"] = eval_str
    report_list.append(run_exp_v2(
      name_tmp, feat_net, datasets=training_datasets,
      train_args=args_tmp, report_args={"pe": f"{pe_str}"}))

  report = Report.merge_reports(report_list)
  tk.register_report(
    f"{exp_prefix}/report.csv",
    values=report.get_values(),
    template=report.get_template())
