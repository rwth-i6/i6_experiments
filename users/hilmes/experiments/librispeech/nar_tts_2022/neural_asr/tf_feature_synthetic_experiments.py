import copy
from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob

from .pipeline import (
  build_training_datasets,
  build_test_dataset,
  training,
  search,
  get_average_checkpoint_v2,
)

from .attention_asr_config import (
  create_config,
  ConformerEncoderArgs,
  RNNDecoderArgs,
)
from .base_config import get_lm_opts, apply_fairseq_init_to_conformer_encoder
from .feature_extraction_net import (
  log10_net_10ms,
)



class LocalReport:
  def __init__(self, header):
    self.format_str = header
    self.values = {}


def train_conformer_with_synthetic_data(synthethic_corpus_dict):

  returnn_exe = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
  )
  returnn_cpu_exe = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
  )
  returnn_root_datasets = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443",
  ).out_repository
  pref_name = "experiments/librispeech/nar_tts_2022/conformer_training/synthetic/"
  for synth_name, synthethic_corpus in synthethic_corpus_dict.items():
    if ("mono" in synth_name and "xvec" in synth_name):
        continue
    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    prefix_name = pref_name + synth_name + "/"
    if "1000" in synth_name:
      partition_epoch = 30
    elif "860" in synth_name:
      partition_epoch = 25
    else:
      partition_epoch = 3
    training_datasets_speedperturbed = build_training_datasets(
      returnn_exe,
      returnn_root_datasets,
      prefix_name,
      bpe_size=2000,
      use_raw_features=True,
      link_speed_perturbation=True,
      synthetic_bliss=synthethic_corpus,
      synthetic_scale=1,
      original_scale=0,
      partition_epoch=partition_epoch,
    )
    # retrain dataset has curriculum options disabled
    training_datasets_speedperturbed_retrain = build_training_datasets(
      returnn_exe,
      returnn_root_datasets,
      prefix_name,
      bpe_size=2000,
      use_raw_features=True,
      link_speed_perturbation=True,
      use_curicculum=False,
      synthetic_bliss=synthethic_corpus,
      synthetic_scale=1,
      original_scale=0,
      partition_epoch=partition_epoch,
    )

    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
      test_dataset_tuples[testset] = build_test_dataset(
        testset,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root_datasets,
        output_path=prefix_name,
        use_raw_features=True,
      )

    conformer_enc_fixed_bn_args = ConformerEncoderArgs(
      num_blocks=12,
      input_layer="lstm-6",
      att_num_heads=8,
      ff_dim=2048,
      enc_key_dim=512,
      conv_kernel_size=32,
      pos_enc="rel",
      dropout=0.1,
      att_dropout=0.1,
      l2=0.0001,
    )

    apply_fairseq_init_to_conformer_encoder(conformer_enc_fixed_bn_args)
    conformer_enc_fixed_bn_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()
    training_args = {}

    # LR scheduling
    training_args["const_lr"] = [50, 20]  # use const LR during pretraining
    training_args["wup_start_lr"] = 0.0002
    training_args["wup"] = 30
    training_args["with_staged_network"] = True
    training_args["speed_pert"] = True

    conformer_enc_fixed_bn_args.batch_norm_opts = {
      "momentum": 0.1,
      "epsilon": 1e-3,
      "update_sample_only_in_training": True,
      "delay_sample_update": True,
    }

    transf_lm_opts = get_lm_opts()

    # conformer round 2
    name = "tf_feature_conformer_12l_lstm_1l_normal_v2/"
    local_training_args = copy.deepcopy(training_args)

    # pretraining
    local_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 22500 * 160}
    local_training_args["pretrain_reps"] = 5
    local_training_args["batch_size"] = 15000 * 160  # frames * samples per frame

    exp_prefix = prefix_name + name
    args = copy.deepcopy(
      {**local_training_args, "encoder_args": conformer_enc_fixed_bn_args, "decoder_args": rnn_dec_args}
    )
    args["name"] = name
    returnn_root = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn", commit="3f62155a08722310f51276792819b3c7c64ad356"
    ).out_repository

    header = "Name,best,,,,,last,,,,,average_4,,,\n,dev-clean,dev-other,test-clean,test-other,,dev-clean,dev-other,test-clean,test-other,,dev-clean,dev-other,test-clean,test-other\n"
    report = LocalReport(header=header)

    def run_exp_v2(
      ft_name, feature_extraction_net, datasets, train_args, search_args=None, search_extraction_net=None, report=None
    ):
      search_args = search_args if search_args is not None else train_args
      search_extraction_net = search_extraction_net if search_extraction_net is not None else feature_extraction_net
      returnn_config = create_config(
        training_datasets=datasets, **train_args, feature_extraction_net=feature_extraction_net
      )
      returnn_search_config = create_config(
        training_datasets=datasets, **search_args, feature_extraction_net=search_extraction_net, is_recog=True
      )
      train_job = training(ft_name, returnn_config, returnn_exe, returnn_root, num_epochs=250)
      # average = get_average_checkpoint(train_job, returnn_exe=returnn_exe, returnn_root=returnn_root, num_average=4)
      average = get_average_checkpoint_v2(train_job, returnn_exe=returnn_cpu_exe, returnn_root=returnn_root, num_average=4)
      from i6_core.returnn.training import GetBestTFCheckpointJob

      best_checkpoint_job = GetBestTFCheckpointJob(
        train_job.out_model_dir, train_job.out_learning_rates, key="dev_score_output/output_prob", index=0
      )

      format_str_1, values_1 = search(
        ft_name + "/default_best",
        returnn_search_config,
        best_checkpoint_job.out_checkpoint,
        test_dataset_tuples,
        returnn_exe,
        returnn_root,
      )
      format_str_2, values_2 = search(
        ft_name + "/default_last",
        returnn_search_config,
        train_job.out_checkpoints[250],
        test_dataset_tuples,
        returnn_exe,
        returnn_root,
      )
      format_str_3, values_3 = search(
        ft_name + "/average_4", returnn_search_config, average, test_dataset_tuples, returnn_exe, returnn_root
      )

      if report:
        report.format_str = (
          report.format_str + ft_name + "," + format_str_1 + ",," + format_str_2 + ",," + format_str_3 + "\n"
        )
        report.values.update(values_1)
        report.values.update(values_2)
        report.values.update(values_3)

      ext_lm_search_args = copy.deepcopy(args)
      ext_lm_search_args["ext_lm_opts"] = transf_lm_opts

      """
            for lm_scale in [0.36, 0.38, 0.4, 0.42, 0.44]:
                search_args = copy.deepcopy(ext_lm_search_args)
                search_args['ext_lm_opts']['lm_scale'] = lm_scale
                returnn_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=search_extraction_net, is_recog=True)
                returnn_config.config["batch_size"] = 10000*200  # smaller size for recognition
                search_single(ft_name + "/default_last_ext_lm_%.2f" % lm_scale,
                              returnn_config,
                              train_job.out_checkpoints[250],
                              test_dataset_tuples["dev-other"][0],
                              test_dataset_tuples["dev-other"][1],
                              returnn_exe,
                              returnn_root)"""
      return train_job

    args_bn_fix = copy.deepcopy(args)
    train_job_bn = run_exp_v2(
      exp_prefix + "raw_log10_bn_fix",
      log10_net_10ms,
      datasets=training_datasets_speedperturbed,
      train_args=args_bn_fix,
      report=report,
    )

    args_retrain = copy.deepcopy(args)
    args_retrain["retrain_checkpoint"] = train_job_bn.out_checkpoints[250]
    train_job = run_exp_v2(
      exp_prefix + "raw_log10_retrain",
      log10_net_10ms,
      datasets=training_datasets_speedperturbed_retrain,
      train_args=args_retrain,
      report=report,
    )

    tk.register_report(prefix_name + "/report.csv", values=report.values, template=report.format_str)
