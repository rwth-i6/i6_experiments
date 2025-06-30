from .tools_paths import RETURNN_ROOT, RETURNN_EXE
from .config_builder import AEDConfigBuilder
from .configs import mini_lstm_config_v1
from .model.aed import aed_model_def, _returnn_v2_get_model
from .train import TrainExperiment
from .model import custom_load_params

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora
from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from sisyphus import Path, tk

LIBRISPEECH_CORPUS = LibrispeechCorpora()
BPE1K_OPTS = dict(
  bpe_codes_path=Path(
    "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/"
    "ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.codes"
  ),
  vocab_path=Path(
    "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/"
    "ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"
  ),
  num_labels=1056,
  bos_idx=0,
  eos_idx=0,
)


def py():
  configs = []

  # --------------------------------------- 11GB multi-GPU experiments ---------------------------------------

  # # Baseline v1
  # configs.append(dict(
  #   model_opts=config_11gb_mgpu_v1["model_opts"],
  #   train_opts=config_11gb_mgpu_v1["train_opts"],
  #   alias="models/v1_11gb_mgpu/baseline",
  # ))

  # --------------------------------------- 24GB single-GPU experiments ---------------------------------------
  # Baseline v1
  configs.append(dict(
    model_opts=mini_lstm_config_v1["model_opts"],
    train_opts=dict_update_deep(
      mini_lstm_config_v1["train_opts"],
      {"preload_from_files": {
        "pretrained_global_att_params": {
          "filename": Path("/work/asr3/zeyer/schmitt/debug/debug_torch_ilm_multi_vs_single_gpu_aed/single_gpu_checkpoint.pt"),
          "init_for_train": True,
          "ignore_missing": True,
          "custom_missing_load_func": custom_load_params.load_missing_params,
        }
      }}
    ),
    alias="ilm-experiments/single-gpu/baseline",
  ))

  for config in configs:
    config_builder = AEDConfigBuilder(
      dataset=LIBRISPEECH_CORPUS,
      vocab_opts=BPE1K_OPTS,
      model_def=aed_model_def,
      get_model_func=_returnn_v2_get_model,
    )
    config_builder.config_dict.update(config["model_opts"])

    train_rqmt = dict(time=20, cpu=4)
    train_rqmt.update(dict(gpu_mem=11, mem=20))

    # train model
    train_exp = TrainExperiment(
      config_builder=config_builder,
      alias=config["alias"],
      train_opts=config["train_opts"],
      train_rqmt=train_rqmt,
    )
    train_exp.run_train()

    # recog_exp.run_analysis(
    #   analysis_opts={
    #     "analyze_gradients": True,
    #     "att_weight_seq_tags": [
    #       "train-other-960/1246-124548-0042/1246-124548-0042",
    #       "train-other-960/40-222-0033/40-222-0033",
    #       "train-other-960/103-1240-0038/103-1240-0038",
    #     ],
    #     "ref_alignment_hdf": LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths[analysis_corpus_key],
    #     "ref_alignment_blank_idx": LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
    #     "ref_alignment_vocab_path": LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
    #     "analyze_gradients_plot_encoder_layers": True,
    #     "analyze_gradients_plot_log_gradients": True,
    #     "dump_self_att": True,
    #   }
    # )
