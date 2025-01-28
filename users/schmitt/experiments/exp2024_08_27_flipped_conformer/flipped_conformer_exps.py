import copy
from typing import Dict, Tuple

from i6_core.returnn import PtCheckpoint
from i6_core.recognition.scoring import ScliteJob

from i6_experiments.users.schmitt.recognition.scoring import GetCheckpointWithBestWer

from .tools_paths import RETURNN_ROOT, RETURNN_EXE
from .config_builder import AEDConfigBuilder
from .configs import config_24gb_v1, config_11gb_mgpu_v1
from .model.aed import aed_model_def, _returnn_v2_get_model
from .model.encoder.global_ import (
  GlobalConformerEncoderWAbsolutePos,
  GlobalConformerEncoderWFinalLayerNorm,
  ConformerEncoderLayerWoFinalLayerNorm,
)
from .train import TrainExperiment
from .recog import RecogExperiment, model_recog, _returnn_v2_forward_step, _returnn_v2_get_forward_callback
from .analysis.gmm_alignments import setup_gmm_alignment, LIBRISPEECH_GMM_WORD_ALIGNMENT

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora
from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

import returnn.frontend as rf
from returnn.frontend.build_from_dict import _get_cls_name

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


def _get_recog_opts() -> Dict:
  return {
    "recog_def": model_recog,
    "beam_size": 12,
    "dataset_opts": {
      "corpus_key": "dev-other",
    },
    "forward_step_func": _returnn_v2_forward_step,
    "forward_callback": _returnn_v2_get_forward_callback,
    "length_normalization_exponent": 1.0,
  }


def _run_recogs(
        alias,
        config_builder,
        model_dir,
        learning_rates,
        checkpoints
):
  recog_checkpoints = AEDConfigBuilder.get_recog_checkpoints(
    model_dir,
    learning_rates,
    key="dev_loss_ce",
    checkpoints=checkpoints,
  )
  checkpoint_to_wer: Dict[Tuple[int, PtCheckpoint], tk.Variable] = dict()
  base_exp_alias: str

  recog_opts = _get_recog_opts()
  for checkpoint_alias, checkpoint in recog_checkpoints.items():
    recog_exp = RecogExperiment(
      alias=alias,
      config_builder=config_builder,
      checkpoint=checkpoint,
      checkpoint_alias=checkpoint_alias,
      recog_opts=recog_opts,
      search_rqmt=dict(cpu=4)
    )
    base_exp_alias = recog_exp.base_exp_alias

    score_job = recog_exp.run_eval()
    checkpoint_to_wer[(checkpoint_alias, checkpoint)] = score_job.out_wer

  best_checkpoint_job = GetCheckpointWithBestWer(checkpoint_to_wer)
  best_checkpoint_job.add_alias(f"{base_exp_alias}/best-wer-checkpoint")
  tk.register_output(best_checkpoint_job.get_one_alias(), best_checkpoint_job.out_checkpoint)

  recog_checkpoints.update({"best-wer": best_checkpoint_job.out_checkpoint})

  return recog_checkpoints


def py():
  # needs to be called so that the GMM word alignments become available
  setup_gmm_alignment(LIBRISPEECH_CORPUS.corpus_paths["train-other-960"])

  configs = []

  # --------------------------------------- 11GB multi-GPU experiments ---------------------------------------

  # # Baseline v1
  # configs.append(dict(
  #   model_opts=config_11gb_mgpu_v1["model_opts"],
  #   train_opts=config_11gb_mgpu_v1["train_opts"],
  #   alias="models/v1_11gb_mgpu/baseline",
  # ))

  # # Mod. baseline v1: no data filtering
  # configs.append(dict(
  #   model_opts=config_11gb_mgpu_v1["model_opts"],
  #   train_opts=dict_update_deep(
  #     config_11gb_mgpu_v1["train_opts"],
  #     {},
  #     ["max_seq_length"]
  #   ),
  #   alias="models/v1_11gb_mgpu/no-data-filtering",
  # ))
  #
  # # Mod. baseline v1: filter targets >75
  # configs.append(dict(
  #   model_opts=config_11gb_mgpu_v1["model_opts"],
  #   train_opts=dict_update_deep(
  #     config_11gb_mgpu_v1["train_opts"],
  #     {"max_seq_length": {"targets": 75}},
  #   ),
  #   alias="models/v1_11gb_mgpu/filter-targets-75",
  # ))
  #
  # # Mod. baseline v1: disable self-attention for 1st epoch
  # configs.append(dict(
  #   model_opts=dict_update_deep(
  #     config_11gb_mgpu_v1["model_opts"],
  #     {"encoder_opts.disable_self_attention": {"num_epochs": 20}},
  #   ),
  #   train_opts=config_11gb_mgpu_v1["train_opts"],
  #   alias="models/v1_11gb_mgpu/disable-self-att-1st-epoch",
  # ))
  #
  # # Mod. baseline v1: 17 layers; 400 dim
  # configs.append(dict(
  #   model_opts=dict_update_deep(
  #     config_11gb_mgpu_v1["model_opts"],
  #     {
  #       "encoder_opts.num_layers": 17,
  #       "encoder_opts.out_dimension": 400,
  #     },
  #   ),
  #   train_opts=config_11gb_mgpu_v1["train_opts"],
  #   alias="models/v1_11gb_mgpu/17-layers_400-dim",
  # ))
  #
  # # Mod. baseline v1: use CTC loss after 4th and 8th layer
  # configs.append(dict(
  #   model_opts=config_11gb_mgpu_v1["model_opts"],
  #   train_opts=dict_update_deep(
  #     config_11gb_mgpu_v1["train_opts"],
  #     {"aux_loss_layers": (4, 8)},
  #   ),
  #   alias="models/v1_11gb_mgpu/use-ctc-loss",
  # ))
  #
  # # Mod. baseline v1: different random seeds
  # configs += [dict(
  #   model_opts=config_11gb_mgpu_v1["model_opts"],
  #   train_opts=dict_update_deep(
  #     config_11gb_mgpu_v1["train_opts"],
  #     {"random_seed": seed},
  #     ["max_seq_length"]
  #   ),
  #   alias=f"models/v1_11gb_mgpu/rand-seed-{seed}",
  # ) for seed in (9999, 1234, 1111, 4321, 5678, 8765, 2222, 3333)]
  #
  # # Mod. baseline v1: with both absolute and relative positional encodings
  # configs.append(dict(
  #   model_opts=dict_update_deep(
  #     config_11gb_mgpu_v1["model_opts"],
  #     {"encoder_opts.class": _get_cls_name(GlobalConformerEncoderWAbsolutePos)},
  #   ),
  #   train_opts=config_11gb_mgpu_v1["train_opts"],
  #   alias="models/v1_11gb_mgpu/abs-and-rel-pos-enc",
  # ))
  #
  # # Mod. baseline v1: conformer layers w/o final layer norm
  # configs.append(dict(
  #   model_opts=dict_update_deep(
  #     config_11gb_mgpu_v1["model_opts"],
  #     {"encoder_opts.encoder_layer.class": _get_cls_name(ConformerEncoderLayerWoFinalLayerNorm)},
  #   ),
  #   train_opts=config_11gb_mgpu_v1["train_opts"],
  #   alias="models/v1_11gb_mgpu/conformer-layers-wo-final-layer-norm",
  # ))

  # --------------------------------------- 24GB single-GPU experiments ---------------------------------------
  # Baseline v1
  configs.append(dict(
    model_opts=config_24gb_v1["model_opts"],
    train_opts=config_24gb_v1["train_opts"],
    alias="models/v1_24gb/baseline",
  ))
  #
  # # Mod. baseline v1: different random seeds
  # configs += [dict(
  #   model_opts=config_24gb_v1["model_opts"],
  #   train_opts=dict_update_deep(
  #     config_24gb_v1["train_opts"],
  #     {"random_seed": seed},
  #     ["max_seq_length"]
  #   ),
  #   alias=f"models/v1_24gb/rand-seed-{seed}",
  # ) for seed in (1337, 8264, 2160, 2222, 5678)]

  for config in configs:
    config_builder = AEDConfigBuilder(
      dataset=LIBRISPEECH_CORPUS,
      vocab_opts=BPE1K_OPTS,
      model_def=aed_model_def,
      get_model_func=_returnn_v2_get_model,
    )
    config_builder.config_dict.update(config["model_opts"])

    train_rqmt = dict(time=80, cpu=4)
    if "torch_distributed" in config["train_opts"]:
      train_rqmt.update(dict(horovod_num_processes=4, distributed_launch_cmd="torchrun", gpu_mem=11, mem=15))
    else:
      # with mem=15, the multi proc dataset gets OOM
      train_rqmt.update(dict(gpu_mem=24, mem=20))

    # train model
    train_exp = TrainExperiment(
      config_builder=config_builder,
      alias=config["alias"],
      train_opts=config["train_opts"],
      train_rqmt=train_rqmt,
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    # do recognition on last, best, best-4-avg checkpoints
    # and determine checkpoint with best WER (recog_checkpoints["best-wer"])
    recog_checkpoints = _run_recogs(
      alias=config["alias"],
      config_builder=config_builder,
      model_dir=model_dir,
      learning_rates=learning_rates,
      checkpoints=checkpoints
    )

    # do some analysis of the trained model
    # first, create a new recog experiment for the analysis. we use the checkpoint with best WER for that
    analysis_corpus_key = "train"
    recog_opts = dict_update_deep(_get_recog_opts(), {"dataset_opts.corpus_key": analysis_corpus_key})
    recog_exp = RecogExperiment(
      alias=config["alias"],
      config_builder=config_builder,
      # checkpoint=recog_checkpoints["best-wer"],
      # checkpoint_alias="best-wer",
      checkpoint=checkpoints[518],
      checkpoint_alias=f"epoch-518",
      recog_opts=recog_opts,
      search_rqmt=dict(cpu=4)
    )

    # run the "analyze_gradients" job ("analyze_gradients": True),
    # which originally only analyzed gradients but now dumps all sorts of stuff
    # this was used to create the plots of Figure 1, 2, 3, 5, 6 of the paper
    # run the "dump_self_att" job ("dump_self_att": True), which dumps the self-attention energies into HDF files
    # which can be used to create a plot such as in Figure 4 of the paper
    recog_exp.run_analysis(
      analysis_opts={
        "analyze_gradients": True,
        "att_weight_seq_tags": [
          "train-other-960/1246-124548-0042/1246-124548-0042",
          "train-other-960/40-222-0033/40-222-0033",
          "train-other-960/103-1240-0038/103-1240-0038",
        ],
        "ref_alignment_hdf": LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths[analysis_corpus_key],
        "ref_alignment_blank_idx": LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
        "ref_alignment_vocab_path": LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
        "analyze_gradients_plot_encoder_layers": True,
        "analyze_gradients_plot_log_gradients": True,
        "dump_self_att": True,
      }
    )

    # run the "dump_gradients" job ("dump_gradients": True), which dumps the gradients into HDF files.
    # we do not set the seq tags for this job which means that it uses a 1% subset of the LS training data and dumps
    # the gradients for all sequences in that subset into an HDF file.
    # we used this to create the gradient alignments in Section V of the paper
    # recog_exp.run_analysis(
    #   analysis_opts={
    #     "att_weight_seq_tags": None,
    #     "ref_alignment_hdf": LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths[analysis_corpus_key],
    #     "ref_alignment_blank_idx": LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
    #     "ref_alignment_vocab_path": LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
    #     "dump_gradients": True,
    #   }
    # )
