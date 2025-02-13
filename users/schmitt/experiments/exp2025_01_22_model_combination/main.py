import copy
from typing import Dict, Tuple

from i6_core.returnn import PtCheckpoint

from .tools_paths import RETURNN_ROOT, RETURNN_EXE
from .model.checkpoints import convert_checkpoints, external_torch_checkpoints, external_tf_checkpoints
from .config_builder import AEDConfigBuilder, TinaAlignmentModelConfigBuilder
from .model.conformer_tina import (
  from_scratch_model_def as tina_conformer_model_def,
  _returnn_v2_get_model as tina_conformer_get_model,
  from_scratch_framewise_prob_model_def,
  from_scratch_diphone_fh_model_def,
  from_scratch_monophone_fh_model_def,
  from_scratch_phon_transducer_model_def,
)
from .model.aed import (
  aed_model_def,
  _returnn_v2_get_model as aed_get_model
)
from .model.ctc import (
  ctc_model_def,
  _returnn_v2_get_model as ctc_get_model,
  model_recog as ctc_model_recog,
  load_missing_params as load_missing_params_ctc,
)
from .model.custom_load_params import load_missing_params_aed
from .analysis.analysis import analyze_encoder
from .analysis.gmm_alignments import setup_gmm_alignment, LIBRISPEECH_GMM_WORD_ALIGNMENT
from .rescoring import rescore
from .configs import (
  config_24gb_v1,
  config_24gb_v2,
  config_ctc_v1,
  config_ctc_v2,
  config_post_hmm_v1,
  config_diphone_fh_v1,
  config_monophone_fh_v1,
  config_phon_transducer_v1,
)
from .train import TrainExperiment
from .recog import RecogExperiment, _returnn_v2_forward_step, _returnn_v2_get_forward_callback

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora

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
  num_labels=1_056,
  bos_idx=0,
  eos_idx=0,
)

BPE10K_OPTS = dict(
  bpe_codes_path=Path(
    "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes"
  ),
  vocab_path=Path(
    "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"
  ),
  num_labels=10_025,
  bos_idx=0,
  eos_idx=0,
)

PHONEME_OPTS = dict(
  bpe_codes_path=Path(
    "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/"
    "ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.codes"
  ),
  vocab_path=Path(
    "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/"
    "ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"
  ),
  num_labels=79,
  bos_idx=0,
  eos_idx=0,
)


def py():
  # needs to be called so that the GMM word alignments become available
  setup_gmm_alignment(LIBRISPEECH_CORPUS.corpus_paths["train-other-960"])

  convert_checkpoints()

  configs = []

  # for model_name in [
  #   "tina-phon-transducer_vit+fs-35-ep",
  # ]:
  #   configs.append(dict(
  #     config_builder_cls=TinaAlignmentModelConfigBuilder,
  #     model_opts=config_phon_transducer_v1["model_opts"],
  #     model_name=model_name,
  #     checkpoint=external_torch_checkpoints[model_name],
  #     model_def=from_scratch_phon_transducer_model_def,
  #     get_model_func=tina_conformer_get_model,
  #   ))

  for model_name in [
    "tina-monoph-fh_fs-50-ep",
  ]:
    configs.append(dict(
      config_builder_cls=TinaAlignmentModelConfigBuilder,
      model_opts=config_monophone_fh_v1["model_opts"],
      model_name=model_name,
      checkpoint=external_torch_checkpoints[model_name],
      model_def=from_scratch_monophone_fh_model_def,
      get_model_func=tina_conformer_get_model,
    ))

  for model_name in [
    "tina-diph-fh_vit+fs-35-ep",
    "tina-diph-fh_viterbi-20-ep",
  ]:
    configs.append(dict(
      config_builder_cls=TinaAlignmentModelConfigBuilder,
      model_opts=config_diphone_fh_v1["model_opts"],
      model_name=model_name,
      checkpoint=external_torch_checkpoints[model_name],
      model_def=from_scratch_diphone_fh_model_def,
      get_model_func=tina_conformer_get_model,
    ))

  configs.append(dict(
    config_builder_cls=TinaAlignmentModelConfigBuilder,
    model_opts=config_ctc_v1["model_opts"],
    model_name="tina-ctc_fs-100-ep",
    checkpoint=external_torch_checkpoints["tina-ctc_fs-100-ep"],
    model_def=from_scratch_framewise_prob_model_def,
    get_model_func=tina_conformer_get_model,
  ))

  for model_name in ["tina-monoph-post-hmm_fs-25-ep", "tina-monoph-post-hmm_fs-50-ep"]:
    configs.append(dict(
      config_builder_cls=TinaAlignmentModelConfigBuilder,
      model_opts=config_post_hmm_v1["model_opts"],
      model_name=model_name,
      checkpoint=external_torch_checkpoints[model_name],
      model_def=from_scratch_framewise_prob_model_def,
      get_model_func=tina_conformer_get_model,
    ))

  for model_name in [
    "aed_1k-bpe",
    "aed_1k-bpe-ctc",
    "context-1-transducer_1k-bpe_full-sum",
    "context-1-transducer_1k-bpe_fixed-path",
    "full-context-transducer_1k-bpe_fixed-path",
    "full-context-transducer_1k-bpe_full-sum",
  ]:
    configs.append(dict(
      config_builder_cls=AEDConfigBuilder,
      model_opts=config_24gb_v1["model_opts"],
      model_name=model_name,
      checkpoint=external_torch_checkpoints[model_name],
      model_def=aed_model_def,
      get_model_func=aed_get_model,
    ))

  for config in configs:
    config_builder_cls = config["config_builder_cls"]
    model_name = config["model_name"]

    feature_extraction = config["model_opts"].get("feature_extraction", "log-mel-on-the-fly")
    config_builder = config_builder_cls(
      dataset=LIBRISPEECH_CORPUS,
      vocab_opts=BPE1K_OPTS,
      model_def=config["model_def"],
      get_model_func=config["get_model_func"],
      feature_dimension=config["model_opts"].get("feature_dimension"),
      feature_extraction=feature_extraction,
      batch_size_factor=1 if feature_extraction is None else 160,
    )
    config_builder.config_dict.update(config["model_opts"])

    config_builder.config_dict["preload_from_files"] = dict(
      pretrained_params=dict(
        filename=config["checkpoint"],
        ignore_missing=True,
    ))
    if config_builder_cls == AEDConfigBuilder:
      # old checkpoint has slightly diff structure
      config_builder.config_dict["preload_from_files"]["pretrained_params"]["custom_missing_load_func"] = (
        load_missing_params_aed)

    analyze_encoder(
      config_builder=config_builder,
      seq_tags=[
        "train-other-960/5278-3072-0042/5278-3072-0042",
        "train-other-960/1246-124548-0042/1246-124548-0042",
        "train-other-960/40-222-0033/40-222-0033",
        "train-other-960/103-1240-0038/103-1240-0038",
      ],
      corpus_key="train",
      checkpoint=None,  # we set the checkpoint above via preload_from_files
      returnn_root=RETURNN_ROOT,
      returnn_python_exe=RETURNN_EXE,
      alias=f"models/{model_name}",
      hdf_targets=None,
      ref_alignment_hdf=LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
      ref_alignment_blank_idx=LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
      ref_alignment_vocab_path=LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
      seq_alias="ground-truth",
    )

  # train AED model using Tina's Conformer and features
  configs = []
  configs.append(dict(
    model_opts=config_24gb_v2["model_opts"],
    train_opts=config_24gb_v2["train_opts"],
    alias="models/v2_24gb_single-gpu/baseline",
  ))

  for config in configs:
    config_builder = AEDConfigBuilder(
      dataset=LIBRISPEECH_CORPUS,
      vocab_opts=BPE1K_OPTS,
      model_def=aed_model_def,
      get_model_func=aed_get_model,
      feature_dimension=config["model_opts"]["feature_dimension"],
      feature_extraction=config["model_opts"]["feature_extraction"],
      batch_size_factor=1,
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

  # CTC recog and rescoring
  configs = []
  configs.append(dict(
    model_opts=config_ctc_v2["model_opts"],
    checkpoint=external_torch_checkpoints["mohammad-aed-5.4"],
    checkpoint_alias="avg",
    model_name="mohammad-aed-5.4/aux-ctc",
  ))

  for config in configs:
    model_name = config["model_name"]

    config_builder = AEDConfigBuilder(
      dataset=LIBRISPEECH_CORPUS,
      vocab_opts=BPE10K_OPTS,
      model_def=ctc_model_def,
      get_model_func=ctc_get_model,
    )
    config_builder.config_dict.update(config["model_opts"])
    config_builder.config_dict["preload_from_files"] = {
      "pretrained_params": {
        "filename": config["checkpoint"],
        "ignore_missing": True,
        "custom_missing_load_func": load_missing_params_ctc,
    }}
    # config_builder.config_dict["ctc_prior_scale"] = 0.4

    recog_exp = RecogExperiment(
      alias=f"models/{model_name}",
      config_builder=config_builder,
      checkpoint=None,  # config["checkpoint"],
      checkpoint_alias=config["checkpoint_alias"],
      recog_opts={
        "recog_def": ctc_model_recog,
        "beam_size": 12,
        "dataset_opts": {
          "corpus_key": "dev-other",
        },
        "forward_step_func": _returnn_v2_forward_step,
        "forward_callback": _returnn_v2_get_forward_callback,
        "length_normalization_exponent": 1.0,
      },
      search_rqmt=dict(cpu=4)
    )
    recog_exp.run_eval()
    #
    # rescore(
    #   config_builder=config_builder,
    #   corpus_key="dev-other",
    #   checkpoint=None,  # we set the checkpoint above via preload_from_files
    #   returnn_root=RETURNN_ROOT,
    #   returnn_python_exe=RETURNN_EXE,
    #   alias=f"models/{model_name}",
    # )
