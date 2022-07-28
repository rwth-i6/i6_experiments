"""
Pipeline file for experiments with the standard CTC TTS model
"""
from sisyphus import tk
from returnn_common.nn import min_returnn_behavior_version
from copy import deepcopy
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.corpus import CorpusReplaceOrthFromReferenceCorpus, MergeCorporaJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_experiments.common.setups.returnn_common.serialization import (
  Collection,
  ExternData,
  Import,
  Network,
  NonhashedCode,
  PythonEnlargeStackWorkaroundCode,
)
from i6_experiments.common.datasets.librispeech import get_corpus_object_dict
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
  TTSTrainingDatasets,
  TTSEvalDataset,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.default_vocoder import get_default_vocoder
from i6_private.users.hilmes.tools.tts import VerifyCorpus, MultiJobCleanup
from i6_private.users.hilmes.util.asr_evaluation import asr_evaluation


def get_training_config(
  returnn_common_root: tk.Path, training_datasets: TTSTrainingDatasets, gauss_up: bool = False, **kwargs
):
  """
  Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_model
  :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
  :param training_datasets: datasets for training
  :param gauss_up: Whether to enable gaussian upsampling in the network
  :param kwargs: arguments to be passed to the network construction
  :return: RETURNN training config
  """
  post_config = {
    "cleanup_old_models": True,
    "use_tensorflow": True,
    "tf_log_memory_usage": True,
    "stop_on_nonfinite_train_score": False,
    "log_batch_size": True,
    "debug_print_layer_output_template": True,
    "cache_size": "0",
  }
  config = {
    "behavior_version": min_returnn_behavior_version,
    ############
    "optimizer": {"class": "adam", "epsilon": 1e-8},
    "accum_grad_multiple_step": 2,
    "gradient_clip": 1,
    "gradient_noise": 0,
    "learning_rate_control": "newbob_multi_epoch",
    "learning_rate_control_min_num_epochs_per_new_lr": 5,
    "learning_rate_control_relative_error_relative_lr": True,
    "learning_rates": [0.001],
    "use_learning_rate_control_always": True,
    "learning_rate_control_error_measure": "dev_score_dec_output",
    ############
    "newbob_learning_rate_decay": 0.9,
    "newbob_multi_num_epochs": 5,
    "newbob_multi_update_interval": 1,
    "newbob_relative_error_threshold": 0,
    #############
    "batch_size": 18000,
    "max_seq_length": {"audio_features": 1600},
    "max_seqs": 60,
  }

  extern_data = [datastream.as_nnet_constructor_data(key) for key, datastream in training_datasets.datastreams.items()]
  config["train"] = training_datasets.train.as_returnn_opts()
  config["dev"] = training_datasets.cv.as_returnn_opts()

  rc_recursionlimit = PythonEnlargeStackWorkaroundCode
  rc_extern_data = ExternData(extern_data=extern_data)
  rc_model = Import("i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.NARTTSModel")
  rc_construction_code = Import("i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.construct_network")

  rc_network = Network(
    net_func_name=rc_construction_code.object_name,
    net_func_map={
      "net_module": rc_model.object_name,
      "phoneme_data": "phonemes",
      "duration_data": "duration_data",
      "label_data": "speaker_labels",
      "audio_data": "audio_features",
      "time_dim": "phonemes_time",
      "label_time_dim": "speaker_labels_time",
      "speech_time_dim": "audio_features_time",
      "duration_time_dim": "duration_data_time",
    },
    net_kwargs={"training": True, "gauss_up": gauss_up, **kwargs},
  )

  serializer = Collection(
    serializer_objects=[
      rc_recursionlimit,
      rc_extern_data,
      rc_model,
      rc_construction_code,
      rc_network,
    ],
    returnn_common_root=returnn_common_root,
    make_local_package_copy=True,
    packages={
      "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks",
      "i6_experiments.users.hilmes.modules"}
  )

  returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])

  return returnn_config


def get_forward_config(returnn_common_root, datasets, gauss_up: bool = False, use_true_durations: bool = False, **kwargs):
  """
  Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_model
  :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
  :param datasets: datasets for training
  :param gauss_up: whether to use gaussian upsampling
  :param kwargs: arguments to be passed to the network construction
  :return: RETURNN forward config
  """
  from copy import deepcopy
  eval_datasets = deepcopy(datasets)
  eval_datasets.datastreams["duration_data"].available_for_inference = use_true_durations

  config = {
    "behavior_version": min_returnn_behavior_version,
    "forward_batch_size": 18000,
    "max_seqs": 60,
    "forward_use_search": True,
    "target": "dec_output",
  }
  extern_data = [datastream.as_nnet_constructor_data(key) for key, datastream in eval_datasets.datastreams.items()]
  config["eval"] = datasets.cv.as_returnn_opts()

  rc_recursionlimit = PythonEnlargeStackWorkaroundCode
  rc_extern_data = ExternData(extern_data=extern_data)
  rc_model = Import("i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.NARTTSModel")
  rc_construction_code = Import("i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.construct_network")

  rc_network = Network(
    net_func_name=rc_construction_code.object_name,
    net_func_map={
      "net_module": rc_model.object_name,
      "phoneme_data": "phonemes",
      "duration_data": "duration_data",
      "label_data": "speaker_labels",
      "audio_data": "audio_features",
      "time_dim": "phonemes_time",
      "label_time_dim": "speaker_labels_time",
      "speech_time_dim": "audio_features_time",
      "duration_time_dim": "duration_data_time",
    },
    net_kwargs={"training": False, "use_true_durations": use_true_durations, **kwargs},
  )

  serializer = Collection(
    serializer_objects=[
      rc_recursionlimit,
      rc_extern_data,
      rc_model,
      rc_construction_code,
      rc_network,
    ],
    returnn_common_root=returnn_common_root,
    make_local_package_copy=True,
    packages={
      "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks",
      "i6_experiments.users.hilmes.modules"}
  )

  returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

  return returnn_config


def gl_swer_on_forward_hdf(name, forward_hdf, vocoder_data, forward_job, returnn_root, returnn_exe):
  """

  :param name:
  :param forward_hdf:
  :param vocoder_data:
  :param forward_job:
  :param returnn_root:
  :param returnn_exe:
  :return:
  """
  default_vocoder = get_default_vocoder(name=name, corpus_data=vocoder_data)
  default_vocoder.train(num_epochs=100, time_rqmt=36, mem_rqmt=12)
  forward_vocoded, vocoder_forward_job = default_vocoder.vocode(forward_hdf, iterations=30, cleanup=True)
  verification = VerifyCorpus(forward_vocoded).out
  cleanup = MultiJobCleanup([forward_job, vocoder_forward_job], verification, output_only=True)
  tk.register_output(name + "/ctc_model" + "/".join(["cleanup", name]), cleanup.out)

  corpus_object_dict = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")
  cv_synth_corpus_job = CorpusReplaceOrthFromReferenceCorpus(
    forward_vocoded, corpus_object_dict["train-clean-100"].corpus_file
  )
  cv_synth_corpus_job.add_input(verification)
  cv_synth_corpus = cv_synth_corpus_job.out_corpus
  librispeech_trafo = tk.Path(
    "/u/rossenbach/experiments/librispeech_tts/config/evaluation/asr/pretrained_configs/trafo.specaug4.12l.ffdim4."
    "pretrain3.natctc_recognize_pretrained.config"
  )
  asr_evaluation(
    config_file=librispeech_trafo,
    corpus=cv_synth_corpus,
    output_path=name + "/ctc_model",
    returnn_root=returnn_root,
    returnn_python_exe=returnn_exe,
  )

