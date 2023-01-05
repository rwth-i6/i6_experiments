"""
Pipeline file for experiments with the standard CTC TTS model
"""
from sisyphus import tk
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.corpus import (
  CorpusReplaceOrthFromReferenceCorpus,
  MergeCorporaJob,
  SegmentCorpusJob,
)
from i6_experiments.common.setups.returnn_common.serialization import (
  Collection,
  ExternData,
  Import,
  Network,
  PythonEnlargeStackWorkaroundNonhashedCode,
)
from i6_experiments.common.datasets.librispeech import get_corpus_object_dict
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
  TTSTrainingDatasets,
  TTSForwardData,
  get_inference_dataset,
)
from i6_private.users.hilmes.tools.tts import VerifyCorpus, MultiJobCleanup
from i6_experiments.users.hilmes.experiments.librispeech.util.asr_evaluation import (
  asr_evaluation,
)
from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import (
  CalculateSpeakerPriorJob,
)
from typing import Optional, List, Dict, Union, Any
from i6_experiments.users.hilmes.tools.tts.analysis import (
  CalculateVarianceFromFeaturesJob,
  CalculateVarianceFromDurations,
)
from i6_core.report import GenerateReportStringJob, MailJob


def get_training_config(
  returnn_common_root: tk.Path,
  training_datasets: TTSTrainingDatasets,
  batch_size=18000,
  **kwargs,
):
  """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_model
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN training config
    """
  post_config = {
    "cleanup_old_models": True,
    "use_tensorflow": True,
    "tf_log_memory_usage": True,
    "stop_on_nonfinite_train_score": True,
    "log_batch_size": True,
    "debug_print_layer_output_template": True,
    "cache_size": "0",
  }
  config = {
    "behavior_version": 12,
    ############
    "optimizer": {"class": "adam", "epsilon": 1e-8},
    "accum_grad_multiple_step": round(18000 * 2 / batch_size),
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
    "batch_size": batch_size,
    "max_seq_length": {"audio_features": 1600},
    "max_seqs": 60,
  }

  extern_data = [datastream.as_nnet_constructor_data(key) for key, datastream in training_datasets.datastreams.items()]
  config["train"] = training_datasets.train.as_returnn_opts()
  config["dev"] = training_datasets.cv.as_returnn_opts()

  rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
  rc_extern_data = ExternData(extern_data=extern_data)
  rc_model = Import("i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.NARTTSModel")
  rc_construction_code = Import(
    "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.construct_network"
  )

  net_func_map = {
    "net_module": rc_model.object_name,
    "phoneme_data": "phonemes",
    "duration_data": "duration_data",
    "label_data": "speaker_labels",
    "audio_data": "audio_features",
    "time_dim": "phonemes_time",
    "label_time_dim": "speaker_labels_time",
    "speech_time_dim": "audio_features_time",
    "duration_time_dim": "duration_data_time",
  }

  if "use_pitch_pred" in kwargs.keys() and kwargs["use_pitch_pred"]:
    net_func_map["pitch"] = "pitch_data"
    net_func_map["pitch_time"] = "pitch_data_time"

  if "use_energy_pred" in kwargs.keys() and kwargs["use_energy_pred"]:
    net_func_map["energy"] = "energy_data"
    net_func_map["energy_time"] = "energy_data_time"

  rc_network = Network(
    net_func_name=rc_construction_code.object_name,
    net_func_map=net_func_map,
    net_kwargs={"training": True, **kwargs},
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
      "i6_experiments.users.hilmes.modules",
    },
  )

  returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])

  return returnn_config


def get_forward_config(
  returnn_common_root,
  forward_dataset: TTSForwardData,
  use_true_durations: bool = False,
  use_calculated_prior: bool = False,
  use_audio_data: bool = False,
  energy_cheat: bool = False,
  pitch_cheat: bool = False,
  batch_size: int = 4000,
  **kwargs,
):
  """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_model
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """

  config = {
    "behavior_version": 12,
    "forward_batch_size": batch_size,
    "max_seqs": 60,
    "forward_use_search": True,
    "target": "dec_output",
  }

  extern_data = [datastream.as_nnet_constructor_data(key) for key, datastream in forward_dataset.datastreams.items()]
  config["eval"] = forward_dataset.dataset.as_returnn_opts()

  rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
  rc_extern_data = ExternData(extern_data=extern_data)
  rc_model = Import("i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.NARTTSModel")
  rc_construction_code = Import(
    "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.construct_network"
  )

  net_func_map = {
    "net_module": rc_model.object_name,
    "phoneme_data": "phonemes",
    "duration_data": "duration_data" if use_true_durations else "None",
    "label_data": "speaker_labels",
    "audio_data": "audio_features" if use_audio_data else "None",
    "time_dim": "phonemes_time",
    "label_time_dim": "speaker_labels_time",
    "speech_time_dim": "audio_features_time" if use_audio_data else "None",
    "duration_time_dim": "duration_data_time" if use_true_durations else "None",
  }
  if use_calculated_prior:
    assert kwargs["use_vae"], "Need to also set use_vae in network kwargs"
    net_func_map["speaker_prior"] = "speaker_prior"
    net_func_map["prior_time"] = "speaker_prior_time"

  if pitch_cheat:
    assert kwargs["use_pitch_pred"], "Need to use pitch pred to use the true pitch"
    net_func_map["pitch"] = "pitch_data"
    net_func_map["pitch_time"] = "pitch_data_time"

  if energy_cheat:
    assert kwargs["use_energy_pred"], "Need to use energy pred to use the true energy"
    net_func_map["energy"] = "energy_data"
    net_func_map["energy_time"] = "energy_data_time"

  rc_network = Network(
    net_func_name=rc_construction_code.object_name,
    net_func_map=net_func_map,
    net_kwargs={
      "training": False,
      "use_true_durations": use_true_durations,
      **kwargs,
    },
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
      "i6_experiments.users.hilmes.modules",
    },
  )

  returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

  return returnn_config


def get_speaker_extraction_config(returnn_common_root, forward_dataset: TTSForwardData, **kwargs):
  """

    :param returnn_common_root:
    :param forward_dataset:
    :param kwargs:
    :return:
    """
  config = {
    "behavior_version": 12,
    "forward_batch_size": 18000,
    "max_seqs": 60,
    "forward_use_search": True,
    "target": "dec_output",
  }
  extern_data = [datastream.as_nnet_constructor_data(key) for key, datastream in forward_dataset.datastreams.items()]
  config["eval"] = forward_dataset.dataset.as_returnn_opts()

  rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
  rc_extern_data = ExternData(extern_data=extern_data)
  rc_model = Import("i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.NARTTSModel")
  rc_construction_code = Import(
    "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.construct_network"
  )

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
    net_kwargs={"training": False, "dump_speaker_embeddings": True, **kwargs},
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
      "i6_experiments.users.hilmes.modules",
    },
  )

  returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

  return returnn_config


def get_vae_prior_config(returnn_common_root, forward_dataset: TTSForwardData, **kwargs):
  """

    :param returnn_common_root:
    :param forward_dataset:
    :param kwargs:
    :return:
    """
  config = {
    "behavior_version": 12,
    "forward_batch_size": 18000,
    "max_seqs": 60,
    "forward_use_search": True,
    "target": "dec_output",
  }
  extern_data = [datastream.as_nnet_constructor_data(key) for key, datastream in forward_dataset.datastreams.items()]
  config["eval"] = forward_dataset.dataset.as_returnn_opts()

  rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
  rc_extern_data = ExternData(extern_data=extern_data)
  rc_model = Import("i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.NARTTSModel")
  rc_construction_code = Import(
    "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.construct_network"
  )

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
    net_kwargs={"training": False, "dump_vae": True, "use_vae": True, **kwargs},
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
      "i6_experiments.users.hilmes.modules",
    },
  )

  returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

  return returnn_config


def tts_training(config, returnn_exe, returnn_root, prefix, num_epochs=200, mem=32):
  """

    :param config:
    :param returnn_exe:
    :param returnn_root:
    :param prefix:
    :param num_epochs:
    :return:
    """
  train_job = ReturnnTrainingJob(
    config,
    log_verbosity=5,
    num_epochs=num_epochs,
    time_rqmt=120,
    mem_rqmt=mem,
    cpu_rqmt=4,
    returnn_python_exe=returnn_exe,
    returnn_root=returnn_root,
  )
  train_job.add_alias(prefix + "/training")
  tk.register_output(prefix + "/training.models", train_job.out_model_dir)

  return train_job


def tts_forward(checkpoint, config, returnn_exe, returnn_root, prefix, hdf_outputs=None):
  """

    :param checkpoint:
    :param config:
    :param returnn_exe:
    :param returnn_root:
    :param prefix:
    :return:
    """
  if not hdf_outputs:
    hdf_outputs = []
  forward_job = ReturnnForwardJob(
    model_checkpoint=checkpoint,
    returnn_config=config,
    hdf_outputs=hdf_outputs,
    returnn_python_exe=returnn_exe,
    returnn_root=returnn_root,
  )
  forward_job.add_alias(prefix + "/forward")

  return forward_job


def gl_swer(name, vocoder, checkpoint, config, returnn_root, returnn_exe):
  """
    Griffin Lin synthetic WER
    :param name:
    :param vocoder:
    :param checkpoint:
    :param config:
    :param returnn_root:
    :param returnn_exe:
    :return:
    """
  forward_job = tts_forward(
    checkpoint=checkpoint,
    config=config,
    returnn_root=returnn_root,
    returnn_exe=returnn_exe,
    prefix=name,
  )
  forward_hdf = forward_job.out_hdf_files["output.hdf"]
  forward_vocoded, vocoder_forward_job = vocoder.vocode(forward_hdf, iterations=30, cleanup=True, name=name)

  verification = VerifyCorpus(forward_vocoded).out
  cleanup = MultiJobCleanup([forward_job, vocoder_forward_job], verification, output_only=True)
  tk.register_output(name + "/ctc_model/cleanup.log", cleanup.out)

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
  # asr_evaluation(
  #    config_file=librispeech_trafo,
  #    corpus=cv_synth_corpus,
  #    output_path=name,
  #    returnn_root=returnn_root,
  #    returnn_python_exe=returnn_exe,
  # )


def synthesize_with_splits(
  name,
  reference_corpus: tk.Path,
  corpus_name: str,
  job_splits: int,
  datasets: TTSForwardData,
  returnn_root,
  returnn_exe,
  returnn_common_root,
  checkpoint,
  vocoder,
  batch_size: int = 4000,
  reconstruction_norm: bool = True,
  segments: Optional[Dict] = None,
  **tts_model_kwargs,
):
  """

    :param name:
    :param reference_corpus: Needs to be the matching corpus for datasets
    :param corpus_name: Name of the corpus for the ReplaceOrthJob
    :param job_splits: number of splits performed
    :param datasets: datasets including datastream supposed to hold the audio data in .train
    :param returnn_root:
    :param returnn_exe:
    :param returnn_common_root:
    :param checkpoint:
    :param vocoder:
    :param tts_model_kwargs: kwargs to be passed to the tts model for synthesis
    :return:
    """
  if segments is None:
    forward_segments = SegmentCorpusJob(reference_corpus, job_splits).out_single_segment_files
  else:
    job_splits = len(segments)
    forward_segments = segments

  verifications = []
  output_corpora = []
  for i in range(job_splits):
    split_name = name + "/synth_corpus/part_%i" % i
    forward_config = get_forward_config(
      returnn_common_root=returnn_common_root,
      forward_dataset=datasets,
      batch_size=batch_size,
      **tts_model_kwargs,
    )
    forward_config.config["eval"]["datasets"]["audio"]["segment_file"] = forward_segments[i + 1]

    last_forward_job = ReturnnForwardJob(
      model_checkpoint=checkpoint,
      returnn_config=forward_config,
      hdf_outputs=[],
      returnn_python_exe=returnn_exe,
      returnn_root=returnn_root,
    )
    last_forward_job.set_keep_value(20)
    last_forward_job.add_alias(split_name + "/forward")
    forward_hdf = last_forward_job.out_hdf_files["output.hdf"]
    tk.register_output(split_name + "/foward.hdf", forward_hdf)

    forward_vocoded, vocoder_forward_job = vocoder.vocode(
      forward_hdf,
      iterations=30,
      cleanup=True,
      name=split_name,
      recon_norm=reconstruction_norm,
    )
    tk.register_output(split_name + "/synthesized_corpus.xml.gz", forward_vocoded)
    output_corpora.append(forward_vocoded)
    verification = VerifyCorpus(forward_vocoded).out
    verifications.append(verification)

    cleanup = MultiJobCleanup([last_forward_job, vocoder_forward_job], verification, output_only=True)
    tk.register_output(split_name + "/cleanup/cleanup.log", cleanup.out)

  from i6_core.corpus.transform import MergeStrategy

  merge_job = MergeCorporaJob(output_corpora, corpus_name, merge_strategy=MergeStrategy.FLAT)
  for verfication in verifications:
    merge_job.add_input(verfication)

  cv_synth_corpus = CorpusReplaceOrthFromReferenceCorpus(
    bliss_corpus=merge_job.out_merged_corpus,
    reference_bliss_corpus=reference_corpus,
  ).out_corpus

  tk.register_output(name + "/synth_corpus/synthesized_corpus.xml.gz", cv_synth_corpus)
  return cv_synth_corpus


def build_speaker_embedding_dataset(
  returnn_common_root,
  returnn_exe,
  returnn_root,
  datasets,
  prefix,
  train_job,
  epoch=200,
  speaker_embedding_size=256,
  **kwargs,
):
  """

    :param returnn_common_root:
    :param returnn_exe:
    :param returnn_root:
    :param datasets:
    :param prefix:
    :param train_job:
    :return:
    """
  extraction_config = get_speaker_extraction_config(
    speaker_embedding_size=speaker_embedding_size,
    training=True,
    returnn_common_root=returnn_common_root,
    forward_dataset=TTSForwardData(
      dataset=datasets.cv,
      datastreams=datasets.datastreams,  # cv is fine here cause we assume all speakers in cv
    ),
    **kwargs
  )
  extraction_job = tts_forward(
    checkpoint=train_job.out_checkpoints[epoch],
    config=extraction_config,
    returnn_exe=returnn_exe,
    returnn_root=returnn_root,
    prefix=prefix + "/extract_speak_emb",
  )
  speaker_embedding_hdf = extraction_job.out_default_hdf
  return speaker_embedding_hdf


def build_vae_speaker_prior_dataset(
  returnn_common_root,
  returnn_exe,
  returnn_root,
  dataset,
  datastreams,
  prefix,
  train_job,
  corpus,
  epoch=200,
  speaker_embedding_size=256,
  **forward_kwargs,
):
  """

    :param returnn_common_root:
    :param returnn_exe:
    :param returnn_root:
    :param datasets:
    :param prefix:
    :param train_job:
    :param corpus
    :return:
    """
  vae_extraction_config = get_vae_prior_config(
    speaker_embedding_size=speaker_embedding_size,
    training=True,
    returnn_common_root=returnn_common_root,
    forward_dataset=TTSForwardData(dataset=dataset, datastreams=datastreams),
    **forward_kwargs,
  )
  vae_extraction_job = tts_forward(
    checkpoint=train_job.out_checkpoints[epoch],
    config=vae_extraction_config,
    returnn_exe=returnn_exe,
    returnn_root=returnn_root,
    prefix=prefix + "/extract_vae_prior",
  )
  vae_prior_hdf = vae_extraction_job.out_default_hdf
  priors = CalculateSpeakerPriorJob(vae_hdf=vae_prior_hdf, corpus_file=corpus).out_prior
  tk.register_output(prefix + "/calculated_priors", priors)
  return priors


def var_rep_format(report) -> str:
  out = []
  out.append(
    f"""
        Name: {report["name"]}
                  Full / No Silence / Weighted / Weighted no Silence
        Features:{str(report["feat_var"])}, {str(report["feat_var_no_sil"])}, {str(report["feat_var_weight"])}, {str(report["feat_var_weight_no_sil"])}
        Durations:{str(report["dur_var"])}, {str(report["dur_var_no_sil"])}, {str(report["dur_var_weight"])}, {str(report["dur_var_weight_no_sil"])}
        Excel: {report["feat_var"].get():.{4}f}/{report["feat_var_weight"].get():.{4}f}, {report["feat_var_no_sil"].get():.{4}f}/{report["feat_var_weight_no_sil"].get():.{4}f}, {report["dur_var"].get():.{4}f}/{report["dur_var_weight"].get():.{4}f}, {report["dur_var_no_sil"].get():.{4}f}/{report["dur_var_weight_no_sil"].get():.{4}f} 
"""
  )
  return "\n".join(out)


def calculate_feature_variance(
  train_job,
  corpus,
  returnn_root,
  returnn_common_root,
  returnn_exe,
  prefix: str,
  training_datasets,
  durations=None,
  speaker_embedding_hdf: Optional = None,
  **kwargs,
):
  if "speaker_embedding_size" in kwargs:
    speaker_embedding_size = kwargs["speaker_embedding_size"]
  else:
    speaker_embedding_size = 256
  if not speaker_embedding_hdf:
    speaker_embedding_hdf = build_speaker_embedding_dataset(
      returnn_common_root=returnn_common_root,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      datasets=training_datasets,
      prefix=prefix,
      train_job=train_job,
      speaker_embedding_size=speaker_embedding_size
    )

  synth_dataset = get_inference_dataset(
    corpus,
    returnn_root=returnn_root,
    returnn_exe=returnn_exe,
    datastreams=training_datasets.datastreams,
    speaker_embedding_hdf=speaker_embedding_hdf,
    durations=durations,
    process_corpus=False,
    shuffle_info=False,
    alias=prefix,
    speaker_embedding_size=speaker_embedding_size
  )
  forward_config = get_forward_config(
    returnn_common_root=returnn_common_root,
    forward_dataset=synth_dataset,
    dump_round_durations=True,
    dump_durations_to_hdf=True,
    **kwargs,
  )
  forward_job = tts_forward(
    checkpoint=train_job.out_checkpoints[200],
    config=forward_config,
    prefix=prefix + "/cov_analysis/full/forward",
    returnn_root=returnn_root,
    returnn_exe=returnn_exe,
    hdf_outputs=["durations.hdf"] if durations is None else None,
  )
  forward_hdf = forward_job.out_hdf_files["output.hdf"]
  tk.register_output(prefix + "/cov_analysis/full/features.hdf", forward_hdf)
  if durations:
    durations_hdf = durations
    tk.register_output(prefix + "/cov_analysis/full/durations.hdf", durations_hdf)
  else:
    durations_hdf = forward_job.out_hdf_files["durations.hdf"]
    tk.register_output(prefix + "/cov_analysis/full/durations.hdf", durations_hdf)
  feat_var_job = CalculateVarianceFromFeaturesJob(feature_hdf=forward_hdf, duration_hdf=durations_hdf, bliss=corpus)
  feat_var_job.add_alias(prefix + "/cov_analysis/full/calculate_feat_var_job")
  tk.register_output(prefix + "/cov_analysis/full/feat_variance", feat_var_job.out_variance)
  dur_var_job = CalculateVarianceFromDurations(duration_hdf=durations_hdf, bliss=corpus)
  dur_var_job.add_alias(prefix + "/cov_analysis/full/calculate_dur_var_job")
  tk.register_output(prefix + "/cov_analysis/full/dur_variance", dur_var_job.out_variance)
  report_dict = {
    "name": prefix,
    "feat_var": feat_var_job.out_variance,
    "feat_var_no_sil": feat_var_job.out_variance_no_sil,
    "feat_var_weight": feat_var_job.out_weight_variance,
    "feat_var_weight_no_sil": feat_var_job.out_weight_variance_no_sil,
    "dur_var": dur_var_job.out_variance,
    "dur_var_no_sil": dur_var_job.out_variance_no_sil,
    "dur_var_weight": dur_var_job.out_weight_variance,
    "dur_var_weight_no_sil": dur_var_job.out_weight_variance_no_sil,
  }
  content = GenerateReportStringJob(report_values=report_dict, report_template=var_rep_format).out_report
  report = MailJob(subject=prefix, result=content, send_contents=True)
  tk.register_output(f"reports/{prefix}", report.out_status)
  cleanup = MultiJobCleanup([forward_job], report.out_status, output_only=True)
  tk.register_output(prefix + "/cleanup/cleanup.log", cleanup.out)

def get_average_checkpoint_v2(training_job, returnn_exe, returnn_root, num_average: int = 4):
  """
  get an averaged checkpoint using n models

  :param training_job:
  :param num_average:
  :return:
  """
  from i6_core.returnn.training import GetBestTFCheckpointJob, AverageTFCheckpointsJob
  epochs = []
  for i in range(num_average):
    best_checkpoint_job = GetBestTFCheckpointJob(
      training_job.out_model_dir,
      training_job.out_learning_rates,
      key="dev_score_nartts_model_mean_absolute_difference_reduce",
      index=i)
    epochs.append(best_checkpoint_job.out_epoch)
  average_checkpoint_job = AverageTFCheckpointsJob(training_job.out_model_dir, epochs=epochs,
    returnn_python_exe=returnn_exe, returnn_root=returnn_root)
  return average_checkpoint_job.out_checkpoint