from sisyphus import tk
from returnn_common.nn import min_returnn_behavior_version
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob
from i6_experiments.common.setups.returnn_common.serialization import (
  Collection,
  ExternData,
  Import,
  Network,
  PythonEnlargeStackWorkaroundCode,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datasets import (
  GenericDataset,
)


def get_training_config(returnn_common_root, training_datasets, **kwargs):
  """
  Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_aligner
  :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
  :param training_datasets: datasets for training
  :param kwargs: arguments to be passed to the network construction
  :return: RETURNN training config
  """

  # changing these does not change the hash
  post_config = {
    "cleanup_old_models": True,
    "use_tensorflow": True,
    "tf_log_memory_usage": True,
    "stop_on_nonfinite_train_score": True,  # this might break now with True
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
    "learning_rate_control_error_measure": "dev_score_reconstruction_output",
    ############
    "newbob_learning_rate_decay": 0.9,
    "newbob_multi_num_epochs": 5,
    "newbob_multi_update_interval": 1,
    "newbob_relative_error_threshold": 0,
    #############
    "batch_size": 28000,
    "max_seq_length": {"audio_features": 1600},
    "max_seqs": 200,
  }

  extern_data = [
    datastream.as_nnet_constructor_data(key)
    for key, datastream in training_datasets.datastreams.items()
  ]

  config["train"] = training_datasets.train.as_returnn_opts()
  config["dev"] = training_datasets.cv.as_returnn_opts()

  rc_recursionlimit = PythonEnlargeStackWorkaroundCode
  rc_extern_data = ExternData(extern_data=extern_data)
  rc_model = Import(
    "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.ctc_aligner.CTCAligner"
  )
  rc_construction_code = Import(
    "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.ctc_aligner.construct_network"
  )

  rc_network = Network(
    net_func_name=rc_construction_code.object_name,
    net_func_map={
      "net_module": rc_model.object_name,
      "audio_data": "audio_features",
      "label_data": "speaker_labels",
      "phoneme_data": "phonemes",
      "audio_time_dim": "audio_features_time",
      "label_time_dim": "speaker_labels_time",
      "phoneme_time_dim": "phonemes_time",
    },
    net_kwargs={**kwargs},
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
      "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks"
    },
  )

  returnn_config = ReturnnConfig(
    config=config, post_config=post_config, python_epilog=[serializer]
  )

  return returnn_config


def get_forward_config(
  returnn_common_root, forward_dataset: GenericDataset, datastreams, **kwargs
):
  """
  Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
  :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
  :param training_datasets: datasets for training
  :param kwargs: arguments to be passed to the network construction
  :return: RETURNN forward config
  """
  config = {
    "behavior_version": min_returnn_behavior_version,
    "forward_batch_size": 28000,
    "max_seq_length": {"audio_features": 1000},
    "max_seqs": 200,
    "forward_use_search": True,
    "target": "extract_alignment",
  }
  config["eval"] = forward_dataset.as_returnn_opts()
  extern_data = [
    datastream.as_nnet_constructor_data(key) for key, datastream in datastreams.items()
  ]

  rc_recursionlimit = PythonEnlargeStackWorkaroundCode
  rc_extern_data = ExternData(extern_data=extern_data)
  rc_model = Import(
    "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.ctc_aligner.CTCAligner"
  )
  rc_construction_code = Import(
    "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.ctc_aligner.construct_network"
  )

  rc_network = Network(
    net_func_name=rc_construction_code.object_name,
    net_func_map={
      "net_module": rc_model.object_name,
      "audio_data": "audio_features",
      "label_data": "speaker_labels",
      "phoneme_data": "phonemes",
      "audio_time_dim": "audio_features_time",
      "label_time_dim": "speaker_labels_time",
      "phoneme_time_dim": "phonemes_time",
    },
    net_kwargs={"training": False, **kwargs},
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
      "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks"
    },
  )

  returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

  return returnn_config


def ctc_training(config, returnn_exe, returnn_root, prefix, num_epochs=100):

  train_job = ReturnnTrainingJob(
    config,
    log_verbosity=5,
    num_epochs=num_epochs,
    time_rqmt=100,
    mem_rqmt=16,
    cpu_rqmt=4,
    returnn_python_exe=returnn_exe,
    returnn_root=returnn_root,
  )
  train_job.add_alias(prefix + "/training")
  tk.register_output(prefix + "/training.models", train_job.out_model_dir)

  return train_job


def ctc_forward(checkpoint, config, returnn_exe, returnn_root, prefix):
  last_forward_job = ReturnnForwardJob(
    model_checkpoint=checkpoint,
    returnn_config=config,
    hdf_outputs=[],
    returnn_python_exe=returnn_exe,
    returnn_root=returnn_root,
  )
  durations_hdf = last_forward_job.out_hdf_files["output.hdf"]
  tk.register_output(prefix + "/training.durations", durations_hdf)

  return durations_hdf
