import os
import numpy as np
from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob

from i6_core.returnn.forward import ReturnnForwardJob
from i6_experiments.users.rossenbach.returnn.training import GetBestCheckpointJob
from i6_experiments.users.rossenbach.setups.returnn_standalone.data.datasets import (
  OggZipDataset,
  MetaDataset,
  HDFDataset,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import (
  AudioFeatureDatastream,
  ReturnnAudioFeatureOptions,
  LinearFilterbankOptions,
)
from .vocoder_data import get_vocoder_data

from ..default_tools import RETURNN_EXE, RETURNN_RC_ROOT

post_config_template = {
  "cleanup_old_models": True,
  "use_tensorflow": True,
  "tf_log_memory_usage": True,
  "stop_on_nonfinite_train_score": False,
  "log_batch_size": True,
  "debug_print_layer_output_template": True,
  "cache_size": "0",
}


class LJSpeechMiniGLVocoder:
  """
  This is a quick implementation of a "Griffin & Lim" Vocoder, which
  consists of a network that converts log-mel-features into linear features for the G&L algorithm

  The network is completely static for now, but this class can serve as a template to have a more generic
  implementation in the future
  """

  def __init__(
    self,
    name: str,
    alias_path: str,
    audio_datastream: AudioFeatureDatastream,
    zip_dataset: tk.Path,
    train_segments: tk.Path,
    dev_segments: tk.Path,
    returnn_gpu_exe: tk.Path,
    returnn_root: tk.Path,
    train_dataset_key: str = "train-clean-100-tts-train",
    dev_dataset_key: str = "train-clean-100-tts-dev",
    model_size: int = 512,
  ):
    """
    :param name: A name for this specific vocoder
    :param alias_path: alias path for the current TTS Experiment
    :param audio_datastream: the (probably log-mel) audio settings (linear settings will be inferred from this)
    :param datagroup: the dataset used for training
    :param train_dataset_key: datagroup key for training
    :param dev_dataset_key: datagroup key for cross validation
    :param model_size:
    """
    self._alias_path = os.path.join(alias_path, name)
    self._zip_dataset = zip_dataset
    self._train_segments = train_segments
    self._dev_segments = dev_segments
    self._train_dataset_key = train_dataset_key
    self._dev_dataset_key = dev_dataset_key
    self._model_size = model_size
    self._source_audio_opts = audio_datastream
    assert (
      audio_datastream.options.sample_rate is not None
    ), "please specify a sample_rate in the AudioFeatureDatastream"
    #options_type = type(audio_datastream.options.feature_options)
    #audio_datastream.options.feature_options = options_type(**params)
    self._sample_rate = audio_datastream.options.sample_rate
    self._linear_size = 2 ** int(
      np.ceil(np.log2(self._sample_rate * audio_datastream.options.window_len)) - 1
    )
    self._target_audio_opts = self._get_target_datastream(
      audio_datastream, self._linear_size
    )

    self._returnn_gpu_exe = returnn_gpu_exe
    self._returnn_root = returnn_root

  def _build_network(self):
    """
    :return: a fixed network for now
    :rtype: dict[str, Any]
    """
    network = {
      "transform": {
        "class": "linear",
        "activation": "relu",
        "from": ["data"],
        "n_out": self._model_size,
      },
      "lstm0_bw": {
        "class": "rec",
        "direction": -1,
        "from": ["data"],
        "n_out": self._model_size,
        "unit": "nativelstm2",
      },
      "lstm0_fw": {
        "class": "rec",
        "direction": 1,
        "from": ["data"],
        "n_out": self._model_size,
        "unit": "nativelstm2",
      },
      "lstm1_bw": {
        "class": "rec",
        "direction": -1,
        "dropout": 0.2,
        "from": ["lstm0_fw", "lstm0_bw"],
        "n_out": self._model_size,
        "unit": "nativelstm2",
      },
      "lstm1_fw": {
        "class": "rec",
        "direction": 1,
        "dropout": 0.2,
        "from": ["lstm0_fw", "lstm0_bw"],
        "n_out": self._model_size,
        "unit": "nativelstm2",
      },
      "combine_0": {"class": "copy", "from": ["lstm0_fw", "lstm0_bw"]},
      "combine_1": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "add": {"class": "combine", "kind": "add", "from": ["combine_0", "combine_1"]},
      "log_output": {
        "class": "linear",
        "from": ["add", "transform"],
        "loss": "mse",
        "target": "layer:log_target",
        "n_out": self._linear_size,
      },
      "output": {
        "class": "eval",
        "eval": "tf.math.pow(10.0, source(0))",
        "from": "log_output",
        "loss": "mse",
        "target": "data_target",
        "loss_scale": 0.05,
      },
      "log_target": {
        "class": "eval",
        "eval": "tf.math.log(source(0) + 1.0e-10)/tf.math.log(10.0)",
        "from": "data:data_target",
      },
    }
    return network

  @classmethod
  def _get_target_datastream(
    self, source_audio_opts: AudioFeatureDatastream, linear_size: int
  ):
    """
    This is a classmethod so that it can be used from outside if needed

    :param source_audio_opts: the (probably log-mel) audio settings (linear settings will be inferred from this)
    :param linear_size: the linear spectrogram dimension (constraint 2*linear_size >= sample_rate*window_len)
    :return:
    """
    options = ReturnnAudioFeatureOptions(
      features="linear_spectrogram",
      num_feature_filters=linear_size,
      window_len=source_audio_opts.options.window_len,
      step_len=source_audio_opts.options.step_len,
      peak_normalization=False,
      preemphasis=source_audio_opts.options.preemphasis,
      sample_rate=source_audio_opts.options.sample_rate,
      feature_options=LinearFilterbankOptions(center=source_audio_opts.options.feature_options.center),
    )
    target_audio = AudioFeatureDatastream(
      available_for_inference=False, options=options
    )
    return target_audio

  def _get_training_dataset(self):
    source_train_dataset = OggZipDataset(
      path=self._zip_dataset,
      audio_opts=self._source_audio_opts.as_returnn_audio_opts(),
      segment_file=self._train_segments,
      partition_epoch=4,
      seq_ordering="laplace:.1000",
    )

    target_train_dataset = OggZipDataset(
      path=self._zip_dataset,
      audio_opts=self._target_audio_opts.as_returnn_audio_opts(),
      segment_file=self._train_segments,
    )

    train_dataset = MetaDataset(
      data_map={"data": ("source", "data"), "data_target": ("target", "data")},
      datasets={"source": source_train_dataset, "target": target_train_dataset},
      seq_order_control_dataset="source",
    )
    return train_dataset.as_returnn_opts()

  def _get_dev_dataset(self):
    source_dev_dataset = OggZipDataset(
      path=self._zip_dataset,
      audio_opts=self._source_audio_opts.as_returnn_audio_opts(),
      segment_file=self._dev_segments,
      partition_epoch=1,
      seq_ordering="sorted_reverse",
    )

    target_dev_dataset = OggZipDataset(
      path=self._zip_dataset,
      audio_opts=self._target_audio_opts.as_returnn_audio_opts(),
      segment_file=self._dev_segments,
    )

    dev_dataset = MetaDataset(
      data_map={"data": ("source", "data"), "data_target": ("target", "data")},
      datasets={"source": source_dev_dataset, "target": target_dev_dataset},
      seq_order_control_dataset="source",
    )
    return dev_dataset.as_returnn_opts()

  def _get_forward_dataset(self, hdf_input):
    source_dev_dataset = HDFDataset(
      files=hdf_input, partition_epoch=1, seq_ordering="sorted_reverse"
    )

    dev_dataset = MetaDataset(
      data_map={"data": ("source", "data")},
      datasets={"source": source_dev_dataset},
      seq_order_control_dataset="source",
    )
    return dev_dataset.as_returnn_opts()

  def build_config(self):
    config = {
      "behavior_version": 1,
      ############
      "optimizer": {"class": "adam", "epsilon": 1e-8},
      "gradient_clip": 5,
      "gradient_noise": 0,
      "learning_rate_control": "newbob_multi_epoch",
      "learning_rate_control_min_num_epochs_per_new_lr": 5,
      "learning_rate_control_relative_error_relative_lr": True,
      "learning_rate_control_error_measure": "dev_score_log_output",
      "learning_rates": [0.001],
      "use_learning_rate_control_always": True,
      ############
      "newbob_learning_rate_decay": 0.8,
      "newbob_multi_num_epochs": 3,
      "newbob_multi_update_interval": 1,
      "newbob_relative_error_threshold": 0,
      #############
      "batch_size": 20000,
      # "batch_size": 1,
      #############
      "network": self._build_network(),
      "train": self._get_training_dataset(),
      "dev": self._get_dev_dataset(),
      "extern_data": {
        "data": self._source_audio_opts.as_returnn_extern_data_opts(),
        "data_target": self._target_audio_opts.as_returnn_extern_data_opts(),
      },
    }

    self.config = ReturnnConfig(config=config, post_config=post_config_template.copy())

  def build_forward_config(self, hdf_input):
    forward_datset = self._get_forward_dataset(hdf_input)
    config = {
      "behavior_version": 1,
      "forward_batch_size": 160000,  # 800 frames * 200 sequences can easily fit
      "network": self._build_network(),
      "eval": forward_datset,
      "target": "data_target",
      "extern_data": {
        "data": self._source_audio_opts.as_returnn_extern_data_opts(),
        "data_target": self._target_audio_opts.as_returnn_extern_data_opts(),
      },
    }
    return ReturnnConfig(config=config, post_config=post_config_template.copy())

  def train(self, num_epochs, time_rqmt, mem_rqmt, cpu_rqmt=4):
    self.num_epochs = num_epochs
    self.train_job = ReturnnTrainingJob(
      self.config,
      log_verbosity=5,
      num_epochs=num_epochs,
      time_rqmt=time_rqmt,
      mem_rqmt=mem_rqmt,
      cpu_rqmt=cpu_rqmt,
      returnn_python_exe=self._returnn_gpu_exe,
      returnn_root=self._returnn_root,
    )
    self.train_job.add_alias(os.path.join(self._alias_path, "training"))

    self.best_checkpoint = GetBestCheckpointJob(
      self.train_job.out_model_dir,
      self.train_job.out_learning_rates,
      key="dev_score_log_output",
    ).out_checkpoint

    tk.register_output(
      os.path.join(self._alias_path, "training.models"), self.train_job.out_model_dir
    )
    tk.register_output(
      os.path.join(self._alias_path, "training.config"),
      self.train_job.out_returnn_config_file,
    )

  def vocode(self, hdf_input, iterations=1, checkpoint=None, name=None, cleanup=False, peak_normalization=True):
    if name is None:
      name = self._alias_path
    forward_config = self.build_forward_config(hdf_input)
    forward_job = ReturnnForwardJob(
      model_checkpoint=self.train_job.out_checkpoints[checkpoint]
      if checkpoint
      else self.best_checkpoint,
      returnn_config=forward_config,
      time_rqmt=1,
      returnn_python_exe=self._returnn_gpu_exe,
      returnn_root=self._returnn_root,
    )
    forward_job.add_alias(name + "/vocoder_forward")
    # this job is never needed to be kept
    forward_job.set_keep_value(20)
    from i6_experiments.users.rossenbach.tts.vocoder.griffin_lim import (
      HDFPhaseReconstruction,
    )

    hdf_reconstruct = HDFPhaseReconstruction(
      hdf_file=forward_job.out_default_hdf,
      backend="legacy",
      iterations=iterations,
      sample_rate=self._sample_rate,
      window_shift=self._source_audio_opts.options.step_len,
      window_size=self._source_audio_opts.options.window_len,
      preemphasis=self._source_audio_opts.options.preemphasis,
      peak_normalization=peak_normalization,
      file_format="ogg",
      mem_rqmt=16,
      time_rqmt=16,
    )
    hdf_reconstruct.add_alias(os.path.join(name, "hdf_reconstruction"))
    if cleanup == True:
      return hdf_reconstruct.out_corpus, forward_job
    return hdf_reconstruct.out_corpus


def default_vocoder(output_path, corpus_data, returnn_exe, returnn_root):

  # Vocoder training

  mini_vocoder = LJSpeechMiniGLVocoder(
    name="vocoder",
    alias_path=output_path,
    zip_dataset=corpus_data.zip,
    dev_segments=corpus_data.dev_segments,
    train_segments=corpus_data.train_segments,
    audio_datastream=corpus_data.audio_opts,
    model_size=512,
    returnn_gpu_exe=returnn_exe,
    returnn_root=returnn_root,
  )

  mini_vocoder.build_config()

  mini_vocoder.train(
    num_epochs=100,
    time_rqmt=36,
    mem_rqmt=12,
  )
  return mini_vocoder


def get_default_vocoder(name):
  returnn_exe = RETURNN_EXE

  corpus_data = get_vocoder_data()
  output_path = name

  mini_vocoder = default_vocoder(output_path, corpus_data, returnn_exe, RETURNN_RC_ROOT)
  mini_vocoder.train(num_epochs=100, time_rqmt=36, mem_rqmt=12)
  return mini_vocoder
