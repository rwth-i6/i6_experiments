from i6_private.users.pzheng.librispeech_luescher import OFFICIAL_LANGUAGE_MODELS

import i6_core.rasr as rasr
import i6_core.returnn as returnn

from sisyphus import *
Path = tk.Path

#---------------------------------------
def get_standard_4gram(name, scale):

  lm_config = rasr.RasrConfig()
  lm_config.file = OFFICIAL_LANGUAGE_MODELS['4gram'].get_path()
  lm_config.type = "ARPA"
  lm_config.scale = scale

  return lm_config

rnnLMs = {}
model_name = "kazuki-lstm_27062019"
vocab_file = "/u/zhou/asr-exps/librispeech/dependencies/kazuki_lstmlm_27062019/vocabulary"
meta_graph = "/u/zhou/am-train/utils/testrun/lm-graph/lm.meta"
model_path = "/u/zhou/asr-exps/librispeech/dependencies/kazuki_lstmlm_27062019/network.040"
model_config = "/u/zhou/asr-exps/librispeech/dependencies/kazuki_lstmlm_27062019/inference.config"
rnnLMs[model_name] = (vocab_file, model_path, meta_graph, model_config)

def get_rnn_lm_config(name, scale, max_batch_size=64, sort_batch_request=None,
                      allow_reduced_history=True, batch_pruning_threshold=None,
                      nativeLSTM=Path("/u/zhou/libs/nativelstm2/tf1.12/NativeLstm2.so"), unknown="<UNK>",
                      as_job=False, **kwargs):
  """
  from /u/zhou/asr-exps/ted-lium2/recipe/setups/zhou/librispeech/lm_setup.py
  """

  assert name in rnnLMs.keys()
  vocab_file, model_path, meta_graph, model_config = rnnLMs[name]

  if meta_graph is None:
    assert model_config is not None, "neither model graph nor model config provided"
    if isinstance(model_config, dict):
      crnn_config = crnn.CRNNConfig(model_config, {}) # crnn_config from model_config
    else: crnn_config = model_config # config file
    compile_graph_job = crnn.CompileTFGraphJob(crnn_config)
    meta_graph = compile_graph_job.graph
    if as_job:
      tk.register_output("graph_%s" %name, meta_graph)

  rnn_lm_config = rasr.RasrConfig()
  rnn_lm_config.type                    = "tfrnn"
  rnn_lm_config.scale                   = scale
  rnn_lm_config.vocab_file              = Path(vocab_file)
  rnn_lm_config.transform_output_negate = True
  rnn_lm_config.vocab_unknown_word      = unknown # Note: change accordingly w.r.t. vocabulary

  rnn_lm_config.loader.type               = "meta"
  rnn_lm_config.loader.meta_graph_file    = meta_graph
  rnn_lm_config.loader.saved_model_file   = rasr.StringWrapper(model_path, Path(model_path+".index"))
  rnn_lm_config.loader.required_libraries = nativeLSTM

  rnn_lm_config.input_map.info_0.param_name             = "word"
  rnn_lm_config.input_map.info_0.tensor_name            = "extern_data/placeholders/delayed/delayed"
  rnn_lm_config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

  rnn_lm_config.output_map.info_0.param_name  = "softmax"
  rnn_lm_config.output_map.info_0.tensor_name = "output/output_batch_major"

  rnn_lm_config.min_batch_size = kwargs.get("min_batch_size", 4)
  rnn_lm_config.opt_batch_size = kwargs.get("opt_batch_size", 64)
  rnn_lm_config.max_batch_size = max_batch_size
  rnn_lm_config.allow_reduced_history = allow_reduced_history

  if sort_batch_request is not None:
    rnn_lm_config.sort_batch_request = sort_batch_request
  if batch_pruning_threshold is not None:
    rnn_lm_config.batch_pruning_threshold = batch_pruning_threshold

  return rnn_lm_config

def get_bigram(name, scale):

  lm_config = rasr.RasrConfig()
  lm_config.file = Path('/work/asr3/irie/data/librispeech/lm/seq_train/960hr/kn2.no_pruning.gz', cached=True)
  lm_config.type = "ARPA"
  lm_config.scale = scale

  return lm_config
