from typing import Optional, Dict, Any, Tuple
import tree
import functools
import copy

from returnn.tensor import Tensor, Dim, single_step_dim, TensorDict
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.state import State

from sisyphus import tk, Path

from i6_core.returnn.training import PtCheckpoint
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchWordsToCTMJob, SearchBPEtoWordsJob, SearchTakeBestJob, SearchOutputRawReplaceJob

from .config_builder import AEDConfigBuilder
from .tools_paths import RETURNN_EXE, RETURNN_ROOT
from .analysis import analysis
from .model.aed import AEDModel

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils


def model_recog(
        *,
        model: AEDModel,
        data: Tensor,
        data_spatial_dim: Dim,
        beam_size: int,
        max_seq_len: Optional[int] = None,
        length_normalization_exponent: float = 1.0,
) -> Tuple[Tensor, Tensor, Dim, Tensor, Dim, Dim]:
  """
  Function is run within RETURNN.

  Earlier we used the generic beam_search function,
  but now we just directly perform the search here,
  as this is overall simpler and shorter.

  :return:
      recog results including beam {batch, beam, out_spatial},
      log probs {batch, beam},
      out_spatial_dim,
      final beam_dim
  """

  # --------------------------------- init encoder, dims, etc ---------------------------------

  enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

  if max_seq_len is None:
    max_seq_len = enc_spatial_dim.get_size_tensor()
  else:
    max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims

  ended = rf.constant(False, dims=batch_dims_)
  out_seq_len = rf.constant(0, dims=batch_dims_)
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)

  # lists of [B, beam] tensors
  seq_targets = []
  seq_backrefs = []

  # --------------------------------- init states ---------------------------------
  decoder_state = model.decoder.decoder_default_initial_state(
    batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)

  # --------------------------------- init targets ---------------------------------

  target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)

  # --------------------------------- main loop ---------------------------------

  i = 0
  while True:
    # --------------------------------- get embeddings ---------------------------------
    if i == 0:
      input_embed = rf.zeros(
        batch_dims_ + [model.decoder.target_embed.out_dim],
        feature_dim=model.decoder.target_embed.out_dim)
    else:
      input_embed = model.decoder.target_embed(target)

    # --------------------------------- decoder step ---------------------------------

    step_out, decoder_state = model.decoder.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed,
      state=decoder_state,
    )
    logits = model.decoder.decode_logits(
      input_embed=input_embed,
      s=step_out["s"],
      att=step_out["att"],
    )

    label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

    # --------------------------------- filter finished beams, pick top-k ---------------------------------

    # Filter out finished beams
    label_log_prob = rf.where(
      ended,
      rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
      label_log_prob,
    )

    seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

    # --------------------------------- update state ---------------------------------
    decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), decoder_state)

    ended = rf.gather(ended, indices=backrefs)
    out_seq_len = rf.gather(out_seq_len, indices=backrefs)
    i += 1

    ended = rf.logical_or(ended, rf.convert_to_tensor(target == model.eos_idx))
    ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
    if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
      break
    out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    if i > 1 and length_normalization_exponent != 0:
      # Length-normalized scores, so we evaluate score_t/len.
      # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
      # Because we count with EOS symbol, shifted by one.
      seq_log_prob *= rf.where(
        ended,
        (i / (i - 1)) ** length_normalization_exponent,
        1.0,
      )

  if i > 0 and length_normalization_exponent != 0:
    seq_log_prob *= (1 / i) ** length_normalization_exponent

  # Backtrack via backrefs, resolve beams.
  seq_targets_ = []
  indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
  for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
    # indices: FinalBeam -> Beam
    # backrefs: Beam -> PrevBeam
    seq_targets_.insert(0, rf.gather(target, indices=indices))
    indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

  seq_targets__ = TensorArray(seq_targets_[0])
  for target in seq_targets_:
    seq_targets__ = seq_targets__.push_back(target)
  out_spatial_dim = Dim(out_seq_len, name="out-spatial")
  seq_targets = seq_targets__.stack(axis=out_spatial_dim)

  best_hyps = rf.reduce_argmax(seq_log_prob, axis=beam_dim)
  best_seq_targets = rf.gather(
    seq_targets,
    indices=best_hyps,
    axis=beam_dim,
  )

  # out_spatial_dim has 2 dims (batch, beam) since seqs can have different lengths for global AED
  # just gather the best seq lengths (same as for the seqs themselves)
  best_seq_targets_spatial_dim = out_spatial_dim.copy()
  best_seq_targets_spatial_dim.dyn_size_ext = rf.gather(
    out_spatial_dim.dyn_size_ext,
    indices=best_hyps,
    axis=beam_dim,
  )

  # replace the old dim with the new one
  best_seq_targets = utils.copy_tensor_replace_dim_tag(
    best_seq_targets, out_spatial_dim, best_seq_targets_spatial_dim
  )

  return best_seq_targets, seq_log_prob, best_seq_targets_spatial_dim, seq_targets, out_spatial_dim, beam_dim


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  import returnn.frontend as rf
  from returnn.tensor import Tensor, Dim, batch_dim
  from returnn.config import get_global_config

  if rf.is_executing_eagerly():
    batch_size = int(batch_dim.get_dim_value())
    for batch_idx in range(batch_size):
      seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
      print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  recog_def = config.typed_value("_recog_def")
  extra = {}
  if config.bool("cheating", False):
    default_target_key = config.typed_value("target")
    targets = extern_data[default_target_key]
    extra.update(dict(targets=targets, targets_spatial_dim=targets.get_time_dim_tag()))

  beam_search_opts = config.typed_value("beam_search_opts", {})  # type: Dict
  extra.update(beam_search_opts)

  recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, **extra)
  hdf_targets = None
  hdf_targets_spatial_dim = None
  if len(recog_out) == 5:
    # recog results including beam {batch, beam, out_spatial},
    # log probs {batch, beam},
    # extra outputs {batch, beam, ...},
    # out_spatial_dim,
    # final beam_dim
    assert len(recog_out) == 5, f"mismatch, got {len(recog_out)} outputs with recog_def_ext=True"
    hyps, scores, extra, out_spatial_dim, beam_dim = recog_out
  elif len(recog_out) == 4:
    # same without extra outputs
    assert len(recog_out) == 4, f"mismatch, got {len(recog_out)} outputs recog_def_ext=False"
    hyps, scores, out_spatial_dim, beam_dim = recog_out
    extra = {}
  elif len(recog_out) == 6:
    # same as with 4, but additionally alignment and alignment_spatial_dim
    hdf_targets, scores, hdf_targets_spatial_dim, hyps, out_spatial_dim, beam_dim = recog_out
    extra = {}
  elif len(recog_out) == 7:
    # same as with 4, but additionally alignment and alignment_spatial_dim
    hdf_targets, scores, hdf_targets_spatial_dim, hyps, out_spatial_dim, beam_dim, ratio_recomb_paths = recog_out
    extra = {}
    rf.get_run_ctx().mark_as_output(
      ratio_recomb_paths, "ratio_recomb_paths", dims=[batch_dim])
  else:
    raise ValueError(f"unexpected num outputs {len(recog_out)} from recog_def")
  assert isinstance(hyps, Tensor) and isinstance(scores, Tensor)
  assert isinstance(out_spatial_dim, Dim) and isinstance(beam_dim, Dim)
  rf.get_run_ctx().mark_as_output(hyps, "hyps", dims=[batch_dim, beam_dim, out_spatial_dim])
  rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, beam_dim])
  assert isinstance(extra, dict)
  for k, v in extra.items():
    assert isinstance(k, str) and isinstance(v, Tensor)
    assert v.dims[:2] == (batch_dim, beam_dim)
    rf.get_run_ctx().mark_as_output(v, k, dims=v.dims)

  if hdf_targets is not None:
    assert isinstance(hdf_targets, Tensor) and isinstance(hdf_targets_spatial_dim, Dim)
    rf.get_run_ctx().mark_as_output(hdf_targets, "hdf_targets", dims=[batch_dim, hdf_targets_spatial_dim])


_v2_forward_out_filename = "output.py.gz"
_v2_forward_ext_out_filename = "output_ext.py.gz"


def _returnn_v2_get_forward_callback():
  from typing import TextIO
  from returnn.tensor import Tensor, TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config
  from returnn.datasets.hdf import SimpleHDFWriter
  from i6_experiments.users.schmitt import hdf
  import numpy as np

  config = get_global_config()
  recog_def_ext = config.bool("__recog_def_ext", False)

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      self.out_file: Optional[TextIO] = None
      self.out_ext_file: Optional[TextIO] = None
      self.hdf_file: Optional[SimpleHDFWriter] = None
      self.out_recomb_path_ratio_file: Optional[TextIO] = None

    def init(self, *, model):
      import gzip

      self.out_file = gzip.open(_v2_forward_out_filename, "wt")
      self.out_file.write("{\n")

      if recog_def_ext:
        self.out_ext_file = gzip.open(_v2_forward_ext_out_filename, "wt")
        self.out_ext_file.write("{\n")

      hdf_target_dim = model.target_dim.dimension

      self.hdf_file = SimpleHDFWriter(
        filename="best_hyp.hdf", dim=hdf_target_dim, ndim=1
      )

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      hdf_targets: Tensor = outputs["hdf_targets"]  # [T]
      hyps: Tensor = outputs["hyps"]  # [beam, out_spatial]
      scores: Tensor = outputs["scores"]  # [beam]
      assert hyps.sparse_dim and hyps.sparse_dim.vocab  # should come from the model
      assert hyps.dims[1].dyn_size_ext, f"hyps {hyps} do not define seq lengths"
      # AED/Transducer etc will have hyps len depending on beam -- however, CTC will not.
      hyps_len = hyps.dims[1].dyn_size_ext  # [beam] or []
      assert hyps.raw_tensor.shape[:1] == scores.raw_tensor.shape  # (beam,)
      if hyps_len.raw_tensor.shape:
        assert scores.raw_tensor.shape == hyps_len.raw_tensor.shape  # (beam,)
      num_beam = hyps.raw_tensor.shape[0]
      # Consistent to old search task, list[(float,str)].
      self.out_file.write(f"{seq_tag!r}: [\n")

      for i in range(num_beam):
        score = float(scores.raw_tensor[i])
        hyp_ids = hyps.raw_tensor[
                  i, : hyps_len.raw_tensor[i] if hyps_len.raw_tensor.shape else hyps_len.raw_tensor
                  ]
        try:
          hyp_serialized = hyps.sparse_dim.vocab.get_seq_labels(hyp_ids)
        except Exception as e:
          print(f"Error in get_seq_labels: {e!r}")
          print(f"hyp_ids: {hyp_ids!r}")
          exit()
        self.out_file.write(f"  ({score!r}, {hyp_serialized!r}),\n")
      self.out_file.write("],\n")

      if self.out_ext_file:
        self.out_ext_file.write(f"{seq_tag!r}: [\n")
        for v in outputs.data.values():
          assert v.dims[0].dimension == num_beam
        for i in range(num_beam):
          d = {k: v.raw_tensor[i].tolist() for k, v in outputs.data.items() if k not in {"hyps", "scores"}}
          self.out_ext_file.write(f"  {d!r},\n")
        self.out_ext_file.write("],\n")

      if self.hdf_file:
        seq_len = hdf_targets.dims[0].dyn_size_ext.raw_tensor.item()
        hdf_targets_raw = hdf_targets.raw_tensor[:seq_len]

        if seq_len > 0:
          hdf.dump_hdf_numpy(
            hdf_dataset=self.hdf_file,
            data=hdf_targets_raw[None],  # [1, T]
            seq_lens=np.array([seq_len]),  # [1]
            seq_tags=[seq_tag],
          )

    def finish(self):
      self.out_file.write("}\n")
      self.out_file.close()
      if self.out_ext_file:
        self.out_ext_file.write("}\n")
        self.out_ext_file.close()
      if self.hdf_file:
        self.hdf_file.close()
      if self.out_recomb_path_ratio_file:
        self.out_recomb_path_ratio_file.write("}\n")
        self.out_recomb_path_ratio_file.close()

  return _ReturnnRecogV2ForwardCallbackIface()


class RecogExperiment:
  def __init__(
          self,
          alias: str,
          config_builder: AEDConfigBuilder,
          checkpoint: PtCheckpoint,
          checkpoint_alias: str,
          recog_opts: Dict,
          search_rqmt: Optional[Dict] = None,
          search_alias: Optional[str] = None,
  ):
    self.config_builder = config_builder
    self.checkpoint_alias = checkpoint_alias
    self.checkpoint = checkpoint
    self.recog_opts = recog_opts

    self.corpus_key = self.recog_opts["dataset_opts"]["corpus_key"]
    self.stm_corpus_key = self.corpus_key

    self.alias = alias

    self.concat_num = self.recog_opts.get("dataset_opts", {}).get("concat_num")
    if self.concat_num is not None:
      self.stm_corpus_key += "_concat-%d" % self.concat_num

    self.search_rqmt = search_rqmt if search_rqmt is not None else {}
    self.search_hyps_file = None
    self.best_search_hyps_hdf = None

    self.alias += "/returnn_decoding/" if search_alias is None else f"/{search_alias}"
    length_normalization_exponent = self.recog_opts.get("length_normalization_exponent")
    if length_normalization_exponent is not None:
      self.alias += f"len-norm-exp-{length_normalization_exponent:.1f}"

    self.alias = f"{self.alias}_beam-{self.recog_opts['beam_size']}"

    self.base_exp_alias = self.alias
    self.alias += f"/{self.checkpoint_alias}-checkpoint/{self.stm_corpus_key}"

    self.score_job = None
    self.analyze_gradients_job = None
    self.dump_gradients_job = None
    self.dump_self_att_job = None

  def run_eval(self) -> ScliteJob:
    stm_path = self.config_builder.dataset.stm_paths[self.stm_corpus_key]
    ctm_path = self.get_ctm_path()

    self.score_job = ScliteJob(
      ref=stm_path,
      hyp=ctm_path
    )

    self.score_job.add_alias("%s/scores" % (self.alias,))
    tk.register_output(self.score_job.get_one_alias(), self.score_job.out_report_dir)

    return self.score_job

  def get_ctm_path(self) -> Path:
    recog_config = self.config_builder.get_recog_config(opts=self.recog_opts)

    search_job = ReturnnForwardJobV2(
      model_checkpoint=self.checkpoint,
      returnn_config=recog_config,
      returnn_root=RETURNN_ROOT,
      returnn_python_exe=RETURNN_EXE,
      output_files=["output.py.gz", "best_hyp.hdf"],
      mem_rqmt=self.search_rqmt.get("mem", 6),
      time_rqmt=self.search_rqmt.get("time", 1),
      cpu_rqmt=self.search_rqmt.get("cpu", 2),
    )
    search_job.rqmt["sbatch_args"] = self.search_rqmt.get("sbatch_args", [])
    search_job.add_alias(f"{self.alias}/search")
    self.search_hyps_file = search_job.out_files["output.py.gz"]
    self.best_search_hyps_hdf = search_job.out_files["best_hyp.hdf"]
    search_take_best_job = SearchTakeBestJob(search_py_output=search_job.out_files["output.py.gz"])
    out_search_file = search_take_best_job.out_best_search_results

    out_search_results = SearchBPEtoWordsJob(out_search_file).out_word_search_results

    search_words_to_ctm_job = SearchWordsToCTMJob(
      out_search_results,
      self.config_builder.dataset.corpus_paths[self.corpus_key])

    return search_words_to_ctm_job.out_ctm_file

  def run_analysis(self, analysis_opts: Optional[Dict] = None):
    att_weight_seq_tags = analysis_opts["att_weight_seq_tags"]

    if analysis_opts.get("analyze_gradients", False):
      self.analyze_gradients_job = analysis.analyze_gradients(
        config_builder=self.config_builder,
        seq_tags=att_weight_seq_tags,
        corpus_key=self.stm_corpus_key,
        checkpoint=self.checkpoint,
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        alias=self.alias,
        hdf_targets=analysis_opts.get("ground_truth_hdf"),
        ref_alignment_hdf=analysis_opts.get("ref_alignment_hdf"),
        ref_alignment_blank_idx=analysis_opts.get("ref_alignment_blank_idx"),
        ref_alignment_vocab_path=analysis_opts.get("ref_alignment_vocab_path"),
        seq_alias="ground-truth",
        do_forced_align_on_gradients=analysis_opts.get("do_forced_align_on_gradients", False),
        plot_encoder_gradient_graph=analysis_opts.get("plot_encoder_gradient_graph", False),
        plot_encoder_layers=analysis_opts.get("analyze_gradients_plot_encoder_layers", False),
        plot_log_gradients=analysis_opts.get("analyze_gradients_plot_log_gradients", False),
      )

    if analysis_opts.get("dump_gradients", False):
      self.dump_gradients_job = analysis.dump_gradients(
        config_builder=self.config_builder,
        seq_tags=att_weight_seq_tags,
        corpus_key=self.stm_corpus_key,
        checkpoint=self.checkpoint,
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        alias=self.alias,
        hdf_targets=analysis_opts.get("ground_truth_hdf"),
        seq_alias="ground-truth",
        input_layer_name=analysis_opts.get("dump_gradients_input_layer_name", "encoder_input"),
      )

    if analysis_opts.get("dump_self_att", False):
      self.dump_self_att_job = analysis.dump_self_att(
        config_builder=self.config_builder,
        seq_tags=att_weight_seq_tags,
        corpus_key=self.stm_corpus_key,
        checkpoint=self.checkpoint,
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        alias=self.alias,
        hdf_targets=analysis_opts.get("ground_truth_hdf"),
        seq_alias="ground-truth",
      )
