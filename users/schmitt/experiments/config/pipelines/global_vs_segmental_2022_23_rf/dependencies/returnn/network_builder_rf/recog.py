from typing import Optional, Dict

from returnn.tensor import TensorDict


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


_v2_forward_out_filename = "output.py.gz"
_v2_forward_ext_out_filename = "output_ext.py.gz"


def _returnn_v2_get_forward_callback():
  from typing import TextIO
  from returnn.tensor import Tensor, TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config

  config = get_global_config()
  recog_def_ext = config.bool("__recog_def_ext", False)

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      self.out_file: Optional[TextIO] = None
      self.out_ext_file: Optional[TextIO] = None

    def init(self, *, model):
      import gzip

      self.out_file = gzip.open(_v2_forward_out_filename, "wt")
      self.out_file.write("{\n")

      if recog_def_ext:
        self.out_ext_file = gzip.open(_v2_forward_ext_out_filename, "wt")
        self.out_ext_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
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
        hyp_serialized = hyps.sparse_dim.vocab.get_seq_labels(hyp_ids)
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

    def finish(self):
      self.out_file.write("}\n")
      self.out_file.close()
      if self.out_ext_file:
        self.out_ext_file.write("}\n")
        self.out_ext_file.close()

  return _ReturnnRecogV2ForwardCallbackIface()
