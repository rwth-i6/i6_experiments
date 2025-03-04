from sisyphus import tk, Path
from typing import Dict, List, Union, Optional
import copy

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.training import PtCheckpoint
from i6_core.text.processing import WriteToTextFileJob
from i6_core.returnn import ReturnnDumpHDFJob
from i6_core.corpus.segments import SegmentCorpusJob

from ..config_builder import AEDConfigBuilder, TinaAlignmentModelConfigBuilder

from .analyze_encoder import _returnn_v2_forward_step as analyze_encoder_forward_step
from .analyze_encoder import _returnn_v2_get_forward_callback as analyze_encoder_get_forward_callback
from .analyze_encoder import analyze_encoder as analyze_encoder_def


def analyze_encoder(
        config_builder: Union[AEDConfigBuilder, TinaAlignmentModelConfigBuilder],
        seq_tags: List[str],
        corpus_key: str,
        checkpoint: Optional[PtCheckpoint],
        returnn_root: Path,
        returnn_python_exe: Path,
        alias: str,
        hdf_targets: Optional[Path],
        ref_alignment_hdf: Path,
        ref_alignment_blank_idx: int,
        ref_alignment_vocab_path: Path,
        seq_alias: str,
):
  assert seq_alias in ("ground-truth", "search")

  segment_file = WriteToTextFileJob(
    content=seq_tags
  )

  config_opts = {
    "dataset_opts": {
      "segment_paths": {corpus_key: segment_file.out_file},
      "corpus_key": corpus_key,
      "use_multi_proc": False,
    },
    "forward_step_func": analyze_encoder_forward_step,
    "forward_callback": analyze_encoder_get_forward_callback,
    "analyze_encoder_def": analyze_encoder_def,
    "json_vocab_path": config_builder.vocab_opts["vocab_path"],
  }

  config_opts.update({
    "ref_alignment_hdf": ref_alignment_hdf,
    "ref_alignment_blank_idx": ref_alignment_blank_idx,
    "ref_alignment_vocab_path": ref_alignment_vocab_path,
  })

  if hdf_targets is not None:
    config_opts["dataset_opts"]["hdf_targets"] = {corpus_key: hdf_targets}

  analyze_encoder_config = config_builder.get_analyze_encoder_config(opts=config_opts)

  output_files = ["encoder_cosine_sim"]

  analyze_gradients_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=analyze_encoder_config,
    returnn_root=returnn_root,
    returnn_python_exe=returnn_python_exe,
    output_files=output_files,
    mem_rqmt=12,
    time_rqmt=2,
    device="cpu",
    cpu_rqmt=3,
  )
  analyze_gradients_job.rqmt["sbatch_args"] = ["--exclude", "cn-260"]
  analyze_gradients_job.add_alias(f"{alias}/analysis/analyze_gradients_{seq_alias}/{'_'.join([tag.split('/')[-1] for tag in seq_tags])}")
  tk.register_output(analyze_gradients_job.get_one_alias(), analyze_gradients_job.out_files["encoder_cosine_sim"])

  return analyze_gradients_job
