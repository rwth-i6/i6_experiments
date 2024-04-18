from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJob

from sisyphus import Path

import subprocess
import tempfile
import shutil
import os
import json
import ast
from typing import Optional

import recipe.i6_experiments.users.schmitt.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)


class DumpPhonemeAlignJob(Job):
  def __init__(
          self,
          rasr_config,
          rasr_post_config,
          time_red,
          rasr_exe,
          state_tying_file,
          returnn_python_exe,
          returnn_root,
          time_rqtm=1,
          mem_rqmt=2,
  ):
    self.returnn_python_exe = returnn_python_exe
    self.returnn_root = returnn_root

    self.rasr_config = rasr_config
    self.rasr_post_config = rasr_post_config
    self.rasr_exe = rasr_exe
    self.state_tying_file = state_tying_file
    self.time_red = str(time_red)

    self.time_rqmt = time_rqtm
    self.mem_rqtm = mem_rqmt

    self.out_align = self.output_path("out_align")
    self.out_phoneme_vocab = self.output_path("phoneme_vocab")

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqtm, "time": self.time_rqmt})

  def create_files(self):
    RasrCommand.write_config(self.rasr_config, self.rasr_post_config, "rasr.config")

  def run(self):
    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(tools_dir, "dump_phoneme_align.py"),
      "rasr.config",
      "--rasr_exe", self.rasr_exe.get_path(),
      "--time_red", str(self.time_red),
      "--returnn_root", self.returnn_root.get_path(),
      # "--state_tying_file", self.state_tying_file.get_path()
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_align", self.out_align.get_path())
    shutil.move("phoneme_vocab", self.out_phoneme_vocab.get_path())

  @classmethod
  def hash(cls, kwargs):
    if kwargs["state_tying_file"].get_path() == "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/state-tying_mono-eow_3-states":
      kwargs.pop("state_tying_file")
    return super().hash(kwargs)


class CompareAlignmentsJob(Job):
  def __init__(
    self, hdf_align1, hdf_align2, seq_tags, blank_idx1, blank_idx2, vocab1, vocab2, name1, name2,
    time_rqtm=1, mem_rqmt=2, returnn_python_exe=None,
    returnn_root=None):
    self.align1_name = name1
    self.align2_name = name2
    self.vocab2 = vocab2
    self.vocab1 = vocab1
    self.blank_idx2 = blank_idx2
    self.blank_idx1 = blank_idx1
    self.seq_tags = seq_tags
    self.hdf_align2 = hdf_align2
    self.hdf_align1 = hdf_align1
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.time_rqmt = time_rqtm
    self.mem_rqtm = mem_rqmt

    self.out_align = self.output_path("out_align")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqtm, "time": self.time_rqmt})

  def run(self):
    with open("seq_file", "w+") as f:
      for tag in self.seq_tags:
        f.write(tag + "\n")
    command = [
      self.returnn_python_exe, os.path.join(tools_dir, "compare_aligns.py"),
      self.hdf_align1.get_path(), self.hdf_align2.get_path(), "--segment_path", "seq_file",
      "--blank_idx_align1", str(self.blank_idx1), "--blank_idx_align2", str(self.blank_idx2),
      "--vocab_align1", self.vocab1.get_path(), "--vocab_align2", self.vocab2.get_path(),
      "--align1_name", self.align1_name, "--align2_name", self.align2_name,
      "--returnn_root", self.returnn_root, ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])


class AugmentBPEAlignmentJob(Job):
  def __init__(self, bpe_align_hdf, phoneme_align_hdf, bpe_blank_idx, phoneme_blank_idx,
               bpe_vocab, phoneme_vocab, phoneme_lexicon, segment_file, time_red_phon_align, time_red_bpe_align,
               time_rqtm=1, mem_rqmt=2, returnn_python_exe=None,
               returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.phoneme_lexicon = phoneme_lexicon
    self.phoneme_vocab = phoneme_vocab
    self.bpe_vocab = bpe_vocab
    self.phoneme_blank_idx = phoneme_blank_idx
    self.bpe_blank_idx = bpe_blank_idx
    self.phoneme_align_hdf = phoneme_align_hdf
    self.bpe_align_hdf = bpe_align_hdf
    self.bpe_upsampling_factor = int(time_red_bpe_align / time_red_phon_align)
    self.segment_file = segment_file

    self.time_rqmt = time_rqtm
    self.mem_rqtm = mem_rqmt

    self.out_align = self.output_path("out_align")
    self.out_vocab = self.output_path("out_vocab")
    self.out_skipped_seqs = self.output_path("out_skipped_seqs")
    self.out_skipped_seqs_var = self.output_var("out_skipped_seqs_var")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqtm, "time": self.time_rqmt})

  def run(self):
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "augment_bpe_align.py"),
      self.bpe_align_hdf.get_path(), self.phoneme_align_hdf.get_path(), "--bpe_blank_idx", str(self.bpe_blank_idx), "--phoneme_blank_idx",
      str(self.phoneme_blank_idx), "--bpe_vocab", self.bpe_vocab.get_path(), "--phoneme_vocab", tk.uncached_path(self.phoneme_vocab),
      "--phoneme_lexicon", self.phoneme_lexicon.get_path(), "--out_align", self.out_align.get_path(),
      "--out_vocab", self.out_vocab.get_path(), "--out_skipped_seqs", self.out_skipped_seqs.get_path(),
      "--segment_file", self.segment_file.get_path(), "--bpe_upsampling_factor", str(self.bpe_upsampling_factor),
      "--returnn_root", self.returnn_root
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    with open(self.out_skipped_seqs.get_path(), "r") as f:
      skipped_seqs = eval(f.read())
      self.out_skipped_seqs_var.set(skipped_seqs)


class AlignmentStatisticsJob(Job):
  def __init__(
          self,
          alignment,
          json_vocab,
          seq_list_filter_file=None,
          blank_idx=0,
          silence_idx=None,
          time_rqmt=2,
          returnn_python_exe=None,
          returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.alignment = alignment
    self.json_vocab = json_vocab
    self.seq_list_filter_file = seq_list_filter_file
    self.blank_idx = blank_idx
    self.silence_idx = silence_idx

    self.out_statistics = self.output_path("statistics")
    self.out_histograms_folder = self.output_path("histograms")
    self.out_label_dependent_mean_lens = self.output_path("label_dependent_mean_lens")
    self.out_label_dependent_mean_lens_var = self.output_var("label_dependent_mean_lens_var", pickle=True)

    self.time_rqmt = time_rqmt

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": self.time_rqmt})

  def run(self):
    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(tools_dir, "alignment_statistics.py"),
      self.alignment.get_path(),
      "--json-vocab", self.json_vocab.get_path(),
      "--blank-idx", str(self.blank_idx),
      "--returnn-root", self.returnn_root.get_path()
    ]

    if self.seq_list_filter_file:
      command += ["--seq-list-filter-file", str(self.seq_list_filter_file)]

    if self.silence_idx is not None:
      command += ["--sil-idx", str(self.silence_idx)]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    with open("label_dependent_mean_lens", "r") as f:
      label_dep_means = ast.literal_eval(f.read())
      label_dep_means = {int(k): v for k, v in label_dep_means.items()}
      label_dep_means = [label_dep_means[idx] for idx in range(len(label_dep_means))]
    self.out_label_dependent_mean_lens_var.set(label_dep_means)

    shutil.move("statistics", self.out_statistics.get_path())
    shutil.move("histograms", self.out_histograms_folder.get_path())
    shutil.move("label_dependent_mean_lens", self.out_label_dependent_mean_lens.get_path())


class AlignmentSplitSilenceJob(Job):
  def __init__(self, hdf_align_path, segment_file, sil_idx, blank_idx, max_len,
               returnn_python_exe=None, returnn_root=None):
    self.max_len = max_len
    self.blank_idx = blank_idx
    self.sil_idx = sil_idx
    self.hdf_align_path = hdf_align_path
    self.segment_file = segment_file
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.out_align = self.output_path("out_align")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 1})

  def run(self):
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "alignment_split_silence.py"),
      self.hdf_align_path.get_path(),
      "--segment_file", tk.uncached_path(self.segment_file),
      "--sil_idx", str(self.sil_idx), "--blank_idx", str(self.blank_idx), "--max_len", str(self.max_len.get()),
      "--returnn_root", self.returnn_root
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_alignment", self.out_align.get_path())


class AlignmentCenterSegBoundaryJob(Job):
  def __init__(self, hdf_align_path, segment_file, blank_idx,
               returnn_python_exe=None, returnn_root=None):
    self.blank_idx = blank_idx
    self.hdf_align_path = hdf_align_path
    self.segment_file = segment_file
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.out_align = self.output_path("out_align")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 1})

  def run(self):
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "alignment_center_seg_boundaries.py"),
      self.hdf_align_path.get_path(),
      "--segment_file", tk.uncached_path(self.segment_file),
      "--blank_idx", str(self.blank_idx),
      "--returnn_root", self.returnn_root
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_alignment", self.out_align.get_path())


class AlignmentAddEosJob(Job):
  def __init__(
          self,
          hdf_align_path: Path,
          segment_file: Path,
          blank_idx: int,
          eos_idx: int,
          returnn_python_exe: Path,
          returnn_root: Path,
  ):
    self.returnn_root = returnn_root
    self.returnn_python_exe = returnn_python_exe
    self.blank_idx = blank_idx
    self.eos_idx = eos_idx
    self.hdf_align_path = hdf_align_path
    self.segment_file = segment_file

    self.out_align = self.output_path("out_align")
    self.out_keep_seqs = self.output_path("out_keep_seqs")
    self.out_exclude_seqs = self.output_path("out_exclude_seqs")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 30, "gpu": 0})

  def run(self):
    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(tools_dir, "alignment_add_eos.py"),
      self.hdf_align_path.get_path(),
      "--blank_idx", str(self.blank_idx),
      "--eos_idx", str(self.eos_idx),
      "--returnn_root", self.returnn_root.get_path()
    ]

    if self.segment_file is not None:
      command += ["--segment_file", self.segment_file.get_path()]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_alignment", self.out_align.get_path())
    shutil.move("out_keep_seqs", self.out_keep_seqs.get_path())
    shutil.move("out_exclude_seqs", self.out_exclude_seqs.get_path())


class ReduceAlignmentJob(Job):
  def __init__(self, hdf_align_path, segment_file, sil_idx, blank_idx, reduction_factor,
               returnn_python_exe=None, returnn_root=None):
    self.reduction_factor = reduction_factor
    self.blank_idx = blank_idx
    self.sil_idx = sil_idx
    self.hdf_align_path = hdf_align_path
    self.segment_file = segment_file
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.out_align = self.output_path("out_align")
    self.out_skipped_seqs = self.output_path("skipped_seqs")
    self.out_skipped_seqs_var = self.output_var("skipped_seqs_var")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 1})

  def run(self):
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "reduce_alignment.py"),
      self.hdf_align_path.get_path(),
      "--segment_file", tk.uncached_path(self.segment_file),
      "--sil_idx", str(self.sil_idx), "--blank_idx", str(self.blank_idx),
      "--reduction_factor", str(self.reduction_factor),
      "--returnn_root", self.returnn_root
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_alignment", self.out_align.get_path())
    shutil.move("skipped_seqs", self.out_skipped_seqs.get_path())

    with open(self.out_skipped_seqs.get_path(), "r") as f:
      skipped_seqs = eval(f.read())
      self.out_skipped_seqs_var.set(skipped_seqs)


class DumpNonBlanksFromAlignmentJob(Job):
  def __init__(self, alignment: Path, returnn_python_exe: Path, returnn_root: Path, blank_idx=0, time_rqmt=2):
    self.returnn_python_exe = returnn_python_exe
    self.returnn_root = returnn_root

    self.alignment = alignment
    self.blank_idx = blank_idx

    self.time_rqmt = time_rqmt

    self.out_labels = self.output_path("out_labels")
    self.out_skipped_seqs = self.output_path("skipped_seqs")
    self.out_skipped_seqs_var = self.output_var("skipped_seqs_var")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": self.time_rqmt})

  def run(self):
    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(tools_dir, "alignment_dump_non_blanks.py"),
      self.alignment.get_path(),
      "--blank_idx", str(self.blank_idx),
      "--returnn-root", self.returnn_root.get_path()
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_labels", self.out_labels.get_path())
    shutil.move("skipped_seqs", self.out_skipped_seqs.get_path())

    with open(self.out_skipped_seqs.get_path(), "r") as f:
      skipped_seqs = eval(f.read())
      self.out_skipped_seqs_var.set(skipped_seqs)

  @classmethod
  def hash(cls, kwargs):
    kwargs.pop("time_rqmt")
    return super().hash(kwargs)


class RemoveLabelFromAlignmentJob(Job):
  def __init__(self, alignment, blank_idx, remove_idx, remove_only_middle, time_rqmt=2, returnn_python_exe=None, returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.alignment = alignment
    self.blank_idx = blank_idx
    self.remove_idx = remove_idx
    self.remove_only_middle = remove_only_middle

    self.time_rqmt = time_rqmt

    self.out_alignment = self.output_path("out_alignment")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": self.time_rqmt})

  def run(self):
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "alignment_remove_label.py"),
      self.alignment.get_path(),
      "--blank_idx", str(self.blank_idx), "--remove_idx", str(self.remove_idx),
      "--returnn-root", self.returnn_root
    ]

    if self.remove_only_middle:
      command += ["--remove_only_middle"]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_alignment", self.out_alignment.get_path())


class SwitchLabelInAlignmentJob(Job):
  def __init__(self, alignment, new_idx, orig_idx, time_rqmt=2, returnn_python_exe=None, returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.alignment = alignment
    self.new_idx = new_idx
    self.orig_idx = orig_idx

    self.time_rqmt = time_rqmt

    self.out_alignment = self.output_path("out_alignment")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": self.time_rqmt})

  def run(self):
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "alignment_switch_label.py"),
      self.alignment.get_path(),
      "--new_idx", str(self.new_idx), "--orig_idx", str(self.orig_idx),
      "--returnn-root", self.returnn_root
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_alignment", self.out_alignment.get_path())


class DumpAlignmentFromTxtJob(Job):
  def __init__(self, alignment_txt, segment_file, num_classes, returnn_python_exe=None, returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.alignment_txt = alignment_txt
    self.segment_file = segment_file
    self.num_classes = num_classes

    self.out_hdf_align = self.output_path("out_hdf_align")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

  def run(self):
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "dump_alignment_from_txt.py"),
      self.alignment_txt.get_path(),
      "--segment_file", self.segment_file.get_path(), "--num_classes", str(self.num_classes),
      "--returnn-root", self.returnn_root
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_hdf_align", self.out_hdf_align.get_path())


class DumpAlignmentFromTxtJobV2(Job):
  def __init__(
    self, rasr_config, rasr_post_config,
    num_classes, returnn_python_exe=None, returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.rasr_config = rasr_config
    self.rasr_post_config = rasr_post_config
    self.num_classes = num_classes

    self.out_hdf_align = self.output_path("out_hdf_align")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 10})

  def run(self):
    RasrCommand.write_config(
      config=self.rasr_config, post_config=self.rasr_post_config, filename="rasr.config"
    )
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "dump_alignment_from_txt_new.py"),
      "--rasr_config_file", "rasr.config",
      "--num_classes", str(self.num_classes),
      "--returnn-root", self.returnn_root
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_hdf_align", self.out_hdf_align.get_path())


class ChooseBestAlignmentJob(Job):
  def __init__(
          self,
          returnn_config: ReturnnConfig,
          rasr_config_path: Path,
          rasr_nn_trainer_exe: Path,
          segment_path: Path,
          align1_hdf_path: Path,
          align2_hdf_path: Path,
          label_name: str,
          blank_idx: int,
          mem_rqmt: int,
          time_rqmt: int,
          returnn_python_exe=None,
          returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.rasr_config_path = rasr_config_path
    self.returnn_config = returnn_config
    self.rasr_nn_trainer_exe = rasr_nn_trainer_exe
    self.segment_path = segment_path
    self.align1_hdf_path = align1_hdf_path
    self.align2_hdf_path = align2_hdf_path
    self.label_name = label_name
    self.blank_idx = blank_idx

    self.mem_rqmt = mem_rqmt
    self.time_rqmt = time_rqmt

    self.out_hdf_align = self.output_path("out_hdf_align")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "gpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

  def run(self):
    self.returnn_config.write("returnn.config")
    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "choose_best_align.py"),
      "returnn.config",
      "--rasr_config_path", self.rasr_config_path.get_path(),
      "--rasr_nn_trainer_exe", self.rasr_nn_trainer_exe.get_path(),
      "--segment_path", self.segment_path.get_path(),
      "--align1_hdf_path", self.align1_hdf_path.get_path(),
      "--align2_hdf_path", self.align2_hdf_path.get_path(),
      "--label_name", self.label_name,
      "--blank_idx", str(self.blank_idx),
      "--output_path", self.out_hdf_align.get_path(),
      "--returnn_root", self.returnn_root
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    # shutil.move("out_hdf_align", self.out_hdf_align.get_path())


class AlignmentRemoveAllBlankSeqsJob(Job):
  """
  Goes through HDF file with alignment sequences and returns another HDF file, which only contains the alignments
  with at least one non-blank label.
  """
  def __init__(
          self,
          hdf_align_path: Path,
          blank_idx: int,
          returnn_python_exe: Path,
          returnn_root: Path,
  ):
    self.returnn_root = returnn_root
    self.returnn_python_exe = returnn_python_exe
    self.blank_idx = blank_idx
    self.hdf_align_path = hdf_align_path

    self.out_align = self.output_path("out_align")
    self.out_segment_file = self.output_path("out_segment_file")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

  def run(self):
    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(tools_dir, "alignment_remove_all_blank_seqs.py"),
      self.hdf_align_path.get_path(),
      "--blank_idx", str(self.blank_idx),
      "--returnn_root", self.returnn_root.get_path()
    ]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("out_alignment", self.out_align.get_path())
    shutil.move("out_segment_file", self.out_segment_file.get_path())


class CompareBpeAndGmmAlignments(Job):
  def __init__(
          self,
          bpe_align_hdf: Path,
          bpe_vocab: Path,
          bpe_blank_idx: int,
          bpe_downsampling_factor: int,
          phoneme_alignment_cache: Path,
          phoneme_downsampling_factor: int,
          allophone_file: Path,
          silence_phone: str,
          segment_file: Optional[Path] = None,
          time_rqmt=1,
          mem_rqmt=4,
          returnn_python_exe=None,
          returnn_root=None
  ):
    self.returnn_python_exe = returnn_python_exe
    self.returnn_root = returnn_root

    self.bpe_align_hdf = bpe_align_hdf
    self.bpe_vocab = bpe_vocab
    self.bpe_blank_idx = bpe_blank_idx
    self.bpe_downsampling_factor = bpe_downsampling_factor
    self.phoneme_alignment_cache = phoneme_alignment_cache
    self.phoneme_downsampling_factor = phoneme_downsampling_factor
    self.allophone_file = allophone_file
    self.segment_file = segment_file
    self.silence_phone = silence_phone

    self.time_rqmt = time_rqmt
    self.mem_rqtm = mem_rqmt

    self.out_statistics = self.output_path("statistics")
    self.out_filtered_segments = self.output_path("filtered_segments")

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqtm, "time": self.time_rqmt})

  def create_files(self):
    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(tools_dir, "compare_bpe_and_gmm_alignment.py"),
      "--bpe_align_hdf", self.bpe_align_hdf.get_path(),
      "--bpe_vocab", self.bpe_vocab.get_path(),
      "--bpe_blank_idx", str(self.bpe_blank_idx),
      "--bpe_downsampling_factor", str(self.bpe_downsampling_factor),
      "--phoneme_align_cache", self.phoneme_alignment_cache.get_path(),
      "--phoneme_downsampling_factor", str(self.phoneme_downsampling_factor),
      "--allophone_file", self.allophone_file.get_path(),
      "--silence_phone", self.silence_phone,
      "--returnn_root", self.returnn_root.get_path(),
    ]

    if self.segment_file is not None:
      command += ["--segment_file", self.segment_file.get_path()]

    create_executable("rnn.sh", command)

  def run(self):
    subprocess.check_call(["./rnn.sh"])

  @classmethod
  def hash(cls, kwargs):
    kwargs.pop("time_rqmt")
    kwargs.pop("mem_rqmt")
    return super().hash(kwargs)
