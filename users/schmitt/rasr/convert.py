from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand

import subprocess
import tempfile
import shutil
import ast
import json


class RASRLatticeToCTMJob(Job):
  def __init__(self, rasr_exe_path, flf_lattice_tool_config, crp, lattice_path, time_rqtm=1, mem_rqmt=2):
    self.lattice_path = lattice_path
    self.crp = crp
    self.flf_lattice_tool_config = flf_lattice_tool_config
    self.rasr_exe_path = rasr_exe_path
    self.mem_rqmt = mem_rqmt
    self.time_rqmt = time_rqtm

    self.out_ctm = self.output_path("lattice.ctm.1")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

  def run(self):
    config, post_config = build_config_from_mapping(self.crp, {
      "corpus": "flf-lattice-tool.corpus",
      "lexicon": "flf-lattice-tool.lexicon"})
    config.flf_lattice_tool._update(self.flf_lattice_tool_config)
    config.flf_lattice_tool.network.archive_reader.path = self.lattice_path
    config.flf_lattice_tool.network.dump_ctm.dump.channel = self.out_ctm

    RasrCommand.write_config(config, post_config, "rasr.config")
    command = [self.rasr_exe_path.get_path(), "--config", "rasr.config", "--*.LOGFILE=sprint.log", ]

    create_executable("run.sh", command)
    subprocess.check_call(["./run.sh"])


class ConvertCTMBPEToWordsJob(Job):
  def __init__(self, bpe_ctm_file):
    self.bpe_ctm_file = bpe_ctm_file

    self.out_ctm_file = self.output_path("words.ctm")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1}, mini_task=True)

  def run(self):
    shutil.copy(self.bpe_ctm_file.get_path(), "bpe_ctm.ctm")

    data = []
    concat = False
    word = ''
    start = None
    duration = 0.0
    with open("bpe_ctm.ctm", 'r') as f:
      for line in f:
        if not line.strip(): continue
        if line.startswith(';'):
          if concat:
            print('segment finished inproperly: %s' % tokens)
            tokens[-2] = word
            data.append(' '.join(tokens))
            concat = False
            duration = 0.0
            word = ''
          data.append(line.strip())
          continue
        tokens = line.split()
        assert len(tokens) == 6
        bpe = tokens[-2]
        if concat:
          tokens[2] = start
          tokens[3] = '%.3f' % (float(tokens[3]) + duration)
          tokens[-2] = word + bpe
        if bpe.endswith('@@'):
          start = tokens[2]
          duration = float(tokens[3])
          word += bpe.replace('@@', '')
          concat = True
        else:
          concat = False
          duration = 0.0
          if tokens[-2] not in ["[NOISE]", "[LAUGHTER]", "[VOCALIZED-NOISE]", "!NULL", "[SILENCE]", "[EOS]"]:
            data.append(' '.join(tokens))
          word = ''
    if concat:
      tokens[-2] = word
      if tokens[-2] not in ["[NOISE]", "[LAUGHTER]", "[VOCALIZED-NOISE]", "[SILENCE]", "[EOS]"]:
        data.append(' '.join(tokens))
    with open(self.out_ctm_file.get_path(), 'w+') as f:
      f.write('\n'.join(data))


class BPEJSONVocabToRasrFormatsJob(Job):
  """
  Convert a JSON-like monophone-eow vocab into a label file, allophone file and state-tying file for RASR to use.
  """

  def __init__(self, json_vocab_path, blank_idx):
    self.blank_idx = blank_idx
    self.json_vocab_path = json_vocab_path
    self.out_rasr_label_file = self.output_path("out_rasr_label_file")
    self.out_allophones = self.output_path("out_allophones")
    self.out_state_tying = self.output_path("out_state_tying")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 1, "time": 1})

  def run(self):
    # load json vocab
    with open(self.json_vocab_path.get_path(), "r") as f:
      json_vocab = ast.literal_eval(f.read())
      json_vocab = {k: v for k, v in json_vocab.items() if k not in ["<s>", "</s>"]}

    # create the label file (<phon> <idx>)
    label_file = []
    for bpe, v in json_vocab.items():
      label_file.append("%s %d\n" % (bpe, int(v)))
    # add blank label
    label_file.append("%s %d\n" % ("[blank]", self.blank_idx))

    with open(self.out_rasr_label_file.get_path(), "w+") as f:
      f.writelines(label_file)

    # create allophone file (one phoneme with ctx and without state info per line)
    allophones = []
    for bpe, v in json_vocab.items():
      allophones.append("%s{#+#}\n" % bpe)
      allophones.append("%s{#+#}@i\n" % bpe)
      allophones.append("%s{#+#}@f\n" % bpe)
      allophones.append("%s{#+#}@i@f\n" % bpe)
    # add all for cases for the blank label
    allophones.append("[blank]{#+#}\n")
    allophones.append("[blank]{#+#}@i\n")
    allophones.append("[blank]{#+#}@f\n")
    allophones.append("[blank]{#+#}@i@f\n")

    with open(self.out_allophones.get_path(), "w+") as f:
      f.writelines(allophones)

    # create state-tying file (<phon with ctx and state> <idx>)
    state_tying = []
    for bpe, v in json_vocab.items():
      state_tying.append("%s{#+#}.0 %d\n" % (bpe, int(v)))
      state_tying.append("%s{#+#}@i.0 %d\n" % (bpe, int(v)))
      state_tying.append("%s{#+#}@f.0 %d\n" % (bpe, int(v)))
      state_tying.append("%s{#+#}@i@f.0 %d\n" % (bpe, int(v)))

    state_tying.append("%s %d\n" % ("[blank]{#+#}.0", self.blank_idx))
    state_tying.append("%s %d\n" % ("[blank]{#+#}@i.0", self.blank_idx))
    state_tying.append("%s %d\n" % ("[blank]{#+#}@f.0", self.blank_idx))
    state_tying.append("%s %d\n" % ("[blank]{#+#}@i@f.0", self.blank_idx))

    with open(self.out_state_tying.get_path(), "w+") as f:
      f.writelines(state_tying)


class PhonJSONVocabToRasrFormatsJob(Job):
  """
  Convert a JSON-like monophone-eow vocab into a label file, allophone file and state-tying file for RASR to use.
  """
  def __init__(self, json_vocab_path, blank_idx):
    self.blank_idx = blank_idx
    self.json_vocab_path = json_vocab_path
    self.out_rasr_label_file = self.output_path("out_rasr_label_file")
    self.out_allophones = self.output_path("out_allophones")
    self.out_state_tying = self.output_path("out_state_tying")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 1, "time": 1})

  def run(self):
    # load json vocab
    with open(self.json_vocab_path.get_path(), "r") as f:
      json_vocab = json.load(f)

    # create the label file (<phon> <idx>)
    label_file = []
    for k, v in json_vocab.items():
      phon = k.split("{")[0]
      if k.endswith("@f.0"):  # mark eof word phonemes with # at the end
        phon += "#"
      label_file.append("%s %d\n" % (phon, int(v)))
    # add blank label
    label_file.append("%s %d\n" % ("[blank]", self.blank_idx))

    with open(self.out_rasr_label_file.get_path(), "w+") as f:
      f.writelines(label_file)

    # create allophone file (one phoneme with ctx and without state info per line)
    allophones = []
    for k, v in json_vocab.items():
      phon, ctx = k.split("{")
      phon_not_eow = phon + "{" + ctx
      phon_eow = phon + "#{" + ctx
      if phon_not_eow.endswith("}.0"):
        allophones.append("%s\n" % (phon_not_eow[:-2]))
        allophones.append("%s@i\n" % (phon_not_eow[:-2]))
        allophones.append("%s@f\n" % (phon_not_eow[:-2]))
        allophones.append("%s@i@f\n" % (phon_not_eow[:-2]))
      else:
        allophones.append("%s\n" % (phon_eow[:-4]))
        allophones.append("%s@i\n" % (phon_eow[:-4]))
        allophones.append("%s@f\n" % (phon_eow[:-4]))
        allophones.append("%s@i@f\n" % (phon_eow[:-4]))
    # add all for cases for the blank label
    allophones.append("[blank]{#+#}\n")
    allophones.append("[blank]{#+#}@i\n")
    allophones.append("[blank]{#+#}@f\n")
    allophones.append("[blank]{#+#}@i@f\n")

    with open(self.out_allophones.get_path(), "w+") as f:
      f.writelines(allophones)

    # create state-tying file (<phon with ctx and state> <idx>)
    state_tying = []
    for k, v in json_vocab.items():
      phon, ctx = k.split("{")
      phon_not_eow = phon + "{" + ctx
      phon_eow = phon + "#{" + ctx
      if phon_not_eow.endswith("}.0"):
        state_tying.append("%s %d\n" % (phon_not_eow, int(v)))
        state_tying.append("%s@i.0 %d\n" % (phon_not_eow[:-2], int(v)))
        state_tying.append("%s@f.0 %d\n" % (phon_not_eow[:-2], int(v)))
        state_tying.append("%s@i@f.0 %d\n" % (phon_not_eow[:-2], int(v)))
      else:
        state_tying.append("%s.0 %d\n" % (phon_eow[:-4], int(v)))
        state_tying.append("%s@i.0 %d\n" % (phon_eow[:-4], int(v)))
        state_tying.append("%s@f.0 %d\n" % (phon_eow[:-4], int(v)))
        state_tying.append("%s@i@f.0 %d\n" % (phon_eow[:-4], int(v)))
    state_tying.append("%s %d\n" % ("[blank]{#+#}.0", self.blank_idx))
    state_tying.append("%s %d\n" % ("[blank]{#+#}@i.0", self.blank_idx))
    state_tying.append("%s %d\n" % ("[blank]{#+#}@f.0", self.blank_idx))
    state_tying.append("%s %d\n" % ("[blank]{#+#}@i@f.0", self.blank_idx))

    with open(self.out_state_tying.get_path(), "w+") as f:
      f.writelines(state_tying)
