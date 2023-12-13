import re
from typing import Dict

from sisyphus import Job, Path, Task


class ConcatStmFileJob(Job):
  def __init__(self, stm_file: Path, concat_num: int):
    self.input_stm_file = stm_file
    self.concat_num = concat_num

    self.out_stm_file = self.output_path("out_stm.stm")

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  def run(self):
    with open(self.input_stm_file, "r") as input_file:
      with open(self.out_stm_file, "w+") as output_file:
        tags = []  # store to-be-conacatenated seq tags
        texts = []  # store to-be-conacatenated transcriptions
        last_seq_id = ""  # store previous seq id to decide whether we want to concatenate
        first_tag = True  # keep track if we are still in the process of dumping the first seq tag

        for line in input_file:
          # line format we are interested in: <tag> <number> <number>  <start>   <end> <flag> <transcription>
          if not line.startswith(";;"):
            m = re.match(
              "^([a-zA-Z0-9_/-]+)\\s+1\\s+([a-zA-Z0-9_]+)\\s+([0-9.]+)\\s+(inf)\\s+<([a-zA-Z0-9,\\-]+)>(.*)$", line)
            assert m, "unexpected line: %r" % line
            tag, tag2, start, end, flags, text = m.groups()
            assert start == "0.00" and end == "inf" and flags == "d0"

            seq_id = tag.split("-")[:2]  # tag has form number-number-number
            seq_id = "-".join(seq_id)  # the first two numbers are the identifier
            if seq_id == last_seq_id or last_seq_id == "":
              # if this tag belongs to the same seq as previously, we can continue concatenating
              # tags.append("dev-other/%s/%s" % (tag, tag))  # append the full seq tag for librispeech
              # texts.append(text)

              if len(tags) == self.concat_num:
                # if we reach the number of desired concatenations, write the concatenated seq to the output file
                prefix = "" if first_tag else "\n"  # leading new line only if we dumped something before
                output_file.write(prefix + "%s 1 %s %s\t%s <%s> %s" % (
                  ";".join(tags), tag2, start, end, flags, " ".join(texts)
                ))
                # start new empty buffer
                tags = []
                texts = []
                first_tag = False
              tags.append("dev-other/%s/%s" % (tag, tag))  # append the full seq tag for librispeech
              texts.append(text)
            else:
              # if this tag does not belong to the same seq as previously, we dump the currently buffered tags
              # and start a new buffer with the current seq tag
              prefix = "" if first_tag else "\n"  # leading new line only if we dumped something before
              if len(tags) > 0:
                output_file.write(prefix + "%s 1 %s %s\t%s <%s> %s" % (
                  ";".join(tags), tag2, start, end, flags, " ".join(texts)
                ))

              # start new buffer with current seq tag
              tags = ["dev-other/%s/%s" % (tag, tag)]
              texts = [text]
              first_tag = False

            last_seq_id = seq_id  # update

        if len(tags) > 0:
          # if the buffer is not empty, dump the remaining tags into the output file
          prefix = "" if first_tag else "\n"  # leading new line only if we dumped something before
          output_file.write(prefix + "%s 1 %s %s\t%s <%s> %s" % (
            ";".join(tags), tag2, start, end, flags, " ".join(texts)
          ))

        output_file.write('\n;; LABEL "d0" "default0" "all other segments of category 0"\n')


class ConcatSeqTagFileJob(Job):
  def __init__(self, seq_tag_file: Path, concat_num: int):
    self.input_file = seq_tag_file
    self.concat_num = concat_num

    self.out_file = self.output_path("out_seq_tags")

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  def run(self):
    with open(self.input_file, "r") as input_file:
      with open(self.out_file, "w+") as output_file:
        first_tag = True
        tags = []
        last_seq_id = ""
        for line in input_file:
          tag = line.strip()
          seq_id = tag.split("/")[1]
          seq_id = seq_id.split("-")[:2]
          seq_id = "-".join(seq_id)

          if seq_id == last_seq_id or last_seq_id == "":
            # tags.append(tag)
            if len(tags) == self.concat_num:
              prefix = "" if first_tag else "\n"  # leading new line only if we dumped something before
              output_file.write(prefix + ";".join(tags))
              tags = []
              first_tag = False
            tags.append(tag)  # might need to move this back above the preceding if
          else:
            if len(tags) > 0:
              prefix = "" if first_tag else "\n"  # leading new line only if we dumped something before
              output_file.write(prefix + ";".join(tags))
            tags = [tag]
            first_tag = False

          last_seq_id = seq_id

        if len(tags) > 0:
          prefix = "" if first_tag else "\n"  # leading new line only if we dumped something before
          output_file.write(prefix + ";".join(tags))


class WordsToCTMJob(Job):
  def __init__(self, stm_path: Path, words_path: Path, dataset_name: str):
    self.words_path = words_path
    self.stm_path = stm_path
    self.dataset_name = dataset_name

    self.out_ctm_file = self.output_path("out.ctm")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):
    from recipe.i6_experiments.users.schmitt.experiments.config.concat_seqs.scoring import ScliteHubScoreJob
    ScliteHubScoreJob.create_ctm(
      name=self.dataset_name, ref_stm_filename=self.stm_path.get_path(), source_filename=self.words_path.get_path(),
      target_filename=self.out_ctm_file.get_path())


class WordsToCTMJobV2(Job):
  def __init__(self, words_path: Path):
    self.words_path = words_path

    self.out_ctm_file = self.output_path("out.ctm")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):
    from recipe.i6_experiments.users.schmitt.experiments.config.concat_seqs.scoring import ScliteJob
    ScliteJob.create_ctm(
      source_filename=self.words_path.get_path(),
      target_filename=self.out_ctm_file.get_path())


def get_concat_dataset_dict(original_dataset_dict: Dict, seq_list_file: Path, seq_len_file: Path):
  return {
    "class": "ConcatSeqsDataset",
    "dataset": original_dataset_dict,
    "seq_list_file": seq_list_file,
    "seq_len_file": seq_len_file
  }
