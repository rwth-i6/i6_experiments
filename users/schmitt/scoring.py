import zipfile
from sisyphus import Path, Job, Task

from i6_core.returnn.config import ReturnnConfig
from i6_core.lib import corpus


class ProcessConcatStmAndCtm(Job):
  """
  ProcessConcatStmAndCtm is a class that reads CTM and STM files, extracts sequence tags,
  replaces these tags with sequential identifiers, and outputs modified CTM and STM files.
  The class also generates a mapping file that records the original sequence tags and their
  corresponding new identifiers.

  This can be used in cases where the seq tags are too long for Sclite to process, in which case an error is thrown
  by Sclite. Example for the error message:

  align_ctm_to_stm: File identifiers do not match,
  hyp file 'dev-other/... and ref file 'dev-other/... not synchronized
  sclite: Alignment failed.  Exiting.

  Attributes:
      ctm_path (Path): Path to the input CTM file.
      stm_path (Path): Path to the input STM file.
      seq_tag_prefix (str): Prefix used to identify sequence tags within the CTM file.
      out_stm (Path): Path for the output STM file with updated identifiers.
      out_ctm (Path): Path for the output CTM file with updated identifiers.
      out_seq_tag_mapping (Path): Path for the output file containing the sequence tag mapping.
  """

  def __init__(self, ctm_path: Path, stm_path: Path, seq_tag_prefix: str):
    # Initialize paths for input files and prefix for sequence tags.
    self.ctm_path = ctm_path
    self.stm_path = stm_path
    self.seq_tag_prefix = seq_tag_prefix

    # Define output paths for modified STM and CTM files, and the mapping file.
    self.out_stm = self.output_path("out.stm")
    self.out_ctm = self.output_path("out.ctm")
    self.out_seq_tag_mapping = self.output_path("seq_tag_mapping.txt")

  def tasks(self):
    # Define the task to run as a mini-task.
    yield Task("run", mini_task=True)

  def run(self):
    # Read all lines from the CTM file.
    with open(self.ctm_path.get_path(), "r") as f:
      ctm_lines = f.readlines()

    # Initialize an empty list to store sequence tags.
    seq_tags = []
    # Extract sequence tags from lines that start with the specified prefix.
    for ctm_line in ctm_lines:
      if ctm_line.startswith(f";; {self.seq_tag_prefix}"):
        # Extract and clean the sequence tag, adding it to seq_tags list.
        seq_tags.append(ctm_line.split()[1].strip())

    # Combine all CTM lines back into a single string.
    new_ctm_content = "".join(ctm_lines)
    print(seq_tags)

    # Read all lines from the STM file.
    with open(self.stm_path.get_path(), "r") as f:
      stm_lines = f.readlines()

    # Combine all STM lines back into a single string.
    new_stm_content = "".join(stm_lines)

    # Replace each sequence tag in both CTM and STM contents with a unique identifier.
    for i, seq_tag in enumerate(seq_tags):
      new_ctm_content = new_ctm_content.replace(seq_tag, f"seq_{i}")
      new_stm_content = new_stm_content.replace(seq_tag, f"seq_{i}")

    # Write the updated CTM content to the output CTM file.
    with open(self.out_ctm.get_path(), "w") as f:
      f.write(new_ctm_content)

    # Write the updated STM content to the output STM file.
    with open(self.out_stm.get_path(), "w") as f:
      f.write(new_stm_content)

    # Write the mapping of original sequence tags to new identifiers to the mapping file.
    with open(self.out_seq_tag_mapping.get_path(), "w") as f:
      print(seq_tags)
      for i, seq_tag in enumerate(seq_tags):
        print(f"{seq_tag} seq_{i}")
        f.write(f"{seq_tag} seq_{i}\n")

