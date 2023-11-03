from typing import List, Optional

from sisyphus import *

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.audio.encoding import BlissChangeEncodingJob

from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.users.schmitt.datasets.dump import DumpDatasetConfigBuilder
from i6_experiments.users.schmitt.datasets import oggzip, concat
from i6_experiments.users.schmitt.datasets.concat import ConcatStmFileJob, ConcatSeqTagFileJob
from i6_experiments.users.schmitt.corpus.statistics import GetSeqLenFileJob

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE_NEW


class LibrispeechCorpora:
  def __init__(self):
    self.oggzip_paths = {
      "train": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.Cbboscd6En6A/output/out.ogg.zip", cached=True)],
      "cv": [
        Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.RvwLniNrgMit/output/out.ogg.zip", cached=True),
        Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.NSdIHfk1iw2M/output/out.ogg.zip", cached=True)
      ],
      "dev-other": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.NSdIHfk1iw2M/output/out.ogg.zip", cached=True)],
      "dev-clean": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.RvwLniNrgMit/output/out.ogg.zip", cached=True)],
      "test-clean": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.uJ4Bsi72tTTX/output/out.ogg.zip", cached=True)],
      "test-other": [Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.aEkXwA7HziQ1/output/out.ogg.zip", cached=True)],
    }
    self.oggzip_paths["devtrain"] = self.oggzip_paths["train"]

    self.segment_paths = {
      "train": None,
      "cv": Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/processing/PipelineJob.gTty7UHs0uBu/output/out", cached=True),
      "devtrain": None,
      "dev-other": None
    }

    self.corpus_paths = get_bliss_corpus_dict()
    self.corpus_paths_wav = {
      key: BlissChangeEncodingJob(
        corpus_file=val, output_format="wav").out_corpus for key, val in self.corpus_paths.items()
    }

    self.stm_jobs = {
      "dev-other": CorpusToStmJob(self.corpus_paths["dev-other"]),
      "dev-clean": CorpusToStmJob(self.corpus_paths["dev-clean"]),
      "test-other": CorpusToStmJob(self.corpus_paths["test-other"]),
      "test-clean": CorpusToStmJob(self.corpus_paths["test-clean"]),
    }

    self.stm_paths = {
      "dev-other": self.stm_jobs["dev-other"].out_stm_path,
      "dev-clean": self.stm_jobs["dev-clean"].out_stm_path,
      "test-other": self.stm_jobs["test-other"].out_stm_path,
      "test-clean": self.stm_jobs["test-clean"].out_stm_path,
    }

    self.segment_corpus_jobs = {
      "dev-other": SegmentCorpusJob(bliss_corpus=self.corpus_paths["dev-other"], num_segments=1)
    }

    self.seq_len_files = {
      "dev-other": self.get_seq_lens_file(dataset_name="dev-other", concat_num=None),
      "train": self.get_seq_lens_file(dataset_name="train", concat_num=None)
    }

    # update self.segment_paths, self.stm_paths and self.seq_len_files
    self.get_concatenated_seqs("dev-other", concat_nums=[1, 2, 4, 8, 10, 20])

    self.corpus_keys = ("train", "cv", "devtrain")
    self.train_corpus_keys = ("train", "cv", "devtrain")
    test_corpus_keys = ()

    self.partition_epoch = 20

  def get_seq_lens_file(self, dataset_name: str, concat_num: Optional[int]):
    dataset_dict = oggzip.get_dataset_dict(
      oggzip_path_list=self.oggzip_paths[dataset_name],
      bpe_file=None,
      vocab_file=None,
      segment_file=None,
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="sorted_reverse",
      epoch_wise_filter=None,
      use_targets=False
    )

    if concat_num:
      # e.g. "dev-other_concat-2"
      concat_dataset_name = "%s_concat-%d" % (dataset_name, concat_num)
      dataset_dict = concat.get_concat_dataset_dict(
        original_dataset_dict=dataset_dict,
        seq_len_file=self.seq_len_files[dataset_name],
        seq_list_file=self.segment_paths[concat_dataset_name]
      )

    extern_data_dict = {
      "data": {"available_for_inference": True, "shape": (None, 1), "dim": 1},
    }

    dump_dataset_returnn_config = DumpDatasetConfigBuilder.get_dump_dataset_config(
      dataset_dict, extern_data_dict, "data")

    get_seq_len_file_job = GetSeqLenFileJob(
      returnn_config=dump_dataset_returnn_config,
      returnn_root=RETURNN_CURRENT_ROOT,
      returnn_python_exe=RETURNN_EXE_NEW,
      time_rqmt=6 if dataset_name == "train" else 1
    )
    get_seq_len_file_job.add_alias("datasets/LibriSpeech/seq_lens/%s" % (
      concat_dataset_name if concat_num else dataset_name
    ))
    tk.register_output(get_seq_len_file_job.get_one_alias(), get_seq_len_file_job.out_seq_len_file)

    return get_seq_len_file_job.out_seq_len_file

  def get_concatenated_seqs(self, dataset_name: str, concat_nums: List[int]):
    """

    :return:
    """

    self.segment_paths.update({
      "%s_concat-%d" % (dataset_name, concat_num): self.get_concat_seq_tag_file(
        dataset_name=dataset_name,
        concat_num=concat_num
      ) for concat_num in concat_nums
    })

    self.stm_paths.update({
      "%s_concat-%d" % (dataset_name, concat_num): self.get_concat_stm_file(
        dataset_name=dataset_name,
        concat_num=concat_num
      ) for concat_num in concat_nums
    })

    self.seq_len_files.update({
      "%s_concat-%d" % (dataset_name, concat_num): self.get_seq_lens_file(
        dataset_name, concat_num) for concat_num in concat_nums
    })

  def get_concat_seq_tag_file(self, dataset_name: str, concat_num: int):
    concat_seq_tag_file_job = ConcatSeqTagFileJob(
      seq_tag_file=self.segment_corpus_jobs[dataset_name].out_single_segment_files[1], concat_num=concat_num
    )

    # concat_seq_tag_file_job.add_alias("concat-seq-tags-%d" % concat_num)

    return concat_seq_tag_file_job.out_file

  def get_concat_stm_file(self, dataset_name: str, concat_num: int):
    concat_stm_file_job = ConcatStmFileJob(
      stm_file=self.stm_paths[dataset_name],
      concat_num=concat_num
    )

    # concat_stm_file_job.add_alias("concat-stms-%d" % concat_num)

    return concat_stm_file_job.out_stm_file
