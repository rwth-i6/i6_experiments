from typing import List, Optional, Dict

from sisyphus import *

from i6_core.corpus.convert import CorpusToStmJob, CorpusToTxtJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.corpus.filter import FilterCorpusBySegmentDurationJob
from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.tools.download import DownloadJob

from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.users.schmitt.datasets.dump import DumpDatasetConfigBuilder
from i6_experiments.users.schmitt.datasets import oggzip, concat
from i6_experiments.users.schmitt.datasets.concat import ConcatStmFileJob, ConcatSeqTagFileJob
from i6_experiments.users.schmitt.corpus.statistics import GetSeqLenFileJob, GetCorrectDataFilteringJob
from i6_experiments.users.schmitt.rasr.convert import ArpaLMToWordListJob, LabelFileToWordListJob

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE_NEW
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo import model_import as trafo_lm_import
from i6_experiments.users.schmitt.corpus.segment_ends import AugmentCorpusSegmentEndsJob
from i6_experiments.users.schmitt.datasets.bpe_lm import build_lm_training_datasets, LMDatasetSettings


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
      "dev-other": None,
      "dev-clean": None
    }

    self.corpus_paths = get_bliss_corpus_dict()
    self.corpus_paths_wav_imported = get_bliss_corpus_dict(audio_format="wav")
    self.corpus_paths_wav = {
      "dev-other": BlissChangeEncodingJob(
        corpus_file=self.corpus_paths["dev-other"], output_format="wav").out_corpus,
      "dev-clean": self.corpus_paths_wav_imported["dev-clean"]
    }

    self.corpus_paths_w_correct_segment_ends = {
      "dev-other": AugmentCorpusSegmentEndsJob(
        bliss_corpous=self.corpus_paths["dev-other"],
        oggzip_path=self.oggzip_paths["dev-other"][0]
      ).out_bliss_corpus
    }
    dev_other_corpus_by_duration = self.get_corpus_filtered_by_duration_bins("dev-other")
    self.corpus_paths.update(dev_other_corpus_by_duration)

    self.stm_jobs = {
      k: CorpusToStmJob(v) for k, v in self.corpus_paths.items() if (k.startswith("dev-") or k.startswith("test-"))
    }

    self.oggzip_paths.update({k: self.oggzip_paths["dev-other"] for k in dev_other_corpus_by_duration.keys()})
    self.segment_paths.update({
      k: SegmentCorpusJob(
        v, num_segments=1).out_single_segment_files[1] for k, v in dev_other_corpus_by_duration.items()
    })

    self.stm_paths = {k: v.out_stm_path for k, v in self.stm_jobs.items()}

    self.corpus_to_txt_job = CorpusToTxtJob(self.corpus_paths["train-other-960"])

    self.segment_corpus_jobs = {
      "dev-other": SegmentCorpusJob(bliss_corpus=self.corpus_paths["dev-other"], num_segments=1)
    }

    self.seq_len_files = {
      "dev-other": self.get_seq_lens_file(
        dataset_dict=self.get_oggzip_dataset_dict("dev-other"),
        concat_num=None,
        corpus_key="dev-other"
      ),
      "train": self.get_seq_lens_file(
        dataset_dict=self.get_oggzip_dataset_dict("train"),
        concat_num=None,
        corpus_key="train"
      ),
      **{
        f"train_lm_{n}k": self.get_seq_lens_file(
          dataset_dict=build_lm_training_datasets(
            prefix=f"lm_{n}k_train_data",
            librispeech_key="train-other-960",
            bpe_size=n * 1_000,
            settings=LMDatasetSettings(
              train_partition_epoch=1,
              train_seq_ordering="laplace:.1000",
            )
          ).train.as_returnn_opts(),
          concat_num=None,
          corpus_key=f"train_lm_{n}k"
        ) for n in (1, 10)
      }
    }

    self.calulate_correct_data_filtering_thresholds(
      seq_len_file1=self.seq_len_files["train_lm_10k"],
      seq_len_file2=self.seq_len_files["train_lm_1k"],
      max_seq_len1=75,
      corpus_name1="train_lm_10k",
      corpus_name2="train_lm_1k"
    )

    # update self.segment_paths, self.stm_paths and self.seq_len_files
    self.get_concatenated_seqs("dev-other", concat_nums=[1, 2, 4, 8, 10, 20])

    self.corpus_keys = ("train", "cv", "devtrain")
    self.train_corpus_keys = ("train", "cv", "devtrain")
    test_corpus_keys = ()

    self.partition_epoch = 20

    self.arpa_lm_paths = {
      "arpa": DownloadJob(
        url="http://www.openslr.org/resources/11/4-gram.arpa.gz",
        target_filename="librispeech_4-gram.arpa.gz",
      ).out_file
    }

    self.nn_lm_meta_graph_paths = {
      "kazuki-lstm": Path("/u/atanas.gruev/setups/librispeech/2023-08-08-zhou-conformer-transducer/work/crnn/compile/CompileTFGraphJob.0dxq1DSvOxuN/output/graph.meta")
    }
    self.nn_lm_checkpoint_paths = {
      "kazuki-lstm": "/u/zhou/asr-exps/librispeech/dependencies/kazuki_lstmlm_27062019/network.040",
      "kazuki-trafo": Path("/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023")
    }
    self.nn_lm_torch_checkpoints = self.get_nn_lm_torch_checkpoints()

    self.nn_lm_vocab_paths = {
      "kazuki-lstm": Path("/work/asr3/zeyer/schmitt/dependencies/librispeech/lm/kazuki_lstmlm_27062019/vocabulary")
    }

    self.lm_word_list_paths = {
      "arpa": ArpaLMToWordListJob(
        arpa_lm_file_path=self.arpa_lm_paths["arpa"],
        labels_to_exclude=["<s>", "</s>", "<UNK>"],
      ).out_word_list_file,
      "kazuki-lstm": LabelFileToWordListJob(
        label_file_path=self.nn_lm_vocab_paths["kazuki-lstm"],
        labels_to_exclude=["<s>", "</s>", "<sb>", "<UNK>"],
      ).out_word_list_file
    }

  def get_nn_lm_torch_checkpoints(self):
    nn_lm_torch_checkpoints = {
      "kazuki-trafo": trafo_lm_import.get_pt_checkpoint_path()
    }
    tk.register_output("kazuki-trafo-pt-checkpoint", nn_lm_torch_checkpoints["kazuki-trafo"])
    return nn_lm_torch_checkpoints

  def get_seq_lens_file(self, dataset_dict: Dict, concat_num: Optional[int], corpus_key: str):
    if concat_num:
      # e.g. "dev-other_concat-2"
      concat_dataset_name = "%s_concat-%d" % (corpus_key, concat_num)
      dataset_dict = concat.get_concat_dataset_dict(
        original_dataset_dict=dataset_dict,
        seq_len_file=self.seq_len_files[corpus_key],
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
      time_rqmt=8 if "train" in corpus_key else 1
    )
    get_seq_len_file_job.add_alias("datasets/LibriSpeech/seq_lens/%s" % (
      concat_dataset_name if concat_num else corpus_key
    ))
    tk.register_output(get_seq_len_file_job.get_one_alias(), get_seq_len_file_job.out_seq_len_file)

    return get_seq_len_file_job.out_seq_len_file

  def get_oggzip_dataset_dict(self, corpus_key: str):
    return oggzip.get_dataset_dict(
      oggzip_path_list=self.oggzip_paths[corpus_key],
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
        dataset_dict=self.get_oggzip_dataset_dict(dataset_name),
        concat_num=concat_num,
        corpus_key=dataset_name
      ) for concat_num in concat_nums
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

  def get_corpus_filtered_by_duration_bins(self, corpus_key, min_duration=0.1, max_duration=31.0, step_size=5.0):
    import numpy as np
    corpus_by_duration = {
      f"{corpus_key}_{min_duration_}-{max_duration_}": FilterCorpusBySegmentDurationJob(
        bliss_corpus=self.corpus_paths_w_correct_segment_ends[corpus_key],
        min_duration=min_duration_,
        max_duration=max_duration_,
      ).out_corpus for min_duration_, max_duration_ in zip(
        np.arange(min_duration, max_duration, step_size),
        np.arange(min_duration + step_size, max_duration + step_size, step_size)
      )
    }
    return corpus_by_duration

  @staticmethod
  def calulate_correct_data_filtering_thresholds(
          seq_len_file1,
          seq_len_file2,
          max_seq_len1,
          corpus_name1,
          corpus_name2,
  ):
    get_correct_data_filtering_job = GetCorrectDataFilteringJob(
      seq_len_file1=seq_len_file1,
      seq_len_file2=seq_len_file2,
      max_seq_len1=max_seq_len1
    )
    get_correct_data_filtering_job.add_alias(f"datasets/LibriSpeech/filtering_thresholds/{corpus_name1}-to-{corpus_name2}")
    tk.register_output(get_correct_data_filtering_job.get_one_alias(), get_correct_data_filtering_job.out_threshold)
