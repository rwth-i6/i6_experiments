import copy
from collections import defaultdict
import random
from functools import lru_cache
import os
import pickle
import numpy as np
from tqdm import tqdm

from sisyphus import Job, Task, tk

from i6_core.util import uopen
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.datasets.loquacious import PrepareLoquaciousDatasetJob
from i6_core.text.processing import HeadJob, PipelineJob

from i6_experiments.common.datasets.loquacious.corpus import get_bliss_corpus_dict
from i6_experiments.users.rossenbach.lib.hdf import SimpleHDFWriter

def get_dev_segments():
    dev = get_bliss_corpus_dict()["dev.all"]
    dev_all_segments = SegmentCorpusJob(dev, 1).out_single_segment_files[1]

    def shuffle_and_head(segment_file, num_lines):
        # only shuffle, this is deterministic
        shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
            segment_file=segment_file,
            split={"shuffle": 1.0},
            shuffle=True
        )
        segment_file = shuffle_segment_file_job.out_segments["shuffle"]
        return HeadJob(segment_file, num_lines=num_lines).out

    dev_all_subset = shuffle_and_head(dev_all_segments, 3000)
    return dev_all_subset



class PrepareLoquaciousTrainSmallSpeakerLabelsJob(PrepareLoquaciousDatasetJob):
    """
    Prepare the Loquacious dataset from HuggingFace.
    """

    def __init__(
            self,
            hf_home_dir: tk.Path,

    ):
        self.hf_home_dir = hf_home_dir

        self.out_speaker_hdf = self.output_path("speaker_labels.hdf")
        self.out_num_speakers = self.output_var("num_speakers")
        self.out_speaker_dict = self.output_path("speaker_dict.pkl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 16, "mem": 32, "time": 4})

    def run(self):
        os.environ["HF_HOME"] = self.hf_home_dir.get_path()
        from datasets import load_dataset
        dataset = load_dataset(
            path="speechbrain/LoquaciousSet",
            name="small",
            split="train",
            num_proc=8,
        )

        speakers = {"unknown": 0}
        segment_speaker_ids = {}

        progress = tqdm(dataset, file=open("/dev/null", "w"))
        for i, entry in enumerate(progress):
            if i % 100 == 0:
                # do not bloat log files, so print manually
                segment_name = f"train.small/{entry['ID']}/0"
                if entry["spk_id"] is None:
                    id = 0
                else:
                    if entry["spk_id"] not in speakers.keys():
                        id = len(speakers)
                        speakers[entry["spk_id"]] = id
                    else:
                        id = speakers[entry["spk_id"]]
                segment_speaker_ids[segment_name] = id

        speaker_by_index = {id: name for name, id in speakers.items()}
        pickle.dump(speaker_by_index, uopen(self.out_speaker_dict, "wb"))

        num_speakers = len(speakers)
        self.out_num_speakers.set(num_speakers)

        hdf_writer = SimpleHDFWriter(self.out_speaker_hdf.get_path(), dim=num_speakers, ndim=1)
        for segment_name, id in segment_speaker_ids.items():
            hdf_writer.insert_batch(np.asarray([[id]], dtype="int32"), [1], [segment_name])
        hdf_writer.close()
