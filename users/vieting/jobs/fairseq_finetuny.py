import soundfile
import glob
import os
import random
import subprocess
import os

from i6_core.lib import corpus

from sisyphus import Job, Task, tk


class CreateTaskDataJob(Job):
    """
    Creates required task files for wav2vec finetuning with fairseq. This includes the following files:
    - train.tsv
    - train.ltr
    - train.wrd
    - valid.tsv
    - valid.ltr
    - valid.wrd
    - dict.ltr.txt

    For the script see https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py for .tsv creation,
    https://github.com/facebookresearch/fairseq/blob/91c364b7ceef8032099363cb10ba19a85b050c1c/examples/wav2vec/libri_labels.py aswell as
    the issue https://github.com/facebookresearch/fairseq/issues/2493 for .wrd and .ltr creation.
    """

    def __init__(
        self,
        corpus_paths,
        file_extension="wav",
        valid_percent=0.01,
        seed=42,
        path_must_contain=None
    ):
        """
        :param [tk.Path]|tk.Path audio_dir_path: list of paths or single path to raw audio files to be included
        :param str file_extension: file extension to look for in audio_dir_path
        :param float valid_percent: percentage of files to be in validation set
        :param int seed: random seed for splitting into train and valid set
        :param str|None path_must_contain: if set, path must contain this substring
            for a file to be included in the task
        """
        if not isinstance(corpus_paths, list):
            corpus_paths = [corpus_paths]
        self.corpus_paths = corpus_paths
        assert all([isinstance(path, tk.Path) for path in self.corpus_paths])
        self.file_extension = file_extension
        self.valid_percent = valid_percent
        assert 0.0 <= self.valid_percent <= 1.0
        self.seed = seed
        self.path_must_contain = path_must_contain

        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}

        self.out_task_path = self.output_path("task", directory=True)

        self.out_train_tsv_path = self.output_path("task/train.tsv")
        self.out_valid_tsv_path = self.output_path("task/valid.tsv")
        
        self.out_dict_ltr_path = self.output_path("task/dict.ltr.txt")

        self.out_train_ltr_path = self.output_path("task/train.ltr")
        self.out_train_wrd_path = self.output_path("task/train.wrd")
        self.out_valid_ltr_path = self.output_path("task/valid.ltr")
        self.out_valid_wrd_path = self.output_path("task/valid.wrd")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        #self.create_manifest()
        self.create_tsv_and_labels()
        self.download_dict_ltr()
    
    def get_common_dir(self):
        """
        Returns the common directory of all audios given in the corpora.
        """
        # TODO test this
        common_dir = None
        for corpus_path in self.corpus_paths:
            c = corpus.Corpus()
            c.load(corpus_path.get())
            for segment in c.segments():
                audio_path = segment.recording.audio
                assert os.path.exists(audio_path), f"Path {audio_path} does not exist."
                if common_dir is None:
                    common_dir = os.path.dirname(audio_path)
                else:
                    common_dir = os.path.commonpath([common_dir, os.path.dirname(audio_path)])
        return common_dir

    def create_tsv_and_labels(self):
        """
        Creates both tsv files (train.tsv, valid.tsv) and labels (train.ltr, train.wrd, valid.ltr, valid.wrd) from the given corpora.
        """
        rand = random.Random(self.seed)
        
        common_dir = self.get_common_dir()

        valid_tsv = (
            open(self.out_valid_tsv_path, "w")
            if self.valid_percent > 0
            else None
        )

        train_tsv = open(self.out_train_tsv_path, "w")

        valid_ltr = (
            open(self.out_valid_ltr_path, "w")
            if self.valid_percent > 0
            else None
        )

        train_ltr = open(self.out_train_ltr_path, "w")

        valid_wrd = (
            open(self.out_valid_wrd_path, "w")
            if self.valid_percent > 0
            else None
        )

        train_wrd = open(self.out_train_wrd_path, "w")

        if valid_tsv is not None:
            print(common_dir, file=valid_tsv)

        print(common_dir, file=train_tsv)

        # iterate over all corpora
        for corpus_path in self.corpus_paths:
            c = corpus.Corpus()
            c.load(corpus_path.get())
            for segment in c.segments():
                audio_path = segment.recording.audio
                audio_trans = segment.orth
                assert os.path.exists(audio_path), f"Path {audio_path} does not exist."

                rel_audio_path = os.path.relpath(audio_path, common_dir)
                frames = soundfile.info(audio_path).frames

                # determine whether to write to train or valid
                if rand.random() >= self.valid_percent:
                    tsv_out = train_tsv
                    ltr_out = train_ltr
                    wrd_out = train_wrd
                else:
                    tsv_out = valid_tsv
                    ltr_out = valid_ltr
                    wrd_out = valid_wrd
                
                # write to tsv files
                print(f"{rel_audio_path}\t{frames}", file=tsv_out)

                # write to transcription files
                print(
                    " ".join(list(audio_trans.replace(" ", "|"))) + " |",
                    file=ltr_out,
                )
                print(audio_trans, file=wrd_out)
        
        # close all files
        if valid_tsv is not None:
            valid_tsv.close()
        if valid_ltr is not None:
            valid_ltr.close()
        if valid_wrd is not None:
            valid_wrd.close()
        
        train_tsv.close()
        train_ltr.close()
        train_wrd.close()



    def download_dict_ltr(self):
        """
        Downloads the dict.ltr.txt file for fairseq wav2vec finetuning.
        """
        subprocess.check_call(["wget", "-O", self.out_dict_ltr_path.get(), "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"])