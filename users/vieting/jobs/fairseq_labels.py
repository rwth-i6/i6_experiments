import soundfile
import glob
import os
import random
import numpy as np
import shutil
import subprocess

from i6_core.lib import corpus

from sisyphus import Job, Task, tk

class FairseqFinetuneCorpusCreationJob(Job):
    """
    Creates a corpus directory for finetuning wav2vec with fairseq. This includes the following files:
    - raw audios files
    - transcriptions.txt
    """
    def __init__(self, corpus_path):
        """
        :param [tk.Path]|tk.Path corpus_path: list of paths or single path to corpus xml files.
        """
        if isinstance(corpus_path, tk.Path):
            self.corpus_paths = [corpus_path]
        else:
            assert isinstance(corpus_path, list)
            self.corpus_paths = corpus_path
        assert min([isinstance(path, tk.Path) for path in self.corpus_paths])

        self.dest_paths = [self.output_path(f"corpus_{i}", directory=True) for i in range(len(self.corpus_paths))]
        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}
        
    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        for corpus_path, dest_path in zip(self.corpus_paths, self.dest_paths):
            print(f"Processing {corpus_path.get()}")
            c = corpus.Corpus()
            c.load(corpus_path.get())
            out_trans_path = os.path.join(dest_path.get(), "transcriptions.txt")
            for segment in c.segments():
                audio_path = segment.recording.audio
                audio_trans = segment.orth
                assert os.path.exists(audio_path), f"Path {audio_path} does not exist."
                # copy audio file
                shutil.copy(audio_path, dest_path)

                # write to transcription file
                with open(out_trans_path, "a") as f:
                    # TODO maybe can write whole path?
                    audio_basename = os.path.basename(audio_path)
                    f.write(f"{audio_basename} {audio_trans}\n")
            print(f"Finished processing {corpus_path.get()}\n")
        print("Finished processing all corpus files.")


class FairseqFinetuneTaskCreationJob(Job):
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
        audio_dir_path,
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
            for a file to be included in the manifest
        """
        if isinstance(audio_dir_path, tk.Path):
            self.audio_dir_paths = [audio_dir_path]
        else:
            assert isinstance(audio_dir_path, list)
            self.audio_dir_paths = audio_dir_path
        assert min([isinstance(path, tk.Path) for path in self.audio_dir_paths])

        
        self.manifest_audio_paths = None

        self.file_extension = file_extension
        self.valid_percent = valid_percent
        assert 0.0 <= self.valid_percent <= 1.0
        self.seed = seed
        self.path_must_contain = path_must_contain

        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}

        self.out_task_path = self.output_path("task", directory=True)

        self.out_train_tsv_path = self.output_path(os.path.join(self.out_task_path.get(), "train.tsv"))
        self.out_valid_tsv_path = self.output_path(os.path.join(self.out_task_path.get(), "valid.tsv"))
        
        self.out_dict_ltr_path = self.output_path(os.path.join(self.out_task_path.get(), "dict.ltr.txt"))

        self.out_train_ltr_path = self.output_path(os.path.join(self.out_task_path.get(), "train.ltr"))
        self.out_train_wrd_path = self.output_path(os.path.join(self.out_task_path.get(), "train.wrd"))
        self.out_valid_ltr_path = self.output_path(os.path.join(self.out_task_path.get(), "valid.ltr"))
        self.out_valid_wrd_path = self.output_path(os.path.join(self.out_task_path.get(), "valid.wrd"))
        """
        self.out_valid_tsv_path = self.output_path("valid.tsv")
        self.out_dict_ltr_path = self.output_path("dict.ltr.txt")
        self.out_train_ltr_path = self.output_path("train.ltr")
        self.out_train_wrd_path = self.output_path("train.wrd")
        self.out_valid_ltr_path = self.output_path("valid.ltr")
        self.out_valid_wrd_path = self.output_path("valid.wrd")
        """

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        #self.create_manifest()
        self.create_tsv()
        self.create_labels(split="train")
        self.create_labels(split="valid")
        self.download_dict_ltr()

    def create_tsv(self):
        """
        Creates the tsv files for fairseq wav2vec finetuning (train.tsv, valid.tsv).
        """
        rand = random.Random(self.seed)
        
        assert self.valid_percent >= 0 and self.valid_percent <= 1.0

        common_dir = os.path.commonpath(self.audio_dir_paths)
        dir_paths = [os.path.realpath(path.get_path()) for path in self.audio_dir_paths]

        all_train_data = []

        valid_f = (
            open(os.path.join(self.output_path("valid.tsv")), "w")
            if self.valid_percent > 0
            else None
        )

        if valid_f is not None:
            print(common_dir, file=valid_f)

        for i, dir_path in enumerate(dir_paths):
            train_data = []
            search_path = os.path.join(dir_path, "**/*." + self.file_extension)

            for path in glob.iglob(search_path, recursive=True):
                frames = soundfile.info(path).frames
                path = os.path.realpath(path)

                #if self.path_must_contain and self.path_must_contain not in path:
                if False:
                    continue

                rel_path = os.path.relpath(path, common_dir)
        
                if rand.random() > self.valid_percent:
                    train_data.append((rel_path, frames))
                else:
                    print(f"{rel_path}\t{frames}", file=valid_f)
            all_train_data.append(train_data)

        if valid_f is not None:
            valid_f.close()

        #with open(self.out_train_tsv_path.get(), "w") as f:
        with open(self.out_train_tsv_path, "w") as f:
            print(common_dir, file=f)
            for train_data in all_train_data:
                for rel_path, frames in train_data:
                    print(f"{rel_path}\t{frames}", file=f)
        

        
    def create_labels(self, split=None):
        """
        Creates the labels for fairseq wav2vec finetuning (train.ltr, train.wrd, valid.ltr, valid.wrd).

        :param str split: either "train" or "valid"
        """
        assert split in ["train", "valid"], f"split must be either 'train' or 'valid', but is {split}"


        transcriptions = {}
        if split == "train":
            tsv_path = self.out_train_tsv_path.get()
            ltr_path = self.out_train_ltr_path.get()
            wrd_path = self.out_train_wrd_path.get()
        else:
            tsv_path = self.out_valid_tsv_path.get()
            ltr_path = self.out_valid_ltr_path.get()
            wrd_path = self.out_valid_wrd_path.get()

        with open(tsv_path, "r") as tsv, open(
            ltr_path, "w"
        ) as ltr_out, open(
            wrd_path, "w"
        ) as wrd_out:
            root = next(tsv).strip()
            print('root',root)
            for line in tsv:
                line = line.strip()

                dir = os.path.dirname(line)

                if dir not in transcriptions:
                    parts = dir.split(os.path.sep)

                    trans_path = f"{os.path.join()}"

                    path = os.path.join(root, dir, "transcriptions.txt")

                    assert os.path.exists(path)
                    texts = {}
                    with open(path, "r") as trans_f:
                        for tline in trans_f:
                            items = tline.strip().split()
                            texts[items[0]] = " ".join(items[1:])

                    transcriptions[dir] = texts
                part = os.path.basename(line).split(".")[0]+f'.{self.file_extension}'

                assert part in transcriptions[dir]
                print(transcriptions[dir][part], file=wrd_out)
                print(
                    " ".join(list(transcriptions[dir][part].replace(" ", "|"))) + " |",
                    file=ltr_out,
                )


    def download_dict_ltr(self):
        """
        Downloads the dict.ltr.txt file for fairseq wav2vec finetuning.
        """
        subprocess.check_call(["wget", "-O", self.out_dict_ltr_path.get(), "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"])
