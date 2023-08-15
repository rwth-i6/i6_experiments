import soundfile
import glob
import os
import random
import subprocess
from typing import List, Union, Optional 
import logging

from sisyphus import Job, Task, tk

from i6_core.lib import corpus

class CreateFairseqLabeledDataJob(Job):
    """
    Creates required task files for wav2vec finetuning with fairseq. This includes the following files:
    - train.tsv
    - train.ltr
    - train.wrd

    If create_letter_dict is set to True, the following file will be created:
    - dict.ltr.txt

    If sample_valid_percent is set > 0, a random sample of the files will be saved as validation set 
    and the following files will be created:
    - valid.tsv
    - valid.ltr
    - valid.wrd
    

    For the script see https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py for .tsv creation,
    https://github.com/facebookresearch/fairseq/blob/91c364b7ceef8032099363cb10ba19a85b050c1c/examples/wav2vec/libri_labels.py 
    as well as the issue https://github.com/facebookresearch/fairseq/issues/2493 for .wrd and .ltr creation.

    See also https://github.com/rwth-i6/i6_experiments/blob/main/users/dierkes/preprocessing/wav2vec.py for only 
    manifest creation job (e.g. for fairseq pre-training).
    """

    def __init__(
        self,
        train_corpus_paths: Union[List[tk.Path], tk.Path],
        valid_corpus_paths: Union[List[tk.Path], tk.Path] = [],
        file_extension: str = "wav",
        sample_valid_percent: float = 0.01,
        seed: int = 42,
        path_must_contain: Optional[str] = None,
        train_dest_name: str = "train",
        valid_dest_name: str = "valid",
        create_letter_dict: bool = True,
    ):
        """
        :param corpus_paths: list of paths or single path to raw audio file directory to be included in training set
        :param valid_corpus_paths: list of paths or single path to raw audio file directory to be included in 
            validation set. Ignored if sample_valid_percent > 0. Default: []
        :param file_extension: file extension to look for in corpus_paths
        :param sample_valid_percent: percentage of files to be randomly sampled as validation set. 
            If > 0, valid_corpus_paths will be ignored. Default: 0.01
        :param seed: random seed for splitting into train and valid set
        :param path_must_contain: if set, path must contain this substring
            for a file to be included in the task
        :param train_dest_name: name of the train label files. Default: "train"
        :param valid_dest_name: name of the valid label files. Default: "valid"
        :param create_letter_dict: if set to True, a dict.ltr.txt file will be created. Default: True
        : 
        """
        if not isinstance(train_corpus_paths, list):
            train_corpus_paths = [train_corpus_paths]
        if not isinstance(valid_corpus_paths, list):
            valid_corpus_paths = [valid_corpus_paths]
    
        self.train_corpus_paths = train_corpus_paths
        self.valid_corpus_paths = valid_corpus_paths
        assert all([isinstance(path, tk.Path) for path in self.train_corpus_paths])
        assert all([isinstance(path, tk.Path) for path in self.valid_corpus_paths])

        self.file_extension = file_extension
        self.valid_percent = sample_valid_percent
        assert 0.0 <= self.valid_percent <= 1.0, "sample_valid_percent must be between 0 and 1."
        self.seed = seed
        self.path_must_contain = path_must_contain
        self.create_letter_dict = create_letter_dict

        if not valid_corpus_paths and self.valid_percent == 0:
            logging.warning("No validation set given and sample_valid_percent is 0. No validation set will be created.")


        self.out_task_path = self.output_path("task", directory=True)

        self.out_train_tsv_path = self.output_path(f"task/{train_dest_name}.tsv")
        self.out_valid_tsv_path = self.output_path(f"task/{valid_dest_name}.tsv")
        
        self.out_dict_ltr_path = self.output_path("task/dict.ltr.txt")

        self.out_train_ltr_path = self.output_path(f"task/{train_dest_name}.ltr")
        self.out_train_wrd_path = self.output_path(f"task/{train_dest_name}.wrd")
        self.out_valid_ltr_path = self.output_path(f"task/{valid_dest_name}.ltr")
        self.out_valid_wrd_path = self.output_path(f"task/{valid_dest_name}.wrd")

        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}


    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        self.create_tsv_and_labels()
        if self.create_letter_dict:
            self.create_dict_ltr()
        if self.valid_percent == 0 and not self.valid_corpus_paths:
            self.delete_valid_files()
        

    def create_tsv_and_labels(self):
        """
        Creates both tsv files (train.tsv, valid.tsv) and labels (train.ltr, train.wrd, valid.ltr, valid.wrd) 
        from the given corpora.
        """
        rand = random.Random(self.seed)
        
        train_common_dir, valid_common_dir = self.get_common_dir()

        valid_tsv = open(self.out_valid_tsv_path, "w")
        dest_tsv = open(self.out_dest_tsv_path, "w")

        valid_ltr = open(self.out_valid_ltr_path, "w")
        dest_ltr = open(self.out_dest_ltr_path, "w")

        valid_wrd = open(self.out_valid_wrd_path, "w") 
        dest_wrd = open(self.out_dest_wrd_path, "w")

        # write common directory (root) to tsv files
        if self.valid_percent > 0:
            print(train_common_dir, file=valid_tsv)
        elif self.valid_corpus_paths:
            print(valid_common_dir, file=valid_tsv)
        print(train_common_dir, file=train_tsv)

        # iterate over all training corpora
        for corpus_path in self.train_corpus_paths:
            corpus_object = corpus.Corpus()
            corpus_object.load(corpus_path.get())
            for segment in corpus_object.segments():
                # extract audio path and transcription from segment
                audio_path = segment.recording.audio
                audio_trans = segment.orth
                assert os.path.exists(audio_path), f"Path {audio_path} does not exist."

                rel_audio_path = os.path.relpath(audio_path, train_common_dir)
                frames = soundfile.info(audio_path).frames

                # determine whether to write to dest or valid
                if rand.random() >= self.valid_percent:
                    tsv_out = dest_tsv
                    ltr_out = dest_ltr
                    wrd_out = dest_wrd
                else:
                    tsv_out = valid_tsv
                    ltr_out = valid_ltr
                    wrd_out = valid_wrd
                
                # write audio path to tsv files
                print(f"{rel_audio_path}\t{frames}", file=tsv_out)

                # write transcription to transcription files
                print(
                    " ".join(list(audio_trans.replace(" ", "|"))) + " |",
                    file=ltr_out,
                )
                print(audio_trans, file=wrd_out)
        
        # iterate over all validation corpora
        if self.valid_corpus_paths and self.valid_percent <= 0:
            for corpus_path in self.valid_corpus_paths:
                corpus_object = corpus.Corpus()
                corpus_object.load(corpus_path.get())
                for segment in corpus_object.segments():
                    # extract audio path and transcription from segment
                    audio_path = segment.recording.audio
                    audio_trans = segment.orth
                    assert os.path.exists(audio_path), f"Path {audio_path} does not exist."

                    rel_audio_path = os.path.relpath(audio_path, valid_common_dir)
                    frames = soundfile.info(audio_path).frames

                    # write audio path to tsv files
                    print(f"{rel_audio_path}\t{frames}", file=valid_tsv)

                    # write transcription to transcription files
                    print(
                        " ".join(list(audio_trans.replace(" ", "|"))) + " |",
                        file=valid_ltr,
                    )
                    print(audio_trans, file=valid_wrd)


        # close all files
        valid_tsv.close()
        valid_ltr.close()
        valid_wrd.close()
        
        dest_tsv.close()
        dest_ltr.close()
        dest_wrd.close()



    def create_dict_ltr(self):
        """
        Creates the dict.ltr.txt file for fairseq wav2vec finetuning, as given by
        https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt
        """
        dict_ltr_content = """| 94802
E 51860
T 38431
A 33152
O 31495
N 28855
I 28794
H 27187
S 26071
R 23546
D 18289
L 16308
U 12400
M 10685
W 10317
C 9844
F 9062
G 8924
Y 8226
P 6890
B 6339
V 3936
K 3456
' 1023
X 636
J 598
Q 437
Z 213
"""
        with open(self.out_dict_ltr_path.get(), 'w') as f:
            f.write(dict_ltr_content)

    
    def get_common_dir(self):
        """
        Returns the common directory of all audios given in the corpora.
        """
        train_common_dir = None
        valid_common_dir = None
        # iterate over all training corpora
        for corpus_path in self.train_corpus_paths:
            corpus_object = corpus.Corpus()
            corpus_object.load(corpus_path.get())
            for segment in corpus_object.segments():
                audio_path = segment.recording.audio
                assert os.path.exists(audio_path), f"Path {audio_path} does not exist."
                if train_common_dir is None:
                    train_common_dir = os.path.dirname(audio_path)
                else:
                    train_common_dir = os.path.commonpath([train_common_dir, os.path.dirname(audio_path)])

        for corpus_path in self.valid_corpus_paths:
            corpus_object = corpus.Corpus()
            corpus_object.load(corpus_path.get())
            for segment in corpus_object.segments():
                audio_path = segment.recording.audio
                assert os.path.exists(audio_path), f"Path {audio_path} does not exist."
                if valid_common_dir is None:
                    valid_common_dir = os.path.dirname(audio_path)
                else:
                    valid_common_dir = os.path.commonpath([valid_common_dir, os.path.dirname(audio_path)])

        return train_common_dir, valid_common_dir

    def delete_valid_files():
        """
        Delete valid set files if no valid set was created.
        """
        os.remove(self.out_valid_tsv_path.get())
        os.remove(self.out_valid_ltr_path.get())
        os.remove(self.out_valid_wrd_path.get())