import soundfile
import glob
import os
import random
import subprocess as sp
import shutil
import logging
from typing import List, Union, Optional 

from sisyphus import Job, Task, tk

from i6_core.lib import corpus

class CreateFairseqLabeledDataJob(Job):
    """
    Creates required task files for wav2vec finetuning with fairseq. This includes the following files:
    - <dest_name>.tsv
    - <dest_name>.ltr
    - <dest_name>.wrd

    For the script see https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py for .tsv creation,
    https://github.com/facebookresearch/fairseq/blob/91c364b7ceef8032099363cb10ba19a85b050c1c/examples/wav2vec/libri_labels.py 
    as well as the issue https://github.com/facebookresearch/fairseq/issues/2493 for .wrd and .ltr creation.

    See also https://github.com/rwth-i6/i6_experiments/blob/main/users/dierkes/preprocessing/wav2vec.py for only 
    manifest creation job (e.g. for fairseq pre-training).
    """

    def __init__(
        self,
        corpus_paths: Union[List[tk.Path], tk.Path],
        file_extension: str = "wav",
        path_must_contain: Optional[str] = None,
        dest_name: str = "train",
    ):
        """
        :param corpus_paths: list of paths or single path to raw audio file directory to be included
        :param file_extension: file extension to look for in corpus_paths
        :param path_must_contain: if set, path must contain this substring
            for a file to be included in the task
        :param dest_name: name of the main label files. Default: "train"
        """
        if not isinstance(corpus_paths, list):
            corpus_paths = [corpus_paths]
        self.corpus_paths = corpus_paths
        assert all([isinstance(path, tk.Path) for path in self.corpus_paths])

        self.file_extension = file_extension
        self.path_must_contain = path_must_contain

        self.out_labels_path = self.output_path("labels", directory=True)

        self.out_tsv_path = self.output_path(f"labels/{dest_name}.tsv")
        self.out_ltr_path = self.output_path(f"labels/{dest_name}.ltr")
        self.out_wrd_path = self.output_path(f"labels/{dest_name}.wrd")

        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}


    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        self.create_tsv_and_labels()

    def create_tsv_and_labels(self):
        """
        Creates both .tsv file and labels (.ltr and .wrd files) from the given corpora.
        """        
        common_dir = self.get_common_dir()

        tsv = open(self.out_tsv_path, "w")
        ltr = open(self.out_ltr_path, "w")
        wrd = open(self.out_wrd_path, "w")

        # write common directory (root) to tsv files
        print(common_dir, file=tsv)

        # iterate over all corpora
        for corpus_path in self.corpus_paths:
            corpus_object = corpus.Corpus()
            corpus_object.load(corpus_path.get())
            for segment in corpus_object.segments():
                # extract audio path and transcription from segment
                audio_path = segment.recording.audio
                audio_trans = segment.orth
                assert os.path.exists(audio_path), f"Path {audio_path} does not exist."

                rel_audio_path = os.path.relpath(audio_path, common_dir)
                frames = soundfile.info(audio_path).frames
                
                # write audio path to tsv file
                print(f"{rel_audio_path}\t{frames}", file=tsv)

                # write transcription to transcription files
                print(
                    " ".join(list(audio_trans.replace(" ", "|"))) + " |",
                    file=ltr,
                )
                print(audio_trans, file=wrd)

        tsv.close()
        ltr.close()
        wrd.close()

    def get_common_dir(self):
        """
        Returns the common directory of all audios given in the corpora.
        """
        audio_paths = []
        # iterate over all corpora
        for corpus_path in self.corpus_paths:
            corpus_object = corpus.Corpus()
            corpus_object.load(corpus_path.get())
            for recording in corpus_object.all_recordings():
                audio_path = recording.audio
                audio_paths.append(audio_path)
                assert os.path.exists(audio_path), f"Path {audio_path} does not exist."
        common_dir = os.path.commonpath(audio_paths)
        return common_dir


class MergeLabeledFairseqDataJob(Job):
    """
    Merge multiple labeled fairseq data directories into one. Each directory must contain the following files:
    - <name>.tsv
    - <name>.ltr
    - <name>.wrd

    If create_letter_dict is set to True, the following file will be created:
    - dict.ltr.txt
    """
    def __init__(
        self,
        labeled_data_paths: Union[List[tk.Path], tk.Path],
        create_letter_dict: bool = True,
    ):
        if not isinstance(labeled_data_paths, list):
            labeled_data_paths = [labeled_data_paths]
        self.labeled_data_paths = labeled_data_paths
        assert all([isinstance(path, tk.Path) for path in self.labeled_data_paths])

        self.create_letter_dict = create_letter_dict

        self.out_task_path = self.output_path("task", directory=True)
        self.out_dict_ltr_path = self.output_path("task/dict.ltr.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        self.merge()
        if self.create_letter_dict:
            self.create_dict_ltr()

    def merge(self):
        for path in self.labeled_data_paths:
            assert os.path.exists(path.get()), f"Path {path} does not exist."

            # iterature through directory "path" and copy all files to self.out_task_path directory
            for file in os.listdir(path.get()):
                if file.endswith(".tsv") or file.endswith(".ltr") or file.endswith(".wrd"):
                    src = os.path.join(path.get(), file)
                    dst = os.path.join(self.out_task_path.get(), file)
                    assert not os.path.exists(dst), f"'{dst}' exists, two inputs have the same names"
                    shutil.copyfile(src, dst)
                else:
                    logging.warning(f"Ignoring File {filename}; is not a valid fairseq file.")
        

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


class FairseqDecodingJob(Job):
    """
    Runs decoding with fairseq for a given fine-tuned model and a given data set.

    Yields hypotheses and targets files in both word format and specified post-processing format.
    """
    def __init__(
            self,
            fairseq_python_exe: tk.Path,
            fairseq_root: tk.Path,
            model_path: tk.Path,
            data_path: tk.Path,
            gen_subset: str,
            nbest: int = 1,
            w2l_decoder: str = "viterbi",
            lm_path: Optional[tk.Path] = None,
            lm_lexicon: Optional[tk.Path] = None,
            lm_weight: float = 2.0,
            word_score: float = -1.0,
            sil_weight: float = 0.0,
            criterion: str = "ctc",
            labels: str = "ltr",
            post_process: str = "letter",
            max_tokens: int = 4000000,
    ):
        """
        :param fairseq_python_exe: path to fairseq python executable
        :param fairseq_root: path to fairseq root directory
        :param model_path: path to fine-tuned model
        :param data_path: path to the data to be decoded. Should be a directory containing
            - [gen_subset].tsv
            - [gen_subset].[labels] (e.g. ltr)
        :param gen_subset: name of the subset to be decoded. The files in data_path should be named correspondingly.
        :param nbest: number of nbest hypotheses to output, default 1
        :param w2l_decoder: decoder to use, default "viterbi". Can be "viterbi", "kenlm" or "fairseqlm".
        :param lm_path: path to language model, default None. Only required if w2l_decoder is "kenlm" or "fairseqlm".
        :param lm_lexicon: path to lexicon for language model, default None.
            Only required if w2l_decoder is "kenlm" or "fairseqlm".
        :param lm_weight: weight of language model, default 2.0
        :param word_score: word score, default 1.0
        :param sil_weight: silence weight, default 0.0
        :param criterion: criterion to use, default "ctc". At the moment, only "ctc" is tested.
        :param labels: labels to use, default "ltr". At the moment, only "ltr" is tested. 
            The data_path folder should contain a file with the corresponding file extension.
        :param post_process: post processing to use, default "letter".
            Can be "letter", "sentencepiece", "wordpiece", "letter", "silence", "subword_nmt", "_EOW" or "none".
            (just use "letter" for now)
        :param max_tokens: maximum number of tokens to decode, default 4000000
        """

        # inputs
        self.fairseq_python_exe = fairseq_python_exe
        self.fairseq_root = fairseq_root
        self.model_path = model_path
        self.data_path = data_path
        self.gen_subset = gen_subset
        self.nbest = nbest

        assert w2l_decoder in ["viterbi", "kenlm", "fairseqlm"], f"Unknown decoder {w2l_decoder}."
        self.w2l_decoder = w2l_decoder

        if self.w2l_decoder in ["kenlm", "fairseqlm"]:
            assert lm_path is not None, "lm_path must be set if w2l_decoder is kenlm or fairseqlm."
            assert lm_lexicon is not None, "lm_lexicon must be set if w2l_decoder is kenlm or fairseqlm."
        self.lm_path = lm_path
        self.lm_lexicon = lm_lexicon

        self.lm_weight = lm_weight
        self.word_score = word_score
        self.sil_weight = sil_weight

        if criterion not in ["ctc"]:
            logging.warning(f"Unknown criterion {criterion}. This might not work.")
        self.criterion = criterion

        if labels not in ["ltr"]:
            logging.warning(f"Unknown labels {labels}. This might not work.")
        self.labels = labels

        if post_process not in ["letter"]:
            logging.warning(f"Unknown post_process {post_process}. This might not work.")
        self.post_process = post_process

        self.max_tokens = max_tokens

        # outputs
        self.out_results = self.output_path("results", directory=True)
        model_name = os.path.basename(self.model_path.get())
        self.out_hyp_word = self.output_path(f"results/hypo.word-{model_name}-{self.gen_subset}.txt")
        self.out_hyp_units = self.output_path(f"results/hypo.units-{model_name}-{self.gen_subset}.txt")
        self.out_ref_word = self.output_path(f"results/ref.word-{model_name}-{self.gen_subset}.txt")
        self.out_ref_units = self.output_path(f"results/ref.units-{model_name}-{self.gen_subset}.txt")

        # rqmt
        self.rqmt = {"time": 6, "mem": 8, "cpu": 1, "gpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        my_env = os.environ
        if "PYTHONPATH" in my_env:
            my_env["PYTHONPATH"] += f":{self.fairseq_root.get()}"
        else:
            my_env["PYTHONPATH"] = f"{self.fairseq_root.get()}"
        run_cmd = self._get_run_cmd()
        logging.info(f"Running decoding with following run command: {' '.join(run_cmd)}")
        sp.check_call(run_cmd, env=my_env)

    def _get_run_cmd(self):
        fairseq_infer_path = os.path.join(self.fairseq_root.get(), "examples", "speech_recognition", "infer.py")
        run_cmd = [
            self.fairseq_python_exe.get(),
            fairseq_infer_path,
            self.data_path.get(),
            "--task", "audio_finetuning",
            "--nbest", str(self.nbest),
            "--path", self.model_path.get(),
            "--gen-subset", self.gen_subset,
            "--w2l-decoder", self.w2l_decoder,
            "--lm-model", self.lm_path.get(),
            "--lexicon", self.lm_lexicon.get(),
            "--lm-weight", str(self.lm_weight),
            "--word-score", str(self.word_score),
            "--sil-weight", str(self.sil_weight),
            "--criterion", self.criterion,
            "--labels", self.labels,
            "--post-process", self.post_process,
            "--max-tokens", str(self.max_tokens),
            "--results-path", self.out_results.get()
        ]

        return run_cmd

