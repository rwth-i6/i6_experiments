import os
from typing import Optional, Type
from sisyphus import Job, Task, tk
import logging
from i6_core.tools.download import DownloadJob
import subprocess as sp
from i6_core.tools.git import CloneGitRepositoryJob

__all__ = ["SetupFairseqJob"]

import logging
import os
import shutil
import subprocess as sp
from typing import Optional

from sisyphus import *

from i6_core.util import get_executable_path

import numpy as np


# "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_u_audio_preprocessing/Wav2VecUFeaturizeAudioJob.TyJVh2DlIY8F/output/audio_features/valid.tsv"
# This function is highly customized for the Librispeech dataset in /u/corpora i6
def get_w2vu_librispeech_transcription(
    manifest_tsv_path: str,
    output_path: str,
    corpus_root: str = "/u/corpora/speech/LibriSpeech/LibriSpeech/train-other-960/",
):
    with open(manifest_tsv_path, "r") as f:
        audios = f.readlines()

    if len(audios) < 2:
        raise ValueError("The manifest file must contain at least two lines. (Fist line is header root)")

    transcriptions = {}

    def join_libri(path):
        return os.path.join(corpus_root, path)

    audios = [line.split("\t")[0] for line in audios[1:]]
    audio_transcriptions = []
    for audio in audios:
        # 147156/2522-147156-0057.flac
        last_slash = audio.rfind("/")
        last_dot = audio.rfind(".")
        audio = audio[last_slash + 1 : last_dot]  # 2522-147156-0057

        path_file = audio.split("-")  # [2522, 147156, 0057]

        path = os.path.join(*path_file[:-1])  #  2522/147156
        specific_file = path_file[-1]  #  0057

        trasncription_file_path = os.path.join(
            path, "-".join(path_file[:-1]) + ".trans.txt"
        )  # 522/147156/522-147156.trans.txt

        if not trasncription_file_path in transcriptions:
            with open(join_libri(trasncription_file_path), "r") as f:
                transcriptions[trasncription_file_path] = f.readlines()

        transcription_text = transcriptions[trasncription_file_path]

        for line in transcription_text:
            if line.startswith(audio):
                transcription_text = line
                break

        transcription_text = transcription_text.split(" ", 1)[1]  # Remove the audio name

        audio_transcriptions.append(transcription_text)

    all_transcriptions_in_a_string = "".join(audio_transcriptions)

    with open(output_path, "w") as f:
        f.write(all_transcriptions_in_a_string)


def cf(file_path: str) -> str:
    return sp.check_output(["cf", file_path]).decode().strip()


def cache_path(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")
    if os.path.isfile(path):
        return cf(path)

    all_cached_paths = []
    for root, dirs, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.exists(file_path):
                temp_path = cf(file_path)
                all_cached_paths.append(temp_path)
    return os.path.commonpath(all_cached_paths)


def compare_npy_files(file1, file2, num_samples=1000, atol=1e-8):
    # Load the arrays
    arr1 = np.load(file1)
    arr2 = np.load(file2)

    # Slice to the first num_samples
    arr1_slice = arr1[:num_samples]
    arr2_slice = arr2[:num_samples]

    # Compare shapes
    if arr1_slice.shape != arr2_slice.shape:
        print(f"Shape mismatch: {arr1_slice.shape} vs {arr2_slice.shape}")
        return

    # Compare values
    if np.allclose(arr1_slice, arr2_slice, atol=atol):
        print("The arrays are the same within the tolerance.")
    else:
        print("The arrays differ. Showing indices where they differ:")
        differences = np.where(np.abs(arr1_slice - arr2_slice) > atol)
        for idx in zip(*differences):
            print(f"Index {idx}: {arr1_slice[idx]} != {arr2_slice[idx]}")


class SetupFairseqJob(Job):
    """
    Set up a fairseq repository. Needed to build Cython components.
    """

    def __init__(self, fairseq_root: tk.Path, python_env: Optional[tk.Path] = None, identifier: Optional[str] = None):
        self.fairseq_root = fairseq_root
        self.python_env = python_env
        self.identifier = (
            identifier  # Recommended to use a different unique id for each container that you run fairseq with
        )
        self.out_fairseq_root = self.output_path("fairseq", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        shutil.copytree(self.fairseq_root.get(), self.out_fairseq_root.get(), dirs_exist_ok=True, symlinks=True)

        args = []

        if self.python_env not in [None, ""]:
            args.append(f"source {os.path.join(self.python_env.get_path(), 'bin/activate')} && ")

        args.append("pip install --editable .")

        args = " ".join(args)

        sp.run(args, cwd=self.out_fairseq_root.get_path(), check=True, shell=True)


class CreateWav2VecManifestJob(Job):
    """
    Run the wav2vec_manifest.py script to generate train and validation manifest files.
    """

    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        audio_root: Type[tk.Path],
        valid_percent: float = 0.01,
        ext: str = "flac",
        seed: int = 42,
        path_must_contain: Optional[str] = None,
    ):
        """
        :param fairseq_root: Path to the Fairseq root directory
        :param audio_root: Path to the directory containing audio files
        :param output_dir: Path to the output directory where manifest files will be stored
        :param valid_percent: Percentage of data to be used as validation set
        :param ext: Audio file extension (default: "flac")
        :param seed: Random seed for validation split
        :param path_must_contain: Optional filter for files
        """
        self.fairseq_root = fairseq_root
        self.audio_root = audio_root
        self.valid_percent = valid_percent
        self.ext = ext
        self.seed = seed
        self.path_must_contain = path_must_contain

        self.out_tsv_file = self.output_path("train.tsv")

        # self.rqmt = {"time": 2, "cpu": 1, "mem": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        logging.info(f"Creating manifest files for Fairseq training in:  {self.output_path('train.tsv')}")
        script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/wav2vec_manifest.py")

        sh_call = f"python3  {script_path}  {self.audio_root.get_path()}  --dest . --valid-percent  {self.valid_percent}  --ext  {self.ext}   --seed   {self.seed}"

        if self.path_must_contain:
            sh_call += f"  --path-must-contain  {self.path_must_contain}"

        # self.sh(sh_call)
        sp.check_call(sh_call, shell=True)

        import shutil

        shutil.move("train.tsv", self.out_tsv_file.get_path())


class PrepareWav2VecTextDataJob(Job):
    """
    Run the prepare_text.sh script to preprocess text data for Wav2Vec training.
    """

    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        language: str,
        text_file_path: Type[tk.Path],
        kenlm_root: Optional[Type[tk.Path]],
        fasttext_model: Type[tk.Path],
        sil_prob: float,
        kaldi_root: Optional[Type[tk.Path]] = None,
        fairseq_python_env: Optional[Type[tk.Path]] = None,
        vocab_size: int = 1000,
        tts_engine: str = "espeak",
    ):
        """
        :param fairseq_root: Path to the fairseq root directory
        :param language: Language code for the text data
        :param text_file_path: Path to the input text file
        :param output_dir: Path to the output directory
        :param vocab_size: Vocabulary size for the text data
        :param tts_engine: Text-to-speech engine to use
        :param sil_prob: Probability of silence insertion
        """
        self.fairseq_root = fairseq_root
        self.kenlm_root = kenlm_root
        self.language = language
        self.text_file_path = text_file_path
        self.vocab_size = vocab_size
        self.tts_engine = tts_engine  # Supported values: "espeak", "espeak-mg", "G2P" (english only)
        self.sil_prob = sil_prob
        self.kaldi_root = kaldi_root
        self.fasttext_model = fasttext_model
        self.fairseq_python_env = fairseq_python_env

        self.out_text_dir = self.output_path("text", directory=True)

        self.rqmt = {"time": 2000, "cpu": 1, "mem": 180}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        logging.info(f"Preparing text data for Wav2Vec training in: {self.output_path}")
        logging.info(f"Kaldi root is: {os.environ.copy().get('KALDI_ROOT')}")

        if self.vocab_size <= 0:
            raise ValueError("Vocabulary size must be greater than 0")

        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/prepare_text.sh"
        )

        env = os.environ.copy()

        # Ensure fairseq_python_env is provided
        if not self.fairseq_python_env:
            raise ValueError("fairseq_python_env must be provided to specify the Python environment")
        if self.kaldi_root:
            env["KALDI_ROOT"] = self.kaldi_root.get_path()

        env["PYTHONPATH"] = f"{self.fairseq_root.get_path()}:" + env.get("PYTHONPATH", "")
        env["FAIRSEQ_ROOT"] = self.fairseq_root.get_path()
        env["PATH"] = f"{self.fairseq_python_env.get_path()}/bin:" + env["PATH"]
        env["VIRTUAL_ENV"] = self.fairseq_python_env.get_path()
        env["KENLM_ROOT"] = self.kenlm_root.get_path() if self.kenlm_root else ""

        activate_path = os.path.join(self.fairseq_python_env.get_path(), "bin/activate")
        activate_path = ["source", activate_path]

        sh_call = [
            "zsh",
            script_path,
            self.language,
            self.text_file_path.get_path(),
            self.out_text_dir.get_path(),
            str(self.vocab_size),
            self.tts_engine,
            self.fasttext_model.get_path(),
            str(self.sil_prob),
        ]

        logging.info(f"Running command: {' '.join(activate_path + ['&&'] + sh_call)}")
        logging.info(f"Environment: {env}")

        sp.run(" ".join(activate_path + [" && "] + sh_call), shell=True, env=env, check=True)
        # sp.run(sh_call, shell=True, env=env, check=True)


def get_rvad_root():
    """
    Clone the rVADfast repository and return its path.
    """
    out_repository = CloneGitRepositoryJob(
        url="https://github.com/zhenghuatan/rVADfast.git",
        checkout_folder_name="rVADfast",
        commit="0ed4c1246ad5fdb1cead801153f455b9cf6d569b",
    ).out_repository.copy()
    tk.register_output("rVADfast", out_repository)
    rvad_root = out_repository.get_path()
    return out_repository


def get_fairseq_root(
    python_env: Optional[tk.Path] = "/usr/bin/python",
    commit: Optional[str] = "ecbf110e1eb43861214b05fa001eff584954f65a",
    fairseq_root: Optional[tk.Path] = None,
    identifier: Optional[str] = None,
):
    """
    :param python_env: path to the python environment where fairseq will be installed
    """

    if fairseq_root is None:
        fairseq_root = CloneGitRepositoryJob(
            "https://github.com/facebookresearch/fairseq", checkout_folder_name="fairseq", commit=commit
        ).out_repository
    else:
        if commit is not None:
            logging.warning("Commit will be ignored since fairseq_root is provided")

    return SetupFairseqJob(fairseq_root, python_env, identifier).out_fairseq_root
