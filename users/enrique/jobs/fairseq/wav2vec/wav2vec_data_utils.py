import gzip
import json
import os
from typing import Dict, Optional, Type, Any, List
from sisyphus import Job, Task, tk, tools
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

import random


def calculate_all_configs(configs, seed_range: range):
    """
    Calculate the number of models to be trained based on the provided configurations.
    This is a mini-task that runs before the main training task.
    """
    n_models = 1
    for value in configs.values():
        if isinstance(value, list):
            n_models *= len(value)

        elif isinstance(value, range):
            n_models *= len(value)

        else:
            n_models *= 1

    n_models *= len(seed_range)

    all_configs = [{}]

    keys = list(configs.keys())

    for key in keys:
        all_configs_aux = []
        for con in all_configs:
            if isinstance(configs[key], list):
                for value in configs[key]:
                    new_config = con.copy()
                    new_config[key] = value
                    all_configs_aux.append(new_config)
            elif isinstance(configs[key], range):
                for value in configs[key]:
                    new_config = con.copy()
                    new_config[key] = value
                    all_configs_aux.append(new_config)
            else:
                new_config = con.copy()
                new_config[key] = configs[key]
                all_configs_aux.append(new_config)
        all_configs = all_configs_aux

    all_configs_aux = []
    for con in all_configs:
        for seed in seed_range:
            new_config = con.copy()
            new_config["common.seed"] = seed
            all_configs_aux.append(new_config)
    all_configs = all_configs_aux

    assert len(all_configs) == n_models

    return all_configs, n_models

    
def w2v_manifest(
    root,
    dest,
    ext,
    valid_percent,
    path_must_contain,
    seed: Optional[int] = 42,
    name_the_manifests_just_train_and_valid: bool = False,
    max_n_audios_per_manifest: Optional[int] = None,
):
    assert valid_percent >= 0 and valid_percent <= 1.0

    if not os.path.exists(dest):
        os.makedirs(dest)

    dir_path = os.path.realpath(root)
    data_corpus = os.path.basename(dir_path.rstrip(os.sep))

    search_path = os.path.join(dir_path, "**/*." + ext)
    rand = random.Random(seed)

    if name_the_manifests_just_train_and_valid:
        valid_name = "valid.tsv"
    else:
        valid_name = data_corpus + "_valid.tsv"

    valid_f = open(os.path.join(dest, valid_name), "w") if valid_percent > 0 else None

    import glob
    import soundfile

    if name_the_manifests_just_train_and_valid:
        tsv_name = "train.tsv"
    else:
        if valid_percent == 0:
            tsv_name = data_corpus + ".tsv"
        else:
            tsv_name = data_corpus + "_train.tsv"

    with open(os.path.join(dest, tsv_name), "w") as train_f:
        print(dir_path, file=train_f)

        if valid_f is not None:
            print(dir_path, file=valid_f)

        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            if max_n_audios_per_manifest is not None and max_n_audios_per_manifest <= 0:
                break

            max_n_audios_per_manifest = max_n_audios_per_manifest - 1 if max_n_audios_per_manifest is not None else None

            if path_must_contain and path_must_contain not in file_path:
                continue

            frames = soundfile.info(fname).frames
            dest = train_f if rand.random() >= valid_percent else valid_f
            print("{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest)
    if valid_f is not None:
        valid_f.close()



# "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_u_audio_preprocessing/Wav2VecUFeaturizeAudioJob.TyJVh2DlIY8F/output/audio_features/valid.tsv"
# This function is highly customized for the Librispeech dataset in /u/corpora i6
def get_w2vu_librispeech_transcription(
    manifest_tsv_path: str,
    output_path: Optional[str] = None,
    corpus_root: str = "/u/corpora/speech/LibriSpeech/LibriSpeech/train-other-960/",
):

    if output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            return f.read()

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

    all_transcriptions_in_a_string = "".join(audio_transcriptions).lower()

    if output_path is not None and not os.path.exists(output_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(all_transcriptions_in_a_string)

    return all_transcriptions_in_a_string


class GetW2VLibriSpeechGroundTruthJob(Job):
    """
    Job to get the transcription of the Librispeech dataset.
    It will download the transcription file and save it in the output path.
    """

    def __init__(
        self, audio_manifest: tk.Path, librispeech_subset: str = "train-other-960", corpus_root: Optional[tk.Path] = tk.Path("/u/corpora/speech/LibriSpeech/LibriSpeech/")
    ):
        self.audio_manifest = audio_manifest
        self.corpus_root = corpus_root 
        self.librispeech_subset = librispeech_subset

        self.output_ground_truth = self.output_path("ground_truth.txt")
        self.ground_truth_formatted = self.output_path("ground_truth_formatted.txt.gz")
        self.ground_truth_stm = self.output_path("ground_truth.stm")



    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        manifest_path = ""
        if os.path.exists(self.audio_manifest.get_path() + "/preprocessed_audio.tsv"):
            manifest_path = self.audio_manifest.get_path() + "/preprocessed_audio.tsv"
        elif os.path.exists(self.audio_manifest.get_path() + "/train.tsv"):
            manifest_path = self.audio_manifest.get_path() + "/train.tsv"
        else:
            raise FileNotFoundError("Could not find a valid manifest file in the provided audio_manifest path.")

        logging.info(f"Processing manifest: {manifest_path}")

        get_w2vu_librispeech_transcription(
            manifest_path,
            self.output_ground_truth.get_path(),
            self.corpus_root.get_path() + "/" + self.librispeech_subset,
        )

        ground_truth_path = self.output_ground_truth.get_path()

        ids = []
        ground_truths = []

        logging.info(f"Formatting ground truth from {ground_truth_path} using manifest {manifest_path}")

        # Build IDs from the manifest
        with open(manifest_path, "r") as manifest_file:
            lines = manifest_file.readlines()
            for line in lines[1:]:  # skip header
                line = line.strip().split(" ")[0]
                slash_index = line.rfind('/')
                flac_index = line.rfind('.flac')
                line_id = line[slash_index + 1:flac_index]
                # Example: "3703-71550-0004" → "train/3703-71550-0004/3703-71550-0004"
                ids.append(f"{self.librispeech_subset}/{line_id}/{line_id}")

        # Load transcriptions
        with open(ground_truth_path, "r") as gt_file:
            for line in gt_file:
                line = line.strip().replace('\n', '').upper()
                ground_truths.append(line)

        assert len(ids) == len(ground_truths), \
            f"Length mismatch: {len(ids)} IDs vs {len(ground_truths)} transcriptions"

        ground_truth_formatted = {ids[i]: ground_truths[i] for i in range(len(ids))}

        # Save gzipped JSON
        with gzip.open(self.ground_truth_formatted.get_path(), "wt", encoding="utf-8") as f:
            json.dump(ground_truth_formatted, f, ensure_ascii=False, indent=2)

        logging.info(f"Formatted ground truth saved to {self.ground_truth_formatted.get_path()}")

        create_stm_from_json(self.ground_truth_formatted.get_path(), self.ground_truth_stm.get_path())



def cf(file_path: str) -> str:
    return sp.check_output(["cf", file_path]).decode().strip()


def cache_path(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")
    if os.path.isfile(path):
        return cf(path)

    all_cached_paths = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for name in files:
                file_path = os.path.join(root, name)
                if os.path.exists(file_path):
                    temp_path = cache_path(file_path)
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
        vocab_size: int = 0,  # TODO: THIS IS NOT THE VOCAB SIZE, IT IS THE MIN NUMBER OF TIMES A PHONEME NEEDS TO APPEAR FOR IT TO NOT BE DISCARTED
        tts_engine: str = "espeak",
        lm_pruning: Optional[list] = [0,0,0,3], # In this example, we only keep the 4-grams that appear at least 4 times
        three_gram_lm: Optional[bool] = False, # TODO: fully integrate differnt ngram orders
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
        self.lm_pruning = lm_pruning
        self.three_gram_lm = three_gram_lm


        self.out_text_dir = self.output_path("text", directory=True)
        self.processed_wrd_data = self.output_path("text/lm.upper.lid.txt")
        self.processed_wrd_LM_bin = self.output_path("text/kenlm.wrd.o4.bin")
        self.processed_wrd_LM_arpa = self.output_path("text/kenlm.wrd.o4.arpa")
        self.processed_phn_data_and_LM = self.output_path("text/phones", directory=True)
        self.processed_dict_phn_txt = self.output_path("text/phones/dict.phn.txt")
        self.lexicon_lst = self.output_path("text/lexicon_filtered.lst")
        self.phn_words_sil_hlg_graph_path = self.output_path("text/fst/phn_to_words_sil/HLG.phn.kenlm.wrd.o4.fst")
        self.phn_words_sil_kaldi_dict_path = self.output_path("text/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o4.txt")
        self.text_phones_lm_filtered_04_bin = self.output_path("text/phones/lm.phones.filtered.04.bin")
        self.text_phones_lm_filtered_04_arpa = self.output_path("text/phones/lm.phones.filtered.04.arpa")
        self.text_phones_lm_filtered_06_bin = self.output_path("text/phones/lm.phones.filtered.06.bin")
        self.text_phones_lm_filtered_06_arpa = self.output_path("text/phones/lm.phones.filtered.06.arpa")
        self.text_phones_lm_filtered_txt = self.output_path("text/phones/lm.phones.filtered.txt")

        self.rqmt = {"time": 2000, "cpu": 1, "mem": 180}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        logging.info(f"Preparing text data for Wav2Vec training in: {self.output_path}")
        logging.info(f"Kaldi root is: {os.environ.copy().get('KALDI_ROOT')}")

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

        
        if self.three_gram_lm:
            script_path = "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/fairseq_2025_06_02-proj/recipe/i6_experiments/users/enrique/experiments/wav2vec_u/prepare_text_3gram.sh"
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
                "0", "0", "0", # no pruning
            ]
        else:
            script_path = "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/fairseq_2025_06_02-proj/recipe/i6_experiments/users/enrique/experiments/wav2vec_u/prepare_text.sh"
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
                str(self.lm_pruning[0]),
                str(self.lm_pruning[1]),
                str(self.lm_pruning[2]),
                str(self.lm_pruning[3]),
            ]

        logging.info(f"Running command: {' '.join(sh_call)}")
        logging.info(f"Environment: {env}")

        sp.run(sh_call, env=env, check=True)
        #sp.run(" ".join(activate_path + [" && "] + sh_call), shell=True, env=env, check=True)
        # sp.run(sh_call, shell=True, env=env, check=True)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v:
                if (k == "lm_pruning" and v == [0,0,0,3]) or (k == "three_gram_lm" and v == False):
                    continue
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement
        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))

class PrepareWav2VecTextDataJob_V2(Job):
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
        vocab_size: int = 1000,  # TODO: THIS IS NOT THE VOCAB SIZE, IT IS THE MIN NUMBER OF TIMES A PHONEME NEEDS TO APPEAR FOR IT TO NOT BE DISCARTED
        tts_engine: str = "espeak",
        lm_pruning: Optional[list] = [0,0,0,3], # In this example, we only keep the 4-grams that appear at least 4 times
        three_gram_lm: Optional[bool] = False, # TODO: fully integrate differnt ngram orders
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
        self.lm_pruning = lm_pruning
        self.three_gram_lm = three_gram_lm


        self.out_text_dir = self.output_path("text", directory=True)
        self.processed_phn_data_and_LM = self.output_path("text/phones", directory=True)

        self.rqmt = {"time": 2000, "cpu": 1, "mem": 180}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        logging.info(f"Preparing text data for Wav2Vec training in: {self.output_path}")
        logging.info(f"Kaldi root is: {os.environ.copy().get('KALDI_ROOT')}")

        if self.vocab_size <= 0:
            raise ValueError("Vocabulary size must be greater than 0")

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

        
        if self.three_gram_lm:
            script_path = "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/fairseq_2025_06_02-proj/recipe/i6_experiments/users/enrique/experiments/wav2vec_u/prepare_text_3gram.sh"
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
                "0", "0", "0", # no pruning
            ]
        else:
            script_path = "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/fairseq_2025_06_02-proj/recipe/i6_experiments/users/enrique/experiments/wav2vec_u/prepare_text.sh"
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
                str(self.lm_pruning[0]),
                str(self.lm_pruning[1]),
                str(self.lm_pruning[2]),
                str(self.lm_pruning[3]),
            ]

        logging.info(f"Running command: {' '.join(sh_call)}")
        logging.info(f"Environment: {env}")

        sp.run(sh_call, env=env, check=True)
        #sp.run(" ".join(activate_path + [" && "] + sh_call), shell=True, env=env, check=True)
        # sp.run(sh_call, shell=True, env=env, check=True)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v:
                if (k == "lm_pruning" and v == [0,0,0,3]) or (k == "three_gram_lm" and v == False):
                    continue
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement
        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))


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
            # logging.warning("Commit will be ignored since fairseq_root is provided")
            pass

    return SetupFairseqJob(fairseq_root, python_env, identifier).out_fairseq_root



def create_stm_from_json(input_gz_path: str, output_stm_path: str):
    """
    Reads a gzipped JSON file (ground truth) and creates a reference STM file.

    The STM format is:
    <filename> <channel> <speaker> <start_time> <end_time> [<label>] <transcript>
    Although start_time, end_time, label and speaker are not available in the source JSON, we will create placeholders.
    """
    print(f"Processing '{input_gz_path}' to create '{output_stm_path}'...")
    with gzip.open(input_gz_path, 'rt', encoding='utf-8') as f_in:
        data = json.load(f_in)

    with open(output_stm_path, 'w', encoding='utf-8') as f_out:
        # Sort by utterance ID for a consistent and ordered output
        for utt_id in sorted(data.keys()):
            transcript = data[utt_id].strip()
            if not transcript:
                continue  # Skip empty transcripts

            # Use placeholders for fields not present in the source JSON
            filename = utt_id
            channel = '1'
            speaker = utt_id  # Often, utterance ID is used as speaker ID
            start_time = '0.00'
            end_time = '1000.00'  # An arbitrary but large enough value
            label = '<d0>'
            
            # Write the formatted line to the STM file
            f_out.write(f"{filename} {channel} {speaker} {start_time} {end_time} {label} {transcript}\n")
        f_out.write(';; LABEL "d0" "default0" "all other segments of category 0"')


def create_ctm_from_json(input_gz_path: str, output_ctm_path: str):
    """
    Reads a gzipped JSON file (system recognition) and creates a CTM file.
    The CTM format is:
    <utterance_id> <channel> <start_time> <duration> <word> <confidence>
    Although start_time, duration, and confidence are not available in the source JSON, we will create placeholders.
    """
    print(f"Processing '{input_gz_path}' to create '{output_ctm_path}'...")
    with gzip.open(input_gz_path, 'rt', encoding='utf-8') as f_in:
        data = json.load(f_in)

    with open(output_ctm_path, 'w', encoding='utf-8') as f_out:
        # Sort by utterance ID for a consistent and ordered output
        for utt_id in sorted(data.keys()):
            transcript = data[utt_id].strip()
            words = transcript.split()
            
            # Since time/confidence are not available, we create placeholders
            channel = '1'
            confidence = '0.99'
            word_duration = 0.0  # Arbitrary duration for each word
            current_time = 0.0

            for word in words:
                # Write the formatted line for each word to the CTM file
                f_out.write(f"{utt_id} {channel} {current_time:.2f} {word_duration:.2f} {word} {confidence}\n")
                # Increment time for the next word to ensure a valid sequence
                current_time += word_duration

# def calculate_perplexity(arpa_file, text_file, normalize=False):
#     import kenlm
#     import math
#     model = kenlm.Model(arpa_file)
    
#     used_tokens = set()

#     total_log_prob = 0.0
#     total_tokens = 0
    
#     with open(text_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
            
#             # Compute log-probability (base 10) for the sentence
#             # KenLM returns log10 probabilities, so we convert to log2 later
#             log_prob = model.score(line, bos=True, eos=True)
#             total_log_prob += log_prob
            
#             # Count tokens (words) in the sentence
#             # KenLM includes <s> and </s> in scoring, so we adjust token count
#             tokens = len(line.split()) + 2  # +2 for <s> and </s>
#             total_tokens += tokens
    
#     # Avoid division by zero
#     if total_tokens == 0:
#         raise ValueError("Text file is empty or contains no valid tokens.")
    
#     # Convert log10 probability to log2 and compute perplexity
#     # KenLM's score is in log10, so convert to log2: log2(p) = log10(p) / log10(2)
#     log2_prob = total_log_prob / math.log10(2)
    
#     # Perplexity = 2^(-1/N * sum(log2(P)))
#     average_log2_prob = log2_prob / total_tokens
#     perplexity = 2 ** (-average_log2_prob)
    
#     return perplexity

def calculate_perplexity(arpa_file, text_file):
    import kenlm
    import math

    # Load KenLM model
    model = kenlm.Model(arpa_file)

    # --- Extract vocabulary (only from unigrams in ARPA) ---
    vocab = set()
    if arpa_file.endswith(".arpa"):
        with open(arpa_file, 'r', encoding='utf-8', errors='ignore') as f:
            in_unigrams = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("\\1-grams:"):
                    in_unigrams = True
                    continue
                if in_unigrams:
                    if line.startswith("\\2-grams:"):
                        break
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[1]
                        vocab.add(token)
    else:
        vocab = None  # Can't extract vocab from binary model

    # --- Compute perplexity ---
    total_log_prob = 0.0
    total_tokens = 0
    used_tokens = set()

    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            log_prob = model.score(line, bos=True, eos=True)
            total_log_prob += log_prob

            tokens = line.split()
            used_tokens.update(tokens)
            total_tokens += len(tokens) + 1  # +1 for </s>

    if total_tokens == 0:
        raise ValueError("Text file is empty or contains no valid tokens.")

    # Convert log10 -> log2
    log2_prob = total_log_prob / math.log10(2)
    avg_log2_prob = log2_prob / total_tokens
    perplexity = 2 ** (-avg_log2_prob)

    # --- Compute vocabulary usage ---
    if vocab:
        vocab_usage = len(used_tokens & vocab) / len(vocab)
    else:
        vocab_usage = 0.0

    return perplexity, vocab_usage, log2_prob, avg_log2_prob



class CalculatePerplexityJob(Job):
    """
    Job to calculate the perplexity of a text file given a KenLM language model in ARPA format.
    """

    def __init__(self, arpa_file: tk.Path, text_file: tk.Path):
        self.arpa_file = arpa_file # either arpa or binary LM
        self.text_file = text_file

        self.output_perplexity = self.output_path("perplexity.txt")
        self.output_perplexity_var = self.output_var("perplexity")
        self.vocab_usage_var = self.output_var("vocab_usage")
        self.output_log2_prob_var = self.output_var("log2_prob")
        self.output_avg_log2_prob_var = self.output_var("avg_log2_prob")

        self.rqmt = {"time": 5, "cpu": 1, "mem": 16}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        perplexity, vocab_usage, log2_prob, avg_log2_prob = calculate_perplexity(
            self.arpa_file.get_path(),
            self.text_file.get_path(),
        )

        self.output_perplexity_var.set(perplexity)
        self.vocab_usage_var.set(vocab_usage)
        self.output_log2_prob_var.set(log2_prob)
        self.output_avg_log2_prob_var.set(avg_log2_prob)
        
        with open(self.output_perplexity.get_path(), 'w') as f:
            f.write(f"Perplexity: {perplexity}\n")
        
        print(f"Calculated Perplexity: {perplexity}")

        

def divide_LibriSpeech_960h_transcriptions_into_100h_360h_and_500h(ds_train_other_960_gzs, ds_train_other_500_gz, ds_train_clean_360_gz, ds_train_clean_100_gz):
    """This is just a formatting function to divide the transcriptions of the full 960h LibriSpeech into 3 subsets"""
    import gzip
    import json
    import os

    root_train_other_500 = "/u/corpora/speech/LibriSpeech/LibriSpeech/train-other-500"
    root_train_clean_360 = "/u/corpora/speech/LibriSpeech/LibriSpeech/train-clean-360"
    root_train_clean_100 = "/u/corpora/speech/LibriSpeech/LibriSpeech/train-clean-100"
    
    ds_train_other_500_gz_content = {}
    ds_train_clean_360_gz_content = {}
    ds_train_clean_100_gz_content = {}

    folder_dict_train_other_500 = {
        d: [sd for sd in os.listdir(os.path.join(root_train_other_500, d))
            if os.path.isdir(os.path.join(root_train_other_500, d, sd))]
        for d in os.listdir(root_train_other_500)
        if os.path.isdir(os.path.join(root_train_other_500, d))
    }

    folder_dict_train_clean_360 = {
        d: [sd for sd in os.listdir(os.path.join(root_train_clean_360, d))
            if os.path.isdir(os.path.join(root_train_clean_360, d, sd))]
        for d in os.listdir(root_train_clean_360)
        if os.path.isdir(os.path.join(root_train_clean_360, d))
    }

    folder_dict_train_clean_100 = {
        d: [sd for sd in os.listdir(os.path.join(root_train_clean_100, d))
            if os.path.isdir(os.path.join(root_train_clean_100, d, sd))]
        for d in os.listdir(root_train_clean_100)
        if os.path.isdir(os.path.join(root_train_clean_100, d))
    }   

    for ds_train_other_960_gz in ds_train_other_960_gzs:
        with gzip.open(ds_train_other_960_gz, 'rt', encoding='utf-8') as f:
            data_960h = json.load(f)
        
        for key, value in data_960h.items():
            key_parts_path = key.split('/')
            key_parts = key_parts_path[1].split('-')
            if key_parts[0] in folder_dict_train_other_500:
                if key_parts[1] in folder_dict_train_other_500[key_parts[0]]:
                    ds_train_other_500_gz_content["train-other-500/" + key_parts_path[1] + "/" + key_parts_path[2]] = value
                    continue
            if key_parts[0] in folder_dict_train_clean_360:
                if key_parts[1] in folder_dict_train_clean_360[key_parts[0]]:
                    ds_train_clean_360_gz_content["train-clean-360/" + key_parts_path[1] + "/" + key_parts_path[2]] = value
                    continue
            if key_parts[0] in folder_dict_train_clean_100:
                if key_parts[1] in folder_dict_train_clean_100[key_parts[0]]:
                    ds_train_clean_100_gz_content["train-clean-100/" + key_parts_path[1] + "/" + key_parts_path[2]] = value
                    continue

    with gzip.open(ds_train_other_500_gz, 'wt', encoding='utf-8') as f:
        json.dump(ds_train_other_500_gz_content, f, indent=2)
    with gzip.open(ds_train_clean_360_gz, 'wt', encoding='utf-8') as f:
        json.dump(ds_train_clean_360_gz_content, f, indent=2)
    with gzip.open(ds_train_clean_100_gz, 'wt', encoding='utf-8') as f:
        json.dump(ds_train_clean_100_gz_content, f, indent=2)

class DivideLibriSpeech960hInto100h360h500hJob(Job):
    """
    Divide the LibriSpeech 960h transcriptions into 100h, 360h, and 500h subsets.
    """

    def __init__(
        self,
        ds_train_other_960_gzs: List[tk.Path],
    ):
        self.ds_train_other_960_gzs = ds_train_other_960_gzs
        self.ds_train_other_500_gz = self.output_path("train_other_500.gz")
        self.ds_train_clean_360_gz = self.output_path("train_clean_360.gz")
        self.ds_train_clean_100_gz = self.output_path("train_clean_100.gz")

        self.out_label_path_dict = self.output_path("label_path_dict.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        divide_LibriSpeech_960h_transcriptions_into_100h_360h_and_500h(
            [p.get_path() for p in self.ds_train_other_960_gzs],
            self.ds_train_other_500_gz.get_path(),
            self.ds_train_clean_360_gz.get_path(),
            self.ds_train_clean_100_gz.get_path(),
        )

        label_path_dict = {"path": {"train-clean-100": self.ds_train_clean_100_gz.get_path(), "train-clean-360": self.ds_train_clean_360_gz.get_path(), "train-other-500": self.ds_train_other_500_gz.get_path()}, "score": -1, "epoch": -1}

        with open(self.out_label_path_dict.get_path(), "w") as f:
            f.write(json.dumps(label_path_dict))
            f.write("\n")


class FairseqNormalizeAndPrepareTextJob(Job):
    """
    Normalize and prepare text data for Fairseq training.
    """

    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        text_file_path: Type[tk.Path],
        language: str, # fr, pt, en
        lid_path: Type[tk.Path], # path to a fasttext model
    ):
        
        self.fairseq_root = fairseq_root
        self.text_file_path = text_file_path
        self.language = language
        self.lid_path = lid_path
        self.fairseq_python_env = fairseq_python_env

        self.output_dir = self.output_path("normalized_text", directory=True)
        self.words_txt = self.output_path("normalized_text/words.txt")
        self.dict_txt = self.output_path("normalized_text/dict.txt")
        self.normalize_text_lid = self.output_path("normalized_text/lm.upper.lid.txt")
        
        self.rqmt = {"time": 200, "cpu": 1, "mem": 16}
    
    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
    
    def sh_call_with_environment(self, sh_call):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.fairseq_root.get_path()}:" + env.get("PYTHONPATH", "")
        env["FAIRSEQ_ROOT"] = self.fairseq_root.get_path()
        env["PATH"] = f"{self.fairseq_python_env.get_path()}/bin:" + env["PATH"]

        sp.run(sh_call, env=env, check=True, shell=True)

    def run(self):
        sh_call = ['python', self.fairseq_root.get_path() + '/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py', '--lang', self.language, '--fasttext-model', self.lid_path.get_path(), '<', self.text_file_path.get_path(), '|', 'grep', '-v', "'\\-\\-\\-'", '>', self.normalize_text_lid.get_path()]
        self.sh_call_with_environment(" ".join(sh_call))
        sh_call = ['python', self.fairseq_root.get_path() + '/fairseq_cli/preprocess.py', '--dataset-impl', 'mmap', '--trainpref', self.normalize_text_lid.get_path(), '--only-source', '--destdir', self.output_dir.get_path(), '--padding-factor', '1', '--dict-only']
        self.sh_call_with_environment(" ".join(sh_call))
        sh_call = ['cut', '-f1', "-d'", "'", self.dict_txt.get_path(), '|', 'grep', '-v', '-x', "'[[:punct:]]*'", '|', 'grep', '-Pv', "'\\d\\d\\d\\d\\d+'", '>', self.words_txt.get_path()]
        self.sh_call_with_environment(" ".join(sh_call))

class CreateLexiconAndPhonemizeTextJob(Job):

    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        words_txt: Type[tk.Path],
        phones_txt: Type[tk.Path],
        text_file_path: Type[tk.Path],
        insert_silence_between_words_prob: float = 0.0,
    ):
        self.fairseq_root = fairseq_root
        self.fairseq_python_env = fairseq_python_env
        self.words_txt = words_txt
        self.phones_txt = phones_txt
        self.text_file_path = text_file_path
        self.insert_silence_between_words_prob = insert_silence_between_words_prob

        self.lexicon_lst = self.output_path("lexicon.lst")
        self.lexicon_filtered_lst = self.output_path("lexicon_filtered.lst")
        self.labeled_text = self.output_path("labels/labeled_text.txt")
        self.labels_folder = self.output_path("labels", directory=True)
        self.dict_txt = self.output_path("labels/dict.txt")
        self.dict_phn_txt = self.output_path("labels/dict.phn.txt")
        self.output_label_list_with_bos_eos_pad_unk =self.output_var("label_list_with_bos_eos_pad_unk")

        self.rqmt = {"time": 3000, "cpu": 2, "mem": 128}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def sh_call_with_environment(self, sh_call):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.fairseq_root.get_path()}:" + env.get("PYTHONPATH", "")
        env["FAIRSEQ_ROOT"] = self.fairseq_root.get_path()
        env["PATH"] = f"{self.fairseq_python_env.get_path()}/bin:" + env["PATH"]

        logging.info(f"Running command: {sh_call}") 
        sp.run(sh_call, env=env, check=True, shell=True)

    def run(self):
        sh_call = f"paste {self.words_txt.get_path()} {self.phones_txt.get_path()} > {self.lexicon_lst.get_path()}"
        self.sh_call_with_environment(sh_call)

        sh_call = f"rm -f {self.dict_txt.get_path()}"
        self.sh_call_with_environment(sh_call)

        sh_call = f"python {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref {self.phones_txt.get_path()} --only-source --destdir {self.labels_folder.get_path()} --padding-factor 1 --dict-only"
        self.sh_call_with_environment(sh_call)

        sh_call = f"python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d {self.dict_txt.get_path()} < {self.lexicon_lst.get_path()} > {self.lexicon_filtered_lst.get_path()}"
        self.sh_call_with_environment(sh_call)

        sh_call = f"python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s {self.insert_silence_between_words_prob} --surround --lexicon {self.lexicon_filtered_lst.get_path()} < {self.text_file_path.get_path()} > {self.labeled_text.get_path()}"
        self.sh_call_with_environment(sh_call)

        sh_call = f"cp {self.labels_folder.get_path()}/dict.txt {self.labels_folder.get_path()}/dict.phn.txt"
        self.sh_call_with_environment(sh_call)

        sh_call = f"echo '<SIL> 0' >> {self.dict_phn_txt.get_path()}"
        self.sh_call_with_environment(sh_call)

        sh_call = f"python {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref {self.labeled_text.get_path()} --workers 1 --only-source --destdir {self.labels_folder.get_path()} --srcdict {self.dict_phn_txt.get_path()}"
        self.sh_call_with_environment(sh_call)

        with open(self.dict_phn_txt.get_path(), 'r') as f:
            labels = [line.split()[0] for line in f.readlines()]
        # Add special tokens
        special_tokens = ['<s>', '<pad>', '</s>', '<unk>']

        self.output_label_list_with_bos_eos_pad_unk.set(special_tokens + labels)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if (k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v) and not k=='fairseq_python_env' and not k=='fairseq_root':
                d[k] = v
    
        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement
        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))



class TrainKenLMJob(Job):

    def __init__(
        self,
        kenlm_root: Type[tk.Path],
        text_file_path: Type[tk.Path],
        lm_order: int,
        discount_fallback: bool=True,
        pruning: Optional[List[int]] = None,
        build_binary: bool = True,
    ):
        self.kenlm_root = kenlm_root
        self.text_file_path = text_file_path
        self.lm_order = lm_order
        self.discount_fallback = discount_fallback
        self.pruning = pruning
        self.build_binary = build_binary

        if pruning is None:
            self.output_arpa = self.output_path(f"kenlm.o{lm_order}.arpa")
            if build_binary:
                self.output_bin = self.output_path(f"kenlm.o{lm_order}.bin")
        else:
            assert len(pruning) == lm_order, "Pruning list length must match LM order"
            self.output_arpa = self.output_path(f"kenlm.o{lm_order}.pruned_{'_'.join(map(str, pruning))}.arpa")
            if build_binary:
                self.output_bin = self.output_path(f"kenlm.o{lm_order}.pruned_{'_'.join(map(str, pruning))}.bin")

        self.rqmt = {"time": 1000, "cpu": 1, "mem": 64}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
    
    def run(self):
        sh_call = f"{self.kenlm_root.get_path()}/lmplz -o {self.lm_order} {' --prune ' + ' '.join(map(str, self.pruning)) if self.pruning else ''} < {self.text_file_path.get_path()} --discount_fallback > {self.output_arpa.get_path()}"
        sp.run(sh_call, check=True, shell=True)

        if self.build_binary:
            sh_call = f"{self.kenlm_root.get_path()}/build_binary {self.output_arpa.get_path()} {self.output_bin.get_path()}"
            sp.run(sh_call, check=True, shell=True)

class TakeNRandomLinesFromTextFileJob(Job):
    def __init__(self, input_text_file: Type[tk.Path], n_lines: int):
        self.input_text_file = input_text_file
        self.n_lines = n_lines
        self.output_text_file = self.output_path(f"random_{n_lines}_lines.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import random

        with open(self.input_text_file.get_path(), "r", encoding="utf-8") as f:
            lines = f.readlines()

        random_lines = random.sample(lines, min(self.n_lines, len(lines)))

        with open(self.output_text_file.get_path(), "w", encoding="utf-8") as f:
            f.writelines(random_lines)

class MergeGenerateShardsJob(Job):
    def __init__(
        self, 
        input_text_files: List[Type[tk.Path]], 
        manifest_path: tk.Path,
        decoding_audio_name: str,
    ):
        self.input_text_files = input_text_files
        self.manifest_path = manifest_path
        self.decoding_audio_name = decoding_audio_name
        
        self.output_text_file = self.output_path("transcription.txt")
        self.manifest_path_output = self.output_path("manifest.tsv")

        self.transcription_generated_formatted = self.output_path(f"search_results.py.gz")
        self.transcription_generated_ctm = self.output_path(f"search_results.ctm")

        self.out_label_path_dict = self.output_path(f"labels_path.json")


    def tasks(self):
        yield Task("run", mini_task=True)
        yield Task("format_transcriptions", mini_task=True)

    def run(self):
        num_shards = len(self.input_text_files)
        print(self.input_text_files)
        lines_of_each_shard = []
        for text_file in self.input_text_files:
            with open(text_file.get_path(), "r", encoding="utf-8") as infile:
                lines = infile.readlines()
                lines_of_each_shard.append(lines)

        all_lines_in_order = [None] * sum(len(lines) for lines in lines_of_each_shard)

        for shard_id, shard_lines in enumerate(lines_of_each_shard):
            for line_id, line in enumerate(shard_lines):
                all_lines_in_order[(line_id*num_shards)+shard_id] = line

        print(f"Total lines to write: {len(all_lines_in_order)}")
        print(f"Shards_lengths for {len(lines_of_each_shard)} shards: {[len(lines) for lines in lines_of_each_shard]}")
        
        with open(self.output_text_file.get_path(), "w", encoding="utf-8") as outfile:
            for lines in all_lines_in_order:
                for line in lines:
                    outfile.write(line)

        with open(self.manifest_path.get_path(), "r") as manifest_file:
            with open(self.manifest_path_output.get_path(), "w", encoding="utf-8") as manifest_out:
                lines = manifest_file.readlines()
                manifest_out.writelines(lines)

    def format_transcriptions(self):
        """
        Format the generated transcription to a more readable format.
        """
        import json
        import gzip

        hypothesis_formatted = {}

        ids = []

        with open(self.manifest_path, "r") as manifest_file:
            lines = manifest_file.readlines()
    
            base = self.decoding_audio_name + '/'

            lines = lines[1:]  # Skip header

            for line in lines[:]:  # Skip header
                line = line.split(' ')
                line_id =  line[0]  # e.g. 71550/3703-71550-0004.flac

                slash_index = line_id.rfind('/')
                # find the last occurrence of ".flac"
                flac_index = line_id.rfind('.flac')
                
                # slice between them 3703-71550-0004
                line_id = line_id[slash_index + 1:flac_index]

                ids.append(base + line_id + '/' + line_id)

        transcriptions = []

        with open(self.output_text_file.get_path(), "r") as gen_file:
            for line in gen_file:
                line = line.strip().replace('\n', '').upper()
                transcriptions.append(line)


        assert len(ids) == len(transcriptions), f"Length mismatch: {len(ids)} ids vs {len(transcriptions)} transcriptions"
        for i in range(len(ids)):
            hypothesis_formatted[ids[i]] = transcriptions[i]

        # save as gzipped JSON
        with gzip.open(self.transcription_generated_formatted, "wt", encoding="utf-8") as f:
            json.dump(hypothesis_formatted, f, ensure_ascii=False, indent=2)

        label_path_dict = {"path": {self.decoding_audio_name: self.transcription_generated_formatted.get_path()}, "score": -1, "epoch": -1}

        with open(self.out_label_path_dict.get_path(), "w") as f:
            f.write(json.dumps(label_path_dict))
            f.write("\n")


        create_ctm_from_json(
            self.transcription_generated_formatted.get_path(), self.transcription_generated_ctm.get_path()
        ) 