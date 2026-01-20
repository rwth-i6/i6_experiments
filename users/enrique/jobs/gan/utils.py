import logging
from copy import deepcopy
from itertools import product
import gzip
import json
import os
import shutil
import subprocess as sp
from typing import Any, Dict, List, Optional, Type
from sisyphus import Job, Task, tk, tools
"""
Defines the external software to be used for the Experiments
"""
from sisyphus import tk

from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_core.lm.kenlm import CompileKenLMJob
from recipe.i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from recipe.i6_experiments.common.tools.sctk import compile_sctk


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


# python from apptainer/singularity/docker
RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

MINI_RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/JackTemaki/MiniReturnn", commit="0dc69329b21ce0acade4fcb2bf1be0dc8cc0b121"
).out_repository.copy()
MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

I6_MODELS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_models",
    commit="933c6c13f7d6c74e5a59af0257e17c208dae9da3",
    checkout_folder_name="i6_models",
).out_repository.copy()
I6_MODELS_REPO_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_MODELS"


# FAIRSEQ_PATH = SetupFairseqJob(
#     fairseq_root = CloneGitRepositoryJob(
#         url="git@github.com:facebookresearch/fairseq.git",
#         checkout_folder_name="fairseq",
#         commit="c7c478b92fe135838a2b9ec8341495c732a92401",
#     ).out_repository.copy(),
#     python_exe = "/usr/bin/python3",
# ).out_fairseq_root
# FAIRSEQ_PATH.hash_overwrite = "DEFAULT_FAIRSEQ_PATH"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12").copy()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"

SUBWORD_NMT_REPO = get_returnn_subword_nmt(
    commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
).copy()
SUBWORD_NMT_REPO.hash_overwrite = "I6_SUBWORD_NMT_V2"

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


def resolve_tk_paths(obj):
    # If it's a tk.Path object → replace it
    if isinstance(obj, tk.Path):
        return obj.get_path()
    
    # If dictionary → process values
    if isinstance(obj, dict):
        return {k: resolve_tk_paths(v) for k, v in obj.items()}
    
    # If list → process elements
    if isinstance(obj, list):
        return [resolve_tk_paths(x) for x in obj]
    
    # If tuple → return a new tuple
    if isinstance(obj, tuple):
        return tuple(resolve_tk_paths(x) for x in obj)
    
    # If set → return a new set
    if isinstance(obj, set):
        return {resolve_tk_paths(x) for x in obj}
    
    # Otherwise just return unchanged
    return obj




class ExpandableIterable:
    """
    A wrapper class for iterables that should trigger a branch expansion.
    """
    def __init__(self, data):
        # Ensure the data passed is actually iterable
        try:
            iter(data)
        except TypeError:
            raise TypeError("ExpandableIterable argument must be an iterable")
        self.data = list(data)  # Store as list for repeated iteration

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"ExpandableIterable({self.data})"
    
    def __len__(self):
        return len(self.data)

def extract_branches(d, prefix=()):
    """
    Traverse nested dictionaries and extract all branching points
    only for ExpandableIterable instances.
    Returns a list of: (key_path, iterable_values)
    """
    branches = []

    for k, v in d.items():
        path = prefix + (k,)
        
        # Recursive step for nested dictionaries
        if isinstance(v, dict):
            branches.extend(extract_branches(v, path))
        
        # Logic change: Only checking for our custom class now
        elif isinstance(v, ExpandableIterable):
            branches.append((path, v.data))

    return branches

def set_in_path(d, path, value):
    """Set a value inside nested dictionaries using a tuple of keys."""
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value

def expand_dictionary(original_dict):
    """
    Expand a nested dictionary into a list of dictionaries,
    one per combination of ExpandableIterable values.
    """
    # 1. Extract only the branches marked by ExpandableIterable
    branches = extract_branches(original_dict)
    
    # 2. If no special branches, return the original dict in a list
    if not branches:
        return [deepcopy(original_dict)]

    paths, values_lists = zip(*branches)
    configurations = []

    # 3. Create cartesian product of all ExpandableIterables found
    for combination in product(*values_lists):
        new_dict = deepcopy(original_dict)
        for path, val in zip(paths, combination):
            set_in_path(new_dict, path, val)
        configurations.append(new_dict)

    return configurations




def validate_no_missing_values(d):
    """Raise an error if any value in the nested dictionary is '???'."""
    for k, v in d.items():
        if isinstance(v, dict):
            validate_no_missing_values(v)
        elif v == "???":
            raise ValueError(f"Missing required value for key '{k}'")


def flatten_dict(d, parent_key=""):
    """
    Recursively flatten a nested dictionary.
    Example:
      {"a": {"b": {"c": 5}}} -> {"a.b.c": 5}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def flatten_dictionary_to_text(config_dict):
    """
    1. Validate no '???' values
    2. Flatten nested dictionary into dotted keys
    3. Create a CLI override string of the form:
         key.subkey=value \ 
         key2.subkey2=value2 \
    """
    # 1. Validate
    validate_no_missing_values(config_dict)

    # 2. Flatten
    flat = flatten_dict(config_dict)

    # 3. Format as key=value \ lines
    lines = []
    for k, v in flat.items():

        # Convert Python booleans to lowercase hydra style
        if isinstance(v, bool):
            v = str(v).lower()

        # Convert lists/tuples to comma format
        # if isinstance(v, (list, tuple)):
        #     v = ",".join(map(str, v))
        if isinstance(v, list):
            v = "[" + ",".join(map(str, v)) + "]"
        if isinstance(v, tuple):
            v = "(" + ",".join(map(str, v)) + ")"

        lines.append(f"{k}={v} \\")

    return "\n".join(lines)

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
    
    __sis_hash_exclude__ = {"corpus_root"}

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
        manifest_path_aux = self.audio_manifest.get_path()
        if os.path.isdir(manifest_path_aux):
            manifest_path = self.audio_manifest.get_path() + "/preprocessed_audio.tsv"
        elif os.path.isfile(manifest_path_aux):
            manifest_path = manifest_path_aux
        else:
            raise FileNotFoundError(f"The manifest path {manifest_path_aux} does not exist.")

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
    
    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__)) 

import os
import json
import gzip

def format_transcriptions(
    manifest_path_input, 
    output_text_file_path, 
    transcription_generated_formatted_path, 
    subset_name, 
    out_label_path_dict_path, 
    transcription_generated_ctm_path
):
    hypothesis_formatted = {}
    ids = []

    # Handle manifest path logic
    if os.path.isdir(manifest_path_input):
        manifest_path = os.path.join(manifest_path_input, "preprocessed_audio.tsv")
    elif os.path.isfile(manifest_path_input):
        manifest_path = manifest_path_input
    else:
        raise FileNotFoundError(f"The manifest path {manifest_path_input} does not exist.")
    
    # Process manifest file to extract IDs
    with open(manifest_path, "r") as manifest_file:
        lines = manifest_file.readlines()
        
        lines = lines[1:]  # Skip header line

        for line in lines:
            parts = line.split(' ')
            line_id = parts[0]  # e.g. 71550/3703-71550-0004.flac

            slash_index = line_id.rfind('/')
            flac_index = line_id.rfind('.flac')
            
            # Extract ID: 3703-71550-0004
            if slash_index != -1 and flac_index != -1:
                clean_id = line_id[slash_index + 1:flac_index]
                # Construct full ID path
                ids.append(subset_name + "/" + clean_id + '/' + clean_id)
            else:
                # Fallback if format is unexpected
                ids.append(subset_name + line_id)

    transcriptions = []

    # Read generated transcriptions
    with open(output_text_file_path, "r") as gen_file:
        for line in gen_file:
            line = line.strip().replace('\n', '').upper()
            transcriptions.append(line)

    # Verify alignment
    assert len(ids) == len(transcriptions), f"Length mismatch: {len(ids)} ids vs {len(transcriptions)} transcriptions"
    
    for i in range(len(ids)):
        hypothesis_formatted[ids[i]] = transcriptions[i]

    # Save as gzipped JSON
    with gzip.open(transcription_generated_formatted_path, "wt", encoding="utf-8") as f:
        json.dump(hypothesis_formatted, f, ensure_ascii=False, indent=2)

    # Save label path dictionary
    label_path_dict = {
        "path": {subset_name: transcription_generated_formatted_path}, 
        "score": -1, 
        "epoch": -1
    }

    with open(out_label_path_dict_path, "w") as f:
        f.write(json.dumps(label_path_dict))
        f.write("\n")

    # Call the external function (assumed to exist in scope or imported)
    create_ctm_from_json(
        transcription_generated_formatted_path, 
        transcription_generated_ctm_path
    )

class FormatToCtmJob(Job):

    def __init__(
        self, 
        manifest_path: tk.Path,
        text_file: tk.Path,
        subset_name: str = "preprocessed_audio",
    ):
        self.manifest_path = manifest_path
        self.text_file = text_file
        self.subset_name = subset_name
        
        self.transcription_generated_formatted = self.output_path(f"search_results.py.gz")
        self.transcription_generated_ctm = self.output_path(f"search_results.ctm")
        self.out_label_path_dict = self.output_path(f"labels_path.json")    

    def tasks(self):
        yield Task("run", mini_task=True)
    def run(self):
        format_transcriptions(
            manifest_path_input=self.manifest_path.get_path(),
            output_text_file_path=self.text_file.get_path(),
            transcription_generated_formatted_path=self.transcription_generated_formatted.get_path(),
            subset_name=self.subset_name,
            out_label_path_dict_path=self.out_label_path_dict.get_path(),
            transcription_generated_ctm_path=self.transcription_generated_ctm.get_path()
        )

class MergeGenerateShardsJob(Job):

    __sis_hash_exclude__ = {"subset_name", "manifest_path"}

    def __init__(
        self, 
        input_text_files: List[Type[tk.Path]], 
        subset_name: str = "preprocessed_audio",
    ):
        self.input_text_files = input_text_files
        self.subset_name = subset_name
        
        self.output_text_file = self.output_path("transcription.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

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
        
    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__)) 
        
def format_transcription(cfg):
    """
    Format the generated transcription to a more readable format.
    """
    import json
    import gzip

    hypothesis_formatted = {}

    manifest_path = cfg.task_data.get_path() + "/" + cfg.gen_subset + ".tsv"

    ids = []

    with open(manifest_path, "r") as manifest_file:
        lines = manifest_file.readlines()

        base = cfg.gen_subset + '/'

        for line in lines[1:]:  # Skip header
            line = line.split(' ')
            line_id =  line[0]  # e.g. 71550/3703-71550-0004.flac

            slash_index = line_id.rfind('/')
            # find the last occurrence of ".flac"
            flac_index = line_id.rfind('.flac')
            
            # slice between them 3703-71550-0004
            line_id = line_id[slash_index + 1:flac_index]

            ids.append(base + line_id + '/' + line_id)

    transcriptions = []

    with open(cfg.transcription_generated.get_path(), "r") as gen_file:
        for line in gen_file:
            line = line.strip().replace('\n', '').upper()
            transcriptions.append(line)


    assert len(ids) == len(transcriptions), f"Length mismatch: {len(ids)} ids vs {len(transcriptions)} transcriptions"
    for i in range(len(ids)):
        hypothesis_formatted[ids[i]] = transcriptions[i]

    # save as gzipped JSON
    with gzip.open(cfg.transcription_generated_formatted, "wt", encoding="utf-8") as f:
        json.dump(hypothesis_formatted, f, ensure_ascii=False, indent=2)

    label_path_dict = {"path": {cfg.gen_subset: cfg.transcription_generated_formatted.get_path()}, "score": -1, "epoch": -1}

    with open(cfg.out_label_path_dict.get_path(), "w") as f:
        f.write(json.dumps(label_path_dict))
        f.write("\n")


    create_ctm_from_json(
        cfg.transcription_generated_formatted.get_path(), cfg.transcription_generated_ctm.get_path()
    ) 

class FormatTranscriptionJob(Job):

    def __init__(
        self, 
        task_data: tk.Path,
        transcription_generated: tk.Path,
        gen_subset: str = "preprocessed_audio",
    ):
        self.task_data = task_data
        self.transcription_generated = transcription_generated
        self.gen_subset = gen_subset

        self.transcription_generated_formatted = self.output_path(f"search_results.py.gz")
        self.transcription_generated_ctm = self.output_path(f"search_results.ctm")
        self.out_label_path_dict = self.output_path(f"labels_path.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        format_transcription(self)

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