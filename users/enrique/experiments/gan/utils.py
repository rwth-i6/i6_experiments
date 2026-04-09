from copy import deepcopy
from itertools import product
import gzip
import json
import os
import shutil
import subprocess as sp
from typing import Optional
from sisyphus import Job, Task, tk
"""
Defines the external software to be used for the Experiments
"""
from sisyphus import tk

from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_core.lm.kenlm import CompileKenLMJob
from recipe.i6_experiments.users.vieting.experiments.librispeech.librispeech_960_pretraining.wav2vec2.fairseq \
    import SetupFairseqJob

from recipe.i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from recipe.i6_experiments.common.tools.sctk import compile_sctk

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


FAIRSEQ_PATH = SetupFairseqJob(
    fairseq_root = CloneGitRepositoryJob(
        url="git@github.com:facebookresearch/fairseq.git",
        checkout_folder_name="fairseq",
        commit="c7c478b92fe135838a2b9ec8341495c732a92401",
    ).out_repository.copy(),
    python_exe = "/usr/bin/python3",
).out_fairseq_root
FAIRSEQ_PATH.hash_overwrite = "DEFAULT_FAIRSEQ_PATH"

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