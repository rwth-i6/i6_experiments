import gzip
import json
import math
import os
from random import random
import shutil
import subprocess as sp
from typing import Optional, Type, Dict, Any
from sisyphus import Job, Task, tk
import logging
import i6_core.util as util
import numpy as np
import glob
from sisyphus import tools
import gc


from sisyphus import tk
from sisyphus.job_path import Variable

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




def process_audio(*, env, fairseq_root, audio_dir, valid_percent, ext, rvad_root, initial_manifests_dir: Optional[tk.Path] = None, concurrent: int = 8, delete_silence_concurrent: int = None, featurize_audio_concurrent: int = None, layer, model_path, dim:Optional[int] = 512, alias_prefix, alias_delete: str = None, alias_feat : str = None, existing_clusters: Optional[tk.Path] = None, existing_pca: Optional[tk.Path] = None, max_n_audios_per_manifest: Optional[int] = None, name_the_manifests_just_train_and_valid: Optional[bool] = True):
    if alias_delete is None:
        alias_delete = alias_prefix+"/audio/delete_silences"
    if alias_feat is None:
        alias_feat = alias_prefix+"/audio/featurize_audio"

    if delete_silence_concurrent is None:
        delete_silence_concurrent = concurrent
    if featurize_audio_concurrent is None:
        featurize_audio_concurrent = concurrent

    
    delete_silences_job = Wav2VecUDeleteSilencesInAudioJob(
        environment=env,
        fairseq_root=fairseq_root,
        audio_dir=audio_dir,
        valid_percent=valid_percent,
        extension=ext,
        rvad_root=rvad_root,
        initial_manifests_dir=initial_manifests_dir,
        concurrent=concurrent,
        name_the_manifests_just_train_and_valid=name_the_manifests_just_train_and_valid,
        max_n_audios_per_manifest=max_n_audios_per_manifest,
    )
    delete_silences_job.add_alias(os.path.join(alias_prefix, alias_delete))
    featurize_job = Wav2VecUFeaturizeAudioJob(
        environment=env,
        fairseq_root=fairseq_root,
        layer=layer,
        existing_clusters=existing_clusters,
        existing_pca=existing_pca,
        w2v2_model_path=model_path,
        input_audio_manifests=delete_silences_job.out_preprocessed_manifest,
        concurrent=concurrent,
    )
    featurize_job.add_alias(os.path.join(alias_prefix, alias_feat))
    return delete_silences_job, featurize_job, delete_silences_job.out_preprocessed_manifest

class Wav2VecUDeleteSilencesInAudioJob(Job):
    """
    Class to perform audio preprocessing for Wav2Vec-U.
    This includes generating manifests, performing VAD, and removing silence from audio files.
    """

    __sis_hash_exclude__ = {"environment":"/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3", "fairseq_root":"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq",  "rvad_root":None, "name_the_manifests_just_train_and_valid":True}

    __sis_hash_constant__ = {"concurrent": 8}

    def __init__(
        self,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        audio_dir: Type[tk.Path],
        valid_percent: Optional[float] = 0,
        extension: Optional[str] = "flac",
        rvad_root: Optional[tk.Path] = None,
        initial_manifests_dir: Optional[tk.Path] = None, # not supported yet
        concurrent: Optional[int] = 8,
        name_the_manifests_just_train_and_valid: Optional[bool] = True,
        max_n_audios_per_manifest: Optional[int] = None,  # Use only for testing purposes
    ):
        """
        :param environment: Path to the virtual environment.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param audio_dir: Path to the directory containing audio files.
        :param valid_percent: Percentage of validation data to use.
        :param extension: Audio file extension (default: "flac").
        :param rvad_root: Path to the RVAD root directory.
        :param initial_manifests_dir: Directory containing initial manifests 
        :param concurrent: Number of concurrent tasks to run (default: 16).
        :param name_the_manifests_just_train_and_valid: If True, the manifests will be named just train and valid, otherwise they will be named with the split name.
        :param max_n_audios_per_manifest: Maximum number of audio files per manifest (default: None).
        """
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.audio_dir = audio_dir
        self.valid_percent = valid_percent if valid_percent is not None else 0
        self.extension = extension
        self.rvad_root = (
            rvad_root
            if rvad_root
            else tk.Path("/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/rVADfast")
        )
        self.initial_manifests_dir = initial_manifests_dir
        self.valid_manifest_exists = False
        self.concurrent = concurrent
        self.name_the_manifests_just_train_and_valid = name_the_manifests_just_train_and_valid
        self.max_n_audios_per_manifest = max_n_audios_per_manifest


        self.out_preprocessed_audio = self.output_path("preprocessed_audio", directory=True)
        self.out_preprocessed_manifest = self.output_path("preprocessed_manifest", directory=True)

        # Resource requirements
        self.rqmt = {"time": 64, "cpu": concurrent//2,"gpu":1, "mem": 64}

        self.out_manifest_and_vads = self.output_path("manifest_and_vads", directory=True)

        # Parallelization
        # Parallel processing outputs
        self.out_manifest_and_vads_dirs = {
            task_id: self.output_path(f"manifest_and_vads_{task_id}", directory=True)
            for task_id in range(1, self.concurrent + 1)
        }

        self.out_manifest_and_vads_patch = util.MultiOutputPath(
            self, "manifest_and_vads_.$(TASK)", self.out_manifest_and_vads_dirs, cached=True
        )

    def tasks(self):
        yield Task("make_initial_manifest", mini_task=True)

        yield Task("split_manifest", mini_task=True)

        yield Task("run_vad", resume="run_vad", rqmt=self.rqmt, args=range(1, self.concurrent + 1))

        yield Task(
            "remove_silence_from_audio",
            resume="remove_silence_from_audio",
            rqmt=self.rqmt,
            args=range(1, self.concurrent + 1),
        )

        yield Task("make_final_manifest", rqmt=self.rqmt)

    def env_call(self):
        env_call_str = (
            f"export FAIRSEQ_ROOT={self.fairseq_root.get_path()} && "
            f"export RVAD_ROOT={self.rvad_root} && "
            "export PYTHONPATH=$RVAD_ROOT:$PYTHONPATH && "
        )
        if self.environment:
            env_call_str += f"source {self.environment.get_path()}/bin/activate && "
        return env_call_str

    def make_initial_manifest(self):

        # If initial manifests are provided, copy them to the output directory
        # and skip the manifest generation step
        
        if self.initial_manifests_dir:
            # Get the source and destination directories
            src_dir = self.initial_manifests_dir.get_path()
            dest_dir = self.out_manifest_and_vads.get_path()

            # Check if source directory exists
            if not os.path.exists(src_dir):
                raise FileNotFoundError(f"Source directory not found: {src_dir}")

            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)

            # Copy all .tsv files from source to destination
            for file_name in os.listdir(src_dir):
                if file_name.endswith('.tsv'):
                    src_path = os.path.join(src_dir, file_name)
                    dest_path = os.path.join(dest_dir, file_name)
                    shutil.copy2(src_path, dest_path)

        else:
            w2v_manifest(root=self.audio_dir.get_path(),
                         valid_percent=0,
                         dest=self.out_manifest_and_vads.get_path(),
                         ext=self.extension,
                         path_must_contain=None,
                         name_the_manifests_just_train_and_valid=self.name_the_manifests_just_train_and_valid,
                         max_n_audios_per_manifest=self.max_n_audios_per_manifest
                     )


    def split_manifest(self):
        """Split the manifest into chunks for parallel processing"""

        tsv_files = glob.glob(os.path.join(self.out_manifest_and_vads.get_path(), "*.tsv"))

        for tsv_file in tsv_files:
            filename = os.path.basename(tsv_file)
            with open(tsv_file, "r") as f:
                lines = f.readlines()
                header = lines[0] if len(lines) > 0 else ""
                data = lines[1:] if len(lines) > 1 else []

                # Split into chunks
                chunk_size = math.ceil(len(data) / self.concurrent)
                chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
                # Write chunks to temporary directories
                for task_id, chunk in enumerate(chunks, 1):
                    chunk_dir = self.out_manifest_and_vads_dirs[task_id].get_path()
                    os.makedirs(chunk_dir, exist_ok=True)
                    
                    chunk_manifest = os.path.join(chunk_dir, filename)
                    with open(chunk_manifest, "w") as f_out:
                        f_out.write(header)
                        f_out.writelines(chunk)
        
    def run_vad(self, task_id):

        
        manifest_and_vad_dir = self.out_manifest_and_vads_dirs[task_id].get_path()

        vads_script = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/vads.py")

        tsv_files = glob.glob(os.path.join(self.out_manifest_and_vads_dirs[task_id].get_path(), "*.tsv"))

        for tsv_file in tsv_files:
            filename = os.path.basename(tsv_file)
            sh_call = (
                f"python {vads_script} -r $RVAD_ROOT " +
                f" < {manifest_and_vad_dir}/"+filename +
                f" > {manifest_and_vad_dir}/"+filename[:-4]+".vads"
            )
            sp.run(self.env_call() + sh_call, shell=True, check=True)




    def remove_silence_from_audio(self, task_id):

        manifest_and_vad_dir = self.out_manifest_and_vads_dirs[task_id].get_path()

        silence_remove_script = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/remove_silence.py"
        )

        tsv_files = glob.glob(os.path.join(self.out_manifest_and_vads.get_path(), "*.tsv"))

        for tsv_file in tsv_files:
            filename = os.path.basename(tsv_file)
            
            sh_call = (
                f"python {silence_remove_script} "
                f"--tsv {manifest_and_vad_dir}/"+filename+
                f" --vads {manifest_and_vad_dir}/"+filename[:-4]+".vads"+
                f" --out {self.out_preprocessed_audio.get_path()} "
            )

            sp.run(self.env_call() + sh_call, shell=True, check=True)

        
    def make_final_manifest(self):
        w2v_manifest(root=self.out_preprocessed_audio.get_path(),
                        valid_percent=self.valid_percent,
                        dest=self.out_preprocessed_manifest.get_path(),
                        ext=self.extension,
                        path_must_contain=None,
                        name_the_manifests_just_train_and_valid=self.name_the_manifests_just_train_and_valid)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v:
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement

        for k, v in cls.__sis_hash_constant__.items():
            d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))


class Wav2VecUFeaturizeAudioJob(Job):
    """
    Job to featurize audio using Wav2Vec for unsupervised learning (Wav2Vec-U).

    This job prepares audio features by running the `prepare_audio.sh` script.
    """

    __sis_hash_exclude__ = {"environment":"/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3", "fairseq_root":"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"}

    __sis_hash_constant__ = {"concurrent": 8}

    def __init__(
        self,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        w2v2_model_path: tk.Path,
        input_audio_manifests: tk.Path,
        dim: Optional[int] = 512,
        layer: Optional[int] = 14,  # Small w2v2 models don't have 14 layers
        existing_clusters: Optional[tk.Path] = None,
        existing_pca: Optional[tk.Path] = None,
        concurrent: Optional[int] = 4,
    ):
        """
        :param environment: Path to the virtual environment.
        :param fairseq_root: Path to the root directory of Fairseq.
        :param w2v2_model_path: Path to the pre-trained Wav2Vec 2.0 model.
        :param input_audio_manifests: Base directory with the .tsv files for audio manifests.
        :param dim: Feature dimension (default: 512).
        :param layer: Layer for feature extraction (default: 14) In the paper, they have base-1 order, they refer to this layer as layer 15
        :param existing_clusters: Path to existing clusters (if any), should just be a centroids.npy file
        :param existing_pca: Path to existing PCA (if any), should be a directory containing 512_pca_A.npy and 512_pca_b.npy
        """
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.w2v2_model_path = w2v2_model_path
        self.input_audio_manifests = input_audio_manifests
        self.dim = dim
        self.layer = layer
        self.existing_clusters = existing_clusters
        self.existing_pca = existing_pca
        self.concurrent = concurrent

        # Paths for outputs
        self.out_features = self.output_path("audio_features", directory=True)
        self.out_features_precompute_pca512_cls128_mean = self.output_path("audio_features/precompute_pca512_cls128_mean", directory=True)
        self.out_features_precompute_pca512_cls128_mean_pooled = self.output_path("audio_features/precompute_pca512_cls128_mean_pooled", directory=True)
        self.out_features_clusters = self.output_path("audio_features/CLUS128/centroids.npy")
        self.out_features_pca = self.output_path("audio_features/pca", directory=True)

        self.out_chunk_manifest = self.output_path("chunk_manifest", directory=True)
        self.out_chunk_manifest_dirs = {
            task_id: self.output_path(f"chunk_manifest_{task_id}", directory=True)
            for task_id in range(1, self.concurrent + 1)
        }

        self.out_chunk_dirs = {
            task_id: self.output_path(f"features_chunk_{task_id}", directory=True)
            for task_id in range(1, self.concurrent + 1)
        }
        self.out_features_chunked = util.MultiOutputPath(
            self, "features_chunk_$(TASK)", self.out_chunk_dirs, cached=True
        )

        # Resource requirements
        self.rqmt = {"time": 100, "gpu": 1, "mem": 120}

    def tasks(self):
        yield Task("copy_manifest_files", mini_task=True)
        yield Task("split_audio_manifest", mini_task=True)
        yield Task("extract_features", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        yield Task("merge_features", rqmt=self.rqmt)
        if self.existing_clusters is None:
            yield Task("cluster_features", rqmt=self.rqmt)
        else:
            yield Task("cluster_features", mini_task=True)

        yield Task("apply_cluster", rqmt=self.rqmt)
        
        if self.existing_pca is None:
            yield Task("compute_pca", rqmt=self.rqmt)
        else:
            yield Task("compute_pca", mini_task=True)

        yield Task("apply_pca", rqmt=self.rqmt)
        yield Task("merge_clusters", rqmt=self.rqmt)
        yield Task("mean_pool", rqmt=self.rqmt)

    def env_call(self):
        """Prepare the environment activation call string."""
        env_call_str = (
            f"export FAIRSEQ_ROOT={self.fairseq_root.get_path()}  &&  export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH && "
        )
        if self.environment:
            env_call_str += f"source {self.environment.get_path()}/bin/activate && "
        return env_call_str

    def run_python_script(self, script_path, args):
        """
        Helper method to execute Python scripts with the given arguments.
        """
        sp.run(
            self.env_call() + f"python {script_path} " + " ".join(args),
            shell=True,
            check=True,
        )

    def copy_manifest_files(self):
        """
        Copy mandatory `.tsv` files (at least one is required) and optional `.wrd`, `.ltr`, `.phn`, and `dict*` files
        from the source directory to the target directory.
        """
        source_dir = self.input_audio_manifests.get_path()
        target_dir = self.out_features.get_path()

        # Ensure there is at least one '.tsv' file in the source directory
        tsv_files = glob.glob(os.path.join(source_dir, "*.tsv"))
        if not tsv_files:
            logging.warning("No .tsv manifest files found in the source directory. At least one is required.")

        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Copy mandatory .tsv files
        for tsv_file in tsv_files:
            shutil.copy(tsv_file, target_dir)

        # Copy optional files
        for pattern in ["*.wrd", "*.ltr", "*.phn", "dict*"]:
            optional_files = glob.glob(os.path.join(source_dir, pattern))
            for optional_file in optional_files:
                shutil.copy(optional_file, target_dir)

    def split_audio_manifest(self):
        """
        Split the input audio manifest files for all splits (e.g., train, valid, test)
        into smaller chunks for parallel processing.
        """
        # Detect all available `.tsv` manifest files in the input directory
        manifest_files = glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        for manifest_file in manifest_files:
            # Extract the split name from the filename (e.g., "train" from "train.tsv")
            split_name = os.path.splitext(os.path.basename(manifest_file))[0]

            # Read and split the manifest into chunks
            with open(manifest_file, "r") as f:
                lines = f.readlines()
                header = lines[0] if lines else ""
                data = lines[1:] if len(lines) > 1 else []
                chunk_size = math.ceil(len(data) / self.concurrent)
                chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

                # Write each chunk to its respective output directory
                for task_id, chunk in enumerate(chunks, 1):
                    # Create dynamically subdirectories based on the split name
                    chunk_path = os.path.join(self.out_chunk_manifest_dirs[task_id].get_path(), f"{split_name}.tsv")
                    os.makedirs(self.out_chunk_manifest_dirs[task_id].get_path(), exist_ok=True)
                    with open(chunk_path, "w") as chunk_file:
                        chunk_file.write(header)
                        chunk_file.writelines(chunk)

    def extract_features(self, task_id):
        """
        Extract features for the specified chunk of audio data in parallel for all splits.
        """
        # Detect all available manifest splits
        manifest_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_chunk_manifest_dirs[task_id].get_path(), "*.tsv"))
        ]

        for split in manifest_splits:
            chunk_dir = self.out_chunk_dirs[task_id].get_path()
            manifest_path = self.out_chunk_manifest_dirs[task_id].get_path()
            script_path = os.path.join(
                self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py"
            )

            # Check if the features already exist in the output directory
            output_feature_file = os.path.join(chunk_dir, f"{split}.npy")
            if os.path.exists(output_feature_file):
                logging.info(f"Features already exist for split '{split}' chunk {task_id}, skipping extraction.")
                continue  # Skip this split

            self.run_python_script(
                script_path,
                [
                    manifest_path,
                    "--split",
                    split,
                    "--save-dir",
                    chunk_dir,
                    "--checkpoint",
                    self.w2v2_model_path.get_path(),
                    "--layer",
                    str(self.layer),
                ],
            )

    def merge_features(self):
        """
        Merge the extracted features from all chunks into a single output directory.
        This handles all splits (`train`, `valid`, etc.) dynamically and processes
        both `.npy` and `.lengths` files.
        """

        # Detect all available splits dynamically based on '.npy' files in the chunk folders
        available_splits = set()
        for task_id in range(1, self.concurrent + 1):
            chunk_dir = self.out_chunk_dirs[task_id].get_path()
            npy_files = glob.glob(os.path.join(chunk_dir, "*.npy"))
            for npy_file in npy_files:
                split_name = os.path.splitext(os.path.basename(npy_file))[0]
                available_splits.add(split_name)

        from npy_append_array import NpyAppendArray

        # Process each available split
        for split in available_splits:
            output_npy_file = os.path.join(self.out_features.get_path(), f"{split}.npy")
            output_lengths_path = os.path.join(self.out_features.get_path(), f"{split}.lengths")

            

            npaa = NpyAppendArray(output_npy_file)

            # Merge the features from each chunk for the current split
            for task_id in range(1, self.concurrent + 1):
                chunk_dir = self.out_chunk_dirs[task_id].get_path()
                feature_file = os.path.join(chunk_dir, f"{split}.npy")

                if os.path.exists(feature_file):
                    data = np.load(feature_file, mmap_mode="r")
                    npaa.append(data)
                    del data
                    gc.collect()
                else:
                    logging.warning(f"Feature file not found: {feature_file}, skipping for chunk {task_id}.")

            # Merge length files for the current split
            with open(output_lengths_path, "w") as fout:
                for task_id in range(1, self.concurrent + 1):
                    chunk_dir = self.out_chunk_dirs[task_id].get_path()
                    length_file = os.path.join(chunk_dir, f"{split}.lengths")

                    if os.path.exists(length_file):
                        with open(length_file, "r") as fin:
                            fout.writelines(fin.readlines())
                    else:
                        logging.warning(f"Length file not found: {length_file}, skipping for chunk {task_id}.")

    def cluster_features(self):
        """
        Cluster features for each non-validation manifest.

        This function now finds all `.tsv` files in the features directory,
        filters out any containing "valid" in their name, and then runs
        clustering for each remaining manifest.
        """
        if self.existing_clusters != None:
            src_file = self.existing_clusters.get_path()
            dst_file = self.out_features_clusters.get_path()
            dst_dir = os.path.dirname(dst_file)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            return


        features_dir = self.out_features.get_path()
        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/wav2vec_cluster_faiss.py"
        )

        # Find all .tsv files that do not contain 'valid' in their name.
        # We sort the results to ensure deterministic behavior.
        manifest_files = sorted([
            f for f in glob.glob(os.path.join(features_dir, "*.tsv"))
            if "valid" not in os.path.basename(f)
        ])

        if not manifest_files:
            logging.warning("No non-validation .tsv manifest files found for clustering.")
            return

        # Note: The clustering script saves to a fixed filename ('CLUS128.pt').
        # If multiple non-validation manifests are processed, the output from the last one
        # will overwrite the previous ones.
        for manifest_file in manifest_files:
            logging.info(f"Running feature clustering for manifest: {os.path.basename(manifest_file)}")
            self.run_python_script(
                script_path,
                [
                    manifest_file,
                    "--checkpoint",
                    self.w2v2_model_path.get_path(),
                    "--save-dir",
                    self.out_features.get_path(),
                    "-f",
                    "CLUS128",
                    "--sample-pct",
                    "0.01",
                    "--layer",
                    str(self.layer),
                ],
            )




    def apply_cluster(self):
        """
        Apply clustering to feature splits using the trained cluster model.
        """
        all_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        ]

        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py"
        )

        for split in all_splits:
            self.run_python_script(
                script_path,
                [
                    self.out_features.get_path(),
                    "--checkpoint",
                    self.w2v2_model_path.get_path(),
                    "--path",
                    os.path.join(self.out_features.get_path(), "CLUS128"),
                    "--split",
                    split,
                    "--layer",
                    str(self.layer),
                ],
            )

    # def compute_pca(self):
    #     """
    #     Task to compute PCA on the processed features.
    #     """
    #     train_split = "train"
    #     train_split_npy = os.path.join(self.out_features.get_path(), f"{train_split}.npy")
    #     output_pca_dir = os.path.join(self.out_features.get_path(), "pca")
    #     script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/pca.py")

    #     self.run_python_script(
    #         script_path,
    #         [
    #             train_split_npy,
    #             "--output",
    #             output_pca_dir,
    #             "--dim",
    #             str(self.dim),
    #         ],
    #     )

    
    def compute_pca(self):
        """
        Task to compute PCA on the features of each non-validation split.

        This function identifies all non-validation splits by looking for `.tsv`
        files without "valid" in their names. It then computes PCA for the
        corresponding `.npy` feature file of each split.
        """

        if self.existing_pca != None:
            for file_name in os.listdir(self.existing_pca.get_path()):
                src_file = os.path.join(self.existing_pca.get_path(), file_name)
                dst_file = os.path.join(self.out_features_pca.get_path(), file_name)
                os.makedirs(self.out_features_pca.get_path(), exist_ok=True)
                shutil.copy2(src_file, dst_file)
            return
            
        
        features_dir = self.out_features.get_path()
        script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/pca.py")

        # Find all .tsv files that do not contain 'valid' in their name.
        # We sort the results to ensure deterministic behavior.
        manifest_files = sorted([
            f for f in glob.glob(os.path.join(features_dir, "*.tsv"))
            if "valid" not in os.path.basename(f)
        ])

        if not manifest_files:
            logging.warning("No non-validation manifest files found for PCA computation.")
            return

        # Note: The PCA script saves to a fixed filename ('pca.pt'). If multiple
        # non-validation splits are processed, the output from the last one will
        # overwrite the previous ones.
        for manifest_file in manifest_files:
            split_name = os.path.splitext(os.path.basename(manifest_file))[0]
            feature_file_npy = os.path.join(features_dir, f"{split_name}.npy")

            if not os.path.exists(feature_file_npy):
                logging.warning(
                    f"Feature file {feature_file_npy} not found, "
                    f"skipping PCA computation for split '{split_name}'."
                )
                continue

            logging.info(f"Computing PCA for split: {split_name}")
            output_pca_dir = os.path.join(self.out_features.get_path(), "pca")

            self.run_python_script(
                script_path,
                [
                    feature_file_npy,
                    "--output",
                    output_pca_dir,
                    "--dim",
                    str(self.dim),
                ],
            )

    def apply_pca(self):
        """
        Task to apply PCA transformation to all feature splits.
        """
        all_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        ]
        output_precompute_pca_dir = os.path.join(self.out_features.get_path(), f"precompute_pca{self.dim}")
        pca_path = os.path.join(self.out_features.get_path(), f"pca/{self.dim}_pca")
        script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/apply_pca.py")

        for split in all_splits:
            self.run_python_script(
                script_path,
                [
                    self.out_features.get_path(),
                    "--split",
                    split,
                    "--save-dir",
                    output_precompute_pca_dir,
                    "--pca-path",
                    pca_path,
                    "--batch-size",
                    "1048000",
                ],
            )

    def merge_clusters(self):
        """
        Task to merge precomputed PCA features with clusters for all splits.
        """
        all_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        ]
        precompute_pca_dir = os.path.join(self.out_features.get_path(), f"precompute_pca{self.dim}")
        cluster_dir = os.path.join(self.out_features.get_path(), "CLUS128")
        output_merge_dir = os.path.join(self.out_features.get_path(), f"precompute_pca{self.dim}_cls128_mean")
        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/merge_clusters.py"
        )

        for split in all_splits:
            self.run_python_script(
                script_path,
                [
                    precompute_pca_dir,
                    "--cluster-dir",
                    cluster_dir,
                    "--split",
                    split,
                    "--save-dir",
                    output_merge_dir,
                    "--pooling",
                    "mean",
                ],
            )

    def mean_pool(self):
        """
        Task to perform mean pooling on the merged PCA and cluster features.
        """
        all_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        ]
        input_mean_dir = os.path.join(self.out_features.get_path(), f"precompute_pca{self.dim}_cls128_mean")
        output_mean_pooled_dir = os.path.join(
            self.out_features.get_path(), f"precompute_pca{self.dim}_cls128_mean_pooled"
        )
        script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/mean_pool.py")

        for split in all_splits:
            self.run_python_script(
                script_path,
                [
                    input_mean_dir,
                    "--save-dir",
                    output_mean_pooled_dir,
                    "--split",
                    split,
                ],
            )

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v:
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement

        for k, v in cls.__sis_hash_constant__.items():
            d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__)) 


  
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

class WordToLetterJob(Job):
    __sis_hash_exclude__ = {"environment":"/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3", "fairseq_root":"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"}

    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        words_file: tk.Path,
    ):
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.words_file = words_file
        self.letters_file = self.output_path("letters.txt")

    def tasks(self):
        yield Task("convert_words_to_letters", rqmt={"time": 1000, "cpu": 1, "mem": 16})

    def convert_words_to_letters(self):
        word_lines = []
        with open(self.words_file.get_path(), "r") as wf:
            word_lines = wf.readlines()

        assert len(word_lines) > 0, "Words file is empty."
        letter_lines = []
        for line in word_lines:
            word = line.strip()
            letters = " ".join(list(word))
            letter_lines.append(f"{letters}\n")

        with open(self.letters_file.get_path(), "w") as lf:
            lf.writelines(letter_lines)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v:
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement


        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))


class FairseqG2PWordToPhnJob(Job):
    """
    Job to convert words to phonemes using a G2P model.
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py --compact < $target_dir/words.txt > $target_dir/phones.txt
    """

    __sis_hash_exclude__ = {"environment":"/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3", "fairseq_root":"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"}

    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        words_file: tk.Path,
    ):
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.words_file = words_file
        self.phn_file = self.output_path("phones.txt")

    def tasks(self):
        yield Task("convert_words_to_phonemes", rqmt={"time": 1000, "cpu": 1, "mem": 16})

    def env_call(self):
        """Prepare the fairseq_python_env activation call string."""
        env_call_str = (
            f"export FAIRSEQ_ROOT={self.fairseq_root.get_path()}  &&  export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH && "
        )
        if self.fairseq_python_env:
            env_call_str += f"source {self.fairseq_python_env.get_path()}/bin/activate && "
        return env_call_str
    
    def convert_words_to_phonemes(self):
        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py"
        )

        sh_call = (
            f"python {script_path} --compact < {self.words_file.get_path()} > {self.phn_file.get_path()}"
        )

        sp.run(self.env_call() + sh_call, shell=True, check=True)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v:
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))
        
class CreateLexiconAndDictionaryJob(Job):
    __sis_hash_exclude__ = {"environment":"/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3", "fairseq_root":"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"}
    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        words_txt: Type[tk.Path],
        tokenized_words: Type[tk.Path],
        label_type: str = "",
        sil_token: str = "<SIL>",
    ):
        self.fairseq_root = fairseq_root
        self.fairseq_python_env = fairseq_python_env
        self.words_txt = words_txt
        self.tokenized_words = tokenized_words
        self.label_type = label_type
        self.silence_token = sil_token

        # Outputs
        self.lexicon_lst = self.output_path("lexicon.lst")
        self.lexicon_filtered_lst = self.output_path("lexicon_filtered.lst")
        
        # Intermediate folder for dict generation
        self.dict_folder = self.output_path("dict_folder", directory=True)
        self.dict_txt = self.output_path("dict_folder/dict.txt")
        self.dict_labeltype_txt = self.output_path(f"dict_folder/dict.{label_type}.txt")

        self.rqmt = {"time": 3000, "cpu": 1, "mem": 16}

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
        # 1. Create raw lexicon
        sh_call = f"paste {self.words_txt.get_path()} {self.tokenized_words.get_path()} > {self.lexicon_lst.get_path()}"
        self.sh_call_with_environment(sh_call)

        # 2. Generate dictionary from phones
        # Note: preprocess.py writes dict.txt to the destdir
        sh_call = f"python {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref {self.tokenized_words.get_path()} --only-source --destdir {self.dict_folder.get_path()} --padding-factor 1 --dict-only"
        self.sh_call_with_environment(sh_call)

        # 3. Filter lexicon based on the generated dictionary
        sh_call = f"python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d {self.dict_txt.get_path()} < {self.lexicon_lst.get_path()} > {self.lexicon_filtered_lst.get_path()}"
        self.sh_call_with_environment(sh_call)
        

        # cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
        # echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt
        # Create final dictionary with silence token
        sh_call = f"cp {self.dict_txt.get_path()} {self.dict_labeltype_txt.get_path()}"
        self.sh_call_with_environment(sh_call)

        sh_call = f"echo \"{self.silence_token} 0\" >> {self.dict_labeltype_txt.get_path()}"
        self.sh_call_with_environment(sh_call)

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

class TokenizeWithLexiconAndSilenceJob(Job):
    """
    Job to phonemize text using a lexicon and optionally insert silence tokens.
    """

    __sis_hash_exclude__ = {"environment":"/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3", "fairseq_root":"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"}

    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        text_file_path: Type[tk.Path],
        lexicon_lst: Type[tk.Path],
        insert_silence_between_words_prob: float = 0.25,
        silence_token: str = "<SIL>",  # only <SIL> is supported currently
        surround_with_silence: bool = True,
    ):
        assert silence_token == "<SIL>", "Currently, only '<SIL>' is supported as silence token."
        self.fairseq_root = fairseq_root
        self.fairseq_python_env = fairseq_python_env
        self.text_file_path = text_file_path
        self.lexicon_lst = lexicon_lst
        self.insert_silence_between_words_prob = insert_silence_between_words_prob
        self.silence_token = silence_token
        self.surround_with_silence = surround_with_silence

        # Outputs
        self.tokenized_text_with_silence = self.output_path("tokenized_text_with_silence.txt")

        self.rqmt = {"time": 3000, "cpu": 1, "mem": 16}
    
    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        wrd_to_phn = {}
        lexicon_path = self.lexicon_lst.get_path()
        lines = open(self.text_file_path.get_path(), "r").readlines()
        # Load lexicon
        with open(lexicon_path, "r") as lf:
            for line in lf:
                items = line.rstrip().split()
                assert len(items) > 1, f"Invalid lexicon line: {line}"
                assert items[0] not in wrd_to_phn, f"Duplicate word: {items[0]}"
                wrd_to_phn[items[0]] = items[1:]

        outputs = []

        # Process input lines
        for line in lines:
            words = line.strip().split()

            # Skip lines with unknown words
            if not all(w in wrd_to_phn for w in words):
                continue

            phones = []
            if self.surround_with_silence:
                phones.append(self.silence_token)

            sample_sil_probs = None
            if self.insert_silence_between_words_prob > 0 and len(words) > 1:
                sample_sil_probs = np.random.random(len(words) - 1)

            for i, w in enumerate(words):
                phones.extend(wrd_to_phn[w])
                if (
                    sample_sil_probs is not None
                    and i < len(sample_sil_probs)
                    and sample_sil_probs[i] < self.insert_silence_between_words_prob
                ):
                    phones.append(self.silence_token)

            if self.surround_with_silence:
                phones.append(self.silence_token)

            outputs.append(" ".join(phones))

        with open(self.tokenized_text_with_silence.get_path(), "w") as f:
            for line in outputs:
                f.write(line + "\n")

class FairseqNormalizeTextAndCreateDictionary(Job):
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

class FairseqPreprocessJob(Job):
    """
    Job to preprocess data for Fairseq training.
    sh_call = f"python {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref {self.labeled_text.get_path()} --workers 1 --only-source --destdir {self.labels_folder.get_path()} --srcdict {self.dict_labeltype_txt.get_path()}"
        self.sh_call_with_environment(sh_call)
    """
    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        tokenized_text: Type[tk.Path],
        dict_labeltype_txt: Type[tk.Path],
        dataset_impl: str = "mmap",
    ):
        
        self.fairseq_root = fairseq_root
        self.tokenized_text = tokenized_text
        self.dict_labeltype_txt = dict_labeltype_txt
        self.fairseq_python_env = fairseq_python_env
        self.dataset_impl = dataset_impl

        self.labels_folder = self.output_path("labels_folder", directory=True)
        
        self.rqmt = {"time": 4242, "cpu": 1, "mem": 32}
    
    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
    
    def sh_call_with_environment(self, sh_call):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.fairseq_root.get_path()}:" + env.get("PYTHONPATH", "")
        env["FAIRSEQ_ROOT"] = self.fairseq_root.get_path()
        env["PATH"] = f"{self.fairseq_python_env.get_path()}/bin:" + env["PATH"]

        sp.run(sh_call, env=env, check=True, shell=True)

    def run(self):
        sh_call = f"python {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py --dataset-impl {self.dataset_impl} --trainpref {self.tokenized_text.get_path()} --workers 1 --only-source --destdir {self.labels_folder.get_path()} --srcdict {self.dict_labeltype_txt.get_path()}"
        self.sh_call_with_environment(sh_call)


def resolve_tk_paths(obj):
    # If it's a tk.Path object → replace it
    if isinstance(obj, tk.Path):
        return obj.get_path()
    
    # If dictionary → process values
    if isinstance(obj, dict):
        return {k: resolve_tk_paths(v, tk.Path) for k, v in obj.items()}
    
    # If list → process elements
    if isinstance(obj, list):
        return [resolve_tk_paths(x, tk.Path) for x in obj]
    
    # If tuple → return a new tuple
    if isinstance(obj, tuple):
        return tuple(resolve_tk_paths(x, tk.Path) for x in obj)
    
    # If set → return a new set
    if isinstance(obj, set):
        return {resolve_tk_paths(x, tk.Path) for x in obj}
    
    # Otherwise just return unchanged
    return obj


class FairseqHydraTrainJob(Job):
    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        fairseq_hydra_config: dict,
    ):
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.fairseq_hydra_config = fairseq_hydra_config

        self.out_dir = self.output_path("out_dir", directory=True)
        self.out_best_model = self.output_path("out_dir/checkpoint_best.pt")

        # Resource requirements
        self.rqmt = {"time": 1000, "gpu": 1, "mem": 16}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def train(self):

        final_fairseq_hydra_config = deepcopy(self.fairseq_hydra_config)

        # Cache the paths for the task data, text data, and KenLM binary
        final_fairseq_hydra_config['task']['data'] = cache_path(self.fairseq_hydra_config['task']['data'].get_path())
        final_fairseq_hydra_config['task']['text_data'] = cache_path(self.fairseq_hydra_config['task']['text_data'].get_path())
        final_fairseq_hydra_config['task']['kenlm_path'] = cache_path(self.fairseq_hydra_config['task']['kenlm_path'].get_path())
        final_fairseq_hydra_config['checkpoint']['save_dir'] = cache_path(self.out_dir.get_path())
        final_fairseq_hydra_config['common']['user_dir'] = cache_path(os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised"))

        final_fairseq_hydra_config = resolve_tk_paths(final_fairseq_hydra_config)
        final_fairseq_hydra_config_str = flatten_dictionary_to_text(final_fairseq_hydra_config)
        
        

        sh_call_str = f"""
            source {self.fairseq_python_env.get_path()}/bin/activate && \
            export HYDRA_FULL_ERROR=1 && \
            PYTHONPATH=$PYTHONPATH:{self.fairseq_root.get_path()} \
            fairseq-hydra-train \
            {final_fairseq_hydra_config_str}
        """
        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__)) 
        


from copy import deepcopy
from itertools import product

def is_iterable_branch(x):
    """Return True if x is an iterable that should expand."""
    if isinstance(x, (str, bytes, dict)):
        return False
    return isinstance(x, (list, tuple, set, range))

def extract_branches(d, prefix=()):
    """
    Traverse nested dictionaries and extract all branching points.
    Returns a list of:
       (key_path, iterable_values)
    """
    branches = []

    for k, v in d.items():
        path = prefix + (k,)
        if isinstance(v, dict):
            branches.extend(extract_branches(v, path))
        elif is_iterable_branch(v):
            branches.append((path, list(v)))

    return branches

def set_in_path(d, path, value):
    """Set a value inside nested dictionaries using a tuple of keys."""
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value

def expand_dictionary(original_dict):
    """
    Expand a nested dictionary into a list of dictionaries,
    one per combination of iterable branching values.
    """
    branches = extract_branches(original_dict)
    if not branches:
        return [deepcopy(original_dict)]

    paths, values_lists = zip(*branches)
    configurations = []

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
        if isinstance(v, (list, tuple)):
            v = ",".join(map(str, v))

        lines.append(f"{k}={v} \\")

    return "\n".join(lines)



class TrainKenLMJob(Job):

    def __init__(
        self,
        kenlm_root: Type[tk.Path],
        text_file_path: Type[tk.Path],
        lm_order: int,
        discount_fallback: bool=True,
        pruning: Optional[list[int]] = None,
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
        sh_call = f"{self.kenlm_root.get_path()}/lmplz -o {self.lm_order} {('--prune ' + ' '.join(map(str, self.pruning))) if not self.pruning is None else ' '} < {self.text_file_path.get_path()} --discount_fallback > {self.output_arpa.get_path()}"
        sp.run(sh_call, check=True, shell=True)

        if self.build_binary:
            sh_call = f"{self.kenlm_root.get_path()}/build_binary {self.output_arpa.get_path()} {self.output_bin.get_path()}"
            sp.run(sh_call, check=True, shell=True)

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


class FairseqKaldiDecodingJob(Job):
    """
    Run the w2vu_generate.py script with specified configurations for Wav2Vec generation.
    """

    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        task_data: Type[tk.Path],
        lexicon_lst_path: Type[tk.Path],
        hlg_graph_path: Type[tk.Path],
        kaldi_dict: Type[tk.Path],
        checkpoints_path: Type[tk.Path],
        gen_subset: Optional[str] = None,
        config_dir: Optional[str] = None,
        config_name: Optional[str] = None,
        dict_phn_txt_path: Optional[tk.Path] = None,
        extra_config: Optional[str] = None,
    ):
        """
        :param decoding_audio_name: Name of the decoding audio
        :param fairseq_python_env: Path to the Python virtual fairseq_python_env.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param task_data: Path to the directory with features.
        :param checkpoints_path: Path to the GAN checkpoints.
        :param config_dir: Path to the configuration directory (default: "config/generate").
        :param config_name: Name of the Hydra configuration (default: "viterbi").
        :param gen_subset: Subset to generate (default: "valid").
        """
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.task_data = task_data
        self.lexicon_lst_path = lexicon_lst_path
        self.hlg_graph_path = hlg_graph_path
        self.kaldi_dict = kaldi_dict
        self.checkpoints_path = checkpoints_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.gen_subset = gen_subset 
        self.extra_config = extra_config if extra_config is not None else ""

        self.results_path = self.output_path("transcriptions", directory=True)
        
        self.shard_id  = None
        if "shard_id=" in self.extra_config:
            shard_id_index = self.extra_config.index("shard_id=") + len("shard_id=")
            shard_id_end_index = self.extra_config.find(" ", shard_id_index)
            if shard_id_end_index == -1:
                shard_id_end_index = len(self.extra_config)
            else:
                self.shard_id = self.extra_config[shard_id_index:shard_id_end_index]+"_"

        self.num_shards = None
        if "num_shards=" in self.extra_config:
            num_shards_index = self.extra_config.index("num_shards=") + len("num_shards=")
            num_shards_end_index = self.extra_config.find(" ", num_shards_index)
            if num_shards_end_index == -1:
                num_shards_end_index = len(self.extra_config)
            self.num_shards = self.extra_config[num_shards_index:num_shards_end_index]

        self.manifest_path = tk.Path(self.task_data.get_path() + "/" + self.gen_subset + ".tsv")

        self.transcription_generated = self.output_path(f"transcriptions/{self.gen_subset}{self.shard_id if self.shard_id is not None else ''}.txt")
        self.transcription_generated_formatted = self.output_path(f"search_results.py.gz")
        self.transcription_generated_ctm = self.output_path(f"search_results.ctm")

        self.out_label_path_dict = self.output_path(f"labels_path.json")

        self.rqmt = {"time": 1000, "gpu": 1, "mem": 32}         
        

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
        yield Task("format_transcription", mini_task=True)

    def run(self):
        if isinstance(self.checkpoints_path, Variable):
            self.checkpoints_path = self.checkpoints_path.get()
        else:
            assert isinstance(self.checkpoints_path, tk.Path), "checkpoints_path should be of type tk.Path or Variable"
            self.checkpoints_path = self.checkpoints_path.get_path() # asume its a tk.Path

        logging.info(f"Using checkpoint: {self.checkpoints_path}")
        
        self.extra_config += f""" \
            lexicon={self.lexicon_lst_path.get_path()} \
            kaldi_decoder_config.hlg_graph_path={self.hlg_graph_path.get_path()} \
            kaldi_decoder_config.output_dict={self.kaldi_dict.get_path()} \
            viterbi_transcript="" \
            """

        sh_call_str = ""
        if self.fairseq_python_env is not None:
            sh_call_str += f"export PYTHONNOUSERSITE=1 && source {self.fairseq_python_env.get_path()}/bin/activate && "

        sh_call_str = (
            sh_call_str
            + f""" \
            export HYDRA_FULL_ERROR=1 && \
            /opt/conda/bin/python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/w2vu_generate.py \
            --config-dir {self.config_dir} \
            --config-name {self.config_name} \
            fairseq.common_eval.path={self.checkpoints_path} \
            fairseq.task.data={self.task_data.get_path()} \
            fairseq.dataset.gen_subset={self.gen_subset} \
            fairseq.common.user_dir={self.fairseq_root.get_path()}/examples/wav2vec/unsupervised \
            results_path={self.results_path.get_path()} \
            """
            + self.extra_config
        )

        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)

    def format_transcription(self):
        """
        Format the generated transcription to a more readable format.
        """
        import json
        import gzip

        hypothesis_formatted = {}

        self.manifest_path = self.task_data.get_path() + "/" + self.gen_subset + ".tsv"

        ids = []

        shard_id_int = int(self.shard_id[:-1]) if self.shard_id is not None else 0
        num_shards_int = int(self.num_shards) if self.num_shards is not None else 1

        with open(self.manifest_path, "r") as manifest_file:
            lines = manifest_file.readlines()
    
            base = self.decoding_audio_name + '/'

            lines = lines[1:]  # Skip header

            if num_shards_int > 1:
                lines = lines[shard_id_int::num_shards_int]

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

        with open(self.transcription_generated.get_path(), "r") as gen_file:
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










############################################################################################################
#audio setup
############################################################################################################

def process_audio(*, env, fairseq_root, audio_dir, valid_percent, ext, rvad_root, initial_manifests_dir: Optional[tk.Path] = None, concurrent: int = 8, delete_silence_concurrent: int = None, featurize_audio_concurrent: int = None, layer, model_path, dim:Optional[int] = 512, alias_prefix, alias_delete: str = None, alias_feat : str = None, existing_clusters: Optional[tk.Path] = None, existing_pca: Optional[tk.Path] = None, max_n_audios_per_manifest: Optional[int] = None, name_the_manifests_just_train_and_valid: Optional[bool] = True):
    if alias_delete is None:
        alias_delete = alias_prefix+"/audio/delete_silences"
    if alias_feat is None:
        alias_feat = alias_prefix+"/audio/featurize_audio"

    if delete_silence_concurrent is None:
        delete_silence_concurrent = concurrent
    if featurize_audio_concurrent is None:
        featurize_audio_concurrent = concurrent

    
    delete_silences_job = Wav2VecUDeleteSilencesInAudioJob(
        environment=env,
        fairseq_root=fairseq_root,
        audio_dir=audio_dir,
        valid_percent=valid_percent,
        extension=ext,
        rvad_root=rvad_root,
        initial_manifests_dir=initial_manifests_dir,
        concurrent=concurrent,
        name_the_manifests_just_train_and_valid=name_the_manifests_just_train_and_valid,
        max_n_audios_per_manifest=max_n_audios_per_manifest,
    )
    delete_silences_job.add_alias(os.path.join(alias_prefix, alias_delete))
    featurize_job = Wav2VecUFeaturizeAudioJob(
        environment=env,
        fairseq_root=fairseq_root,
        layer=layer,
        existing_clusters=existing_clusters,
        existing_pca=existing_pca,
        w2v2_model_path=model_path,
        input_audio_manifests=delete_silences_job.out_preprocessed_manifest,
        concurrent=concurrent,
    )
    featurize_job.add_alias(os.path.join(alias_prefix, alias_feat))
    return delete_silences_job, featurize_job, delete_silences_job.out_preprocessed_manifest

class Wav2VecUDeleteSilencesInAudioJob(Job):
    """
    Class to perform audio preprocessing for Wav2Vec-U.
    This includes generating manifests, performing VAD, and removing silence from audio files.
    """

    __sis_hash_exclude__ = {"environment":"/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3", "fairseq_root":"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq",  "rvad_root":None, "name_the_manifests_just_train_and_valid":True}

    __sis_hash_constant__ = {"concurrent": 8}

    def __init__(
        self,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        audio_dir: Type[tk.Path],
        valid_percent: Optional[float] = 0,
        extension: Optional[str] = "flac",
        rvad_root: Optional[tk.Path] = None,
        initial_manifests_dir: Optional[tk.Path] = None, # not supported yet
        concurrent: Optional[int] = 8,
        name_the_manifests_just_train_and_valid: Optional[bool] = True,
        max_n_audios_per_manifest: Optional[int] = None,  # Use only for testing purposes
    ):
        """
        :param environment: Path to the virtual environment.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param audio_dir: Path to the directory containing audio files.
        :param valid_percent: Percentage of validation data to use.
        :param extension: Audio file extension (default: "flac").
        :param rvad_root: Path to the RVAD root directory.
        :param initial_manifests_dir: Directory containing initial manifests 
        :param concurrent: Number of concurrent tasks to run (default: 16).
        :param name_the_manifests_just_train_and_valid: If True, the manifests will be named just train and valid, otherwise they will be named with the split name.
        :param max_n_audios_per_manifest: Maximum number of audio files per manifest (default: None).
        """
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.audio_dir = audio_dir
        self.valid_percent = valid_percent if valid_percent is not None else 0
        self.extension = extension
        self.rvad_root = (
            rvad_root
            if rvad_root
            else tk.Path("/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/rVADfast")
        )
        self.initial_manifests_dir = initial_manifests_dir
        self.valid_manifest_exists = False
        self.concurrent = concurrent
        self.name_the_manifests_just_train_and_valid = name_the_manifests_just_train_and_valid
        self.max_n_audios_per_manifest = max_n_audios_per_manifest


        self.out_preprocessed_audio = self.output_path("preprocessed_audio", directory=True)
        self.out_preprocessed_manifest = self.output_path("preprocessed_manifest", directory=True)

        # Resource requirements
        self.rqmt = {"time": 64, "cpu": concurrent//2,"gpu":1, "mem": 64}

        self.out_manifest_and_vads = self.output_path("manifest_and_vads", directory=True)

        # Parallelization
        # Parallel processing outputs
        self.out_manifest_and_vads_dirs = {
            task_id: self.output_path(f"manifest_and_vads_{task_id}", directory=True)
            for task_id in range(1, self.concurrent + 1)
        }

        self.out_manifest_and_vads_patch = util.MultiOutputPath(
            self, "manifest_and_vads_.$(TASK)", self.out_manifest_and_vads_dirs, cached=True
        )

    def tasks(self):
        yield Task("make_initial_manifest", mini_task=True)

        yield Task("split_manifest", mini_task=True)

        yield Task("run_vad", resume="run_vad", rqmt=self.rqmt, args=range(1, self.concurrent + 1))

        yield Task(
            "remove_silence_from_audio",
            resume="remove_silence_from_audio",
            rqmt=self.rqmt,
            args=range(1, self.concurrent + 1),
        )

        yield Task("make_final_manifest", rqmt=self.rqmt)

    def env_call(self):
        env_call_str = (
            f"export FAIRSEQ_ROOT={self.fairseq_root.get_path()} && "
            f"export RVAD_ROOT={self.rvad_root} && "
            "export PYTHONPATH=$RVAD_ROOT:$PYTHONPATH && "
        )
        if self.environment:
            env_call_str += f"source {self.environment.get_path()}/bin/activate && "
        return env_call_str

    def make_initial_manifest(self):

        # If initial manifests are provided, copy them to the output directory
        # and skip the manifest generation step
        
        if self.initial_manifests_dir:
            # Get the source and destination directories
            src_dir = self.initial_manifests_dir.get_path()
            dest_dir = self.out_manifest_and_vads.get_path()

            # Check if source directory exists
            if not os.path.exists(src_dir):
                raise FileNotFoundError(f"Source directory not found: {src_dir}")

            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)

            # Copy all .tsv files from source to destination
            for file_name in os.listdir(src_dir):
                if file_name.endswith('.tsv'):
                    src_path = os.path.join(src_dir, file_name)
                    dest_path = os.path.join(dest_dir, file_name)
                    shutil.copy2(src_path, dest_path)

        else:
            w2v_manifest(root=self.audio_dir.get_path(),
                         valid_percent=0,
                         dest=self.out_manifest_and_vads.get_path(),
                         ext=self.extension,
                         path_must_contain=None,
                         name_the_manifests_just_train_and_valid=self.name_the_manifests_just_train_and_valid,
                         max_n_audios_per_manifest=self.max_n_audios_per_manifest
                     )


    def split_manifest(self):
        """Split the manifest into chunks for parallel processing"""

        tsv_files = glob.glob(os.path.join(self.out_manifest_and_vads.get_path(), "*.tsv"))

        for tsv_file in tsv_files:
            filename = os.path.basename(tsv_file)
            with open(tsv_file, "r") as f:
                lines = f.readlines()
                header = lines[0] if len(lines) > 0 else ""
                data = lines[1:] if len(lines) > 1 else []

                # Split into chunks
                chunk_size = math.ceil(len(data) / self.concurrent)
                chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
                # Write chunks to temporary directories
                for task_id, chunk in enumerate(chunks, 1):
                    chunk_dir = self.out_manifest_and_vads_dirs[task_id].get_path()
                    os.makedirs(chunk_dir, exist_ok=True)
                    
                    chunk_manifest = os.path.join(chunk_dir, filename)
                    with open(chunk_manifest, "w") as f_out:
                        f_out.write(header)
                        f_out.writelines(chunk)
        
    def run_vad(self, task_id):

        
        manifest_and_vad_dir = self.out_manifest_and_vads_dirs[task_id].get_path()

        vads_script = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/vads.py")

        tsv_files = glob.glob(os.path.join(self.out_manifest_and_vads_dirs[task_id].get_path(), "*.tsv"))

        for tsv_file in tsv_files:
            filename = os.path.basename(tsv_file)
            sh_call = (
                f"python {vads_script} -r $RVAD_ROOT " +
                f" < {manifest_and_vad_dir}/"+filename +
                f" > {manifest_and_vad_dir}/"+filename[:-4]+".vads"
            )
            sp.run(self.env_call() + sh_call, shell=True, check=True)




    def remove_silence_from_audio(self, task_id):

        manifest_and_vad_dir = self.out_manifest_and_vads_dirs[task_id].get_path()

        silence_remove_script = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/remove_silence.py"
        )

        tsv_files = glob.glob(os.path.join(self.out_manifest_and_vads.get_path(), "*.tsv"))

        for tsv_file in tsv_files:
            filename = os.path.basename(tsv_file)
            
            sh_call = (
                f"python {silence_remove_script} "
                f"--tsv {manifest_and_vad_dir}/"+filename+
                f" --vads {manifest_and_vad_dir}/"+filename[:-4]+".vads"+
                f" --out {self.out_preprocessed_audio.get_path()} "
            )

            sp.run(self.env_call() + sh_call, shell=True, check=True)

        
    def make_final_manifest(self):
        w2v_manifest(root=self.out_preprocessed_audio.get_path(),
                        valid_percent=self.valid_percent,
                        dest=self.out_preprocessed_manifest.get_path(),
                        ext=self.extension,
                        path_must_contain=None,
                        name_the_manifests_just_train_and_valid=self.name_the_manifests_just_train_and_valid)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v:
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement

        for k, v in cls.__sis_hash_constant__.items():
            d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))


class Wav2VecUFeaturizeAudioJob(Job):
    """
    Job to featurize audio using Wav2Vec for unsupervised learning (Wav2Vec-U).

    This job prepares audio features by running the `prepare_audio.sh` script.
    """

    __sis_hash_exclude__ = {"environment":"/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3", "fairseq_root":"/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"}

    __sis_hash_constant__ = {"concurrent": 8}

    def __init__(
        self,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        w2v2_model_path: tk.Path,
        input_audio_manifests: tk.Path,
        dim: Optional[int] = 512,
        layer: Optional[int] = 14,  # Small w2v2 models don't have 14 layers
        existing_clusters: Optional[tk.Path] = None,
        existing_pca: Optional[tk.Path] = None,
        concurrent: Optional[int] = 4,
    ):
        """
        :param environment: Path to the virtual environment.
        :param fairseq_root: Path to the root directory of Fairseq.
        :param w2v2_model_path: Path to the pre-trained Wav2Vec 2.0 model.
        :param input_audio_manifests: Base directory with the .tsv files for audio manifests.
        :param dim: Feature dimension (default: 512).
        :param layer: Layer for feature extraction (default: 14) In the paper, they have base-1 order, they refer to this layer as layer 15
        :param existing_clusters: Path to existing clusters (if any), should just be a centroids.npy file
        :param existing_pca: Path to existing PCA (if any), should be a directory containing 512_pca_A.npy and 512_pca_b.npy
        """
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.w2v2_model_path = w2v2_model_path
        self.input_audio_manifests = input_audio_manifests
        self.dim = dim
        self.layer = layer
        self.existing_clusters = existing_clusters
        self.existing_pca = existing_pca
        self.concurrent = concurrent

        # Paths for outputs
        self.out_features = self.output_path("audio_features", directory=True)
        self.out_features_precompute_pca512_cls128_mean = self.output_path("audio_features/precompute_pca512_cls128_mean", directory=True)
        self.out_features_precompute_pca512_cls128_mean_pooled = self.output_path("audio_features/precompute_pca512_cls128_mean_pooled", directory=True)
        self.out_features_clusters = self.output_path("audio_features/CLUS128/centroids.npy")
        self.out_features_pca = self.output_path("audio_features/pca", directory=True)

        self.out_chunk_manifest = self.output_path("chunk_manifest", directory=True)
        self.out_chunk_manifest_dirs = {
            task_id: self.output_path(f"chunk_manifest_{task_id}", directory=True)
            for task_id in range(1, self.concurrent + 1)
        }

        self.out_chunk_dirs = {
            task_id: self.output_path(f"features_chunk_{task_id}", directory=True)
            for task_id in range(1, self.concurrent + 1)
        }
        self.out_features_chunked = util.MultiOutputPath(
            self, "features_chunk_$(TASK)", self.out_chunk_dirs, cached=True
        )

        # Resource requirements
        self.rqmt = {"time": 100, "gpu": 1, "mem": 120}

    def tasks(self):
        yield Task("copy_manifest_files", mini_task=True)
        yield Task("split_audio_manifest", mini_task=True)
        yield Task("extract_features", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        yield Task("merge_features", rqmt=self.rqmt)
        if self.existing_clusters is None:
            yield Task("cluster_features", rqmt=self.rqmt)
        else:
            yield Task("cluster_features", mini_task=True)

        yield Task("apply_cluster", rqmt=self.rqmt)
        
        if self.existing_pca is None:
            yield Task("compute_pca", rqmt=self.rqmt)
        else:
            yield Task("compute_pca", mini_task=True)

        yield Task("apply_pca", rqmt=self.rqmt)
        yield Task("merge_clusters", rqmt=self.rqmt)
        yield Task("mean_pool", rqmt=self.rqmt)

    def env_call(self):
        """Prepare the environment activation call string."""
        env_call_str = (
            f"export FAIRSEQ_ROOT={self.fairseq_root.get_path()}  &&  export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH && "
        )
        if self.environment:
            env_call_str += f"source {self.environment.get_path()}/bin/activate && "
        return env_call_str

    def run_python_script(self, script_path, args):
        """
        Helper method to execute Python scripts with the given arguments.
        """
        sp.run(
            self.env_call() + f"python {script_path} " + " ".join(args),
            shell=True,
            check=True,
        )

    def copy_manifest_files(self):
        """
        Copy mandatory `.tsv` files (at least one is required) and optional `.wrd`, `.ltr`, `.phn`, and `dict*` files
        from the source directory to the target directory.
        """
        source_dir = self.input_audio_manifests.get_path()
        target_dir = self.out_features.get_path()

        # Ensure there is at least one '.tsv' file in the source directory
        tsv_files = glob.glob(os.path.join(source_dir, "*.tsv"))
        if not tsv_files:
            logging.warning("No .tsv manifest files found in the source directory. At least one is required.")

        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Copy mandatory .tsv files
        for tsv_file in tsv_files:
            shutil.copy(tsv_file, target_dir)

        # Copy optional files
        for pattern in ["*.wrd", "*.ltr", "*.phn", "dict*"]:
            optional_files = glob.glob(os.path.join(source_dir, pattern))
            for optional_file in optional_files:
                shutil.copy(optional_file, target_dir)

    def split_audio_manifest(self):
        """
        Split the input audio manifest files for all splits (e.g., train, valid, test)
        into smaller chunks for parallel processing.
        """
        # Detect all available `.tsv` manifest files in the input directory
        manifest_files = glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        for manifest_file in manifest_files:
            # Extract the split name from the filename (e.g., "train" from "train.tsv")
            split_name = os.path.splitext(os.path.basename(manifest_file))[0]

            # Read and split the manifest into chunks
            with open(manifest_file, "r") as f:
                lines = f.readlines()
                header = lines[0] if lines else ""
                data = lines[1:] if len(lines) > 1 else []
                chunk_size = math.ceil(len(data) / self.concurrent)
                chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

                # Write each chunk to its respective output directory
                for task_id, chunk in enumerate(chunks, 1):
                    # Create dynamically subdirectories based on the split name
                    chunk_path = os.path.join(self.out_chunk_manifest_dirs[task_id].get_path(), f"{split_name}.tsv")
                    os.makedirs(self.out_chunk_manifest_dirs[task_id].get_path(), exist_ok=True)
                    with open(chunk_path, "w") as chunk_file:
                        chunk_file.write(header)
                        chunk_file.writelines(chunk)

    def extract_features(self, task_id):
        """
        Extract features for the specified chunk of audio data in parallel for all splits.
        """
        # Detect all available manifest splits
        manifest_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_chunk_manifest_dirs[task_id].get_path(), "*.tsv"))
        ]

        for split in manifest_splits:
            chunk_dir = self.out_chunk_dirs[task_id].get_path()
            manifest_path = self.out_chunk_manifest_dirs[task_id].get_path()
            script_path = os.path.join(
                self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py"
            )

            # Check if the features already exist in the output directory
            output_feature_file = os.path.join(chunk_dir, f"{split}.npy")
            if os.path.exists(output_feature_file):
                logging.info(f"Features already exist for split '{split}' chunk {task_id}, skipping extraction.")
                continue  # Skip this split

            self.run_python_script(
                script_path,
                [
                    manifest_path,
                    "--split",
                    split,
                    "--save-dir",
                    chunk_dir,
                    "--checkpoint",
                    self.w2v2_model_path.get_path(),
                    "--layer",
                    str(self.layer),
                ],
            )

    def merge_features(self):
        """
        Merge the extracted features from all chunks into a single output directory.
        This handles all splits (`train`, `valid`, etc.) dynamically and processes
        both `.npy` and `.lengths` files.
        """

        # Detect all available splits dynamically based on '.npy' files in the chunk folders
        available_splits = set()
        for task_id in range(1, self.concurrent + 1):
            chunk_dir = self.out_chunk_dirs[task_id].get_path()
            npy_files = glob.glob(os.path.join(chunk_dir, "*.npy"))
            for npy_file in npy_files:
                split_name = os.path.splitext(os.path.basename(npy_file))[0]
                available_splits.add(split_name)

        from npy_append_array import NpyAppendArray

        # Process each available split
        for split in available_splits:
            output_npy_file = os.path.join(self.out_features.get_path(), f"{split}.npy")
            output_lengths_path = os.path.join(self.out_features.get_path(), f"{split}.lengths")

            

            npaa = NpyAppendArray(output_npy_file)

            # Merge the features from each chunk for the current split
            for task_id in range(1, self.concurrent + 1):
                chunk_dir = self.out_chunk_dirs[task_id].get_path()
                feature_file = os.path.join(chunk_dir, f"{split}.npy")

                if os.path.exists(feature_file):
                    data = np.load(feature_file, mmap_mode="r")
                    npaa.append(data)
                    del data
                    gc.collect()
                else:
                    logging.warning(f"Feature file not found: {feature_file}, skipping for chunk {task_id}.")

            # Merge length files for the current split
            with open(output_lengths_path, "w") as fout:
                for task_id in range(1, self.concurrent + 1):
                    chunk_dir = self.out_chunk_dirs[task_id].get_path()
                    length_file = os.path.join(chunk_dir, f"{split}.lengths")

                    if os.path.exists(length_file):
                        with open(length_file, "r") as fin:
                            fout.writelines(fin.readlines())
                    else:
                        logging.warning(f"Length file not found: {length_file}, skipping for chunk {task_id}.")

    def cluster_features(self):
        """
        Cluster features for each non-validation manifest.

        This function now finds all `.tsv` files in the features directory,
        filters out any containing "valid" in their name, and then runs
        clustering for each remaining manifest.
        """
        if self.existing_clusters != None:
            src_file = self.existing_clusters.get_path()
            dst_file = self.out_features_clusters.get_path()
            dst_dir = os.path.dirname(dst_file)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            return


        features_dir = self.out_features.get_path()
        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/wav2vec_cluster_faiss.py"
        )

        # Find all .tsv files that do not contain 'valid' in their name.
        # We sort the results to ensure deterministic behavior.
        manifest_files = sorted([
            f for f in glob.glob(os.path.join(features_dir, "*.tsv"))
            if "valid" not in os.path.basename(f)
        ])

        if not manifest_files:
            logging.warning("No non-validation .tsv manifest files found for clustering.")
            return

        # Note: The clustering script saves to a fixed filename ('CLUS128.pt').
        # If multiple non-validation manifests are processed, the output from the last one
        # will overwrite the previous ones.
        for manifest_file in manifest_files:
            logging.info(f"Running feature clustering for manifest: {os.path.basename(manifest_file)}")
            self.run_python_script(
                script_path,
                [
                    manifest_file,
                    "--checkpoint",
                    self.w2v2_model_path.get_path(),
                    "--save-dir",
                    self.out_features.get_path(),
                    "-f",
                    "CLUS128",
                    "--sample-pct",
                    "0.01",
                    "--layer",
                    str(self.layer),
                ],
            )

    def apply_cluster(self):
        """
        Apply clustering to feature splits using the trained cluster model.
        """
        all_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        ]

        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py"
        )

        for split in all_splits:
            self.run_python_script(
                script_path,
                [
                    self.out_features.get_path(),
                    "--checkpoint",
                    self.w2v2_model_path.get_path(),
                    "--path",
                    os.path.join(self.out_features.get_path(), "CLUS128"),
                    "--split",
                    split,
                    "--layer",
                    str(self.layer),
                ],
            )
    
    def compute_pca(self):
        """
        Task to compute PCA on the features of each non-validation split.

        This function identifies all non-validation splits by looking for `.tsv`
        files without "valid" in their names. It then computes PCA for the
        corresponding `.npy` feature file of each split.
        """

        if self.existing_pca != None:
            for file_name in os.listdir(self.existing_pca.get_path()):
                src_file = os.path.join(self.existing_pca.get_path(), file_name)
                dst_file = os.path.join(self.out_features_pca.get_path(), file_name)
                os.makedirs(self.out_features_pca.get_path(), exist_ok=True)
                shutil.copy2(src_file, dst_file)
            return
            
        
        features_dir = self.out_features.get_path()
        script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/pca.py")

        # Find all .tsv files that do not contain 'valid' in their name.
        # We sort the results to ensure deterministic behavior.
        manifest_files = sorted([
            f for f in glob.glob(os.path.join(features_dir, "*.tsv"))
            if "valid" not in os.path.basename(f)
        ])

        if not manifest_files:
            logging.warning("No non-validation manifest files found for PCA computation.")
            return

        # Note: The PCA script saves to a fixed filename ('pca.pt'). If multiple
        # non-validation splits are processed, the output from the last one will
        # overwrite the previous ones.
        for manifest_file in manifest_files:
            split_name = os.path.splitext(os.path.basename(manifest_file))[0]
            feature_file_npy = os.path.join(features_dir, f"{split_name}.npy")

            if not os.path.exists(feature_file_npy):
                logging.warning(
                    f"Feature file {feature_file_npy} not found, "
                    f"skipping PCA computation for split '{split_name}'."
                )
                continue

            logging.info(f"Computing PCA for split: {split_name}")
            output_pca_dir = os.path.join(self.out_features.get_path(), "pca")

            self.run_python_script(
                script_path,
                [
                    feature_file_npy,
                    "--output",
                    output_pca_dir,
                    "--dim",
                    str(self.dim),
                ],
            )

    def apply_pca(self):
        """
        Task to apply PCA transformation to all feature splits.
        """
        all_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        ]
        output_precompute_pca_dir = os.path.join(self.out_features.get_path(), f"precompute_pca{self.dim}")
        pca_path = os.path.join(self.out_features.get_path(), f"pca/{self.dim}_pca")
        script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/apply_pca.py")

        for split in all_splits:
            self.run_python_script(
                script_path,
                [
                    self.out_features.get_path(),
                    "--split",
                    split,
                    "--save-dir",
                    output_precompute_pca_dir,
                    "--pca-path",
                    pca_path,
                    "--batch-size",
                    "1048000",
                ],
            )

    def merge_clusters(self):
        """
        Task to merge precomputed PCA features with clusters for all splits.
        """
        all_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        ]
        precompute_pca_dir = os.path.join(self.out_features.get_path(), f"precompute_pca{self.dim}")
        cluster_dir = os.path.join(self.out_features.get_path(), "CLUS128")
        output_merge_dir = os.path.join(self.out_features.get_path(), f"precompute_pca{self.dim}_cls128_mean")
        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/merge_clusters.py"
        )

        for split in all_splits:
            self.run_python_script(
                script_path,
                [
                    precompute_pca_dir,
                    "--cluster-dir",
                    cluster_dir,
                    "--split",
                    split,
                    "--save-dir",
                    output_merge_dir,
                    "--pooling",
                    "mean",
                ],
            )

    def mean_pool(self):
        """
        Task to perform mean pooling on the merged PCA and cluster features.
        """
        all_splits = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(os.path.join(self.out_features.get_path(), "*.tsv"))
        ]
        input_mean_dir = os.path.join(self.out_features.get_path(), f"precompute_pca{self.dim}_cls128_mean")
        output_mean_pooled_dir = os.path.join(
            self.out_features.get_path(), f"precompute_pca{self.dim}_cls128_mean_pooled"
        )
        script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/mean_pool.py")

        for split in all_splits:
            self.run_python_script(
                script_path,
                [
                    input_mean_dir,
                    "--save-dir",
                    output_mean_pooled_dir,
                    "--split",
                    split,
                ],
            )

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__ or cls.__sis_hash_exclude__[k] != v:
                d[k] = v

        for k, org, replacement in cls.__sis_hash_overwrite__:
            if k in d and d[k] == org:
                d[k] = replacement

        for k, v in cls.__sis_hash_constant__.items():
            d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__)) 


