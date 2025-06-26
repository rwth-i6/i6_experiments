import math
import os
import shutil
import subprocess as sp
from typing import Optional, Type, List, Dict, Any
from sisyphus import Job, Task, tk
import logging
from .wav2vec_data_utils import get_rvad_root, get_fairseq_root
import i6_core.util as util
from npy_append_array import NpyAppendArray
import numpy as np
import glob
import h5py
import gc


class Wav2VecUDeleteSilencesInAudioJob(Job):
    """
    Class to perform audio preprocessing for Wav2Vec-U.
    This includes generating manifests, performing VAD, and removing silence from audio files.
    """

    def __init__(
        self,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        audio_dir: Type[tk.Path],
        valid_percent: Optional[float] = 0.01,
        extension: Optional[str] = "flac",
        rvad_root: Optional[tk.Path] = None,
        initial_manifests_dir: Optional[tk.Path] = None,
        w2v2_model_path: tk.Path = None,
        concurrent: Optional[int] = 8,
    ):
        """
        :param environment: Path to the virtual environment.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param audio_dir: Path to the directory containing audio files.
        :param valid_percent: Percentage of validation data to use.
        :param extension: Audio file extension (default: "flac").
        :param rvad_root: Path to the RVAD root directory.
        :param initial_manifests_dir: Directory containing initial manifests train.tsv and valid.tsv.
        :param w2v2_model_path: Path to the Wav2Vec 2.0 model.
        :param concurrent: Number of concurrent tasks to run (default: 16).
        """
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.audio_dir = audio_dir
        self.valid_percent = valid_percent
        self.extension = extension
        self.rvad_root = (
            rvad_root
            if rvad_root
            else tk.Path("/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/rVADfast")
        )
        self.w2v2_model_path = w2v2_model_path
        self.initial_manifests_dir = initial_manifests_dir
        self.valid_manifest_exists = False
        self.concurrent = concurrent

        # self.out_manifest_and_vads = self.output_path("manifest_and_vads", directory=True)
        self.out_preprocessed_audio = self.output_path("preprocessed_audio", directory=True)
        self.out_preprocessed_manifest = self.output_path("preprocessed_manifest", directory=True)

        # Resource requirements
        self.rqmt = {"time": 64, "cpu": 4, "gpu": 4, "mem": 64}

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
        yield Task("make_initial_manifest", rqmt=self.rqmt)

        yield Task("split_manifest", rqmt=self.rqmt)

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
            train_manifest = os.path.join(self.initial_manifests_dir.get_path(), "train.tsv")
            valid_manifest = os.path.join(self.initial_manifests_dir.get_path(), "valid.tsv")

            if not os.path.exists(train_manifest):
                raise FileNotFoundError(f"train.tsv not found in {self.initial_manifests_dir.get_path()}")

            self.valid_manifest_exists = os.path.exists(valid_manifest)

            os.makedirs(self.out_manifest_and_vads.get_path(), exist_ok=True)
            sp.run(f"cp {train_manifest} {self.out_manifest_and_vads.get_path()}/train.tsv", shell=True, check=True)

            if self.valid_manifest_exists:
                sp.run(f"cp {valid_manifest} {self.out_manifest_and_vads.get_path()}/valid.tsv", shell=True, check=True)

        else:
            manifest_script = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/wav2vec_manifest.py")
            sh_call = (
                f"python {manifest_script} {self.audio_dir.get_path()} "
                f"--ext {self.extension} --dest {self.out_manifest_and_vads.get_path()} "
                f"--valid-percent 0"
            )
            sp.run(self.env_call() + sh_call, shell=True, check=True)

    def split_manifest(self):
        """Split the manifest into chunks for parallel processing"""
        train_manifest = os.path.join(self.out_manifest_and_vads.get_path(), "train.tsv")
        valid_manifest = os.path.join(self.out_manifest_and_vads.get_path(), "valid.tsv")

        with open(train_manifest, "r") as f:
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

                chunk_manifest = os.path.join(chunk_dir, "train.tsv")
                with open(chunk_manifest, "w") as f_out:
                    f_out.write(header)
                    f_out.writelines(chunk)

        if self.valid_manifest_exists:
            logging.info("Valid manifest exists, splitting it as well")
            with open(valid_manifest, "r") as f:
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

                    chunk_manifest = os.path.join(chunk_dir, "valid.tsv")
                    with open(chunk_manifest, "w") as f_out:
                        f_out.write(header)
                        f_out.writelines(chunk)

    def run_vad(self, task_id):

        manifest_and_vad_dir = self.out_manifest_and_vads_dirs[task_id].get_path()

        vads_script = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/vads.py")
        sh_call = (
            f"python {vads_script} -r $RVAD_ROOT "
            f"< {manifest_and_vad_dir}/train.tsv "
            f"> {manifest_and_vad_dir}/train.vads"
        )
        sp.run(self.env_call() + sh_call, shell=True, check=True)

    def remove_silence_from_audio(self, task_id):

        manifest_and_vad_dir = self.out_manifest_and_vads_dirs[task_id].get_path()

        silence_remove_script = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/remove_silence.py"
        )

        sh_call = (
            f"python {silence_remove_script} "
            f"--tsv {manifest_and_vad_dir}/train.tsv "
            f"--vads {manifest_and_vad_dir}/train.vads "
            f"--out {self.out_preprocessed_audio.get_path()} "
        )
        sp.run(self.env_call() + sh_call, shell=True, check=True)

    def make_final_manifest(self):
        manifest_script = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/wav2vec_manifest.py")
        sh_call = (
            f"python {manifest_script} {self.out_preprocessed_audio.get_path()} "
            f"--ext {self.extension} --dest {self.out_preprocessed_manifest.get_path()} "
            f"--valid-percent {self.valid_percent}"
        )
        sp.run(self.env_call() + sh_call, shell=True, check=True)


class Wav2VecUFeaturizeAudioJob(Job):
    """
    Job to featurize audio using Wav2Vec for unsupervised learning (Wav2Vec-U).

    This job prepares audio features by running the `prepare_audio.sh` script.
    """

    def __init__(
        self,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        w2v2_model_path: tk.Path,
        input_audio_manifests: tk.Path,
        dim: Optional[int] = 512,
        layer: Optional[int] = 14,  # Small w2v2 models don't have 14 layers
        concurrent: Optional[int] = 4,
    ):
        """
        :param environment: Path to the virtual environment.
        :param fairseq_root: Path to the root directory of Fairseq.
        :param w2v2_model_path: Path to the pre-trained Wav2Vec 2.0 model.
        :param input_audio_manifests: Base directory with the .tsv files for audio manifests.
        :param dim: Feature dimension (default: 512).
        :param layer: Layer for feature extraction (default: 14) In the paper, they have base-1 order, they refer to this layer as layer 15
        """
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.w2v2_model_path = w2v2_model_path
        self.input_audio_manifests = input_audio_manifests
        self.dim = dim
        self.layer = layer
        self.concurrent = concurrent

        # Paths for outputs
        self.out_features = self.output_path("audio_features", directory=True)

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
        self.rqmt = {"time": 100, "cpu": 1, "gpu": 2, "mem": 150}

    def tasks(self):
        yield Task("copy_manifest_files", rqmt=self.rqmt)
        yield Task("split_audio_manifest", rqmt=self.rqmt)
        yield Task("extract_features", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        yield Task("merge_features", rqmt=self.rqmt)
        yield Task("cluster_features", rqmt=self.rqmt)
        yield Task("apply_cluster", rqmt=self.rqmt)
        yield Task("compute_pca", rqmt=self.rqmt)
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
                    data = np.load(feature_file)
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
        Cluster features across all chunks.
        """
        combined_manifest = os.path.join(self.out_features.get_path(), "train.tsv")
        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/wav2vec_cluster_faiss.py"
        )

        self.run_python_script(
            script_path,
            [
                combined_manifest,
                "--checkpoint",
                self.w2v2_model_path.get_path(),
                "--save-dir",
                self.out_features.get_path(),
                "-f",
                "CLUS128",
                "--sample-pct",
                "0.01",
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
                ],
            )

    def compute_pca(self):
        """
        Task to compute PCA on the processed features.
        """
        train_split = "train"
        train_split_npy = os.path.join(self.out_features.get_path(), f"{train_split}.npy")
        output_pca_dir = os.path.join(self.out_features.get_path(), "pca")
        script_path = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/pca.py")

        self.run_python_script(
            script_path,
            [
                train_split_npy,
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
