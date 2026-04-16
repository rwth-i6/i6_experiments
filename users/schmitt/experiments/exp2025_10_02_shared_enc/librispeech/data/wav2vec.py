###########################################################
# Imports
###########################################################
import os
from typing import Optional, Dict, Any, List, Union, Tuple
import gc
import tempfile
import os
import shutil
import random

import numpy as np

from i6_core.text.processing import ConcatenateJob, PipelineJob, TailJob, HeadJob
from i6_core.tools.download import DownloadJob
from i6_core.util import uopen

from .audio_preprocessing import process_audio
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.audio_preprocessing import process_audio as process_audio_enrique
from i6_experiments.users.schmitt.hdf import dump_hdf_numpy
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import (
    get_rvad_root,
    get_fairseq_root,
)

from sisyphus import tk, Path, Job, Task
from sisyphus.delayed_ops import DelayedFormat, DelayedBase

from returnn.datasets.hdf import SimpleHDFWriter


def run_meta_experiments(
        librispeech_key: str,
        vad_concurrent: int = 1,
        dump_hdf_concurrent: int = 1,
        existing_clusters: Optional[Path] = None,
        fixed_random_subset: Optional[int] = None,
        max_abs_value: Optional[float] = None,
        use_tsv_for_cluster_ids: bool = False,
        remove_cluster_repetitions: bool = False,
        use_correct_seq_tags_for_cluster_ids: bool = False,
) -> Tuple[List[Path], Path, List[Path]]:
    environment = "/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3"
    fairseq_root = get_fairseq_root(
        python_env=tk.Path(environment),
        fairseq_root=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
        ),
    )


    ################################################################
    ########## Configuration for the Wav2VecU pipeline ##########
    ################################################################

    max_audios_per_manifest = (
        None  # Used to limit the number of audio files in each manifest file, mainly for debugging purposes
    )

    w2v2model = "large_60kh"  # Options: "base", "large_960h", "large_60kh"
    feature_extraction_layer = 14  # Layer to extract features from the Wav2Vec2 model, w2v-u paper uses layer 14
    assert not (
        (w2v2model == "base" and feature_extraction_layer > 11) or feature_extraction_layer > 23
    ), "not so many layers in the model"

    training_audio = librispeech_key  # Options: "train-clean-100", "train-clean-360", "train-other-500", "train-other-960", "dev-clean", "dev-complete", "dev-other", "test-clean", "test-other"
    training_audio_extension = "flac"  # Options: "flac", "wav"
    training_valid_percent = 0.01  # Percentage of the training data to be used for validation

    alias = "wav2vec_u_librispeech_gan_training_" + training_audio + "_" + w2v2model
    audio_alias = os.path.join(alias, "audio")
    training_audio_alias = os.path.join(audio_alias, training_audio)

    ################################################################
    ########### Prepare the aduio and featurize it ############
    ################################################################

    environment = tk.Path(environment)

    if w2v2model == "large_960h":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt",
            target_filename="wav2vec2_large_960h_no_finetune.pt",
        ).out_file
    if w2v2model == "large_60kh":
        assert w2v2model == "large_60kh"
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt",
            target_filename="wav2vec_60kh_no_finetune.pt",
        ).out_file
    if w2v2model == "base":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
            target_filename="wav2vec_small.pt",
        ).out_file  # All of this models are fully unsupervised

    ################### Training data preprocessing (runs once) ############################

    training_audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", training_audio))
    delete_silences_job, featurize_training_audio_job = process_audio(
        env_vad=environment,
        env_features=tk.Path("/work/asr4/schmitt/venvs/fairseq_env"),
        fairseq_root=fairseq_root,
        audio_dir=training_audio_dir,
        valid_percent=training_valid_percent,
        ext=training_audio_extension,
        rvad_root=get_rvad_root(),
        concurrent=vad_concurrent,
        layer=feature_extraction_layer,
        model_path=w2v2_model_path,
        alias_prefix=training_audio_alias,
        alias_delete="delete_silences/"
        + w2v2model
        + "/layer_"
        + str(feature_extraction_layer)
        + "valid_"
        + str(training_valid_percent),
        alias_feat="featurize_audio/"
        + w2v2model
        + "/layer_"
        + str(feature_extraction_layer)
        + "valid_"
        + str(training_valid_percent),
        max_n_audios_per_manifest=max_audios_per_manifest,
        name_the_manifests_just_train_and_valid=True,
        existing_clusters=existing_clusters,
    )

    npy_file, seq_lengths_file, tsv_file = [DelayedFormat(
      f"{{}}/{file_name}",
      featurize_training_audio_job.out_features_precompute_pca512_cls128_mean_pooled
    ) for file_name in ("train.npy", "train.lengths", "train.tsv")]

    dump_features_to_hdf_job = DumpNumpyFeaturesToHdfJobV2(
        npy_file=npy_file,
        seq_lengths_file=seq_lengths_file,
        tsv_file=tsv_file,
        concurrent=dump_hdf_concurrent,
        fixed_random_subset=fixed_random_subset,
        max_abs_value=max_abs_value,
    )
    dump_features_to_hdf_job.add_alias(
        f"data/librispeech/segmented_wav2vec_features_hdf/{librispeech_key}{'-maxabs-' + str(max_abs_value) if max_abs_value else ''}"
        f"{'_subset-' + str(fixed_random_subset) if fixed_random_subset else ''}"
    )
    # tk.register_output(
    #     dump_features_to_hdf_job.get_one_alias(),
    #     dump_features_to_hdf_job.out_hdfs[0]
    # )

    cluster_id_file_train, cluster_id_file_valid = [DelayedFormat(
        f"{{}}/CLUS128/{set}.src",
        featurize_training_audio_job.out_features
    ) for set in ("train", "valid")]
    cluster_id_file_train = HeadJob(cluster_id_file_train, num_lines="-0").out
    cluster_id_file_valid = HeadJob(cluster_id_file_valid, num_lines="-0").out
    cluster_id_file = ConcatenateJob([cluster_id_file_train, cluster_id_file_valid]).out
    tsv_file_clusters_train, tsv_file_clusters_valid = [DelayedFormat(
        f"{{}}/CLUS128/{file_name}",
        featurize_training_audio_job.out_features
    ) for file_name in ("train.tsv", "valid.tsv")]
    # remove header line from tsv (this also converts the DelayedFormat to a Path as a nice side effect)
    tsv_path_train = TailJob(tsv_file_clusters_train, num_lines="+2").out
    tsv_path_valid = TailJob(tsv_file_clusters_valid, num_lines="+2").out
    tsv_path = ConcatenateJob([tsv_path_train, tsv_path_valid]).out
    # reformat line "146197/1061-146197-0004.flac    237120" -> "train-other-960/1061-146197-0004/1061-146197-0004"
    reformat_seq_tags = PipelineJob(
        tsv_path,
        [
            rf"sed -E 's|^[^/]*/([^.]+)\.flac.*|{librispeech_key}/\1/\1|'"
        ]
    ).out
    dump_cluster_ids_to_hdf_job = DumpClusterIndicesToHdfJob(
        text_file=cluster_id_file,
        num_clusters=128,
        concurrent=dump_hdf_concurrent,
        fixed_random_subset=fixed_random_subset,
        tsv_file=tsv_file if use_tsv_for_cluster_ids else None,
        seq_tag_file=reformat_seq_tags if use_correct_seq_tags_for_cluster_ids else None,
        remove_cluster_repetitions=remove_cluster_repetitions,
    )
    # output/data/librispeech/segmented_wav2vec_cluster_ids_hdf/train-other-960/.
    dump_cluster_ids_to_hdf_job.add_alias(
        f"data/librispeech/segmented_wav2vec_cluster_ids_hdf/{librispeech_key}{'_subset-' + str(fixed_random_subset) if fixed_random_subset else ''}"
        f"{'_no_reps' if remove_cluster_repetitions else ''}"
    )
    # tk.register_output(
    #     dump_cluster_ids_to_hdf_job.get_one_alias(),
    #     dump_cluster_ids_to_hdf_job.out_hdfs[0]
    # )

    return list(dump_features_to_hdf_job.out_hdfs.values()), featurize_training_audio_job.out_features_clusters, list(dump_cluster_ids_to_hdf_job.out_hdfs.values())


class DumpNumpyFeaturesToHdfJob(Job):
    def __init__(
            self,
            npy_file: DelayedBase,
            seq_lengths_file: DelayedBase,
            tsv_file: DelayedBase,
            concurrent: int = 10,
    ):
        self.npy_file = npy_file
        self.seq_lengths_file = seq_lengths_file
        self.tsv_file = tsv_file
        self.concurrent = concurrent

        self.out_hdfs = {
            i: self.output_path(f"data_{i}.hdf") for i in range(self.concurrent)
        }

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 32, "time": 4}, args=range(1, self.concurrent + 1))

    def run(self, task_id):
        import gc
        import tempfile
        import os
        import shutil

        with uopen(self.seq_lengths_file.get(), "r") as f:
            seq_lengths = [int(line.strip()) for line in f.readlines()]
        with uopen(self.tsv_file.get(), "r") as f:
            seq_tags = [line.strip().split("\t")[0].split(".flac")[0] for line in f.readlines()[1:]]  # skip header line

        data = np.load(self.npy_file.get(), mmap_mode="r")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_hdf = os.path.join(tmp_dir, f"data_{task_id}.hdf")

            hdf_writer = SimpleHDFWriter(filename=tmp_hdf, dim=512, ndim=2)
            prev_pos = 0
            num_seqs = len(seq_lengths) // self.concurrent
            num_processed = 0
            for i, (seq_len, seq_tag) in enumerate(zip(seq_lengths, seq_tags)):
                if (i % self.concurrent) == (task_id - 1):
                    seq_data = data[prev_pos:prev_pos + seq_len].astype(np.float32, copy=False)  # (T, F)
                    is_nan = np.isnan(seq_data).any()
                    if is_nan:
                        print("Found NaN values in sequence:", seq_tag)
                        continue

                    seq_data = seq_data[None, ...]  # (1, T, F)

                    seq_lens = {0: np.array([seq_len])}
                    batch_seq_sizes = np.expand_dims(seq_lens[0], 1)

                    hdf_writer.insert_batch(
                        seq_data,
                        seq_len=seq_lens,
                        seq_tag=[seq_tag],
                        extra={"seq_sizes": batch_seq_sizes}
                    )

                    num_processed += 1
                    if num_processed % 5000 == 0:
                        gc.collect()
                        print(f"Processed sequence {num_processed}/{num_seqs} ({num_processed / num_seqs * 100:.2f}%)")

                prev_pos += seq_len

            hdf_writer.close()
            shutil.move(tmp_hdf, self.out_hdfs[task_id - 1].get_path())


class DumpNumpyFeaturesToHdfJobV2(Job):
    """
    Convert a concatenated NumPy feature file into shuffled per-worker HDF files.

    This job reads:
      - a `.npy` file with all feature frames concatenated along time,
      - a file with per-sequence lengths,
      - a TSV file with sequence tags.

    It creates a global deterministic shuffle of sequence indices (seed=42),
    splits them across `concurrent` workers via modulo partitioning, and each
    worker writes its assigned sequences into an HDF file using SimpleHDFWriter.

    Parameters
    ----------
    npy_file : DelayedBase
        Path to the concatenated `.npy` feature file.
    seq_lengths_file : DelayedBase
        File containing one sequence length per line.
    tsv_file : DelayedBase
        TSV with sequence tags (first column, header skipped).
    concurrent : int
        Number of workers; determines how shuffled indices are partitioned.
    fixed_random_subset : int, optional
        Maximum number of sequences *per worker* after partitioning.

    Each worker produces one HDF file containing only its assigned (and optionally
    truncated) shuffled sequences.
    """
    def __init__(
            self,
            npy_file: DelayedBase,
            seq_lengths_file: DelayedBase,
            tsv_file: DelayedBase,
            concurrent: int = 10,
            fixed_random_subset: Optional[int] = None,
            max_abs_value: Optional[float] = None,
    ):
        self.npy_file = npy_file
        self.seq_lengths_file = seq_lengths_file
        self.tsv_file = tsv_file
        self.concurrent = concurrent
        self.fixed_random_subset = fixed_random_subset
        self.max_abs_value = max_abs_value

        self.out_hdfs = {
            i: self.output_path(f"data_{i}.hdf") for i in range(self.concurrent)
        }

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 32, "time": 4}, args=range(1, self.concurrent + 1))

    def run(self, task_id):
        with uopen(self.seq_lengths_file.get(), "r") as f:
            seq_lengths = [int(line.strip()) for line in f.readlines()]
        seq_starts = np.cumsum([0] + seq_lengths[:-1]).tolist()
        with uopen(self.tsv_file.get(), "r") as f:
            seq_tags = [line.strip().split("\t")[0].split(".flac")[0] for line in f.readlines()[1:]]  # skip header line
        assert len(seq_lengths) == len(seq_tags), (
            f"mismatch between seq lengths and seq tags: {len(seq_lengths)} vs {len(seq_tags)}"
        )

        data = np.load(self.npy_file.get(), mmap_mode="r")
        assert data.shape[0] == sum(seq_lengths), "data length does not match sum of seq lengths"

        num_seqs = len(seq_lengths)
        indices = list(range(num_seqs))
        random.Random(42).shuffle(indices)
        worker_indices = [i for i in indices if (i % self.concurrent) == (task_id - 1)]
        if self.fixed_random_subset is not None:
            worker_indices = worker_indices[:self.fixed_random_subset]
        num_seqs = len(worker_indices)

        seq_lengths = [seq_lengths[i] for i in worker_indices]
        seq_starts = [seq_starts[i] for i in worker_indices]
        seq_tags = [seq_tags[i] for i in worker_indices]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_hdf = os.path.join(tmp_dir, f"data_{task_id}.hdf")

            hdf_writer = SimpleHDFWriter(filename=tmp_hdf, dim=data.shape[1], ndim=2)

            num_processed = 0
            for seq_start, seq_len, seq_tag in zip(seq_starts, seq_lengths, seq_tags):
                seq_data = data[seq_start:seq_start + seq_len].astype(np.float32, copy=False)  # (T, F)
                is_nan = np.isnan(seq_data).any()
                if is_nan:
                    print("Found NaN values in sequence:", seq_tag, ". Skipping it.")
                    continue
                if self.max_abs_value is not None:
                    max_abs = np.max(np.abs(seq_data))
                    if max_abs > self.max_abs_value:
                        print("Found extreme values in sequence:", seq_tag, " (max abs value:", max_abs, "). Skipping it.")
                        continue

                seq_data = seq_data[None, ...]  # (1, T, F)

                seq_lens = {0: np.array([seq_len])}
                batch_seq_sizes = seq_lens[0][:, None]

                hdf_writer.insert_batch(
                    seq_data,
                    seq_len=seq_lens,
                    seq_tag=[seq_tag],
                    extra={"seq_sizes": batch_seq_sizes}
                )

                num_processed += 1
                if num_processed % 5000 == 0:
                    gc.collect()
                    print(f"Processed sequence {num_processed}/{num_seqs} ({num_processed / num_seqs * 100:.2f}%)")

            hdf_writer.close()
            shutil.move(tmp_hdf, self.out_hdfs[task_id - 1].get_path())

    @classmethod
    def hash(cls, kwargs):
        if kwargs.get("max_abs_value", None) is None:
            kwargs.pop("max_abs_value")

        return super().hash(kwargs)


class DumpClusterIndicesToHdfJob(Job):
    def __init__(
            self,
            text_file: DelayedBase,
            num_clusters: int,
            concurrent: int = 10,
            fixed_random_subset: Optional[int] = None,
            tsv_file: Optional[DelayedBase] = None,
            seq_tag_file: Optional[Path] = None,
            remove_cluster_repetitions: bool = False,
    ):
        """

        Args:
            text_file: each line contains a sequence of phonemes separated by space
            phoneme_file: line format: <phoneme> <integer (count?)>
            concurrent: number of concurrent hdf files to dump
            fixed_random_subset: if given, only use a fixed random subset of the data
        """
        self.text_file = text_file
        self.num_clusters = num_clusters
        self.concurrent = concurrent
        self.fixed_random_subset = fixed_random_subset
        self.tsv_file = tsv_file
        self.seq_tag_file = seq_tag_file
        self.remove_cluster_repetitions = remove_cluster_repetitions

        assert not (self.tsv_file is not None and self.seq_tag_file is not None), (
            "only one of tsv_file and seq_tag_file can be given")

        self.out_hdfs = {
            i: self.output_path(f"data_{i}.hdf") for i in range(self.concurrent)
        }
        self.out_vocab = self.output_path("phonemes.vocab")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 4}, args=range(1, self.concurrent + 1))

    def run(self, task_id):
        import gc
        import tempfile
        import random

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_hdf = os.path.join(tmp_dir, f"data_{task_id}.hdf")
            hdf_writer = SimpleHDFWriter(filename=tmp_hdf, dim=self.num_clusters, ndim=1)

            if self.tsv_file is not None:
                with uopen(self.tsv_file.get(), "r") as f:
                    seq_tags = [line.strip().split("\t")[0].split(".flac")[0] for line in f.readlines()[1:]]  # skip header line
            else:
                with uopen(self.seq_tag_file.get(), "r") as f:
                    seq_tags = [line.strip() for line in f.readlines()]
            random.Random(42).shuffle(seq_tags)

            with uopen(self.text_file.get(), "r") as f:
                lines = f.readlines()
            random.Random(42).shuffle(lines)

            num_lines = len(lines)

            assert num_lines == len(seq_tags), (
                f"mismatch between seq lengths and seq tags: {num_lines} vs {len(seq_tags)}"
            )

            lines = [lines[i] for i in range(num_lines) if (i % self.concurrent) == (task_id - 1)]
            seq_tags = [seq_tags[i] for i in range(num_lines) if (i % self.concurrent) == (task_id - 1)]
            if self.fixed_random_subset is not None:
                lines = lines[:self.fixed_random_subset]
                seq_tags = seq_tags[:self.fixed_random_subset]
            num_lines = len(lines)
            gc.collect()

            for i, line in enumerate(lines):
                cluster_ids = line.strip().split()
                data = [int(id) for id in cluster_ids]
                if self.remove_cluster_repetitions:
                    data = [i for (i, j) in zip(data, [None] + data) if i != j]
                seq_len = len(data)
                data = np.array([data])  # (1, T)

                seq_lens = {0: np.array([seq_len])}
                batch_seq_sizes = np.expand_dims(seq_lens[0], 1)

                hdf_writer.insert_batch(
                    data,
                    seq_len=seq_lens,
                    seq_tag=[seq_tags[i]],
                    extra={"seq_sizes": batch_seq_sizes}
                )

                if i % 10_000 == 0:
                    gc.collect()
                    print(f"Processed sequence {i}/{num_lines} ({i / num_lines * 100:.1f}%)")

            hdf_writer.close()
            shutil.move(tmp_hdf, self.out_hdfs[task_id - 1].get_path())

    @classmethod
    def hash(cls, kwargs):
        if kwargs.get("tsv_file", None) is None:
            kwargs.pop("tsv_file")
        if kwargs.get("seq_tag_file", None) is None:
            kwargs.pop("seq_tag_file")
        if kwargs.get("remove_cluster_repetitions", False) is False:
            kwargs.pop("remove_cluster_repetitions")

        return super().hash(kwargs)
