import copy
import ast
import gzip
import os
import re
import zipfile
from dataclasses import asdict
import xml.etree.ElementTree as ET

import numpy as np
from sisyphus import Job, Task, tk

from i6_core.lib import corpus
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from ...data.phmm_common import TrainingDatasets
from ...experiments.phmm_phon.rasr_table_decoding import OggZipTextToBlissSubsetJob, SelectSegmentSubsetJob
from ...lm_score_dump_jobs import DumpNgramConvLmScoreTableJob
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_lm import create_lm_image_for_lexicon
from ...phmm_pipeline import ASRModel, search, training
from ...phmm_rasr import CreateLibrasrVenvJob, build_librasr_phone_table_recognition_config
from ...pytorch_networks.phmm.cluster_sequence_one_to_one_generative_cfg import ModelConfig as OneToOneModelConfig
from ...pytorch_networks.phmm.cluster_sequence_phmm_generative_cfg import ModelConfig
from ...pytorch_networks.phmm.decoder.phoneme_argmax_decoder_v1 import DecoderConfig as ArgmaxDecoderConfig
from ...pytorch_networks.phmm.decoder.rasr_trainable_table_decoding_v1 import DecoderConfig
from ...pytorch_networks.phmm.vector_sequence_phmm_generative_cfg import ModelConfig as VectorModelConfig
from ...segment_clustering_jobs import MakeSparseClusterHDFDatasetCompatibleJob


class _ReturnnDataset:
    def __init__(self, opts):
        self.opts = opts

    def as_returnn_opts(self):
        return copy.deepcopy(self.opts)


def _num_str(value: float) -> str:
    return ("%g" % value).replace(".", "p")


class ExtractHDFSubsetBySegmentsJob(Job):
    def __init__(self, *, hdf_files, segment_file: tk.Path, output_filename: str = "subset.hdf"):
        if not isinstance(hdf_files, (list, tuple)):
            hdf_files = [hdf_files]
        self.hdf_files = list(hdf_files)
        self.segment_file = segment_file
        self.out_hdf = self.output_path(output_filename)
        self.out_report = self.output_path("subset_stats.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 24, "time": 5})

    @staticmethod
    def _as_str(tag):
        if isinstance(tag, bytes):
            return tag.decode("utf-8")
        return str(tag)

    def run(self):
        import h5py

        with open(tk.uncached_path(self.segment_file), "rt", encoding="utf-8") as f:
            wanted = {line.strip() for line in f if line.strip()}
        if not wanted:
            raise ValueError(f"No segments found in {self.segment_file}")

        selected = []
        input_dim = None
        labels_dtype = None
        attrs = {}
        for hdf_file in self.hdf_files:
            hdf_path = tk.uncached_path(hdf_file)
            with h5py.File(hdf_path, "r") as hdf:
                if input_dim is None:
                    input_dim = int(hdf["inputs"].shape[1]) if hdf["inputs"].ndim == 2 else 1
                    labels_dtype = hdf["labels"].dtype if "labels" in hdf else h5py.special_dtype(vlen=bytes)
                    attrs = dict(hdf.attrs)
                seq_lengths = hdf["seqLengths"][:]
                seq_tags = hdf["seqTags"][:]
                input_offsets = np.concatenate(([0], np.cumsum(seq_lengths[:, 0], dtype=np.int64)))
                for seq_idx, raw_tag in enumerate(seq_tags):
                    tag = self._as_str(raw_tag)
                    if tag in wanted:
                        start = int(input_offsets[seq_idx])
                        end = int(input_offsets[seq_idx + 1])
                        selected.append((hdf_path, seq_idx, tag, start, end, seq_lengths[seq_idx].copy()))

        selected_tags = {item[2] for item in selected}
        missing = wanted - selected_tags
        if missing:
            raise ValueError(f"Missing {len(missing)} requested segments in HDFs, e.g. {sorted(missing)[:5]}")

        total_timesteps = int(sum(end - start for _, _, _, start, end, _ in selected))
        seq_lengths_out = np.asarray([item[5] for item in selected], dtype=np.int32)
        with h5py.File(self.out_hdf.get_path(), "w") as out_hdf:
            inputs_out = out_hdf.create_dataset(
                "inputs",
                shape=(total_timesteps, input_dim),
                dtype=np.float32,
                chunks=(min(100_000, max(1, total_timesteps)), input_dim),
            )
            write_offset = 0
            open_hdfs = {}
            try:
                for hdf_path, _seq_idx, _tag, start, end, _seq_len in selected:
                    hdf = open_hdfs.get(hdf_path)
                    if hdf is None:
                        hdf = h5py.File(hdf_path, "r")
                        open_hdfs[hdf_path] = hdf
                    length = end - start
                    inputs_out[write_offset : write_offset + length] = hdf["inputs"][start:end]
                    write_offset += length
            finally:
                for hdf in open_hdfs.values():
                    hdf.close()

            out_hdf.create_dataset("seqLengths", data=seq_lengths_out)
            out_hdf.create_dataset(
                "seqTags",
                data=np.asarray([item[2].encode("utf-8") for item in selected], dtype=object),
                dtype=h5py.special_dtype(vlen=bytes),
            )
            out_hdf.create_dataset("labels", shape=(0,), dtype=labels_dtype)
            for key, value in attrs.items():
                out_hdf.attrs[key] = value
            out_hdf.attrs["numSeqs"] = len(selected)
            out_hdf.attrs["numTimesteps"] = total_timesteps
            out_hdf.attrs["inputPattSize"] = input_dim
            out_hdf.attrs["numLabels"] = input_dim

        with open(self.out_report.get_path(), "wt", encoding="utf-8") as f:
            f.write(f"input_hdfs\t{len(self.hdf_files)}\n")
            f.write(f"selected_sequences\t{len(selected)}\n")
            f.write(f"selected_timesteps\t{total_timesteps}\n")
            f.write(f"input_dim\t{input_dim}\n")


class ClusterSequenceDecodingSummaryJob(Job):
    def __init__(self, rows):
        self.rows = rows
        self.out_report = self.output_path("cluster_sequence_decoding_summary.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _read_wer(wer_var) -> float:
        if hasattr(wer_var, "get"):
            return float(wer_var.get())
        with open(tk.uncached_path(wer_var), "rt") as f:
            text = f.read().strip()
        matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if not matches:
            raise ValueError(f"Could not parse WER from {wer_var}: {text!r}")
        return float(matches[0])

    @staticmethod
    def _get_sclite_dtl_path(wer_var) -> str:
        if not hasattr(wer_var, "get_path"):
            raise TypeError(f"Expected Sisyphus variable with get_path(), got {wer_var!r}")
        return os.path.join(os.path.dirname(wer_var.get_path()), "reports", "sclite.dtl")

    @staticmethod
    def _parse_sclite_error_breakdown(dtl_path: str):
        keys = {
            "Percent Substitution": "sub",
            "Percent Deletions": "del",
            "Percent Insertions": "ins",
        }
        result = {}
        pattern = re.compile(r"^(Percent (?:Substitution|Deletions|Insertions))\s*=\s*([0-9.]+)%\s*\(")
        with open(dtl_path, "rt", errors="ignore") as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    result[keys[match.group(1)]] = float(match.group(2))
        missing = [key for key in ("sub", "del", "ins") if key not in result]
        if missing:
            raise ValueError(f"Could not parse {missing} from {dtl_path}")
        return result

    def run(self):
        results = []
        for row in self.rows:
            breakdown = self._parse_sclite_error_breakdown(self._get_sclite_dtl_path(row["wer_file"]))
            results.append(
                {
                    "lm_scale": row["lm_scale"],
                    "wer": self._read_wer(row["wer_file"]),
                    **breakdown,
                }
            )
        results.sort(key=lambda item: item["wer"])
        with open(self.out_report.get_path(), "wt") as f:
            f.write("lm_scale\twer\tsub_percent\tdel_percent\tins_percent\n")
            for item in results:
                f.write(
                    f"{item['lm_scale']:g}\t"
                    f"{item['wer']:.4f}\t"
                    f"{item['sub']:.4f}\t"
                    f"{item['del']:.4f}\t"
                    f"{item['ins']:.4f}\n"
                )


class TuneAndCvDecodingSummaryJob(Job):
    def __init__(self, *, tune_rows, cv_rows):
        self.tune_rows = tune_rows
        self.cv_rows = cv_rows
        self.out_report = self.output_path("tune_and_cv_decoding_summary.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _row_to_result(row):
        breakdown = ClusterSequenceDecodingSummaryJob._parse_sclite_error_breakdown(
            ClusterSequenceDecodingSummaryJob._get_sclite_dtl_path(row["wer_file"])
        )
        return {
            "lm_scale": row["lm_scale"],
            "wer": ClusterSequenceDecodingSummaryJob._read_wer(row["wer_file"]),
            **breakdown,
        }

    def run(self):
        tune_results = [self._row_to_result(row) for row in self.tune_rows]
        cv_results = [self._row_to_result(row) for row in self.cv_rows]
        tune_results.sort(key=lambda item: item["wer"])
        cv_by_lm = {item["lm_scale"]: item for item in cv_results}
        best_tune = tune_results[0]
        if best_tune["lm_scale"] not in cv_by_lm:
            raise ValueError(f"Best tuning lm_scale {best_tune['lm_scale']} missing from CV rows")
        selected_cv = cv_by_lm[best_tune["lm_scale"]]
        with open(self.out_report.get_path(), "wt") as f:
            f.write("selected_by_tuning\n")
            f.write("dataset\tlm_scale\twer\tsub_percent\tdel_percent\tins_percent\n")
            for name, item in (("tune_train_0p5pct", best_tune), ("cv_1pct", selected_cv)):
                f.write(
                    f"{name}\t"
                    f"{item['lm_scale']:g}\t"
                    f"{item['wer']:.4f}\t"
                    f"{item['sub']:.4f}\t"
                    f"{item['del']:.4f}\t"
                    f"{item['ins']:.4f}\n"
                )
            f.write("\nall_tuning_results\n")
            f.write("lm_scale\twer\tsub_percent\tdel_percent\tins_percent\n")
            for item in tune_results:
                f.write(
                    f"{item['lm_scale']:g}\t"
                    f"{item['wer']:.4f}\t"
                    f"{item['sub']:.4f}\t"
                    f"{item['del']:.4f}\t"
                    f"{item['ins']:.4f}\n"
                )


class OggZipTextToPhonemeBlissSubsetJob(Job):
    def __init__(
        self,
        *,
        ogg_zip: tk.Path,
        segment_file: tk.Path,
        lexicon_file: tk.Path,
        silence_label: str = "[SILENCE]",
    ):
        self.ogg_zip = ogg_zip
        self.segment_file = segment_file
        self.lexicon_file = lexicon_file
        self.silence_label = silence_label
        self.out_corpus = self.output_path("phoneme_corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _open_text(path):
        path = tk.uncached_path(path)
        if path.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "rt", encoding="utf-8")

    def _read_first_pronunciation_lexicon(self):
        with self._open_text(self.lexicon_file) as f:
            root = ET.parse(f).getroot()
        mapping = {}
        for lemma in root.findall(".//lemma"):
            orths = [orth.text.strip() for orth in lemma.findall("orth") if orth.text and orth.text.strip()]
            phons = [phon.text.strip() for phon in lemma.findall("phon") if phon.text and phon.text.strip()]
            if not orths or not phons:
                continue
            pron = [phone.replace("#", "") for phone in phons[0].split()]
            pron = [phone for phone in pron if phone and phone != self.silence_label]
            for orth in orths:
                mapping.setdefault(orth, pron)
        return mapping

    def run(self):
        lexicon = self._read_first_pronunciation_lexicon()
        with open(tk.uncached_path(self.segment_file), "rt") as f:
            wanted_segments = {line.strip() for line in f if line.strip()}

        with zipfile.ZipFile(tk.uncached_path(self.ogg_zip), "r") as zf:
            text_members = [name for name in zf.namelist() if name.endswith(".txt")]
            if len(text_members) != 1:
                raise ValueError(f"Expected exactly one metadata txt in {self.ogg_zip}, found {text_members}")
            entries = ast.literal_eval(zf.read(text_members[0]).decode("utf-8"))

        bliss = corpus.Corpus()
        bliss.name = "train-other-960"
        recordings = {}
        for entry in entries:
            seq_name = entry["seq_name"]
            if seq_name not in wanted_segments:
                continue
            corpus_name, recording_name, segment_name = seq_name.split("/", 2)
            if corpus_name != bliss.name:
                raise ValueError(f"Unexpected corpus prefix {corpus_name!r} in seq {seq_name!r}")
            phonemes = []
            for word in entry["text"].split():
                try:
                    phonemes.extend(lexicon[word])
                except KeyError as exc:
                    raise KeyError(f"Word {word!r} from seq {seq_name!r} not found in {self.lexicon_file}") from exc
            rec = recordings.get(recording_name)
            if rec is None:
                rec = corpus.Recording()
                rec.name = recording_name
                rec.audio = entry.get("file", "")
                bliss.add_recording(rec)
                recordings[recording_name] = rec
            seg = corpus.Segment(
                start=0.0,
                end=float(entry["duration"]),
                orth=" ".join(phonemes),
            )
            seg.name = segment_name
            rec.add_segment(seg)

        found_segments = {seg.fullname() for seg in bliss.segments()}
        missing = wanted_segments - found_segments
        if missing:
            raise ValueError(f"Missing {len(missing)} requested segments in OggZip metadata, e.g. {sorted(missing)[:5]}")
        bliss.dump(self.out_corpus.get_path())


def _make_hdf_dataset(*, hdf_files, partition_epoch, seq_ordering, segment_file):
    if not isinstance(hdf_files, (list, tuple)):
        hdf_files = [hdf_files]
    opts = {
        "class": "HDFDataset",
        "files": list(hdf_files),
        "use_cache_manager": True,
        "partition_epoch": partition_epoch,
        "seq_ordering": seq_ordering,
    }
    if segment_file is not None:
        opts["seq_list_filter_file"] = segment_file
    return _ReturnnDataset(opts)


def _make_training_datasets(*, hdf_files, train_segments, cv_segments):
    train_dataset = _make_hdf_dataset(
        hdf_files=hdf_files,
        partition_epoch=10,
        seq_ordering="laplace:.100",
        segment_file=train_segments,
    )
    cv_dataset = _make_hdf_dataset(
        hdf_files=hdf_files,
        partition_epoch=1,
        seq_ordering="sorted",
        segment_file=cv_segments,
    )
    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=cv_dataset,
        datastreams={},
        prior=None,
    )


def _generative2_style_lr(num_epochs: int, init_lr: float, peak_lr: float):
    epoch_1 = int(num_epochs * 0.45)
    epoch_2 = num_epochs - 2 * epoch_1
    return (
        list(np.linspace(init_lr, peak_lr, epoch_1))
        + list(np.linspace(peak_lr, init_lr, epoch_1))
        + list(np.linspace(init_lr, 1e-6, epoch_2))
    )


def eow_phon_phmm_ls960_cluster_sequence_generative():
    prefix_name = (
        "example_setups/librispeech/phmm_standalone_2024/"
        "ls960_cluster_sequence_generative"
    )

    returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    train_segments = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/corpus/segments/"
        "ShuffleAndSplitSegmentsJob.RorARIMr0Hr0/output/train.segments"
    )
    cv_segments = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/corpus/segments/"
        "ShuffleAndSplitSegmentsJob.RorARIMr0Hr0/output/cv.segments"
    )
    train_ogg_zip = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/returnn/oggzip/"
        "BlissToOggZipJob.mNyFEqof29Q5/output/out.ogg.zip"
    )
    lm_table = tk.Path(
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_phoneme_cipher/phoneme_ngram_conv_lm_v2_context3_ep200_log_probs.pt"
    )
    sil_p30_lm_checkpoint = tk.Path(
        "/work/asr4/zyang/mini/work/i6_core/returnn/training/"
        "ReturnnTrainingJob.nCTqjn4nnAF2/output/models/epoch.400.pt"
    )
    context8_lm_v3_checkpoint = tk.Path(
        "/work/asr4/zyang/mini/work/i6_core/returnn/training/"
        "ReturnnTrainingJob.Yny90iRrDAzS/output/models/epoch.020.pt"
    )
    context8_lm_v3_config = {
        "vocab_size": 41,
        "embedding_dim": 128,
        "conv_channels": 512,
        "conv_kernel_sizes": (4, 5),
        "conv_dilations": (1, 1),
        "num_conv_layers": 2,
        "projection_dim": 512,
        "dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 40,
    }
    lexicon = tk.Path(
        "/work/asr4/zyang/corpora/librispeech/960/lexicon/"
        "phmm_no_eow_special_phonemes.lexicon.xml.gz"
    )
    phoneme_vocab = tk.Path(
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_oggzip_to_phoneme_audio_hdf/train_960_phoneme_vocab.txt"
    )

    k128_source_prefix = (
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca/segment_clustering/"
        "layer15_pca512_peak_prom0p02_full_train_k128_4m"
    )
    cluster_hdfs = [
        tk.Path(f"{k128_source_prefix}/cluster_labels_k128.{idx:03d}.hdf")
        for idx in range(10)
    ]
    compat_job = MakeSparseClusterHDFDatasetCompatibleJob(cluster_hdfs)
    compat_job.add_alias(prefix_name + "/k128_unmerged/make_hdf_returnn_compatible")
    for idx, compat_hdf in enumerate(compat_job.out_hdf_files):
        tk.register_output(
            prefix_name + f"/k128_unmerged/cluster_labels_k128.{idx:03d}.returnn_compat.hdf",
            compat_hdf,
        )

    cv_bliss_job = OggZipTextToBlissSubsetJob(
        ogg_zip=train_ogg_zip,
        segment_file=cv_segments,
    )
    cv_bliss_job.add_alias(prefix_name + "/train_cv_1pct_reference")
    tk.register_output(prefix_name + "/train_cv_1pct_reference/corpus.xml.gz", cv_bliss_job.out_corpus)
    phoneme_cv_bliss_job = OggZipTextToPhonemeBlissSubsetJob(
        ogg_zip=train_ogg_zip,
        segment_file=cv_segments,
        lexicon_file=lexicon,
    )
    phoneme_cv_bliss_job.add_alias(prefix_name + "/train_cv_1pct_phoneme_reference")
    tk.register_output(
        prefix_name + "/train_cv_1pct_phoneme_reference/corpus.xml.gz",
        phoneme_cv_bliss_job.out_corpus,
    )

    sil_p30_lm_dump_job = DumpNgramConvLmScoreTableJob(
        lm_checkpoint=sil_p30_lm_checkpoint,
        python_exe=returnn_exe,
        recipe_root=tk.Path("/u/zyang/setups/mini/recipe"),
        context_length=3,
        vocab_size=41,
        model_config_dict={
            "vocab_size": 41,
            "embedding_dim": 128,
            "conv_channels": 256,
            "conv_kernel_size": 3,
            "projection_dim": 256,
            "dropout": 0.3,
            "pad_token_id": 0,
            "bos_token_id": 40,
        },
        output_filename="phoneme_ngram_conv_lm_v2_no_eow_sil_p30_context3_ep400_log_probs.pt",
        mem_rqmt=8,
        time_rqmt=2,
    )
    sil_p30_lm_dump_job.add_alias(prefix_name + "/lm_tables/no_eow_sil_p30_context3_ep400")
    tk.register_output(
        prefix_name + "/lm_tables/no_eow_sil_p30_context3_ep400/log_probs.pt",
        sil_p30_lm_dump_job.out_scores,
    )
    tk.register_output(
        prefix_name + "/lm_tables/no_eow_sil_p30_context3_ep400/stats.txt",
        sil_p30_lm_dump_job.out_stats,
    )

    base_lm_config, lm_image = create_lm_image_for_lexicon(
        lexicon_file=lexicon,
        scale=1.0,
        output_prefix=prefix_name + "/no_eow_4gram_lm_image",
    )

    num_epochs = 300
    decode_epochs = [100, 200, 300]

    def run_training_and_search(
        *,
        run_name_suffix: str,
        network_module: str,
        model_config,
        hdf_files,
        batch_size: int = 7_000,
        gpu_mem: int = 48,
        search_batch_size: int = 5_000,
        search_mem: int = 32,
        lm_scales=(0.6, 0.4, 0.8, 1.0),
        num_epochs = 300,
        decode_epochs=[100,200,300],
        logfile_prefix: str,
        simple_argmax_decode: bool = False,
        learning_rates=None,
        extra_train_config=None,
        max_seqs=200,
        accum_grad=4,
        extract_search_hdf_subset: bool = False,
        tune_lm_on_train_subset: bool = False,
        tune_subset_ratio: float = 0.005,
    ):
        train_config = {
            "optimizer": {"class": "adamw", "epsilon": 1e-08, "weight_decay": 1e-2},
            "learning_rates": learning_rates
            if learning_rates is not None
            else _generative2_style_lr(num_epochs, init_lr=3e-5, peak_lr=3e-4),
            "batch_size": batch_size,
            "max_seqs": max_seqs,
            "num_workers_per_gpu": 0,
            "accum_grad_multiple_step": accum_grad,
            "gradient_clip_norm": 1.0,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": decode_epochs,
            },
        }
        if extra_train_config:
            train_config.update(copy.deepcopy(extra_train_config))
        datasets = _make_training_datasets(
            hdf_files=hdf_files,
            train_segments=train_segments,
            cv_segments=cv_segments,
        )
        training_name = prefix_name + "/" + run_name_suffix
        train_job = training(
            training_name=training_name,
            datasets=datasets,
            train_args={
                "network_module": network_module,
                "config": train_config,
                "net_args": {"model_config_dict": asdict(model_config)},
                "use_training_config_v2": True,
            },
            num_epochs=num_epochs,
            returnn_exe=returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
        )
        train_job.rqmt["gpu_mem"] = gpu_mem
        for checkpoint_epoch in decode_epochs:
            tk.register_output(
                training_name + f"/epoch.{checkpoint_epoch}.pt",
                train_job.out_checkpoints[checkpoint_epoch].path,
            )
        cv_search_hdf_files = hdf_files
        cv_search_segment_file = cv_segments
        if extract_search_hdf_subset:
            cv_search_subset_job = ExtractHDFSubsetBySegmentsJob(
                hdf_files=hdf_files,
                segment_file=cv_segments,
                output_filename="train_cv_1pct.hdf",
            )
            cv_search_subset_job.add_alias(training_name + "/search_hdf_subset/train_cv_1pct")
            tk.register_output(training_name + "/search_hdf_subset/train_cv_1pct.hdf", cv_search_subset_job.out_hdf)
            tk.register_output(
                training_name + "/search_hdf_subset/train_cv_1pct_stats.txt",
                cv_search_subset_job.out_report,
            )
            cv_search_hdf_files = cv_search_subset_job.out_hdf
            cv_search_segment_file = None
        cv_search_dataset = _make_hdf_dataset(
            hdf_files=cv_search_hdf_files,
            partition_epoch=1,
            seq_ordering="sorted",
            segment_file=cv_search_segment_file,
        )

        tune_search_dataset = None
        tune_bliss = None
        tune_phoneme_bliss = None
        if tune_lm_on_train_subset:
            tune_segments_job = SelectSegmentSubsetJob(
                segment_file=train_segments,
                ratio=tune_subset_ratio,
                random_seed=1,
            )
            tune_segments_job.add_alias(training_name + "/train_tune_0p5pct_segments")
            tk.register_output(training_name + "/train_tune_0p5pct_segments/segments", tune_segments_job.out_segments)
            tune_bliss_job = OggZipTextToBlissSubsetJob(
                ogg_zip=train_ogg_zip,
                segment_file=tune_segments_job.out_segments,
            )
            tune_bliss_job.add_alias(training_name + "/train_tune_0p5pct_reference")
            tk.register_output(training_name + "/train_tune_0p5pct_reference/corpus.xml.gz", tune_bliss_job.out_corpus)
            tune_phoneme_bliss_job = OggZipTextToPhonemeBlissSubsetJob(
                ogg_zip=train_ogg_zip,
                segment_file=tune_segments_job.out_segments,
                lexicon_file=lexicon,
            )
            tune_phoneme_bliss_job.add_alias(training_name + "/train_tune_0p5pct_phoneme_reference")
            tk.register_output(
                training_name + "/train_tune_0p5pct_phoneme_reference/corpus.xml.gz",
                tune_phoneme_bliss_job.out_corpus,
            )
            tune_search_subset_job = ExtractHDFSubsetBySegmentsJob(
                hdf_files=hdf_files,
                segment_file=tune_segments_job.out_segments,
                output_filename="train_tune_0p5pct.hdf",
            )
            tune_search_subset_job.add_alias(training_name + "/search_hdf_subset/train_tune_0p5pct")
            tk.register_output(training_name + "/search_hdf_subset/train_tune_0p5pct.hdf", tune_search_subset_job.out_hdf)
            tk.register_output(
                training_name + "/search_hdf_subset/train_tune_0p5pct_stats.txt",
                tune_search_subset_job.out_report,
            )
            tune_search_dataset = _make_hdf_dataset(
                hdf_files=tune_search_subset_job.out_hdf,
                partition_epoch=1,
                seq_ordering="sorted",
                segment_file=None,
            )
            tune_bliss = tune_bliss_job.out_corpus
            tune_phoneme_bliss = tune_phoneme_bliss_job.out_corpus

        def run_rasr_search(*, asr_model, checkpoint_epoch, lm_scale, dataset_name, dataset, reference, name_part):
            lm_config = base_lm_config._copy()
            lm_config.scale = lm_scale
            lm_scale_name = _num_str(lm_scale) if isinstance(lm_scale, (int, float)) else "best"
            if name_part == "cv" and not tune_lm_on_train_subset:
                logfile_suffix = f"{logfile_prefix}_ep{checkpoint_epoch}_lm{lm_scale:g}"
            else:
                logfile_suffix = f"{logfile_prefix}_{name_part}_ep{checkpoint_epoch}_lm{lm_scale_name}"
            recog_config = build_librasr_phone_table_recognition_config(
                lexicon_path=lexicon,
                lm_config=lm_config,
                collapse_repeated_labels=True,
                score_threshold=18.0,
                intermediate_score_threshold=18.0,
                logfile_suffix=logfile_suffix,
            )
            decoder_config = DecoderConfig(
                rasr_config_file=recog_config,
                lexicon=lexicon,
                data_key="data",
                lm_image_file=lm_image,
            )
            search_name = training_name + f"/rasr_{name_part}_decoding/ep_{checkpoint_epoch}/lm{lm_scale_name}"
            _search_jobs, wers = search(
                search_name,
                forward_config={
                    "batch_size": search_batch_size,
                    "max_seqs": 64,
                    "num_workers_per_gpu": 0,
                },
                asr_model=asr_model,
                decoder_module="phmm.decoder.rasr_trainable_table_decoding_v1",
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={dataset_name: (dataset, reference)},
                returnn_exe=returnn_exe,
                returnn_root=MINI_RETURNN_ROOT,
                mem_rqmt=search_mem,
                use_gpu=False,
            )
            return {"lm_scale": lm_scale, "wer_file": wers[search_name + "/" + dataset_name]}

        def run_argmax_search(*, asr_model, checkpoint_epoch, dataset_name, dataset, reference, name_part):
            argmax_decoder_config = ArgmaxDecoderConfig(
                phoneme_vocab=phoneme_vocab,
                silence_label="[SILENCE]",
                data_key="data",
            )
            search_name = training_name + f"/argmax_phoneme_{name_part}_decoding/ep_{checkpoint_epoch}"
            search(
                search_name,
                forward_config={
                    "batch_size": search_batch_size,
                    "max_seqs": 64,
                    "num_workers_per_gpu": 0,
                },
                asr_model=asr_model,
                decoder_module="phmm.decoder.phoneme_argmax_decoder_v1",
                decoder_args={"config": asdict(argmax_decoder_config)},
                test_dataset_tuples={dataset_name: (dataset, reference)},
                returnn_exe=returnn_exe,
                returnn_root=MINI_RETURNN_ROOT,
                mem_rqmt=search_mem,
                use_gpu=False,
            )

        for checkpoint_epoch in decode_epochs:
            asr_model = ASRModel(
                checkpoint=train_job.out_checkpoints[checkpoint_epoch],
                net_args={"model_config_dict": asdict(model_config)},
                network_module=network_module,
                prior_file=None,
                prior_files=None,
                prefix_name=training_name,
            )
            cv_summary_rows = []
            tune_summary_rows = []
            if tune_lm_on_train_subset:
                tune_parameters = []
                tune_values = []
                for lm_scale in lm_scales:
                    tune_parameters.append((lm_scale,))
                    tune_summary_rows.append(
                        run_rasr_search(
                            asr_model=asr_model,
                            checkpoint_epoch=checkpoint_epoch,
                            lm_scale=lm_scale,
                            dataset_name="train_tune_0p5pct",
                            dataset=tune_search_dataset,
                            reference=tune_bliss,
                            name_part="train_tune_0p5pct",
                        )
                    )
                    tune_values.append(tune_summary_rows[-1]["wer_file"])
                pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                    parameters=tune_parameters,
                    values=tune_values,
                    mode="minimize",
                )
                pick_optimal_params_job.add_alias(
                    training_name + f"/rasr_train_tune_0p5pct_decoding/ep_{checkpoint_epoch}/pick_best_lm"
                )
                cv_summary_rows.append(
                    run_rasr_search(
                        asr_model=asr_model,
                        checkpoint_epoch=checkpoint_epoch,
                        lm_scale=pick_optimal_params_job.out_optimal_parameters[0],
                        dataset_name="train_cv_1pct",
                        dataset=cv_search_dataset,
                        reference=cv_bliss_job.out_corpus,
                        name_part="cv_best_tuned",
                    )
                )
            else:
                for lm_scale in lm_scales:
                    cv_summary_rows.append(
                        run_rasr_search(
                            asr_model=asr_model,
                            checkpoint_epoch=checkpoint_epoch,
                            lm_scale=lm_scale,
                            dataset_name="train_cv_1pct",
                            dataset=cv_search_dataset,
                            reference=cv_bliss_job.out_corpus,
                            name_part="cv",
                        )
                    )

            if simple_argmax_decode:
                if tune_lm_on_train_subset:
                    run_argmax_search(
                        asr_model=asr_model,
                        checkpoint_epoch=checkpoint_epoch,
                        dataset_name="train_tune_0p5pct",
                        dataset=tune_search_dataset,
                        reference=tune_phoneme_bliss,
                        name_part="train_tune_0p5pct",
                    )
                run_argmax_search(
                    asr_model=asr_model,
                    checkpoint_epoch=checkpoint_epoch,
                    dataset_name="train_cv_1pct",
                    dataset=cv_search_dataset,
                    reference=phoneme_cv_bliss_job.out_corpus,
                    name_part="cv",
                )

            if tune_lm_on_train_subset:
                tune_summary_job = ClusterSequenceDecodingSummaryJob(tune_summary_rows)
                tune_summary_job.add_alias(training_name + f"/rasr_train_tune_0p5pct_decoding/ep_{checkpoint_epoch}/summary")
                tk.register_output(
                    training_name + f"/rasr_train_tune_0p5pct_decoding/ep_{checkpoint_epoch}/summary.txt",
                    tune_summary_job.out_report,
                )
            else:
                summary_job = ClusterSequenceDecodingSummaryJob(cv_summary_rows)
                summary_job.add_alias(training_name + f"/rasr_cv_decoding/ep_{checkpoint_epoch}/summary")
                tk.register_output(
                    training_name + f"/rasr_cv_decoding/ep_{checkpoint_epoch}/summary.txt",
                    summary_job.out_report,
                )
        return train_job

    merged_cluster_hdf = tk.Path(
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca_merge_cluster_ids/k128_peak_prom0p02/"
        "cluster_labels_k128_merged.hdf"
    )

    # run_training_and_search(
    #     run_name_suffix="cluster_sequence_one_to_one_generative.k128_merged_2conv_h512_kernel4_stride1_b10000_sil_lm_ep300",
    #     network_module="phmm.cluster_sequence_one_to_one_generative",
    #     model_config=OneToOneModelConfig(
    #         input_vocab_size=128,
    #         label_target_size=40,
    #         lm_table_path=sil_p30_lm_dump_job.out_scores,
    #         hidden_size=512,
    #         conv_kernel_size=4,
    #         conv_stride=1,
    #         conv_dilation=1,
    #         dropout=0.1,
    #         lm_vocab_size=41,
    #         lm_context_length=3,
    #         beam_size=200,
    #         lm_scale=0.6,
    #         am_scale=1.0,
    #         sampling_type="batch",
    #         sampling_ratio=0.2,
    #         share_samples=False,
    #         ratio_corrector=1.0,
    #     ),
    #     hdf_files=merged_cluster_hdf,
    #     batch_size=10_000,
    #     gpu_mem=48,
    #     search_batch_size=5_000,
    #     search_mem=32,
    #     num_epochs=300,
    #     decode_epochs=[200, 300],
    #     logfile_prefix="cluster_sequence_one_to_one_generative_k128_merged_sil_lm",
    #     simple_argmax_decode=True,
    # )
    #
    # b7000_train_job = run_training_and_search(
    #     run_name_suffix="cluster_sequence_generative.k128_unmerged_kernel4_stride1_b7000_ep300",
    #     network_module="phmm.cluster_sequence_phmm_generative",
    #     model_config=ModelConfig(
    #         input_vocab_size=128,
    #         label_target_size=40,
    #         lm_table_path=lm_table,
    #         hidden_size=256,
    #         conv_kernel_size=4,
    #         conv_stride=1,
    #         conv_dilation=1,
    #         dropout=0.1,
    #         lm_vocab_size=41,
    #         lm_context_length=3,
    #         beam_size=200,
    #         lm_scale=0.6,
    #         am_scale=1.0,
    #         sampling_type="batch",
    #         sampling_ratio=0.2,
    #         share_samples=False,
    #         ratio_corrector=1.0,
    #     ),
    #     hdf_files=compat_job.out_hdf_files,
    #     batch_size=7_000,
    #     gpu_mem=48,
    #     search_batch_size=5_000,
    #     search_mem=32,
    #     num_epochs=300,
    #     decode_epochs=[100, 200, 300],
    #     logfile_prefix="cluster_sequence_generative_k128",
    # )
    # run_training_and_search(
    #     run_name_suffix="cluster_sequence_generative.k128_unmerged_kernel4_stride1_b7000_ep300_continue_lr5e-5_b40000_sr0p5_share_ep100",
    #     network_module="phmm.cluster_sequence_phmm_generative",
    #     model_config=ModelConfig(
    #         input_vocab_size=128,
    #         label_target_size=40,
    #         lm_table_path=lm_table,
    #         hidden_size=256,
    #         conv_kernel_size=4,
    #         conv_stride=1,
    #         conv_dilation=1,
    #         dropout=0.1,
    #         lm_vocab_size=41,
    #         lm_context_length=3,
    #         beam_size=200,
    #         lm_scale=0.6,
    #         am_scale=1.0,
    #         sampling_type="batch",
    #         sampling_ratio=0.5,
    #         share_samples=True,
    #         ratio_corrector=1.0,
    #     ),
    #     hdf_files=compat_job.out_hdf_files,
    #     batch_size=40_000,
    #     gpu_mem=48,
    #     search_batch_size=5_000,
    #     search_mem=32,
    #     num_epochs=100,
    #     decode_epochs=[100],
    #     logfile_prefix="cluster_sequence_generative_k128_continue_lr5e-5_b40000",
    #     simple_argmax_decode=True,
    #     learning_rates=[5e-5] * 100,
    #     extra_train_config={
    #         "preload_from_files": {
    #             "previous": {
    #                 "filename": b7000_train_job.out_checkpoints[300].path,
    #                 "init_for_train": True,
    #                 "checkpoint_key": "model",
    #             },
    #         },
    #         "cleanup_old_models": {
    #             "keep_last_n": 4,
    #             "keep_best_n": 4,
    #             "keep": [100],
    #         },
    #     },
    # )
    # # with shared samples:
    # run_training_and_search(
    #     run_name_suffix="cluster_sequence_generative.k128_unmerged_kernel4_stride1_b80000_share_sample_ep300",
    #     network_module="phmm.cluster_sequence_phmm_generative",
    #     model_config=ModelConfig(
    #         input_vocab_size=128,
    #         label_target_size=40,
    #         lm_table_path=lm_table,
    #         hidden_size=256,
    #         conv_kernel_size=4,
    #         conv_stride=1,
    #         conv_dilation=1,
    #         dropout=0.1,
    #         lm_vocab_size=41,
    #         lm_context_length=3,
    #         beam_size=200,
    #         lm_scale=0.6,
    #         am_scale=1.0,
    #         sampling_type="batch",
    #         sampling_ratio=0.8,
    #         share_samples=True,
    #         ratio_corrector=1.0,
    #     ),
    #     hdf_files=compat_job.out_hdf_files,
    #     batch_size=80_000,
    #     gpu_mem=48,
    #     search_batch_size=5_000,
    #     search_mem=32,
    #     num_epochs=300,
    #     decode_epochs=[200, 300],
    #     logfile_prefix="cluster_sequence_generative_k128",
    # )
    #
    # segment_representation_prefix = (
    #     "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
    #     "ls960_wav2vec2_large_layer15_pca/segment_representations/"
    #     "layer15_pca512_peak_prom0p02_full_train"
    # )
    # segment_representation_hdfs = [
    #     tk.Path(f"{segment_representation_prefix}/segment_representations.{idx:03d}.hdf")
    #     for idx in range(10)
    # ]
    # run_training_and_search(
    #     run_name_suffix="vector_sequence_generative.pca512_segments_kernel4_stride1_b7000_ep300",
    #     network_module="phmm.vector_sequence_phmm_generative",
    #     model_config=VectorModelConfig(
    #         input_dim=512,
    #         label_target_size=40,
    #         lm_table_path=lm_table,
    #         conv_kernel_size=4,
    #         conv_stride=1,
    #         conv_dilation=1,
    #         dropout=0.1,
    #         input_time_batch_norm=True,
    #         input_residual_linear=True,
    #         lm_vocab_size=41,
    #         lm_context_length=3,
    #         beam_size=200,
    #         lm_scale=0.6,
    #         am_scale=1.0,
    #         sampling_type="batch",
    #         sampling_ratio=0.2,
    #         share_samples=False,
    #         ratio_corrector=1.0,
    #     ),
    #     hdf_files=segment_representation_hdfs,
    #     batch_size=7_000,
    #     gpu_mem=48,
    #     search_batch_size=5_000,
    #     search_mem=32,
    #     num_epochs=300,
    #     decode_epochs=[100, 200, 300],
    #     logfile_prefix="vector_sequence_generative_pca512",
    # )

    for training_lm in [0.3,0.6,1.0]:
        run_training_and_search(
            run_name_suffix=f"cluster_sequence_generative.k128_unmerged_kernel5_stride1_lm{training_lm}b80000_share_sample_context8_lmv3_ep300",
            network_module="phmm.cluster_sequence_phmm_generative",
            model_config=ModelConfig(
                input_vocab_size=128,
                label_target_size=40,
                lm_checkpoint_path=context8_lm_v3_checkpoint,
                lm_model_config_dict=context8_lm_v3_config,
                hidden_size=512,
                conv_kernel_size=5,
                conv_stride=1,
                conv_dilation=1,
                dropout=0.1,
                lm_vocab_size=41,
                lm_context_length=8,
                beam_size=200,
                lm_scale=training_lm,
                am_scale=1.0,
                sampling_type="batch",
                sampling_ratio=0.5,
                share_samples=True,
                ratio_corrector=1.0,
            ),
            hdf_files=compat_job.out_hdf_files,
            batch_size=80_000,
            gpu_mem=48,
            search_batch_size=5_000,
            search_mem=32,
            num_epochs=300,
            max_seqs=1000,
            accum_grad=2,
            decode_epochs=[100, 200, 300],
            logfile_prefix="cluster_sequence_generative_k128_context8_lmv3",
            simple_argmax_decode=True,
        )

    layer15_pca512_frame_hdf = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/feature_pca_jobs/"
        "ApplyPcaToFeatureHDFJob.JfBOA9zHcXBA/output/features_layer15_pca512.hdf"
    )
    run_training_and_search(
        run_name_suffix="vector_sequence_generative.layer15_pca512_frames_kernel4_stride2_lm08_lp05_b180000_share_sample_context8_lmv3_ep300",
        network_module="phmm.vector_sequence_phmm_generative",
        model_config=VectorModelConfig(
            input_dim=512,
            label_target_size=40,
            lm_checkpoint_path=context8_lm_v3_checkpoint,
            lm_model_config_dict=context8_lm_v3_config,
            conv_kernel_size=4,
            conv_stride=2,
            conv_dilation=1,
            dropout=0.1,
            input_time_batch_norm=True,
            input_residual_linear=True,
            lm_vocab_size=41,
            lm_context_length=8,
            beam_size=200,
            lm_scale=0.8,
            am_scale=1.0,
            sampling_type="batch",
            sampling_ratio=0.5,
            share_samples=True,
            ratio_corrector=1.0,
        ),
        hdf_files=layer15_pca512_frame_hdf,
        batch_size=180_000,
        gpu_mem=48,
        search_batch_size=5_000,
        search_mem=32,
        lm_scales=(0.1, 0.05, 0.2, 0.4, 0.6),
        num_epochs=300,
        max_seqs=1000,
        accum_grad=2,
        extra_train_config={"loop_penalty": 1.05},
        decode_epochs=[100, 200, 300],
        logfile_prefix="vector_sequence_generative_layer15_pca512_frames_context8_lmv3",
        simple_argmax_decode=True,
        extract_search_hdf_subset=True,
        tune_lm_on_train_subset=True,
        tune_subset_ratio=0.005,
    )
    run_training_and_search(
        run_name_suffix="vector_sequence_generative.layer15_pca512_frames_kernel4_stride2_lm08_lp05_b160000_share_sample_4gram_table_ep300",
        network_module="phmm.vector_sequence_phmm_generative",
        model_config=VectorModelConfig(
            input_dim=512,
            label_target_size=40,
            lm_table_path=lm_table,
            conv_kernel_size=4,
            conv_stride=2,
            conv_dilation=1,
            dropout=0.1,
            input_time_batch_norm=True,
            input_residual_linear=True,
            lm_vocab_size=41,
            lm_context_length=3,
            beam_size=200,
            lm_scale=0.8,
            am_scale=1.0,
            sampling_type="batch",
            sampling_ratio=0.5,
            share_samples=True,
            ratio_corrector=1.0,
        ),
        hdf_files=layer15_pca512_frame_hdf,
        batch_size=160_000,
        gpu_mem=48,
        search_batch_size=5_000,
        search_mem=32,
        lm_scales=(0.1, 0.05, 0.2, 0.4, 0.6),
        num_epochs=300,
        max_seqs=1000,
        accum_grad=2,
        extra_train_config={"loop_penalty": 1.05},
        decode_epochs=[100, 200, 300],
        logfile_prefix="vector_sequence_generative_layer15_pca512_frames_4gram_table",
        simple_argmax_decode=True,
        extract_search_hdf_subset=True,
        tune_lm_on_train_subset=True,
        tune_subset_ratio=0.005,
    )
    run_training_and_search(
        run_name_suffix="vector_sequence_generative.layer15_pca512_frames_kernel4_stride2_lm04_lp00_b160000_share_sample_4gram_table_ep300",
        network_module="phmm.vector_sequence_phmm_generative",
        model_config=VectorModelConfig(
            input_dim=512,
            label_target_size=40,
            lm_table_path=lm_table,
            conv_kernel_size=4,
            conv_stride=2,
            conv_dilation=1,
            dropout=0.1,
            input_time_batch_norm=True,
            input_residual_linear=True,
            lm_vocab_size=41,
            lm_context_length=3,
            beam_size=200,
            lm_scale=0.4,
            am_scale=1.0,
            sampling_type="batch",
            sampling_ratio=0.5,
            share_samples=True,
            ratio_corrector=1.0,
        ),
        hdf_files=layer15_pca512_frame_hdf,
        batch_size=160_000,
        gpu_mem=48,
        search_batch_size=5_000,
        search_mem=32,
        lm_scales=(0.1, 0.05, 0.2, 0.4, 0.6),
        num_epochs=300,
        max_seqs=1000,
        accum_grad=2,
        extra_train_config={"loop_penalty": 1.00},
        decode_epochs=[100, 200, 300],
        logfile_prefix="vector_sequence_generative_layer15_pca512_frames_4gram_table_lm04",
        simple_argmax_decode=True,
        extract_search_hdf_subset=True,
        tune_lm_on_train_subset=True,
        tune_subset_ratio=0.005,
    )

    run_training_and_search(
        run_name_suffix="vector_sequence_generative.layer15_pca512_frames_kernel4_stride2_lm04_lp00_b180000_share_sample_context8_lmv3_ep300",
        network_module="phmm.vector_sequence_phmm_generative",
        model_config=VectorModelConfig(
            input_dim=512,
            label_target_size=40,
            lm_checkpoint_path=context8_lm_v3_checkpoint,
            lm_model_config_dict=context8_lm_v3_config,
            conv_kernel_size=4,
            conv_stride=2,
            conv_dilation=1,
            dropout=0.1,
            input_time_batch_norm=True,
            input_residual_linear=True,
            lm_vocab_size=41,
            lm_context_length=8,
            beam_size=200,
            lm_scale=0.4,
            am_scale=1.0,
            sampling_type="batch",
            sampling_ratio=0.5,
            share_samples=True,
            ratio_corrector=1.0,
        ),
        hdf_files=layer15_pca512_frame_hdf,
        batch_size=180_000,
        gpu_mem=48,
        search_batch_size=5_000,
        search_mem=32,
        lm_scales=(0.1, 0.05, 0.2, 0.4, 0.6),
        num_epochs=300,
        max_seqs=1000,
        accum_grad=2,
        extra_train_config={"loop_penalty": 1.00},
        decode_epochs=[100, 200, 300],
        logfile_prefix="vector_sequence_generative_layer15_pca512_frames_context8_lmv3",
        simple_argmax_decode=True,
        extract_search_hdf_subset=True,
        tune_lm_on_train_subset=True,
        tune_subset_ratio=0.005,
    )

    run_training_and_search(
        run_name_suffix="vector_sequence_generative.layer15_pca512_frames_kernel9_stride3_lm08_lp2_b160000_share_sample_4gram_table_ep300",
        network_module="phmm.vector_sequence_phmm_generative",
        model_config=VectorModelConfig(
            input_dim=512,
            label_target_size=40,
            lm_table_path=lm_table,
            conv_kernel_size=9,
            conv_stride=3,
            conv_dilation=1,
            dropout=0.1,
            input_time_batch_norm=True,
            input_residual_linear=True,
            lm_vocab_size=41,
            lm_context_length=3,
            beam_size=200,
            lm_scale=0.8,
            am_scale=1.0,
            sampling_type="batch",
            sampling_ratio=0.5,
            share_samples=True,
            ratio_corrector=1.0,
        ),
        hdf_files=layer15_pca512_frame_hdf,
        batch_size=160_000,
        gpu_mem=48,
        search_batch_size=5_000,
        search_mem=32,
        lm_scales=(0.1, 0.05, 0.2, 0.4, 0.6),
        num_epochs=300,
        max_seqs=1000,
        accum_grad=2,
        extra_train_config={"loop_penalty": 1.2},
        decode_epochs=[100, 200, 300],
        logfile_prefix="vector_sequence_generative_layer15_pca512_frames_4gram_table",
        simple_argmax_decode=True,
        extract_search_hdf_subset=True,
        tune_lm_on_train_subset=True,
        tune_subset_ratio=0.005,
    )



py = eow_phon_phmm_ls960_cluster_sequence_generative
