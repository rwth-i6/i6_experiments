import ast
import glob
import os
import re
import shutil
import zipfile
from dataclasses import asdict

import numpy as np
from sisyphus import Job, Task, tk

from i6_core.lib import corpus
from i6_experiments.common.setups.returnn.datasets import MetaDataset
from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_lm import create_lm_image_for_lexicon
from ...phmm_pipeline import ASRModel, search
from ...phmm_rasr import CreateLibrasrVenvJob, build_librasr_phone_table_recognition_config
from ...pytorch_networks.phmm.decoder.rasr_table_decoding_v1 import DecoderConfig as RasrTableDecoderConfig
from ...pytorch_networks.phmm.rasr_table_model_cfg import ModelConfig


def _num_str(value: float) -> str:
    return ("%g" % value).replace(".", "p")


class OggZipTextToBlissSubsetJob(Job):
    def __init__(self, *, ogg_zip: tk.Path, segment_file: tk.Path):
        self.ogg_zip = ogg_zip
        self.segment_file = segment_file
        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
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
                orth=entry["text"],
            )
            seg.name = segment_name
            rec.add_segment(seg)

        found_segments = {seg.fullname() for seg in bliss.segments()}
        missing = wanted_segments - found_segments
        if missing:
            raise ValueError(f"Missing {len(missing)} requested segments in OggZip metadata, e.g. {sorted(missing)[:5]}")
        bliss.dump(self.out_corpus.get_path())


class SelectSegmentSubsetJob(Job):
    def __init__(self, *, segment_file: tk.Path, ratio: float = 0.2, random_seed: int = 1):
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.segment_file = segment_file
        self.ratio = ratio
        self.random_seed = random_seed
        self.out_segments = self.output_path("segments")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(tk.uncached_path(self.segment_file), "rt") as f:
            segments = [line.strip() for line in f if line.strip()]
        rng = np.random.default_rng(self.random_seed)
        num_selected = max(1, int(round(len(segments) * self.ratio)))
        selected_indices = np.sort(rng.choice(len(segments), size=num_selected, replace=False))
        with open(self.out_segments.get_path(), "wt") as f:
            for idx in selected_indices:
                f.write(segments[int(idx)] + "\n")


class MakeSparseHDFDatasetCompatibleJob(Job):
    """
    RETURNN HDFDataset expects a dummy target-length column for legacy sparse HDFs.
    The cluster-id HDFs only store input lengths, so add a second zero column.
    """

    def __init__(self, hdf_files):
        self.hdf_files = hdf_files
        self.out_hdf_files = [
            self.output_path(os.path.basename(str(hdf_file)).replace(".hdf", ".returnn_compat.hdf"))
            for hdf_file in hdf_files
        ]

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import h5py

        for src, dst in zip(self.hdf_files, self.out_hdf_files):
            shutil.copyfile(tk.uncached_path(src), dst.get_path())
            with h5py.File(dst.get_path(), "r+") as f:
                seq_lengths = f["seqLengths"][...]
                if seq_lengths.ndim != 2:
                    raise ValueError(f"{src}: expected 2-D seqLengths, got {seq_lengths.shape}")
                if seq_lengths.shape[1] == 1:
                    fixed = np.concatenate([seq_lengths, np.zeros_like(seq_lengths)], axis=1)
                    del f["seqLengths"]
                    f.create_dataset("seqLengths", data=fixed, dtype=seq_lengths.dtype)
                elif seq_lengths.shape[1] != 2:
                    raise ValueError(f"{src}: expected 1 or 2 seqLengths columns, got {seq_lengths.shape}")


class DumpAlignmentFrameLengthsHDFJob(Job):
    """Dump one 10ms frame-length scalar per utterance from GMM alignment HDFs."""

    def __init__(self, alignment_hdfs, *, output_filename: str = "alignment_frame_lengths.hdf"):
        self.alignment_hdfs = alignment_hdfs if isinstance(alignment_hdfs, (list, tuple)) else [alignment_hdfs]
        self.out_hdf = self.output_path(output_filename)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import h5py

        seq_tags = []
        frame_lengths = []
        for hdf_path in self.alignment_hdfs:
            with h5py.File(tk.uncached_path(hdf_path), "r") as hdf:
                seq_tags.extend(hdf["seqTags"][:])
                frame_lengths.extend(int(length) for length in hdf["seqLengths"][:, 0])

        inputs = np.asarray(frame_lengths, dtype="int32")[:, None]
        seq_lengths = np.ones((len(frame_lengths), 2), dtype="int32")
        seq_lengths[:, 1] = 0
        with h5py.File(self.out_hdf.get_path(), "w") as out_hdf:
            out_hdf.create_dataset("inputs", data=inputs)
            out_hdf.create_dataset("seqLengths", data=seq_lengths)
            dt = h5py.special_dtype(vlen=bytes)
            out_hdf.create_dataset("seqTags", data=np.asarray(seq_tags, dtype=object), dtype=dt)
            out_hdf.create_dataset("labels", data=np.asarray([], dtype="S5"))
            out_hdf.attrs["numTimesteps"] = int(inputs.shape[0])
            out_hdf.attrs["inputPattSize"] = 1
            out_hdf.attrs["numDims"] = 1
            out_hdf.attrs["numLabels"] = 1
            out_hdf.attrs["numSeqs"] = int(inputs.shape[0])


class TableDecodingTuneSummaryJob(Job):
    def __init__(self, rows):
        self.rows = rows
        self.out_report = self.output_path("tuning_summary.txt")

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
            raise TypeError(f"Expected a Sisyphus Variable with get_path(), got {wer_var!r}")
        return os.path.join(os.path.dirname(wer_var.get_path()), "reports", "sclite.dtl")

    @staticmethod
    def _parse_sclite_error_breakdown(dtl_path: str):
        keys = {
            "Percent Substitution": "sub",
            "Percent Deletions": "del",
            "Percent Insertions": "ins",
        }
        result = {}
        pattern = re.compile(r"^(Percent (?:Substitution|Deletions|Insertions))\s*=\s*([0-9.]+)%\s*\(\s*(\d+)\)")
        with open(dtl_path, "rt", errors="ignore") as f:
            for line in f:
                match = pattern.match(line)
                if not match:
                    continue
                prefix = keys[match.group(1)]
                result[f"{prefix}_percent"] = float(match.group(2))
        missing = [
            key
            for key in (
                "sub_percent",
                "del_percent",
                "ins_percent",
            )
            if key not in result
        ]
        if missing:
            raise ValueError(f"Could not parse {missing} from {dtl_path}")
        return result

    def run(self):
        results = []
        for row in self.rows:
            wer = self._read_wer(row["wer_file"])
            breakdown = self._parse_sclite_error_breakdown(self._get_sclite_dtl_path(row["wer_file"]))
            results.append({
                "lm_scale": row["lm_scale"],
                "score_threshold": row["score_threshold"],
                "collapse_repeated_labels": row["collapse_repeated_labels"],
                "wer": wer,
                **breakdown,
            })
        results.sort(key=lambda item: item["wer"])
        with open(self.out_report.get_path(), "wt") as f:
            f.write("lm_scale\tscore_threshold\tif_collapse\twer\tsub_percent\tdel_percent\tins_percent\n")
            for item in results:
                f.write(
                    f"{item['lm_scale']:g}\t"
                    f"{item['score_threshold']:g}\t"
                    f"{item['collapse_repeated_labels']}\t"
                    f"{item['wer']:.4f}\t"
                    f"{item['sub_percent']:.4f}\t"
                    f"{item['del_percent']:.4f}\t"
                    f"{item['ins_percent']:.4f}\n"
                )


def _get_cluster_hdfs(num_clusters: int):
    alias_pattern = (
        "alias/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca/segment_clustering/"
        f"layer15_pca512_peak_prom0p02_full_train_k{num_clusters}_4m/assign_k{num_clusters}/part_*"
    )
    hdfs = []
    for part_alias in glob.glob(alias_pattern):
        job_dir = os.path.realpath(part_alias)
        hdfs.extend(glob.glob(os.path.join(job_dir, "output", f"cluster_labels_k{num_clusters}.*.hdf")))
    hdfs = sorted(set(hdfs), key=lambda path: os.path.basename(path))
    if len(hdfs) != 10:
        raise ValueError(f"Expected 10 k{num_clusters} cluster HDF files, found {len(hdfs)}: {hdfs}")
    return [tk.Path(path) for path in hdfs]


def _build_cluster_dataset(prefix_name: str, num_clusters: int, cv_segments: tk.Path):
    cluster_hdfs = _get_cluster_hdfs(num_clusters)
    compat_hdf_job = MakeSparseHDFDatasetCompatibleJob(cluster_hdfs)
    compat_hdf_job.add_alias(prefix_name + f"/make_k{num_clusters}_hdf_returnn_compatible")
    return HDFDataset(
        files=compat_hdf_job.out_hdf_files,
        partition_epoch=1,
        segment_file=cv_segments,
        seq_ordering="sorted_reverse",
    )


def _build_frame_cluster_dataset(
    *,
    prefix_name: str,
    num_clusters: int,
    subset_segments: tk.Path,
    segment_start_hdf: tk.Path,
    frame_length_hdf: tk.Path,
):
    cluster_hdfs = _get_cluster_hdfs(num_clusters)
    cluster_compat_job = MakeSparseHDFDatasetCompatibleJob(cluster_hdfs)
    cluster_compat_job.add_alias(prefix_name + f"/frame_mode_make_k{num_clusters}_cluster_hdf_returnn_compatible")
    starts_compat_job = MakeSparseHDFDatasetCompatibleJob([segment_start_hdf])
    starts_compat_job.add_alias(prefix_name + "/frame_mode_make_segment_starts_hdf_returnn_compatible")

    cluster_dataset = HDFDataset(
        files=cluster_compat_job.out_hdf_files,
        partition_epoch=1,
        segment_file=subset_segments,
        seq_ordering="sorted_reverse",
    )
    starts_dataset = HDFDataset(
        files=starts_compat_job.out_hdf_files,
        partition_epoch=1,
        segment_file=subset_segments,
        seq_ordering="sorted_reverse",
    )
    frame_length_dataset = HDFDataset(
        files=frame_length_hdf,
        partition_epoch=1,
        segment_file=subset_segments,
        seq_ordering="sorted_reverse",
    )
    return MetaDataset(
        data_map={
            "data": ("cluster_dataset", "data"),
            "segment_starts": ("starts_dataset", "data"),
            "frame_lengths": ("frame_length_dataset", "data"),
        },
        datasets={
            "cluster_dataset": cluster_dataset,
            "starts_dataset": starts_dataset,
            "frame_length_dataset": frame_length_dataset,
        },
        seq_order_control_dataset="cluster_dataset",
    )


def eow_phon_phmm_ls960_rasr_table_decoding():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_rasr_table_decoding"

    train_ogg_zip = tk.Path("/u/zyang/setups/mini/work/i6_core/returnn/oggzip/BlissToOggZipJob.mNyFEqof29Q5/output/out.ogg.zip")
    cv_segments = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/corpus/segments/"
        "ShuffleAndSplitSegmentsJob.RorARIMr0Hr0/output/cv.segments"
    )
    subset_segments_job = SelectSegmentSubsetJob(segment_file=cv_segments, ratio=0.2, random_seed=1)
    subset_segments_job.add_alias(prefix_name + "/train_cv_0p2pct_segments")
    tk.register_output(prefix_name + "/train_cv_0p2pct_segments/segments", subset_segments_job.out_segments)
    train_cv_0p2_segments = subset_segments_job.out_segments

    alignment_hdfs = [tk.Path(f"output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf") for i in range(1, 201)]
    frame_length_job = DumpAlignmentFrameLengthsHDFJob(alignment_hdfs)
    frame_length_job.add_alias(prefix_name + "/alignment_frame_lengths")
    tk.register_output(prefix_name + "/alignment_frame_lengths.hdf", frame_length_job.out_hdf)
    p02_start_hdf = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/segmenter_jobs/DetectPeaksFromScoreHDFJob.Bf3WdJsaZWqv/"
        "output/peak_segment_starts.hdf"
    )

    segment_cluster_datasets = {
        40: _build_cluster_dataset(prefix_name, 40, train_cv_0p2_segments),
        60: _build_cluster_dataset(prefix_name, 60, train_cv_0p2_segments),
        128: _build_cluster_dataset(prefix_name, 128, train_cv_0p2_segments),
    }
    frame_cluster_datasets = {
        num_clusters: _build_frame_cluster_dataset(
            prefix_name=prefix_name,
            num_clusters=num_clusters,
            subset_segments=train_cv_0p2_segments,
            segment_start_hdf=p02_start_hdf,
            frame_length_hdf=frame_length_job.out_hdf,
        )
        for num_clusters in (40, 60, 128)
    }

    subset_bliss_job = OggZipTextToBlissSubsetJob(
        ogg_zip=train_ogg_zip,
        segment_file=train_cv_0p2_segments,
    )
    subset_bliss_job.add_alias(prefix_name + "/train_cv_0p2pct_reference")
    subset_bliss = subset_bliss_job.out_corpus
    tk.register_output(prefix_name + "/train_cv_0p2pct_reference/corpus.xml.gz", subset_bliss)

    returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    lexicon = tk.Path("/work/asr4/zyang/corpora/librispeech/960/lexicon/phmm_no_eow_special_phonemes.lexicon.xml.gz")
    base_lm_config, lm_image = create_lm_image_for_lexicon(
        lexicon_file=lexicon,
        scale=1.0,
        output_prefix=prefix_name + "/no_eow_4gram_lm_image",
    )

    table_variants = [
        (
            "p02_k60_joint",
            60,
            "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
            "ls960_wav2vec2_large_layer15_pca_cluster_eval/p02_segment_k60_1pct_joint_probability.npz",
        ),
        (
            "p02_k40_joint",
            40,
            "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
            "ls960_wav2vec2_large_layer15_pca_cluster_eval/p02_segment_k40_1pct_joint_probability.npz",
        ),
        (
            "p02_k128_joint",
            128,
            "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
            "ls960_wav2vec2_large_layer15_pca_cluster_eval/p02_segment_k128_1pct_joint_probability.npz",
        ),
    ]
    lm_scales = [0.1, 0.2, 1.0, 0.05]
    segment_lm_scales = [0.1, 0.2, 1.0, 0.05, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    score_threshold = 24.0
    collapse_repeated_labels = False

    decode_variants = (
        ("segment", "unmerged", False),
        ("segment", "merged", True),
        ("frame", "", False),
    )

    for table_name, num_clusters, table_file in table_variants:
        for decode_model, input_merge_name, merge_consecutive_cluster_ids in decode_variants:
            dataset = (
                frame_cluster_datasets[num_clusters] if decode_model == "frame" else segment_cluster_datasets[num_clusters]
            )
            model_config = ModelConfig(
                table_file=table_file,
                decode_model=decode_model,
                merge_consecutive_same_cluster_ids_in_segment_mode=merge_consecutive_cluster_ids,
                segment_start_downsample_rate=2,
                max_segment_start_mismatch_ratio=0.1,
                max_num_huge_segment_start_mismatches=10,
            )
            asr_model = ASRModel(
                checkpoint=None,
                net_args={"model_config_dict": asdict(model_config)},
                network_module="phmm.rasr_table_model",
                prior_file=None,
                prior_files=None,
                prefix_name=prefix_name
                + "/"
                + table_name
                + "_"
                + (decode_model if not input_merge_name else decode_model + "_" + input_merge_name),
            )
            summary_rows = []
            lm_scales_for_variant = segment_lm_scales if decode_model == "segment" else lm_scales
            for lm_scale in lm_scales_for_variant:
                lm_config = base_lm_config._copy()
                lm_config.scale = lm_scale
                recog_config = build_librasr_phone_table_recognition_config(
                    lexicon_path=lexicon,
                    lm_config=lm_config,
                    collapse_repeated_labels=collapse_repeated_labels,
                    score_threshold=score_threshold,
                    intermediate_score_threshold=score_threshold,
                    logfile_suffix=(
                        f"phone_table_{table_name}_"
                        f"{decode_model if not input_merge_name else decode_model + '_' + input_merge_name}"
                        f"_lm{lm_scale:g}"
                    ),
                )
                decoder_config = RasrTableDecoderConfig(
                    rasr_config_file=recog_config,
                    lexicon=lexicon,
                    data_key="data",
                    segment_starts_key="segment_starts",
                    frame_lengths_key="frame_lengths",
                    logprob_mode="generative",
                    lm_image_file=lm_image,
                )
                search_name = (
                    prefix_name
                    + "/joint_probability_decoding"
                    + f"/{table_name}"
                    + f"/{decode_model if not input_merge_name else decode_model + '_' + input_merge_name}"
                    + f"/lm{_num_str(lm_scale)}"
                    + f"_collapse{int(collapse_repeated_labels)}"
                )
                _search_jobs, wers = search(
                    search_name,
                    forward_config={
                        "batch_size": 1000,
                        "max_seqs": 64,
                        "num_workers_per_gpu": 0,
                    },
                    asr_model=asr_model,
                    decoder_module="phmm.decoder.rasr_table_decoding_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples={"train_cv_0p2pct": (dataset, subset_bliss)},
                    returnn_exe=returnn_exe,
                    returnn_root=MINI_RETURNN_ROOT,
                    mem_rqmt=16,
                    use_gpu=False,
                )
                summary_rows.append(
                    {
                        "lm_scale": lm_scale,
                        "score_threshold": score_threshold,
                        "collapse_repeated_labels": collapse_repeated_labels,
                        "wer_file": wers[search_name + "/train_cv_0p2pct"],
                    }
                )

            summary_job = TableDecodingTuneSummaryJob(summary_rows)
            summary_job.add_alias(
                prefix_name
                + f"/joint_probability_decoding/{table_name}/"
                + f"{decode_model if not input_merge_name else decode_model + '_' + input_merge_name}/summary"
            )
            tk.register_output(
                prefix_name
                + f"/joint_probability_decoding/{table_name}/"
                + f"{decode_model if not input_merge_name else decode_model + '_' + input_merge_name}/summary.txt",
                summary_job.out_report,
            )

    # table_variants = [
    #     ("k60_l2sq", 60, "/u/zyang/setups/mini/script/mlm_embedding_l2sq_ot_transport_plan.npz"),
    #     ("k40_l2sq_sinkhorn_eps0p01", 40, "/u/zyang/setups/mini/script/mlm_embedding_k40_l2sq_sinkhorn_eps0p01_transport.npz"),
    #     ("k40_l2sq_sinkhorn_eps0p1", 40, "/u/zyang/setups/mini/script/mlm_embedding_k40_l2sq_sinkhorn_eps0p1_transport.npz"),
    #     ("k60_l2sq_linear_map_iter", 60, "/u/zyang/setups/mini/script/mlm_embedding_l2sq_ot_linear_map_iter_final_transport.npz"),
    # ]
    # lm_scales = [0.05, 0.1, 0.01]
    # score_thresholds = [24, 32]
    # collapse_options = [True, False]
    #
    # for table_name, num_clusters, table_file in table_variants:
    #     model_config = ModelConfig(table_file=table_file)
    #     asr_model = ASRModel(
    #         checkpoint=None,
    #         net_args={"model_config_dict": asdict(model_config)},
    #         network_module="phmm.rasr_table_model",
    #         prior_file=None,
    #         prior_files=None,
    #         prefix_name=prefix_name + "/" + table_name,
    #     )
    #     summary_rows = []
    #     for collapse_repeated_labels in collapse_options:
    #         for score_threshold in score_thresholds:
    #             for lm_scale in lm_scales:
    #                 lm_config = base_lm_config._copy()
    #                 lm_config.scale = lm_scale
    #                 recog_config = build_librasr_phone_table_recognition_config(
    #                     lexicon_path=lexicon,
    #                     lm_config=lm_config,
    #                     collapse_repeated_labels=collapse_repeated_labels,
    #                     score_threshold=score_threshold,
    #                     intermediate_score_threshold=score_threshold,
    #                     logfile_suffix=(
    #                         f"phone_table_{table_name}_lm{lm_scale:g}_thr{score_threshold:g}_"
    #                         f"collapse{int(collapse_repeated_labels)}"
    #                     ),
    #                 )
    #                 decoder_config = RasrTableDecoderConfig(
    #                     rasr_config_file=recog_config,
    #                     lexicon=lexicon,
    #                     data_key="data",
    #                     logprob_mode="generative",
    #                     lm_image_file=lm_image,
    #                 )
    #                 search_name = (
    #                     prefix_name
    #                     + "/tuning"
    #                     + f"/{table_name}"
    #                     + f"/lm{_num_str(lm_scale)}"
    #                     + f"_thr{_num_str(score_threshold)}"
    #                     + f"_collapse{int(collapse_repeated_labels)}"
    #                 )
    #                 _search_jobs, wers = search(
    #                     search_name,
    #                     forward_config={
    #                         "batch_size": 1000,
    #                         "max_seqs": 128,
    #                         "num_workers_per_gpu": 0,
    #                     },
    #                     asr_model=asr_model,
    #                     decoder_module="phmm.decoder.rasr_table_decoding_v1",
    #                     decoder_args={"config": asdict(decoder_config)},
    #                     test_dataset_tuples={"train_cv_1pct": (cluster_datasets[num_clusters], subset_bliss)},
    #                     returnn_exe=returnn_exe,
    #                     returnn_root=MINI_RETURNN_ROOT,
    #                     mem_rqmt=16,
    #                     use_gpu=False,
    #                 )
    #                 summary_rows.append(
    #                     {
    #                         "lm_scale": lm_scale,
    #                         "score_threshold": score_threshold,
    #                         "collapse_repeated_labels": collapse_repeated_labels,
    #                         "wer_file": wers[search_name + "/train_cv_1pct"],
    #                     }
    #                 )
    #
    #     summary_job = TableDecodingTuneSummaryJob(summary_rows)
    #     summary_job.add_alias(prefix_name + f"/tuning/{table_name}/summary")
    #     tk.register_output(prefix_name + f"/tuning/{table_name}/summary.txt", summary_job.out_report)


py = eow_phon_phmm_ls960_rasr_table_decoding
