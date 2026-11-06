import json
import os
from typing import List, NamedTuple, Optional, Type

import numpy as np
import torch
from i6_core.corpus.convert import CorpusToTextDictJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.text.processing import HeadJob
from i6_core.util import DelayedFormat
from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.common.setups.serialization import Call, Import
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import Job, Task, tk
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

from recipe.data.base import OggZipDataConfig
from recipe.data.librispeech.bpe import get_default_bpe_target_config, vocab_to_bpe_size
from recipe.model_pipelines.common.memrecog import convert_model_for_memristor
from recipe.model_pipelines.common.serializers import get_model_serializers, recipe_imports
from recipe.model_pipelines.common.train import TrainedModel
from recipe.model_pipelines.qat_ffnn_transducer.pytorch_modules import (
    QATFFNNTransducerEncoder as DigitalQATFFNNTransducerEncoder,
    QATFFNNTransducerModel,
    QATFFNNTransducerModelNoJointNet7Offload,
    QATFFNNTransducerModelNoJointOffload,
    QATFFNNTransducerModelNoPredictionOffload,
    QATFFNNTransducerRecogConfig as DigitalQATFFNNTransducerRecogConfig,
    QATFFNNTransducerScorer as DigitalQATFFNNTransducerScorer,
)
from recipe.model_pipelines.qat_ffnn_transducer.mem_inited.pytorch_modules import (
    QATFFNNTransducerEncoder,
    QATFFNNTransducerRecogConfig,
    QATFFNNTransducerJointADCRecogConfig,
    QATFFNNTransducerScorer,
    QATFFNNTransducerScorerNoJointNet7Offload,
    QATFFNNTransducerScorerNoJointOffload,
    QATFFNNTransducerScorerNoPredictionOffload,
)
from recipe.tools import returnn_python_exe, returnn_root


class WriteSegmentsJob(Job):
    def __init__(self, segments: List[str]):
        self.segments = segments
        self.out_segments = self.output_path("segments.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.out_segments.get_path(), "w") as segment_file:
            for segment in self.segments:
                segment_file.write(f"{segment}\n")


def extract_scores(
    model: TrainedModel,
    encoder_class: Type[torch.nn.Module],
    scorer_class: Type[torch.nn.Module],
    data_config: OggZipDataConfig,
) -> ReturnnForwardJobV2:
    target_size = model.model_config.target_size
    returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "classes": {"dim": target_size - 1, "dtype": "int32", "sparse": True},
            },
            "tensor_encoder_time": CodeWrapper('Tensor(name="encoder_time", dims=[batch_dim], dtype="int32")'),
            "dim_encoder_time": CodeWrapper('Dim(dimension=tensor_encoder_time, name="encoder_time")'),
            "tensor_reference_position": CodeWrapper(
                'Tensor(name="reference_position", dims=[batch_dim], dtype="int32")'
            ),
            "dim_reference_position": CodeWrapper(
                'Dim(dimension=tensor_reference_position, name="reference_position")'
            ),
            "dim_vocab": CodeWrapper(f'Dim({target_size}, name="vocab")'),
            "model_outputs": {
                "logits": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_encoder_time, dim_reference_position, dim_vocab)"),
                    "dtype": "float32",
                },
                "labels": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_reference_position)"),
                    "dtype": "int32",
                },
            },
            "backend": "torch",
            "batch_size": 360 * 16_000,
            "max_seqs": 1,
        },
        python_prolog=recipe_imports + ["import torch", "from returnn.tensor import Dim, Tensor, batch_dim"],
        python_epilog=[
            get_model_serializers(model_class=encoder_class, model_config=model.model_config),
            Import(f"{scorer_class.__module__}.{scorer_class.__name__}", import_as="ScorerModel"),
            Import(f"{_make_reference_score_step.__module__}.{_make_reference_score_step.__name__}"),
            Call(
                _make_reference_score_step.__name__,
                kwargs=[
                    ("scorer_class", CodeWrapper("ScorerModel")),
                    ("scorer_config", CodeWrapper("cfg")),
                    ("checkpoint", DelayedFormat('"{}"', model.get_checkpoint().path)),
                ],
                return_assign_variables="forward_step",
            ),
            Import(
                f"{ReferenceScoreCallback.__module__}.{ReferenceScoreCallback.__name__}",
                import_as="forward_callback",
            ),
        ],
        sort_config=False,
    )
    returnn_config.update(data_config.get_returnn_data("forward_data"))

    job = ReturnnForwardJobV2(
        model_checkpoint=model.get_checkpoint(),
        returnn_config=returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=["scores"],
        device="gpu",
        cpu_rqmt=4,
        mem_rqmt=24,
        time_rqmt=24,
    )
    job.rqmt["gpu_mem"] = 48
    return job


def _make_reference_score_step(*, scorer_class, scorer_config, checkpoint: str):
    scorer = None

    def forward_step(*, model: QATFFNNTransducerEncoder, extern_data: TensorDict, **_):
        nonlocal scorer
        raw_data = extern_data.as_raw_tensor_dict()
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        labels = raw_data["classes"].long()
        label_lengths = raw_data["classes:size1"].to(device=audio_samples.device)

        if scorer is None:
            scorer = scorer_class(cfg=scorer_config)
            checkpoint_data = torch.load(checkpoint, map_location="cpu")
            incompatible_keys = scorer.load_state_dict(checkpoint_data["model"], strict=False)
            assert not incompatible_keys.missing_keys, incompatible_keys.missing_keys
            scorer.to(device=audio_samples.device)
            scorer.eval()

        assert audio_samples.size(0) == 1, "Reference score extraction requires max_seqs=1"
        encoder_states, encoder_lengths = model.forward(audio_samples, audio_samples_size)
        prediction_states = scorer.forward_prediction_network(labels)
        logits = scorer.forward_joint_network(encoder_states, encoder_lengths, prediction_states, label_lengths)

        encoder_length = int(encoder_lengths[0].item())
        reference_positions = int(label_lengths[0].item()) + 1
        logits = logits.reshape(1, encoder_length, reference_positions, scorer.target_size)

        # The last reference position is the history after the final reference label.
        labels_with_final_position = torch.nn.functional.pad(
            labels[:, : reference_positions - 1], [0, 1], value=-1
        ).to(dtype=torch.int32)

        import returnn.frontend as rf

        run_ctx = rf.get_run_ctx()
        assert run_ctx.expected_outputs is not None
        run_ctx.expected_outputs["logits"].dims[1].dyn_size_ext.raw_tensor = encoder_lengths
        run_ctx.expected_outputs["logits"].dims[2].dyn_size_ext.raw_tensor = label_lengths + 1
        run_ctx.expected_outputs["labels"].dims[1].dyn_size_ext.raw_tensor = label_lengths + 1
        run_ctx.mark_as_output(logits, name="logits")
        run_ctx.mark_as_output(labels_with_final_position, name="labels")

    return forward_step


class ReferenceScoreCallback(ForwardCallbackIface):
    def init(self, **_):
        os.makedirs("scores", exist_ok=True)
        self.manifest = {}

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        raw_outputs = outputs.as_raw_tensor_dict()
        filename = seq_tag.replace("/", "__") + ".npz"
        logits = np.asarray(raw_outputs["logits"], dtype=np.float32)
        labels = np.asarray(raw_outputs["labels"], dtype=np.int32)
        blank_label = logits.shape[-1] - 1
        np.savez_compressed(
            os.path.join("scores", filename),
            logits=logits,
            labels=labels[:-1],
            encoder_length=np.int32(logits.shape[0]),
            label_length=np.int32(labels.shape[0] - 1),
            blank_label=np.int32(blank_label),
        )
        self.manifest[seq_tag] = {
            "file": filename,
            "logits_shape": list(logits.shape),
            "label_length": int(labels.shape[0] - 1),
            "blank_label": blank_label,
        }

    def finish(self):
        with open("scores/manifest.json", "w") as manifest_file:
            json.dump(self.manifest, manifest_file, indent=2, sort_keys=True)


def _monotonic_rnnt_alignment_posteriors(logits: np.ndarray, labels: np.ndarray, blank_label: int):
    shifted_logits = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True) # subtract max, no logic change.
    log_probs = shifted_logits - np.log(np.sum(np.exp(shifted_logits), axis=-1, keepdims=True)) # softmax
    num_frames, reference_positions, _ = logits.shape
    num_labels = labels.shape[0]
    assert reference_positions == num_labels + 1

    alpha = np.full((num_frames, num_labels + 1), -np.inf, dtype=np.float64)
    for t in range(num_frames):
        for u in range(num_labels + 1): # all (t, u)
            previous_blank = 0.0 if t == 0 and u == 0 else (alpha[t - 1, u] if t > 0 else -np.inf)
            blank_path = previous_blank + log_probs[t, u, blank_label]

            previous_label = 0.0 if t == 0 and u == 1 else (alpha[t - 1, u - 1] if t > 0 and u > 0 else -np.inf)
            label_path = (
                previous_label + log_probs[t, u - 1, labels[u - 1]] if u > 0 else -np.inf
            )
            alpha[t, u] = np.logaddexp(blank_path, label_path)

    log_likelihood = alpha[-1, -1]
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -1] = 0.0
    for t in range(num_frames - 2, -1, -1):
        for u in range(num_labels + 1): # reverse loop but (t, u)
            blank_path = log_probs[t + 1, u, blank_label] + beta[t + 1, u]
            label_path = (
                log_probs[t + 1, u, labels[u]] + beta[t + 1, u + 1] if u < num_labels else -np.inf
            )
            beta[t, u] = np.logaddexp(blank_path, label_path)

    label_posteriors = np.zeros((num_frames, num_labels), dtype=np.float64)
    for t in range(num_frames):
        for u in range(num_labels):
            previous = 0.0 if t == 0 and u == 0 else (alpha[t - 1, u] if t > 0 else -np.inf)
            label_posteriors[t, u] = np.exp(
                previous + log_probs[t, u, labels[u]] + beta[t, u + 1] - log_likelihood
            )

    return log_likelihood, label_posteriors


class AnalyzeReferenceScoresJob(Job):
    def __init__(
        self, score_dirs: dict[str, tk.Path], transcript_dict: tk.Path, reference_model: str = "base_model"
    ):
        self.score_dirs = score_dirs
        self.transcript_dict = transcript_dict
        self.reference_model = reference_model
        self.out_summary = self.output_path("summary.json")
        self.out_table = self.output_path("summary.tsv")
        self.out_margin_heatmaps = self.output_path("margin_heatmaps", directory=True)
        self.out_focused_margin_heatmaps = self.output_path("focused_margin_heatmaps", directory=True)
        self.out_model_centered_margin_heatmaps = self.output_path(
            "model_centered_margin_heatmaps", directory=True
        )

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 1, "gpu": 1, "gpu_mem": 11})

    def run(self):
        import ast

        from i6_native_ops.monotonic_rnnt import monotonic_rnnt_loss

        with open(self.transcript_dict.get_path()) as transcript_file:
            transcripts = ast.literal_eval(transcript_file.read())

        manifests = {}
        for model_name, score_dir in self.score_dirs.items():
            with open(os.path.join(score_dir.get_path(), "manifest.json")) as manifest_file:
                manifests[model_name] = json.load(manifest_file)

        reference_segments = set(manifests[self.reference_model])
        assert all(set(manifest) == reference_segments for manifest in manifests.values())
        assert reference_segments <= transcripts.keys()

        display_names = {
            "digital_model": "QAT: Encoder + Predictor + Joiner Layer 1 + Joiner Output Layer; On-device: -",
            "no_joint_model": "QAT: Joiner Layer 1 + Joiner Output Layer; On-device: Encoder + Predictor",
            "no_joint_7_model": "QAT: Joiner Output Layer; On-device: Encoder + Predictor + Joiner Layer 1",
            "no_pred_model": "QAT: Predictor; On-device: Encoder + Joiner Layer 1 + Joiner Output Layer",
            "base_model": "QAT: - ; On-device: Encoder + Predictor + Joiner Layer 1 + Joiner Output Layer",
        }
        summary = {}
        margin_heatmaps = {}
        for seq_tag in sorted(reference_segments):
            model_results = {}
            lattices = {}
            expected_emission_frames = {}
            alignment_posteriors = {}
            blank_probabilities_by_model = {}
            oracle_probabilities_by_model = {}
            for model_name, score_dir in self.score_dirs.items():
                filename = manifests[model_name][seq_tag]["file"]
                with np.load(os.path.join(score_dir.get_path(), filename)) as data:
                    logits = np.asarray(data["logits"], dtype=np.float32)
                    labels = np.asarray(data["labels"], dtype=np.int64)
                    blank_label = int(data["blank_label"])

                input_length, reference_positions, vocab_size = logits.shape
                label_length = labels.shape[0]
                assert reference_positions == label_length + 1
                #### COMPARE NEXT BLANK VS NEXT ORACLE PROBABLITY #######
                shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
                probabilities = np.exp(shifted_logits)
                probabilities /= np.sum(probabilities, axis=-1, keepdims=True) # get back to probabilities, softmax
                blank_probabilities = probabilities[:, :label_length, blank_label] # blank at all reference positions
                oracle_probabilities = np.take_along_axis(
                    probabilities[:, :label_length, :],
                    np.broadcast_to(labels[None, :, None], (input_length, label_length, 1)),
                    axis=-1,
                )[..., 0] # probability of the correct label at each reference position
                ## END ## BEGIN - CHECK LOSS AND ALIGNMENT POSTERIORS ####
                acts = torch.from_numpy(logits.reshape(-1, vocab_size)).requires_grad_(True)
                loss = monotonic_rnnt_loss(
                    acts=acts,
                    labels=torch.from_numpy(labels[None, :]),
                    input_lengths=torch.tensor([input_length], dtype=torch.int32),
                    label_lengths=torch.tensor([label_length], dtype=torch.int32),
                    blank_label=blank_label,
                ).item()
                log_probability = -loss
                forward_log_probability, label_posteriors = _monotonic_rnnt_alignment_posteriors(
                    logits=logits, labels=labels, blank_label=blank_label
                )
                assert np.isclose(forward_log_probability, log_probability, atol=1e-3), (
                    forward_log_probability,
                    log_probability,
                ) # just to check custom implementation. ## END ## BEGIN - model specific alignment metrics
                emission_frames = np.sum(
                    label_posteriors * np.arange(input_length, dtype=np.float64)[:, None], axis=0
                ) # 
                posterior_entropy = -np.sum(
                    label_posteriors * np.log(np.maximum(label_posteriors, np.finfo(np.float64).tiny)), axis=0
                )
                emission_weighted_oracle_probability = np.sum( # Around the frames this model selects as its favorable reference-emission region, how strong is its local token score?
                    label_posteriors * oracle_probabilities, axis=0
                ) # frames where the alignment posterior says token u was likely emitted, what local probability did the scorer assign to that token
                emission_weighted_blank_probability = np.sum(
                    label_posteriors * blank_probabilities, axis=0
                ) # At those same token-emission locations, how much local probability did the scorer assign to the competing blank symbol
                emission_weighted_label_blank_log_margin = np.sum(
                    label_posteriors
                    * (
                        np.log(np.maximum(oracle_probabilities, np.finfo(np.float64).tiny))
                        - np.log(np.maximum(blank_probabilities, np.finfo(np.float64).tiny))
                    ),
                    axis=0,
                )
                model_results[model_name] = {
                    "model_display_name": display_names.get(model_name, model_name),
                    "log_probability": log_probability,
                    "log_probability_per_token": log_probability / label_length,
                    "mean_blank_probability": float(np.mean(blank_probabilities)),
                    "mean_oracle_token_probability": float(np.mean(oracle_probabilities)),
                    "blank_favored_over_oracle_fraction": float(
                        np.mean(blank_probabilities > oracle_probabilities)
                    ),
                    "mean_expected_emission_frame": float(np.mean(emission_frames)),
                    "mean_expected_emission_frame_normalized": float(
                        np.mean(emission_frames / max(input_length - 1, 1))
                    ),
                    "expected_first_token_emission_frame": float(emission_frames[0]),
                    "mean_alignment_posterior_entropy": float(np.mean(posterior_entropy)),
                    "mean_emission_weighted_oracle_probability": float(
                        np.mean(emission_weighted_oracle_probability)
                    ),
                    "mean_emission_weighted_blank_probability": float(
                        np.mean(emission_weighted_blank_probability)
                    ),
                    "mean_emission_weighted_label_blank_log_margin": float(
                        np.mean(emission_weighted_label_blank_log_margin)
                    ),
                    "expected_emission_frames": emission_frames.tolist(),
                    "num_encoder_frames": input_length,
                    "num_labels": label_length,
                }
                lattices[model_name] = logits
                expected_emission_frames[model_name] = emission_frames
                alignment_posteriors[model_name] = label_posteriors
                blank_probabilities_by_model[model_name] = blank_probabilities
                oracle_probabilities_by_model[model_name] = oracle_probabilities

            reference_result = model_results[self.reference_model]
            reference_logits = lattices[self.reference_model]
            reference_emission_frames = expected_emission_frames[self.reference_model]
            reference_alignment_posteriors = alignment_posteriors[self.reference_model]
            for model_name, result in model_results.items():
                delta = lattices[model_name] - reference_logits
                emission_frame_delta = expected_emission_frames[model_name] - reference_emission_frames
                model_blank_probabilities = blank_probabilities_by_model[model_name]
                model_oracle_probabilities = oracle_probabilities_by_model[model_name]
                reference_alignment_oracle_probabilities = np.sum(
                    reference_alignment_posteriors * model_oracle_probabilities, axis=0
                )
                reference_alignment_blank_probabilities = np.sum(
                    reference_alignment_posteriors * model_blank_probabilities, axis=0
                )
                reference_alignment_log_margins = np.sum(
                    reference_alignment_posteriors
                    * (
                        np.log(np.maximum(model_oracle_probabilities, np.finfo(np.float64).tiny))
                        - np.log(np.maximum(model_blank_probabilities, np.finfo(np.float64).tiny))
                    ),
                    axis=0,
                )
                result["log_probability_delta"] = (
                    result["log_probability"] - reference_result["log_probability"]
                )
                result["log_probability_per_token_delta"] = (
                    result["log_probability_per_token"] - reference_result["log_probability_per_token"]
                )
                result["lattice_mean_absolute_delta"] = float(np.mean(np.abs(delta)))
                result["lattice_max_absolute_delta"] = float(np.max(np.abs(delta)))
                result["lattice_root_mean_square_delta"] = float(np.sqrt(np.mean(np.square(delta))))
                result["mean_emission_frame_delta"] = float(np.mean(emission_frame_delta))
                result["mean_absolute_emission_frame_delta"] = float(np.mean(np.abs(emission_frame_delta)))
                result["first_token_emission_frame_delta"] = float(emission_frame_delta[0])
                result["reference_alignment_mean_oracle_probability"] = float(
                    np.mean(reference_alignment_oracle_probabilities)
                )
                result["reference_alignment_mean_blank_probability"] = float(
                    np.mean(reference_alignment_blank_probabilities)
                )
                result["reference_alignment_mean_label_blank_log_margin"] = float(
                    np.mean(reference_alignment_log_margins)
                )

            summary[seq_tag] = model_results
            margin_heatmaps[seq_tag] = {
                model_name: np.log(
                    np.maximum(oracle_probabilities_by_model[model_name], np.finfo(np.float64).tiny)
                )
                - np.log(
                    np.maximum(blank_probabilities_by_model[model_name], np.finfo(np.float64).tiny)
                )
                for model_name in model_results
            }

        with open(self.out_summary.get_path(), "w") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)

        columns = [
            "seq_tag",
            "model",
            "model_display_name",
            "log_probability",
            "log_probability_per_token",
            "log_probability_delta",
            "log_probability_per_token_delta",
            "mean_blank_probability",
            "mean_oracle_token_probability",
            "blank_favored_over_oracle_fraction",
            "mean_expected_emission_frame",
            "mean_expected_emission_frame_normalized",
            "expected_first_token_emission_frame",
            "mean_alignment_posterior_entropy",
            "mean_emission_weighted_oracle_probability",
            "mean_emission_weighted_blank_probability",
            "mean_emission_weighted_label_blank_log_margin",
            "reference_alignment_mean_oracle_probability",
            "reference_alignment_mean_blank_probability",
            "reference_alignment_mean_label_blank_log_margin",
            "mean_emission_frame_delta",
            "mean_absolute_emission_frame_delta",
            "first_token_emission_frame_delta",
            "lattice_mean_absolute_delta",
            "lattice_max_absolute_delta",
            "lattice_root_mean_square_delta",
            "num_encoder_frames",
            "num_labels",
        ]
        with open(self.out_table.get_path(), "w") as table_file:
            table_file.write("\t".join(columns) + "\n")
            for seq_tag, model_results in summary.items():
                for model_name, result in model_results.items():
                    values = [seq_tag, model_name] + [str(result[column]) for column in columns[2:]]
                    table_file.write("\t".join(values) + "\n")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import textwrap

        preferred_order = [
            "digital_model",
            "no_joint_model",
            "no_joint_7_model",
            "no_pred_model",
            "base_model",
        ]
        model_order = [name for name in preferred_order if name in self.score_dirs]
        os.makedirs(self.out_margin_heatmaps.get_path(), exist_ok=True)
        os.makedirs(self.out_focused_margin_heatmaps.get_path(), exist_ok=True)
        os.makedirs(self.out_model_centered_margin_heatmaps.get_path(), exist_ok=True)
        for seq_tag, model_heatmaps in margin_heatmaps.items():
            figure_title = f"{seq_tag}\n{textwrap.fill(transcripts[seq_tag], width=85)}"
            all_margins = np.concatenate([model_heatmaps[name].ravel() for name in model_order])
            color_limit = max(1.0, float(np.percentile(np.abs(all_margins), 95)))
            fig, axes = plt.subplots(
                len(model_order),
                1,
                figsize=(12, 2.8 * len(model_order)),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for ax, model_name in zip(axes, model_order):
                image = ax.imshow(
                    model_heatmaps[model_name].T,
                    aspect="auto",
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-color_limit,
                    vmax=color_limit,
                    interpolation="nearest",
                )
                emission_frames = np.asarray(
                    summary[seq_tag][model_name]["expected_emission_frames"], dtype=np.float64
                )
                ax.plot(emission_frames, np.arange(emission_frames.size), color="black", linewidth=1.0)
                ax.set_title(display_names[model_name], loc="left", fontsize=11)
                ax.set_ylabel("Reference token index")
            axes[-1].set_xlabel("Encoder frame")
            colorbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
            colorbar.set_label("log P(reference token) - log P(blank)")
            fig.suptitle(figure_title, fontsize=9)
            filename = seq_tag.replace("/", "__") + ".png"
            fig.savefig(os.path.join(self.out_margin_heatmaps.get_path(), filename), dpi=180)
            plt.close(fig)

            frame_offsets = np.arange(-8, 9)
            reference_emission_frames = np.asarray(
                summary[seq_tag][self.reference_model]["expected_emission_frames"], dtype=np.float64
            )
            reference_frame_indices = np.rint(reference_emission_frames).astype(np.int64)
            focused_heatmaps = {}
            for model_name in model_order:
                full_heatmap = model_heatmaps[model_name]
                frame_indices = reference_frame_indices[:, None] + frame_offsets[None, :]
                valid = (frame_indices >= 0) & (frame_indices < full_heatmap.shape[0])
                clipped_frame_indices = np.clip(frame_indices, 0, full_heatmap.shape[0] - 1)
                token_indices = np.arange(full_heatmap.shape[1])[:, None]
                focused_heatmap = full_heatmap[clipped_frame_indices, token_indices]
                focused_heatmap[~valid] = np.nan
                focused_heatmaps[model_name] = focused_heatmap

            focused_values = np.concatenate(
                [focused_heatmaps[name][np.isfinite(focused_heatmaps[name])] for name in model_order]
            )
            focused_color_limit = max(1.0, float(np.percentile(np.abs(focused_values), 95)))
            fig, axes = plt.subplots(
                len(model_order),
                1,
                figsize=(9, 2.8 * len(model_order)),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for ax, model_name in zip(axes, model_order):
                image = ax.imshow(
                    focused_heatmaps[model_name],
                    aspect="auto",
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-focused_color_limit,
                    vmax=focused_color_limit,
                    interpolation="nearest",
                    extent=(frame_offsets[0] - 0.5, frame_offsets[-1] + 0.5, -0.5, reference_emission_frames.size - 0.5),
                )
                model_emission_frames = np.asarray(
                    summary[seq_tag][model_name]["expected_emission_frames"], dtype=np.float64
                )
                ax.plot(
                    model_emission_frames - reference_emission_frames,
                    np.arange(reference_emission_frames.size),
                    color="black",
                    linewidth=1.0,
                )
                ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
                ax.set_title(display_names[model_name], loc="left", fontsize=11)
                ax.set_ylabel("Reference token index")
            axes[-1].set_xlabel("Encoder-frame offset from digital expected emission")
            colorbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
            colorbar.set_label("log P(reference token) - log P(blank)")
            fig.suptitle(figure_title, fontsize=9)
            fig.savefig(os.path.join(self.out_focused_margin_heatmaps.get_path(), filename), dpi=180)
            plt.close(fig)

            model_centered_heatmaps = {}
            for model_name in model_order:
                full_heatmap = model_heatmaps[model_name]
                model_emission_frames = np.asarray(
                    summary[seq_tag][model_name]["expected_emission_frames"], dtype=np.float64
                )
                model_frame_indices = np.rint(model_emission_frames).astype(np.int64)
                frame_indices = model_frame_indices[:, None] + frame_offsets[None, :]
                valid = (frame_indices >= 0) & (frame_indices < full_heatmap.shape[0])
                clipped_frame_indices = np.clip(frame_indices, 0, full_heatmap.shape[0] - 1)
                token_indices = np.arange(full_heatmap.shape[1])[:, None]
                model_centered_heatmap = full_heatmap[clipped_frame_indices, token_indices]
                model_centered_heatmap[~valid] = np.nan
                model_centered_heatmaps[model_name] = model_centered_heatmap

            model_centered_values = np.concatenate(
                [
                    model_centered_heatmaps[name][np.isfinite(model_centered_heatmaps[name])]
                    for name in model_order
                ]
            )
            model_centered_color_limit = max(
                1.0, float(np.percentile(np.abs(model_centered_values), 95))
            )
            fig, axes = plt.subplots(
                len(model_order),
                1,
                figsize=(9, 2.8 * len(model_order)),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for ax, model_name in zip(axes, model_order):
                image = ax.imshow(
                    model_centered_heatmaps[model_name],
                    aspect="auto",
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-model_centered_color_limit,
                    vmax=model_centered_color_limit,
                    interpolation="nearest",
                    extent=(frame_offsets[0] - 0.5, frame_offsets[-1] + 0.5, -0.5, reference_emission_frames.size - 0.5),
                )
                model_emission_frames = np.asarray(
                    summary[seq_tag][model_name]["expected_emission_frames"], dtype=np.float64
                )
                ax.plot(
                    reference_emission_frames - model_emission_frames,
                    np.arange(reference_emission_frames.size),
                    color="black",
                    linewidth=1.0,
                )
                ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
                ax.set_title(display_names[model_name], loc="left", fontsize=11)
                ax.set_ylabel("Reference token index")
            axes[-1].set_xlabel("Encoder-frame offset from each model's expected emission")
            colorbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
            colorbar.set_label("log P(reference token) - log P(blank)")
            fig.suptitle(figure_title, fontsize=9)
            fig.savefig(
                os.path.join(self.out_model_centered_margin_heatmaps.get_path(), filename), dpi=180
            )
            plt.close(fig)


class FitBlankPenaltyJob(Job):
    """Fit digital-to-fully-on-device blank offsets from extracted score lattices."""

    def __init__(self, digital_scores: tk.Path, on_device_scores: tk.Path):
        self.digital_scores = digital_scores
        self.on_device_scores = on_device_scores
        self.out_summary = self.output_path("blank_penalty_fit.json")
        self.out_report = self.output_path("blank_penalty_fit.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 1})

    def run(self):
        from pathlib import Path

        from diagnostics.qat_ffnn_scores.fit_blank_penalty import analyze_scores, format_results

        results = analyze_scores(Path(self.digital_scores.get_path()), Path(self.on_device_scores.get_path()))
        with open(self.out_summary.get_path(), "w") as summary_file:
            json.dump(results, summary_file, indent=2)
        with open(self.out_report.get_path(), "w") as report_file:
            report_file.write(format_results(results))


def _convert_model_for_memristor(
    model_class,
    model: TrainedModel,
    descriptor_suffix: str,
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    correction_settings: Optional[CycleCorrectionSettings],
    num_cycles: int,
    recog_config: QATFFNNTransducerRecogConfig,
    regrid_embedding: bool = True,
    joint_correction_settings: Optional[CycleCorrectionSettings] = None,
) -> TrainedModel:
    if joint_correction_settings is not None:
        descriptor_suffix += (
            f"_joint_correction_{joint_correction_settings.num_cycles}cycles"
            f"_{joint_correction_settings.test_input_value}input"
            f"_{joint_correction_settings.relative_deviation}dev"
            f"_ideal_{joint_correction_settings.ideal_programming}"
        )
    if not regrid_embedding:
        descriptor_suffix += "_no_embedding_regrid"
    memristor_model, _ = convert_model_for_memristor(
        checkpoint=model.get_checkpoint(),
        config=model.model_config,
        model_class=model_class,
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        correction_settings=correction_settings,
        num_cycles=num_cycles,
        regrid_embedding=regrid_embedding,
        joint_correction_settings=joint_correction_settings,
        alias=f"{model.descriptor}{descriptor_suffix}_memristor_num_cycles_{num_cycles}",
    )

    return TrainedModel(
        descriptor=f"memristor_{model.descriptor}{descriptor_suffix}",
        model_config=recog_config,
        checkpoints={0: memristor_model},
    )


class PreparedReferenceScores(NamedTuple):
    score_dirs: dict[str, tk.Path]
    bliss_corpus: tk.Path
    segment_file: tk.Path
    output_prefix: str


def prepare_scores(
    model: TrainedModel,
    corpus_name: str = "dev-other",
    segments: Optional[List[str]] = None,
    num_segments: int | float = 3,
    model_names: Optional[List[str]] = None,
    output_prefix: str = "diagnostics/qat_ffnn_scores",
    converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    joint_l1_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    joint_l2_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    regrid_embedding: bool = True,
    joint_correction_settings: Optional[CycleCorrectionSettings] = None,
) -> PreparedReferenceScores:
    """Create extraction jobs; num_segments <= 1 selects the first 1/N corpus split."""
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    output_prefix = output_prefix.strip("/")
    if not output_prefix:
        raise ValueError("output_prefix must not be empty")
    if converter_hardware_settings is None:
        converter_hardware_settings = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=4,
            output_range_bits=4,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )

    if pos_enc_converter_hardware_settings is None:
        pos_enc_converter_hardware_settings = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=1,
            output_range_bits=7,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
    
    model_specs = {
        "base_model": (QATFFNNTransducerModel, QATFFNNTransducerScorer, ""),
        "no_joint_model": (
            QATFFNNTransducerModelNoJointOffload,
            QATFFNNTransducerScorerNoJointOffload,
            "_no_joint_offload",
        ),
        "no_joint_7_model": (
            QATFFNNTransducerModelNoJointNet7Offload,
            QATFFNNTransducerScorerNoJointNet7Offload,
            "_no_joint_net_7_offload",
        ),
        "no_pred_model": (
            QATFFNNTransducerModelNoPredictionOffload,
            QATFFNNTransducerScorerNoPredictionOffload,
            "_no_pred_offload",
        ),
    }
    if model_names is None:
        model_names = list(model_specs)
    unknown_models = set(model_names) - model_specs.keys()
    if unknown_models:
        raise ValueError(f"Unknown on-device models: {sorted(unknown_models)}")

    memristor_config = model.model_config.with_replaced(
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        correction_settings=None,
        num_cycles=1,
    )
    if joint_l1_converter_hardware_settings is not None:
        if joint_l2_converter_hardware_settings is None:
            raise ValueError("joint_l2_converter_hardware_settings must be provided if joint_l1_converter_hardware_settings is provided")
        recog_config = QATFFNNTransducerJointADCRecogConfig(
        **vars(memristor_config),
        ilm_scale=0.0,
        blank_penalty=0.0,
        joint_l1_converter_hardware_settings=joint_l1_converter_hardware_settings,
        joint_l2_converter_hardware_settings=joint_l2_converter_hardware_settings
    )
    else:
        recog_config = QATFFNNTransducerRecogConfig(
            **vars(memristor_config),
            ilm_scale=0.0,
            blank_penalty=0.0,
        )

    bliss_corpus = get_bliss_corpus_dict("wav")[corpus_name]
    if segments is None:
        if num_segments <= 1:
            segment_file = SegmentCorpusJob(bliss_corpus, max(1, round(1 / num_segments))).out_single_segment_files[1]
        else:
            all_segments = SegmentCorpusJob(bliss_corpus, 1).out_single_segment_files[1]
            segment_file = HeadJob(all_segments, num_lines=num_segments).out
    else:
        assert segments, "segments must not be empty"
        segment_file = WriteSegmentsJob(segments).out_segments

    bpe_size = vocab_to_bpe_size(model.model_config.target_size - 1)
    data_config = OggZipDataConfig.from_bliss(
        bliss_corpus_files=[bliss_corpus],
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
        target_config=get_default_bpe_target_config(bpe_size),
        segment_file=segment_file,
    )
    tk.register_output(f"{output_prefix}/segments", segment_file)

    digital_recog_config = DigitalQATFFNNTransducerRecogConfig(
        **vars(model.model_config),
        ilm_scale=0.0,
        blank_penalty=0.0,
    )
    digital_model = TrainedModel(
        descriptor=f"{model.descriptor}_digital",
        model_config=digital_recog_config,
        checkpoints={0: model.get_checkpoint()},
    )
    digital_job = extract_scores(
        model=digital_model,
        encoder_class=DigitalQATFFNNTransducerEncoder,
        scorer_class=DigitalQATFFNNTransducerScorer,
        data_config=data_config,
    )
    digital_job.add_alias(f"{output_prefix}/digital_model")
    digital_scores = digital_job.out_files["scores"]
    tk.register_output(f"{output_prefix}/digital_model", digital_scores)
    outputs = {"digital_model": digital_scores}

    for model_name in dict.fromkeys(model_names):
        conversion_class, scorer_class, descriptor_suffix = model_specs[model_name]
        converted_model = _convert_model_for_memristor(
            model_class=conversion_class,
            model=model,
            descriptor_suffix=descriptor_suffix,
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=None,
            num_cycles=1,
            recog_config=recog_config,
            regrid_embedding=regrid_embedding,
            joint_correction_settings=joint_correction_settings,
        )
        job = extract_scores(
            model=converted_model,
            encoder_class=QATFFNNTransducerEncoder,
            scorer_class=scorer_class,
            data_config=data_config,
        )
        job.add_alias(f"{output_prefix}/{model_name}")
        scores = job.out_files["scores"]
        tk.register_output(f"{output_prefix}/{model_name}", scores)
        outputs[model_name] = scores

    return PreparedReferenceScores(
        score_dirs=outputs, bliss_corpus=bliss_corpus, segment_file=segment_file, output_prefix=output_prefix
    )


def run_analysis(prepared: PreparedReferenceScores) -> AnalyzeReferenceScoresJob:
    """Analyze all prepared model scores using a shared digital alignment."""
    output_prefix = prepared.output_prefix
    transcript_dict = CorpusToTextDictJob(
        bliss_corpus=prepared.bliss_corpus, segment_file=prepared.segment_file
    ).out_dictionary
    analysis_job = AnalyzeReferenceScoresJob(
        score_dirs=prepared.score_dirs, transcript_dict=transcript_dict, reference_model="digital_model"
    )
    analysis_job.add_alias(f"{output_prefix}/analysis")
    tk.register_output(f"{output_prefix}/summary.json", analysis_job.out_summary)
    tk.register_output(f"{output_prefix}/summary.tsv", analysis_job.out_table)
    tk.register_output(f"{output_prefix}/margin_heatmaps", analysis_job.out_margin_heatmaps)
    tk.register_output(
        f"{output_prefix}/focused_margin_heatmaps", analysis_job.out_focused_margin_heatmaps
    )
    tk.register_output(
        f"{output_prefix}/model_centered_margin_heatmaps",
        analysis_job.out_model_centered_margin_heatmaps,
    )
    return analysis_job


def run_blank_penalty_fit(prepared: PreparedReferenceScores) -> FitBlankPenaltyJob:
    """Fit digital-to-fully-on-device blank penalties from the prepared scores."""
    output_prefix = prepared.output_prefix
    penalty_job = FitBlankPenaltyJob(
        digital_scores=prepared.score_dirs["digital_model"], on_device_scores=prepared.score_dirs["base_model"]
    )
    penalty_job.add_alias(f"{output_prefix}/blank_penalty_fit")
    tk.register_output(f"{output_prefix}/blank_penalty_fit.json", penalty_job.out_summary)
    tk.register_output(f"{output_prefix}/blank_penalty_fit.txt", penalty_job.out_report)

    return penalty_job
