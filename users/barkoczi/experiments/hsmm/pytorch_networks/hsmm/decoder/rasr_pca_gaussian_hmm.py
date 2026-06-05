"""
LibRASR search for PCA-Gaussian phoneme scores.

This decoder keeps the acoustic model used in the frame-accuracy sweep:
wav2vec2 hidden states -> fixed PCA -> diagonal Gaussian scores from
precomputed per-phoneme means and pooled variance. The resulting frame scores
are reordered to the RASR lexicon phoneme inventory and passed to librasr.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Protocol, Union

import numpy as np
import torch
from sisyphus import tk


class TracebackItem(Protocol):
    lemma: str
    am_score: float
    lm_score: float
    start_time: int
    end_time: int


def _traceback_to_string(traceback: List[TracebackItem]) -> str:
    traceback_str = " ".join(item.lemma for item in traceback)
    for token in ("<s>", "</s>", "<blank>", "[BLANK]", "[SILENCE]", "<silence>", "[SENTENCE-END]"):
        traceback_str = traceback_str.replace(token, "")
    return " ".join(traceback_str.split())


@dataclass
class DecoderConfig:
    rasr_config_file: Union[str, tk.Path]
    lexicon: Union[str, tk.Path]
    returnn_vocab: Union[str, tk.Path]
    phoneme_means_pt: Union[str, tk.Path]
    pooled_variance_pt: Union[str, tk.Path]
    silence_label: str = "[SILENCE]"
    target_layer: int = -1
    variance_floor: float = 1.0e-5
    use_label_priors: bool = True
    missing_label_score: float = -1.0e6
    normalize_frame_scores: bool = True


@dataclass
class ExtraConfig:
    print_rtf: bool = True
    sample_rate: int = 16000
    print_hypothesis: bool = True


def _as_path(path) -> str:
    return path.get_path() if hasattr(path, "get_path") else str(path)


def _load_pt(path):
    return torch.load(_as_path(path), map_location="cpu", weights_only=False)


def _load_returnn_labels(vocab_path) -> List[str]:
    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.util.basic import cf

    vocab = Vocabulary.create_vocab(vocab_file=cf(vocab_path), unknown_label=None)
    return list(vocab.labels)


def forward_init_hook(run_ctx, **kwargs):
    from i6_core.lib import lexicon
    from librasr import Configuration, SearchAlgorithm
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    extra_config = ExtraConfig(**kwargs.get("extra_config", {}))

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    lex = lexicon.Lexicon()
    lex.load(cf(config.lexicon))
    run_ctx.label_inventory = list(lex.phonemes.keys())
    if config.silence_label not in lex.phonemes:
        raise ValueError(f"Silence label {config.silence_label!r} not found in lexicon {config.lexicon!r}")

    returnn_labels = _load_returnn_labels(config.returnn_vocab)
    means_payload = _load_pt(config.phoneme_means_pt)
    variance_payload = _load_pt(config.pooled_variance_pt)

    stats_label_ids = torch.as_tensor(means_payload["label_ids"], dtype=torch.long)
    means = torch.as_tensor(means_payload["means"], dtype=torch.float32)
    variance = torch.as_tensor(variance_payload["variance"], dtype=torch.float32)
    inv_variance = torch.reciprocal(torch.clamp(variance, min=config.variance_floor))

    if means.ndim != 2:
        raise ValueError(f"Expected mean matrix [labels, dim], got {tuple(means.shape)}")
    if inv_variance.shape != (means.shape[1],):
        raise ValueError(f"Expected variance dim {means.shape[1]}, got {tuple(inv_variance.shape)}")

    frame_counts = torch.as_tensor(means_payload.get("frame_counts", np.ones(len(stats_label_ids))), dtype=torch.float32)
    priors = frame_counts / torch.clamp(frame_counts.sum(), min=1.0)
    log_priors = torch.log(torch.clamp(priors, min=1.0e-30))

    token_to_stats_idx = {}
    for stats_idx, label_id in enumerate(stats_label_ids.tolist()):
        if 0 <= label_id < len(returnn_labels):
            token_to_stats_idx[returnn_labels[label_id]] = stats_idx

    lex_to_stats_idx = []
    missing_labels = []
    for label in run_ctx.label_inventory:
        stats_idx = token_to_stats_idx.get(label)
        if stats_idx is None:
            lex_to_stats_idx.append(-1)
            missing_labels.append(label)
        else:
            lex_to_stats_idx.append(stats_idx)

    run_ctx.lex_to_stats_idx = torch.tensor(lex_to_stats_idx, dtype=torch.long)
    run_ctx.means = means
    run_ctx.inv_variance = inv_variance
    run_ctx.log_priors = log_priors
    run_ctx.use_label_priors = config.use_label_priors
    run_ctx.missing_label_score = float(config.missing_label_score)
    run_ctx.normalize_frame_scores = bool(config.normalize_frame_scores)
    run_ctx.target_layer = config.target_layer

    rasr_config = Configuration()
    rasr_config.set_from_file(cf(config.rasr_config_file))
    run_ctx.search_algorithm = SearchAlgorithm(config=rasr_config)

    run_ctx.sample_rate = extra_config.sample_rate
    run_ctx.print_rtf = extra_config.print_rtf
    run_ctx.print_hypothesis = extra_config.print_hypothesis

    if missing_labels:
        print(
            "RASR PCA-Gaussian decoder: no statistics for %d/%d lexicon labels: %s"
            % (len(missing_labels), len(run_ctx.label_inventory), ", ".join(missing_labels[:20]))
        )
    print(
        "RASR PCA-Gaussian decoder: mapped %d/%d lexicon labels to statistics."
        % (len(run_ctx.label_inventory) - len(missing_labels), len(run_ctx.label_inventory))
    )

    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0.0
        run_ctx.total_am_time = 0.0
        run_ctx.total_search_time = 0.0


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.print_rtf and run_ctx.running_audio_len_s > 0:
        total_proc_time = run_ctx.total_am_time + run_ctx.total_search_time
        print("Total-AM-Time: %.2fs, AM-RTF: %.3f" % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s))
        print(
            "Total-Search-Time: %.2fs, Search-RTF: %.3f"
            % (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
        )
        print("Total-time: %.2f, Batch-RTF: %.3f" % (total_proc_time, total_proc_time / run_ctx.running_audio_len_s))


def _emission_scores(features: torch.Tensor, run_ctx) -> torch.Tensor:
    if features.shape[-1] < run_ctx.means.shape[-1]:
        raise ValueError(f"Feature dim {features.shape[-1]} is smaller than mean dim {run_ctx.means.shape[-1]}")
    features = features[..., : run_ctx.means.shape[-1]]
    means = run_ctx.means.to(features.device)
    inv_variance = run_ctx.inv_variance.to(features.device)

    diff = features[:, None, :] - means[None, :, :]
    stats_scores = -0.5 * torch.sum(diff * diff * inv_variance[None, None, :], dim=-1)
    if run_ctx.use_label_priors:
        stats_scores = stats_scores + run_ctx.log_priors.to(features.device)[None, :]

    lex_to_stats_idx = run_ctx.lex_to_stats_idx.to(features.device)
    lex_scores = torch.full(
        (features.shape[0], lex_to_stats_idx.shape[0]),
        run_ctx.missing_label_score,
        device=features.device,
        dtype=stats_scores.dtype,
    )
    mapped_mask = lex_to_stats_idx >= 0
    lex_scores[:, mapped_mask] = stats_scores[:, lex_to_stats_idx[mapped_mask]]
    return lex_scores


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / run_ctx.sample_rate
        run_ctx.running_audio_len_s += audio_len_batch

    am_start = perf_counter()
    all_hidden_states, output_lengths = model.extract_hidden_states(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
    transformed_hidden_states = model.transform_hidden_states(all_hidden_states, output_lengths, update_pca=False)
    layer_features = transformed_hidden_states[model.return_layers.index(run_ctx.target_layer)]
    am_time = perf_counter() - am_start

    search_start = perf_counter()
    hypotheses = []
    for seq_features, seq_len in zip(layer_features, output_lengths):
        seq_features = seq_features[:seq_len]
        scores = _emission_scores(seq_features, run_ctx)
        if run_ctx.normalize_frame_scores:
            scores = scores - torch.max(scores, dim=-1, keepdim=True).values
        features = (-scores).detach().cpu().numpy().astype("float32", copy=False)
        features = np.ascontiguousarray(features)
        traceback = run_ctx.search_algorithm.recognize_segment(features=features)
        hypotheses.append(_traceback_to_string(traceback))
    search_time = perf_counter() - search_start

    if run_ctx.print_rtf:
        run_ctx.total_am_time += am_time
        run_ctx.total_search_time += search_time
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    tags = data["seq_tag"]
    for hyp, tag in zip(hypotheses, tags):
        if run_ctx.print_hypothesis:
            print(hyp)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(hyp)))
