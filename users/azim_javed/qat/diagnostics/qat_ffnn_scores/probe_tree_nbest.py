"""Probe existing LibRASR N-best access using one saved recognition-job config.

Run on a GPU in the recognition container. No manager or new recognition jobs
are created. Original configs/checkpoints are read; probe files go to output-dir.
The default job is the full-quant, cycle-1, no-LM tree_mbs1024 run with joint
L2 ADC 2prec/6range. The default anchor is the local penalty estimate 1.44.

This checks candidate collection, traceback score agreement, and whether the
lexicon makes token counts identifiable. It does not optimize a blank penalty.
"""

import argparse
import ast
import configparser
import gzip
import io
import json
import math
import shlex
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from time import perf_counter


SETUP = Path(__file__).resolve().parents[2]
DEFAULT_JOB = SETUP / (
    "alias/recognition/memristor_qat_ffnn_transducer_full_quant_v2_lowbs"
    "_joint_l1_4prec_4range_l2_2prec_6range_separated_joint_adc"
    "__recog_num_cycle_1_batched_decoder_cache_tree_mbs1024/dev-other/shard_1_5"
)
DEFAULT_SISYPHUS = Path("/work/smt4/azim.javed/repositories/sisyphus_repo")
ESTIMATED_BLANK_PENALTY = 1.44
PROBE_ROOT = SETUP / "output/diagnostics/qat_ffnn_scores/tree_nbest_probe_jl2_2prec_6range"


def job_paths(job_dir):
    """Resolve the saved command without executing its shell script."""
    command = shlex.split((job_dir / "work/rnn.sh").read_text())
    rnn_index = next(i for i, value in enumerate(command) if value.endswith("/rnn.py"))
    return Path(command[rnn_index + 1]), Path(command[rnn_index]).parent


def rasr_path_from_returnn_config(config_path):
    """Find the forward-step's tk.Path literal without loading models."""
    tree = ast.parse(config_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "recog_rasr_config_file":
                value = keyword.value
                if isinstance(value, ast.Call) and value.args:
                    return Path(ast.literal_eval(value.args[0]))
    raise ValueError(f"Cannot find a literal recog_rasr_config_file in {config_path}")


def read_rasr_config(path):
    config = configparser.ConfigParser(interpolation=None, strict=False)
    config.optionxform = str
    with path.open() as stream:
        config.read_file(stream)
    if config.get("lib-rasr.search-algorithm", "type") != "tree-timesync-beam-search":
        raise ValueError("This probe requires a tree-timesync-beam-search configuration")
    return config


def joint_l2_settings(config):
    section = "lib-rasr.label-scorer.recognition.memristor"
    values = config.get(section, "joint-l2-converter-hardware-settings", fallback=None)
    if values is None:
        values = config.get(section, "converter-hardware-settings")
    parts = values.split()
    if len(parts) != 5:
        raise ValueError("Expected five joint-L2 converter hardware settings")
    return {
        "input_bits": int(parts[0]), "output_precision_bits": int(parts[1]),
        "output_range_bits": int(parts[2]), "hardware_input_vmax": float(parts[3]),
        "hardware_output_current_scaling": float(parts[4]),
    }


def require_joint_l2_analysis_settings(config):
    settings = joint_l2_settings(config)
    if (settings["output_precision_bits"], settings["output_range_bits"]) != (2, 6):
        raise ValueError("This analysis requires joint L2 ADC 2prec/6range; use the matching recognition job/probe")
    return settings


def config_without_decoder(config_path):
    """Load the saved Python config without its eager forward_step construction.

    Removing this assignment in memory avoids loading a second scorer and
    leaves the original config file and recognition pipeline untouched.
    """
    tree = ast.parse(config_path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "forward_step" for target in node.targets)
    )]
    return "#!rnn.py\n" + ast.unparse(tree)


def read_lexicon(path):
    """Map all orthographic lemma names to their possible pronunciation strings."""
    pronunciations = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != "lemma":
                continue
            variants = {tuple((phon.text or "").split()) for phon in element.findall("phon")}
            for orth in element.findall("orth"):
                name = (orth.text or "").strip()
                if name:
                    pronunciations.setdefault(name, set()).update(variants)
            element.clear()
    return pronunciations


def infer_token_count(items, pronunciations, *, encoder_frames):
    """Infer counts from lemma pronunciations; never claim exported token paths.

    Valid only for complete RNA paths with one frame per blank/label transition.
    The caller disables sentence-end fallback and rejects CTC tree topologies.
    Multiple pronunciations with the same length still identify the count,
    although they do not identify the actual model-token sequence.
    """
    details = []
    total = 0
    identifiable = bool(items)
    for item in items:
        variants = pronunciations.get(item["lemma"], set())
        lengths = sorted({sum(token != "<blank>" for token in variant) for variant in variants})
        details.append({
            "lemma": item["lemma"], "pronunciation_variants": len(variants),
            "possible_nonblank_token_counts": lengths,
        })
        if len(lengths) != 1:
            identifiable = False
        else:
            total += lengths[0]
    if total > encoder_frames:
        identifiable = False
    return {
        "token_count_source": "inferred from lexicon; not exported by LibRASR",
        "token_count_identifiable": identifiable,
        "nonblank_token_count": total if identifiable else None,
        "blank_count": encoder_frames - total if identifiable else None,
        "lemma_token_counts": details,
    }


def word_errors(reference, hypothesis):
    """Diagnostic word edit distance; final experiments should use Sclite."""
    ref, hyp = reference.split(), hypothesis.split()
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, predicted in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (word != predicted)))
        previous = current
    return previous[-1]


def traceback_record(traceback, reference, frames, pronunciations, transcription_fn):
    items = [{
        "lemma": item.lemma,
        "am_score": float(item.am_score), "lm_score": float(item.lm_score),
        "start_time": int(item.start_time), "end_time": int(item.end_time),
        "confidence_score": (float(item.confidence_score)
                             if item.confidence_score is not None else None),
    } for item in traceback]
    transcription = transcription_fn(traceback)
    score = items[-1]["am_score"] + items[-1]["lm_score"] if items else None
    if score is not None and not math.isfinite(score):
        score = None
    result = {
        "transcription": transcription,
        "word_errors": word_errors(reference, transcription),
        "traceback_terminal_cost": score,
        "score_source": "last returned item's cumulative am_score + lm_score",
        "traceback": items,
        **infer_token_count(items, pronunciations, encoder_frames=frames),
    }
    # Empty or invalid tracebacks cannot provide a usable complete-path slope.
    if score is None:
        result.update(token_count_identifiable=False, nonblank_token_count=None, blank_count=None)
    return result


def prepare_effective_config(original, output_dir, blank_penalty):
    """Copy config, redirect logs, and require complete paths for count inference."""
    config = read_rasr_config(original)
    section = "lib-rasr.search-algorithm"
    if config.get(section, "tree-builder-type", fallback="") != "rna":
        raise ValueError("Token-count inference here requires the RNA tree builder")
    if config.getboolean(section, "collapse-repeated-labels", fallback=False):
        raise ValueError("Token-count inference does not support collapsed CTC repeats")
    config.set(section, "sentence-end-fall-back", "no")
    if not config.has_section("*"):
        config.add_section("*")
    for kind in ("log", "error", "warning", "statistics"):
        config.set("*", f"{kind}.channel", str(output_dir / "rasr.probe.log"))
    scorer = "lib-rasr.label-scorer"
    if not config.has_section(scorer):
        raise ValueError("First probe supports the single label-scorer configuration")
    config.set(scorer, "presoftmax-blank-penalty", "no")
    recognition = f"{scorer}.recognition"
    config.set(recognition, "device", "cuda")
    if blank_penalty is not None:
        config.set(recognition, "blank-penalty", str(blank_penalty))
    path = output_dir / "rasr.probe.config"
    with path.open("w") as stream:
        config.write(stream)
    return path, config


def run_probe(args, config_path, returnn_root, rasr_path, pronunciations):
    sys.path.insert(0, str(returnn_root))
    sys.path.insert(0, str(DEFAULT_SISYPHUS))
    sys.path.insert(0, str(SETUP))
    import numpy as np
    import torch
    from returnn.config import Config, set_global_config
    from returnn.datasets import init_dataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this probe in the recognition container on a GPU node.")
    config = Config()
    set_global_config(config)
    config.load_file(io.StringIO(config_without_decoder(config_path)))

    from recipe.model_pipelines.common.recog import _traceback_to_transcription
    from recipe.model_pipelines.common.pyscorers import register_pyscorers
    from librasr import Configuration, SearchAlgorithm

    effective_path, effective = prepare_effective_config(rasr_path, args.output_dir, args.blank_penalty)
    register_pyscorers()
    rasr_config = Configuration()
    rasr_config.set_from_file(str(effective_path))
    decoder = SearchAlgorithm(config=rasr_config)
    if not hasattr(decoder, "recognize_segment_n_best"):
        raise RuntimeError("The loaded LibRASR does not expose recognize_segment_n_best")

    checkpoint_path = config.typed_value("load")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch, step = checkpoint.get("epoch", 1), checkpoint.get("step", 0)
    model = config.typed_value("get_model")(epoch=epoch, step=step)
    missing, unexpected = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    if missing:
        raise RuntimeError(f"Encoder checkpoint is missing parameters: {missing}")
    # The shared checkpoint also contains prediction/joint-network parameters,
    # so unexpected encoder keys are normal and recorded for reproducibility.
    del checkpoint
    model.eval().to("cuda")

    data_options = dict(config.typed_value("forward_data"))
    data_options["use_cache_manager"] = False
    if args.whole_corpus:
        data_options.pop("seq_list_filter_file", None)
    dataset = init_dataset(data_options)
    dataset.init_seq_order(epoch=1, seq_list=args.seq_tag or None)
    anchor = float(effective.get("lib-rasr.label-scorer.recognition", "blank-penalty"))
    scale = float(effective.get("lib-rasr.label-scorer.recognition", "scale", fallback="1.0"))
    summary = {
        "returnn_config": str(config_path), "returnn_root": str(returnn_root),
        "original_rasr_config": str(rasr_path), "effective_rasr_config": str(effective_path),
        "checkpoint": checkpoint_path, "anchor_penalty": anchor, "scorer_scale": scale,
        "requested_nbest": args.nbest,
        "noise_handling": "Active readout noise; no seed resets or RNG replay",
        "joint_l2_converter_hardware_settings": joint_l2_settings(effective),
        "local_penalty_estimate": ESTIMATED_BLANK_PENALTY,
        "whole_corpus": args.whole_corpus,
        "sentence_end_fallback": False, "presoftmax_blank_penalty": False,
        "encoder_unexpected_checkpoint_keys": unexpected,
        "lexicon": effective.get("lib-rasr", "lexicon.file"),
        "note": "Traceback totals/counts are probes, not yet verified authoritative reranking metadata.",
        "utterances": [],
    }
    jsonl_path = args.output_dir / "nbest.jsonl"
    with jsonl_path.open("w") as stream, torch.no_grad():
        for index in range(args.num_seqs):
            if not dataset.is_less_than_num_seqs(index):
                break
            dataset.load_seqs(index, index + 1)
            tag = dataset.get_tag(index)
            raw = np.asarray(dataset.get_data(index, "raw"))
            if raw.size == 1:
                text = raw.item()
                reference = text.decode("utf-8") if isinstance(text, bytes) else str(text)
            else:
                reference = " ".join(map(str, raw.flat))
            audio = np.asarray(dataset.get_data(index, "data"), dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[:, None]
            if audio.ndim != 2 or audio.shape[1] != 1:
                raise ValueError(f"Unexpected audio shape {audio.shape} for {tag}")
            samples = torch.from_numpy(audio).unsqueeze(0).to("cuda")
            lengths = torch.tensor([len(audio)], dtype=torch.int32, device="cuda")
            start = perf_counter()
            states, state_lengths = model(samples, lengths)
            frames = int(state_lengths[0].item())
            features = np.ascontiguousarray(states[0, :frames].detach().cpu().numpy(), dtype=np.float32)
            encoder_seconds = perf_counter() - start
            start = perf_counter()
            tracebacks = decoder.recognize_segment_n_best(features=features, n=args.nbest)
            search_seconds = perf_counter() - start
            direct = traceback_record(decoder.get_current_best_traceback(), reference, frames,
                                      pronunciations, _traceback_to_transcription)
            candidates = [traceback_record(tb, reference, frames, pronunciations,
                                          _traceback_to_transcription) for tb in tracebacks]
            for rank, candidate in enumerate(candidates, 1):
                candidate["rank"] = rank
                count, cost = candidate["blank_count"], candidate["traceback_terminal_cost"]
                candidate["tentative_penalty_coefficient"] = scale * count if count is not None else None
                candidate["tentative_zero_penalty_cost"] = (
                    cost - anchor * scale * count if count is not None and cost is not None else None
                )
            top = candidates[0] if candidates else None
            gap = (top["traceback_terminal_cost"] - direct["traceback_terminal_cost"]
                   if top and top["traceback_terminal_cost"] is not None
                   and direct["traceback_terminal_cost"] is not None else None)
            record = {
                "seq_tag": tag, "reference": reference, "reference_words": len(reference.split()),
                "encoder_frames": frames, "encoder_seconds": encoder_seconds,
                "search_seconds": search_seconds, "num_candidates": len(candidates),
                "direct_best": direct, "top_nbest_matches_direct_transcript": (
                    top["transcription"] == direct["transcription"] if top else False
                ), "top_nbest_minus_direct_cost": gap, "candidates": candidates,
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            summary["utterances"].append({key: value for key, value in record.items()
                                          if key not in ("candidates", "direct_best")})
            print(f"\n{tag}\nReference: {reference}", flush=True)
            print(f"Frames={frames}; candidates={len(candidates)}; top/direct cost gap={gap}", flush=True)
            print("rank | terminal cost | errors | inferred tokens/blanks | transcript", flush=True)
            for candidate in candidates:
                print(f"{candidate['rank']:4} | {candidate['traceback_terminal_cost']} | "
                      f"{candidate['word_errors']} | {candidate['nonblank_token_count']}/"
                      f"{candidate['blank_count']} | {candidate['transcription']}", flush=True)
            if args.save_encoder:
                np.save(args.output_dir / f"encoder_{index:03d}.npy", features)
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"\nProbe results: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recog-job", type=Path, default=DEFAULT_JOB,
                        help="Existing recognition job directory containing work/rnn.sh")
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--seq-tag", action="append", help="Optional exact tag; repeat to select several")
    parser.add_argument("--nbest", type=int, default=20)
    parser.add_argument("--blank-penalty", type=float, default=ESTIMATED_BLANK_PENALTY,
                        help="Additive anchor penalty (default: 1.44 for joint L2 ADC 2prec/6range)")
    parser.add_argument("--whole-corpus", action="store_true",
                        help="Remove a source job's shard segment filter for this small diagnostic")
    parser.add_argument("--save-encoder", action="store_true")
    parser.add_argument("--inspect-only", action="store_true",
                        help="CPU preflight: inspect saved config and lexicon without loading models")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: ADC-specific probe root / bp<blank-penalty>")
    args = parser.parse_args()
    if args.num_seqs < 1 or args.nbest < 1:
        parser.error("num-seqs and nbest must be positive")
    if args.blank_penalty is not None and not math.isfinite(args.blank_penalty):
        parser.error("blank-penalty must be finite")
    config_path, returnn_root = job_paths(args.recog_job.resolve())
    rasr_path = rasr_path_from_returnn_config(config_path)
    rasr = read_rasr_config(rasr_path)
    hardware = require_joint_l2_analysis_settings(rasr)
    lexicon_path = Path(rasr.get("lib-rasr", "lexicon.file"))
    pronunciations = read_lexicon(lexicon_path)
    counts = Counter(len(variants) for variants in pronunciations.values())
    ambiguous_lengths = sum(len({len(p) for p in variants}) > 1 for variants in pronunciations.values())
    print(f"RETURNN config: {config_path}\nRETURNN root: {returnn_root}\nRASR config: {rasr_path}\n"
          f"Lexicon: {lexicon_path}\nLemma names: {len(pronunciations)}\n"
          f"Pronunciation-variant histogram: {dict(sorted(counts.items()))}\n"
          f"Lemma names with varying pronunciation lengths: {ambiguous_lengths}", flush=True)
    print(f"Joint L2 ADC: {hardware}; probe anchor: {args.blank_penalty}", flush=True)
    if args.inspect_only:
        return
    args.output_dir = (args.output_dir or PROBE_ROOT / f"bp{args.blank_penalty:g}").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_probe(args, config_path, returnn_root, rasr_path, pronunciations)


if __name__ == "__main__":
    main()
