"""
Adapted from here: https://github.com/huggingface/open_asr_leaderboard/blob/5c03c1f85a84ab7a991dcc1b3f14905ec6d632c9/nemo_asr/run_eval.py
"""
from __future__ import annotations

import argparse

import os
import sys
import shutil

from typing import Any, Tuple, List

import tree

import torch

import soundfile

from tqdm import tqdm
from normalizer import data_utils

from datasets import load_from_disk

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.aed_multitask_models import MultiTaskTranscriptionConfig
from nemo.collections.asr.parts.mixins.transcription import GenericTranscriptionType
from nemo.collections.common.parts.transformer_utils import mask_padded_tokens

sys.path.insert(0, "/u/zeineldeen/setups/ubuntu_22_setups/2024-06-07--canary-aed/recipe")

from i6_experiments.users.zeyer.decoding.beam_search_torch.interface import (
    LabelScorerIntf,
    StateObjIgnored,
)
from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_v5 import (
    beam_search_v5,
    BeamSearchOptsV5,
)

DATA_CACHE_DIR = "/var/tmp/audio_cache"


def compute_wer(predictions, references):
    from jiwer import compute_measures

    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    return incorrect / total


def dataset_iterator(dataset):
    for i, item in enumerate(dataset):
        yield {
            **item["audio"],
            "reference": item["norm_text"],
            "audio_filename": f"file_{i}",
            "sample_rate": 16_000,
            "sample_id": i,
        }


def write_audio(buffer, cache_prefix) -> list:
    cache_dir = os.path.join(DATA_CACHE_DIR, cache_prefix)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.makedirs(cache_dir)

    data_paths = []
    for idx, data in enumerate(buffer):
        fn = os.path.basename(data["audio_filename"])
        fn = os.path.splitext(fn)[0]
        path = os.path.join(cache_dir, f"{idx}_{fn}.wav")
        data_paths.append(path)

        soundfile.write(path, data["array"], samplerate=data["sample_rate"])

    return data_paths


def pack_results(results: list, buffer, transcriptions):
    for sample, transcript in zip(buffer, transcriptions):
        result = {"reference": sample["reference"], "pred_text": transcript}
        results.append(result)
    return results


def get_our_canary_label_scorer(
    model: ASRModel, enc: torch.Tensor, enc_input_mask: torch.Tensor, pad_id: int, bos_prefix_seq: torch.Tensor
) -> LabelScorerIntf:
    """
    Creates a CanaryLabelScorer object that is used in the beam search implementation.

    :param model: nemo ASRModel object
    :param enc: [B,T]
    :param enc_input_mask: [B,T]
    :param pad_id:
    :param bos_prefix_seq:
    """

    trafo_decoder_module = model.transf_decoder  # type: torch.nn.Module
    log_softmax_module = model.log_softmax  # type: torch.nn.Module

    class CanaryLabelScorer(LabelScorerIntf):
        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            return {
                "step": StateObjIgnored(0),
                # "prefix_input": torch.tile(
                #     torch.tensor(bos_prefix_seq, device=device)[None, :], [batch_size, 1]
                # ),  # [Batch,InputSeqLen]
                "prefix_input": bos_prefix_seq,  # [Batch,InputSeqLen]
                "model_state": None,
            }

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """
            :param prev_state: state of the scorer (decoder). any nested structure.
                all tensors are expected to have shape [Batch, Beam, ...].
            :param prev_label: shape [Batch, Beam] -> index in [0...Label-1]
            :return: (scores, state).
                scores: shape [Batch, Beam, Label], log-prob-like scores.
                    Broadcasting is allowed for any of the dims (e.g. think of :class:`LengthRewardScorer`).
                state: all tensors are expected to have shape [Batch, Beam, ...].
            """
            batch_size, beam_size = prev_label.shape

            # Convert all [batch,beam,...] tensors to [batch*beam,...].
            def _map(x):
                if x is None:
                    return None
                assert isinstance(x, torch.Tensor) and x.shape[:2] == (batch_size, beam_size)
                return x.flatten(0, 1)

            prev_model_state = tree.map_structure(_map, prev_state["model_state"])

            input = prev_state["prefix_input"]  # [batch*beam,in_seq_len] or None
            if input is None:
                input = prev_label.flatten(0, 1)[:, None]  # [batch*beam,1]

            # start_pos is used for the positional encoding added to the embeddings
            dec_embed = trafo_decoder_module.embedding.forward(
                input, start_pos=prev_state["step"].content
            )  # [batch*beam,in_seq_len|1,D]
            dec_input_mask = mask_padded_tokens(input, pad_id=pad_id).float()

            _, enc_len, enc_dim = enc.size()
            enc_input_mask_ = enc_input_mask.unsqueeze(1).expand(-1, beam_size, -1).contiguous().view(-1, enc_len)
            enc_ = enc.unsqueeze(1).expand(-1, beam_size, -1, -1).contiguous().view(-1, enc_len, enc_dim)

            # decoder_mems_list is a list of size num_layers that cache output activations of shape
            # [batch*beam,history,D]
            decoder_mems_list = trafo_decoder_module.decoder.forward(
                decoder_states=dec_embed,
                decoder_mask=dec_input_mask,
                encoder_states=enc_,
                encoder_mask=enc_input_mask_,
                decoder_mems_list=prev_model_state,
                return_mems=True,
                return_mems_as_list=True,
            )

            # decoder_mems_list[-1][:, -1:] is the output of the last layer at the current decoding step position
            log_probs = log_softmax_module.forward(hidden_states=decoder_mems_list[-1][:, -1:])  # [batch*beam,1,V]

            def _map(x):
                assert isinstance(x, torch.Tensor) and x.shape[:1] == (batch_size * beam_size,)
                return x.unflatten(0, (batch_size, beam_size))

            log_probs = log_probs.squeeze(1)  # [batch*beam,V]
            log_probs = _map(log_probs)  # [batch,beam,V]

            def _map(x):
                assert isinstance(x, torch.Tensor) and x.shape[:1] == (batch_size * beam_size,)
                return x.unflatten(0, (batch_size, beam_size))

            decoder_mems_list = tree.map_structure(_map, decoder_mems_list)

            return log_probs, {
                "step": StateObjIgnored(prev_state["step"].content + input.size(1)),
                "prefix_input": None,
                "model_state": decoder_mems_list,
            }

    return CanaryLabelScorer()


def _transcribe_output_processing_our_beam_search(
    outputs, trcfg: MultiTaskTranscriptionConfig
) -> GenericTranscriptionType:
    # outputs are returned from `_transcribe_forward` function call
    enc_states = outputs.pop("encoder_states")
    enc_lens = outputs.pop("encoded_lengths")
    enc_mask = outputs.pop("encoder_mask")
    dec_input_ids = outputs.pop("decoder_input_ids")

    canary_label_scorer = get_our_canary_label_scorer(
        model=asr_model,
        enc=enc_states,
        enc_input_mask=enc_mask,
        pad_id=asr_model.tokenizer.pad_id,
        bos_prefix_seq=dec_input_ids,  # [3, 4, 8, 4, 10]
    )

    seq_targets, _, out_seq_len = beam_search_v5(
        canary_label_scorer,
        batch_size=enc_states.size(0),
        max_seq_len=enc_lens,
        device=enc_states.device,
        opts=beam_search_v5_opts,
    )  # [B,Beam,L]

    best_hyps = []
    for i in range(seq_targets.shape[0]):
        best_hyp_int = seq_targets[i, 0, : out_seq_len[i, 0]].tolist()
        best_hyp_text = asr_model.tokenizer.ids_to_text(best_hyp_int)
        best_hyps.append(asr_model.decoding.strip_special_tokens(best_hyp_text))
    return best_hyps


def buffer_audio_and_transcribe(
    model: ASRModel, dataset, batch_size: int, pnc: bool, cache_prefix: str, verbose: bool = True
):
    buffer = []
    results = []
    for sample in tqdm(dataset_iterator(dataset), desc="Evaluating: Sample id", unit="", disable=not verbose):
        buffer.append(sample)

        if len(buffer) == batch_size:
            filepaths = write_audio(buffer, cache_prefix)

            if pnc is not None:
                transcriptions = model.transcribe(
                    filepaths, batch_size=batch_size, pnc=False, verbose=False, num_workers=4
                )
            else:
                transcriptions = model.transcribe(filepaths, batch_size=batch_size, verbose=False, num_workers=4)

            # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
            if type(transcriptions) == tuple and len(transcriptions) == 2:
                transcriptions = transcriptions[0]
            results = pack_results(results, buffer, transcriptions)
            buffer.clear()

    if len(buffer) > 0:
        filepaths = write_audio(buffer, cache_prefix)
        if pnc is not None:
            transcriptions = model.transcribe(filepaths, batch_size=batch_size, pnc=False, verbose=False)
        else:
            transcriptions = model.transcribe(filepaths, batch_size=batch_size, verbose=False)
        # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
        if type(transcriptions) == tuple and len(transcriptions) == 2:
            transcriptions = transcriptions[0]
        results = pack_results(results, buffer, transcriptions)
        buffer.clear()

    # Delete temp cache dir
    if os.path.exists(DATA_CACHE_DIR):
        shutil.rmtree(DATA_CACHE_DIR)

    return results


def main(args):
    import better_exchook

    better_exchook.install()

    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    global asr_model
    asr_model = ASRModel.restore_from(args.model_path, map_location=device)
    asr_model.freeze()

    global beam_search_v5_opts
    beam_search_v5_opts = BeamSearchOptsV5(
        beam_size=args.beam_size,
        bos_label=asr_model.tokenizer.bos_id,
        eos_label=asr_model.tokenizer.eos_id,
        num_labels=len(asr_model.tokenizer.vocab),
        length_normalization_exponent=1,
        pruning_threshold=args.pruning_threshold,
        adaptive_pruning=args.adaptive_pruning,
    )

    # hook our beam search implementation
    asr_model._transcribe_output_processing = _transcribe_output_processing_our_beam_search

    dataset = load_from_disk(args.dataset_path)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples !")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)

    predictions = []
    references = []

    # run streamed inference
    cache_prefix = (
        f"{args.model_id.replace('/', '-')}-{args.dataset_path.replace('/', '')}-"
        f"{args.dataset.replace('/', '-')}-{args.split}"
    )
    results = buffer_audio_and_transcribe(asr_model, dataset, args.batch_size, args.pnc, cache_prefix, verbose=True)
    for sample in results:
        predictions.append(data_utils.normalizer(sample["pred_text"]))
        references.append(sample["reference"])

    # Write manifest results to args.manifest_path. This required modification in normalizer/eval_utils.py script
    manifest_path = data_utils.write_manifest(
        args.manifest_path, references, predictions, args.model_id, args.dataset_path, args.dataset, args.split
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = compute_wer(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer, "%")

    if args.wer_out_path:
        with open(args.wer_out_path, "w") as f:
            f.write(f"{wer}\n")
        print(f"Wrote WER (%) to {args.wer_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="Model ID.")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to nemo model.",
    )

    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split.")

    parser.add_argument("--manifest_path", type=str, required=True, help="Path to save the search output.")

    parser.add_argument("--wer_out_path", type=str, default=None, help="Path to save the WER output.")

    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--pruning_threshold", type=float, default=0.0)
    parser.add_argument("--adaptive_pruning", type=bool, default=False)

    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--pnc",
        type=bool,
        default=None,
        help="flag to indicate inferene in pnc mode for models that support punctuation and capitalization",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=True)

    main(args)
