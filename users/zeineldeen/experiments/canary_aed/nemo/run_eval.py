"""
Adapted from here: https://github.com/huggingface/open_asr_leaderboard/blob/5c03c1f85a84ab7a991dcc1b3f14905ec6d632c9/nemo_asr/run_eval.py
"""
import argparse

import os
import sys

# sys.path.append("/u/zeineldeen/setups/ubuntu_22_setups/2024-06-07--canary-aed/recipe/open_asr_leaderboard/normalizer")

import shutil
import torch

import soundfile

from tqdm import tqdm
from normalizer import data_utils

from datasets import load_from_disk

from nemo.collections.asr.models import ASRModel

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
    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    asr_model = ASRModel.restore_from(args.model_path, map_location=device)
    asr_model.freeze()

    # TODO: how to set the num_workers?
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
