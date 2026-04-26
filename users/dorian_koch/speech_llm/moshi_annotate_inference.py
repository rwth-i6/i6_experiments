import sphn
import json
from pathlib import Path
from datasets import load_from_disk, Dataset
import argparse
import os
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
import functools

# ln -s ../projects/moshi-finetune moshi_finetune
from moshi_finetune.annotate import (
    init_logging,
    Params,
    logger,
    torch,
    whisper,
    load_audio_paths,
    process_one,
)


def extract_hf_to_files_multproc(output_dir, dataset: Dataset, num_proc):
    print(f"{len(dataset)} rows in dataset")
    print(f"sharding into {num_proc}")
    shards = [dataset.shard(num_shards=num_proc, index=i) for i in range(num_proc)]

    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        # list() is used here to consume the generator and ensure execution finishes/catches errors
        list(executor.map(functools.partial(extract_hf_to_files, output_dir), shards))


def extract_hf_to_files(output_dir, dataset: Dataset):

    num_files_written = 0
    for row in dataset:
        row_id = row["id"]

        user_audio = None
        assistant_audio = None
        sample_rate = None

        d = row["speaker_audio"]
        speaker_tracks = [dict(zip(d.keys(), values)) for values in zip(*d.values())]

        # 1. Extract audio arrays for both speakers
        for track in speaker_tracks:
            speaker_name = track["speaker"]

            # HuggingFace Audio feature decodes into a dict with 'array' and 'sampling_rate'
            audio_array = track["audio"]["array"]
            sr = track["audio"]["sampling_rate"]

            # Ensure the array is 1D (sometimes HF loads mono as shape (1, N))
            audio_array = np.squeeze(audio_array)

            if speaker_name == "user":
                user_audio = audio_array
                sample_rate = sr  # We assume both tracks have the same sampling rate
            elif speaker_name == "assistant":
                assistant_audio = audio_array
                sample_rate = sr
            else:
                raise ValueError(f"Invalid speaker name {speaker_name}")

        # Skip if for some reason a row is missing one of the speakers
        if user_audio is None or assistant_audio is None:
            print(f"Skipping row {row_id}: Missing user or assistant audio.")
            raise ValueError("some audio is missing")
            continue

        # 2. Pad arrays with zeros if they are not the exact same length
        max_len = max(len(user_audio), len(assistant_audio))

        if len(assistant_audio) < max_len:
            assistant_audio = np.pad(
                assistant_audio, (0, max_len - len(assistant_audio))
            )
            raise ValueError(
                f"Audios not equal length {len(assistant_audio)} != {len(user_audio)}"
            )
        if len(user_audio) < max_len:
            user_audio = np.pad(user_audio, (0, max_len - len(user_audio)))
            raise ValueError(
                f"Audios not equal length {len(assistant_audio)} != {len(user_audio)}"
            )

        # 3. Combine into a Stereo array
        # soundfile expects shape (frames, channels)
        # Column 0 (Left) = Assistant, Column 1 (Right) = User
        stereo_audio = np.column_stack((assistant_audio, user_audio))

        # 4. Save to WAV file
        output_filename = f"{row_id}.wav"
        output_path = os.path.join(output_dir, output_filename)
        assert not os.path.exists(output_path)

        sf.write(output_path, stereo_audio, sample_rate)
        num_files_written += 1

    print(
        f"Successfully extracted and saved {num_files_written} stereo WAV files to '{output_dir}'"
    )


# TODO "temporary" setup, just convert our hf to something the moshi-finetune code can read
def hf_to_moshi_format(out_dir: str, dataset: Dataset):

    out_dir_audios = os.path.join(out_dir, "wav-dir/")
    os.makedirs(out_dir_audios, exist_ok=True)

    extract_hf_to_files_multproc(out_dir_audios, dataset, num_proc=6)

    paths = [str(f) for f in Path(out_dir_audios).glob("*.wav")]
    durations = sphn.durations(paths)

    data_json_path = os.path.join(out_dir, "data.jsonl")

    with open(data_json_path, "w") as fobj:
        for p, d in zip(paths, durations):
            if d is None:
                continue
            json.dump({"path": p, "duration": d}, fobj)
            fobj.write("\n")
    return data_json_path


def run(
    params: "Params", shard: int = 0
):  # copy of run() in annotate.py, cuda device logic removed
    init_logging(params.verbose)
    # local_rank = dora.distrib.get_distrib_spec().local_rank
    # shard += local_rank
    # local_rank = 0
    logger.info("Hello, world, this is shard %d / %d.", shard, params.shards)
    params.shard = shard
    # torch.cuda.set_device(local_rank)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["OMP_NUM_THREADS"] = "2"

    logger.info("Loading all models.")
    device = torch.device("cuda")
    w_model = whisper.load_model(params.whisper_model, device=device)

    logger.info("Loading egs %s.", params.egs)
    paths = load_audio_paths(params.egs)
    kept_paths = paths[shard :: params.shards]
    logger.info("Processing % 8d files out of % 8d.", len(kept_paths), len(paths))
    del paths

    for idx, path in enumerate(kept_paths):
        if (idx + 1) % 100 == 0:
            logger.info("Processing % 8d / % 8d files.", idx + 1, len(kept_paths))
        out_file = path.with_suffix(".json")
        err_file = path.with_suffix(".json.err")
        if out_file.exists():
            continue
        if err_file.exists() and not params.rerun_errors:
            continue
        try:
            if path.stat().st_size < 1000:
                logger.warning("Small file detected: %s", path)
                continue
            logger.debug("Processing file %s, out file is %s", path, out_file)
            process_one(
                path,
                out_file,
                channel=0,
                language=params.lang,
                w_model=w_model,
                params=params,
            )
        except Exception as err:
            if "cuda" in repr(err).lower():
                raise
            logger.exception("Error processing %s", path)
            err_file.touch()
            continue


def main():
    parser = argparse.ArgumentParser(description="Run moshi annotate Inference")
    parser.add_argument(
        "--in_hf",
        type=str,
        required=True,
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--in_hf_shard", type=int, required=False)
    parser.add_argument("--in_hf_num_shards", type=int, required=False)
    args = parser.parse_args()
    dataset = load_from_disk(args.in_hf)
    print("Dataset loaded successfully!")
    if args.in_hf_shard is not None and args.in_hf_num_shards is not None:
        dataset = dataset.shard(
            num_shards=args.in_hf_num_shards, index=args.in_hf_shard
        )
        print(f"Using shard {args.in_hf_shard} of {args.in_hf_num_shards}")

    data_path = hf_to_moshi_format(args.out_dir, dataset)

    print("Now starting moshi annotate.py")

    params = Params(
        egs=data_path,
        verbose=False,
        lang="en",
        whisper_model="medium",
        keep_silence_in_segments=True,
        rerun_errors=False,
        shards=1,
        shard=0,
    )
    run(params)


if __name__ == "__main__":
    main()
