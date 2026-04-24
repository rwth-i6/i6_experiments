import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS
import argparse
import json
import os
import random
from typing import Any
from datasets import Features, Value, Sequence, Audio, Dataset, load_from_disk

print("Imports successful", flush=True)

SPEAKER_ALIAS = {}

dialogue_features = Features(
    {
        # Unique identifier for the conversation
        "id": Value("string"),
        # The full-length audio tracks, one per speaker
        "speaker_audio": Sequence(
            {
                "speaker": Value("string"),
                "audio": Audio(),  # HF converts your file paths to audio bytes here
            }
        ),
        # The chronological array of conversation turns
        "turns": Sequence(
            {
                "speaker": Value("string"),
                "start_time": Value("float32"),  # float32 is perfect for timestamps
                "text": Value("string"),
            }
        ),
    }
)


def available_speakers(speaker_dir: str) -> list[str]:
    """List available speakers based on the files in the speaker directory."""
    speakers = []
    for filename in os.listdir(speaker_dir):
        if filename.endswith(".wav"):
            speaker_name = os.path.splitext(filename)[0]
            speakers.append(speaker_name)
    return speakers


def speaker_name_to_path(speaker_name: str, speaker_dir: str) -> str:
    speaker_name = SPEAKER_ALIAS.get(speaker_name, speaker_name)
    based = os.path.basename(speaker_name)
    if based.startswith("rng_"):
        al = available_speakers(
            os.path.join(speaker_dir, os.path.dirname(speaker_name))
        )
        # special token for random
        if len(al) == 0:
            raise ValueError(
                f"No available speakers in directory {os.path.dirname(speaker_name)} for random selection"
            )
        chosen = random.choice(al)
        print(f"Randomly chose speaker {chosen} from {al} for {speaker_name}")
        return os.path.join(speaker_dir, os.path.dirname(speaker_name), chosen + ".wav")

    return os.path.join(speaker_dir, speaker_name + ".wav")


def gen_conversation(
    model, dialogue, device, speaker_dir, silence_length_sampler
) -> tuple[dict[str, torch.Tensor], list[Any]]:

    print("Generating conversation...")

    speaker_to_path = {}
    paths_used = set()

    # TODO get all speakers
    speakers = set(
        (turn["speaker"], turn.get("exaggeration", 0.5)) for turn in dialogue
    )
    speak_map = {}
    for s in speakers:
        if s[0] not in speaker_to_path:
            for _ in range(5):  # try a few times to resolve random speaker if needed
                p = speaker_name_to_path(s[0], speaker_dir)
                if p not in paths_used:
                    break
            speaker_to_path[s[0]] = p
            paths_used.add(p)

        model.prepare_conditionals(
            speaker_to_path[s[0]], exaggeration=s[1], norm_loudness=True
        )
        speak_map[s] = model.conds

    # We synthesize per-speaker streams, then align them based on the sampled
    # silence lengths between consecutive turns.
    #
    # Let each speaker stream be: previous utterances + (silence/overlap) + next utterance.
    # - For silence_length > 0: insert silence after each turn for other speakers.
    # - For silence_length < 0: overlap means the *other* speaker starts their next
    #   utterance earlier by -silence_length seconds.
    #
    # To implement this robustly for all configurations, we build explicit utterance
    # start times and then render each speaker's timeline.

    # First pass: precompute each utterance wav and each utterance start time.
    utterances = []  # list of dict: speaker, wav, start_sample
    t_samples = 0
    for idx, turn in enumerate(dialogue):
        print(f"Synthesizing {turn['speaker']}'s line...")

        if "pre_silence" in turn:
            assert turn["pre_silence"] >= 0, "Pre-silence must be non-negative"
            print(
                f"Adding pre-silence of {turn['pre_silence']} seconds for {turn['speaker']}"
            )
            t_samples += int(model.sr * turn["pre_silence"])

        model.conds = speak_map[(turn["speaker"], turn.get("exaggeration", 0.5))]
        wav = model.generate(
            text=turn["text"],
            audio_prompt_path=None,  # we hack conditionals in directly
            exaggeration=turn.get("exaggeration", 0.5),
            cfg_weight=turn.get("cfg", 0.5),
        )
        model.conds = None

        wav = wav.to(device)
        start = t_samples
        utterances.append(
            {
                "speaker": turn["speaker"],
                "wav": wav,
                "start": start,
                "text": turn["text"],
            }
        )

        t_samples += wav.shape[-1]
        # Advance global time by the silence length between this and next turn.
        # Negative silence means the next turn starts earlier (overlap), so we
        # subtract samples but never let time go below current utterance start.
        silence_seconds = float(silence_length_sampler())
        delta = int(model.sr * silence_seconds)
        if silence_seconds >= 0:
            t_samples += delta
        else:
            t_samples = max(start, t_samples + delta)

    # Second pass: render per-speaker timelines using utterance start times.
    # Determine total length.
    end_samples = 0
    for u in utterances:
        end_samples = max(end_samples, u["start"] + u["wav"].shape[-1])

    rendered = {}
    for s, exagg in speakers:
        # single channel tensor
        rendered[s] = torch.zeros(
            1, end_samples, device=device, dtype=utterances[0]["wav"].dtype
        )

    for u in utterances:
        s = u["speaker"]
        st = u["start"]
        en = st + u["wav"].shape[-1]
        rendered[s][0, st:en] += u["wav"][0]

    return rendered, utterances


def process_dialogue(
    model, dialogue, device, speaker_dir, silence_length_sampler, out_dir, diag_id
):

    assert isinstance(dialogue, list), (
        "Each line in the input text file should be a json array that contains dialogues"
    )
    assert len(dialogue) > 0, "Each dialogue should contain at least one turn"
    assert all("speaker" in turn and "text" in turn for turn in dialogue), (
        "Each turn in the dialogue should contain 'speaker', 'text' fields"
    )

    output_dir = os.path.join(out_dir, f"dialogue_{diag_id}")
    os.makedirs(output_dir, exist_ok=True)

    audios_per_speaker, utterances = gen_conversation(
        model, dialogue, device, speaker_dir, silence_length_sampler
    )

    for speaker, audio in audios_per_speaker.items():
        torchaudio.save(f"{output_dir}/{speaker}.wav", audio.cpu(), model.sr)
        print(f"Saved {speaker}'s audio to {output_dir}/{speaker}.wav")

    # save utterances without audio as metadata.json
    metadata = []
    for u in utterances:
        metadata.append(
            {
                "speaker": u["speaker"],
                "text": u["text"],
                "start_time": u["start"] / model.sr,
            }
        )
    with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


# chatterbox inference gets its own venv, so we let the job execute this file directly
@torch.inference_mode()
def main():
    global SPEAKER_ALIAS

    try:
        import torchcodec
    except:
        print(
            "torchcodec could not be imported. If on slurm cluster, run ml load FFmpeg"
        )
    import torchcodec
    # fail early. If this fails, its likely that the user doesn't have ffmpeg installed, or its broken somehow. use InstallFFmpeg job here

    parser = argparse.ArgumentParser(description="Run Chatterbox TTS Inference")
    parser.add_argument(
        "--in_jsonl",
        type=str,
        required=False,
        help="Path to input JSONL file. Input JSONL file is a line separated list of json arrays that contain dialogues",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        help="Limit the number of dialogues to process. Useful for testing.",
    )
    parser.add_argument(
        "--in_hf",
        type=str,
        required=False,
        help="Path to input in Hugging Face dataset format. If provided, will ignore --in_jsonl and read dialogues from the Hugging Face dataset",
    )
    parser.add_argument(
        "--in_hf_shard",
        type=int,
        required=False,
        help="If --in_hf is provided and the dataset is sharded, provide the shard index to read from. If not provided, will read from the first shard (index 0)",
    )
    parser.add_argument(
        "--in_hf_num_shards",
        type=int,
        required=False,
        help="If --in_hf is provided and the dataset is sharded, provide the total number of shards. If not provided, will assume the dataset is not sharded",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save output audio. Each line in the input text file will be saved as a directory that contains the audio channel for every speaker",
    )
    parser.add_argument(
        "--out_hf",
        type=str,
        required=False,
        help="Path to save output in Hugging Face dataset format. If not provided, will not save in Hugging Face format",
    )
    parser.add_argument(
        "--speaker_directory",
        type=str,
        required=True,
        help="Directory that contains reference audio for each speaker.",
    )
    parser.add_argument("--speaker_alias", type=str, default="{}", help="")
    args = parser.parse_args()

    SPEAKER_ALIAS = json.loads(args.speaker_alias)

    random.seed(42)  # For reproducibility

    # TODO figure out good way to sample silence
    def silence_length_sampler():
        val = random.gauss(0.2, 0.4)
        while val < -0.3 or val > 0.6:  # TODO vibe based, make this better later
            val = random.gauss(0.2, 0.4)
        return val

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    model = ChatterboxTurboTTS.from_pretrained(device=device)

    # TODO there are some projects that try to make chatterbox faster
    # https://github.com/rsxdalv/chatterbox/blob/faster/src/chatterbox/tts.py
    # or chatterbox-vllm
    # but I am not certain that they have no regressions. Just use shards for now for "speedup"
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")
    # model.generate = torch.compile(model.generate, dynamic=True)
    model.t3.inference_turbo = torch.compile(
        model.t3.inference_turbo,
        dynamic=True,  # fullgraph=True, backend="cudagraphs"
    )

    assert args.in_jsonl is not None or args.in_hf is not None, (
        "Either --in_jsonl or --in_hf must be provided"
    )
    assert not (args.in_jsonl is not None and args.in_hf is not None), (
        "Cannot provide both --in_jsonl and --in_hf. Please choose one."
    )

    last_diag_id = 0
    if args.in_hf is not None:
        assert args.limit is None, (
            "Limiting number of dialogues is not supported when reading from Hugging Face dataset"
        )
        print(f"Loading dialogues from Hugging Face dataset at {args.in_hf}...")
        dataset = load_from_disk(args.in_hf)
        print("Dataset loaded successfully!")
        if args.in_hf_shard is not None and args.in_hf_num_shards is not None:
            dataset = dataset.shard(
                num_shards=args.in_hf_num_shards, index=args.in_hf_shard
            )
            print(f"Using shard {args.in_hf_shard} of {args.in_hf_num_shards}")
        for i, example in enumerate(dataset):
            print(f"Processing dialogue {i}...")
            dialogue = json.loads(example["dialogue"])
            process_dialogue(
                model,
                dialogue,
                device,
                args.speaker_directory,
                silence_length_sampler,
                args.out_dir,
                i,
            )
            last_diag_id = i
    else:
        with open(args.in_jsonl, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.limit is not None and i >= args.limit:
                    print(f"Reached limit of {args.limit} dialogues. Stopping.")
                    break
                print(f"Processing dialogue {i}...")

                # each line is a json array that contains dialogues
                dialogue = json.loads(line)
                process_dialogue(
                    model,
                    dialogue,
                    device,
                    args.speaker_directory,
                    silence_length_sampler,
                    args.out_dir,
                    i,
                )
                last_diag_id = i
    print("All dialogues processed successfully!")

    if args.out_hf is not None:
        print(f"Saving output in Hugging Face dataset format to {args.out_hf}...")

        def gen():
            for i in range(last_diag_id + 1):
                dialogue_dir = os.path.join(args.out_dir, f"dialogue_{i}")
                metadata_path = os.path.join(dialogue_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    raise FileNotFoundError(f"metadata.json not found for dialogue {i}")
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                speaker_audio = {}
                for item in metadata:
                    speaker = item["speaker"]
                    if speaker in speaker_audio:
                        continue
                    audio_path = os.path.join(dialogue_dir, f"{speaker}.wav")
                    if not os.path.exists(audio_path):
                        raise FileNotFoundError(
                            f"Audio file for speaker {speaker} not found in dialogue {i}"
                        )
                    speaker_audio[speaker] = audio_path
                yield {
                    "id": f"dialogue_{i}",
                    "speaker_audio": {
                        "speaker": list(speaker_audio.keys()),
                        "audio": list(speaker_audio.values()),
                    },
                    "turns": {
                        "speaker": [turn["speaker"] for turn in metadata],
                        "start_time": [turn["start_time"] for turn in metadata],
                        "text": [turn["text"] for turn in metadata],
                    },
                }

        dataset = Dataset.from_generator(
            gen,
            features=dialogue_features,
        )
        dataset.save_to_disk(args.out_hf)
        print(f"Dataset saved successfully to {args.out_hf}!")


if __name__ == "__main__":
    main()
