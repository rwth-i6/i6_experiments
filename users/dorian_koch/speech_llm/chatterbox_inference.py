import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS
import argparse
import json
import os
import random
from typing import Any

SAMPLE_RATE = 24000  # Chatterbox default output is 24kHz
SPEAKER_ALIAS = {}


def _silence_audio(device: str, wav_like: torch.Tensor, seconds: float) -> torch.Tensor:
    """Create (possibly negative) silence.

    Semantics for negative silence lengths (speaker overlap):
    - Positive seconds: return that many zeros (delay).
    - Zero: return empty tensor.
    - Negative seconds: return empty tensor.

    Negative overlap is handled later by truncating/aligning each speaker's
    concatenation, so at this level we never create a "negative" tensor.
    """
    if seconds <= 0.0:
        return torch.empty(1, 0, device=device, dtype=wav_like.dtype)
    n = int(SAMPLE_RATE * seconds)
    if n <= 0:
        return torch.empty(1, 0, device=device, dtype=wav_like.dtype)
    return torch.zeros(1, n, device=device, dtype=wav_like.dtype)


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
            t_samples += int(SAMPLE_RATE * turn["pre_silence"])

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
        delta = int(SAMPLE_RATE * silence_seconds)
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


# chatterbox inference gets its own venv, so we let the job execute this file directly
def main():
    # READ args --in_text and --out_dir
    parser = argparse.ArgumentParser(description="Run Chatterbox TTS Inference")
    parser.add_argument(
        "--in_text",
        type=str,
        required=True,
        help="Path to input text file. Input text file is a line seperated list of json arrays that contain dialogues",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save output audio. Each line in the input text file will be saved as a directory that contains the audio channel for every speaker",
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
    model = ChatterboxTurboTTS.from_pretrained(device=device)

    # read in_text line by line
    with open(args.in_text, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(f"Processing dialogue {i}...")
            dialogue = json.loads(
                line
            )  # each line is a json array that contains dialogues
            assert isinstance(dialogue, list), (
                "Each line in the input text file should be a json array that contains dialogues"
            )
            assert len(dialogue) > 0, "Each dialogue should contain at least one turn"
            assert all("speaker" in turn and "text" in turn for turn in dialogue), (
                "Each turn in the dialogue should contain 'speaker', 'text' fields"
            )

            output_dir = os.path.join(args.out_dir, f"dialogue_{i}")
            os.makedirs(output_dir, exist_ok=True)

            # speakers must not contain path symbols
            # assert all(
            #    "/" not in turn["speaker"] and "\\" not in turn["speaker"]
            #    for turn in dialogue
            # ), "Speaker names must not contain path symbols"

            audios_per_speaker, utterances = gen_conversation(
                model, dialogue, device, args.speaker_directory, silence_length_sampler
            )

            for speaker, audio in audios_per_speaker.items():
                torchaudio.save(f"{output_dir}/{speaker}.wav", audio.cpu(), SAMPLE_RATE)
                print(f"Saved {speaker}'s audio to {output_dir}/{speaker}.wav")

            # save utterances without audio as metadata.json
            metadata = []
            for u in utterances:
                metadata.append(
                    {
                        "speaker": u["speaker"],
                        "text": u["text"],
                        "start_time": u["start"] / SAMPLE_RATE,
                    }
                )
            with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        print("All dialogues processed successfully!")


if __name__ == "__main__":
    main()
