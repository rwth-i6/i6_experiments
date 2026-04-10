import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS
import argparse
import json
import os
import random

SAMPLE_RATE = 24000 # Chatterbox default output is 24kHz

def gen_conversation(model, dialogue, device, speaker_dir, silence_length_sampler) -> map[str, torch.Tensor]:

    print("Generating conversation...")

    # TODO get all speakers
    speakers = set(turn["speaker"] for turn in dialogue)
    speak_map = {}
    for s in speakers:
        model.prepare_conditionals(os.path.join(speaker_dir, s), exaggeration=turn["exaggeration"], norm_loudness=True)
        speak_map[s] = model.conds

    audio_segments = {s: [] for s in speakers}

    for turn in dialogue:
        print(f"Synthesizing {turn['speaker']}'s line...")
        
        # TODO defaults for these
        model.conds = speak_map[turn["speaker"]]
        wav = model.generate(
            text=turn["text"],
            audio_prompt_path=None, # we hack conditionals in directly
            exaggeration=turn["exaggeration"],
            cfg=turn["cfg"]
        )
        model.conds = None

        wav_silence = torch.zeros_like(wav).to(device)
        silence = torch.zeros(1, int(SAMPLE_RATE * silence_length_sampler())).to(device)

        for s in speakers:
            if s == turn["speaker"]:
                audio_segments[s].append(wav)
                audio_segments[s].append(silence)
            else:
                audio_segments[s].append(wav_silence)
                audio_segments[s].append(silence)

    return {s: torch.cat(audio_segments[s][:-1], dim=-1) for s in speakers}
    

# chatterbox inference gets its own venv, so we let the job execute this file directly
def main():
    # READ args --in_text and --out_dir
    parser = argparse.ArgumentParser(description="Run Chatterbox TTS Inference")
    parser.add_argument("--in_text", type=str, required=True, help="Path to input text file. Input text file is a line seperated list of json arrays that contain dialogues")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output audio. Each line in the input text file will be saved as a directory that contains the audio channel for every speaker")
    parser.add_argument("--speaker_directory", type=str, required=True, help="Directory that contains reference audio for each speaker.")
    args = parser.parse_args()

    # TODO figure out good way to sample silence
    def silence_length_sampler():
        return 0.5 + random.random() # between 0.5 and 1.5 seconds of silence between turns

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)

    # 2. Define the script with conversational tags and parameter tuning
    # dialogue =[
    #     {
    #         "speaker": "Alice",
    #         "audio_ref": "speaker_A_ref.wav", 
    #         "text": "Hey there! [laugh] I really didn't expect to run into you today.",
    #         "exaggeration": 0.6, # Slightly increased for a happy, expressive tone
    #         "cfg": 0.4           # Lowering CFG slightly allows for more deliberate pacing
    #     },
    #     {
    #         "speaker": "Bob",
    #         "audio_ref": "speaker_B_ref.wav", 
    #         "text": "Oh, hi Alice! [chuckle] Yeah, I decided to take an early break from the office.",
    #         "exaggeration": 0.5, # Default neutral expression
    #         "cfg": 0.5
    #     },
    #     {
    #         "speaker": "Alice",
    #         "audio_ref": "speaker_A_ref.wav", 
    #         "text": "Well, it's a beautiful day for it. [sigh] Do you want to grab a coffee?",
    #         "exaggeration": 0.5,
    #         "cfg": 0.4
    #     }
    # ]

    # read in_text line by line
    with open(args.in_text, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            dialogue = json.loads(line) # each line is a json array that contains dialogues
            assert isinstance(dialogue, list), "Each line in the input text file should be a json array that contains dialogues"
            assert len(dialogue) > 0, "Each dialogue should contain at least one turn"
            assert all("speaker" in turn and "text" in turn for turn in dialogue), "Each turn in the dialogue should contain 'speaker', 'text' fields"

            output_dir = os.path.join(args.out_dir, f"dialogue_{i}")
            os.makedirs(output_dir, exist_ok=True)

            # speakers must not contain path symbols
            assert all("/" not in turn["speaker"] and "\\" not in turn["speaker"] for turn in dialogue), "Speaker names must not contain path symbols"

            audios_per_speaker = gen_conversation(model, dialogue, device, args.speaker_directory)

            for speaker, audio in audios_per_speaker.items():
                torchaudio.save(f"{output_dir}/{speaker}.wav", audio.cpu(), SAMPLE_RATE)
                print(f"Saved {speaker}'s audio to {output_dir}/{speaker}.wav")
