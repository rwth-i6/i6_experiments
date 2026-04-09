import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS
import argparse

SAMPLE_RATE = 24000 # Chatterbox default output is 24kHz

def gen_conversation(model, dialogue, device):
	audio_segments =[]

	# TODO randomize silence
	silence = torch.zeros(1, int(SAMPLE_RATE * 0.5)).to(device)

	print("Generating conversation...")

	for turn in dialogue:
		print(f"Synthesizing {turn['speaker']}'s line...")
		
		# TODO defaults for these
		wav = model.generate(
			text=turn["text"],
			audio_prompt_path=turn["audio_ref"],
			exaggeration=turn["exaggeration"],
			cfg=turn["cfg"]
		)
		
		audio_segments.append(wav)
		audio_segments.append(silence)

	# 4. Concatenate all pieces along the time dimension (excluding trailing silence)
	final_audio = torch.cat(audio_segments[:-1], dim=-1)
	return final_audio.cpu()
	

# chatterbox inference gets its own venv, so we let the job execute this file directly
def main():
	# READ args --in_text and --out_dir
	parser = argparse.ArgumentParser(description="Run Chatterbox TTS Inference")
	parser.add_argument("--in_text", type=str, required=True, help="Path to input text file")
	parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output audio")
	args = parser.parse_args()

	# 1. Initialize the Turbo model (fastest, natively supports conversational tags)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = ChatterboxTurboTTS.from_pretrained(device=device)

	# 2. Define the script with conversational tags and parameter tuning
	dialogue =[
		{
			"speaker": "Alice",
			"audio_ref": "speaker_A_ref.wav", 
			"text": "Hey there! [laugh] I really didn't expect to run into you today.",
			"exaggeration": 0.6, # Slightly increased for a happy, expressive tone
			"cfg": 0.4           # Lowering CFG slightly allows for more deliberate pacing
		},
		{
			"speaker": "Bob",
			"audio_ref": "speaker_B_ref.wav", 
			"text": "Oh, hi Alice! [chuckle] Yeah, I decided to take an early break from the office.",
			"exaggeration": 0.5, # Default neutral expression
			"cfg": 0.5
		},
		{
			"speaker": "Alice",
			"audio_ref": "speaker_A_ref.wav", 
			"text": "Well, it's a beautiful day for it. [sigh] Do you want to grab a coffee?",
			"exaggeration": 0.5,
			"cfg": 0.4
		}
	]

	final_audio = gen_conversation(model, dialogue, device)

	# Save the final conversation to disk
	torchaudio.save(f"{args.out_dir}/realistic_dialog.wav", final_audio, SAMPLE_RATE)
	print(f"Saved to {args.out_dir}/realistic_dialog.wav")
