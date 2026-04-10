import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import argparse
import os

def gen_voice(model, tokenizer, text_prompt, voice_description, device):
    print("Generating voice...")

    # Tokenize the description and the text
    input_ids = tokenizer(voice_description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)

    # Generate the audio
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

    # Convert the generated tensor to a numpy array
    audio_arr = generation.cpu().numpy().squeeze()

    return audio_arr

def main():
    parser = argparse.ArgumentParser(description="Run parlertts TTS Inference")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output audio.")
    parser.add_argument("--voices_per_prompt", type=int, default=1, help="Number of different voices to generate per prompt.")
    args = parser.parse_args()

    # Automatically use your GPU if available, otherwise fallback to CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Loading Parler-TTS Large... (This might take a moment, it's a 2.2B parameter model)")

    # Load the Parler-TTS Large model and its tokenizer
    model_name = "parler-tts/parler-tts-large-v1"
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # 1. The script you want the voice to read 
    # (Keep it around 10-15 seconds for Chatterbox cloning)
    text_prompt = "The quick brown fox jumps over the lazy dog. We promptly judged antique ivory buckles for the next prize. It is easy to tell the depth of a well by dropping a stone into it."

    # 2. The description of the random voice you want to generate
    # Change the gender, pitch, and speed to get completely different voices.
    # ALWAYS keep the quality descriptions at the end!

    # TODO random description
    voice_description = (
        "A male speaker delivers a slightly expressive and animated speech with a moderate speed and deep pitch. "
        "The recording is of very high quality, with the speaker's voice sounding clear and very close up, "
        "recorded in a soundproof studio with zero background noise."
    )


    for i in range(args.voices_per_prompt):
        audio_arr = gen_voice(model, tokenizer, text_prompt, voice_description, device)

        # Save the generated audio to a WAV file
        output_filename = f"voice_{i}.wav"
        output_filename = os.path.join(args.out_dir, output_filename)
        sf.write(output_filename, audio_arr, model.config.sampling_rate)

        print(f"Success! High-quality voice saved to {output_filename}")

if __name__ == "__main__":
    main()