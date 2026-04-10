import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf
import argparse
import os

def gen_voice(model, tokenizer, text_prompt, voice_description, device):
    print("Generating voice...")

    # Tokenize the description and the text
    input = tokenizer(voice_description, return_tensors="pt").to(device)
    prompt = tokenizer(text_prompt, return_tensors="pt").to(device)

    # Generate the audio
    generation = model.generate(input_ids=input.input_ids, attention_mask=input.attention_mask, prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask, do_sample=True)

    # Convert the generated tensor to a numpy array
    audio_arr = generation.cpu().numpy().squeeze()

    return audio_arr

DEFAULT_TEXT_PROMPT = "The quick brown fox jumps over the lazy dog. Who could have possibly predicted that trees would drop leaves in October? But the bus driver hasn't got a clue where he's going, misses the turning, and ends up taking a massive detour through some dreary industrial estate in London."


def main():
    parser = argparse.ArgumentParser(description="Run parlertts TTS Inference")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output audio.")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT_PROMPT, help="Text prompt to generate speech from.")
    parser.add_argument("--voices_per_prompt", type=int, default=1, help="Number of different voices to generate per prompt.")
    # allow multiple voice prompts
    parser.add_argument("--voice_description", type=str, action="append", help="Description of the voice to generate.")
    args = parser.parse_args()

    # Automatically use your GPU if available, otherwise fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)  # For reproducibility
    # Load the Parler-TTS Large model and its tokenizer
    model_name = "parler-tts/parler-tts-mini-v1.1"
    print(f"Loading {model_name}...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 1. The script you want the voice to read 
    # (Keep it around 10-15 seconds for Chatterbox cloning)
    
    # 2. The description of the random voice you want to generate
    # Change the gender, pitch, and speed to get completely different voices.
    # ALWAYS keep the quality descriptions at the end!

    # TODO random description
    # voice_description = (
    #     "A male speaker delivers a slightly expressive and animated speech with a moderate speed and deep pitch. "
    #     "The recording is of very high quality, with the speaker's voice sounding clear and very close up, "
    #     "recorded in a soundproof studio with zero background noise."
    # )

    for voice_prompt in args.voice_description:
        for i in range(args.voices_per_prompt): # TODO batch https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md
            print(f"Generating voice {i+1}/{args.voices_per_prompt}")
            audio_arr = gen_voice(model, tokenizer, args.text, voice_prompt, device)

            # Save the generated audio to a WAV file
            output_filename = f"voice_{i}.wav"
            output_filename = os.path.join(args.out_dir, output_filename)
            sf.write(output_filename, audio_arr, model.config.sampling_rate)

            print(f"Success! High-quality voice saved to {output_filename}")

if __name__ == "__main__":
    main()