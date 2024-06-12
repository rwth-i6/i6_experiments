Some code here is based on: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard

- The `normalizer` folder is taken from here: https://github.com/huggingface/open_asr_leaderboard/tree/main/normalizer. The `write_manifest` function was modified in order to pass the manifest output path as parameter to the function.
- `run_eval.py` reads the dataset path and model path from input instead.
