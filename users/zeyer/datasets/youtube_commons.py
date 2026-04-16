"""
YouTube-Commons

https://huggingface.co/datasets/PleIAs/YouTube-Commons/
"""

from __future__ import annotations
from typing import Any, Dict


def _load():
    from datasets import Features, Value, VerificationMode, load_dataset, concatenate_datasets

    # https://huggingface.co/datasets/PleIAs/YouTube-Commons/discussions/7
    f = Features(
        {
            "video_id": Value("string"),
            "video_link": Value("string"),
            "title": Value("string"),
            "text": Value("string"),
            "channel": Value("string"),
            "channel_id": Value("string"),
            "date": Value("string"),
            "license": Value("string"),
            "original_language": Value("string"),
            "transcription_language": Value("string"),
            "word_count": Value("int64"),
            "character_count": Value("int64"),
        }
    )
    f1 = Features({"language_id_method": Value("string"), **f})
    f2 = Features({"language_id_method": Value("string"), "__index_level_0__": Value("int64"), **f})
    f3 = Features({"source_language": Value("string"), **f})

    ds1 = load_dataset(
        "PleIAs/YouTube-Commons",
        split="train",
        data_files=[f"cctube_{i}.parquet" for i in range(0, 234)],
        features=f1,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    ds1 = ds1.remove_columns([key for key in ds1.column_names if key not in f])
    ds2 = load_dataset(
        "PleIAs/YouTube-Commons",
        split="train",
        data_files=[f"cctube_{i}.parquet" for i in range(234, 287)],
        features=f2,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    ds2 = ds2.remove_columns([key for key in ds2.column_names if key not in f])
    ds3 = load_dataset(
        "PleIAs/YouTube-Commons",
        split="train",
        data_files=[f"cctube_{i}.parquet" for i in range(287, 439)],
        features=f3,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    ds3 = ds3.remove_columns([key for key in ds3.column_names if key not in f])
    ds = concatenate_datasets([ds1, ds2, ds3])
    # Dataset({
    #     features: ['video_id', 'video_link', 'title', 'text', 'channel', 'channel_id', 'date', 'license', 'original_language', 'transcription_language', 'word_count', 'character_count'],
    #     num_rows: 22684737
    # })
    return ds


def _load_en():
    ds = _load()
    ds = ds.filter(_filter_func_lang_en, num_proc=8)
    # Dataset({
    #     features: ['video_id', 'video_link', 'title', 'text', 'channel', 'channel_id', 'date', 'license', 'original_language', 'transcription_language', 'word_count', 'character_count'],
    #     num_rows: 3262750
    # })
    # sum(ds["word_count"]) = 6_664_116_242
    return ds


def _filter_func_lang_en(example: Dict[str, Any]) -> bool:
    return example["transcription_language"] == "en"


# TODO The rows contain long paragraphs. For most of our models, we probably need to split them into sentences.
#   For that, need to use some model. E.g. https://huggingface.co/kredor/punctuate-all.


def _demo():
    from datasets import Dataset
    from transformers import pipeline
    import nltk

    # Ensure the NLTK sentence splitter is downloaded
    nltk.download("punkt", quiet=True)

    # 1. Start with your unpunctuated dataset
    original_dataset = Dataset.from_dict(
        {
            "text": [
                "this is the first document it has two sentences",
                "hugging face is a company based in new york city they are known for their transformers library",
                "what is the weather like today in aachen",
            ]
        }
    )

    # 2. Load the punctuation restoration pipeline
    # This model is lightweight and works well for English.
    # It will run on CUDA if available, otherwise CPU.
    punctuator = pipeline(
        "token-classification",
        model="kredor/punctuate-all-distilbert-base-cased",
        aggregation_strategy="word",  # This makes processing the output much easier
    )

    # 3. Define the mapping function to restore punctuation and then split
    def restore_and_split(examples):
        # The pipeline can process a list of texts directly
        punctuated_texts = punctuator(examples["text"])

        # The output needs to be reconstructed into strings
        reconstructed_texts = []
        for result in punctuated_texts:
            # Each 'result' is a list of words and their predicted punctuation
            # e.g., [{'word': 'hello', 'entity_group': 'O'}, {'word': 'world', 'entity_group': '.'}]
            full_text = "".join([item["word"] + item["entity_group"].replace("O", " ") for item in result]).strip()
            reconstructed_texts.append(full_text)

        # Now, split the newly punctuated texts into sentences
        all_sentences = [sentence for text in reconstructed_texts for sentence in nltk.sent_tokenize(text)]

        return {"sentence": all_sentences}

    # 4. Apply the function using .map()
    # We use a larger batch size because pipeline operations are faster on batches.
    sentence_dataset = original_dataset.map(
        restore_and_split,
        batched=True,
        batch_size=8,  # Adjust batch size based on your hardware (GPU memory)
        remove_columns=original_dataset.column_names,
    )
    return sentence_dataset
