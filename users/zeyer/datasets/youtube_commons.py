"""
YouTube-Commons

https://huggingface.co/datasets/PleIAs/YouTube-Commons/
"""


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
    return ds
