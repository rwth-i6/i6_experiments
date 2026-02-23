from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
import glob
import numpy as np
import math
import random
from itertools import chain


def get_voxpopuli_data_per_lang(audio_base_dir, target_base_dir, split="train", lang_list="de", partition_epoch=1):
    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir + "/" + lang + "/" + split + ".hdf" for lang in lang_list],
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.1000",
    }
    target_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [target_base_dir + "/" + lang + "/" + split + ".hdf" for lang in lang_list],
    }

    return MetaDataset(data_map={"data": ("features", "data"), "targets": ("targets", "data")},
                       datasets={
                           "features": raw_audio_dataset_dict,
                           "targets": target_dataset_dict,
                       },
                       seq_order_control_dataset="features",
                       )


def get_voxpopuli_data_with_upsampling(audio_base_dir, target_base_dir, split="train",
                                       lang_list=["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt",
                                                  "nl", "pl", "ro", "sk", "sl"],
                                       partition_epoch=1, upsample_alpha=0.5, seed=42, ):
    rng = np.random.default_rng(seed)
    files_by_lang = {f"{lang}": glob.glob(f"{target_base_dir}/{lang}/{split}_*.hdf") for lang in lang_list}
    lang_duration = {
        "en": 543,
        "de": 282,
        "fr": 211,
        "es": 166,
        "pl": 111,
        "it": 91,
        "ro": 89,
        "hu": 63,
        "cs": 62,
        "nl": 53,
        "fi": 27,
        "hr": 43,
        "sk": 35,
        "sl": 10,
        "et": 3,
        "lt": 2
    }
    lang_upsample_ratio = {l: lang_duration[l] / sum(list(lang_duration.values())) for l in lang_list}
    lang_upsample_ratio = {l: lang_upsample_ratio[l] / max(list(lang_upsample_ratio.values())) for l in lang_list}
    lang_upsample_ratio = {l: lang_upsample_ratio[l] ** upsample_alpha / lang_upsample_ratio[l] for l in
                           lang_upsample_ratio.keys()}

    upsampled_files_by_lang = {}

    for lang, files in files_by_lang.items():
        if lang not in lang_upsample_ratio:
            raise ValueError(f"Missing upsample rate for language: {lang}")

        rate = lang_upsample_ratio[lang]
        n = len(files)
        if n == 0 or rate <= 0:
            upsampled_files_by_lang[lang] = []
            continue

        result = []

        if rate >= 1.0:
            integer_part = int(math.floor(rate))
            fractional_part = rate - integer_part

            # Full duplications
            for _ in range(integer_part):
                result.extend(files)

            # Fractional sampling
            k = int(round(fractional_part * n))
            if k > 0:
                indices = rng.choice(n, size=min(k, n), replace=False)
                result.extend([files[i] for i in indices])

        else:
            # Downsampling case (rate < 1)
            k = int(round(rate * n))
            if k > 0:
                indices = rng.choice(n, size=min(k, n), replace=False)
                result.extend([files[i] for i in indices])

        upsampled_files_by_lang[lang] = result

    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir + "/" + lang + "/" + split + ".hdf" for lang in lang_list],
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.1000",
    }
    target_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": list(chain.from_iterable(upsampled_files_by_lang.values()))
    }

    return MetaDataset(data_map={"data": ("features", "data"), "targets": ("targets", "data")},
                       datasets={
                           "features": raw_audio_dataset_dict,
                           "targets": target_dataset_dict,
                       },
                       seq_order_control_dataset="features",
                       )


def get_voxpopuli_data_with_language_id(audio_base_dir, target_base_dir, split="train",
                                       lang_list=["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt",
                                                  "nl", "pl", "ro", "sk", "sl"],
                                       partition_epoch=1, upsample_alpha=0.5, seed=42):
    if split == "train":
        rng = np.random.default_rng(seed)
        files_by_lang = {f"{lang}": glob.glob(f"{target_base_dir}/{lang}/{split}_*.hdf") for lang in lang_list}
        lang_duration = {
            "en": 543,
            "de": 282,
            "fr": 211,
            "es": 166,
            "pl": 111,
            "it": 91,
            "ro": 89,
            "hu": 63,
            "cs": 62,
            "nl": 53,
            "fi": 27,
            "hr": 43,
            "sk": 35,
            "sl": 10,
            "et": 3,
            "lt": 2
        }
        lang_upsample_ratio = {l: lang_duration[l] / sum(list(lang_duration.values())) for l in lang_list}
        lang_upsample_ratio = {l: lang_upsample_ratio[l] / max(list(lang_upsample_ratio.values())) for l in lang_list}
        lang_upsample_ratio = {l: lang_upsample_ratio[l] ** upsample_alpha / lang_upsample_ratio[l] for l in
                               lang_upsample_ratio.keys()}

        upsampled_files_by_lang = {}

        for lang, files in files_by_lang.items():
            if lang not in lang_upsample_ratio:
                raise ValueError(f"Missing upsample rate for language: {lang}")

            rate = lang_upsample_ratio[lang]
            n = len(files)
            if n == 0 or rate <= 0:
                upsampled_files_by_lang[lang] = []
                continue

            result = []

            if rate >= 1.0:
                integer_part = int(math.floor(rate))
                fractional_part = rate - integer_part

                # Full duplications
                for _ in range(integer_part):
                    result.extend(files)

                # Fractional sampling
                k = int(round(fractional_part * n))
                if k > 0:
                    indices = rng.choice(n, size=min(k, n), replace=False)
                    result.extend([files[i] for i in indices])

            else:
                # Downsampling case (rate < 1)
                k = int(round(rate * n))
                if k > 0:
                    indices = rng.choice(n, size=min(k, n), replace=False)
                    result.extend([files[i] for i in indices])

            upsampled_files_by_lang[lang] = result

        target_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": list(chain.from_iterable(upsampled_files_by_lang.values()))
        }
    else:
        target_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": [target_base_dir + "/" + l + "/" + split + ".hdf" for l in lang_list],
        }

    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir + "/" + lang + "/" + split + ".hdf" for lang in lang_list],
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.1000",
    }

    lang_id_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": glob.glob(f"/u/jxu/setups/voxpopuli/2026-01-14-multilingual-bpe-ctc/output/target_hdf/language_id/{split}/*.hdf")
    }

    return MetaDataset(data_map={"data": ("features", "data"), "targets": ("targets", "data"), "language": ("language", "data")},
                       datasets={
                           "features": raw_audio_dataset_dict,
                           "targets": target_dataset_dict,
                           "language": lang_id_dataset_dict,
                       },
                       seq_order_control_dataset="features",
                       )