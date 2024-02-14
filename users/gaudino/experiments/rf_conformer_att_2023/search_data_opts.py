from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output

search_data_opts_ted2 = {
    "class": "MetaDataset",
    "data_map": {
        "audio_features": ("zip_dataset", "data"),
        "bpe_labels": ("zip_dataset", "classes"),
    },
    "datasets": {
        "zip_dataset": {
            "class": "OggZipDataset",
            "path": generic_job_output(
                "i6_core/returnn/oggzip/BlissToOggZipJob.Wgp6724p1XD2/output/out.ogg.zip"
            ).get_path(),
            # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.Wgp6724p1XD2/output/out.ogg.zip",
            "use_cache_manager": True,
            "audio": {
                "features": "raw",
                "peak_normalization": True,
                "preemphasis": None,
            },
            "targets": {
                "class": "BytePairEncoding",
                "bpe_file": generic_job_output(
                    "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes"
                ).get_path(),
                # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes",
                "vocab_file": generic_job_output(
                    "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab"
                ).get_path(),
                # "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab",
                "unknown_label": "<unk>",
                "seq_postfix": [0],
            },
            "segment_file": None,
            "partition_epoch": 1,
            # "seq_ordering": "sorted_reverse",
        }
    },
    "seq_order_control_dataset": "zip_dataset",
}

search_data_opts_librispeech960 = {
    "class": "MetaDataset",
    "data_map": {
        "audio_features": ("zip_dataset", "data"),
        "bpe_labels": ("zip_dataset", "classes"),
    },
    "datasets": {
        "zip_dataset": {
            "class": "OggZipDataset",
            "path": generic_job_output(
                "i6_core/returnn/oggzip/BlissToOggZipJob.NSdIHfk1iw2M/output/out.ogg.zip"
            ).get_path(),
            "use_cache_manager": True,
            "audio": {
                "features": "raw",
                "peak_normalization": True,
                "preemphasis": None,
            },
            "targets": {
                "class": "BytePairEncoding",
                "bpe_file": generic_job_output(
                    "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes"
                ).get_path(),
                "vocab_file": generic_job_output(
                    "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"
                ).get_path(),
                "unknown_label": "<unk>",
                "seq_postfix": [0],
            },
            "segment_file": None,
            "partition_epoch": 1,
            # "seq_ordering": "sorted_reverse",
        }
    },
    "seq_order_control_dataset": "zip_dataset",
}