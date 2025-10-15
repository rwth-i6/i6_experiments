from i6_experiments.common.setups.returnn.datasets.base import MetaDataset


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
