from i6_experiments.common.setups.returnn.datasets.base import MetaDataset

vox_lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
vox_audio_dir = "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr"
vox_target_dir = "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_512"


def get_voxpopuli_data_per_lang(audio_base_dir=vox_audio_dir, target_base_dir=vox_target_dir, split="train", lang_list=vox_lang_list, partition_epoch=1):
    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir + "/" + l + "/" + split + ".hdf" for l in lang_list],
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.1000",
    }
    target_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [target_base_dir + "/" + l + "/" + split + ".hdf" for l in lang_list],
    }

    return MetaDataset(data_map={"data": ("features", "data"), "targets": ("targets", "data")},
                       datasets={
                           "features": raw_audio_dataset_dict,
                           "targets": target_dataset_dict,
                       },
                       seq_order_control_dataset="features",
                       )

def get_csfleurs_data_per_set(audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_asr", split="test", set_list=['mms', 'read', 'xtts'], partition_epoch=1):
    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir + "/" + dataset + "/" + split + ".hdf" for dataset in set_list],
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.1000",
    }

    return MetaDataset(data_map={"data": ("features", "data")},#, "targets": ("targets", "data")},
                       datasets={
                           "features": raw_audio_dataset_dict,
                           #"targets": target_dataset_dict,
                       },
                       seq_order_control_dataset="features",
                       )
    
def get_miami_data_per_set(audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr", split="test", set_list=['full', 'spa', 'eng'], partition_epoch=1):
    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir + "/" + dataset + "/" + split + ".hdf" for dataset in set_list],
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.1000",
    }

    return MetaDataset(data_map={"data": ("features", "data")},#, "targets": ("targets", "data")},
                       datasets={
                           "features": raw_audio_dataset_dict,
                           #"targets": target_dataset_dict,
                       },
                       seq_order_control_dataset="features",
                       )

def get_fleurs_data_per_set(audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/fleurs_asr", split=['mms', 'read', 'xtts'], set_list="read", partition_epoch=1):
    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir + "/" + set_list + "/" + split + ".hdf"],
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.1000",
    }

    return MetaDataset(data_map={"data": ("features", "data")},#, "targets": ("targets", "data")},
                       datasets={
                           "features": raw_audio_dataset_dict,
                           #"targets": target_dataset_dict,
                       },
                       seq_order_control_dataset="features",
                       )

def get_switchlingua_data_per_set(audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr", target_base_dir= "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_lex_hdf", split="dev", partition_epoch=1):
    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir + "/" + split + ".hdf"],
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.1000",
    }

    target_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": [target_base_dir + "/" + split + ".hdf"],
        }

    return MetaDataset(data_map={"data": ("features", "data"), "targets": ("targets", "data")},
                           datasets={
                               "features": raw_audio_dataset_dict,
                               "targets": target_dataset_dict,
                           },
                           seq_order_control_dataset="features",
                           )
    
def get_data_per_set(corpus, lexicon_path = None, split= None, partition_epoch= None):
    if corpus == "switchlingua":
        return get_switchlingua_data_per_set(split=split, partition_epoch=partition_epoch)
    elif corpus == "switchlingua-tts":
        return get_switchlingua_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr_tts",
                                    target_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_tts_asr_lexicon",
                                    split="train", partition_epoch=partition_epoch)
    elif corpus == "csfleurs":
        return get_csfleurs_data_per_set(split=split, partition_epoch=partition_epoch)
    elif corpus == "fleurs":
        return get_fleurs_data_per_set(split=split, partition_epoch=partition_epoch)
    else:
        return get_voxpopuli_data_per_lang(split=split, partition_epoch=partition_epoch)


def get_data_hdf(audio_base_dir: str):
    raw_audio_dataset_dict = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": [audio_base_dir],
        "partition_epoch": 1,
        "seq_ordering": "laplace:.1000",
    }
    return MetaDataset(data_map={"data": ("features", "data")},#, "targets": ("targets", "data")},
                       datasets={
                           "features": raw_audio_dataset_dict,
                           #"targets": target_dataset_dict,
                       },
                       seq_order_control_dataset="features",
                       )