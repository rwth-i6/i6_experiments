from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.librispeech_long.librispeech_long import (
    get_ogg_zip_dict,
    get_bliss_corpus_dict,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import (
    get_bpe_datastream,
    get_audio_raw_datastream,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream
from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset


@lru_cache()
def build_test_dataset(dataset_key, bpe_size=10000, use_raw_features=False, preemphasis=None):
    ogg_zip_dict = get_ogg_zip_dict("corpora")
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]
    from i6_core.corpus.convert import CorpusToTextDictJob

    test_reference_dict_file = CorpusToTextDictJob(bliss_dict[dataset_key]).out_dictionary

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=True)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream(preemphasis)  # TODO: experimenting
    else:
        raise NotImplementedError

    data_map = {"audio_features": ("zip_dataset", "data"), "bpe_labels": ("zip_dataset", "classes")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, test_reference_dict_file, bliss_dict[dataset_key]
