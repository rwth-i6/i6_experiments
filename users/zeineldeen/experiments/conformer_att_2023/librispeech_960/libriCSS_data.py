import sisyphus.toolkit as tk
import os

from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import (
    get_bpe_datastream,
    get_audio_raw_datastream,
)
from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset
from i6_core.corpus.convert import CorpusToTextDictJob
from i6_core.returnn.oggzip import BlissToOggZipJob

from .default_tools import RETURNN_ROOT, RETURNN_CPU_EXE


output_prefix = "libriCSS"


def build_libriCSS_test_sets(dataset_key, bpe_size=10000, use_raw_features=False, preemphasis=None):
    dev_bliss_corpus = tk.Path(
        "/work/asr3/converse/data/libri_css/corpus_files/libriCSS_singlechannel_77_62000_124/dev.xml.gz"
    )
    eval_bliss_corpus = tk.Path(
        "/work/asr3/converse/data/libri_css/corpus_files/libriCSS_singlechannel_77_62000_124/eval.xml.gz"
    )

    bliss_corpus_dict = {
        "dev": dev_bliss_corpus,
        "eval": eval_bliss_corpus,
    }

    ogg_audio_format = {"output_format": "ogg", "codec": "libvorbis"}

    converted_bliss_corpus_dict = {}
    for corpus_name, corpus in bliss_corpus_dict.items():
        bliss_change_encoding_job = BlissChangeEncodingJob(
            corpus_file=corpus,
            sample_rate=16000,
            **ogg_audio_format,
        )
        bliss_change_encoding_job.add_alias(
            os.path.join(
                output_prefix,
                "ogg_conversion",
                corpus_name,
            )
        )
        converted_bliss_corpus_dict[corpus_name] = bliss_change_encoding_job.out_corpus

    # get Oggs

    ogg_zip_dict = {}
    for name, bliss_corpus in converted_bliss_corpus_dict.items():
        ogg_zip_job = BlissToOggZipJob(
            bliss_corpus,
            no_conversion=True,
            returnn_python_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
        )
        ogg_zip_job.add_alias(os.path.join(output_prefix, "%s_ogg_zip_job" % name))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    # create dataset

    test_reference_dict_file = CorpusToTextDictJob(converted_bliss_corpus_dict[dataset_key]).out_dictionary
    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=True)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream(preemphasis)
    else:
        raise NotImplementedError

    data_map = {"audio_features": ("zip_dataset", "data"), "bpe_labels": ("zip_dataset", "classes")}

    test_zip_dataset = OggZipDataset(
        files=[ogg_zip_dict[dataset_key]],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, test_reference_dict_file, converted_bliss_corpus_dict[dataset_key]
