from sisyphus import tk

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any, Tuple

from i6_core.returnn import CodeWrapper

from i6_experiments.common.datasets.switchboard.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.switchboard.corpus_train import get_spoken_form_train_bliss_corpus_ldc, \
    get_train_bliss_corpus_ldc
from i6_experiments.common.datasets.switchboard.corpus_eval import get_hub5e00, SwitchboardEvalDataset
from i6_experiments.common.datasets.switchboard.bpe import get_subword_nmt_bpe

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import AudioRawDatastream, \
    ReturnnAudioRawOptions
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset

from ..default_tools import RETURNN_ROOT, RETURNN_CPU_EXE, RETURNN_EXE

@lru_cache()
def get_bpe_datastream(bpe_size, is_recog, use_spoken_form):
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param int bpe_size: size for the bpe labels
    :param bool is_recog: removes the UNK label when not in training
    :return:
    """
    # build dataset
    bpe_settings = get_subword_nmt_bpe(
        bpe_size=bpe_size,
        use_spoken_form=use_spoken_form,
        unk_label='<unk>',
        subdir_prefix="experiments/switchboard/attention_test/bpe/"
    )
    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog
    )
    return bpe_targets


@lru_cache()
def get_audio_raw_datastream():
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=True)
    )
    return audio_datastream


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    extern_data: Dict[str, Dict[str, Any]]

@tk.block()
def build_training_datasets(
        bpe_size=1000,
        partition_epoch=6,
        use_curicculum=True,
        seq_ordering="laplace:.1000",
        link_speed_perturbation=True,
        use_raw_features=True,
        use_spoken_form=False,
    ):
    """

    :param returnn_python_exe:
    :param returnn_root:
    :param output_path:
    :param int bpe_size:
    :param int bpe_partition_epoch:
    :param str seq_ordering:
    :return:
    """
    from i6_core.returnn.oggzip import BlissToOggZipJob

    if use_spoken_form:
        train_corpus = get_spoken_form_train_bliss_corpus_ldc()
    else:
        train_corpus = get_train_bliss_corpus_ldc()
    switchboard_ogg = BlissToOggZipJob(
        bliss_corpus=train_corpus,
        segments=None,
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_CPU_EXE
    ).out_ogg_zip


    from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob
    from i6_core.corpus.convert import CorpusToTxtJob, CorpusReplaceOrthFromTxtJob
    from i6_core.text.processing import PipelineJob

    filtered_hub5_ogg = FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=get_hub5e00().bliss_corpus, bliss_lexicon=get_bliss_lexicon(), all_unknown=False,
    ).out_corpus
    text = CorpusToTxtJob(filtered_hub5_ogg).out_txt
    lowercased_text = PipelineJob(text, pipeline=["tr '[:upper:]' '[:lower:]'"], mini_task=True).out
    filtered_lowercased_hub5_ogg = CorpusReplaceOrthFromTxtJob(filtered_hub5_ogg, lowercased_text).out_corpus

    hub5_ogg = BlissToOggZipJob(
        bliss_corpus=filtered_lowercased_hub5_ogg,
        segments=None,
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_CPU_EXE
    ).out_ogg_zip

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, use_spoken_form=use_spoken_form, is_recog=False)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream()
    else:
        raise NotImplementedError

    extern_data = {
        'audio_features': audio_datastream.as_returnn_extern_data_opts(),
        'bpe_labels': train_bpe_datastream.as_returnn_extern_data_opts()
    }

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}


    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if link_speed_perturbation:
        training_audio_opts["pre_process"] = CodeWrapper("speed_pert")

    train_zip_dataset = OggZipDataset(
        files=switchboard_ogg,
        audio_options=training_audio_opts,
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=partition_epoch,
        seq_ordering=seq_ordering,
        additional_options={"epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}} if use_curicculum else None}  # still hardcoded, future work
    )
    train_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": train_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments
    cv_zip_dataset = OggZipDataset(
        files=[hub5_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=None,
        seq_ordering="sorted_reverse"
    )
    cv_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": cv_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    devtrain_zip_dataset = OggZipDataset(
        files=switchboard_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": devtrain_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        extern_data=extern_data
    )


@lru_cache()
def build_test_datasets(bpe_size, use_spoken_form, use_raw_features=False) -> Tuple[Dataset, SwitchboardEvalDataset]:


    eval_dataset = get_hub5e00()

    from i6_core.returnn.oggzip import BlissToOggZipJob

    eval_ogg = BlissToOggZipJob(
        bliss_corpus=eval_dataset.bliss_corpus,
        segments=None,
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_CPU_EXE
    ).out_ogg_zip

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, use_spoken_form=use_spoken_form, is_recog=True)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream()
    else:
        raise NotImplementedError

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}

    test_zip_dataset = OggZipDataset(
        files=[eval_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse"
    )
    test_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": test_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, eval_dataset
