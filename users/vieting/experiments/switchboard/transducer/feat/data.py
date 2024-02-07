"""
Create data for switchboard transducer pipeline.
"""
import copy
from typing import Any, Dict, List, Optional, Union

from sisyphus import tk
from i6_core import corpus
from i6_core import text
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn import ReturnnDumpHDFJob, RasrAlignmentDumpHDFJob, BlissToOggZipJob
from i6_core.text.processing import ConcatenateJob
from i6_experiments.common.datasets.switchboard import (
    get_train_corpus_object_ldc,
    get_hub5e00,
    get_hub5e00_corpus_object,
    get_bliss_lexicon,
)
from i6_experiments.common.setups.rasr.util import RasrDataInput, HdfDataInput, OggZipHdfDataInput
from i6_experiments.users.berger.recipe.lexicon.modification import (
    DeleteEmptyOrthJob,
    EnsureSilenceFirstJob,
    MakeBlankLexiconJob,
)
from i6_experiments.users.vieting.jobs.returnn import PeakyAlignmentJob
from i6_experiments.users.vieting.util.returnn import iterate_returnn_datasets
from .default_tools import RETURNN_ROOT, RETURNN_EXE


def get_switchboard_data():
    """
    Get train and test data for switchboard dataset. This is on the level of RASR corpus and segment file, lexicon,
    language model etc., not yet involving anything RETURNN specific.
    """
    # create train and cv sets
    train_corpus = get_train_corpus_object_ldc()
    total_train_num_segments = 249536
    cv_size = 300 / total_train_num_segments

    segments = {
        "all": corpus.SegmentCorpusJob(
            train_corpus.corpus_file, 1
        ).out_single_segment_files[1]
    }

    split_segments_job = corpus.ShuffleAndSplitSegmentsJob(
        segments["all"], {"train": 1 - cv_size, "cv": cv_size}
    )
    segments["train"] = split_segments_job.out_segments["train"]
    segments["cv"] = split_segments_job.out_segments["cv"]

    blacklisted_segments = [
        "switchboard-1/sw02986A/sw2986A-ms98-a-0013",
        "switchboard-1/sw02663A/sw2663A-ms98-a-0022",
        "switchboard-1/sw02691A/sw2691A-ms98-a-0017",
        "switchboard-1/sw04091A/sw4091A-ms98-a-0063",
        "switchboard-1/sw04103A/sw4103A-ms98-a-0022",
        "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
        "switchboard-1/sw04318A/sw4318A-ms98-a-0024",
    ]
    segments["train"] = corpus.FilterSegmentsByListJob(
        segment_files={1: segments["train"]},
        filter_list=blacklisted_segments,
    ).out_single_segment_files[1]
    segments["cv"] = corpus.FilterSegmentsByListJob(
        segment_files={1: segments["cv"]},
        filter_list=blacklisted_segments,
    ).out_single_segment_files[1]
    tail_job = text.TailJob(
        segments["train"], num_lines=300, zip_output=False
    )
    tail_job.out.path = "out.gz"  # fix for hash break, see i6_core/#479
    segments["devtrain"] = tail_job.out
    segments["all_filtered"] = corpus.FilterSegmentsByListJob(
        segment_files={1: segments["all"]},
        filter_list=blacklisted_segments,
    ).out_single_segment_files[1]

    # create lexica
    lexicon_base = get_bliss_lexicon()
    lexicon_base = DeleteEmptyOrthJob(lexicon_base).out_lexicon
    lexicon_rasr_loss = MakeBlankLexiconJob(lexicon_base).out_lexicon
    non_word_phones = ["[LAUGHTER]", "[NOISE]", "[VOCALIZEDNOISE]"]
    lexicon_recog_ctc = AddEowPhonemesToLexiconJob(lexicon_rasr_loss, nonword_phones=non_word_phones).out_lexicon
    lexicon_recog_transducer = AddEowPhonemesToLexiconJob(lexicon_base, nonword_phones=non_word_phones).out_lexicon
    lexicon_recog_transducer = EnsureSilenceFirstJob(lexicon_recog_transducer).out_lexicon
    lexicon_args = {
        "normalize_pronunciation": False,
        "add_all": True,
        "add_from_lexicon": False,
    }

    # lm args
    lm_args = {
        "filename": tk.Path(
            "/work/asr4/vieting/setups/swb/dependencies/swb.fsh.4gr.voc30k.LM.gz",
            hash_overwrite="/home/tuske/work/ASR/switchboard/corpus/lm/data/mylm/swb.fsh.4gr.voc30k.LM.gz",
        ),
        "type": "ARPA",
    }

    train_corpus = RasrDataInput(
        corpus_object=train_corpus,
        lexicon={"filename": lexicon_rasr_loss, **lexicon_args},
        lm=lm_args,
    )
    hub5e00 = get_hub5e00()
    dev_corpora = {
        "ctc": {
            "hub5e00": RasrDataInput(
                corpus_object=get_hub5e00_corpus_object(),
                lexicon={"filename": lexicon_recog_ctc, **lexicon_args},
                lm=lm_args,
                stm=hub5e00.stm,
                glm=hub5e00.glm,
            ),
        },
        "transducer": {
            "hub5e00": RasrDataInput(
                corpus_object=get_hub5e00_corpus_object(),
                lexicon={"filename": lexicon_recog_transducer, **lexicon_args},
                lm=lm_args,
                stm=hub5e00.stm,
                glm=hub5e00.glm,
            ),
        },
    }

    return train_corpus, dev_corpora, segments


def get_returnn_base_data(
    partition_epoch: Optional[Dict[str, int]] = None,
    context_window: Optional[Dict[str, int]] = None,
) -> Dict[str, Union[OggZipHdfDataInput, Dict[str, OggZipHdfDataInput]]]:
    """
    Get basic dataset with ogg input and hdf targets.
    """
    train_corpus, _, segments = get_switchboard_data()

    # oggzip
    segments_ogg_parallel = corpus.SplitSegmentFileJob(segments["all"], concurrent=20).out_segment_path
    returnn_root = copy.deepcopy(RETURNN_ROOT)
    returnn_root.hash_overwrite = "SWITCHBOARD_DEFAULT_RETURNN_ROOT"
    ogg_zip_job = BlissToOggZipJob(
        train_corpus.corpus_object.corpus_file,
        raw_sample_rate=8000,
        feat_sample_rate=100,
        segments=segments_ogg_parallel,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=returnn_root,
    )
    ogg_zip_job.rqmt = {"time": 8.0, "cpu": 2}
    meta_args = {"data_map": {"classes": ("hdf", "data"), "data": ("ogg", "data")}}
    if context_window is not None:
        meta_args["context_window"] = context_window
    audio = {
        "features": "raw",
        "peak_normalization": True
    }
    ogg_zip_base_args = dict(
        oggzip_files=[ogg_zip_job.out_ogg_zip],
        alignments=[],
        audio=audio,
        meta_args=meta_args,
    )

    # nn data
    partition_epoch = partition_epoch or {"train": 6, "dev": 1}
    assert set(partition_epoch.keys()) == {"train", "dev"}
    nn_train_data = OggZipHdfDataInput(
        partition_epoch=partition_epoch["train"],
        ogg_args={"segment_file": segments["train"], "targets": None},
        seq_ordering="laplace:.384",
        **ogg_zip_base_args,
    )

    nn_cv_data = OggZipHdfDataInput(
        partition_epoch=partition_epoch["dev"],
        seq_ordering="sorted_reverse",
        ogg_args={"segment_file": segments["cv"], "targets": None},
        **ogg_zip_base_args,
    )

    nn_devtrain_data = OggZipHdfDataInput(
        partition_epoch=partition_epoch["dev"],
        seq_ordering="sorted_reverse",
        ogg_args={"segment_file": segments["devtrain"], "targets": None},
        **ogg_zip_base_args,
    )

    returnn_datasets = {
        "train": nn_train_data,
        "dev": nn_cv_data,
        "eval_datasets": {"devtrain": nn_devtrain_data},
    }
    return returnn_datasets


def get_returnn_ogg_datasets(**kwargs) -> Dict[str, Dict]:
    """
    Get only ogg input datasets without targets.
    """
    returnn_datasets = get_returnn_base_data(**kwargs)
    returnn_datasets = {
        "train": returnn_datasets["train"].get_data_dict()["datasets"]["ogg"],
        "dev": returnn_datasets["dev"].get_data_dict()["datasets"]["ogg"],
        "eval_datasets": {"devtrain": returnn_datasets["eval_datasets"]["devtrain"].get_data_dict()["datasets"]["ogg"]},
    }
    return returnn_datasets


def get_returnn_datasets_transducer_viterbi(
        features: str = "waveform",
        alignment: Union[List[tk.Path], str] = "wei",
        partition_epoch: Optional[Dict[str, int]] = None,
        context_window: Optional[Dict[str, int]] = None,
):
    _, _, segments = get_switchboard_data()
    returnn_datasets = get_returnn_base_data(partition_epoch, context_window)
    sync_ogg_audio = False
    if alignment == "wei":
        sync_ogg_audio = True
        alignment_caches_train = [tk.Path(
            "/u/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/mm/alignment/AlignmentJob.fWmd1ZVWfcFA/output/"
            f"alignment.cache.{idx}",
            hash_overwrite=f"wei_ctc_blstm_ss4_alignment_train_{idx}"
        ) for idx in range(1, 101)]
        alignment_caches_dev = [tk.Path(
            "/u/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/mm/alignment/AlignmentJob.ETS2qXk7kdOY/output/"
            f"alignment.cache.{idx}",
            hash_overwrite=f"wei_ctc_blstm_ss4_alignment_dev_{idx}"
        ) for idx in range(1, 11)]
        allophone_file = tk.Path(
            "/u/vieting/setups/swb/20230406_feat/dependencies/allophones",
            hash_overwrite="SWB_ALLOPHONE_FILE_WEI"
        )
        state_tying_file = tk.Path(
            "/u/vieting/setups/swb/20230406_feat/dependencies/state-tying",
            hash_overwrite="SWB_STATE_TYING_FILE_MONO_EOW_NOCTX_WEI"
        )
        targets = RasrAlignmentDumpHDFJob(
            alignment_caches=alignment_caches_train + alignment_caches_dev,
            allophone_file=allophone_file,
            state_tying_file=state_tying_file,
            sparse=True,
            returnn_root=RETURNN_ROOT,
        )
        alignment = [PeakyAlignmentJob(hdf_file).out_hdf for hdf_file in targets.out_hdf_files]

        returnn_datasets["train"].ogg_args["segment_file"] = corpus.FilterSegmentsByListJob(
            {1: returnn_datasets["train"].ogg_args["segment_file"]},
            targets.out_excluded_segments,
        ).out_single_segment_files[1]
        returnn_datasets["dev"].ogg_args["segment_file"] = corpus.FilterSegmentsByListJob(
            {1: returnn_datasets["dev"].ogg_args["segment_file"]},
            targets.out_excluded_segments,
        ).out_single_segment_files[1]
    assert isinstance(alignment, list)
    assert len(alignment) > 0
    assert isinstance(alignment[0], tk.Path)

    feature_cache_bundle_train = tk.Path(
        "/u/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/features/extraction/"
        "FeatureExtraction.Gammatone.OKQT9hEV3Zgd/output/gt.cache.bundle",
        hash_overwrite="wei_ls960_gammatone_train_bundle",
        cached=False,
    )
    feature_cache_bundle_dev = tk.Path(
        "/u/zhou/asr-exps/swb1/2020-07-27_neural_transducer/work/features/extraction/"
        "FeatureExtraction.Gammatone.pp9W8m2Z8mHU/output/gt.cache.bundle",
        hash_overwrite="wei_ls960_gammatone_dev_bundle",
        cached=False,
    )
    feature_bundle = ConcatenateJob(
        [feature_cache_bundle_train, feature_cache_bundle_dev],
        zip_out=False,
        out_name="gt.cache.bundle",
    ).out

    segments.update({
        "train": returnn_datasets["train"].ogg_args["segment_file"],
        "dev": returnn_datasets["dev"].ogg_args["segment_file"],
        "dev.wei": tk.Path(
            "/u/vieting/setups/swb/20230406_feat/dependencies/segments.wei.dev",
            hash_overwrite="swb_segments_dev_wei",
        ),
    })

    def _add_targets_to_dataset(dataset: OggZipHdfDataInput) -> Dict[str, Any]:
        dataset = dataset.get_data_dict()
        if sync_ogg_audio:
            # Wei's alignment used DC-detection, so synchronize waveforms here
            ogg_zip_job = dataset["datasets"]["ogg"]["path"][0].creator
            synced_ogg_zip_job = BlissToOggZipJob(
                bliss_corpus=ogg_zip_job.bliss_corpus,
                segments=ogg_zip_job.segments,
                rasr_cache=feature_bundle,
                raw_sample_rate=ogg_zip_job.raw_sample_rate,
                feat_sample_rate=ogg_zip_job.feat_sample_rate,
                returnn_python_exe=ogg_zip_job.returnn_python_exe,
                returnn_root=ogg_zip_job.returnn_root,
            )
            synced_ogg_zip_job.rqmt = {"time": 8.0, "cpu": 2}
            dataset["datasets"]["ogg"]["path"] = [synced_ogg_zip_job.out_ogg_zip]
        dataset["datasets"]["hdf"]["files"] = alignment
        return dataset

    if features == "waveform":
        returnn_datasets["train"] = _add_targets_to_dataset(returnn_datasets["train"])
        returnn_datasets["dev"] = _add_targets_to_dataset(returnn_datasets["dev"])
        returnn_datasets["eval_datasets"]["devtrain"] = _add_targets_to_dataset(
            returnn_datasets["eval_datasets"]["devtrain"])
        returnn_datasets["eval_datasets"]["dev.wei"] = copy.deepcopy(returnn_datasets["eval_datasets"]["devtrain"])
        returnn_datasets["eval_datasets"]["dev.wei"]["datasets"]["hdf"]["seq_list_filter_file"] = (
            segments["dev.wei"]
        )
    elif features == "wei":
        feat_dataset = {
            "class": "SprintCacheDataset",
            "data": {
                "data": {
                    "filename": feature_bundle,
                    "data_type": "feat",
                },
            },
        }
        features = ReturnnDumpHDFJob(feat_dataset, returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
        for dataset in ["train", "dev"]:
            returnn_datasets[dataset] = HdfDataInput(
                features=[features.out_hdf],
                alignments=alignment,
                seq_ordering=returnn_datasets[dataset].seq_ordering,
                segment_file=returnn_datasets[dataset].ogg_args["segment_file"],
                partition_epoch=returnn_datasets[dataset].partition_epoch,
            )
            returnn_datasets[dataset].meta_dataset.seq_order_control_dataset = "alignment"
            returnn_datasets[dataset] = returnn_datasets[dataset].get_data_dict()
            returnn_datasets[dataset]["datasets"]["feat"].pop("partition_epoch")
            if returnn_datasets[dataset]["datasets"]["align"]["partition_epoch"] == 1:
                returnn_datasets[dataset]["datasets"]["align"].pop("partition_epoch")
            else:
                returnn_datasets[dataset]["partition_epoch"] = (
                    returnn_datasets[dataset]["datasets"]["align"]["partition_epoch"]
                )
        returnn_datasets["eval_datasets"] = {
            "devtrain": copy.deepcopy(returnn_datasets["dev"]),
            "dev.wei": copy.deepcopy(returnn_datasets["dev"]),
        }
        returnn_datasets["eval_datasets"]["devtrain"]["datasets"]["align"]["seq_list_filter_file"] = (
            segments["devtrain"]
        )
        returnn_datasets["eval_datasets"]["dev.wei"]["datasets"]["align"]["seq_list_filter_file"] = (
            segments["dev.wei"]
        )
    else:
        raise NotImplementedError

    # preserve hash
    for dataset in iterate_returnn_datasets(returnn_datasets):
        alignment_key = "hdf" if "hdf" in dataset["datasets"] else "align"
        feature_key = "features"
        if "ogg" in dataset["datasets"]:
            feature_key = "ogg"
        if "feat" in dataset["datasets"]:
            feature_key = "feat"
        dataset["datasets"]["alignment"] = dataset["datasets"].pop(alignment_key)
        dataset["data_map"]["classes"] = ("alignment", "data")
        if feature_key == "feat":
            dataset["datasets"]["features"] = dataset["datasets"].pop(feature_key)
            dataset["data_map"]["data"] = ("features", "data")
        else:
            dataset["partition_epoch"] = dataset["datasets"][feature_key]["partition_epoch"]
    return returnn_datasets
