from typing import List, Optional
import copy

from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from i6_experiments.users.berger.recipe.lexicon.modification import EnsureSilenceFirstJob
from . import data
from ..general import CTCSetupData, build_feature_hdf_dataset_config
from sisyphus import tk


def get_switchboard_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    feature_type: FeatureType = FeatureType.SAMPLES,
    dc_detection: bool = False,
    use_wei_data: bool = False,
    **kwargs,
) -> CTCSetupData:
    if cv_keys is None:
        cv_keys = ["hub5e00"]
    if dev_keys is None:
        dev_keys = ["hub5e00"]
    if test_keys is None:
        test_keys = ["hub5e01", "rt03s"]

    # ********** Data inputs **********

    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = data.get_data_inputs(
        train_key=train_key,
        cv_keys=cv_keys,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        add_all_allophones=True,
        **kwargs,
    )

    # ********** Train data **********

    if use_wei_data:
        config_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/sprint.train.config")
        feature_flow_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/train.feature.flow")
        train_data_config = {
            "class": "ExternSprintDataset",
            "partitionEpoch": 6,
            "sprintConfigStr": f"--config={config_file} --*.LOGFILE=nn-trainer.train.log --*.TASK=1 "
            f"--*.corpus.segment-order-shuffle=true --*.segment-order-sort-by-time-length=true "
            f"--*.segment-order-sort-by-time-length-chunk-size=300 --feature-extraction.file={feature_flow_file}",
            "sprintTrainerExecPath": rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}"),
        }
        train_lexicon = tk.Path("/work/asr4/berger/dependencies/switchboard/lexicon/wei_train_ctc.lexicon.v2.xml")
    else:
        train_data_config = build_feature_hdf_dataset_config(
            data_inputs=[train_data_inputs[train_key]],
            feature_type=feature_type,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            dc_detection=dc_detection,
            extra_config={
                "partition_epoch": 6,
                "seq_ordering": "laplace:.1000",
            },
        )
        train_lexicon = train_data_inputs[train_key].lexicon.filename

    # ********** CV data **********

    if use_wei_data:
        config_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/sprint.dev.config")
        feature_flow_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/dev.feature.flow")
        cv_data_config = {
            "class": "ExternSprintDataset",
            "partitionEpoch": 1,
            "sprintConfigStr": f"--config={config_file} --*.LOGFILE=nn-trainer.dev.log --*.TASK=1 "
            f"--*.corpus.segment-order-shuffle=true --*.segment-order-sort-by-time-length=true "
            f"--*.segment-order-sort-by-time-length-chunk-size=50 --feature-extraction.file={feature_flow_file}",
            "sprintTrainerExecPath": rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}"),
        }
    else:
        cv_data_config = build_feature_hdf_dataset_config(
            data_inputs=[cv_data_inputs[key] for key in cv_keys],
            feature_type=feature_type,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            dc_detection=dc_detection,
            single_hdf=True,
            extra_config={
                "partition_epoch": 1,
                "seq_ordering": "sorted",
            },
        )

    # ********** Loss corpus **********

    if use_wei_data:
        loss_corpus = tk.Path("/work/asr4/berger/dependencies/switchboard/corpus/wei_train-dev.corpus.gz")
    else:
        loss_corpus = corpus.MergeCorporaJob(
            [train_data_inputs[train_key].corpus_object.corpus_file]
            + [cv_data_inputs[key].corpus_object.corpus_file for key in cv_keys],
            name="loss-corpus",
            merge_strategy=corpus.MergeStrategy.SUBCORPORA,
        ).out_merged_corpus

    loss_lexicon = train_lexicon

    # ********** Recog lexicon **********

    recog_lexicon = EnsureSilenceFirstJob(train_lexicon).out_lexicon
    recog_lexicon = AddEowPhonemesToLexiconJob(
        recog_lexicon, nonword_phones=["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]
    ).out_lexicon

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = recog_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    align_lexicon = EnsureSilenceFirstJob(train_lexicon).out_lexicon
    align_lexicon = AddEowPhonemesToLexiconJob(
        align_lexicon, nonword_phones=["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]
    ).out_lexicon

    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = align_lexicon

    return CTCSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        loss_corpus=loss_corpus,
        loss_lexicon=loss_lexicon,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )
