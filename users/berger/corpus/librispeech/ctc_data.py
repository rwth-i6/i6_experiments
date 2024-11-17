from typing import List, Optional
import copy

from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from i6_experiments.users.berger.corpus.general.experiment_data import PytorchCTCSetupData
from i6_experiments.users.berger.corpus.general.hdf import build_feature_label_meta_dataset_config
from i6_experiments.users.berger.recipe.lexicon import CreateBPELexiconJob
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from . import data
from ..general import CTCSetupData, build_feature_hdf_dataset_config
from sisyphus import tk
from i6_experiments.users.berger.corpus.general.ogg import (
    build_oggzip_dataset_config,
    build_oggzip_label_meta_dataset_config,
)


def get_librispeech_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train-other-960",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    use_wei_lexicon: bool = False,
    feature_type: FeatureType = FeatureType.SAMPLES,
    dc_detection: bool = False,
    partition_epoch: int = 20,
    ogg_dataset: bool = False,
    **kwargs,
) -> CTCSetupData:
    if cv_keys is None:
        cv_keys = ["dev-clean", "dev-other"]
    if dev_keys is None:
        dev_keys = ["dev-clean", "dev-other"]
    if test_keys is None:
        test_keys = ["test-clean", "test-other"]

    # ********** Data inputs **********

    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = data.get_data_inputs(
        train_key=train_key,
        cv_keys=cv_keys,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        use_wei_lexicon=use_wei_lexicon,
        add_all_allophones=True,
        audio_format="wav",
        **kwargs,
    )

    # ********** Train data **********

    train_lexicon = train_data_inputs[train_key].lexicon.filename

    if ogg_dataset:
        train_dataset_builder = MetaDatasetBuilder()
        train_dataset_builder.add_dataset(
            name="data",
            dataset_config=build_oggzip_dataset_config(
                data_inputs=[train_data_inputs[train_key]],
                returnn_root=returnn_root,
                returnn_python_exe=returnn_python_exe,
                audio_config={
                    "features": "raw",
                    "peak_normalization": True,
                },
                extra_config={
                    "partition_epoch": partition_epoch,
                    "seq_ordering": "laplace:.1000",
                },
            ),
            key_mapping={"data": "data"},
            control=True,
        )
        train_data_config = train_dataset_builder.get_dict()
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
                "partition_epoch": partition_epoch,
                "seq_ordering": "laplace:.1000",
            },
        )

    # ********** CV data **********

    if ogg_dataset:
        cv_dataset_builder = MetaDatasetBuilder()
        cv_dataset_builder.add_dataset(
            name="data",
            dataset_config=build_oggzip_dataset_config(
                data_inputs=[cv_data_inputs[key] for key in cv_keys],
                returnn_root=returnn_root,
                returnn_python_exe=returnn_python_exe,
                audio_config={
                    "features": "raw",
                    "peak_normalization": True,
                },
                extra_config={
                    "partition_epoch": 1,
                    "seq_ordering": "sorted",
                },
            ),
            key_mapping={"data": "data"},
            control=True,
        )
        cv_data_config = cv_dataset_builder.get_dict()
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

    loss_corpus = corpus.MergeCorporaJob(
        [train_data_inputs[train_key].corpus_object.corpus_file]
        + [cv_data_inputs[key].corpus_object.corpus_file for key in cv_keys],
        name="loss-corpus",
        merge_strategy=corpus.MergeStrategy.SUBCORPORA,
    ).out_merged_corpus
    loss_lexicon = train_lexicon

    # ********** Recog lexicon **********

    if use_wei_lexicon:
        recog_lexicon = tk.Path("/work/asr4/berger/dependencies/librispeech/lexicon/recog.lexicon.wei.xml")
        for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
            rasr_input.lexicon.filename = recog_lexicon
    else:
        for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
            rasr_input.lexicon = copy.deepcopy(rasr_input.lexicon)
            rasr_input.lexicon.filename = AddEowPhonemesToLexiconJob(rasr_input.lexicon.filename).out_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    align_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon
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


def get_librispeech_data_dumped_labels(
    num_classes: int,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train-other-960",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    feature_type: FeatureType = FeatureType.SAMPLES,
    dc_detection: bool = False,
    partition_epoch: int = 20,
    ogg_dataset: bool = False,
    **kwargs,
) -> PytorchCTCSetupData:
    if cv_keys is None:
        cv_keys = ["dev-clean", "dev-other"]
    if dev_keys is None:
        dev_keys = ["dev-clean", "dev-other"]
    if test_keys is None:
        test_keys = ["test-clean", "test-other"]

    # ********** Data inputs **********

    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy(
        data.get_data_inputs(
            train_key=train_key,
            cv_keys=cv_keys,
            dev_keys=dev_keys,
            test_keys=test_keys,
            ctc_lexicon=True,
            add_all_allophones=True,
            audio_format="wav",
            **kwargs,
        )
    )

    # ********** Train data **********

    train_lexicon = train_data_inputs[train_key].lexicon.filename
    eow_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon

    if ogg_dataset:
        train_data_config = build_oggzip_label_meta_dataset_config(
            label_dim=num_classes - 1,
            data_inputs=[train_data_inputs[train_key]],
            lexicon=eow_lexicon,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            audio_config={
                "features": "raw",
                "peak_normalization": True,
            },
            extra_config={
                "partition_epoch": partition_epoch,
                "seq_ordering": "laplace:.1000",
            },
        )
    else:
        train_data_config = build_feature_label_meta_dataset_config(
            label_dim=num_classes - 1,
            data_inputs=[train_data_inputs[train_key]],
            lexicon=eow_lexicon,
            feature_type=feature_type,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            dc_detection=dc_detection,
            extra_config={
                "partition_epoch": partition_epoch,
                "seq_ordering": "laplace:.1000",
            },
        )

    # ********** CV data **********

    if ogg_dataset:
        cv_data_config = build_oggzip_label_meta_dataset_config(
            label_dim=num_classes - 1,
            data_inputs=[cv_data_inputs[key] for key in cv_keys],
            lexicon=eow_lexicon,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            audio_config={
                "features": "raw",
                "peak_normalization": True,
            },
            extra_config={
                "partition_epoch": 1,
                "seq_ordering": "sorted",
            },
        )
    else:
        cv_data_config = build_feature_label_meta_dataset_config(
            label_dim=num_classes - 1,
            data_inputs=[cv_data_inputs[key] for key in cv_keys],
            lexicon=eow_lexicon,
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

    # ********** Recog lexicon **********

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = eow_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = eow_lexicon

    return PytorchCTCSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )


def get_librispeech_data_bpe(
    bpe_size: int,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    train_key: str = "train-other-960",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    partition_epoch: int = 20,
    **kwargs,
) -> PytorchCTCSetupData:
    if cv_keys is None:
        cv_keys = ["dev-clean", "dev-other"]
    if dev_keys is None:
        dev_keys = ["dev-clean", "dev-other"]
    if test_keys is None:
        test_keys = ["test-clean", "test-other"]

    # ********** Data inputs **********

    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy(
        data.get_data_inputs(
            train_key=train_key,
            cv_keys=cv_keys,
            dev_keys=dev_keys,
            test_keys=test_keys,
            ctc_lexicon=True,
            add_all_allophones=True,
            audio_format="wav",
            **kwargs,
        )
    )

    # ********** Train data **********

    audio_config = {
        "features": "raw",
        "peak_normalization": True,
    }

    bpe_settings = data.get_bpe(bpe_size)
    bpe_config = {
        "class": "BytePairEncoding",
        "unknown_label": None,
        "bpe_file": bpe_settings.bpe_codes,
        "vocab_file": bpe_settings.bpe_vocab,
    }

    train_dataset_builder = MetaDatasetBuilder()
    train_dataset_builder.add_dataset(
        name="data",
        dataset_config=build_oggzip_dataset_config(
            data_inputs=[train_data_inputs[train_key]],
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            audio_config=audio_config,
            target_config=bpe_config,
            extra_config={
                "partition_epoch": partition_epoch,
                "seq_ordering": "laplace:.1000",
            },
        ),
        key_mapping={"data": "data", "classes": "classes"},
        control=True,
    )
    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********

    cv_dataset_builder = MetaDatasetBuilder()
    cv_dataset_builder.add_dataset(
        name="data",
        dataset_config=build_oggzip_dataset_config(
            data_inputs=[cv_data_inputs[key] for key in cv_keys],
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            audio_config=audio_config,
            target_config=bpe_config,
            extra_config={
                "partition_epoch": 1,
                "seq_ordering": "sorted",
            },
        ),
        key_mapping={"data": "data", "classes": "classes"},
        control=True,
    )
    cv_data_config = cv_dataset_builder.get_dict()

    # ********** Recog lexicon **********

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon = copy.deepcopy(rasr_input.lexicon)
        rasr_input.lexicon.filename = CreateBPELexiconJob(
            base_lexicon_path=rasr_input.lexicon.filename,
            bpe_codes=bpe_settings.bpe_codes,
            bpe_vocab=bpe_settings.bpe_vocab,
            subword_nmt_repo=bpe_settings.subword_nmt_repo_path,
        ).out_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    for data_input in align_data_inputs.values():
        data_input.lexicon = copy.deepcopy(data_input.lexicon)
        data_input.lexicon.filename = CreateBPELexiconJob(
            base_lexicon_path=data_input.lexicon.filename,
            bpe_codes=bpe_settings.bpe_codes,
            bpe_vocab=bpe_settings.bpe_vocab,
            subword_nmt_repo=bpe_settings.subword_nmt_repo_path,
        ).out_lexicon

    return PytorchCTCSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )
