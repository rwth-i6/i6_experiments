import copy
from i6_core import corpus
from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_core.tools import CloneGitRepositoryJob
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from i6_experiments.users.berger.corpus.general.experiment_data import (
    BpeSetupData,
)
from i6_experiments.users.berger.corpus.tedlium2 import get_bpe
from i6_experiments.users.berger.helpers.hdf import build_rasr_feature_hdfs
from i6_experiments.users.berger.recipe.lexicon import CreateBPELexiconJob, DeleteEmptyOrthJob, MakeBlankLexiconJob
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from sisyphus import tk

from . import data


def get_tedlium2_pytorch_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    add_unknown: bool = False,
    augmented_lexicon: bool = True,
    feature_type: FeatureType = FeatureType.GAMMATONE_16K,
    bpe_size: int = 500,
) -> BpeSetupData:
    # ********** Data inputs **********
    train_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy(
        data.get_data_inputs(
            ctc_lexicon=False,
            use_augmented_lexicon=augmented_lexicon,
            add_all_allophones=True,
            add_unknown_phoneme_and_mapping=add_unknown,
        )
    )

    # ********** BPE **********

    subword_nmt_repo = CloneGitRepositoryJob("https://github.com/albertz/subword-nmt.git").out_repository
    bpe_job = get_bpe(bpe_size)
    bpe_lex = CreateBPELexiconJob(
        train_data_inputs["train"].lexicon.filename,
        bpe_codes=bpe_job.out_bpe_codes,
        bpe_vocab=bpe_job.out_bpe_vocab,
        subword_nmt_repo=subword_nmt_repo,
    ).out_lexicon
    bpe_lex = DeleteEmptyOrthJob(bpe_lex).out_lexicon
    bpe_lex = MakeBlankLexiconJob(bpe_lex).out_lexicon

    # ********** Train data **********
    train_corpus_object = copy.deepcopy(train_data_inputs["train"].corpus_object)
    assert train_corpus_object.corpus_file is not None
    train_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
        train_corpus_object.corpus_file, bpe_lex, all_unknown=False
    ).out_corpus

    train_dataset_builder = MetaDatasetBuilder()
    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        train_feature_hdf = build_rasr_feature_hdfs(
            train_corpus_object,
            split=train_data_inputs["train"].concurrent,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        train_feature_hdf = [
            BlissToPcmHDFJob(
                train_corpus_object.corpus_file,
                rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
                returnn_root=returnn_root,
            ).out_hdf
        ]
    else:
        raise NotImplementedError

    train_dataset_builder.add_hdf_dataset(
        train_feature_hdf,
        name="features",
        key_mapping={"data": "data"},
        dataset_config={
            "partition_epoch": 5,
            "seq_ordering": "laplace:.1000",
        },
        control=True,
    )

    train_targets_hdf = BlissCorpusToTargetHdfJob(
        train_corpus_object.corpus_file, bpe_lex, returnn_root=returnn_root
    ).out_hdf
    train_dataset_builder.add_hdf_dataset(
        train_targets_hdf,
        name="targets",
        key_mapping={"data": "targets"},
        control=False,
    )

    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********
    cv_corpus_object = copy.deepcopy(dev_data_inputs["dev"].corpus_object)
    assert cv_corpus_object.corpus_file is not None
    cv_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
        cv_corpus_object.corpus_file, bpe_lex, all_unknown=False
    ).out_corpus

    cv_dataset_builder = MetaDatasetBuilder()

    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        cv_feature_hdf = build_rasr_feature_hdfs(
            cv_corpus_object,
            split=1,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        cv_feature_hdf = [
            BlissToPcmHDFJob(
                cv_corpus_object.corpus_file,
                rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
                returnn_root=returnn_root,
            ).out_hdf
        ]
    else:
        raise NotImplementedError

    cv_dataset_builder.add_hdf_dataset(
        cv_feature_hdf,
        name="features",
        key_mapping={"data": "data"},
        dataset_config={
            "partition_epoch": 1,
            "seq_ordering": "sorted",
        },
        control=True,
    )

    cv_targets_hdf = BlissCorpusToTargetHdfJob(cv_corpus_object.corpus_file, bpe_lex, returnn_root=returnn_root).out_hdf
    cv_dataset_builder.add_hdf_dataset(
        cv_targets_hdf,
        name="targets",
        key_mapping={"data": "targets"},
        control=False,
    )

    cv_data_config = cv_dataset_builder.get_dict()

    # ********** dev data **********
    dev_corpus_object = copy.deepcopy(dev_data_inputs["dev"].corpus_object)
    assert dev_corpus_object.corpus_file is not None

    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        dev_feature_hdf = build_rasr_feature_hdfs(
            dev_corpus_object,
            split=1,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        dev_feature_hdf = [
            BlissToPcmHDFJob(
                dev_corpus_object.corpus_file,
                rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
                returnn_root=returnn_root,
            ).out_hdf
        ]
    else:
        raise NotImplementedError

    dev_data_config = {
        "class": "HDFDataset",
        "files": dev_feature_hdf,
        "use_cache_manager": True,
        "seq_ordering": "sorted",
        "partition_epoch": 1,
    }

    # ********** eval data **********
    test_corpus_object = copy.deepcopy(test_data_inputs["test"].corpus_object)
    assert test_corpus_object.corpus_file is not None

    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        test_feature_hdf = build_rasr_feature_hdfs(
            test_corpus_object,
            split=1,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        test_feature_hdf = [
            BlissToPcmHDFJob(
                dev_corpus_object.corpus_file,
                rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
                returnn_root=returnn_root,
            ).out_hdf
        ]
    else:
        raise NotImplementedError

    test_data_config = {
        "class": "HDFDataset",
        "files": test_feature_hdf,
        "use_cache_manager": True,
        "seq_ordering": "sorted",
        "partition_epoch": 1,
    }

    return BpeSetupData(
        train_key="train",
        dev_keys=["dev"],
        test_keys=["test"],
        align_keys=[],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        forward_data_config={
            "dev": dev_data_config,
            "test": test_data_config,
        },
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
        },
        bpe_lexicon=bpe_lex,
    )
