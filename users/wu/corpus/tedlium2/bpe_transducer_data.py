import copy
from typing import List, Optional
from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_core.tools import CloneGitRepositoryJob
from i6_experiments.users.berger.corpus.general.experiment_data import (
    ReturnnSearchSetupData,
)
from i6_experiments.users.berger.corpus.tedlium2 import get_bpe
from i6_experiments.users.berger.recipe.lexicon import DeleteEmptyOrthJob, MakeBlankLexiconJob
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from ..general import build_feature_label_meta_dataset_config, build_feature_hdf_dataset_config
from sisyphus import tk

from . import data


def get_tedlium2_data_dumped_bpe_labels(
    num_classes: int,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    add_unknown: bool = False,
    augmented_lexicon: bool = True,
    feature_type: FeatureType = FeatureType.GAMMATONE_16K,
    bpe_size: int = 500,
) -> ReturnnSearchSetupData:
    if cv_keys is None:
        cv_keys = ["dev"]
    if dev_keys is None:
        dev_keys = ["dev"]
    if test_keys is None:
        test_keys = ["test"]

    # ********** Data inputs **********
    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy(
        data.get_data_inputs(
            ctc_lexicon=False,
            use_augmented_lexicon=augmented_lexicon,
            add_all_allophones=True,
            add_unknown_phoneme_and_mapping=add_unknown,
            filter_unk_from_corpus=True,
        )
    )

    # ********** BPE **********

    subword_nmt_repo = CloneGitRepositoryJob("https://github.com/albertz/subword-nmt.git").out_repository
    bpe_job = get_bpe(bpe_size, subword_nmt_repo)
    bpe_lex = CreateBPELexiconJob(
        base_lexicon_path=train_data_inputs[train_key].lexicon.filename,
        bpe_codes=bpe_job.out_bpe_codes,
        bpe_vocab=bpe_job.out_bpe_vocab,
        subword_nmt_repo=subword_nmt_repo,
        vocab_blacklist=["<s>", "</s>"],
        keep_special_lemmas=True,
    ).out_lexicon
    bpe_lex = DeleteEmptyOrthJob(bpe_lex).out_lexicon
    bpe_lex = MakeBlankLexiconJob(bpe_lex).out_lexicon

    # ********** Train data **********

    train_data_config = build_feature_label_meta_dataset_config(
        label_dim=num_classes - 1,
        data_inputs=[train_data_inputs[train_key]],
        lexicon=bpe_lex,
        feature_type=feature_type,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
        extra_config={
            "partition_epoch": 5,
            "seq_ordering": "laplace:.1000",
        },
    )

    # ********** CV data **********

    cv_data_config = build_feature_label_meta_dataset_config(
        label_dim=num_classes - 1,
        data_inputs=[cv_data_inputs[key] for key in cv_keys],
        lexicon=bpe_lex,
        feature_type=feature_type,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
        single_hdf=True,
        extra_config={
            "partition_epoch": 1,
            "seq_ordering": "sorted",
        },
    )

    # ********** forward data **********

    forward_data_config = {
        key: build_feature_hdf_dataset_config(
            data_inputs=[data_input],
            feature_type=feature_type,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            single_hdf=True,
            extra_config={
                "partition_epoch": 1,
                "seq_ordering": "sorted",
            },
        )
        for key, data_input in {**dev_data_inputs, **test_data_inputs}.items()
    }

    # ********** Recog lexicon **********

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = bpe_lex

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = bpe_lex

    return ReturnnSearchSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        forward_data_config=forward_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )
