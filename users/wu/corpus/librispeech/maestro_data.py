from typing import Dict, List, Optional
import copy
import textwrap
import string
import pprint

from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
import i6_core.returnn as returnn
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType
from . import data
from ..general import BasicSetupData, build_feature_label_meta_dataset_config, build_feature_hdf_dataset_config, build_feature_alignment_meta_dataset_config, filter_unk_in_corpus_object
from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat
from i6_experiments.users.wu.recipe.returnn.hdf import TextFileToTargetHdfJob
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data


def get_librispeech_data(
    num_classes: int,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    alignments: Dict[str, AlignmentData],
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    paired_key: str = "train-clean-100",
    audio_key: str = "train-other-960",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    feature_type: FeatureType = FeatureType.SAMPLES,
    dc_detection: bool = False,
    text_only_start_epoch = 10, # when to start loading text-only data in dataset
    **kwargs,
) -> BasicSetupData:
    # force the usage of these two
    cv_keys = ["dev-clean", "dev-other"]
    if dev_keys is None:
        dev_keys = ["dev-clean", "dev-other"]
    if test_keys is None:
        test_keys = ["test-clean", "test-other"]

    # ********** Data inputs **********

    audio_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy( 
        data.get_data_inputs(
            train_key=audio_key,
            cv_keys=cv_keys,
            dev_keys=dev_keys,
            test_keys=test_keys,
            ctc_lexicon=True,
            add_all_allophones=True,
            audio_format="wav",  # Note: OGGZip dataset lead to length mismatches between features and alignment
            **kwargs,
        )
    )

    paired_data_inputs, _, _, _ = copy.deepcopy( 
        data.get_data_inputs(
            train_key=paired_key,
            cv_keys=cv_keys,
            dev_keys=dev_keys,
            test_keys=test_keys,
            ctc_lexicon=True,
            add_all_allophones=True,
            audio_format="wav",  # Note: OGGZip dataset lead to length mismatches between features and alignment
            **kwargs,
        )
    )

    # ********** Train data **********

    train_lexicon = audio_data_inputs[audio_key].lexicon.filename  # use the largest possible dataset for lexicon
    eow_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon

    # paired part
    paired_data_config = build_feature_alignment_meta_dataset_config(
        data_inputs=[paired_data_inputs[paired_key]],
        feature_type=feature_type,
        alignments=[alignments[f"{paired_key}_align"]],
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
        dc_detection=dc_detection,
        extra_config={
            "partition_epoch": 1,
            "seq_ordering": "laplace:.1000",
        },
    )

    # audio only part 
    audio_data_config = build_feature_hdf_dataset_config(
        data_inputs=[audio_data_inputs[audio_key]],
        feature_type=feature_type,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
        dc_detection=dc_detection,
        extra_config={
            "partition_epoch": 1,
            "seq_ordering": "laplace:.1000",
        },
    )

    # text only part
    lm_data = get_librispeech_normalized_lm_data()
    create_text_hdf = TextFileToTargetHdfJob(
        lm_data,
        eow_lexicon,
        returnn_root,
        dim=num_classes-1,
        num_splits=100,
    )
    text_data_config = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": create_text_hdf.out_hdfs,
        'partition_epoch': 1,
        'seq_ordering': 'laplace:.1000',
    }

    # VariableDataset to load text-only data later
    train_data_config = {
        "class": "VariableDataset",
        "get_dataset": returnn.CodeWrapper("get_train_dataset"),
        "dataset_lru_cache_size": 2,
    }

    # recursive helper for using string path in ditionary
    def _resolve_sis_objects(x):
        if isinstance(x, tk.Path):
            return str(x)

        if isinstance(x, dict):
            return {k: _resolve_sis_objects(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = type(x)
            return t(_resolve_sis_objects(v) for v in x)
        return x

    audio_data_config = _resolve_sis_objects(audio_data_config)
    paired_data_config = _resolve_sis_objects(paired_data_config)
    text_data_config = _resolve_sis_objects(text_data_config)

    audio_data_config  = pprint.pformat(audio_data_config,  width=120)
    paired_data_config  = pprint.pformat(paired_data_config,  width=120)
    text_data_config  = pprint.pformat(text_data_config,  width=120)

    # Librispeech 960 has 281241 seqs -> 28000 corresponds to partition_epoch 10
    # In LM data, 40418261 - 1429869 = 38988392 seqs, partition epoch around 700
    train_data_func = DelayedFormat(
        textwrap.dedent(
            """\
                def get_train_dataset(*, self=None, epoch: int, **_):
                    sampling_sizes = {{
                        "audio":  28000,
                        "paired": 28000,
                        "text":   0 if epoch <= {text_only_start_epoch} else 56000,
                    }}
                    return {{
                        "class": "CombinedDataset",
                        "datasets": {{
                            "audio":  {audio_data_config},
                            "text":   {text_data_config},    # keep it present so data_map stays valid
                            "paired": {paired_data_config},
                        }},
                        "data_map": {{
                            ("audio",  "data"):    "audio",
                            ("text",   "data"):    "text",
                            ("paired", "data"):    "paired_audio",
                            ("paired", "classes"): "paired_align",
                        }},
                        "seq_ordering": "random",
                        "sampling_sizes": sampling_sizes,
                    }}            
            """
        ),
        audio_data_config=audio_data_config,
        text_data_config=text_data_config,
        paired_data_config=paired_data_config,
        text_only_start_epoch=text_only_start_epoch,
    )

    # ********** CV data **********
    # for paired cv: use dev_clean
    paired_cv_data_config = build_feature_alignment_meta_dataset_config(
        data_inputs=[cv_data_inputs[key] for key in cv_keys],
        feature_type=feature_type,
        alignments=[alignments[f"{key}_align"] for key in cv_keys],
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
    # for audio cv: use dev_other
    audio_cv_data_config = build_feature_hdf_dataset_config(
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
    # for text cv: also use dev_other
    text_cv_data_config = build_feature_label_meta_dataset_config(
        data_inputs=[cv_data_inputs[key] for key in cv_keys],
        feature_type=feature_type,
        lexicon=eow_lexicon,
        label_dim=num_classes-1,
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
    )["datasets"]["classes"]
    text_cv_data_config["partition_epoch"] = 1
    text_cv_data_config["seq_ordering"] = "sorted"

    cv_data_config = {
        "class": "CombinedDataset",
        "datasets": {
            "audio": audio_cv_data_config,
            "text": text_cv_data_config,
            "paired": paired_cv_data_config,
        },
        "data_map": {
            ("audio", "data"): "audio",
            ("text", "data"): "text",
            ("paired", "data"): "paired_audio",
            ("paired", "classes"): "paired_align",
        },
        "seq_ordering": "interleave",
    }

    # ********** Recog lexicon **********

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = eow_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**paired_data_inputs, **cv_data_inputs}.items()
    }
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = eow_lexicon
        filter_unk_in_corpus_object(data_input.corpus_object, eow_lexicon)  # TODO: Remove!

    return BasicSetupData(
        train_key=[audio_key, paired_key],
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{audio_key}_align", f"{paired_key}_align", *[f"{cv_key}_align" for cv_key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **audio_data_inputs,
            **paired_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    ), train_data_func 
