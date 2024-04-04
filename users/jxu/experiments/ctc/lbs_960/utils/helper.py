import copy
import os
from typing import Any, Dict, Optional, List
from functools import lru_cache

from sisyphus import tk

from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn import BlissToOggZipJob
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection as TorchCollection,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.berger.recipe import returnn as custom_returnn
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.flashlight_phon_ctc.serializer import get_pytorch_serializer_v3
from i6_experiments.common.datasets.librispeech.corpus import get_corpus_object_dict
from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.data import get_audio_raw_datastream
from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset, MetaDataset
from i6_experiments.common.setups.serialization import Import, PartialImport
from i6_experiments.common.setups.serialization import ExternalImport
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import \
    conformer_ctc_d_model_512_num_layers_12_raw_wave as conformer_ctc
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_models.config import ModelConfiguration
from i6_experiments.common.setups.serialization import Import, SerializerObject
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection,
    PyTorchModel,
    build_config_constructor_serializers,
)


DATA_PREFIX = "lbs-960/"


# modified from i6_experiments/users/rossenbach/experiments/rescale/tedlium2_standalone_2023/data.py
def get_eow_vocab_datastream(lexicon) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream for Tedlium-2

    :param with_blank: datastream for CTC training
    """
    blacklist = {"[SILENCE]"}
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon, blacklist=blacklist)
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size
    )
    return vocab_datastream


# modified from i6_experiments/users/berger/systems/functors/rasr_base.py
def get_prior_file(checkpoint, prior_config):
    forward_job = custom_returnn.ReturnnForwardComputePriorJob(
        model_checkpoint=checkpoint,
        returnn_config=prior_config,
        returnn_root=default_tools_v2.returnn_root,
        returnn_python_exe=default_tools_v2.returnn_python_exe,
    )
    return forward_job.out_prior_txt_file


# modified from i6_experiments/users/rossenbach/experiments/rescale/tedlium2_standalone_2023/flashlight_phon_ctc/config.py
def get_search_config(
        network_module: str,
        net_args: Dict[str, Any],
        decoder: [str],
        decoder_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
        use_custom_engine=False,
        **kwargs,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN training config
    """

    # changing these does not change the hash
    post_config = {
    }

    base_config = {
        #############
        "batch_size": 24000 * 160,
        "max_seqs": 60,
        #############
        # dataset is added later in the pipeline during search_single
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    serializer = get_pytorch_serializer_v3(
        network_module=network_module,
        net_args=net_args,
        debug=debug,
        use_custom_engine=use_custom_engine,
        decoder=decoder,
        decoder_args=decoder_args,
    )
    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )
    return returnn_config


# modified from i6_experiments/users/rossenbach/experiments/rescale/tedlium2_standalone_2023/flashlight_phon_ctc/serializer.py
def get_pytorch_serializer_v3(
        network_module: str,
        net_args: Dict[str, Any],
        decoder: Optional[str] = None,
        decoder_args: Optional[Dict[str, Any]] = None,
        post_decoder_args: Optional[Dict[str, Any]] = None,
        prior=False,
        debug=False,
        **kwargs
) -> TorchCollection:
    """

    :param network_module: path to the pytorch config file containing Model
    :param net_args: extra arguments for the model
    :param decoder: path to the search decoder, if provided will link search functions
    :param decoder_args:
    :param post_decoder_args:
    :param prior: build config for prior computation
    :param debug: run training in debug mode (linking from recipe instead of copy)
    :param kwargs:
    :return:
    """
    PACKAGE = "i6_experiments.users.jxu.experiments.ctc.lbs_960"
    package = PACKAGE + ".pytorch_networks"

    # pytorch_model_import = PartialImport(
    #     code_object_path=package + ".%s.ConformerCTCModel" % network_module,
    #     unhashed_package_root=PACKAGE,
    #     hashed_arguments=net_args,
    #     unhashed_arguments={},
    #     import_as="get_model",
    # )
    # pytorch_train_step = Import(
    #     code_object_path="i6_experiments.users.berger.pytorch.train_steps.ctc.train_step",
    #     unhashed_package_root="i6_experiments.users.berger"
    # )
    # # i6_models_repo = CloneGitRepositoryJob(
    # #     url="https://github.com/rwth-i6/i6_models",
    # #     commit="1e94a4d9d1aa48fe3ac7f60de2cd7bd3fea19c3e",
    # #     checkout_folder_name="i6_models"
    # # ).out_repository
    # i6_models_repo = tk.Path("/u/jxu/setups/librispeech-960/2023-10-17-torch-conformer-ctc/recipe/i6_models")
    # i6_models_repo.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_MODELS"
    # i6_models = ExternalImport(import_path=i6_models_repo)
    #
    # serializer_objects = [
    #     i6_models,
    #     pytorch_model_import,
    #     pytorch_train_step,
    # ]
    num_outputs = 79
    model_config = conformer_ctc.get_default_config_v1(num_inputs=50, num_outputs=num_outputs)
    # additional_serializer_objects = [Import(f"i6_experiments.users.berger.pytorch.train_steps.ctc.train_step")]
    # use nick train step
    additional_serializer_objects = [Import(f"i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pytorch_networks.ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v2.train_step")]
    model_import = Import("i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.conformer_ctc_d_model_512_num_layers_12_raw_wave.ConformerCTCModel")
    serializer_objects: List[SerializerObject] = [model_import]
    constructor_call, imports = build_config_constructor_serializers(model_config, "cfg")
    serializer_objects.extend(imports)
    serializer_objects.append(constructor_call)
    model_kwargs = {}
    model_kwargs["cfg"] = CodeWrapper("cfg")
    serializer_objects.append(PyTorchModel(model_class_name=model_import.object_name, model_kwargs=model_kwargs))
    serializer_objects.extend(additional_serializer_objects)

    if decoder:
        nick_package = "i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pytorch_networks"
        nick_PACKAGE = "i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023"
        # Just a hack to test the phoneme-based recognition
        forward_step = Import(
            code_object_path=nick_package + ".%s.forward_step" % decoder,
            unhashed_package_root=nick_PACKAGE,
        )
        init_hook = PartialImport(
            code_object_path=nick_package + ".%s.forward_init_hook" % decoder,
            unhashed_package_root=nick_PACKAGE,
            hashed_arguments=decoder_args or {},
            unhashed_arguments=post_decoder_args or {},
            )
        finish_hook = Import(
            code_object_path=nick_package + ".%s.forward_finish_hook" % decoder,
            unhashed_package_root=nick_PACKAGE,
        )
        serializer_objects.extend(
            [forward_step, init_hook, finish_hook]
        )
    serializer = TorchCollection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )
    return serializer


# modified from i6_experiments/users/rossenbach/experiments/rescale/tedlium2_standalone_2023/data.py
def get_test_bliss_and_zip(corpus_key):
    """
    for now just return the original ogg zip

    :param corpus_key: e.g. "train", "dev", "test"
    :return:
    """
    bliss = get_corpus_object_dict(audio_format="wav")[corpus_key].corpus_file
    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss,
        no_conversion=False,
        returnn_python_exe=default_tools_v2.returnn_python_exe,
        returnn_root=default_tools_v2.returnn_root,
    ).out_ogg_zip
    return bliss, zip_dataset


# modified from i6_experiments/users/rossenbach/experiments/rescale/tedlium2_standalone_2023/data.py
@lru_cache()
def build_test_dataset(dataset_key: str):
    """
    :param dataset_key: test dataset to generate ("eval" or "test")
    """

    _, test_ogg = get_test_bliss_and_zip(dataset_key)
    bliss_dict = get_corpus_object_dict()  # unmodified bliss

    audio_datastream = get_audio_raw_datastream()

    data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": test_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, bliss_dict[dataset_key].corpus_file