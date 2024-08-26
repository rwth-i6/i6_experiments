from typing import Dict, List, Optional

from i6_core.returnn.oggzip import BlissToOggZipJob
from sisyphus import tk

from i6_experiments.users.berger.args.returnn.dataset import (
    MetaDatasetBuilder,
    hdf_config_dict_for_files,
    oggzip_config_dict_for_files,
)
from i6_experiments.users.berger.helpers.rasr import RasrDataInput
from i6_experiments.users.berger.recipe.corpus.transform import ReplaceUnknownWordsJob
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob


def build_oggzip_datset_config(
    data_inputs: List[RasrDataInput],
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    audio_config: Optional[dict] = None,
    extra_config: Optional[dict] = None,
    segment_files: Optional[Dict[int, tk.Path]] = None,
) -> dict:
    oggzip_files = [
        BlissToOggZipJob(
            bliss_corpus=data_input.corpus_object.corpus_file,
            segments=segment_files,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
        ).out_ogg_zip
        for data_input in data_inputs
    ]
    return oggzip_config_dict_for_files(oggzip_files, audio_config=audio_config, extra_config=extra_config)


def build_oggzip_label_meta_dataset_config(
    data_inputs: List[RasrDataInput],
    lexicon: tk.Path,
    label_dim: int,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    audio_config: Optional[dict] = None,
    extra_config: Optional[dict] = None,
    segment_files: Optional[Dict[int, tk.Path]] = None,
) -> dict:
    dataset_builder = MetaDatasetBuilder()

    feature_oggzip_config = build_oggzip_datset_config(
        data_inputs=data_inputs,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        audio_config=audio_config,
        extra_config=extra_config,
        segment_files=segment_files,
    )
    dataset_builder.add_dataset(
        name="data", dataset_config=feature_oggzip_config, key_mapping={"data": "data"}, control=True
    )

    label_hdf_files = [
        BlissCorpusToTargetHdfJob(
            ReplaceUnknownWordsJob(data_input.corpus_object.corpus_file, lexicon_file=lexicon).out_corpus_file,
            bliss_lexicon=lexicon,
            returnn_root=returnn_root,
            dim=label_dim,
        ).out_hdf
        for data_input in data_inputs
    ]
    label_hdf_config = hdf_config_dict_for_files(files=label_hdf_files)
    dataset_builder.add_dataset(
        name="classes", dataset_config=label_hdf_config, key_mapping={"data": "classes"}, control=False
    )
    return dataset_builder.get_dict()
