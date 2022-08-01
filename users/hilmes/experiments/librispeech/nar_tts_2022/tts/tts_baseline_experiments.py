"""
Pipeline file for experiments with the standard CTC TTS model
"""
from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.datasets.librispeech import (
    get_corpus_object_dict,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    get_tts_data_from_ctc_align,
    get_tts_data_from_rasr_alignment,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.ctc_align.ctc_experiments import (
    get_baseline_ctc_alignment,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.tts.tts_pipeline import (
    get_training_config,
    tts_training,
    get_forward_config,
    tts_forward,
    gl_swer,
    synthesize_with_splits,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.default_vocoder import (
    get_default_vocoder,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    TTSForwardData,
)


def ctc_baseline():
    """
    baseline for returnn_common ctc model with network constructor
    :return:
    """
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
    ).out_repository
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="79876b18552f61a3af7c21c670475fee51ef3991",
        checkout_folder_name="returnn_common",
    ).out_repository
    name = (
        "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/ctc_baseline"
    )
    alignment = get_baseline_ctc_alignment()
    training_datasets, vocoder_data = get_tts_data_from_ctc_align(
        name + "/datasets",
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        alignment=alignment,
    )
    default_vocoder = get_default_vocoder(name=name, corpus_data=vocoder_data)
    default_vocoder.train(num_epochs=100, time_rqmt=36, mem_rqmt=12)

    name = name + "/repeat"
    train_config = get_training_config(
        returnn_common_root=returnn_common_root,
        training_datasets=training_datasets,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=False,
    )
    train_job = tts_training(
        config=train_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=name,
        num_epochs=200,
    )
    forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(
            dataset=training_datasets.cv, datastreams=training_datasets.datastreams
        ),
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=False,
    )
    gl_swer(
        name=name,
        vocoder=default_vocoder,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
    )

    name = name + "/gauss_up"
    train_config = get_training_config(
        returnn_common_root=returnn_common_root,
        training_datasets=training_datasets,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=True,
    )
    train_job = tts_training(
        config=train_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=name,
        num_epochs=200,
    )
    forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(
            dataset=training_datasets.cv, datastreams=training_datasets.datastreams
        ),
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=True,
    )
    gl_swer(
        name=name,
        vocoder=default_vocoder,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
    )
