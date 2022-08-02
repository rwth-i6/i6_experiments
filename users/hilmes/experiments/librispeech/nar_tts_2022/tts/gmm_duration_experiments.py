from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.datasets.librispeech import (
    get_corpus_object_dict,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    get_tts_data_from_rasr_alignment,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.tts.tts_pipeline import (
    get_training_config,
    tts_training,
    synthesize_with_splits,
    build_speaker_embedding_dataset
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.default_vocoder import (
    get_default_vocoder,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    get_inference_dataset,
)


def gmm_duration_cheat(rasr_alignment, rasr_allophones):
    """
    Experiments with duration predictor cheating
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
    name = "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_duration_cheat"
    (
        training_datasets,
        vocoder_data,
        new_corpus,
        durations_hdf,
    ) = get_tts_data_from_rasr_alignment(
        name + "/datasets",
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        rasr_alignment=rasr_alignment,
        rasr_allophones=rasr_allophones,
    )
    reference_corpus = get_corpus_object_dict(
        audio_format="ogg", output_prefix="corpora"
    )["train-clean-100"]
    default_vocoder = get_default_vocoder(name=name, corpus_data=vocoder_data)
    default_vocoder.train(num_epochs=100, time_rqmt=36, mem_rqmt=12)
    job_splits = 10
    synthetic_data_dict = {}

    exp_name = name + "/repeat"
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
        prefix=exp_name,
        num_epochs=200,
    )
    # synthesis

    # no cheating
    speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job
    )
    synth_dataset = get_inference_dataset(
        reference_corpus.corpus_file,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        durations=durations_hdf,
        process_corpus=False,
    )

    synth_corpus = synthesize_with_splits(
        name=exp_name + "/real",
        reference_corpus=reference_corpus.corpus_file,
        corpus_name="train-clean-100",
        job_splits=job_splits,
        datasets=synth_dataset,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root,
        checkpoint=train_job.out_checkpoints[200],
        vocoder=default_vocoder,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=False,
    )
    synthetic_data_dict["gmm_repeat_real"] = synth_corpus

    # duration cheating
    synth_corpus = synthesize_with_splits(
        name=exp_name + "/cheat",
        reference_corpus=reference_corpus.corpus_file,
        corpus_name="train-clean-100",
        job_splits=job_splits,
        datasets=synth_dataset,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root,
        checkpoint=train_job.out_checkpoints[200],
        vocoder=default_vocoder,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=False,
        use_true_durations=True,
    )
    synthetic_data_dict["gmm_repeat_cheat"] = synth_corpus

    exp_name = name + "/gauss_up"
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
        prefix=exp_name,
        num_epochs=200,
    )
    # synthesis

    speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job
    )
    synth_dataset = get_inference_dataset(
        reference_corpus.corpus_file,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        durations=durations_hdf,
        process_corpus=False,
    )

    # no cheating
    synth_corpus = synthesize_with_splits(
        name=exp_name + "/real",
        reference_corpus=reference_corpus.corpus_file,
        corpus_name="train-clean-100",
        job_splits=job_splits,
        datasets=synth_dataset,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root,
        checkpoint=train_job.out_checkpoints[200],
        vocoder=default_vocoder,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=True,
    )
    synthetic_data_dict["gmm_gauss_up_real"] = synth_corpus

    # duration cheating
    synth_corpus = synthesize_with_splits(
        name=exp_name + "/cheat",
        reference_corpus=reference_corpus.corpus_file,
        corpus_name="train-clean-100",
        job_splits=job_splits,
        datasets=synth_dataset,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root,
        checkpoint=train_job.out_checkpoints[200],
        vocoder=default_vocoder,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=True,
        use_true_durations=True,
    )
    synthetic_data_dict["gmm_gauss_up_cheat"] = synth_corpus

    return synthetic_data_dict
