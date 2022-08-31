"""
Pipeline file for experiments with the standard CTC TTS model
"""
from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    get_tts_data_from_ctc_align, get_librispeech_tts_segments
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.ctc_align.ctc_experiments import (
    get_baseline_ctc_alignment, get_loss_scale_alignments
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.tts.tts_pipeline import (
    get_training_config,
    tts_training,
    get_forward_config,
    gl_swer,
    synthesize_with_splits,
    build_speaker_embedding_dataset,
    build_vae_speaker_prior_dataset
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.default_vocoder import (
    get_default_vocoder,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    TTSForwardData,
    get_inference_dataset_old,
    get_inference_dataset,
    get_ls_100_f0_hdf,
    extend_meta_datasets_with_f0,
    extend_meta_datasets_with_pitch
)
from i6_experiments.common.datasets.librispeech import (
    get_corpus_object_dict
)
from i6_experiments.users.hilmes.experiments.librispeech.util.asr_evaluation import (
    asr_evaluation,
)
from copy import deepcopy

def ctc_baseline():
    """
    baseline for returnn_common ctc model with network constructor
    :return:
    """
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_exe = tk.Path(
        "/u/hilmes/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root_job = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="240f119b54d52a4324ab300c301f8e003e0a398c",
    )
    returnn_root_job.hash_overwrite = "ctc_baseline_returnn"
    returnn_root = returnn_root_job.out_repository
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="79876b18552f61a3af7c21c670475fee51ef3991",
        checkout_folder_name="returnn_common",
    ).out_repository
    name = (
        "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/ctc_baseline"
    )
    alignment = get_baseline_ctc_alignment()
    training_datasets, corpus, durations = get_tts_data_from_ctc_align(
        name + "/datasets",
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        alignment=alignment,
    )
    reference_corpus = get_corpus_object_dict(
        audio_format="ogg", output_prefix="corpora"
    )["train-clean-100"]
    default_vocoder = get_default_vocoder(name=name)
    synthetic_data_dict = {}
    job_splits = 10

    librispeech_trafo = tk.Path(
        "/u/rossenbach/experiments/librispeech_tts/config/evaluation/asr/pretrained_configs/trafo.specaug4.12l.ffdim4."
        "pretrain3.natctc_recognize_pretrained.config"
    )
    train_segments, cv_segments = get_librispeech_tts_segments()
    asr_evaluation(
        config_file=librispeech_trafo,
        corpus=reference_corpus.corpus_file,
        output_path=name,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_exe,
        segment_file=cv_segments
    )

    for upsampling in ["repeat", "gauss"]:
        exp_name = name + f"/{upsampling}"
        train_config = get_training_config(
            returnn_common_root=returnn_common_root,
            training_datasets=training_datasets,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
        )
        if upsampling == "gauss":
            train_config.config["learning_rates"] = [0.0001, 0.001]
        train_job = tts_training(
            config=train_config,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            prefix=exp_name,
            num_epochs=200,
        )
        forward_config = get_forward_config(
            returnn_common_root=returnn_common_root,
            forward_dataset=TTSForwardData(
                dataset=training_datasets.cv, datastreams=training_datasets.datastreams
            ),
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            calc_speaker_embedding=True,
        )
        gl_swer(
            name=exp_name,
            vocoder=default_vocoder,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            checkpoint=train_job.out_checkpoints[200],
            config=forward_config,
        )
        speaker_embedding_hdf = build_speaker_embedding_dataset(
          returnn_common_root=returnn_common_root,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root,
          datasets=training_datasets,
          prefix=exp_name,
          train_job=train_job
        )
        synth_dataset = get_inference_dataset_old(
          corpus,
          returnn_root=returnn_root,
          returnn_exe=returnn_exe,
          datastreams=training_datasets.datastreams,
          speaker_embedding_hdf=speaker_embedding_hdf,
          durations=durations,
          process_corpus=False,
        )
        for duration in ["pred", "cheat"]:
            synth_corpus = synthesize_with_splits(
                name=exp_name + f"/{duration}",
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
                gauss_up=(upsampling == "gauss"),
                use_true_durations=(duration == "cheat"),
            )
            synthetic_data_dict[f"ctc_{upsampling}_{duration}"] = synth_corpus


    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
        checkout_folder_name="returnn_common",
    ).out_repository
    returnn_root_job = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="e75ee20b5830808062aac2821512131fdffc521d",  # fix dim tag declare same as
    )
    returnn_root = returnn_root_job.out_repository
    exp_name = name + "/vae"
    vae_dataset = deepcopy(training_datasets)
    vae_dataset.datastreams["audio_features"].available_for_inference = True
    train_config = get_training_config(
        returnn_common_root=returnn_common_root,
        training_datasets=vae_dataset,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=False,
        use_vae=True,
        batch_size=12000
    )
    train_job = tts_training(
        config=train_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=exp_name,
        num_epochs=200,
    )
    vae_swer_dataset = deepcopy(training_datasets.cv)
    vae_swer_datastreams = deepcopy(training_datasets.datastreams)
    vae_swer_datastreams["audio_features"].available_for_inference = True
    forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(
            dataset=vae_swer_dataset, datastreams=vae_swer_datastreams
        ),
        embedding_size=256,
        speaker_embedding_size=256,
        calc_speaker_embedding=True,
        use_vae=True,
        use_audio_data=True,
    )
    gl_swer(
        name=exp_name,
        vocoder=default_vocoder,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
    )
    vae_dataset = deepcopy(training_datasets.cv)
    vae_dataset.datasets["audio"]["segment_file"] = None
    vae_datastreams = deepcopy(training_datasets.datastreams)
    vae_datastreams["audio_features"].available_for_inference = True
    speaker_prior_hdf = build_vae_speaker_prior_dataset(
          returnn_common_root=returnn_common_root,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root,
          dataset=vae_dataset,
          datastreams=vae_datastreams,
          prefix=exp_name,
          train_job=train_job,
          corpus=reference_corpus.corpus_file
        )
    speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job
    )
    synth_dataset = get_inference_dataset(
        corpus,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        speaker_prior_hdf=speaker_prior_hdf,
        durations=durations,
        process_corpus=False,
    )
    for duration in ["pred", "cheat"]:
        synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{duration}",
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
            use_true_durations=(duration == "cheat"),
            use_vae=True,
            use_calculated_prior=True,
            batch_size=3000 if (duration == "pred") else 4000,
        )
        synthetic_data_dict[f"ctc_vae_{duration}"] = synth_corpus


    exp_name = name + "/f0_pred"
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
        checkout_folder_name="returnn_common",
    ).out_repository
    returnn_root_job = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="ce4366ff0caafc2e4b349fd2a189870f3c76f630",  # fixes shape error in normal
    )
    returnn_root = returnn_root_job.out_repository
    f0_hdf = get_ls_100_f0_hdf(durations=durations, returnn_exe=returnn_exe, returnn_root=returnn_root, prefix=exp_name)
    f0_training_datasets = extend_meta_datasets_with_f0(datasets=training_datasets, f0_dataset=f0_hdf)

    train_config = get_training_config(
        returnn_common_root=returnn_common_root,
        training_datasets=f0_training_datasets,
        embedding_size=256,
        speaker_embedding_size=256,
        use_pitch_pred=True,
    )
    train_job = tts_training(
        config=train_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=exp_name,
        num_epochs=200,
    )
    forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(
            dataset=f0_training_datasets.cv, datastreams=f0_training_datasets.datastreams
        ),
        embedding_size=256,
        speaker_embedding_size=256,
        calc_speaker_embedding=True,
        use_pitch_prediction=True
    )
    gl_swer(
        name=exp_name,
        vocoder=default_vocoder,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
    )
    speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job
    )
    synth_dataset = get_inference_dataset(
        corpus,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        speaker_prior_hdf=speaker_prior_hdf,
        durations=durations,
        process_corpus=False,
    )
    for duration in ["pred", "cheat"]:
        synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{duration}",
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
            use_true_durations=(duration == "cheat"),
            use_pitch_prediction=True,
        )
        synthetic_data_dict[f"ctc_f0_{duration}"] = synth_corpus

    exp_name = name + "/energy_pred"
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
        checkout_folder_name="returnn_common",
    ).out_repository
    returnn_root_job = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="ce4366ff0caafc2e4b349fd2a189870f3c76f630",  # fixes shape error in normal
    )
    returnn_root = returnn_root_job.out_repository
    pitch_training_datasets = extend_meta_datasets_with_pitch(datasets=training_datasets)

    train_config = get_training_config(
        returnn_common_root=returnn_common_root,
        training_datasets=pitch_training_datasets,
        embedding_size=256,
        speaker_embedding_size=256,
        use_energy_prediction=True,
    )
    train_job = tts_training(
        config=train_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=exp_name,
        num_epochs=200,
    )
    forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(
            dataset=pitch_training_datasets.cv, datastreams=pitch_training_datasets.datastreams
        ),
        embedding_size=256,
        speaker_embedding_size=256,
        calc_speaker_embedding=True,
        use_energy_prediction=True
    )
    gl_swer(
        name=exp_name,
        vocoder=default_vocoder,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
    )
    speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job
    )
    synth_dataset = get_inference_dataset(
        corpus,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        speaker_prior_hdf=speaker_prior_hdf,
        durations=durations,
        process_corpus=False,
    )
    for duration in ["pred", "cheat"]:
        synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{duration}",
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
            use_true_durations=(duration == "cheat"),
            use_energy_prediction=True,
        )
        synthetic_data_dict[f"ctc_energy_{duration}"] = synth_corpus

    return synthetic_data_dict


def ctc_loss_scale():
    """
        baseline for returnn_common ctc model with network constructor
        :return:
        """
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root_job = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="240f119b54d52a4324ab300c301f8e003e0a398c",
    )
    returnn_root_job.hash_overwrite = "ctc_baseline_returnn"
    returnn_root = returnn_root_job.out_repository
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="79876b18552f61a3af7c21c670475fee51ef3991",
        checkout_folder_name="returnn_common",
    ).out_repository
    name = (
        "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/loss_scale"
    )
    reference_corpus = get_corpus_object_dict(
        audio_format="ogg", output_prefix="corpora"
    )["train-clean-100"]
    default_vocoder = get_default_vocoder(name=name)
    synthetic_data_dict = {}
    job_splits = 10
    alignments = get_loss_scale_alignments()

    for scale, alignment in alignments.items():
        name = "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/loss_scale" + f"_{scale}"
        training_datasets, corpus, durations = get_tts_data_from_ctc_align(
            name + "/datasets",
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            alignment=alignment,
        )
        for upsampling in ["repeat", "gauss"]:
            if upsampling == "gauss" and float(scale) not in [0, 0.25, 0.75, 1.0]:
                continue
            exp_name = name + f"_{upsampling}"

            if upsampling == "gauss":
                train_config = get_training_config(
                    returnn_common_root=returnn_common_root,
                    training_datasets=training_datasets,
                    embedding_size=256,
                    speaker_embedding_size=256,
                    gauss_up=True,
                )
                train_config.config["learning_rates"] = [0.0001, 0.001]
            else:
                train_config = get_training_config(
                    returnn_common_root=returnn_common_root,
                    training_datasets=training_datasets,
                    embedding_size=256,
                    speaker_embedding_size=256,
                )
            train_job = tts_training(
                config=train_config,
                returnn_exe=returnn_exe,
                returnn_root=returnn_root,
                prefix=exp_name,
                num_epochs=200,
            )
            if upsampling == "gauss":
                forward_config = get_forward_config(
                    returnn_common_root=returnn_common_root,
                    forward_dataset=TTSForwardData(
                        dataset=training_datasets.cv, datastreams=training_datasets.datastreams
                    ),
                    embedding_size=256,
                    speaker_embedding_size=256,
                    calc_speaker_embedding=True,
                    gauss_up=(upsampling == "gauss"),
                )
            else:
                forward_config = get_forward_config(
                    returnn_common_root=returnn_common_root,
                    forward_dataset=TTSForwardData(
                        dataset=training_datasets.cv, datastreams=training_datasets.datastreams
                    ),
                    embedding_size=256,
                    speaker_embedding_size=256,
                    calc_speaker_embedding=True,
                )
            gl_swer(
                name=exp_name + "/gl_swer",
                vocoder=default_vocoder,
                returnn_root=returnn_root,
                returnn_exe=returnn_exe,
                checkpoint=train_job.out_checkpoints[200],
                config=forward_config,
            )
            if float(scale) in [0, 0.25, 1.0]:
                speaker_embedding_hdf = build_speaker_embedding_dataset(
                    returnn_common_root=returnn_common_root,
                    returnn_exe=returnn_exe,
                    returnn_root=returnn_root,
                    datasets=training_datasets,
                    prefix=exp_name,
                    train_job=train_job
                )
                for dur_pred in ["pred", "cheat"]:
                    synth_dataset = get_inference_dataset_old(
                        corpus,
                        returnn_root=returnn_root,
                        returnn_exe=returnn_exe,
                        datastreams=training_datasets.datastreams,
                        speaker_embedding_hdf=speaker_embedding_hdf,
                        durations=durations if dur_pred == "cheat" else None,
                        process_corpus=False,
                    )

                    synth_corpus = synthesize_with_splits(
                        name=exp_name + f"/{dur_pred}",
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
                        gauss_up=(upsampling == "gauss"),
                        use_true_durations=(dur_pred == "cheat"),
                    )
                    synthetic_data_dict[f"ctc_{scale}_{upsampling}_{dur_pred}"] = synth_corpus
    return synthetic_data_dict
