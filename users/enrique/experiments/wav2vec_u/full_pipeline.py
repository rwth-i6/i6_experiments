###########################################################
# Imports
###########################################################
import os


from i6_core.tools.download import DownloadJob

from i6_experiments.users.enrique.jobs.fairseq.wav2vec.audio_preprocessing import (
    Wav2VecUDeleteSilencesInAudioJob,
    Wav2VecUFeaturizeAudioJob,
    process_audio,
)
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_u_GAN import FairseqHydraTrainWav2VecUJob
from i6_experiments.users.enrique.experiments.wav2vec_u.default_tools import KENLM_BINARY_PATH

from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.w2vu_generate_job import FairseqGenerateWav2VecUJob
import logging
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import (
    get_rvad_root,
    get_fairseq_root,
    PrepareWav2VecTextDataJob,
    calculate_all_configs,
)
from i6_core.returnn.search import ReturnnComputeWERJob

from sisyphus import tk


def run_meta_experiments():
    environment = "/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3"
    fairseq_root = get_fairseq_root(
        python_env=tk.Path(environment),
        fairseq_root=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
        ),
    )


    ################################################################
    ########## Configuration for the Wav2VecU pipeline ##########
    ################################################################

    max_audios_per_manifest = (
        None  # Used to limit the number of audio files in each manifest file, mainly for debugging purposes
    )
    max_models_for_each_decoding = 1  # Number of models to use for each decoding job, this is useful to limit the number of models used for decoding, mainly for debugging purposes
    decode_training_data = True

    w2v2model = "large_60kh"  # Options: "base", "large_960h", "large_60kh"
    feature_extraction_layer = 14  # Layer to extract features from the Wav2Vec2 model, w2v-u paper uses layer 14
    assert not (
        (w2v2model == "base" and feature_extraction_layer > 11) or feature_extraction_layer > 23
    ), "not so many layers in the model"

    training_audio = "train-other-960"  # Options: "train-clean-100", "train-clean-360", "train-other-500", "train-other-960", "dev-clean", "dev-complete", "dev-other", "test-clean", "test-other"
    training_audio_extension = "flac"  # Options: "flac", "wav"
    training_valid_percent = 0.01  # Percentage of the training data to be used for validation
    training_concurrent = 8  # Number of concurrent processes to run for audio processing

    # --- Start of new configuration section for multiple decodings ---
    # Define all decoding datasets here. Add a new dict for each new dataset.
    decoding_datasets = [
        {
            "name": "test-clean",
            "extension": "flac",
            "subset": "test-clean",
        },
        {
            "name": "dev-other",
            "extension": "flac",
            "subset": "dev-other",
        },
        {
            "name": "test-other",
            "extension": "flac",
            "subset": "test-other",
        },
        {
            "name": "dev-clean",
            "extension": "flac",
            "subset": "dev-clean",
        },
    ]

    decoding_concurrent = 2  # Number of concurrent processes to run for audio processing
    decoding_config_dir = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/EXTERNAL_SOFTWARE/fairseq_w2vu/fairseq/examples/wav2vec/unsupervised/config/generate"
    decoding_config_name = "kike_kaldi_pruned_2"
    extra_config = None

    # Text configuration
    language = "en"  # Language of the text data
    tts_engine = "G2P"  # Text-to-speech engine to use for text normalization
    text_file_path = "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"
    #text_file_path = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/data/text_raw/BNCCorpus.txt"
    text_file_path = tk.Path(text_file_path)
    sil_prob = 0.25
    vocab_size = 1000  # TODO: THIS IS NOT THE VOCAB SIZE, IT IS THE MIN NUMBER OF TIMES A PHONEME NEEDS TO APPEAR FOR IT TO NOT BE DISCARTED
    fasttext_model = DownloadJob(
        url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", target_filename="lid.176.bin"
    ).out_file
    training_lm_pruning = [0,0,0,3]

    #decoding_lm_prunings = [[0,0,0,4], [0,0,1,4], [0,0,2,5]]
    decoding_lm_prunings = [[0,0,1,4], [0,0,2,5], [0,0,3,6]]
    #decoding_lm_prunings = [[0,0,1,4]]

    alias = "wav2vec_u_librispeech_gan_training_" + training_audio + "_" + w2v2model
    audio_alias = os.path.join(alias, "audio")
    training_audio_alias = os.path.join(audio_alias, training_audio)

    # Training hyperparameters
    config_dir = fairseq_root.get_path() + "/examples/wav2vec/unsupervised/config/gan"
    config_name = "w2vu"
    training_configs = {
        "model.code_penalty": [2, 4],
        "model.gradient_penalty": [1.5, 2],
        "model.smoothness_weight": [0.5, 0.75],
    }
    model_seed_range = range(0, 5)

    ################################################################
    ########### Prepare the aduio and featurize it ############
    ################################################################

    environment = tk.Path(environment)

    if w2v2model == "large_960h":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt",
            target_filename="wav2vec2_large_960h_no_finetune.pt",
        ).out_file
    if w2v2model == "large_60kh":
        assert w2v2model == "large_60kh"
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt",
            target_filename="wav2vec_60kh_no_finetune.pt",
        ).out_file
    if w2v2model == "base":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
            target_filename="wav2vec_small.pt",
        ).out_file  # All of this models are fully unsupervised

    ################### Training data preprocessing (runs once) ############################

    training_audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", training_audio))
    delete_silences_job, featurize_training_audio_job = process_audio(
        env=environment,
        fairseq_root=fairseq_root,
        audio_dir=training_audio_dir,
        valid_percent=training_valid_percent,
        ext=training_audio_extension,
        rvad_root=get_rvad_root(),
        concurrent=training_concurrent,
        layer=feature_extraction_layer,
        model_path=w2v2_model_path,
        alias_prefix=training_audio_alias,
        alias_delete="delete_silences/"
        + w2v2model
        + "/layer_"
        + str(feature_extraction_layer)
        + "valid_"
        + str(training_valid_percent),
        alias_feat="featurize_audio/"
        + w2v2model
        + "/layer_"
        + str(feature_extraction_layer)
        + "valid_"
        + str(training_valid_percent),
        max_n_audios_per_manifest=max_audios_per_manifest,
        name_the_manifests_just_train_and_valid=True,
    )

    ################################################################
    ########### text data and LM ############
    ################################################################

    prepare_text_job_training = PrepareWav2VecTextDataJob(
        fairseq_root=fairseq_root,
        language=language,
        text_file_path=text_file_path,
        kenlm_root=KENLM_BINARY_PATH,
        tts_engine=tts_engine,
        fasttext_model=fasttext_model,
        sil_prob=sil_prob,
        fairseq_python_env=environment,
        vocab_size=vocab_size,
        lm_pruning=training_lm_pruning,
    )

    prepare_text_job_training.add_alias(os.path.join(alias, "text_data"))

    prepare_text_job_decoding = []
    for prun in decoding_lm_prunings:
        job = PrepareWav2VecTextDataJob(
            fairseq_root=fairseq_root,
            language=language,
            text_file_path=text_file_path,
            kenlm_root=KENLM_BINARY_PATH,
            tts_engine=tts_engine,
            fasttext_model=fasttext_model,
            sil_prob=sil_prob,
            fairseq_python_env=environment,
            vocab_size=vocab_size,
            lm_pruning=prun,
        )
        job.add_alias(os.path.join(alias, f"text_data_pruning_{prun}"))
        prepare_text_job_decoding.append(job)


    ################################################################
    ########### Training (runs once) ############
    ################################################################

    # Caculate all possible configurations for the GAN training, so that each model has a different configuration, one for each different combination of hyperparameters
    all_configs, n_different_configs = calculate_all_configs(training_configs, model_seed_range)

    training_jobs = []
    for conf in all_configs:
        GAN_job = FairseqHydraTrainWav2VecUJob(
            environment=environment,
            task_data=featurize_training_audio_job.out_features_precompute_pca512_cls128_mean_pooled,
            task_text_data=prepare_text_job_training.processed_phn_data_and_LM,
            fairseq_root=fairseq_root,
            prefix=alias,
            config_dir=config_dir,
            config_name=config_name,
            extra_configs=conf,
        )
        training_jobs.append(GAN_job)
        GAN_job.add_alias(os.path.join(alias, f"GAN_training_{conf}"))
        #tk.register_output(f"{alias}", GAN_job.out_dir)

    ################################################################
    ########### Decoding/Recognition/Generate (loops over datasets) ############
    ################################################################
    fairseq_root_decoding = get_fairseq_root(
            python_env=tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3"),
            fairseq_root=tk.Path(
                "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/EXTERNAL_SOFTWARE/fairseq_w2vu/fairseq"
            ),
        )
    # This is the main loop for decoding multiple datasets
    for dec_config in decoding_datasets:
        decoding_audio_name = dec_config["name"]
        decoding_audio_alias = os.path.join(audio_alias, decoding_audio_name)

        # Preprocess audio for the current decoding dataset
        decoding_audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", decoding_audio_name))
        decoding_delete_silences_job, decoding_featurize_job = process_audio(
            env=environment,
            fairseq_root=fairseq_root,
            audio_dir=decoding_audio_dir,
            valid_percent=dec_config.get("valid_percent"),
            ext=dec_config["extension"],
            rvad_root=get_rvad_root(),
            concurrent=decoding_concurrent,
            layer=feature_extraction_layer,
            model_path=w2v2_model_path,
            alias_prefix=decoding_audio_alias,
            alias_delete="delete_silences"
            + "/"
            + w2v2model
            + "/layer_"
            + str(feature_extraction_layer)
            + "valid_"
            + str(dec_config.get("valid_percent")),
            alias_feat="featurize_audio/"
            + w2v2model
            + "/layer_"
            + str(feature_extraction_layer)
            + "valid_"
            + str(dec_config.get("valid_percent")),
            existing_clusters=featurize_training_audio_job.out_features_clusters,
            existing_pca=featurize_training_audio_job.out_features_pca,
            max_n_audios_per_manifest=max_audios_per_manifest,
            name_the_manifests_just_train_and_valid=False,
        )

        # Generate results for the current decoding dataset using all trained models
        for training_job in training_jobs[:max_models_for_each_decoding]:

            for lm_job in prepare_text_job_decoding:
                generate_job = FairseqGenerateWav2VecUJob(
                    decoding_audio_name=decoding_audio_name,
                    environment=environment,
                    fairseq_root=fairseq_root_decoding,
                    task_data=decoding_featurize_job.out_features_precompute_pca512_cls128_mean,
                    prepare_text=lm_job.out_text_dir,
                    checkpoints_path=training_job.out_dir,
                    config_name=decoding_config_name,
                    config_dir=decoding_config_dir,
                    extra_config=extra_config,
                )

                # Create a unique alias for each generation job
                job_alias = os.path.join(
                    alias,
                    "decoding",
                    decoding_audio_name,
                    f"generate_{dec_config['subset']}_pruning_{lm_job.lm_pruning}",
                )
                output_path = os.path.join(
                    alias, "decoding_results", decoding_audio_name, f"pruning_{lm_job.lm_pruning}"
                )

                generate_job.add_alias(job_alias)
                tk.register_output(output_path, generate_job.results_path)

                # wer_job = ReturnnComputeWERJob(reference=generate_job.ground_truth_transcription, hypothesis=generate_job.transcription_generated)
                # wer_job.add_alias(job_alias + "_WER")
                # tk.register_output(os.path.join(output_path, "WER"), wer_job.out_wer)

    if decode_training_data:
        

        # Generate results for the training data itself, using all trained models
        decoding_audio_name = training_audio
        decoding_audio_alias = os.path.join(audio_alias, decoding_audio_name)

        for training_job in training_jobs[:1]:

            for lm_job in prepare_text_job_decoding[:1]:
                generate_job = FairseqGenerateWav2VecUJob(
                    decoding_audio_name=decoding_audio_name,
                    environment=environment,
                    fairseq_root=fairseq_root_decoding,
                    task_data=featurize_training_audio_job.out_features_precompute_pca512_cls128_mean,
                    prepare_text=lm_job.out_text_dir,
                    checkpoints_path=training_job.out_dir,
                    config_name=decoding_config_name,
                    #config_name="viterbi",
                    config_dir=decoding_config_dir,
                    extra_config=extra_config,
                    gen_subset="train"
                )

                # Create a unique alias for each generation job
                job_alias = os.path.join(
                    alias,
                    "decoding",
                    decoding_audio_name,
                    f"generate_train_pruning_{lm_job.lm_pruning}",
                )
                output_path = os.path.join(
                    alias, "decoding_results", decoding_audio_name, f"pruning_{lm_job.lm_pruning}"
                )

                generate_job.add_alias(job_alias)
                tk.register_output(output_path, generate_job.results_path)

                # wer_job = ReturnnComputeWERJob(reference=generate_job.ground_truth_transcription, hypothesis=generate_job.transcription_generated)
                # wer_job.add_alias(job_alias + "_WER")
                # tk.register_output(os.path.join(output_path, "WER"), wer_job.out_wer)
    
                

def py():
    run_meta_experiments()
