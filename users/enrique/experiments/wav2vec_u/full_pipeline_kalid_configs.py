###########################################################
# Imports
###########################################################
import sys
import os
from recipe.i6_core.recognition.scoring import ScliteJob
from recipe.i6_core.tools.download import DownloadJob

from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.audio_preprocessing import (
    Wav2VecUDeleteSilencesInAudioJob,
    Wav2VecUFeaturizeAudioJob,
    process_audio,
)
from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_u_GAN import FairseqHydraTrainWav2VecUJob
from recipe.i6_experiments.users.enrique.experiments.wav2vec_u.default_tools import KENLM_BINARY_PATH, SCTK_BINARY_PATH
from recipe.i6_experiments.users.enrique.experiments.wav2vec_u.unsupervised_metric import UnsupervisedVocabUsageAndLMScoreMetric

from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.w2vu_generate_job import FairseqKaldiDecodingJob, ViterbiGenerateWav2VecUJob
from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import (
    get_rvad_root,
    get_fairseq_root,
    PrepareWav2VecTextDataJob,
    calculate_all_configs,
    CalculatePerplexityJob,
    DivideLibriSpeech960hInto100h360h500hJob,
    MergeGenerateShardsJob,
)
from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import GetW2VLibriSpeechGroundTruthJob

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

    #models_to_decode_with = [0, 4, 8, 12, 16, 20, 24, 28]
    #models_to_decode_with = [1, 5, 9, 13, 17, 21, 25, 29]
    #models_to_decode_with = [2, 6, 10, 14, 18, 22, 26, 30]
    #models_to_decode_with = [3, 7, 11, 15, 19, 23, 27, 31]
    #models_to_decode_with = [4, 8, 12, 16, 20, 24, 28, 32]
    
    #models_to_decode_with = [0,1,2,3,4]
    #models_to_decode_with = [5,6,7,8,9]
    #models_to_decode_with = [10,11,12,13,14]
    #models_to_decode_with = [15,16,17,18,19]
    models_to_decode_with = [20,21,22,23,24]
    #models_to_decode_with = [i for i in range(0, 41)] 
    #models_to_decode_with = [0, 9, 29, 6, 15 ,28, 19 ,38 ,37]
    #models_to_decode_with = [9, 2, 29]

    decode_training_data = True
    train_data_shards = 32
    models_to_decode_training_data_with = [] # 9, 29 best WER  6 15 28 best word per  19 38 37 best phn per

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
        # {
        #     "name": "test-clean",
        #     "extension": "flac",
        #     "subset": "test-clean",
        # },
        # {
        #     "name": "dev-other",
        #     "extension": "flac",
        #     "subset": "dev-other",
        # },
        {
            "name": "test-other",
            "extension": "flac",
            "subset": "test-other",
        },
        # {
        #     "name": "dev-clean",
        #     "extension": "flac",
        #     "subset": "dev-clean",
        # },
        # {
        #     "name": "train-clean-100",
        #     "extension": "flac",
        #     "subset": "train-clean-100",
        # },
        # {
        #     "name": "train-clean-360",
        #     "extension": "flac",
        #     "subset": "train-clean-360",
        # },
        # {
        #     "name": "train-other-500",
        #     "extension": "flac",
        #     "subset": "train-other-500",
        # },
        # {
        #     "name": "train-other-960",
        #     "extension": "flac",
        #     "subset": "train-other-960",
        #     "valid_percent": 0.01,
        # }
    ]

    decoding_concurrent = 16  # Number of concurrent processes to run for audio processing
    decoding_config_dir = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/EXTERNAL_SOFTWARE/fairseq_w2vu/fairseq/examples/wav2vec/unsupervised/config/generate"
    decoding_config_name = "kike_kaldi_pruned_2"


    # kaldi_decoding_extra_configs_to_try = [f"""\
    #         beam={b} \
    #         decode_stride={ds} \
    #         blank_weight={bw} \
    #         kaldi_decoder_config.acoustic_scale={acoustic_scale}\
    #         no_softmax={sftm}\
    #     """
    #     for b in [20, 30]
    #     for ds in [1]
    #     for bw in [0, 1.5, 2.0, 2.5, 4, 5, 8, 10]
    #     for acoustic_scale in [0.45, 0.5, 0.55, 0.6]
    #     for sftm in [True]
    # ]

    # kaldi_decoding_extra_configs_to_try = [f"""\
    #         beam={b} \
    #         decode_stride={ds} \
    #         blank_weight={bw} \
    #         kaldi_decoder_config.acoustic_scale={acoustic_scale}\
    #     """
    #     for b in [15, 20]
    #     for ds in [1]
    #     for bw in [1.0, 2.0]
    #     for acoustic_scale in [0.4, 0.5, 0.7, 0.85]
    # ]

    kaldi_decoding_extra_configs_to_try = [f"""\
            beam={b} \
            decode_stride={ds} \
            blank_weight={bw} \
            kaldi_decoder_config.acoustic_scale={acoustic_scale}\
            no_softmax={sftm}\
        """
        for b in [20]
        for ds in [1]
        for bw in [5]
        for acoustic_scale in [0.6]
        for sftm in [True]
    ]







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
    training_lm_pruning = [0,0,0,3] # This KenLM is only used to select the best epoch of a GAN model after training

    #decoding_lm_prunings = [[0,0,1,4], [0,0,2,5], [0,0,3,6]]
    decoding_lm_prunings = [[0,0,1,4]]

    alias = "wav2vec_u_librispeech_gan_training_" + training_audio + "_" + w2v2model
    audio_alias = os.path.join(alias, "audio")
    text_alias = os.path.join(alias, "text_data")
    training_alias = os.path.join(alias, "GAN_training")
    training_audio_alias = os.path.join(audio_alias, training_audio)

    # Training hyperparameters
    config_dir = fairseq_root.get_path() + "/examples/wav2vec/unsupervised/config/gan"
    config_name = "w2vu"

    training_configs = {
        "model.code_penalty": [2, 4],
        "model.gradient_penalty": [1.5, 2],
        "model.smoothness_weight": [0.5, 0.75],
    }
    # training_configs = {
    #     "model.code_penalty": [2],
    #     "model.gradient_penalty": [1.5],
    #     "model.smoothness_weight": [0.5],
    # }
    # training_configs = {
    #     "model.code_penalty": [2],
    #     "model.gradient_penalty": [1.5],
    #     "model.smoothness_weight": [0.75],
    # }
    # training_configs = {
    #     "model.code_penalty": [2],
    #     "model.gradient_penalty": [2],
    #     "model.smoothness_weight": [0.5],
    # }
    # training_configs = {
    #     "model.code_penalty": [2],
    #     "model.gradient_penalty": [2],
    #     "model.smoothness_weight": [0.75],
    # }

    # training_configs = {
    #     "model.code_penalty": [2],
    #     "model.gradient_penalty": [1.5],
    #     "model.smoothness_weight": [0.5],
    # }
    # training_configs = {
    #     "model.code_penalty": [2],
    #     "model.gradient_penalty": [1.5],
    #     "model.smoothness_weight": [0.75],
    # }
    # training_configs = {
    #     "model.code_penalty": [2],
    #     "model.gradient_penalty": [2],
    #     "model.smoothness_weight": [0.5],
    # }
    # training_configs = {
    #     "model.code_penalty": [2],
    #     "model.gradient_penalty": [2],
    #     "model.smoothness_weight": [0.75],
    # }

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
    delete_silences_job, featurize_training_audio_job, process_training_audio_manifest = process_audio(
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
    prepare_text_job_training.add_alias(os.path.join(text_alias, "pruning_" + "_".join(map(str, training_lm_pruning))))


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
            #three_gram_lm=True, # Only temporal, usually use 4gram
        )
        if job.three_gram_lm:
            job.add_alias(os.path.join(text_alias, f"3gram_LM"))
            tk.register_output(os.path.join(text_alias, f"3gram_LM"), job.out_text_dir)
            return
        job.add_alias(os.path.join(text_alias, f"pruning_" + "_".join(map(str, prun))))
        prepare_text_job_decoding.append(job)
        
    ################################################################
    ########### Training (runs once) ############
    ################################################################

    # Caculate all possible configurations for the GAN training, so that each model has a different configuration, one for each different combination of hyperparameters
    all_configs, n_different_configs = calculate_all_configs(training_configs, model_seed_range)
    training_configs["common.seed"] = model_seed_range
    GAN_job_all_configs = FairseqHydraTrainWav2VecUJob(
            environment=environment,
            task_data=featurize_training_audio_job.out_features_precompute_pca512_cls128_mean_pooled,
            task_text_data=prepare_text_job_training.processed_phn_data_and_LM,
            fairseq_root=fairseq_root,
            prefix=alias,
            config_dir=config_dir,
            config_name=config_name,
            extra_configs=training_configs,
        )
    GAN_job_all_configs.add_alias(os.path.join(training_alias, f"all_configs"))
    tk.register_output(os.path.join(training_alias, f"all_configs"), GAN_job_all_configs.out_dir)

    generate_viterbi_jobs = []
    training_jobs = []
    t=0
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
        if t in models_to_decode_with:
            print(conf)
        training_jobs.append(GAN_job)
        t += 1
        GAN_job.add_alias(os.path.join(training_alias, f"model_seed_{conf['common.seed']}_" + "_".join([f"{k}_{v}" for k, v in conf.items() if k != "common.seed"])))

    training_jobs.append(GAN_job_all_configs)

    ################################################################
    ########### Decoding/Recognition/Generate (loops over datasets) ############
    ################################################################
    stats = {}
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
        decoding_delete_silences_job, decoding_featurize_job, process_audio_manifest = process_audio(
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
        stats[decoding_audio_name]={}
        # Generate results for the current decoding dataset using all trained models
        for it in models_to_decode_with:
            if it >= len(training_jobs):
                print(f"Skipping model {it} as it is out of range")
                continue
            training_job = training_jobs[it]
            stats[decoding_audio_name][it]={}
            for lm_job in prepare_text_job_decoding:
                lm_pruning_alias = f"pruning_" + "_".join(map(str, lm_job.lm_pruning))
                stats[decoding_audio_name][it][lm_pruning_alias] = {}
                for extra_config in kaldi_decoding_extra_configs_to_try:
                    extra_config = extra_config+ f""" \
                        fairseq.task.text_data={lm_job.processed_phn_data_and_LM} \
                        fairseq.task.labels=phn \
                    """
                    extra_config_alias = "_".join(extra_config.replace("=", "_").split()).replace(" ", "__")




                    #Debugging: limit to just one extra config
                    debugging = False
                    if debugging:
                        generate_job = FairseqKaldiDecodingJob(
                            decoding_audio_name=decoding_audio_name,
                            environment=environment,
                            fairseq_root=fairseq_root_decoding,
                            task_data=decoding_featurize_job.out_features_precompute_pca512_cls128_mean,
                            lexicon_lst_path=lm_job.lexicon_lst,
                            hlg_graph_path=lm_job.phn_words_sil_hlg_graph_path,
                            kaldi_dict=lm_job.phn_words_sil_kaldi_dict_path,
                            checkpoints_path=tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/gan/hydra_train/FairseqHydraTrainJob.OIgmsxxzHuo9/output/out_dir/checkpoint_best.pt"),
                            gen_subset="preprocessed_audio",
                            config_name=decoding_config_name,
                            config_dir=decoding_config_dir,
                            extra_config=extra_config,
                        )
                        print("AAAAAAAAA")
                        print(generate_job.out_label_path_dict.get_path())
                    else:
                        generate_job = FairseqKaldiDecodingJob(
                            decoding_audio_name=decoding_audio_name,
                            environment=environment,
                            fairseq_root=fairseq_root_decoding,
                            task_data=decoding_featurize_job.out_features_precompute_pca512_cls128_mean,
                            lexicon_lst_path=lm_job.lexicon_lst,
                            hlg_graph_path=lm_job.phn_words_sil_hlg_graph_path,
                            kaldi_dict=lm_job.phn_words_sil_kaldi_dict_path,
                            checkpoints_path=training_job.out_best_model,
                            gen_subset="preprocessed_audio",
                            config_name=decoding_config_name,
                            config_dir=decoding_config_dir,
                            extra_config=extra_config,
                        )









                    



                    
                    
                    
                    
                    # Create a unique alias for each generation job
                    job_alias = os.path.join(
                        alias,
                        "decoding",
                        decoding_audio_name,
                        f"t_{it}", lm_pruning_alias, extra_config_alias,
                    )
                    output_path = os.path.join(
                        alias, "decoding_results", decoding_audio_name, f"pruning_{lm_job.lm_pruning}"
                    )

                    tk.register_output(output_path, generate_job.results_path)

                    generate_job.add_alias(job_alias)

                    ground_truth_job = GetW2VLibriSpeechGroundTruthJob(audio_manifest=process_audio_manifest, librispeech_subset=decoding_audio_name)
                    ref = ground_truth_job.ground_truth_stm
                    hyp = generate_job.transcription_generated_ctm

                    sclite_job = ScliteJob(ref=ref, hyp=hyp, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
                    sclite_job.add_alias(os.path.join(job_alias, "wer"))
                    tk.register_output(os.path.join(output_path, "wer"), sclite_job.out_report_dir)
                    stats[decoding_audio_name][it][lm_pruning_alias][extra_config] = {}
                    if sclite_job.out_wer.get() is not None:
                        stats[decoding_audio_name][it][lm_pruning_alias][extra_config]["wer"] = sclite_job.out_wer.get()

                    calculate_word_perplexity_job = CalculatePerplexityJob( # On word level
                        arpa_file=lm_job.processed_wrd_LM_bin,
                        text_file=generate_job.transcription_generated,
                    )
                    tk.register_output(os.path.join(text_alias, lm_pruning_alias, extra_config, "perplexity"), calculate_word_perplexity_job.output_perplexity_var)
                    calculate_word_perplexity_job.add_alias(os.path.join(job_alias, "word_perplexity"))
                    if calculate_word_perplexity_job.output_perplexity_var.get() is not None:
                        stats[decoding_audio_name][it][lm_pruning_alias][extra_config]["word_per"] = calculate_word_perplexity_job.output_perplexity_var.get()

                    calculate_word_perplexity_job = CalculatePerplexityJob( # On word level, just ground_truth
                        arpa_file=lm_job.processed_wrd_LM_bin,
                        text_file=ground_truth_job.output_ground_truth,
                    )
                    tk.register_output(os.path.join(text_alias, lm_pruning_alias, extra_config, "word_perplexity"), calculate_word_perplexity_job.output_perplexity_var)
                    calculate_word_perplexity_job.add_alias(os.path.join(text_alias, lm_pruning_alias, extra_config, "word_perplexity_ground_truth"))


                    generate_viterbi_job = ViterbiGenerateWav2VecUJob(
                        decoding_audio_name=decoding_audio_name,
                        environment=environment,
                        fairseq_root=fairseq_root_decoding,
                        task_data=decoding_featurize_job.out_features_precompute_pca512_cls128_mean,
                        prepare_text=lm_job.out_text_dir,
                        checkpoints_path=training_job.out_best_model,
                    )
                    generate_viterbi_jobs.append((generate_viterbi_job, job_alias))
                    generate_viterbi_job.add_alias(os.path.join(job_alias, "viterbi_generate"))

                    calculate_phoneme_perplexity_job = CalculatePerplexityJob( # On phoneme level
                        arpa_file=lm_job.text_phones_lm_filtered_06_arpa,
                        text_file=generate_viterbi_job.transcription_generated,
                    )
                    tk.register_output(os.path.join(text_alias, lm_pruning_alias, "phoneme_perplexity"), calculate_phoneme_perplexity_job.output_log2_prob_var)
                    calculate_phoneme_perplexity_job.add_alias(os.path.join(job_alias, "phoneme_perplexity"))
                    if calculate_phoneme_perplexity_job.output_log2_prob_var.get() is not None:
                        stats[decoding_audio_name][it][lm_pruning_alias][extra_config]["log2_prob"] = calculate_phoneme_perplexity_job.output_log2_prob_var.get()

    





    transcripts_and_models = []
    for generate_viterbi_job, job_alias in generate_viterbi_jobs:
        transcripts_and_models.append((generate_viterbi_job.transcription_generated, generate_viterbi_job.checkpoints_path))

    # best_models_job = UnsupervisedVocabUsageAndLMScoreMetric(transcripts_and_models=transcripts_and_models, vocab_path=prepare_text_job_training.processed_dict_phn_txt, kenlm_path=prepare_text_job_training.text_phones_lm_filtered_04_bin, sil="<SIL>", allow_margin=1.2)
    # tk.register_output("best_models", best_models_job.best_to_worst_models[0])


    n_best = 500
    for dec_config in stats.keys():
        results = []
        for model_it in stats[dec_config].keys():
            for lm_prun in stats[dec_config][model_it].keys():
                for extra_config in stats[dec_config][model_it][lm_prun].keys():
                    wer = stats[dec_config][model_it][lm_prun][extra_config].get("wer", float('inf'))
                    word_per = stats[dec_config][model_it][lm_prun][extra_config].get("word_per", float('inf'))
                    log2_prob = stats[dec_config][model_it][lm_prun][extra_config].get("log2_prob", float('-inf'))
                    results.append((model_it, extra_config, lm_prun, wer, word_per, log2_prob))

        top_by_wer = sorted(results, key=lambda x: x[3])[:n_best]
        top_by_word_per = sorted(results, key=lambda x: x[4])[:n_best]
        top_by_log2_prob = sorted(results, key=lambda x: x[5])[-n_best:]

        print(f"\nTop {n_best} models for {dec_config} by WER:")
        for model_it, extra_config, lm_prun, wer, word_per, log2_prob in top_by_wer:
            print(f"  Model {model_it}, LM pruning {lm_prun}, extra_config={str(extra_config).replace(' ', '')}:\n WER={wer}, plexity={word_per}, log_prob phoneme={log2_prob}")

        # print(f"\nTop {n_best} models for {dec_config} by Word Perplexity:")
        # for model_it, lm_prun, wer, word_per, log2_prob in top_by_word_per:
        #     print(f"  Model {model_it}, LM pruning {lm_prun}, extra_config={str(extra_config).replace(' ', '')}:\n WER={wer}, Word Perplexity={word_per}, log_prob phoneme={log2_prob}")
        # print(f"\nTop {n_best} models for {dec_config} by log_prob phoneme:")
        # for model_it, lm_prun, wer, word_per, log2_prob in top_by_log2_prob:
        #     print(f"  Model {model_it}, LM pruning {lm_prun}, extra_config={str(extra_config).replace(' ', '')}:\n WER={wer}, Word Perplexity={word_per}, log_prob phoneme={log2_prob}")




    divide_jobs=[]
    if decode_training_data:
        for t in models_to_decode_training_data_with:
            if t >= len(training_jobs):
                print(f"Skipping model {t} as it is out of range")
                continue
            training_job = training_jobs[t]
            for lm_job in prepare_text_job_decoding[:1]:
                generate_job_valid = FairseqKaldiDecodingJob(
                    decoding_audio_name=training_audio,
                    environment=environment,
                    fairseq_root=fairseq_root_decoding,
                    task_data=featurize_training_audio_job.out_features_precompute_pca512_cls128_mean,
                    lexicon_lst_path=lm_job.lexicon_lst,
                    hlg_graph_path=lm_job.phn_words_sil_hlg_graph_path,
                    kaldi_dict=lm_job.phn_words_sil_kaldi_dict_path,
                    checkpoints_path=training_job.out_best_model,
                    gen_subset="valid",
                    config_dir=decoding_config_dir,
                    config_name=decoding_config_name,
                    extra_config=extra_config,
                )

                # Create a unique alias for each generation job
                job_alias = os.path.join(
                    alias,
                    "decoding",
                    training_audio,
                    "valid",
                    f"t_{t}", "pruning_" + "_".join(map(str, lm_job.lm_pruning))
                )
                generate_job_valid.add_alias(job_alias)

                shard_transcriptions = []
                for shard in range(train_data_shards):
                    extra_config_shard = extra_config + f"""\
                        fairseq.dataset.shard_id={shard} \
                        fairseq.dataset.num_shards={train_data_shards} \
                    """

                    generate_job_train = FairseqKaldiDecodingJob(
                        decoding_audio_name=training_audio,
                        environment=environment,
                        fairseq_root=fairseq_root_decoding,
                        task_data=featurize_training_audio_job.out_features_precompute_pca512_cls128_mean,
                        lexicon_lst_path=lm_job.lexicon_lst,
                        hlg_graph_path=lm_job.phn_words_sil_hlg_graph_path,
                        kaldi_dict=lm_job.phn_words_sil_kaldi_dict_path,
                        checkpoints_path=training_job.out_best_model,
                        gen_subset="train",
                        config_dir=decoding_config_dir,
                        config_name=decoding_config_name,
                        extra_config=extra_config_shard,
                    )
                    # Create a unique alias for each generation job
                    job_alias = os.path.join(
                        alias,
                        "decoding",
                        training_audio,
                        "train",
                        f"t_{t}_shard{shard}", "pruning_" + "_".join(map(str, lm_job.lm_pruning))
                    )
                    generate_job_train.add_alias(job_alias)
                    tk.register_output(os.path.join(alias, "decoding", training_audio, "train", f"t_{t}_shard{shard}", "pruning_" + "_".join(map(str, lm_job.lm_pruning))), generate_job_train.results_path)
                    shard_transcriptions.append(generate_job_train.transcription_generated)

                merge_transcriptions = MergeGenerateShardsJob(shard_transcriptions, generate_job_train.manifest_path, decoding_audio_name=training_audio)

                merge_transcriptions.add_alias(os.path.join(alias, "decoding", training_audio, "train", f"t_{t}", "pruning_" + "_".join(map(str, lm_job.lm_pruning))))
                tk.register_output(os.path.join(alias, "decoding", training_audio, "train", f"t_{t}", "pruning_" + "_".join(map(str, lm_job.lm_pruning))), merge_transcriptions.output_text_file)
                
                divide_job = DivideLibriSpeech960hInto100h360h500hJob([merge_transcriptions.transcription_generated_formatted, generate_job_valid.transcription_generated_formatted])
                divide_job.add_alias(os.path.join(alias, "decoding", training_audio, "train_valid_divided", f"t_{t}", "pruning_" + "_".join(map(str, lm_job.lm_pruning))))
                tk.register_output(os.path.join(alias, "decoding", training_audio, "train_valid_divided", f"t_{t}", "pruning_" + "_".join(map(str, lm_job.lm_pruning))), divide_job.out_label_path_dict)
                divide_jobs.append(divide_job.out_label_path_dict.get_path())

def py():
    run_meta_experiments()
