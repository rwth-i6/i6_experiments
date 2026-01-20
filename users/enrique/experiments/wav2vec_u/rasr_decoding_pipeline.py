###########################################################
# Imports
###########################################################
import sys
import os
from i6_core.recognition.scoring import ScliteJob
from i6_core.tools.download import DownloadJob

from i6_experiments.users.enrique.jobs.fairseq.wav2vec.audio_preprocessing import (
    process_audio,
)
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_u_GAN import FairseqHydraTrainWav2VecUJob
from i6_experiments.users.enrique.experiments.wav2vec_u.default_tools import KENLM_BINARY_PATH, SCTK_BINARY_PATH

from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.w2vu_generate_job import (
    FairseqGenerateWav2VecUJob,
    ViterbiGenerateWav2VecUJob,
)
import logging
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import (
    get_rvad_root,
    get_fairseq_root,
    PrepareWav2VecTextDataJob,
    calculate_all_configs,
    CalculatePerplexityJob,
    DivideLibriSpeech960hInto100h360h500hJob,
)
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import GetW2VLibriSpeechGroundTruthJob

from sisyphus import tk

from recipe.rasr_decod.gan_rasr import FairseqRasrDecode
from recipe.rasr_decod.rasr_utils import LexiconFromTextFileJob, GetTreeTimesyncRecogConfigJob, get_default_phn_order


rasr_binary_path = tk.Path(
    "/work/asr3/berger/hiwis/kleppel/rasr_dev/hmm-treebuilder/rasr_monophone_2/arch/linux-x86_64-standard"
)

language_models_to_try = [  # "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/word_level_lms/kenlm.wrd.o40002.arpa",
    # "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/word_level_lms/kenlm.wrd.o40001.arpa",
    "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/word_level_lms/kenlm.wrd.o40000.arpa",
    # "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/word_level_lms/kenlm.wrd.o2.bin",
    "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/word_level_lms/kenlm.wrd.o3.arpa"
]
language_models_to_try = [tk.Path(lm) for lm in language_models_to_try]

# lm_scales_to_try = [0.2, 1, 1.7, 3]
# lm_scales_to_try = [1.7, 3]
# lm_scales_to_try = [0.7, 1.0, 1.4, 1.8]
# lm_scales_to_try = [0.2, 0.4, 0.5, 0.6, 0.7, 1.0]
# lm_scales_to_try = [0.6, 0.65, 0.7, 0.75, 0.8]
# lm_scales_to_try = [0.67, 0.68, 0.69]
lm_scales_to_try = [0.69]
am_scales_to_try = [1.0]
blank_weights_to_try = [0.0, 0.005, 0.01, 0.07]
blank_mode_to_try = ["add"]
# decode_strides_to_try = [1, 2]
decode_strides_to_try = [1]
beam_sizes_to_try = [1024, 2048, 4096]
# beam_sizes_to_try = [4096]
# score_thresholds_to_try = [12.0, 16.0, 20.0]
score_thresholds_to_try = [16.0]

gan_rasr_init_configs = []
gan_rasr_configs = []

gan_rasr_config = {
    "rasr_binary_path": rasr_binary_path,
    #'checkpoint_path': tk.Path('/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/wav2vec_u_GAN/FairseqHydraTrainWav2VecUJob.zH065mv9MSOF/output/out_dir/checkpoint_best.pt'),
    #'audio_features_to_decode_folder': tk.Path('/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/audio_preprocessing/Wav2VecUFeaturizeAudioJob.mxMPm0Lou129/output/audio_features/precompute_pca512_cls128_mean'),
    #'gen_subset': 'preprocessed_audio',
    "language_model": {
        "lexicon_path": tk.Path(
            "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/wav2vec_data_utils/PrepareWav2VecTextDataJob.RZfllsI3R2Pd/output/text/lexicon_filtered.lst"
        ),
        #'lm_path': tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/word_level_lms/kenlm.wrd.o2.bin"),
        #'scale': 0.1
    },
    "acoustic_model": {
        #'scale': 1.0,
        #'blank_weight': 0.0,
        #'blank_mode': 'add',
        #'decode_stride': 1,
        "sil_is_blank": True,
    },
    #'max_beam_size': 128,
    #'intermediate_max_beam_size': 128,
    #'score_threshold': 18.0,
    #'intermediate_score_threshold': 18.0,
}
for lm_path in language_models_to_try:
    for lm_scale in lm_scales_to_try:
        lm_scale = str(lm_scale)
        for am_scale in am_scales_to_try:
            for blank_weight in blank_weights_to_try:
                for blank_mode in blank_mode_to_try:
                    for decode_stride in decode_strides_to_try:
                        for beam_size in beam_sizes_to_try:
                            for score_threshold in score_thresholds_to_try:
                                config = gan_rasr_config.copy()
                                config["language_model"] = config["language_model"].copy()
                                config["language_model"]["lm_path"] = lm_path
                                config["language_model"]["scale"] = lm_scale
                                config["acoustic_model"] = config["acoustic_model"].copy()
                                config["acoustic_model"]["scale"] = am_scale
                                config["acoustic_model"]["blank_weight"] = blank_weight
                                config["acoustic_model"]["blank_mode"] = blank_mode
                                config["acoustic_model"]["decode_stride"] = decode_stride
                                config["max_beam_size"] = beam_size
                                config["intermediate_max_beam_size"] = beam_size
                                config["score_threshold"] = score_threshold
                                config["intermediate_score_threshold"] = score_threshold
                                gan_rasr_init_configs.append(config)


def run_rasr_experimets():
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

    models_to_decode_with = [6, 15, 9, 29, 28, 19, 38]

    w2v2model = "large_60kh"  # Options: "base", "large_960h", "large_60kh"
    feature_extraction_layer = 14  # Layer to extract features from the Wav2Vec2 model, w2v-u paper uses layer 14
    assert not (
        (w2v2model == "base" and feature_extraction_layer > 11) or feature_extraction_layer > 23
    ), "not so many layers in the model"

    training_audio = "train-other-960"  # Options: "train-clean-100", "train-clean-360", "train-other-500", "train-other-960", "dev-clean", "dev-complete", "dev-other", "test-clean", "test-other"
    training_audio_extension = "flac"  # Options: "flac", "wav"
    training_valid_percent = 0.01  # Percentage of the training data to be used for validation
    training_concurrent = 8  # Number of concurrent processes to run for audio processing

    decode_training_data = False
    # models_to_decode_training_data_with = [6, 15, 9, 29, 28, 19, 38]
    models_to_decode_training_data_with = [9]

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
    extra_config = None

    # Text configuration
    language = "en"  # Language of the text data
    tts_engine = "G2P"  # Text-to-speech engine to use for text normalization
    text_file_path = "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"
    # text_file_path = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/data/text_raw/BNCCorpus.txt"
    text_file_path = tk.Path(text_file_path)
    sil_prob = 0.25
    vocab_size = 1000  # TODO: THIS IS NOT THE VOCAB SIZE, IT IS THE MIN NUMBER OF TIMES A PHONEME NEEDS TO APPEAR FOR IT TO NOT BE DISCARTED
    fasttext_model = DownloadJob(
        url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", target_filename="lid.176.bin"
    ).out_file
    training_lm_pruning = [0, 0, 0, 3]  # This KenLM is only used to select the best epoch of a GAN model after training

    decode_training_data_lm = tk.Path("")

    # decoding_lm_prunings = [[0,0,1,4], [0,0,2,5], [0,0,3,6]]
    decoding_lm_prunings = [[0, 0, 1, 4]]

    alias = "wav2vec_u_librispeech_gan_training_" + training_audio + "_" + w2v2model
    audio_alias = os.path.join(alias, "audio")
    text_alias = os.path.join(alias, "text_data")
    training_alias = os.path.join(audio_alias, "GAN_training")
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
    print(featurize_training_audio_job.out_features_precompute_pca512_cls128_mean_pooled.get_path())

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
        )
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
        GAN_job.add_alias(
            os.path.join(
                training_alias,
                f"model_seed_{conf['common.seed']}_"
                + "_".join([f"{k}_{v}" for k, v in conf.items() if k != "common.seed"]),
            )
        )

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
        # Generate results for the current decoding dataset using all trained models
        for it in models_to_decode_with:
            for rasr_config in gan_rasr_init_configs:
                config_copy = rasr_config.copy()
                config_copy["checkpoint_path"] = training_jobs[it].out_best_model
                config_copy["audio_features_to_decode_folder"] = (
                    decoding_featurize_job.out_features_precompute_pca512_cls128_mean
                )
                config_copy["audio_features_manifest"] = process_audio_manifest
                config_copy["decoding_audio_name"] = decoding_audio_name
                config_copy["gen_subset"] = "preprocessed_audio"
                config_copy["acoustic_model_id"] = it
                gan_rasr_configs.append(config_copy)

    stats = {}
    for gan_rasr_config in gan_rasr_configs:
        lexicon_xml_path = LexiconFromTextFileJob(
            text_file=gan_rasr_config["language_model"]["lexicon_path"], labels=get_default_phn_order()
        ).out_bliss_lexicon

        rasr_config_file = GetTreeTimesyncRecogConfigJob(
            rasr_binary_path=gan_rasr_config["rasr_binary_path"],
            max_beam_size=gan_rasr_config["max_beam_size"],
            intermediate_max_beam_size=gan_rasr_config["intermediate_max_beam_size"],
            score_threshold=gan_rasr_config["score_threshold"],
            intermediate_score_threshold=gan_rasr_config["intermediate_score_threshold"],
            logfile_suffix="decoding",
            lm_path=gan_rasr_config["language_model"]["lm_path"],
            lm_scale=gan_rasr_config["language_model"]["scale"],
            am_scale=gan_rasr_config["acoustic_model"]["scale"],
            lexicon_file_path=lexicon_xml_path,
        ).output_rasr_config_file_path

        alias = os.path.join(
            "rasr_decoding_results",
            f"rasr_decoding_{gan_rasr_config['decoding_audio_name']}_am_{gan_rasr_config['acoustic_model_id']}_lm_{os.path.basename(gan_rasr_config['language_model']['lm_path'])}_lmScale_{gan_rasr_config['language_model']['scale']}_amScale_{gan_rasr_config['acoustic_model']['scale']}_blankWeight_{gan_rasr_config['acoustic_model']['blank_weight']}_decodeStride_{gan_rasr_config['acoustic_model']['decode_stride']}_beamSize_{gan_rasr_config['max_beam_size']}_scoreThreshold_{gan_rasr_config['score_threshold']}",
        )

        fairseq_rasr_decode_job = FairseqRasrDecode(gan_rasr_config, rasr_config_file)
        fairseq_rasr_decode_job.add_alias(alias + "/fairseq_rasr_decode")

        ground_truth_job = GetW2VLibriSpeechGroundTruthJob(
            audio_manifest=gan_rasr_config["audio_features_manifest"],
            librispeech_subset=gan_rasr_config["decoding_audio_name"],
        )
        ref = ground_truth_job.ground_truth_stm
        hyp = fairseq_rasr_decode_job.transcription_generated_ctm

        sclite_job = ScliteJob(ref=ref, hyp=hyp, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
        sclite_job.add_alias(alias + "/sclite")
        tk.register_output("wer", sclite_job.out_report_dir)

        if sclite_job.out_wer.get() is not None:
            stats[alias] = {"sclite_job": sclite_job, "configuration": gan_rasr_config}

    wer_results = []
    for k, v in stats.items():
        if "test-other" in k:
            wer_results.append(
                {
                    "wer": float(v["sclite_job"].out_wer.get()),
                    "alias": k,
                    # "configuration": v["configuration"]
                }
            )
    wer_results.sort(key=lambda x: x["wer"])

    n_best = 10000
    for n in range(min(n_best, len(wer_results))):
        print(f"Rank {n+1}: WER={wer_results[n]['wer']:.2f}%, Alias={wer_results[n]['alias']}")

    divide_jobs = []
    if decode_training_data:
        # Generate results for the training data itself, using all trained models
        decoding_audio_name = training_audio
        decoding_audio_alias = os.path.join(audio_alias, decoding_audio_name)

        for rasr_config in gan_rasr_init_configs:
            config_copy = rasr_config.copy()
            config_copy["checkpoint_path"] = training_jobs[it].out_best_model
            config_copy["audio_features_to_decode_folder"] = (
                decoding_featurize_job.out_features_precompute_pca512_cls128_mean
            )
            config_copy["audio_features_manifest"] = process_audio_manifest
            config_copy["decoding_audio_name"] = decoding_audio_name
            config_copy["gen_subset"] = "preprocessed_audio"
            config_copy["acoustic_model_id"] = it
            gan_rasr_configs.append(config_copy)

        for t in models_to_decode_training_data_with:
            training_job = training_jobs[t]

            for rasr_config in gan_rasr_init_configs:
                gan_rasr_config = rasr_config.copy()
                gan_rasr_config["checkpoint_path"] = training_job.out_best_model
                gan_rasr_config["audio_features_to_decode_folder"] = (
                    featurize_training_audio_job.out_features_precompute_pca512_cls128_mean
                )
                gan_rasr_config["audio_features_manifest"] = process_training_audio_manifest
                gan_rasr_config["decoding_audio_name"] = decoding_audio_name
                gan_rasr_config["gen_subset"] = "train"
                gan_rasr_config["acoustic_model_id"] = t

                lexicon_xml_path = LexiconFromTextFileJob(
                    text_file=gan_rasr_config["language_model"]["lexicon_path"], labels=get_default_phn_order()
                ).out_bliss_lexicon

                rasr_config_file = GetTreeTimesyncRecogConfigJob(
                    rasr_binary_path=gan_rasr_config["rasr_binary_path"],
                    max_beam_size=gan_rasr_config["max_beam_size"],
                    intermediate_max_beam_size=gan_rasr_config["intermediate_max_beam_size"],
                    score_threshold=gan_rasr_config["score_threshold"],
                    intermediate_score_threshold=gan_rasr_config["intermediate_score_threshold"],
                    logfile_suffix="decoding",
                    lm_path=gan_rasr_config["language_model"]["lm_path"],
                    lm_scale=gan_rasr_config["language_model"]["scale"],
                    am_scale=gan_rasr_config["acoustic_model"]["scale"],
                    lexicon_file_path=lexicon_xml_path,
                ).output_rasr_config_file_path

                alias = os.path.join(
                    "rasr_decoding_results",
                    f"rasr_decoding_{gan_rasr_config['decoding_audio_name']}_train_{1-training_valid_percent}_am_{gan_rasr_config['acoustic_model_id']}_lm_{os.path.basename(gan_rasr_config['language_model']['lm_path'])}_lmScale_{gan_rasr_config['language_model']['scale']}_amScale_{gan_rasr_config['acoustic_model']['scale']}_blankWeight_{gan_rasr_config['acoustic_model']['blank_weight']}_decodeStride_{gan_rasr_config['acoustic_model']['decode_stride']}_beamSize_{gan_rasr_config['max_beam_size']}_scoreThreshold_{gan_rasr_config['score_threshold']}",
                )

                fairseq_rasr_decode_train_job = FairseqRasrDecode(gan_rasr_config, rasr_config_file)
                fairseq_rasr_decode_train_job.add_alias(alias + "/fairseq_rasr_decode")

                gan_rasr_config = rasr_config.copy()
                gan_rasr_config["checkpoint_path"] = training_job.out_best_model
                gan_rasr_config["audio_features_to_decode_folder"] = (
                    featurize_training_audio_job.out_features_precompute_pca512_cls128_mean
                )
                gan_rasr_config["audio_features_manifest"] = process_training_audio_manifest
                gan_rasr_config["decoding_audio_name"] = decoding_audio_name
                gan_rasr_config["gen_subset"] = "valid"
                gan_rasr_config["acoustic_model_id"] = t

                lexicon_xml_path = LexiconFromTextFileJob(
                    text_file=gan_rasr_config["language_model"]["lexicon_path"], labels=get_default_phn_order()
                ).out_bliss_lexicon

                rasr_config_file = GetTreeTimesyncRecogConfigJob(
                    rasr_binary_path=gan_rasr_config["rasr_binary_path"],
                    max_beam_size=gan_rasr_config["max_beam_size"],
                    intermediate_max_beam_size=gan_rasr_config["intermediate_max_beam_size"],
                    score_threshold=gan_rasr_config["score_threshold"],
                    intermediate_score_threshold=gan_rasr_config["intermediate_score_threshold"],
                    logfile_suffix="decoding",
                    lm_path=gan_rasr_config["language_model"]["lm_path"],
                    lm_scale=gan_rasr_config["language_model"]["scale"],
                    am_scale=gan_rasr_config["acoustic_model"]["scale"],
                    lexicon_file_path=lexicon_xml_path,
                ).output_rasr_config_file_path

                alias = os.path.join(
                    "rasr_decoding_results",
                    f"rasr_decoding_{gan_rasr_config['decoding_audio_name']}_valid_{training_valid_percent}_am_{gan_rasr_config['acoustic_model_id']}_lm_{os.path.basename(gan_rasr_config['language_model']['lm_path'])}_lmScale_{gan_rasr_config['language_model']['scale']}_amScale_{gan_rasr_config['acoustic_model']['scale']}_blankWeight_{gan_rasr_config['acoustic_model']['blank_weight']}_decodeStride_{gan_rasr_config['acoustic_model']['decode_stride']}_beamSize_{gan_rasr_config['max_beam_size']}_scoreThreshold_{gan_rasr_config['score_threshold']}",
                )

                fairseq_rasr_decode_valid_job = FairseqRasrDecode(gan_rasr_config, rasr_config_file)
                fairseq_rasr_decode_valid_job.add_alias(alias + "/fairseq_rasr_decode")

                divide_job = DivideLibriSpeech960hInto100h360h500hJob(
                    [
                        fairseq_rasr_decode_train_job.transcription_generated_formatted,
                        fairseq_rasr_decode_valid_job.transcription_generated_formatted,
                    ]
                )
                divide_job.add_alias(
                    os.path.join(alias, "rasr", "decoding", decoding_audio_name, "train_valid_divided", f"t_{t}")
                )
                tk.register_output(
                    os.path.join(alias, "rasr", "decoding", decoding_audio_name, "train_valid_divided", f"t_{t}"),
                    divide_job.out_label_path_dict,
                )
                divide_jobs.append(divide_job.out_label_path_dict.get_path())


def py():
    run_rasr_experimets()
