from i6_core.recognition.scoring import ScliteJob

from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import GetW2VLibriSpeechGroundTruthJob, TakeNRandomLinesFromTextFileJob

from recipe.rasr_decod.gan_rasr import FairseqRasrDecode
from recipe.rasr_decod.rasr_utils import LexiconFromTextFileJob, GetTreeTimesyncRecogConfigJob
import os
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.audio_preprocessing import process_audio
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_u_GAN import FairseqHydraTrainWav2VecUJob

from i6_core.tools.download import DownloadJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob
from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import FairseqNormalizeAndPrepareTextJob, CreateLexiconAndPhonemizeTextJob, TrainKenLMJob, calculate_all_configs, get_rvad_root
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import get_fairseq_root
from sisyphus import tk, Job, Task


from i6_experiments.users.enrique.experiments.wav2vec_u.default_tools import KENLM_BINARY_PATH, SCTK_BINARY_PATH

def pipeline():

    environment = "/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3"
    environment = tk.Path(environment)
    fairseq_root = get_fairseq_root(
        python_env=environment,
        fairseq_root=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
        ),
    )
    fasttext_model = DownloadJob(
        url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", target_filename="lid.176.bin"
    ).out_file

    word_level_lm_text_file_path = "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"
    #word_level_lm_text_file_path = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/small_text_test/text.txt"
    word_level_lm_text_file_path = tk.Path(word_level_lm_text_file_path)
    n_lines_for_sentencepiece_training = 1000000

    sentencepiece_text_file_path = TakeNRandomLinesFromTextFileJob(
        input_text_file=word_level_lm_text_file_path, n_lines=n_lines_for_sentencepiece_training
    ).output_text_file

    
    #vocab_sizes = [50, 100, 200, 300, 500, 1200, 2000]
    vocab_sizes = [50]
    
    vocab_type = "unigram"  # options: "bpe" or "unigram"
    assert vocab_type in ["bpe", "unigram"], "vocab_type must be either 'bpe' or 'unigram'"

    label_level_lm_order = 4
    word_level_lm_order = 4
    silence_prob = 0.25


    training_configs = {
        "model.code_penalty": [2, 4],
        "model.gradient_penalty": [1.5, 2],
        "model.smoothness_weight": [0.5, 0.75],
    }
    config_dir = fairseq_root.get_path() + "/examples/wav2vec/unsupervised/config/gan"
    config_name = "w2vu"

    model_seed_range = range(0, 3)

    decoding_datasets = [
        {
            "name": "test-other",
            "extension": "flac",
            "subset": "test-other",
            #"valid_percent": None,
        },
    ]
    prepare_decoding_audio_concurrent = 16
    
    # word level LM
    train_word_level_kenlm = TrainKenLMJob(  
        kenlm_root=KENLM_BINARY_PATH,
        text_file_path=word_level_lm_text_file_path,
        lm_order=word_level_lm_order,
    ).output_arpa
    
    rasr_binary_path = tk.Path("/work/asr3/berger/hiwis/kleppel/rasr_dev/hmm-treebuilder/rasr_monophone_2/arch/linux-x86_64-standard")
    beam_sizes_to_try = [128, 512, 1024, 2048]
    #beam_sizes_to_try = [128]
    lm_scales_to_try = [0.5, 0.7, 0.9, 1.1]
    #lm_scales_to_try = [0.69]
    gan_rasr_configs = [
        {
            'rasr_binary_path': rasr_binary_path,
            'language_model': {
                'lm_path': train_word_level_kenlm,
                'scale': lm_scale,
            },
            'acoustic_model': {
                'scale': 1.0,
                'blank_weight': 0.0,
                'blank_mode': 'add',
                'decode_stride': 1,
                'sil_is_blank': True,
            },
            'max_beam_size': beam_size,
            'intermediate_max_beam_size': beam_size,
            'score_threshold': 14.0,
            'intermediate_score_threshold': 14.0,
        } for beam_size in beam_sizes_to_try for lm_scale in lm_scales_to_try
    ]
    



    w2v2model = "large_60kh"  # Options: "base", "large_960h", "large_60kh"
    feature_extraction_layer = 14  # Layer to extract features from the Wav2Vec2 model, w2v-u paper uses layer 14
    assert not (
        (w2v2model == "base" and feature_extraction_layer > 11) or feature_extraction_layer > 23
    ), "not so many layers in the model"

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

    training_audio = "train-other-960"  # Options: "train-clean-100", "train-clean-360", "train-other-500", "train-other-960", "dev-clean", "dev-complete", "dev-other", "test-clean", "test-other"
    training_audio_extension = "flac"  # Options: "flac", "wav"
    training_valid_percent = 0.01  # Percentage of the training data to be used for validation
    training_concurrent = 8  # Number of concurrent processes to run for audio processing

    alias = "wav2vec_u_librispeech_gan_training_" + training_audio + "_" + w2v2model
    audio_alias = os.path.join(alias, "audio")
    training_audio_alias = os.path.join(audio_alias, training_audio)

    training_audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", training_audio))
    _, featurize_training_audio_job, process_training_audio_manifest = process_audio(
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
        name_the_manifests_just_train_and_valid=True,
    )

    


    all_stats = {}
    for vocab_size in vocab_sizes:
        vocab_alias = os.path.join(alias, f"{vocab_type}_vocab_size_{vocab_size}")

        stats = {}
        train_sentence_piece_job = TrainSentencePieceJob(
            training_text=sentencepiece_text_file_path,
            vocab_size=vocab_size,
            model_type=SentencePieceType.BPE if vocab_type == "bpe" else SentencePieceType.UNIGRAM, #  SentencePieceType.BPE or SentencePieceType.UNIGRAM
            additional_options={"max_sentence_length": 1000000, "shuffle_input_sentence": False}
        )

        fairseq_normalize_and_prepare_text_job = FairseqNormalizeAndPrepareTextJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            text_file_path=word_level_lm_text_file_path,
            language='en',
            lid_path=fasttext_model,
        )

        apply_sentencepiece_job = ApplySentencepieceToTextJob(text_file=fairseq_normalize_and_prepare_text_job.words_txt, sentencepiece_model=train_sentence_piece_job.out_model, gzip_output=False)

        create_lexicon_and_phonemize_text_job = CreateLexiconAndPhonemizeTextJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            text_file_path=fairseq_normalize_and_prepare_text_job.normalize_text_lid,
            words_txt=fairseq_normalize_and_prepare_text_job.words_txt,
            phones_txt=apply_sentencepiece_job.out_sentencepiece_text,
            insert_silence_between_words_prob=silence_prob,
        )

        # phoneme or bpe level LM
        train_unit_level_kenlm_job = TrainKenLMJob(  
            kenlm_root=KENLM_BINARY_PATH,
            text_file_path=create_lexicon_and_phonemize_text_job.labeled_text,
            lm_order=label_level_lm_order,
        )       

        training_configs["common.seed"] = model_seed_range
        GAN_job_all_configs = FairseqHydraTrainWav2VecUJob(
            environment=environment,
            task_data=featurize_training_audio_job.out_features_precompute_pca512_cls128_mean_pooled,
            task_text_data=create_lexicon_and_phonemize_text_job.labels_folder,
            fairseq_root=fairseq_root,
            prefix=vocab_alias,
            config_dir=config_dir,
            config_name=config_name,
            extra_configs=training_configs,
            task_kenlm_path=train_unit_level_kenlm_job.output_bin,
        )
        GAN_job_all_configs.add_alias(vocab_alias + "/gan_training_all_configs")

        for gan_rasr_config in gan_rasr_configs:
            for decoding_dataset in decoding_datasets:
                decoding_audio_name = decoding_dataset["name"]
                decoding_audio_alias = os.path.join(audio_alias, decoding_audio_name)
                decoding_delete_silences_job, decoding_featurize_job, process_audio_manifest = process_audio(
                    env=environment,
                    fairseq_root=fairseq_root,
                    audio_dir=tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", decoding_audio_name)),
                    valid_percent=decoding_dataset.get("valid_percent"),
                    ext=decoding_dataset["extension"],
                    rvad_root=get_rvad_root(),
                    concurrent=prepare_decoding_audio_concurrent,
                    layer=feature_extraction_layer,
                    model_path=w2v2_model_path,
                    alias_prefix=decoding_audio_alias,
                    alias_delete="delete_silences"
                    + "/"
                    + w2v2model
                    + "/layer_"
                    + str(feature_extraction_layer)
                    + "valid_"
                    + str(decoding_dataset.get("valid_percent")),
                    alias_feat="featurize_audio/"
                    + w2v2model
                    + "/layer_"
                    + str(feature_extraction_layer)
                    + "valid_"
                    + str(decoding_dataset.get("valid_percent")),
                    existing_clusters=featurize_training_audio_job.out_features_clusters,
                    existing_pca=featurize_training_audio_job.out_features_pca,
                    name_the_manifests_just_train_and_valid=False,
                )
                gan_rasr_config["checkpoint_path"] = GAN_job_all_configs.out_best_model
                gan_rasr_config['audio_features_to_decode_folder'] = decoding_featurize_job.out_features_precompute_pca512_cls128_mean
                gan_rasr_config['gen_subset'] = 'preprocessed_audio'
                gan_rasr_config['decoding_audio_name'] = decoding_audio_name
                gan_rasr_config['acoustic_model_id'] = f"all_configs"
                gan_rasr_config['audio_features_manifest'] = process_audio_manifest
                gan_rasr_config['labeled_text_folder'] = create_lexicon_and_phonemize_text_job.labels_folder





                lexicon_xml_path = LexiconFromTextFileJob(text_file=create_lexicon_and_phonemize_text_job.lexicon_filtered_lst, labels=create_lexicon_and_phonemize_text_job.output_label_list_with_bos_eos_pad_unk).out_bliss_lexicon

                rasr_config_file = GetTreeTimesyncRecogConfigJob(
                    rasr_binary_path=gan_rasr_config["rasr_binary_path"],
                    max_beam_size=gan_rasr_config["max_beam_size"],
                    intermediate_max_beam_size=gan_rasr_config["intermediate_max_beam_size"],
                    score_threshold=gan_rasr_config["score_threshold"],
                    intermediate_score_threshold=gan_rasr_config["intermediate_score_threshold"],
                    logfile_suffix="decoding",
                    lm_path=train_unit_level_kenlm_job.output_bin,
                    lm_scale=gan_rasr_config["language_model"]["scale"],
                    am_scale=gan_rasr_config["acoustic_model"]["scale"],
                    lexicon_file_path=lexicon_xml_path).output_rasr_config_file_path
                
                decode_job_alias = os.path.join("rasr_decoding_results", f"vocab_size{vocab_size}", f"rasr_decoding_{gan_rasr_config['decoding_audio_name']}_valid_{gan_rasr_config.get('valid_percent', '')}_am_{gan_rasr_config['acoustic_model_id']}_lm_{os.path.basename(train_word_level_kenlm.get_path())}_lmScale_{gan_rasr_config['language_model']['scale']}_amScale_{gan_rasr_config['acoustic_model']['scale']}_blankWeight_{gan_rasr_config['acoustic_model']['blank_weight']}_decodeStride_{gan_rasr_config['acoustic_model']['decode_stride']}_beamSize_{gan_rasr_config['max_beam_size']}_scoreThreshold_{gan_rasr_config['score_threshold']}")

                fairseq_rasr_decode_valid_job = FairseqRasrDecode(gan_rasr_config, rasr_config_file)
                fairseq_rasr_decode_valid_job.add_alias(decode_job_alias + "/fairseq_rasr_decode")

                ground_truth_job = GetW2VLibriSpeechGroundTruthJob(audio_manifest=gan_rasr_config['audio_features_manifest'], librispeech_subset=gan_rasr_config['decoding_audio_name'])
                ref = ground_truth_job.ground_truth_stm
                hyp = fairseq_rasr_decode_valid_job.transcription_generated_ctm

                
                sclite_job = ScliteJob(ref=ref, hyp=hyp, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
                sclite_job.add_alias(decode_job_alias + "/sclite")
                tk.register_output("wer",sclite_job.out_report_dir)

                if sclite_job.out_wer.get() is not None:
                    stats[decode_job_alias] = {"sclite_job": sclite_job, "configuration": gan_rasr_config}

        all_stats[vocab_size] = stats
    
    for vocab_size, stats in all_stats.items():
        for alias, stat in stats.items():
            sclite_job = stat["sclite_job"]
            wer = sclite_job.out_wer.get()
            print(f"Vocab size: {vocab_size}, Alias: {alias}, WER: {wer}, path{sclite_job.out_report_dir.get_path()}")
    

def py():
    pipeline()