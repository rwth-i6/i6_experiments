###########################################################
# Imports
###########################################################

import sys
from copy import deepcopy
import os
from i6_core.recognition.scoring import ScliteJob
from i6_core.tools.download import DownloadJob

import logging
from sisyphus import tk

from recipe.rasr_decod.gan_rasr import FairseqRasrDecode
from recipe.rasr_decod.rasr_utils import LexiconFromTextFileJob, GetTreeTimesyncRecogConfigJob

from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob

from recipe.i6_experiments.users.enrique.experiments.gan.prepare_text import (
    FairseqG2PWordToPhnJob,
    CreateLexiconAndDictionaryJob,
    TokenizeWithLexiconAndSilenceJob,
    FairseqNormalizeTextAndCreateDictionary,
    FairseqPreprocessJob,
    WordToLetterJob,
    TrainKenLMJob,
    FairseqNormalizeAndPrepareTextJob,
)
from recipe.i6_experiments.users.enrique.experiments.gan.process_audio import process_audio
from recipe.i6_experiments.users.enrique.experiments.gan.hydra_train import FairseqHydraTrainJob
from recipe.i6_experiments.users.enrique.experiments.gan.utils import get_fairseq_root, KENLM_BINARY_PATH, get_rvad_root, expand_dictionary, ExpandableIterable
from recipe.i6_experiments.users.enrique.experiments.gan.decoding import GANw2vGenerateJob#, UnsupervisedVocabUsageAndLMScoreMetric

def get_w2v2_model_path(model_name: str) -> str:
    if model_name == "large_960h":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt",
            target_filename="wav2vec2_large_960h_no_finetune.pt",
        ).out_file
    if model_name == "large_60kh":
        assert model_name == "large_60kh"
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt",
            target_filename="wav2vec_60kh_no_finetune.pt",
        ).out_file
    if model_name == "base":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
            target_filename="wav2vec_small.pt",
        ).out_file
    return w2v2_model_path

def run_gan_experiments():
    ###########################################################
    ## Environment and Fairseq Root
    ###########################################################
    environment = "/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3"
    environment = tk.Path(environment)
    fairseq_root = "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
    fairseq_root = tk.Path(fairseq_root)
    fairseq_root = get_fairseq_root( # installs fairseq in environment ONLY WHEN JOBS ARE NOT ALREADY RUNNED
        python_env=environment,
        fairseq_root=fairseq_root,
    )
    rvad_root = get_rvad_root()

    ###########################################################
    ## Audio Processing
    ###########################################################
    training_datasets = {
        "LibriSpeech-train-other-960": { # whichever name you prefer, must be unique
            "path": tk.Path("/u/corpora/speech/LibriSpeech/LibriSpeech/train-other-960"),
            "dataset": "LibriSpeech", 
            "extension": "flac", # audio file extension to search for
            "subset": "train-other-960",
            "valid_percent": 0.01, # percentage of data to use for validation, CAREFUL IT BREAKS THE HASH
            "w2v2": {
                "model": "large_60kh",  # Options: "base", "large_960h", "large_60kh"
                "feature_extraction_layer": 14,  # which layer to extract features from w2v2 model
            },
            "concurrent_processing": 40, # number of concurrent processes to use for audio processing
            "name_the_manifests_just_train_and_valid": True, # whether to name the manifests just train and valid or not
        }
    }
    
    decoding_datasets = {
        "LibriSpeech-test-other": {
            "path": tk.Path("/u/corpora/speech/LibriSpeech/LibriSpeech/test-other"),
            "dataset": "LibriSpeech", 
            "extension": "flac",
            "subset": "test-other",
            "w2v2": {
                "model": "large_60kh",  
                "feature_extraction_layer": 14  
            },
            "name_of_anchor_dataset": "LibriSpeech-train-other-960", # PCA matrix and clusters will be extracted from this job
        },

        "LibriSpeech-test-clean": {
            "path": tk.Path("/u/corpora/speech/LibriSpeech/LibriSpeech/test-clean"),
            "dataset": "LibriSpeech", 
            "extension": "flac",
            "subset": "test-clean",
            "w2v2": {
                "model": "large_60kh",
                "feature_extraction_layer": 14 
            },
            "name_of_anchor_dataset": "LibriSpeech-train-other-960",
        },
    }
    all_audio_datasets = training_datasets | decoding_datasets

    audio_jobs = {}
    for name, dataset in list(training_datasets.items()) + list(decoding_datasets.items()):
        w2v2model = dataset["w2v2"]["model"]
        feature_extraction_layer = dataset["w2v2"]["feature_extraction_layer"]
        assert not (
            (w2v2model == "base" and feature_extraction_layer > 11) or feature_extraction_layer > 23
        ), "not that many layers in the model"
        
        if "name_of_anchor_dataset" in dataset:
            assert all_audio_datasets[dataset["name_of_anchor_dataset"]]["w2v2"]["feature_extraction_layer"] == feature_extraction_layer, "Feature extraction layer must be the same for training and decoding datasets that are linked."
        if name in training_datasets:
            assert dataset["name_the_manifests_just_train_and_valid"], "For training datasets, name_the_manifests_just_train_and_valid must be True." # TODO: this should be fixed in the future, the whole process_audio function should be more flexible

        w2v2_model_path = get_w2v2_model_path(model_name=w2v2model)

        audio_alias = "audio/" + name + "/" + w2v2model + "/" + "layer_" + str(feature_extraction_layer) + "/" + ("valid_" + str(dataset.get("valid_percent", 0.0)) if dataset.get("valid_percent", 0.0) > 0.0 else "full")

        delete_silences_job, featurize_training_audio_job, process_training_audio_manifest = process_audio(
            env=environment,
            fairseq_root=fairseq_root,
            audio_dir=dataset["path"],
            valid_percent=dataset.get("valid_percent", 0.0),
            ext=dataset.get("extension", "wav"),
            rvad_root=rvad_root,
            concurrent=dataset.get("concurrent_processing", 8),
            delete_silence_concurrent=dataset.get("concurrent_processing_silence_removal", None),
            featurize_audio_concurrent=dataset.get("concurrent_processing_featurize_audio", None),
            layer=dataset["w2v2"]["feature_extraction_layer"],
            model_path=w2v2_model_path,
            alias_prefix=audio_alias,
            name_the_manifests_just_train_and_valid=dataset.get("name_the_manifests_just_train_and_valid", False),
            existing_clusters=None if "name_of_anchor_dataset" not in dataset else audio_jobs[dataset["name_of_anchor_dataset"]]["featurize_job"].out_features_clusters,
            existing_pca=None if "name_of_anchor_dataset" not in dataset else audio_jobs[dataset["name_of_anchor_dataset"]]["featurize_job"].out_features_pca,

        )
        audio_jobs[name] = {
            "config": {"name": name, **dataset},
            "delete_silence_job": delete_silences_job,
            "featurize_job": featurize_training_audio_job,
            "manifest": process_training_audio_manifest,
        }
   
    ############################################################
    ## Text data preparation, lexicon, LMs, etc.
    ############################################################

    fasttext_model = DownloadJob(
        url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", target_filename="lid.176.bin"
    ).out_file
    word_level_text_language_models = {
        "LibriSpeech-no-LibriVox_wrd-4-gram-0-0-1-4":{
            "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
            "level": "word",
            "order": 4,
            "pruning": [0, 0, 1, 4],
            "lid_path": fasttext_model,
        }
    }

    letter_level_text_language_models = {
        "LibriSpeech-no-LibriVox_chr-6-gram":{
            "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
            "level": "chr",
            "order": 6,
            "lid_path": fasttext_model,
            "insert_silence_between_words_prob": 0.25,
            "silence_token": "<SIL>",
        }
    }

    phoneme_level_text_language_models = {
        "LibriSpeech-no-LibriVox_phn-4-gram":{
            "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
            "level": "phn",
            "order": 4,
            "phonemizer": "g2p_en",
            "lid_path": fasttext_model,
            "insert_silence_between_words_prob": 0.25,
            "silence_token": "<SIL>",
        },
        "LibriSpeech-no-LibriVox_phn-6-gram":{
            "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
            "level": "phn",
            "order": 6,
            "phonemizer": "g2p_en",
            "lid_path": fasttext_model,
            "insert_silence_between_words_prob": 0.25,
            "silence_token": "<SIL>",
        }
    }
    bpe_level_text_language_models = {
        "LibriSpeech-no-LibriVox_bpe_50_6-gram":{
            "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.bpe.txt"),
            "level": "bpe",
            "vocab_size": 50,
            "order": 6,
            "insert_silence_between_words_prob": 0.25,
            "silence_token": "<SIL>",
        }
    }
    unigram_level_text_language_models = {
        "LibriSpeech-no-LibriVox_unigram_50_4-gram":{
            "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.bpe.txt"),
            "level": "uni",
            "vocab_size": 50,
            "order": 4,
            "insert_silence_between_words_prob": 0.25,
            "silence_token": "<SIL>",
        }
    }

    word_level_text_language_models_jobs = {}
    for lm_name, lm_info in word_level_text_language_models.items():
        normalize_text_job = FairseqNormalizeAndPrepareTextJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            text_file_path=lm_info["text_file"],
            language=lm_info.get("language", "en"),
            lid_path=lm_info.get("lid_path", None),
        )

        train_kenlm_job = TrainKenLMJob(
            kenlm_root=KENLM_BINARY_PATH,
            text_file_path=normalize_text_job.normalize_text_lid,
            lm_order=lm_info["order"],
            pruning=lm_info["pruning"],
        )
        word_level_text_language_models_jobs[lm_name] = {
            "config": lm_info,
            "normalize_text_job": normalize_text_job,
            "train_kenlm_job": train_kenlm_job,
        }

    letter_level_text_language_models_jobs = {}
    for lm_name, lm_info in letter_level_text_language_models.items():
        normalize_text_job = FairseqNormalizeTextAndCreateDictionary(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            text_file_path=lm_info["text_file"],
            language=lm_info.get("language", "en"),
            lid_path=lm_info.get("lid_path"),
        )     

        fairseq_g2p_word_to_phn_job = WordToLetterJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            words_file=normalize_text_job.words_txt,
        )

        create_lexicon_job = CreateLexiconAndDictionaryJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            words_txt=normalize_text_job.words_txt,
            tokenized_words=fairseq_g2p_word_to_phn_job.letters_file,
        )

        label_text_with_lexicon_and_silence_job = TokenizeWithLexiconAndSilenceJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            text_file_path=normalize_text_job.normalize_text_lid,
            lexicon_lst=create_lexicon_job.lexicon_lst,
            insert_silence_between_words_prob=lm_info.get("insert_silence_between_words_prob"),
            silence_token=lm_info.get("silence_token"),
            surround_with_silence=True,
        )

        fairseq_preprocess_job = FairseqPreprocessJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            tokenized_text=label_text_with_lexicon_and_silence_job.tokenized_text_with_silence,
            dict_labeltype_txt=create_lexicon_job.dict_labeltype_txt,
        )
        train_kenlm_job_with_silence = TrainKenLMJob(
            kenlm_root=KENLM_BINARY_PATH,
            text_file_path=normalize_text_job.normalize_text_lid,
            lm_order=lm_info["order"],
            pruning=lm_info.get("pruning", None),
        )

        label_text_with_lexicon_job = TokenizeWithLexiconAndSilenceJob( # no silence included
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            text_file_path=normalize_text_job.normalize_text_lid,
            lexicon_lst=create_lexicon_job.lexicon_lst,
            insert_silence_between_words_prob=0.0,
            silence_token=lm_info.get("silence_token"),
            surround_with_silence=True,
        )

        train_kenlm_job_no_silence = TrainKenLMJob(
            kenlm_root=KENLM_BINARY_PATH,
            text_file_path=label_text_with_lexicon_job.tokenized_text_with_silence, # NOTE: actually no silence
            lm_order=lm_info["order"],
            pruning=lm_info.get("pruning", None),
        )
        letter_level_text_language_models_jobs[lm_name] = {
            "config": lm_info,
            "normalize_text_job": normalize_text_job,
            "fairseq_g2p_word_to_phn_job": fairseq_g2p_word_to_phn_job,
            "create_lexicon_job": create_lexicon_job,
            "label_text_with_lexicon_and_silence_job": label_text_with_lexicon_and_silence_job,
            "fairseq_preprocess_job": fairseq_preprocess_job,
            "train_kenlm_job_with_silence": train_kenlm_job_with_silence,
            "train_kenlm_job_no_silence": train_kenlm_job_no_silence,
        }


    phoneme_level_text_language_models_jobs = {}
    for lm_name, lm_info in phoneme_level_text_language_models.items():
        assert ("phonemizer" in lm_info and lm_info["phonemizer"] == "g2p_en"), "Only g2p_en phonemizer is supported currently."
        if "phonemizer" in lm_info and lm_info["phonemizer"] == "g2p_en":
            normalize_text_job = FairseqNormalizeTextAndCreateDictionary(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=lm_info["text_file"],
                language=lm_info.get("language", "en"),
                lid_path=lm_info.get("lid_path"),
            )     

            fairseq_g2p_word_to_phn_job = FairseqG2PWordToPhnJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                words_file=normalize_text_job.words_txt,
            )

            create_lexicon_job = CreateLexiconAndDictionaryJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                words_txt=normalize_text_job.words_txt,
                tokenized_words=fairseq_g2p_word_to_phn_job.phn_file,
            )

            label_text_with_lexicon_and_silence_job = TokenizeWithLexiconAndSilenceJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=normalize_text_job.normalize_text_lid,
                lexicon_lst=create_lexicon_job.lexicon_lst,
                insert_silence_between_words_prob=lm_info.get("insert_silence_between_words_prob"),
                silence_token=lm_info.get("silence_token"),
                surround_with_silence=True,
            )

            fairseq_preprocess_job = FairseqPreprocessJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                tokenized_text=label_text_with_lexicon_and_silence_job.tokenized_text_with_silence,
                dict_labeltype_txt=create_lexicon_job.dict_labeltype_txt,
            )
            train_kenlm_job_with_silence = TrainKenLMJob(
                kenlm_root=KENLM_BINARY_PATH,
                text_file_path=label_text_with_lexicon_and_silence_job.tokenized_text_with_silence,
                lm_order=lm_info["order"],
                pruning=lm_info.get("pruning", None),
            )

            label_text_with_lexicon_job = TokenizeWithLexiconAndSilenceJob( # no silence included
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=normalize_text_job.normalize_text_lid,
                lexicon_lst=create_lexicon_job.lexicon_lst,
                insert_silence_between_words_prob=0.0,
                silence_token=lm_info.get("silence_token"),
                surround_with_silence=True,
            )

            train_kenlm_job_no_silence = TrainKenLMJob(
                kenlm_root=KENLM_BINARY_PATH,
                text_file_path=label_text_with_lexicon_job.tokenized_text_with_silence, # NOTE: actually no silence
                lm_order=lm_info["order"],
                pruning=lm_info.get("pruning", None),
            )

        phoneme_level_text_language_models_jobs[lm_name] = {
            "config": lm_info,
            "normalize_text_job": normalize_text_job,
            "fairseq_g2p_word_to_phn_job": fairseq_g2p_word_to_phn_job,
            "create_lexicon_job": create_lexicon_job,
            "label_text_with_lexicon_and_silence_job": label_text_with_lexicon_and_silence_job,
            "fairseq_preprocess_job": fairseq_preprocess_job,
            "train_kenlm_job_with_silence": train_kenlm_job_with_silence,
            "train_kenlm_job_no_silence": train_kenlm_job_no_silence,
        }

        bpe_level_text_language_models_jobs = {}
        for lm_name, lm_info in bpe_level_text_language_models.items():
            normalize_text_job = FairseqNormalizeTextAndCreateDictionary(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=lm_info["text_file"],
                language=lm_info.get("language", "en"),
                lid_path=lm_info.get("lid_path"),
            )   

            train_sentence_piece_job = TrainSentencePieceJob(
                training_text=normalize_text_job.normalize_text_lid,
                vocab_size=lm_info["vocab_size"],
                model_type=SentencePieceType.BPE,
                additional_options={"max_sentence_length": 1000000, "shuffle_input_sentence": False}
            )

            apply_sentencepiece_job = ApplySentencepieceToTextJob(text_file=normalize_text_job.words_txt, sentencepiece_model=train_sentence_piece_job.out_model, gzip_output=False)

            create_lexicon_job = CreateLexiconAndDictionaryJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                words_txt=apply_sentencepiece_job.out_sentencepiece_text,
                tokenized_words=apply_sentencepiece_job.out_sentencepiece_text,
            )

            label_text_with_lexicon_and_silence_job = TokenizeWithLexiconAndSilenceJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=normalize_text_job.normalize_text_lid,
                lexicon_lst=create_lexicon_job.lexicon_lst,
                insert_silence_between_words_prob=lm_info.get("insert_silence_between_words_prob"),
                silence_token=lm_info.get("silence_token"),
                surround_with_silence=True,
            )

            fairseq_preprocess_job = FairseqPreprocessJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                tokenized_text=label_text_with_lexicon_and_silence_job.tokenized_text_with_silence,
                dict_labeltype_txt=create_lexicon_job.dict_labeltype_txt,
            )
            train_kenlm_job_with_silence = TrainKenLMJob(
                kenlm_root=KENLM_BINARY_PATH,
                text_file_path=normalize_text_job.normalize_text_lid,
                lm_order=lm_info["order"],
                pruning=lm_info.get("pruning", None),
            )
            label_text_with_lexicon_job = TokenizeWithLexiconAndSilenceJob( # no silence included
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=normalize_text_job.normalize_text_lid,
                lexicon_lst=create_lexicon_job.lexicon_lst,
                insert_silence_between_words_prob=0.0,
                silence_token=lm_info.get("silence_token"),
                surround_with_silence=True,
            )

            train_kenlm_job_no_silence = TrainKenLMJob(
                kenlm_root=KENLM_BINARY_PATH,
                text_file_path=label_text_with_lexicon_job.tokenized_text_with_silence, # NOTE: actually no silence
                lm_order=lm_info["order"],
                pruning=lm_info.get("pruning", None),
            )
            bpe_level_text_language_models_jobs[lm_name] = {
                "config": lm_info,
                "normalize_text_job": normalize_text_job,
                "train_sentence_piece_job": train_sentence_piece_job,
                "apply_sentencepiece_job": apply_sentencepiece_job,
                "create_lexicon_job": create_lexicon_job,
                "label_text_with_lexicon_and_silence_job": label_text_with_lexicon_and_silence_job,
                "fairseq_preprocess_job": fairseq_preprocess_job,
                "train_kenlm_job_with_silence": train_kenlm_job_with_silence,
                "train_kenlm_job_no_silence": train_kenlm_job_no_silence,
            }


        unigram_level_text_language_models_jobs = {}
        for lm_name, lm_info in unigram_level_text_language_models.items():
            normalize_text_job = FairseqNormalizeTextAndCreateDictionary(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=lm_info["text_file"],
                language=lm_info.get("language", "en"),
                lid_path=lm_info.get("lid_path"),
            )   

            train_sentence_piece_job = TrainSentencePieceJob(
                training_text=normalize_text_job.normalize_text_lid,
                vocab_size=lm_info["vocab_size"],
                model_type=SentencePieceType.UNIGRAM,
                additional_options={"max_sentence_length": 1000000, "shuffle_input_sentence": False}
            )

            apply_sentencepiece_job = ApplySentencepieceToTextJob(text_file=normalize_text_job.words_txt, sentencepiece_model=train_sentence_piece_job.out_model, gzip_output=False)

            create_lexicon_job = CreateLexiconAndDictionaryJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                words_txt=apply_sentencepiece_job.out_sentencepiece_text,
                tokenized_words=apply_sentencepiece_job.out_sentencepiece_text,
            )

            label_text_with_lexicon_and_silence_job = TokenizeWithLexiconAndSilenceJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=normalize_text_job.normalize_text_lid,
                lexicon_lst=create_lexicon_job.lexicon_lst,
                insert_silence_between_words_prob=lm_info.get("insert_silence_between_words_prob"),
                silence_token=lm_info.get("silence_token"),
                surround_with_silence=True,
            )

            fairseq_preprocess_job = FairseqPreprocessJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                tokenized_text=label_text_with_lexicon_and_silence_job.tokenized_text_with_silence,
                dict_labeltype_txt=create_lexicon_job.dict_labeltype_txt, 
            )
            train_kenlm_job_with_silence = TrainKenLMJob(
                kenlm_root=KENLM_BINARY_PATH,
                text_file_path=normalize_text_job.normalize_text_lid,
                lm_order=lm_info["order"],
                pruning=lm_info.get("pruning", None),
            )
            label_text_with_lexicon_job = TokenizeWithLexiconAndSilenceJob( # no silence included
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=normalize_text_job.normalize_text_lid,
                lexicon_lst=create_lexicon_job.lexicon_lst,
                insert_silence_between_words_prob=0.0,
                silence_token=lm_info.get("silence_token"),
                surround_with_silence=True,
            )

            train_kenlm_job_no_silence = TrainKenLMJob(
                kenlm_root=KENLM_BINARY_PATH,
                text_file_path=label_text_with_lexicon_job.tokenized_text_with_silence, # NOTE: actually no silence
                lm_order=lm_info["order"],
                pruning=lm_info.get("pruning", None),
            )
            unigram_level_text_language_models_jobs[lm_name] = {
                "config": lm_info,
                "normalize_text_job": normalize_text_job,
                "train_sentence_piece_job": train_sentence_piece_job,
                "apply_sentencepiece_job": apply_sentencepiece_job,
                "create_lexicon_job": create_lexicon_job,
                "label_text_with_lexicon_and_silence_job": label_text_with_lexicon_and_silence_job,
                "fairseq_preprocess_job": fairseq_preprocess_job,
                "train_kenlm_job_with_silence": train_kenlm_job_with_silence,
                "train_kenlm_job_no_silence": train_kenlm_job_no_silence,
            }
        
        all_levels_text_language_models_jobs = {
            **word_level_text_language_models_jobs,
            **letter_level_text_language_models_jobs,
            **phoneme_level_text_language_models_jobs,
            **bpe_level_text_language_models_jobs,
            **unigram_level_text_language_models_jobs,
        }


    ############################################################
    ## End of text data preparation
    ############################################################

    ###########################################################
    ## GAN training
    ###########################################################

    gan_hydra_train_configs = {
        "default_w2vu_gan_train_job":{
            "description": "Wav2Vec-U GAN training with default hyperparameters, imitating the original paper's setup.",
            "text_data_name": "LibriSpeech-no-LibriVox_phn-4-gram",
            "audio_data_name": "LibriSpeech-train-other-960",
            "fairseq_hydra_config":{
                "common": {
                    "fp16": False,
                    "fp16_no_flatten_grads": True,
                    "log_format": "json",
                    "log_interval": 100,
                    "tensorboard_logdir": "tb",
                    "reset_logging": False,
                    "suppress_crashes": False,
                    "user_dir": "???",
                    #"seed": ExpandableIterable(range(0,5)), # TODO: enable to train all models
                    "seed": 0,
                },
                "checkpoint": {
                    "save_interval": 1000,
                    "save_interval_updates": 1000,
                    "no_epoch_checkpoints": True,
                    "best_checkpoint_metric": "weighted_lm_ppl",
                    "save_dir": "???"
                },
                "distributed_training": {
                    "distributed_world_size": 1
                },
                "task": {
                    "_name": "unpaired_audio_text",
                    "data": "???",
                    "text_data": "???", 
                    "labels": "???",
                    "kenlm_path": "???", # label level, not word level
                    "sort_by_length": False,
                    "unfiltered": False,
                    "max_length": None,
                    "append_eos": False,
                },
                "dataset": {
                    "num_workers": 2,
                    "batch_size": 160,
                    "skip_invalid_size_inputs_valid_test": True,
                    "valid_subset": "valid",
                    "validate_interval": 1000,
                    "validate_interval_updates": 1000,
                    "dataset_impl": "mmap",
                },
                "criterion": {
                    "_name": "model",
                    "log_keys": [
                        "accuracy_dense",
                        "accuracy_token",
                        "temp",
                        "code_ppl"
                    ]
                },
                "optimization": {
                    "max_update": 150000,
                    "clip_norm": 5,
                    "lr": [
                        0
                    ]
                },
                "optimizer": {
                    "_name": "composite",
                    "groups": {
                        "generator": {
                            "lr": [
                                0.0004
                            ],
                            "lr_float": None,
                            "optimizer": {
                                "_name": "adam",
                                "adam_betas": [
                                    0.5,
                                    0.98
                                ],
                                "adam_eps": 0.000001,
                                "weight_decay": 0,
                                #"amsgrad": False
                            },
                            "lr_scheduler": {
                                "_name": "fixed",
                                "warmup_updates": 0
                            }
                        },
                        "discriminator": {
                            "lr": [
                                0.0005
                            ],
                            "lr_float": None,
                            "optimizer": {
                                "_name": "adam",
                                "adam_betas": [
                                    0.5,
                                    0.98
                                ],
                                "adam_eps": 0.000001,
                                "weight_decay": 0.0001,
                                #"amsgrad": False
                            },
                            "lr_scheduler": {
                                "_name": "fixed",
                                "warmup_updates": 0
                            }
                        }
                    }
                },
                "lr_scheduler": "pass_through",
                "model": {
                    "_name": "wav2vec_u",
                    # "code_penalty": ExpandableIterable([2, 4]),
                    # "gradient_penalty": ExpandableIterable([1.5, 2.0]),
                    # "smoothness_weight": ExpandableIterable([0.5, 0.75]),
                    "code_penalty": 2,
                    "gradient_penalty": 1.5,
                    "smoothness_weight": 0.5,
                    "discriminator_dim": 384,
                    "discriminator_depth": 2,
                    "discriminator_kernel": 6,
                    "discriminator_linear_emb": False,
                    "discriminator_causal": True,
                    "discriminator_max_pool": False,
                    "discriminator_act_after_linear": False,
                    "discriminator_dropout": 0,
                    "discriminator_weight_norm": False,
                    "generator_stride": 1,
                    "generator_kernel": 4,
                    "generator_bias": False,
                    "generator_dropout": 0.1,
                    "smoothing": 0,
                    "smoothing_one_sided": False,
                    "gumbel": False,
                    "hard_gumbel": False,
                    "temp": [
                        2,
                        0.1,
                        0.99995
                    ],
                    "input_dim": 512,
                    "segmentation": {
                        "type": "JOIN",
                        "mean_pool_join": False,
                        "remove_zeros": False
                    }
                }
            }
        }
    }
    
    gan_hydra_train_jobs = {}
    for gan_hydra_train_config_name, gan_hydra_train_config in gan_hydra_train_configs.items():
        fairseq_hydra_configs = deepcopy(gan_hydra_train_config["fairseq_hydra_config"])
        fairseq_hydra_configs["task"]["data"] = audio_jobs[gan_hydra_train_config["audio_data_name"]]["featurize_job"].out_features_precompute_pca512_cls128_mean_pooled
        fairseq_hydra_configs["task"]["text_data"] = all_levels_text_language_models_jobs[gan_hydra_train_config["text_data_name"]]["fairseq_preprocess_job"].labels_folder
        fairseq_hydra_configs["task"]["labels"] = all_levels_text_language_models_jobs[gan_hydra_train_config["text_data_name"]]["config"]["level"]
        fairseq_hydra_configs["task"]["kenlm_path"] = all_levels_text_language_models_jobs[gan_hydra_train_config["text_data_name"]]["train_kenlm_job_with_silence"].output_bin

        for fairseq_hydra_config in expand_dictionary(fairseq_hydra_configs):
            gan_hydra_train_job = FairseqHydraTrainJob(
                fairseq_python_env=environment,
                fairseq_root=fairseq_root,
                fairseq_hydra_config=fairseq_hydra_config,
            )

            
            gan_hydra_train_jobs[gan_hydra_train_config_name] = {
                "config": gan_hydra_train_config,
                "train_job": gan_hydra_train_job,
            }
    ############################################################
    ## End of GAN training
    ############################################################

    ############################################################
    ## Decoding experiments setup
    ############################################################

    ############################################################
    ## VIterbi decoding and selecting models based on WER, Phoneme Error Rate, word level perplexity...
    ############################################################
    viterbi_config = {
        "fairseq": {
            "task": {
                "_name": "unpaired_audio_text",
                "labels": "phn",
                "data": "???",
                "sort_by_length": False,
                "shuffle": False,
                "text_data": ""
            },
            "common_eval": {
                "path": "???",
                "quiet": False
            },
            "dataset": {
                "gen_subset": "???",
                "batch_size": 1,
                "total_shards": 1,
                "shard_id": 0,
            },
            "common":{
                "user_dir": "???"
            },
        },
        "w2l_decoder": "VITERBI",
        "post_process": "silence",

        "results_path": "???",
    }

    gan_viterbi_decode_jobs = {}
    for decoding_dataset_name, decoding_dataset in decoding_datasets.items():
        gan_viterbi_decode_jobs[decoding_dataset_name] = {}
        for gan_hydra_train_config_name, gan_hydra_train_config in gan_hydra_train_jobs.items():
            viterbi_config = deepcopy(viterbi_config)
            viterbi_config["fairseq"]["task"]["data"] = audio_jobs[decoding_dataset_name]["featurize_job"].out_features_precompute_pca512_cls128_mean_pooled
            viterbi_config["fairseq"]["common_eval"]["path"] = gan_hydra_train_config["train_job"].out_best_model
            viterbi_config["fairseq"]["task"]["gen_subset"] = "preprocessed_audio"

            gan_viterbi_decode_job = GANw2vGenerateJob(
                fairseq_python_env=environment,
                fairseq_root=fairseq_root,
                decoding_config=viterbi_config,
            )

            gan_viterbi_decode_jobs[decoding_dataset_name][gan_hydra_train_config_name] = gan_viterbi_decode_job
    
    ##########################################################
    ## Decoding with Kaldi and Fairseq using the GAN model
    ###########################################################
    
    kaldi_configs = {
        "default_kaldi_gan_decode_config": {
            "w2l_decoder": "KALDI",
            "post_process": "silence",
            "blank_weight": 4,
            "beam": 20,
            "sil_is_blank": True,
            "blank_mode": "add",
            "unsupervised_tuning": 0,
            "decode_stride": 1,
            "fairseq": {
                "task": {
                    "_name": "unpaired_audio_text",
                    "data": "???",
                    "text_data": "",
                    "labels": "phn",
                    "sort_by_length": False,
                    "shuffle": False
                },
                "common_eval": {
                    "path": "???",
                    "quiet": True
                },
                "dataset": {
                    "gen_subset": "???",
                    "batch_size": 1,
                    "dataset_impl": "mmap",
                    "total_shards": 8, #TODO: assert this matches the number of shard ids
                    "shard_id": ExpandableIterable([0,1,2,3,4,5,6,7]),
                }
            },
            "targets": "wrd",
            "lexicon": "???",
            "viterbi_transcript": "???",
            "kaldi_decoder_config": {
                "hlg_graph_path": "???",
                "output_dict": "???",
                "acoustic_scale": 0.59
            },
            "no_softmax": True,
            "results_path": "???"
        }
    }

    gan_kaldi_decode_jobs = {}
    for decoding_dataset_name, decoding_dataset in decoding_datasets.items():
        gan_kaldi_decode_jobs[decoding_dataset_name] = {}
        for gan_hydra_train_config_name, gan_hydra_train_config in gan_hydra_train_jobs.items():
            gan_kaldi_decode_jobs[decoding_dataset_name][gan_hydra_train_config_name] = {}
            for kaldi_decode_config_name, kaldi_decode_config in kaldi_configs.items():
                kaldi_decode_config_aux = deepcopy(kaldi_decode_config)
                kaldi_decode_config_aux["fairseq"]["task"]["data"] = audio_jobs[decoding_dataset_name]["featurize_job"].out_features_precompute_pca512_cls128_mean_pooled
                kaldi_decode_config_aux["fairseq"]["common_eval"]["path"] = gan_hydra_train_config["train_job"].out_best_model
                kaldi_decode_config_aux["fairseq"]["task"]["gen_subset"] = "preprocessed_audio"

                for kaldi_decode_config_branch in expand_dictionary(kaldi_decode_config_aux):
                    gan_kaldi_decode_job = GANw2vGenerateJob(
                        fairseq_python_env=environment,
                        fairseq_root=fairseq_root,
                        decoding_config=kaldi_decode_config_branch,
                    )
                    tk.register_output("generate", gan_kaldi_decode_job.results_path)
                    gan_kaldi_decode_jobs[decoding_dataset_name][gan_hydra_train_config_name][kaldi_decode_config_name] = {
                        "config": kaldi_decode_config_branch,
                        "decode_job": gan_kaldi_decode_job,
                    }

    ###########################################################
    ## Decoding with RASR using the GAN model
    ###########################################################
    #TODO: to be imported/adapted from /u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/fairseq_2025_06_02-proj/recipe/i6_experiments/users/enrique/experiments/wav2vec_u/rasr_decoding_pipeline.py
    # possibly merged with the above Kaldi decoding section, they follow very similar code
    #
    ############################################################# End of Decoding experiments setup
    ###########################################################

    ###########################################################
    ## models evaluations
    ###########################################################
    ## We use an unsupervised metric proposed by Meta in https://ai.meta.com/research/publications/unsupervised-speech-recognition/

    phn_level_unsupervised_metric_language_model_job = all_levels_text_language_models_jobs["LibriSpeech-no-LibriVox_phn-4-gram"]["train_kenlm_job_no_silence"]
    bpe_level_unsupervised_metric_language_model_job = all_levels_text_language_models_jobs["LibriSpeech-no-LibriVox_bpe_50_6-gram"]["train_kenlm_job_no_silence"]
    unigram_level_unsupervised_metric_language_model_job = all_levels_text_language_models_jobs["LibriSpeech-no-LibriVox_unigram_50_4-gram"]["train_kenlm_job_no_silence"]


    for decoding_dataset_name, decoding_dataset in gan_viterbi_decode_jobs.items():
        for gan_hydra_train_config_name, gan_viterbi_decode_job in decoding_dataset.items():
            unsupervised_metric_job = UnsupervisedSpeechRecognitionMetricJob(
                fairseq_python_env=environment,
                fairseq_root=fairseq_root,
                decoded_phonemes=gan_viterbi_decode_job.out_decoded_phonemes,
                decoded_bpe=gan_viterbi_decode_job.out_decoded_bpe,
                decoded_unigram=gan_viterbi_decode_job.out_decoded_unigram,
                phn_level_lm=phn_level_unsupervised_metric_language_model_job.output_bin,
                bpe_level_lm=bpe_level_unsupervised_metric_language_model_job.output_bin,
                unigram_level_lm=unigram_level_unsupervised_metric_language_model_job.output_bin,
            )




    

    




def py():
    run_gan_experiments()