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

from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob

from recipe.i6_experiments.users.enrique.jobs.gan.prepare_text import (
    FairseqG2PWordToPhnJob,
    CreateLexiconAndDictionaryJob,
    TokenizeWithLexiconAndSilenceJob,
    FairseqNormalizeTextAndCreateDictionary,
    FairseqPreprocessJob,
    WordToLetterJob,
    TrainKenLMJob,
    CreateKaldiFSTJob,
)
from recipe.i6_experiments.users.enrique.jobs.gan.process_audio import process_audio
from recipe.i6_experiments.users.enrique.jobs.gan.hydra_train import FairseqHydraTrainJob
from recipe.i6_experiments.users.enrique.jobs.gan.utils import get_fairseq_root, KENLM_BINARY_PATH, get_rvad_root, TakeNRandomLinesFromTextFileJob, GetW2VLibriSpeechGroundTruthJob, SCTK_BINARY_PATH, MergeGenerateShardsJob, FormatToCtmJob, FormatTranscriptionJob, DivideLibriSpeech960hInto100h360h500hJob
from recipe.i6_experiments.users.enrique.jobs.gan.decoding import GANw2vGenerateJob #, UnsupervisedVocabUsageAndLMScoreMetric a
from recipe.i6_experiments.users.enrique.jobs.gan.scoring import ScoreTextWithLMJob

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
    fairseq_root = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/EXTERNAL_SOFTWARE/fairseq_w2vu/fairseq"
    fairseq_root = tk.Path(fairseq_root)
    fairseq_root = get_fairseq_root( # installs fairseq in environment ONLY WHEN JOBS ARE NOT ALREADY RUNNED
        python_env=environment,
        fairseq_root=fairseq_root,
    )
    rvad_root = get_rvad_root()
    
    ########################################################### 
    # THe actual experiment configurations are defined in configs.py, /u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/fairseq_2025_06_02-proj/recipe/i6_experiments/users/enrique/experiments/gan/configs.py
    from recipe.i6_experiments.users.enrique.experiments.gan.configs import(
        training_datasets,
        decoding_datasets,
        word_level_text_language_models,
        letter_level_text_language_models,
        phoneme_level_text_language_models,
        bpe_level_text_language_models,
        unigram_level_text_language_models,
        gan_hydra_train_configs,
        kaldi_configs,
        viterbi_config_template,
        phn_level_unsupervised_metric_language_model_job_name,
        letter_level_unsupervised_metric_language_model_job_name,
        bpe_level_unsupervised_metric_language_model_job_name,
        unigram_level_unsupervised_metric_language_model_job_name,
        how_to_decode_training_data,
    )
    decode_training_data = False


    ###########################################################
    ## Audio Processing
    ###########################################################
    

    all_audio_datasets = training_datasets | decoding_datasets

    audio_jobs = {}
    for name, dataset in list(training_datasets.items()) + list(decoding_datasets.items()):
        w2v2model = dataset["w2v2"]["model"]
        feature_extraction_layer = dataset["w2v2"]["feature_extraction_layer"]
        assert not (
            (w2v2model == "base" and feature_extraction_layer > 11) or feature_extraction_layer > 23
        ), "not that many layers in the model"
        
        if "name_of_anchor_dataset" in dataset:
            assert w2v2model == all_audio_datasets[dataset["name_of_anchor_dataset"]]["w2v2"]["model"], "W2V2 model must be the same for training and decoding datasets that are linked."
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
            "config": dataset,
            "delete_silence_job": delete_silences_job,
            "featurize_job": featurize_training_audio_job,
            "manifest": process_training_audio_manifest,
        }
   
    ############################################################
    ## Text data preparation, lexicon, LMs, etc.
    ############################################################


    word_level_text_language_models_jobs = {}
    for lm_name, lm_info in word_level_text_language_models.items():
        normalize_text_job = FairseqNormalizeTextAndCreateDictionary(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            text_file_path=lm_info["text_file"],
            language=lm_info.get("language", "en"),
            lid_path=lm_info.get("lid_path"),
            thresholdsrc=lm_info.get("word_count_threshold", 0),
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
            thresholdsrc=lm_info.get("word_count_threshold", 0),
        )     

        word_to_letter_job = WordToLetterJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            words_file=normalize_text_job.words_txt,
        )

        create_lexicon_job = CreateLexiconAndDictionaryJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            words_txt=normalize_text_job.words_txt,
            tokenized_words=word_to_letter_job.letters_file,
            label_type=lm_info["level"],
            each_label_count_threshold=lm_info.get("each_label_count_threshold", 0),
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
            "word_to_letter_job": word_to_letter_job,
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
                thresholdsrc=lm_info.get("word_count_threshold", 0),
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
                label_type=lm_info["level"],
                each_label_count_threshold=lm_info.get("each_label_count_threshold", 0),
            )

            label_text_with_lexicon_and_silence_job = TokenizeWithLexiconAndSilenceJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=normalize_text_job.normalize_text_lid,
                lexicon_lst=create_lexicon_job.lexicon_lst,
                insert_silence_between_words_prob=lm_info.get("insert_silence_between_words_prob"),
                silence_token=lm_info.get("silence_token"),
                surround_with_silence=True, #default
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
                thresholdsrc=lm_info.get("word_count_threshold", 0),
            )   

            sentencepiece_text_file_path = TakeNRandomLinesFromTextFileJob(
                input_text_file=normalize_text_job.normalize_text_lid, n_lines=lm_info.get("sentencepiece_training_text_n_lines", 1000000)
            ).output_text_file

            train_sentence_piece_job = TrainSentencePieceJob(
                training_text=sentencepiece_text_file_path,
                vocab_size=lm_info["vocab_size"],
                model_type=SentencePieceType.BPE,
                additional_options={"max_sentence_length": 1000000, "shuffle_input_sentence": False}
            )

            apply_sentencepiece_job = ApplySentencepieceToTextJob(text_file=normalize_text_job.words_txt, sentencepiece_model=train_sentence_piece_job.out_model, gzip_output=False)

            create_lexicon_job = CreateLexiconAndDictionaryJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                words_txt=normalize_text_job.words_txt,
                tokenized_words=apply_sentencepiece_job.out_sentencepiece_text,
                label_type=lm_info["level"],
                each_label_count_threshold=lm_info.get("each_label_count_threshold", 0),
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
                thresholdsrc=lm_info.get("word_count_threshold", 0),
            )   

            sentencepiece_text_file_path = TakeNRandomLinesFromTextFileJob(
                input_text_file=normalize_text_job.normalize_text_lid, n_lines=lm_info.get("sentencepiece_training_text_n_lines", 1000000)
            ).output_text_file
                    
            train_sentence_piece_job = TrainSentencePieceJob(
                training_text=sentencepiece_text_file_path,
                vocab_size=lm_info["vocab_size"],
                model_type=SentencePieceType.UNIGRAM,
                additional_options={"max_sentence_length": 1000000, "shuffle_input_sentence": False}
            )

            apply_sentencepiece_job = ApplySentencepieceToTextJob(text_file=normalize_text_job.words_txt, sentencepiece_model=train_sentence_piece_job.out_model, gzip_output=False)

            create_lexicon_job = CreateLexiconAndDictionaryJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                words_txt=normalize_text_job.words_txt,
                tokenized_words=apply_sentencepiece_job.out_sentencepiece_text,
                label_type=lm_info["level"],
                each_label_count_threshold=lm_info.get("each_label_count_threshold", 0),
            )

            label_text_with_lexicon_and_silence_job = TokenizeWithLexiconAndSilenceJob(
                fairseq_root=fairseq_root,
                fairseq_python_env=environment,
                text_file_path=normalize_text_job.normalize_text_lid,
                lexicon_lst=create_lexicon_job.lexicon_lst,
                insert_silence_between_words_prob=lm_info.get("insert_silence_between_words_prob"),
                silence_token=lm_info.get("silence_token"),
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

    
    gan_hydra_train_jobs = {}
    for gan_hydra_train_config_name, gan_hydra_train_config in gan_hydra_train_configs:
        fairseq_hydra_config = deepcopy(gan_hydra_train_config["fairseq_hydra_config"])
        fairseq_hydra_config["task"]["data"] = audio_jobs[gan_hydra_train_config["audio_data_name"]]["featurize_job"].out_features_precompute_pca512_cls128_mean_pooled
        fairseq_hydra_config["task"]["text_data"] = all_levels_text_language_models_jobs[gan_hydra_train_config["text_data_name"]]["fairseq_preprocess_job"].labels_folder
        fairseq_hydra_config["task"]["labels"] = all_levels_text_language_models_jobs[gan_hydra_train_config["text_data_name"]]["config"]["level"]
        fairseq_hydra_config["task"]["kenlm_path"] = all_levels_text_language_models_jobs[gan_hydra_train_config["text_data_name"]]["train_kenlm_job_with_silence"].output_bin

        gan_hydra_train_job = FairseqHydraTrainJob(
            fairseq_python_env=environment,
            fairseq_root=fairseq_root,
            fairseq_hydra_config=fairseq_hydra_config,
        )
        gan_hydra_train_jobs[gan_hydra_train_config_name] = {
            "config": gan_hydra_train_config,
            "fairseq_hydra_config": fairseq_hydra_config,
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
    gan_viterbi_decode_jobs = {}
    for gan_hydra_train_config_name, gan_hydra_train_config in gan_hydra_train_jobs.items():
        gan_viterbi_decode_jobs[gan_hydra_train_config_name] = {}
        for decoding_dataset_name, decoding_dataset in decoding_datasets.items():
            viterbi_config = deepcopy(viterbi_config_template)
            viterbi_config["fairseq"]["task"]["data"] = audio_jobs[decoding_dataset_name]["featurize_job"].out_features_precompute_pca512_cls128_mean_pooled
            viterbi_config["fairseq"]["common_eval"]["path"] = gan_hydra_train_config["train_job"].out_best_model
            viterbi_config["fairseq"]["dataset"]["gen_subset"] = "preprocessed_audio"
            viterbi_config["fairseq"]["task"]["labels"] = gan_hydra_train_config["fairseq_hydra_config"]["task"]["labels"]
            viterbi_config["fairseq"]["task"]["text_data"] = gan_hydra_train_config["fairseq_hydra_config"]["task"]["text_data"]

            assert viterbi_config.get("fairseq").get("dataset").get("num_shards", 1) <= 1, "Viterbi decoding does not support sharded datasets currently."

            gan_viterbi_decode_job = GANw2vGenerateJob(
                fairseq_python_env=environment,
                fairseq_root=fairseq_root,
                decoding_config=viterbi_config,
            )
            gan_viterbi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name] = gan_viterbi_decode_job
    

    ##########################################################
    ## Decoding with Kaldi and Fairseq using the GAN model
    ###########################################################

    gan_kaldi_decode_jobs = {}
    for gan_hydra_train_config_name, gan_hydra_train_config in gan_hydra_train_jobs.items():
        gan_kaldi_decode_jobs[gan_hydra_train_config_name] = {}
        for decoding_dataset_name, decoding_dataset in decoding_datasets.items():
            gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name] = {}
            for kaldi_decode_config_name, kaldi_decode_config in kaldi_configs:
                gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name] = {}
                for lm_name, lm_info in word_level_text_language_models_jobs.items():
                    gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name] = {}
                    kaldi_decode_config_aux = deepcopy(kaldi_decode_config)
                    kaldi_decode_config_aux["fairseq"]["task"]["data"] = audio_jobs[decoding_dataset_name]["featurize_job"].out_features_precompute_pca512_cls128_mean_pooled
                    kaldi_decode_config_aux["fairseq"]["common_eval"]["path"] = gan_hydra_train_config["train_job"].out_best_model
                    kaldi_decode_config_aux["fairseq"]["dataset"]["gen_subset"] = "preprocessed_audio"
                    kaldi_decode_config_aux["fairseq"]["task"]["labels"] = gan_hydra_train_config["fairseq_hydra_config"]["task"]["labels"]
                    kaldi_decode_config_aux["lexicon"] = all_levels_text_language_models_jobs[gan_hydra_train_config["config"]["text_data_name"]]["create_lexicon_job"].lexicon_filtered_lst
                    kaldi_decode_config_aux["fairseq"]["task"]["text_data"] = all_levels_text_language_models_jobs[gan_hydra_train_config["config"]["text_data_name"]]["fairseq_preprocess_job"].labels_folder
                    kaldi_decode_config_aux["viterbi_transcript"] = gan_viterbi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name].output_trans

                    create_kaldi_fst_job = CreateKaldiFSTJob(
                        fairseq_root=fairseq_root,
                        fairseq_python_env=environment,
                        lm_arpa=lm_info["train_kenlm_job"].output_arpa,
                        lexicon_lst=all_levels_text_language_models_jobs[gan_hydra_train_config["config"]["text_data_name"]]["create_lexicon_job"].lexicon_filtered_lst,
                        data_dir=all_levels_text_language_models_jobs[gan_hydra_train_config["config"]["text_data_name"]]["create_lexicon_job"].dict_folder,
                        label_type=gan_hydra_train_config["fairseq_hydra_config"]["task"]["labels"],
                    )
                    kaldi_decode_config_aux["kaldi_decoder_config"]["hlg_graph_path"] = create_kaldi_fst_job.out_hlg_graph
                    kaldi_decode_config_aux["kaldi_decoder_config"]["output_dict"] = create_kaldi_fst_job.out_kaldi_dict


                    if kaldi_decode_config_aux.get("fairseq").get("dataset").get("num_shards", 1) > 1:
                        num_shards = kaldi_decode_config_aux["fairseq"]["dataset"]["num_shards"]
                        gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name] = []
                        for shard_id in range(num_shards):
                            kaldi_decode_config_aux_shard = deepcopy(kaldi_decode_config_aux)
                            kaldi_decode_config_aux_shard["fairseq"]["dataset"]["shard_id"] = shard_id
                            

                            gan_kaldi_decode_job = GANw2vGenerateJob(
                                fairseq_python_env=environment,
                                fairseq_root=fairseq_root,
                                decoding_config=kaldi_decode_config_aux_shard,
                            )

                            
                            gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name].append({
                                "shard_id": shard_id,
                                "config": kaldi_decode_config_aux_shard,
                                "decode_job": gan_kaldi_decode_job,
                            })


                    else:
                        gan_kaldi_decode_job = GANw2vGenerateJob(
                            fairseq_python_env=environment,
                            fairseq_root=fairseq_root,
                            decoding_config=kaldi_decode_config_aux,
                        )
                        

                        gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name] = {
                            "config": kaldi_decode_config_aux,
                            "decode_job": gan_kaldi_decode_job,
                        }

                        
                    
    ###########################################################
    ## Decoding with RASR using the GAN model
    ###########################################################
    #TODO: to be imported/adapted from /u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/fairseq_2025_06_02-proj/recipe/i6_experiments/users/enrique/experiments/wav2vec_u/rasr_decoding_pipeline.py
    # possibly merged with the above Kaldi decoding section, they follow very similar code

    ############################################################# 
    #############################################################  End of Decoding experiments setup


    ###########################################################
    ## models evaluations
    ###########################################################
    
    ## Get LibriSpeech ground truth transcriptions
    librispeech_ground_truth_transcriptions = {}
    for decoding_dataset_name, decoding_dataset in decoding_datasets.items():
        lb_subset = ""
        if "test-other" in decoding_dataset_name:
            lb_subset = "test-other"
        elif "test-clean" in decoding_dataset_name:
            lb_subset = "test-clean"
        elif "dev-other" in decoding_dataset_name:
            lb_subset = "dev-other"
        elif "dev-clean" in decoding_dataset_name:
            lb_subset = "dev-clean"
        elif "train-clean-100" in decoding_dataset_name:
            lb_subset = "train-clean-100"
        elif "train-clean-360" in decoding_dataset_name:
            lb_subset = "train-clean-360"
        elif "train-other-500" in decoding_dataset_name:
            lb_subset = "train-other-500"
        elif "train-other-960" in decoding_dataset_name:
            lb_subset = "train-other-960"
        else:
            logging.warning(f"Decoding dataset {decoding_dataset_name} is not a recognized LibriSpeech subset, skipping ground truth extraction.")
            continue

        gt_job = GetW2VLibriSpeechGroundTruthJob(
            librispeech_subset=lb_subset,
            audio_manifest=audio_jobs[decoding_dataset_name]["manifest"],
        )

        ground_truth = gt_job.ground_truth_stm


        for gan_hydra_train_config_name, gan_hydra_train_config in gan_hydra_train_jobs.items():
            for kaldi_decode_config_name, kaldi_decode_config in kaldi_configs:
                for lm_name, lm_info in word_level_text_language_models_jobs.items():
                
                    
                    if type(gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name]) == list:
                        shard_transcriptions = []
                        for shard in gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name]:
                            shard_transcriptions.append(shard["decode_job"].output_trans)
                        #merge_job = MergeGenerateShardsJob(input_text_files=shard_transcriptions, manifest_path=audio_jobs[decoding_dataset_name]["manifest"], subset_name=lb_subset)
                        merge_job = MergeGenerateShardsJob(input_text_files=shard_transcriptions, subset_name=lb_subset)
                        hyp = FormatToCtmJob(manifest_path=audio_jobs[decoding_dataset_name]["manifest"], text_file=merge_job.output_text_file, subset_name=lb_subset).transcription_generated_ctm
                    else:
                        trans_file = gan_kaldi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name]["decode_job"].output_trans
                        hyp = FormatToCtmJob(manifest_path=audio_jobs[decoding_dataset_name]["manifest"], text_file=trans_file, subset_name=lb_subset).transcription_generated_ctm

    
                    sclite_job = ScliteJob(ref=ground_truth, hyp=hyp, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
                    #print(decoding_dataset_name, gan_hydra_train_config_name, kaldi_decode_config_name, lm_name)
                    print("WER: ", sclite_job.out_wer.get())
                    tk.register_output("sclite", sclite_job.out_wer)


    ## We use an unsupervised metric proposed by Meta in https://ai.meta.com/research/publications/unsupervised-speech-recognition/

    phn_level_unsupervised_metric_language_model_job = all_levels_text_language_models_jobs[phn_level_unsupervised_metric_language_model_job_name]["train_kenlm_job_no_silence"].output_bin
    letter_level_unsupervised_metric_language_model_job = all_levels_text_language_models_jobs[letter_level_unsupervised_metric_language_model_job_name]["train_kenlm_job_no_silence"].output_bin
    bpe_level_unsupervised_metric_language_model_job = all_levels_text_language_models_jobs[bpe_level_unsupervised_metric_language_model_job_name]["train_kenlm_job_no_silence"].output_bin
    unigram_level_unsupervised_metric_language_model_job = all_levels_text_language_models_jobs[unigram_level_unsupervised_metric_language_model_job_name]["train_kenlm_job_no_silence"].output_bin


    
    for gan_hydra_train_config_name, gan_hydra_train_config in gan_viterbi_decode_jobs.items():
        for decoding_dataset_name, decoding_dataset in gan_viterbi_decode_jobs[gan_hydra_train_config_name].items():
            viterbi_transcription = gan_viterbi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name].output_trans
            
            metric_lm = None
            level = gan_viterbi_decode_jobs[gan_hydra_train_config_name][decoding_dataset_name].decoding_config["fairseq"]["task"]["labels"]
            if level == "phn":
                metric_lm = phn_level_unsupervised_metric_language_model_job
            elif level == "chr":
                metric_lm = letter_level_unsupervised_metric_language_model_job
            elif level == "bpe":
                metric_lm = bpe_level_unsupervised_metric_language_model_job
            elif level == "uni":
                metric_lm = unigram_level_unsupervised_metric_language_model_job
            
            assert metric_lm is not None, f"Unsupported level for unsupervised metric scoring: {level}"

            # we delete the SIL tokens from the transcription before scoring
            lm_score = ScoreTextWithLMJob(text=viterbi_transcription, kenlm_model_path=metric_lm, symbols_to_remove=["<SIL>"]).total_score

            
    gan_kaldi_decode_training_dataset_jobs = {}
    for how_to_decode in how_to_decode_training_data:
        gan_hydra_train_config_name, decoding_dataset_name, kaldi_decode_config_name, lm_name = how_to_decode


        if gan_hydra_train_config_name not in gan_kaldi_decode_training_dataset_jobs:
            gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name] = {}
        if decoding_dataset_name not in gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name]:
            gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name][decoding_dataset_name] = {}
        if kaldi_decode_config_name not in gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name][decoding_dataset_name]:
            gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name] = {}
        if lm_name not in gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name]:
            gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name] = None    
        
        gan_hydra_train_config = gan_hydra_train_jobs[gan_hydra_train_config_name]
        
        kaldi_decode_config_aux = deepcopy(kaldi_decode_config)
        kaldi_decode_config_aux["fairseq"]["task"]["data"] = audio_jobs[gan_hydra_train_config["config"]["audio_data_name"]]["featurize_job"].out_features_precompute_pca512_cls128_mean_pooled
        kaldi_decode_config_aux["fairseq"]["common_eval"]["path"] = gan_hydra_train_config["train_job"].out_best_model
        kaldi_decode_config_aux["fairseq"]["dataset"]["gen_subset"] = "train"
        kaldi_decode_config_aux["fairseq"]["task"]["labels"] = gan_hydra_train_config["fairseq_hydra_config"]["task"]["labels"]
        kaldi_decode_config_aux["lexicon"] = all_levels_text_language_models_jobs[gan_hydra_train_config["config"]["text_data_name"]]["create_lexicon_job"].lexicon_filtered_lst
        kaldi_decode_config_aux["fairseq"]["task"]["text_data"] = all_levels_text_language_models_jobs[gan_hydra_train_config["config"]["text_data_name"]]["fairseq_preprocess_job"].labels_folder
        kaldi_decode_config_aux["viterbi_transcript"] = ""

        create_kaldi_fst_job = CreateKaldiFSTJob(
            fairseq_root=fairseq_root,
            fairseq_python_env=environment,
            lm_arpa=lm_info["train_kenlm_job"].output_arpa,
            lexicon_lst=all_levels_text_language_models_jobs[gan_hydra_train_config["config"]["text_data_name"]]["create_lexicon_job"].lexicon_filtered_lst,
            data_dir=all_levels_text_language_models_jobs[gan_hydra_train_config["config"]["text_data_name"]]["create_lexicon_job"].dict_folder,
            label_type=gan_hydra_train_config["fairseq_hydra_config"]["task"]["labels"],
        )
        
        kaldi_decode_config_aux["kaldi_decoder_config"]["hlg_graph_path"] = create_kaldi_fst_job.out_hlg_graph
        kaldi_decode_config_aux["kaldi_decoder_config"]["output_dict"] = create_kaldi_fst_job.out_kaldi_dict

        if kaldi_decode_config_aux.get("fairseq").get("dataset").get("num_shards", 1) > 1:
            num_shards = kaldi_decode_config_aux["fairseq"]["dataset"]["num_shards"]
            gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name] = []
            transcriptions_files_train = []
            transcriptions_files_valid = []
            for shard_id in range(num_shards):
                kaldi_decode_config_aux_shard = deepcopy(kaldi_decode_config_aux)
                kaldi_decode_config_aux_shard["fairseq"]["dataset"]["shard_id"] = shard_id
                
                kaldi_decode_config_aux_shard["fairseq"]["dataset"]["gen_subset"] = "train"
                gan_kaldi_decode_trainsb_job = GANw2vGenerateJob(
                    fairseq_python_env=environment,
                    fairseq_root=fairseq_root,
                    decoding_config=kaldi_decode_config_aux_shard,
                )
                transcriptions_files_train.append(gan_kaldi_decode_trainsb_job.output_trans)

                kaldi_decode_config_aux_shard["fairseq"]["dataset"]["gen_subset"] = "valid"
                gan_kaldi_decode_validsb_job = GANw2vGenerateJob(
                    fairseq_python_env=environment,
                    fairseq_root=fairseq_root,
                    decoding_config=kaldi_decode_config_aux_shard,
                )
                
                transcriptions_files_valid.append(gan_kaldi_decode_validsb_job.output_trans)

            
            merge_train_job = MergeGenerateShardsJob(input_text_files=transcriptions_files_train, subset_name="train")
            format_transcription_train_job = FormatTranscriptionJob(task_data=kaldi_decode_config_aux_shard["fairseq"]["task"]["data"], transcription_generated=merge_train_job.output_text_file, gen_subset="train")
            tk.register_output("train_shard_decode_job", format_transcription_train_job.out_label_path_dict)
            merge_valid_job = MergeGenerateShardsJob(input_text_files=transcriptions_files_valid, subset_name="valid")
            format_transcription_valid_job = FormatTranscriptionJob(task_data=kaldi_decode_config_aux_shard["fairseq"]["task"]["data"], transcription_generated=merge_valid_job.output_text_file, gen_subset="valid")
            tk.register_output("train_shard_decode_job", format_transcription_valid_job.out_label_path_dict)

            tk.register_output("train_decode_job", merge_train_job.output_text_file)
            tk.register_output("valid_decode_job", merge_valid_job.output_text_file)

            gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name] = {
                "config": kaldi_decode_config_aux,
                "decode_job_sb": {"train": format_transcription_train_job, "valid": format_transcription_valid_job},
            }

        else:
            gan_kaldi_decode_trainsb_job = GANw2vGenerateJob(
                fairseq_python_env=environment,
                fairseq_root=fairseq_root,
                decoding_config=kaldi_decode_config_aux,
            )
            tk.register_output("train_decode_job", gan_kaldi_decode_trainsb_job.output_trans)

            kaldi_decode_config_aux["fairseq"]["dataset"]["gen_subset"] = "valid"
            gan_kaldi_decode_validsb_job = GANw2vGenerateJob(
                fairseq_python_env=environment,
                fairseq_root=fairseq_root,
                decoding_config=kaldi_decode_config_aux,
            )
            format_transcription_train_job = FormatTranscriptionJob(task_data=kaldi_decode_config_aux_shard["fairseq"]["task"]["data"], transcription_generated=gan_kaldi_decode_trainsb_job.output_trans, gen_subset="train")
            tk.register_output("train_shard_decode_job", format_transcription_train_job.out_label_path_dict)
            merge_valid_job = MergeGenerateShardsJob(input_text_files=transcriptions_files_valid, subset_name="valid")
            format_transcription_valid_job = FormatTranscriptionJob(task_data=kaldi_decode_config_aux_shard["fairseq"]["task"]["data"], transcription_generated=gan_kaldi_decode_validsb_job.output_trans, gen_subset="valid")
            tk.register_output("train_shard_decode_job", format_transcription_valid_job.out_label_path_dict)

            tk.register_output("valid_decode_job", gan_kaldi_decode_validsb_job.output_trans)

            gan_kaldi_decode_training_dataset_jobs[gan_hydra_train_config_name][decoding_dataset_name][kaldi_decode_config_name][lm_name] = {
                "config": kaldi_decode_config_aux,
                "decode_job_sb": {"train": format_transcription_train_job, "valid": format_transcription_valid_job},
            }

        if decoding_dataset_name == "LibriSpeech-train-other-960":
            divide_100h_360h_500h_job = DivideLibriSpeech960hInto100h360h500hJob(
                ds_train_other_960_gzs=[format_transcription_train_job.transcription_generated_formatted,
                                        format_transcription_valid_job.transcription_generated_formatted]
            )
            tk.register_output("divide_100h_360h_500h_job", divide_100h_360h_500h_job.out_label_path_dict)

        
        


                        







    

    




def py():
    run_gan_experiments()