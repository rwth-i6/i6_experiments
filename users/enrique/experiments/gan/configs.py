###########################################################
# Imports
###########################################################

import sys
from copy import deepcopy
import os
from i6_core.tools.download import DownloadJob

import logging
from sisyphus import tk


##############################################################


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
        "name_of_anchor_dataset": "LibriSpeech-train-other-960", # PCA matrices and clusters will be extracted from this job
    },

    # "LibriSpeech-test-clean": {
    #     "path": tk.Path("/u/corpora/speech/LibriSpeech/LibriSpeech/test-clean"),
    #     "dataset": "LibriSpeech", 
    #     "extension": "flac",
    #     "subset": "test-clean",
    #     "w2v2": {
    #         "model": "large_60kh",
    #         "feature_extraction_layer": 14 
    #     },
    #     "name_of_anchor_dataset": "LibriSpeech-train-other-960", # PCA matrices and clusters will be extracted from this job
    # },
}


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
        "word_count_threshold": 2,
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
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
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
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
    "LibriSpeech-no-LibriVox_phn-6-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "phn",
        "order": 6,
        "phonemizer": "g2p_en",
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    }
}

bpe_level_text_language_models = {
    "LibriSpeech-no-LibriVox_bpe_50_6-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "bpe",
        "vocab_size": 50,
        "order": 6,
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
    "LibriSpeech-no-LibriVox_bpe_100_6-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "bpe",
        "vocab_size": 100,
        "order": 6,
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
    "LibriSpeech-no-LibriVox_bpe_300_6-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "bpe",
        "vocab_size": 300,
        "order": 6,
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
    "LibriSpeech-no-LibriVox_bpe_500_6-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "bpe",
        "vocab_size": 500,
        "order": 6,
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
    
}

unigram_level_text_language_models = {
    "LibriSpeech-no-LibriVox_unigram_50_4-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "uni",
        "vocab_size": 50,
        "order": 4,
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
    "LibriSpeech-no-LibriVox_unigram_100_4-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "uni",
        "vocab_size": 100,
        "order": 4,
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
    "LibriSpeech-no-LibriVox_unigram_300_4-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "uni",
        "vocab_size": 300,
        "order": 4,
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
    "LibriSpeech-no-LibriVox_unigram_500_4-gram":{
        "text_file": tk.Path("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"),
        "level": "uni",
        "vocab_size": 500,
        "order": 4,
        "lid_path": fasttext_model,
        "insert_silence_between_words_prob": 0.25,
        "silence_token": "<SIL>",
        "word_count_threshold": 2,
        "each_label_count_threshold": 1000,
    },
}

all_levels_text_language_models = {
    **phoneme_level_text_language_models,
    **letter_level_text_language_models,
    **bpe_level_text_language_models,
    **unigram_level_text_language_models,
    **word_level_text_language_models,
}


default_gan_hydra_train_configs = [
    (
        f"default_w2vu_gan_train_job_cp_{code_penalty}_gp_{gradient_penalty}_sw_{smoothness_weight}_seed_{random_seed}", {
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
                    "seed": random_seed,
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
                    "code_penalty": code_penalty,
                    "gradient_penalty": gradient_penalty,
                    "smoothness_weight": smoothness_weight,
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
    )
    for random_seed in range(0,5)
    for code_penalty in [2,4]
    for gradient_penalty in [1.5,2.0]
    for smoothness_weight in [0.5,0.75]
] 


all_levels_gan_hydra_train_configs = [(
        f"{text_data}_gan_training_config_seed_{random_seed}_cp_{code_penalty}_gp_{gradient_penalty}_sw_{smoothness_weight}_dislr_{discriminator_lr}_genlr_{generator_lr}_gendrop_{generator_dropout}_genk_{generator_kernel}", {
            "description": "variations in vocabulary and other parameters",
            "text_data_name": text_data,
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
                    "seed": random_seed,
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
                                generator_lr
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
                                discriminator_lr
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
                    "code_penalty": code_penalty,
                    "gradient_penalty": gradient_penalty,
                    "smoothness_weight": smoothness_weight,
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
                    "generator_kernel": generator_kernel,
                    "generator_bias": False,
                    "generator_dropout": generator_dropout,
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
    )
    for random_seed in [1]
    for code_penalty in [2]
    for gradient_penalty in [1.5]
    for smoothness_weight in [0.7]
    for discriminator_lr in [0.00023]
    for generator_lr in [0.00018]
    for generator_dropout in [0.1, 0.2,0.3]
    for generator_kernel in [4,5,7]
    for text_data in [
        #"LibriSpeech-no-LibriVox_phn-4-gram",
        #"LibriSpeech-no-LibriVox_chr-6-gram",
        #  "LibriSpeech-no-LibriVox_unigram_50_4-gram",
        #  "LibriSpeech-no-LibriVox_unigram_100_4-gram",
        #  "LibriSpeech-no-LibriVox_unigram_300_4-gram",
        #  "LibriSpeech-no-LibriVox_unigram_500_4-gram",
        "LibriSpeech-no-LibriVox_bpe_50_6-gram",
        "LibriSpeech-no-LibriVox_bpe_100_6-gram",
        "LibriSpeech-no-LibriVox_bpe_300_6-gram",
        "LibriSpeech-no-LibriVox_bpe_500_6-gram"        
    ]
]

#gan_hydra_train_configs = default_gan_hydra_train_configs + all_levels_gan_hydra_train_configs
#gan_hydra_train_configs = default_gan_hydra_train_configs 
gan_hydra_train_configs = all_levels_gan_hydra_train_configs
print("total gan training configs:", len(gan_hydra_train_configs))

viterbi_config_template = {
    "fairseq": {
        "task": {
            "_name": "unpaired_audio_text",
            "labels": "???",
            "data": "???",
            "sort_by_length": False,
            "shuffle": False,
            "text_data": "???"
        },
        "common_eval": {
            "path": "???",
            "quiet": False
        },
        "dataset": {
            "gen_subset": "???",
            "batch_size": 1,
            "num_shards": 1,
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



kaldi_configs = [
    (
        f"default_kaldi_gan_decode_config_shards_{total_shards}",
        {
            "w2l_decoder": "KALDI",
            "post_process": "silence",
            "blank_weight": blank_weight,
            "no_softmax": softm,
            "beam": beam,
            "sil_is_blank": True,
            "blank_mode": "add",
            "unsupervised_tuning": 0,
            "decode_stride": decode_stride,
            "fairseq": {
                "task": {
                    "_name": "unpaired_audio_text",
                    "data": "???",
                    "text_data": "???",
                    "labels": "???",
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
                    "num_shards": total_shards,
                    "shard_id": "???",
                },
                "common":{
                    "user_dir": "???"
                },
            },
            "targets": "wrd",
            "lexicon": "???",
            "viterbi_transcript": "???",
            "kaldi_decoder_config": {
                "hlg_graph_path": "???",
                "output_dict": "???",
                "acoustic_scale": acoustic_scale,
            },
            "no_softmax": True,
            "results_path": "???"
        }
    )
    for total_shards in [8]
    for blank_weight in [4]
    for beam in [20]
    for decode_stride in [1]
    #for acoustic_scale in [0.59, 1.2]
    for acoustic_scale in [0.59]
    for softm in [True]
]


phn_level_unsupervised_metric_language_model_job_name = "LibriSpeech-no-LibriVox_phn-4-gram"
letter_level_unsupervised_metric_language_model_job_name = "LibriSpeech-no-LibriVox_chr-6-gram"
bpe_level_unsupervised_metric_language_model_job_name = "LibriSpeech-no-LibriVox_bpe_50_6-gram"
unigram_level_unsupervised_metric_language_model_job_name = "LibriSpeech-no-LibriVox_unigram_50_4-gram"


how_to_decode_training_data = [ # (gan_hydra_train_config_name, decoding_dataset_name, kaldi_decode_config_name, lm_name)
    ("LibriSpeech-no-LibriVox_bpe_100_6-gram_gan_training_config_seed_1_cp_2_gp_1.5_sw_0.7_dislr_0.00023_genlr_0.00018_gendrop_0.1_genk_7", "LibriSpeech-train-other-960", "default_kaldi_gan_decode_config_shards_8", "LibriSpeech-no-LibriVox_wrd-4-gram-0-0-1-4"),
]