
# /work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/

from sisyphus import gs, tk, Path


import os
import sys
from typing import Optional, Union, Dict

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr
import i6_core.text as text

from i6_core.tools import CloneGitRepositoryJob

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.hybrid_system as hybrid_system
import i6_experiments.common.setups.rasr.util as rasr_util
import i6_experiments.common.datasets.librispeech as lbs_dataset

import copy
import numpy as np

import i6_core.returnn as returnn

from i6_experiments.users.luescher.helpers.search_params import get_search_parameters

# TODO remove these
import i6_experiments.users.luescher.setups.librispeech.pipeline_base_args as lbs_gmm_setups


def run():
  filename_handle = os.path.splitext(os.path.basename(__file__))[0]
  gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
  rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

  nn_args = get_nn_args()

  nn_steps = rasr_util.RasrSteps()
  nn_steps.add_step("nn", nn_args)

  hybrid_init_args = default_gmm_hybrid_init_args()

  train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(use_eval_data_subset=True)

  lbs_gmm_system = gmm_system.GmmSystem()
  lbs_gmm_system.init_system(
    hybrid_init_args=hybrid_init_args,
    train_data=train_data_inputs,
    dev_data=dev_data_inputs,
    test_data=test_data_inputs,
  )

  nn_train_data = hybrid_system.ReturnnRasrDataInput(
    name=name,
    crp=copy.deepcopy(self.crp),
    alignments=self.alignments,
    feature_flow=self.feature_flows[feature_flow_key],
    features=self.features[feature_flow_key],
    acoustic_mixtures=self.acoustic_mixtures,
    feature_scorers=self.feature_scorers,
    shuffle_data=shuffle_data,
  )

  nn_train_data = lbs_gmm_system.outputs["train-other-960"][
    "final"
  ].as_returnn_rasr_data_input(shuffle_data=True)
  nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
  nn_train_data_inputs = {
    "train-other-960.train": nn_train_data,
  }

  nn_cv_data = lbs_gmm_system.outputs["train-other-960"][
    "final"
  ].as_returnn_rasr_data_input()
  nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
  nn_cv_data_inputs = {
    "train-other-960.cv": nn_cv_data,
  }

  nn_devtrain_data = lbs_gmm_system.outputs["train-other-960"][
    "final"
  ].as_returnn_rasr_data_input()
  nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
  nn_devtrain_data_inputs = {
    "train-other-960.devtrain": nn_devtrain_data,
  }
  nn_dev_data_inputs = {
    # "dev-clean": lbs_gmm_system.outputs["dev-clean"][
    #    "final"
    # ].as_returnn_rasr_data_input(),
    "dev-other": lbs_gmm_system.outputs["dev-other"][
      "final"
    ].as_returnn_rasr_data_input(),
  }
  nn_test_data_inputs = {
    # "test-clean": lbs_gmm_system.outputs["test-clean"][
    #    "final"
    # ].as_returnn_rasr_data_input(),
    # "test-other": lbs_gmm_system.outputs["test-other"][
    #    "final"
    # ].as_returnn_rasr_data_input(),
  }

  nn_args = get_nn_args()

  nn_steps = rasr_util.RasrSteps()
  nn_steps.add_step("nn", nn_args)

  # TODO ...
  lbs_nn_system = hybrid_system.HybridSystem()
  lbs_nn_system.init_system(
    hybrid_init_args=hybrid_init_args,
    train_data=nn_train_data_inputs,
    cv_data=nn_cv_data_inputs,
    devtrain_data=nn_devtrain_data_inputs,
    dev_data=nn_dev_data_inputs,
    test_data=nn_test_data_inputs,
    train_cv_pairing=[tuple(["train-other-960.train", "train-other-960.cv"])],
  )
  lbs_nn_system.run(nn_steps)

  gs.ALIAS_AND_OUTPUT_SUBDIR = ""


def default_gmm_hybrid_init_args():
  # hybrid_init_args = lbs_gmm_setups.get_init_args()
  dc_detection: bool = True
  scorer: Optional[str] = None
  mfcc_filter_width: Union[float, Dict] = 268.258

  am_args = {
    "state_tying": "monophone",
    "states_per_phone": 3,
    "state_repetitions": 1,
    "across_word_model": True,
    "early_recombination": False,
    "tdp_scale": 1.0,
    "tdp_transition": (3.0, 0.0, 30.0, 0.0),  # loop, forward, skip, exit
    "tdp_silence": (0.0, 3.0, "infinity", 20.0),
    "tying_type": "global",
    "nonword_phones": "",
    "tdp_nonword": (
      0.0,
      3.0,
      "infinity",
      6.0,
    ),  # only used when tying_type = global-and-nonword
  }

  costa_args = {"eval_recordings": True, "eval_lm": False}
  default_mixture_scorer_args = {"scale": 0.3}

  mfcc_cepstrum_options = {
    "normalize": False,
    "outputs": 16,
    "add_epsilon": False,
  }

  feature_extraction_args = {
    "mfcc": {
      "num_deriv": 2,
      "num_features": None,  # 33 (confusing name: # max features, above -> clipped)
      "mfcc_options": {
        "warping_function": "mel",
        "filter_width": mfcc_filter_width,
        "normalize": True,
        "normalization_options": None,
        "without_samples": False,
        "samples_options": {
          "audio_format": "wav",
          "dc_detection": dc_detection,
        },
        "cepstrum_options": mfcc_cepstrum_options,
        "fft_options": None,
      },
    },
    "gt": {
      "gt_options": {
        "minfreq": 100,
        "maxfreq": 7500,
        "channels": 50,
        # "warp_freqbreak": 7400,
        "tempint_type": "hanning",
        "tempint_shift": 0.01,
        "tempint_length": 0.025,
        "flush_before_gap": True,
        "do_specint": False,
        "specint_type": "hanning",
        "specint_shift": 4,
        "specint_length": 9,
        "normalize": True,
        "preemphasis": True,
        "legacy_scaling": False,
        "without_samples": False,
        "samples_options": {
          "audio_format": "wav",
          "dc_detection": dc_detection,
        },
        "normalization_options": {},
      }
    },
    "energy": {
      "energy_options": {
        "without_samples": False,
        "samples_options": {
          "audio_format": "wav",
          "dc_detection": dc_detection,
        },
        "fft_options": {},
      }
    },
  }

  return rasr_util.RasrInitArgs(
    costa_args=costa_args,
    am_args=am_args,
    feature_extraction_args=feature_extraction_args,
    default_mixture_scorer_args=default_mixture_scorer_args,
    scorer=scorer,
  )


def get_data_inputs(
      train_corpus="train-other-960",
      add_unknown_phoneme_and_mapping=True,
      use_eval_data_subset: bool = False,
  ):
    corpus_object_dict = lbs_dataset.get_corpus_object_dict(
      audio_format="wav",
      output_prefix="corpora",
    )

    lm = {
      "filename": lbs_dataset.get_arpa_lm_dict()["4gram"],
      "type": "ARPA",
      "scale": 10,
    }

    use_stress_marker = False

    original_bliss_lexicon = {
      "filename": lbs_dataset.get_bliss_lexicon(
        use_stress_marker=use_stress_marker,
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
      ),
      "normalize_pronunciation": False,
    }

    augmented_bliss_lexicon = {
      "filename": lbs_dataset.get_g2p_augmented_bliss_lexicon_dict(
        use_stress_marker=use_stress_marker,
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
      )[train_corpus],
      "normalize_pronunciation": False,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[train_corpus] = rasr_util.RasrDataInput(
      corpus_object=corpus_object_dict[train_corpus],
      concurrent=300,
      lexicon=augmented_bliss_lexicon,
    )

    dev_corpus_keys = (
      ["dev-other"] if use_eval_data_subset else ["dev-clean", "dev-other"]
    )
    test_corpus_keys = [] if use_eval_data_subset else ["test-clean", "test-other"]

    for dev_key in dev_corpus_keys:
      dev_data_inputs[dev_key] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict[dev_key],
        concurrent=20,
        lexicon=original_bliss_lexicon,
        lm=lm,
      )

    for tst_key in test_corpus_keys:
      test_data_inputs[tst_key] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict[tst_key],
        concurrent=20,
        lexicon=original_bliss_lexicon,
        lm=lm,
      )

    return train_data_inputs, dev_data_inputs, test_data_inputs


def orig_chris_run():
    # ******************** Settings ********************

    filename_handle = os.path.splitext(os.path.basename(__file__))[0]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** GMM Init ********************

    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_gmm_setups.get_data_inputs(
        use_eval_data_subset=True,
    )
    hybrid_init_args = lbs_gmm_setups.get_init_args()
    mono_args = lbs_gmm_setups.get_monophone_args(allow_zero_weights=True)
    cart_args = lbs_gmm_setups.get_cart_args()
    tri_args = lbs_gmm_setups.get_triphone_args()
    vtln_args = lbs_gmm_setups.get_vtln_args(allow_zero_weights=True)
    sat_args = lbs_gmm_setups.get_sat_args(allow_zero_weights=True)
    vtln_sat_args = lbs_gmm_setups.get_vtln_sat_args()
    final_output_args = lbs_gmm_setups.get_final_output()

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", hybrid_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    # ******************** GMM System ********************

    lbs_gmm_system = gmm_system.GmmSystem()
    lbs_gmm_system.init_system(
        hybrid_init_args=hybrid_init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    lbs_gmm_system.run(steps)

    train_corpus_path = lbs_gmm_system.corpora["train-other-960"].corpus_file
    total_train_num_segments = 281241
    cv_size = 3000 / total_train_num_segments

    all_segments = corpus_recipe.SegmentCorpusJob(
        train_corpus_path, 1
    ).out_single_segment_files[1]

    splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_segments, {"train": 1 - cv_size, "cv": cv_size}
    )
    train_segments = splitted_segments_job.out_segments["train"]
    cv_segments = splitted_segments_job.out_segments["cv"]
    devtrain_segments = text.TailJob(
        train_segments, num_lines=1000, zip_output=False
    ).out

    # ******************** NN Init ********************

    nn_train_data = lbs_gmm_system.outputs["train-other-960"][
        "final"
    ].as_returnn_rasr_data_input(shuffle_data=True)
    nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
    nn_train_data_inputs = {
        "train-other-960.train": nn_train_data,
    }

    nn_cv_data = lbs_gmm_system.outputs["train-other-960"][
        "final"
    ].as_returnn_rasr_data_input()
    nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
    nn_cv_data_inputs = {
        "train-other-960.cv": nn_cv_data,
    }

    nn_devtrain_data = lbs_gmm_system.outputs["train-other-960"][
        "final"
    ].as_returnn_rasr_data_input()
    nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
    nn_devtrain_data_inputs = {
        "train-other-960.devtrain": nn_devtrain_data,
    }
    nn_dev_data_inputs = {
        # "dev-clean": lbs_gmm_system.outputs["dev-clean"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
        "dev-other": lbs_gmm_system.outputs["dev-other"][
            "final"
        ].as_returnn_rasr_data_input(),
    }
    nn_test_data_inputs = {
        # "test-clean": lbs_gmm_system.outputs["test-clean"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
        # "test-other": lbs_gmm_system.outputs["test-other"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
    }

    from pprint import pprint
    pprint(dict(
      hybrid_init_args=hybrid_init_args,
      train_data=nn_train_data_inputs,
      cv_data=nn_cv_data_inputs,
      devtrain_data=nn_devtrain_data_inputs,
      dev_data=nn_dev_data_inputs,
      test_data=nn_test_data_inputs,
    ))
    sys.exit(1)

    nn_args = get_nn_args()

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_repo = CloneGitRepositoryJob(
        url="git@github.com:rwth-i6/returnn.git", checkout_folder_name="returnn"
    )

    lbs_nn_system = hybrid_system.HybridSystem(returnn_root=returnn_repo.out_repository)
    lbs_nn_system.init_system(
        hybrid_init_args=hybrid_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-other-960.train", "train-other-960.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""


# from i6_experiments.users.luescher.setups.librispeech.pipeline_hybrid_args

def get_nn_args(num_outputs: int = 9001, num_epochs: int = 500):
    returnn_configs = get_returnn_configs(
        num_inputs=40, num_outputs=num_outputs, batch_size=24000, num_epochs=num_epochs
    )

    training_args = {
        "log_verbosity": 4,
        "num_epochs": num_epochs,
        "num_classes": num_outputs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 7,
        "cpu_rqmt": 3,
        "partition_epochs": {"train": 20, "dev": 1},
        "use_python_control": False,
    }
    recognition_args = {
        "dev-other": {
            "epochs": list(np.arange(250, num_epochs + 1, 10)),
            "feature_flow_key": "gt",
            "prior_scales": [0.3],
            "pronunciation_scales": [6.0],
            "lm_scales": [20.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": get_search_parameters(),
            "lattice_to_ctm_kwargs": {
                "fill_empty_segments": True,
                "best_path_algo": "bellman-ford",
            },
            "optimize_am_lm_scale": False,
            "rtf": 50,
            "mem": 8,
            "parallelize_conversion": True,
        },
    }
    test_recognition_args = None

    nn_args = rasr_util.HybridArgs(
        returnn_training_configs=returnn_configs,
        returnn_recognition_configs=returnn_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args


def get_returnn_configs(
    num_inputs: int, num_outputs: int, batch_size: int, num_epochs: int
):
    # ******************** blstm base ********************

    base_config = {
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "extern_data": {
            "data": {"dim": num_inputs},
            "classes": {"dim": num_outputs, "sparse": True},
        },
    }
    base_post_config = {
        "cleanup_old_models": {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": returnn.CodeWrapper(f"list(np.arange(10, {num_epochs + 1}, 10))"),
        },
    }

    blstm_base_config = copy.deepcopy(base_config)
    blstm_base_config.update(
        {
            "batch_size": batch_size,  # {"classes": batch_size, "data": batch_size},
            "chunking": "100:50",
            "optimizer": {"class": "nadam"},
            "optimizer_epsilon": 1e-8,
            "gradient_noise": 0.1,
            "learning_rates": returnn.CodeWrapper("list(np.linspace(3e-4, 8e-4, 10))"),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 40,
            "newbob_multi_update_interval": 1,
            "network": {
              # TODO ....
                "output": {
                    "class": "softmax",
                    "loss": "ce",
                    "dropout": 0.1,
                    "from": "data",
                },
            },
        }
    )

    blstm_base_returnn_config = returnn.ReturnnConfig(
        config=blstm_base_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={"numpy": "import numpy as np"},
        pprint_kwargs={"sort_dicts": False},
    )

    return {
        "dummy_nn": blstm_base_returnn_config,
    }
