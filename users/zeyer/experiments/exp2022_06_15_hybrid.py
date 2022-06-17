
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

# TODO remove these
import i6_experiments.users.luescher.setups.librispeech.pipeline_base_args as lbs_gmm_setups
import i6_experiments.users.luescher.setups.librispeech.pipeline_hybrid_args as lbs_hybrid_setups


def run():
  filename_handle = os.path.splitext(os.path.basename(__file__))[0]
  gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
  rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

  nn_args = lbs_hybrid_setups.get_nn_args()

  nn_steps = rasr_util.RasrSteps()
  nn_steps.add_step("nn", nn_args)

  hybrid_init_args = default_gmm_hybrid_init_args()

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


run()
