
# /work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/

from sisyphus import gs, tk, Path


import os
import sys

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


run()
