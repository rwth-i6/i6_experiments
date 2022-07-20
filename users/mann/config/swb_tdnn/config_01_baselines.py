from sisyphus import *

from i6_experiments.users.mann.setups.swb.swbsystem import SWBNNSystem
from i6_experiments.users.mann.setups.nn_system.base_system import NNSystem

from i6_experiments.users.mann.setups.nn_system.switchboard import get_legacy_switchboard_system, get_cart
from i6_experiments.users.mann.nn.tdnn import make_baseline

epochs = [12, 24, 32, 80, 160, 240, 320]

swb_system = get_legacy_switchboard_system()

# swb_system.run("baseline_viterbi_lstm")

baseline_viterbi = swb_system.baselines["viterbi_lstm"]()

print(swb_system.crp["train"].nn_trainer_exe)

print(swb_system.returnn_root)

swb_system.nn_and_recog(
    name="baseline_viterbi_lstm",
    crnn_config=baseline_viterbi,
    epochs=epochs
)

baseline_tdnn = make_baseline(num_input=40)

swb_system.nn_and_recog(
    name="baseline_viterbi_tdnn",
    crnn_config=baseline_tdnn,
    epochs=epochs
)

# make 1-state cart
get_cart(swb_system, hmm_partition=1)


swb_system.nn_and_recog(
    name="baseline_viterbi_lstm.1s",
    crnn_config=baseline_viterbi,
    epochs=epochs
)

swb_system.nn_and_recog(
    name="baseline_viterbi_tdnn.1s",
    crnn_config=baseline_tdnn,
    epochs=epochs
)

