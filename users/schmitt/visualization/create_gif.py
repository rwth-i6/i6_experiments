import sys
import os

sys.path.append("/work/asr3/zeyer/schmitt/venvs/imageio-2.35.0")
import imageio

# example_path = "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/train_from_scratch/500-epochs_wo-ctc-loss_mgpu-4/returnn_decoding/epoch-30-checkpoint/no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/work/cross-att/enc-layer-12/energies/plots/plot.dev-other_3660-6517-0005_3660-6517-0005.png"

# decoding_path = (
#   "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/"
#   "bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/train_from_scratch/500-epochs_wo-ctc-loss_mgpu-4/"
#   "returnn_decoding"
# )
decoding_path = sys.argv[1]
from_epoch = int(sys.argv[2])
to_epoch = int(sys.argv[3])
dirname = sys.argv[4]

analyze_gradients_path = "no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/work/"

plot_paths = {}

# plot_paths.update({
#   "energy": "cross-att/enc-layer-12/energies/plots/plot.dev-other_3660-6517-0005_3660-6517-0005.png",
# })
#
# plot_paths.update({
#   f"grad_wrt_enc{i}": (
#     f"log-prob-grads_wrt_enc-{i}_log-space/plots/plot.dev-other_3660-6517-0005_3660-6517-0005.png"
#   ) for i in range(12)
# })
#
# plot_paths.update({
#   f"enc_cos_sim{i}": (
#     f"enc-{i}_cosine_sim/cos_sim_dev-other_3660-6517-0005_3660-6517-0005_.png"
#   ) for i in range(12)
# })

plot_paths.update({
  f"enc_att_key_heads_cos_sim": f"enc_att_key_heads_cos_sim/enc_att_key_heads_cos_sim_dev-other_3660-6517-0005_3660-6517-0005.png"
})

plot_paths.update({
  f"enc_att_query_heads_cos_sim": f"enc_att_key_heads_cos_sim/enc_att_key_heads_cos_sim_dev-other_3660-6517-0005_3660-6517-0005.png"
})

plot_paths.update({
  f"enc_att_value_heads_cos_sim": f"enc_att_key_heads_cos_sim/enc_att_key_heads_cos_sim_dev-other_3660-6517-0005_3660-6517-0005.png"
})

script_dir = os.path.dirname(os.path.realpath(__file__))

for plot_name, plot_path in plot_paths.items():
  images = []
  for epoch_path in [f"epoch-{ep}-checkpoint" for ep in range(from_epoch, to_epoch + 1, 1)]:
    epoch_plot_path = os.path.join(decoding_path, epoch_path, analyze_gradients_path, plot_path)
    images.append(imageio.imread(epoch_plot_path))

  gif_dir = os.path.join(script_dir, dirname)
  os.makedirs(gif_dir, exist_ok=True)
  imageio.mimsave(f"{gif_dir}/{plot_name}.gif", images, fps=1, loop=0)
