import sys
import os

sys.path.append("/work/asr3/zeyer/schmitt/venvs/imageio-2.35.0")
import imageio

# example_path = "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/train_from_scratch/500-epochs_wo-ctc-loss_mgpu-4/returnn_decoding/epoch-30-checkpoint/no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/work/cross-att/enc-layer-12/energies/plots/plot.dev-other_3660-6517-0005_3660-6517-0005.png"
base_path = (
  "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/"
  "bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/train_from_scratch/500-epochs_wo-ctc-loss_mgpu-4/"
  "returnn_decoding"
)

plot_paths = {
  "energy": (
    "no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/work/cross-att/enc-layer-12/energies/"
    "plots/plot.dev-other_3660-6517-0005_3660-6517-0005.png"
  ),
}

plot_paths.update({
  f"grad_wrt_enc{i}": (
    f"no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/work/log-prob-grads_wrt_enc-{i}_log-space/plots/"
    "plot.dev-other_3660-6517-0005_3660-6517-0005.png"
  ) for i in range(12)
})

plot_paths.update({
  f"grad_wrt_enc{i}": (
    f"no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/work/enc-{i}_cosine_sim/plots/"
    "cos_sim_dev-other_3660-6517-0005_3660-6517-0005.png"
  ) for i in range(12)
})

script_dir = os.path.dirname(os.path.realpath(__file__))

for plot_name, plot_path in plot_paths.items():
  images = []
  for epoch_path in [f"epoch-{ep}-checkpoint" for ep in range (30, 60, 1)]:
    epoch_plot_path = os.path.join(base_path, epoch_path, plot_path)
    images.append(imageio.imread(epoch_plot_path))

  imageio.mimsave(f"{script_dir}/{plot_name}.gif", images, fps=2, loop=0)
