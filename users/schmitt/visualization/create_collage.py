import sys
import os
from PIL import Image, ImageDraw, ImageFont

# example_path = "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/train_from_scratch/500-epochs_wo-ctc-loss_mgpu-4/returnn_decoding/epoch-30-checkpoint/no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/work/cross-att/enc-layer-12/energies/plots/plot.dev-other_3660-6517-0005_3660-6517-0005.png"

# decoding_path = (
#   "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/"
#   "bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/train_from_scratch/500-epochs_wo-ctc-loss_mgpu-4/"
#   "returnn_decoding"
# )
decoding_path = sys.argv[1]
filename = sys.argv[2]
epoch_range = eval(sys.argv[3])
layer_range = eval(sys.argv[4])
scale = sys.argv[5]

analyze_gradients_path = "no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/work/"

plot_paths = {}

# plot_paths.update({
#   f"enc-{i}_cos_sim": (
#     f"enc-{i}_cosine_sim/cos_sim_dev-other_7697-105815-0051_7697-105815-0051_.png"
#   ) for i in range(12)
# })
# crop = ("50", "60", "img.size[0] * 0.85", "img.size[1]")

# head = 7
# head_to_ij_mapping = {
#   0: (0, 0),
#   1: (1, 0),
#   2: (2, 0),
#   3: (3, 0),
#   4: (0, 1),
#   5: (1, 1),
#   6: (2, 1),
#   7: (3, 1),
# }
# plot_paths.update({
#   f"enc-{i}_self_att_head-{head}": (
#     f"enc-{i}/self_att/energies_grid/energies_dev-other_7697-105815-0051_7697-105815-0051.png"
#   ) for i in layer_range
# })
# plot_paths.update({
#   f"enc-{i}_att_key_heads": (
#     f"enc-{i}_att_key_heads_cos_sim_and_pca/enc-{i}_att_key_heads_cosine_sim_dev-other_7697-105815-0051_7697-105815-0051.png"
#   ) for i in layer_range
# })
# i, j = head_to_ij_mapping[head]
# crop = (
#   f"img.size[0] * 0.25 * {i} + 30",  # left
#   f"{1 - j} * 60 + {j} * img.size[1] * 0.525",  # upper
#   f"img.size[0] * 0.25 * ({i + 1})",  # right
#   f"img.size[1] * 0.525 if {j} == 0 else img.size[1]"  # lower
# )

# plot_paths.update({
#   f"enc-{i}_att_key_heads": (
#     f"enc-{i}/intermediate_layer_pca/x_conv_out/pca_dev-other_7697-105815-0051_7697-105815-0051.png"
#   ) for i in layer_range
# })
# crop = ("50", "100", "img.size[0]", "img.size[1]")

plot_paths.update({
  f"enc-{i}_att_key_heads": (
    f"enc-{i}/log-prob-grads_wrt_enc-{i}_log-space/plots/plot.dev-other_7697-105815-0051_7697-105815-0051.png"
  ) for i in layer_range
})
# plot_paths.update({
#   f"enc-{i}_att_key_heads": (
#     f"log-prob-grads_wrt_enc-{i}_log-space/plots/plot.dev-other_7697-105815-0051_7697-105815-0051.png"
#   ) for i in layer_range
# })
plot_paths["cross-att"] = (
  "cross-att/enc-layer-12/energies/plots/plot.dev-other_7697-105815-0051_7697-105815-0051.png"
)
crop = ("0", "0", "img.size[0]", "img.size[1]")


script_dir = os.path.dirname(os.path.realpath(__file__))

collage = None
draw = None
font = None
collage_width = 0
collage_height = 0
for j, (plot_name, plot_path) in enumerate(plot_paths.items()):
  images = []
  # epochs = range(11, 162, 10)

  for i, epoch_path in enumerate([f"epoch-{ep}-checkpoint" for ep in epoch_range]):
    epoch_plot_path = os.path.join(decoding_path, epoch_path, analyze_gradients_path, plot_path)
    img = Image.open(epoch_plot_path)
    # crop away individual plot title
    img = img.crop((eval(crop[0]), eval(crop[1]), eval(crop[2]), eval(crop[3])))
    img = img.resize((int(img.size[0] * float(scale)), int(img.size[1] * float(scale))))
    img_size = img.size
    if i == 0 and j == 0:
      # create collage
      # collage is as wide as all images together - subtract_from_x for each image
      collage_width = (img_size[0]) * len(epoch_range)
      collage_height = img_size[1] * len(plot_paths)
      collage = Image.new("RGB", (collage_width, collage_height), color="white")

    collage.paste(img, (i * (img_size[0]), j * img_size[1]))

# # add title to collage
# collage_w_title = Image.new("RGB", (collage_width, collage_height + 50))
# collage_w_title.paste(collage, (0, 50))
# draw = ImageDraw.Draw(collage_w_title)
# font = ImageFont.load_default()
# draw.text((10, 10), f"Sub-epochs", (255, 255, 255), font=font)

# save image
collage_path = os.path.join(script_dir, f"{filename}.png")
collage_dir = os.path.dirname(collage_path)
os.makedirs(collage_dir, exist_ok=True)
collage.save(collage_path)


