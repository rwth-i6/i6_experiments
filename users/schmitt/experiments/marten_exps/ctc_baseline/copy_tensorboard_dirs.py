import os
import shutil

alias_dir = "/u/schmitt/experiments/2025_03_10_ctc_usr/alias/ctc"
runs_dir = "/u/schmitt/experiments/2025_03_10_ctc_usr/runs"

for top_dir in os.listdir(alias_dir):
  if not "resc-ff-8" in top_dir:
    continue

  top_dir_path = os.path.join(alias_dir, top_dir)

  for bottom_dir in os.listdir(top_dir_path):
    bottom_dir_path = os.path.join(top_dir_path, bottom_dir)

    run_dir = os.path.join(bottom_dir_path, "train", "work", "runs")
    if not os.path.exists(run_dir):
      continue

    # os.makedirs(os.path.join(runs_dir, top_dir), exist_ok=True)
    # os.symlink(run_dir, os.path.join(runs_dir, top_dir, bottom_dir))
    shutil.copytree(run_dir, os.path.join(runs_dir, f"{top_dir[len('ctc-wo_aux_loss-ds100US_accum40-bpe128_model-frPR_no-sp-recog_albert_'):]}_{bottom_dir}"))
