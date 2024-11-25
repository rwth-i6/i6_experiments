from sisyphus import Path, tk

from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJobV2
from i6_experiments.users.schmitt.util.util import CombineNpyFilesJob


def plot_att_weights_mohammad():
  combined_npy_file = CombineNpyFilesJob(
    npy_files=[
      Path("/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.vBWiyIKwiOk9/output/att_weights_ep20/returnn_ep020_data_0_0.npy"),
      Path(
        "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.vBWiyIKwiOk9/output/att_weights_ep20/returnn_ep020_data_1_1.npy"),
    ]
  ).out_file


  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=combined_npy_file,
    targets_hdf=combined_npy_file,
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=None,
    ref_alignment_hdf=None,
    json_vocab_path=Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab"),
    ctc_alignment_hdf=None,
    segment_whitelist=None,
    ref_alignment_json_vocab_path=None,
    plot_w_cog=False,
  )
  plot_att_weights_job.add_alias("tedlium/att_weights")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)
