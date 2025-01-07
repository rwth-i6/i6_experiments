"""
Compute feature statistics of a dataset
"""

import torch
import json
import numpy as np

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim, TensorDict
from returnn.forward_iface import ForwardCallbackIface
from returnn.config import get_global_config

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import Model
from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import kldiv_ctc_lm_loss
from i6_experiments.users.yang.torch.utils.tensor_ops import mask_eos_label
from i6_experiments.users.phan.utils.masking import get_seq_mask

def forward_compute_features_mean_stddev(
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    non_blank_targets: Tensor,
    non_blank_targets_spatial_dim: Dim,
):
    data_raw = data.raw_tensor
    targets_raw = non_blank_targets.raw_tensor

    config = get_global_config()  # noqa
    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    features, features_spatial_dim = model.feature_extraction(
        data, in_spatial_dim=data_spatial_dim
    )

    features_np = features.raw_tensor.detach().cpu().numpy()
    seq_len = features_spatial_dim.dyn_size_ext.raw_tensor.detach().cpu().numpy()

    rf.get_run_ctx().mark_as_output(
        features_np,
        "features",
    )

    rf.get_run_ctx().mark_as_output(
        seq_len,
        "seq_len",
    )


default_out_file_mean = "mean"
default_out_file_stddev = "std_dev"
default_out_file_info = "info.json"
output_files = [
    default_out_file_mean,
    default_out_file_stddev,
    default_out_file_info
]

class Compute_Features_Mean_StdDev_Forward_Callback(ForwardCallbackIface):
    def __init__(
        self,
        output_file_mean=default_out_file_mean,
        output_file_stddev=default_out_file_stddev,
        output_file_info=default_out_file_info,
    ):
        self.output_file_mean = output_file_mean
        self.output_file_stddev = output_file_stddev
        self.output_file_info = output_file_info

    def init(self, model):
        self.n_features = model.in_dim.dimension
        self.n_seqs = 0
        self.n_frames = 0
        self.mean = np.zeros((self.n_features,))
        self.mean_of_squares = np.zeros((self.n_features,)) # 1/N sum x_i**2
        print(f"Number of features: {self.n_features}")
        

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        features = outputs["features"].raw_tensor # (T, F)
        seq_len = outputs["seq_len"].raw_tensor # scalar T
        # print("seq tag", seq_tag)
        # print("features", features)
        # print("seq len", seq_len)
        for i in range(seq_len):
            self.n_frames += 1
            self.mean += (features[i] - self.mean) / self.n_frames
            self.mean_of_squares += (np.square(features[i]) - self.mean_of_squares) / self.n_frames
        self.n_seqs += 1
        # print("n frames", self.n_frames)
        # print("n seqs", self.n_seqs)
        # print("mean", self.mean)
        # print("mean of squares", self.mean_of_squares)

    def finish(self):
        info = {
            "n_frames": self.n_frames,
            "n_seqs": self.n_seqs,
            "n_features": self.n_features,
        }
        self.std_dev = np.sqrt(self.mean_of_squares - np.square(self.mean))
        with open(self.output_file_info, "w") as out_info:
            json.dump(info, out_info, indent=4)
        with open(self.output_file_mean, "w") as out_mean:
            np.savetxt(out_mean, self.mean)
        with open(self.output_file_stddev, "w") as out_stddev:
            np.savetxt(out_stddev, self.std_dev)

def forward_callback_wrapper():
    return Compute_Features_Mean_StdDev_Forward_Callback()
