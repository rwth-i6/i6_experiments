import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .feature_extraction import DbMelFeatureExtraction
from ..x_vectors.feature_config import DbMelFeatureExtractionConfig

from torchaudio.functional import mask_along_axis

def apply_spec_aug(input, num_repeat_time, max_dim_time, num_repeat_feat, max_dim_feat):
    """
    :param Tensor input: the input audio features (B,T,F)
    :param int num_repeat_time: number of repetitions to apply time mask
    :param int max_dim_time: number of columns to be masked on time dimension will be uniformly sampled from [0, mask_param]
    :param int num_repeat_feat: number of repetitions to apply feature mask
    :param int max_dim_feat: number of columns to be masked on feature dimension will be uniformly sampled from [0, mask_param]
    """
    for _ in range(num_repeat_time):
        input = mask_along_axis(input, mask_param=max_dim_time, mask_value=0.0, axis=1)

    for _ in range(num_repeat_feat):
        input = mask_along_axis(input, mask_param=max_dim_feat, mask_value=0.0, axis=2)
    return input

class Model(nn.Module):

    def __init__(self, input_dim = 80, num_classes = 8, p_dropout=0.5, spec_augment=False, **kwargs):
        super(Model, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, dilation=1)
        self.bn_tdnn3 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(2*128,128)
        self.bn_fc1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(128,64)
        self.bn_fc2 = nn.BatchNorm1d(64, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(64,num_classes)

        fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

        self.spec_augment = spec_augment
        self.net_kwargs = {
            "repeat_per_num_frames": 100,
            "max_dim_feat": 8,
            "num_repeat_feat": 5,
            "max_dim_time": 20,
        }

    def forward(self, raw_audio, raw_audio_lengths, eps, step):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        with torch.no_grad():
            squeezed_audio = torch.squeeze(raw_audio)
            x, x_lengths = self.feature_extraction(
                squeezed_audio, raw_audio_lengths
            )  # [B, T, F]

        if self.training and self.spec_augment:
            x = apply_spec_aug(
                x,
                num_repeat_time=torch.max(x_lengths).detach().cpu().numpy()
                // self.net_kwargs["repeat_per_num_frames"],
                max_dim_time=self.net_kwargs["max_dim_time"],
                num_repeat_feat=self.net_kwargs["num_repeat_feat"],
                max_dim_feat=self.net_kwargs["max_dim_feat"],
            )

        x = x.transpose(1,2) # [B, F, T]

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))

        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        if (raw_audio.shape[0] > 1):
            x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
            x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        else:
            x = self.dropout_fc1(F.relu(self.fc1(stats)))
            x = self.dropout_fc2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
def train_step(*, model: Model, data, run_ctx, **kwargs):
    tags = data["seq_tag"]
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    # phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    # phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :].long()  # [B, 1] (sparse)
    tags = list(np.array(tags)[indices.detach().cpu().numpy()])


    pred_logits = model(audio_features, audio_features_len, run_ctx.epoch, run_ctx.global_step)
    loss = nn.functional.cross_entropy(pred_logits, speaker_labels.squeeze(1))

    run_ctx.mark_as_loss(name="ce", loss=loss)
