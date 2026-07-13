import torch
import torch.nn.functional as F
from i6_models.primitives.specaugment import specaugment_v1_by_length
from returnn.torch.context import get_run_ctx

from .i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first import mask_tensor
from .i6modelsV1_VGG4LayerActFrontendV1_v6_generative_gmm import Model as GmmModel


class Model(GmmModel):
    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)
            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )

        mask = mask_tensor(audio_features, audio_features_len)
        conformer_out, out_mask = self.conformer(audio_features, mask)
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)
        return logits.float(), torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to("cpu")
    alignments = data["alignments"].to(torch.long)
    alignment_lengths = data["alignments:size1"].to(torch.long)

    logits, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
    factor = model.alignment_subsampling_factor
    downsampled_alignments = alignments[:, ::factor]
    shared_time_dim = min(logits.shape[1], downsampled_alignments.shape[1])
    if shared_time_dim == 0:
        return

    logits = logits[:, :shared_time_dim]
    downsampled_alignments = downsampled_alignments[:, :shared_time_dim]
    if torch.any(downsampled_alignments < 0) or torch.any(
        downsampled_alignments >= model.alignment_to_ctc_map.shape[0]
    ):
        raise ValueError("GMM alignment contains a label id outside the configured alignment label map")
    remapped_alignments = model.alignment_to_ctc_map[downsampled_alignments]

    valid_seq_len = torch.div(alignment_lengths, factor, rounding_mode="floor").to(audio_features_len.device)
    valid_seq_len = torch.minimum(valid_seq_len, audio_features_len)
    valid_seq_len = torch.clamp(valid_seq_len, max=shared_time_dim)
    valid_mask = mask_tensor(logits, valid_seq_len)
    num_valid_frames = valid_mask.sum()
    if num_valid_frames == 0:
        return

    log_posteriors = F.log_softmax(logits, dim=-1)
    valid_log_posteriors = log_posteriors[valid_mask]
    log_prior = torch.logsumexp(valid_log_posteriors, dim=0) - torch.log(
        num_valid_frames.to(log_posteriors.dtype)
    )
    posterior_prior_scores = valid_log_posteriors - log_prior[None, :]
    loss = F.cross_entropy(
        posterior_prior_scores,
        remapped_alignments[valid_mask],
        reduction="sum",
    )
    run_ctx.mark_as_loss(
        name="gmm_posterior_prior_ce",
        loss=loss,
        inv_norm_factor=num_valid_frames.to(torch.float32),
    )
