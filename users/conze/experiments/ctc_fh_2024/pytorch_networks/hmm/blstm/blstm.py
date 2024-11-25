import torch
from torch import nn

from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.rasr_fsa import RasrFsaBuilder

from returnn.torch.context import get_run_ctx

from .blstm_cfg import BlstmPoolingEncoderConfig, BlstmModelConfig

class BlstmPoolingEncoder(nn.Module):
    """
    BLSTM block module with pooling layers
    """

    def __init__(self, config: BlstmPoolingEncoderConfig):
        super().__init__()
        self.dropout = config.dropout
        self.enforce_sorted = config.enforce_sorted

        config.pooling_layer_positions.sort()

        for p_pos in config.pooling_layer_positions:
            assert p_pos < config.num_layers

        # Assert no duplicates
        assert len(config.pooling_layer_positions) == len(set(config.pooling_layer_positions))

        self.blstm_stack = nn.ModuleList()
        for idx, p_pos in enumerate(config.pooling_layer_positions):
            num_layers=p_pos if idx == 0 else (p_pos - config.pooling_layer_positions[idx - 1])
            self.blstm_stack.append(nn.LSTM(
                input_size=config.input_dim if idx == 0 else 2*config.hidden_dim,
                hidden_size=config.hidden_dim,
                bidirectional=True,
                num_layers=num_layers,
                batch_first=True,
                dropout=self.dropout if num_layers > 1 else 0.0,  # Dropout for single-layer LSTM gives warning that you need to add a final dropout, which we do
            ))

            # Add final dropout
            self.blstm_stack.append(nn.Dropout(
                p=self.dropout,
            ))
            self.blstm_stack.append(nn.MaxPool1d(
                kernel_size=2,
                ceil_mode=True,  # Adds padding to the end
            ))

        self.blstm_stack.append(nn.LSTM(
            input_size=config.input_dim if len(config.pooling_layer_positions) == 0 else 2*config.hidden_dim,
            hidden_size=config.hidden_dim,
            bidirectional=True,
            num_layers=config.num_layers - (
                config.pooling_layer_positions[-1] if len(config.pooling_layer_positions) > 0 else 0),
            batch_first=True,
            dropout=self.dropout,
        ))

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            # during graph mode we have to assume all Tensors are on the correct device,
            # otherwise move lengths to the CPU if they are on GPU
            if seq_len.get_device() >= 0:
                seq_len = seq_len.cpu()

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=seq_len,
            enforce_sorted=self.enforce_sorted,
            batch_first=True,
        )

        for module in self.blstm_stack:
            if isinstance(module, nn.LSTM):
                blstm_packed_in, _ = module(blstm_packed_in)
            elif isinstance(module, nn.Dropout):
                # Assumes that this is preceded by a BLSTM layer and followed by a MaxPool1d layer
                blstm_padded, seq_lens_before_pooling = nn.utils.rnn.pad_packed_sequence(blstm_packed_in, padding_value=0.0, batch_first=True)
                blstm_padded = module(blstm_padded)
                # Not re-packed here, because MaxPool1d is following
            elif isinstance(module, nn.MaxPool1d):
                blstm_padded = blstm_padded.permute(0, 2, 1)  # Pool along sequence length
                blstm_pooled = module(blstm_padded)
                blstm_pooled = blstm_pooled.permute(0, 2, 1)
                seq_lens_after_pooling = (seq_lens_before_pooling + module.kernel_size - 1) // module.kernel_size
                blstm_packed_in = nn.utils.rnn.pack_padded_sequence(
                    input=blstm_pooled,
                    lengths=seq_lens_after_pooling,
                    enforce_sorted=self.enforce_sorted,
                    batch_first=True,
                )
            else:
                raise NotImplementedError

        blstm_out, subsampled_seq_len = nn.utils.rnn.pad_packed_sequence(blstm_packed_in, padding_value=0.0, batch_first=True)

        return blstm_out, subsampled_seq_len


class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = BlstmModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.blstm = BlstmPoolingEncoder(config=frontend_config)
        self.final_linear = nn.Linear(2*frontend_config.hidden_dim, self.cfg.label_target_size)
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specaug_start_epoch

        self.builder = RasrFsaBuilder(self.cfg.fsa_config_path, self.cfg.tdp_scale)

    def forward(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels]
        """

        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,  # TODO: make configurable
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        blstm_in = audio_features_masked_2

        blstm_out, subsampled_audio_features_len = self.blstm(blstm_in, audio_features_len)
        blstm_out = self.final_dropout(blstm_out)
        logits = self.final_linear(blstm_out)

        log_probs = torch.log_softmax(logits, dim=2)

        # Cast to int tensor
        subsampled_audio_features_len = subsampled_audio_features_len.int()
        return log_probs, subsampled_audio_features_len.to("cuda")

def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    raw_audio_len, indices = torch.sort(raw_audio_len, descending=True)
    raw_audio = raw_audio[indices, :, :]
    seq_tags = [data["seq_tag"][i] for i in indices]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    am_scaled_logprobs = logprobs.mul(model.cfg.am_scale)

    weighted_fsa = model.builder.build_batch(seq_tags).to(run_ctx.device)

    from i6_native_ops.fbw import fbw_loss
    fbw_loss = fbw_loss(am_scaled_logprobs, weighted_fsa, audio_features_len)

    # Normalization
    num_output_frames = torch.sum(audio_features_len)

    run_ctx.mark_as_loss(name="hmm-fbw", loss=fbw_loss.sum(), inv_norm_factor=num_output_frames)


def prior_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", 'w') as f:
        np.savetxt(f, log_average_probs, delimiter=' ')
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))

