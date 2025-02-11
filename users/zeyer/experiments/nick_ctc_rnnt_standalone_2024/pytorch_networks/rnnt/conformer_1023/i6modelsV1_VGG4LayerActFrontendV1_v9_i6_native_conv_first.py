"""
Modified from v4 with proper configuration for the predictor and using i6models feature extraction

Sets joiner dropout correctly

v8: with external CTC loss possibility
"""
import os

import numpy as np
import torch
import torchaudio
from torch import nn
from typing import List, Optional, Tuple

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerEncoderV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerConvolutionV1Config,
    ConformerMHSAV1,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)

from returnn.torch.context import get_run_ctx

from .i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import ModelConfig, PredictorConfig




def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


class ConformerBlockV1ConvFirst(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = 0.5 * self.ff1(x) + x  #  [B, T, F]
        x = self.conv(x) + x  #  [B, T, F]
        x = self.mhsa(x, sequence_mask) + x  #  [B, T, F]
        x = 0.5 * self.ff2(x) + x  #  [B, T, F]
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


class ConformerEncoderV1ConvFirst(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV1ConvFirst(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        return x, sequence_mask


class Predictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Taken from torchaudio
    """

    def __init__(
        self,
        cfg: PredictorConfig,
        label_target_size: int,
        output_dim: int
    ) -> None:
        """

        :param cfg: model configuration for the predictor
        :param label_target_size: shared value from model
        :param output_dim: shared value from model
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(label_target_size, cfg.symbol_embedding_dim)
        self.embedding_dropout = nn.Dropout(cfg.emebdding_dropout)
        self.input_layer_norm = torch.nn.LayerNorm(cfg.symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                nn.LSTM(
                    input_size=cfg.symbol_embedding_dim if idx == 0 else cfg.lstm_hidden_dim,
                    hidden_size=cfg.lstm_hidden_dim,
                )
                for idx in range(cfg.num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=cfg.lstm_dropout)
        self.linear = torch.nn.Linear(cfg.lstm_hidden_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.lstm_dropout = cfg.lstm_dropout

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        embedding_out = self.embedding_dropout(embedding_out)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else [s.permute(1, 0, 2) for s in state[layer_idx]])
            lstm_out = self.dropout(lstm_out)
            state_out.append([s.permute(1, 0, 2) for s in lstm_state_out])

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), lengths, state_out


class Joiner(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

    Taken directly from torchaudio
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu", dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        joint_encodings = self.dropout(joint_encodings)
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return output, source_lengths, target_lengths


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=self.cfg.frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=self.cfg.conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=self.cfg.conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=self.cfg.conformer_size, kernel_size=self.cfg.conv_kernel_size, dropout=self.cfg.conv_dropout, activation=nn.functional.silu,
                    norm=LayerNormNC(self.cfg.conformer_size)
                ),
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderV1ConvFirst(cfg=conformer_config)
        self.predictor = Predictor(
            cfg=self.cfg.predictor_config,
            label_target_size=self.cfg.label_target_size + 1,  # ctc blank added
            output_dim=self.cfg.joiner_dim
        )
        self.joiner = Joiner(
            input_dim=self.cfg.joiner_dim,
            output_dim=self.cfg.label_target_size + 1,
            activation=self.cfg.joiner_activation,
            dropout=self.cfg.joiner_dropout
        )
        self.encoder_out_linear = nn.Linear(self.cfg.conformer_size, self.cfg.joiner_dim)
        if self.cfg.ctc_output_loss > 0:
            self.encoder_ctc = nn.Linear(self.cfg.conformer_size, self.cfg.label_target_size + 1)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch


    def forward(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: torch.Tensor,
            labels: torch.Tensor,
            labels_len: torch.Tensor
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, N]
        :param labels_len: length of N as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        # torch._dynamo.mark_dynamic(conformer_in, 0)
        # torch._dynamo.mark_dynamic(conformer_in, 1)

        # torch._dynamo.mark_dynamic(mask, 0)
        # torch._dynamo.mark_dynamic(mask, 1)

        conformer_out, out_mask = self.conformer(conformer_in, mask)
        conformer_joiner_out = self.encoder_out_linear(conformer_out)
        conformer_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        output_logits, src_len, tgt_len = self.joiner(
            source_encodings=conformer_joiner_out,
            source_lengths=conformer_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len
        )  # output is [B, T, N, #vocab]

        if self.cfg.ctc_output_loss > 0:
            ctc_logprobs = torch.log_softmax(self.encoder_ctc(conformer_out), dim=-1)
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs


def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    from i6_native_ops import warp_rnnt
    # if not hasattr(run_ctx, "did_compile") or run_ctx.did_compile == False:
    #     from torch._dynamo.utils import CompileProfiler
    #     run_ctx.prof = CompileProfiler()
    #     run_ctx.engine._model.conformer = torch.compile(model.conformer, dynamic=True, fullgraph=True, backend=run_ctx.prof)
    #     print("DID COMPILE!")
    #     run_ctx.did_compile = True

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    prepended_targets = labels.new_empty([labels.size(0), labels.size(1) + 1])
    prepended_targets[:, 1:] = labels
    prepended_targets[:, 0] = model.cfg.label_target_size  # blank is last index
    prepended_target_lengths = labels_len + 1

    logits, audio_features_len, ctc_logprobs = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        labels=prepended_targets,
        labels_len=prepended_target_lengths
    )

    # if run_ctx.global_step == 2:
    #     print(run_ctx.prof.report())
    #     assert False

    logprobs = torch.log_softmax(logits, dim=-1)

    rnnt_loss = warp_rnnt.rnnt_loss(
        log_probs=logprobs,
        frames_lengths=audio_features_len.to(dtype=torch.int32),
        labels=labels,
        labels_lengths=labels_len.to(dtype=torch.int32),
        blank=model.cfg.label_target_size,
        fastemit_lambda=0.0,
        reduction="sum",
        gather=True,
    )

    num_phonemes = torch.sum(labels_len)

    if ctc_logprobs is not None:
        transposed_logprobs = torch.permute(ctc_logprobs, (1, 0, 2))  # CTC needs [T, B, F]
        ctc_loss = nn.functional.ctc_loss(
            transposed_logprobs,
            labels,
            input_lengths=audio_features_len,
            target_lengths=labels_len,
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )
        run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes, scale=model.cfg.ctc_output_loss)


    run_ctx.mark_as_loss(name="rnnt", loss=rnnt_loss, inv_norm_factor=num_phonemes)


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
