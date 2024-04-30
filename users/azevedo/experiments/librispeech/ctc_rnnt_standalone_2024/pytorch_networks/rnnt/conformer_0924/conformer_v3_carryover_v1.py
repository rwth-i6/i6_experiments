import torch
from torch import nn
from typing import Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

from i6_models.util import compat

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config

from ..conformer_2.conformer_v3 import (
    mask_tensor,
    ConformerBlockV3,
)

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config


from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import mask_tensor
from i6_models.primitives.specaugment import specaugment_v1_by_length

# frontend imports
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from ..streaming_conformer.generic_frontend_v2 import GenericFrontendV2

# feature extract and conformer module imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import (
    prior_step,
    prior_init_hook,
    prior_finish_hook,
    
    Predictor,
    Joiner
)

from returnn.torch.context import get_run_ctx



class ConformerBlockV3COV1(ConformerBlockV3):
    def __init__(self, cfg: ConformerBlockV1Config):
        super().__init__(cfg)

    def infer(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        states: Optional[List[torch.Tensor]],
        lookahead_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        input should be assumed to be streamed (C+R, F')
            where C is the chunk_size and R the lookahead_size
        sequence_mask: (1, C+R)
        states: List[Tensor(C+R, F')] corresponding to previous chunk output of lower layer
        """
        ext_chunk_sz = input.size(0)

        if states is not None:
            all_curr_chunks = torch.cat((*states, input), dim=0)  # ((i+1)*(C+R), F') = (t, F')
            all_curr_chunks = all_curr_chunks.reshape(-1, all_curr_chunks.size(-1))  # (t, F')
            
            seq_mask = torch.zeros(ext_chunk_sz, sequence_mask.size(-1), device=input.device, dtype=bool)
            seq_mask[:-1, -lookahead_size:] = False  # dont attend to lookahead
            seq_mask[-1] = sequence_mask[0]
            seq_mask = seq_mask.view(-1)  # (t)
            
        else:
            all_curr_chunks = input  # (C+R, F')
            seq_mask = sequence_mask  # (1, C+R)


        #
        # Block forward computation
        #
        x = 0.5 * self.ff1(input) + input  # [C+R, F']

        # multihead attention
        y = self.mhsa.layernorm(x)

        inv_sequence_mask = compat.logical_not(seq_mask)
        output_tensor, _ = self.mhsa.mhsa(
            y, all_curr_chunks, all_curr_chunks, key_padding_mask=inv_sequence_mask, need_weights=False
        )  # [C+R, F]
        x = output_tensor + x  # [C+R, F]

        # chunk-independent convolution
        # TODO: test
        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)
        x = self.conv(x) + x  # [C+R, F]

        x = 0.5 * self.ff2(x) + x  # [C+R, F]
        x = self.final_layer_norm(x)  # [C+R, F]

        return x


class ConformerEncoderCOV1(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV3COV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F] or [B, N, C, F] where C is the chunk size
            and N = T'/C the number of chunks
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T'] or [B, N, C]
        :param lookahead_size: number of lookahead frames chunk is able to attend to
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        batch_size: int = data_tensor.size(0)
        use_chunks: bool = (data_tensor.dim() == sequence_mask.dim() + 1 == 4)
        attn_mask: Optional[torch.Tensor] = None

        if use_chunks:
            # chunking by reshaping [B, N, C, F] to [B*N, C, F] for frontend (pooling + conv2d)
            data_tensor = data_tensor.view(-1, data_tensor.size(-2), data_tensor.size(-1))
            sequence_mask = sequence_mask.view(-1, sequence_mask.size(-1))

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # x shape: [B, T, F'] or [B*N, C', F']
        #print(f"{x.shape = }, {sequence_mask.shape = }, {lookahead_size = }")

        if use_chunks:
            # we are chunking, thus reshape to [B, N, C'+R, F'] where R = lookahead_size
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # [B, N, C', F']
            sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, C']

            # [B, N, C', F'] -> [B, N, C'+R, F']
            x, sequence_mask = self.add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

            # create chunk-causal attn_mask
            attn_mask = self.create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                          chunk_size=x.size(2)-lookahead_size, 
                                          lookahead_size=lookahead_size, device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)  # [B, T, F'] or [B, N, C'+R, F']

        # remove lookahead frames from every chunk
        if use_chunks:
            x = x[:, :, :-lookahead_size].contiguous()
            sequence_mask = sequence_mask[:, :, :-lookahead_size].contiguous()

        return x, sequence_mask     # [B, N, C', F']

    def create_chunk_mask(self, seq_len: int, chunk_size: int, lookahead_size: int = 0, 
                          carry_over: Optional[int] = None, device: Union[torch.device, str] = "cpu"
    ) -> torch.Tensor:
        """
        chunk_size := num. subsampled frames in one chunk
        seq_len = N * (chunk_size + lookahead_size) with N = #chunks
        output of some embed may see every embed in the past and in the current chunk
        """
        attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        chunk_ext_size = chunk_size + lookahead_size

        if carry_over is None:
            carry_over = chunk_ext_size
        
        # TODO: test
        for i in range(0, seq_len, chunk_ext_size):
            # attend to past chunk(s)
            attn_mask[i:i+chunk_ext_size, :i] = True

            # attend to current chunk and its lookahead
            attn_mask[i:i+chunk_ext_size, i:i + chunk_ext_size] = True

        # remove redundant lookahead
        attn_mask = attn_mask.view(seq_len, -1, chunk_ext_size)  # [seq_len, N, C+R]
        attn_mask[:, :, :-lookahead_size] = False
        attn_mask = attn_mask.view(seq_len, seq_len)

        return attn_mask

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
        lookahead_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (1, C, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F'] where
        :param lookahead_size: number of lookahead frames chunk is able to attend to
            P = num. of previous chunks (we only save previous chunk => P = 1)
            L = num. of layers = num. of encoder blocks (L = 0 corresponds to frames of prev. chunk after frontend)
            F' = conformer dim after frontend
        """
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, 2*C)

        input = input.view(2, -1, input.size(2))    # (2, C, F)
        sequence_mask = sequence_mask.view(2, -1)   # (2, C)

        x, sequence_mask = self.frontend(input, sequence_mask)  # (2, C', F')

        x = x.unsqueeze(0)  # (1, 2, C', F')
        sequence_mask = sequence_mask.unsqueeze(0)  # (1, 2, C')

        x, sequence_mask = self.add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)  # (1, 2, C'+R)
        x = x[0, 0]  # (C' + R, F')
        sequence_mask = sequence_mask[0, 0].unsqueeze(0)  # (1, C'+R)

        # save layer outs for next chunk (state)
        layer_outs: List[torch.Tensor] = [x]
        state: Optional[torch.Tensor] = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                # state = states[-1][i]
                state = [prev_chunk[i] for prev_chunk in states]

            x = module.infer(x, sequence_mask, states=state, lookahead_size=lookahead_size)
            layer_outs.append(x)

        x = x[:-lookahead_size].unsqueeze(0)  # (1, C', F')
        sequence_mask = sequence_mask[:, :-lookahead_size]   # (1, C')

        return x, sequence_mask, layer_outs








class Mode(Enum):
    STREAMING = 0
    OFFLINE = 1

from ..streaming_conformer.streaming_conformer_v2_cfg import ModelConfig as ModelConfigStreamingLAH
# FastConformer relevant imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from ..streaming_conformer.generic_frontend_v2 import GenericFrontendV2Config

# complete Transducer config
from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
    SpecaugConfig,
    PredictorConfig,
    VGG4LayerActFrontendV1Config_mod,
)

@dataclass
class ModelConfig(ModelConfigStreamingLAH):
    online_model_scale: float
    carry_over_size: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])

        if d.get("use_vgg", False):
            print(d["frontend_config"])
            d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        else:
            d["frontend_config"] = GenericFrontendV2Config.from_dict(d["frontend_config"])

        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["predictor_config"] = PredictorConfig.from_dict(d["predictor_config"])

        return ModelConfig(**d)


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        # net_args are passed as a dict to returnn and here the config is retransformed into its dataclass
        self.cfg = ModelConfig.from_dict(model_config_dict)

        if self.cfg.use_vgg:
            frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=self.cfg.frontend_config)
        else:
            frontend = ModuleFactoryV1(module_class=GenericFrontendV2, cfg=self.cfg.frontend_config)

        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=frontend,
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
                    channels=self.cfg.conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(self.cfg.conformer_size),
                ),
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderCOV1(cfg=conformer_config)
        self.predictor = Predictor(
            cfg=self.cfg.predictor_config,
            label_target_size=self.cfg.label_target_size + 1,  # ctc blank added
            output_dim=self.cfg.joiner_dim,
        )
        self.joiner = Joiner(
            input_dim=self.cfg.joiner_dim,
            output_dim=self.cfg.label_target_size + 1,
            activation=self.cfg.joiner_activation,
            dropout=self.cfg.joiner_dropout,
        )
        self.encoder_out_linear = nn.Linear(self.cfg.conformer_size, self.cfg.joiner_dim)
        if self.cfg.ctc_output_loss > 0:
            self.encoder_ctc = nn.Linear(self.cfg.conformer_size, self.cfg.label_target_size + 1)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        self.lookahead_size = self.cfg.lookahead_size
        self.carry_over_size = self.cfg.carry_over_size

        self.mode: Optional[Mode] = None

    def extract_features(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
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
        mask = mask_tensor(conformer_in, audio_features_len)

        return conformer_in, mask

    def num_samples_to_frames(self, num_samples: int) -> int:
        n_fft = self.feature_extraction.n_fft
        hop_length = self.feature_extraction.hop_length

        if self.feature_extraction.center:
            return (num_samples // hop_length) + 1
        else:
            return ((num_samples - n_fft) // hop_length) + 1
        
    def get_chunk_size(self, **kwargs) -> int:
        chunk_size = int(self.cfg.chunk_size)
        return self.num_samples_to_frames(num_samples=chunk_size)

    def prep_streaming_input(self, conformer_in, mask):
        batch_size = conformer_in.size(0)

        chunk_size_frames = self.get_chunk_size()

        # pad conformer time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -conformer_in.size(1) % chunk_size_frames
        # (B, T, ...) -> (B, T+time_dim_pad, ...) = (B, T', ...)
        conformer_in = torch.nn.functional.pad(conformer_in, (0, 0, 0, time_dim_pad),
                                               "constant", 0)
        mask = torch.nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

        # separate chunks to signal the conformer that we are chunking input
        conformer_in = conformer_in.view(batch_size, -1, chunk_size_frames,
                                         conformer_in.size(-1))  # (B, (T'/C), C, F) = (B, N, C, F)
        mask = mask.view(batch_size, -1, chunk_size_frames)  # (B, N, C)

        return conformer_in, mask

    def merge_chunks(self, conformer_out, out_mask):
        batch_size = conformer_out.size(0)

        conformer_out = conformer_out.view(batch_size, -1, conformer_out.size(-1))  # (B, C'*N, F')
        out_mask = out_mask.view(batch_size, -1)  # (B, C'*N)

        return conformer_out, out_mask

    def forward(
        self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
            labels: torch.Tensor, labels_len: torch.Tensor, 
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, N]
        :param labels_len: length of N as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        assert self.mode is not None
        #print(f"> Currently running in {self.mode}.")

        conformer_in, mask = self.extract_features(raw_audio, raw_audio_len)

        if self.mode == Mode.STREAMING:
            conformer_in, mask = self.prep_streaming_input(conformer_in, mask)

        carry_over_frames = self.num_samples
        conformer_out, out_mask = self.conformer(conformer_in, mask, 
                                                 lookahead_size=self.lookahead_size,
                                                 carry_over=self.carry_over_size)

        if self.mode == Mode.STREAMING:
            conformer_out, out_mask = self.merge_chunks(conformer_out, out_mask)

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
            target_lengths=labels_len,
        )  # output is [B, T, N, #vocab]

        if self.cfg.ctc_output_loss > 0:
            ctc_logprobs = torch.log_softmax(self.encoder_ctc(conformer_out), dim=-1)
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs


def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    from i6_native_ops import warp_rnnt

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    prepended_targets = labels.new_empty([labels.size(0), labels.size(1) + 1])
    prepended_targets[:, 1:] = labels
    prepended_targets[:, 0] = model.cfg.label_target_size  # blank is last index
    prepended_target_lengths = labels_len + 1

    for encoder_mode in [Mode.STREAMING]:#, Mode.OFFLINE]:
        model.mode = encoder_mode
        logits, audio_features_len, ctc_logprobs = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
            labels=prepended_targets, labels_len=prepended_target_lengths
        )

        logprobs = torch.log_softmax(logits, dim=-1)
        fastemit_lambda = model.cfg.fastemit_lambda

        rnnt_loss = warp_rnnt.rnnt_loss(
            log_probs=logprobs,
            frames_lengths=audio_features_len.to(dtype=torch.int32),
            labels=labels,
            labels_lengths=labels_len.to(dtype=torch.int32),
            blank=model.cfg.label_target_size,
            fastemit_lambda=fastemit_lambda if fastemit_lambda is not None else 0.0,
            reduction="sum",
            gather=True,
        )

        num_phonemes = torch.sum(labels_len)

        scale = model.cfg.online_model_scale if encoder_mode == Mode.STREAMING else (1 - model.cfg.online_model_scale)
        mode_str = encoder_mode.name.lower()[:3]

        if ctc_logprobs is not None:
            transposed_logprobs = torch.permute(ctc_logprobs, (1, 0, 2))  # CTC needs [T, B, #vocab + 1]

            ctc_loss = nn.functional.ctc_loss(
                transposed_logprobs,
                labels,
                input_lengths=audio_features_len,
                target_lengths=labels_len,
                blank=model.cfg.label_target_size,
                reduction="sum",
                zero_infinity=True,
            )
            run_ctx.mark_as_loss(name="ctc.%s" % mode_str, loss=ctc_loss, inv_norm_factor=num_phonemes,
                                 scale=model.cfg.ctc_output_loss * scale)

        run_ctx.mark_as_loss(name="rnnt.%s" % mode_str, loss=rnnt_loss, inv_norm_factor=num_phonemes,
                             scale=scale)