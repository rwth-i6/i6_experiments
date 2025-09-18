from dataclasses import dataclass
from typing import Union, Callable, Tuple

import torch
import torchaudio

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import Import, PartialImport
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
import i6_models.parts.conformer as conformer_parts_i6
import i6_models.assemblies.conformer as conformer_i6
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from .util import lengths_to_padding_mask

from ..custom_parts.specaugment import (
    SpecaugmentConfigV1,
    SpecaugmentModuleV1,
)


@dataclass
class FFNNTransducerConfig(ModelConfiguration):
    specaugment: ModuleFactoryV1
    conformer: ModuleFactoryV1
    encoder_dim: int
    prediction_layers: int
    prediction_layer_size: int
    prediction_act: Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    prediction_dropout: float
    joint_layer_size: int
    joint_act: torch.nn.Module
    context_history_size: int
    context_embedding_size: int
    target_size: int
    blank_idx: int


class FFNNTransducer(torch.nn.Module):
    def __init__(self, step: int, cfg: FFNNTransducerConfig, **_):
        super().__init__()
        self.specaugment = cfg.specaugment()
        self.conformer = cfg.conformer()

        self.target_size = cfg.target_size

        self.blank_idx = cfg.blank_idx

        self.encoder_linear = torch.nn.Linear(cfg.encoder_dim, self.target_size)

        self.embedding = torch.nn.Embedding(num_embeddings=self.target_size, embedding_dim=cfg.context_embedding_size)

        self.context_history_size = cfg.context_history_size
        prediction_layers = []
        prev_size = self.context_history_size * cfg.context_embedding_size
        for _ in range(cfg.prediction_layers):
            prediction_layers.append(torch.nn.Dropout(cfg.prediction_dropout))
            prediction_layers.append(torch.nn.Linear(prev_size, cfg.prediction_layer_size))
            prediction_layers.append(cfg.prediction_act)
            prev_size = cfg.prediction_layer_size

        self.prediction_network = torch.nn.ModuleList(prediction_layers)

        self.prediction_linear = torch.nn.Linear(prev_size, self.target_size)

        self.joint = torch.nn.Sequential(
            torch.nn.Linear(self.target_size, cfg.joint_layer_size),
            cfg.joint_act,
            torch.nn.Linear(cfg.joint_layer_size, self.target_size),
        )

    def forward_encoder(
        self,
        features: torch.Tensor,  # [B, T, F]
        features_len: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, T, C], [B]
        with torch.no_grad():
            assert features_len is not None
            sequence_mask = lengths_to_padding_mask(features_len)
            sequence_mask = torch.nn.functional.pad(
                sequence_mask, (0, features.size(1) - sequence_mask.size(1)), value=0
            )

            x = self.specaugment(features)  # [B, T, F]

        x, sequence_mask = self.conformer(x, sequence_mask)  # [B, T, E], [B, T]

        x = self.encoder_linear(x)  # [B, T, C]

        return x, torch.sum(sequence_mask, dim=1).to(torch.int32)  # [B, T, C], [B]

    def _extend_context_history(
        self,
        context: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:  # [B, S, H]
        if self.context_history_size == 1:
            return context.unsqueeze(-1)  # [B, S, H] = [B, S, 1]

        context_list = [context]
        for _ in range(self.context_history_size - 1):
            context_list.append(
                torch.nn.functional.pad(context_list[-1][:, :-1], pad=(1, 0), value=self.blank_idx)
            )  # [B, S]
        context_stack = torch.stack(context_list, dim=-1)  # [B, S, H]

        return context_stack  # [B, S, H]

    def _context_from_targets(
        self,
        targets: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:  # [B, S, H]
        context = torch.nn.functional.pad(targets, pad=(1, 0), value=self.blank_idx)  # [B, S+1]

        return self._extend_context_history(context)  # [B, S+1, H]

    def _context_from_targets_viterbi(
        self,
        targets: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:  # [B, T, H]
        blank_mask = (targets == self.blank_idx).to(torch.float32)  # [B, T]
        non_blank_mask = 1 - blank_mask  # [B, T]
        context = torch.nn.functional.pad(targets[:, :-1], pad=(1, 0), value=self.blank_idx)  # [B, T]
        for t in range(1, targets.size(1)):
            context[:, t] = (
                blank_mask[:, t - 1] * context[:, t - 1] + non_blank_mask[:, t - 1] * context[:, t]
            )  # [B, T]

        return self._extend_context_history(context)  # [B, T, H]

    def forward_predictor(
        self,
        context: torch.Tensor,  # [B, T, H] or [B, H]
    ) -> torch.Tensor:  # [B, T, C] or [B, C]
        a = self.embedding(context)  # [B, T, H, A] or [B, H, A]
        a = torch.reshape(a, shape=[*(a.shape[:-2]), a.shape[-2] * a.shape[-1]])  # [B, T, H*A]
        for layer in self.prediction_network:
            a = layer(a)  # [B, T, P] or [B, P]

        a = self.prediction_linear(a)
        return a  # [B, T, C] or [B, C]

    def forward_viterbi(
        self,
        features: torch.Tensor,  # [B, T, F]
        features_len: torch.Tensor,  # [B]
        targets: torch.Tensor,  # [B, T']
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # [B, T', C], [B, T', C], [B, T', C], [B]
        enc, enc_lens = self.forward_encoder(features, features_len)  # [B, T', C], [B]
        enc_log_probs = torch.log_softmax(enc, dim=2)  # [B, T', C]

        context = self._context_from_targets_viterbi(targets)  # [B, T', H]

        pred = self.forward_predictor(context)  # [B, T', C]
        pred_log_probs = torch.log_softmax(pred, dim=2)  # [B, T', C]

        logits = self.joint(enc + pred)  # [B, T', C]
        log_probs = torch.log_softmax(logits, dim=2)  # [B, T', C]

        return log_probs, pred_log_probs, enc_log_probs, enc_lens

    def forward_with_pruned_loss(
        self,
        features: torch.Tensor,  # [B, T, F]
        features_len: torch.Tensor,  # [B]
        targets: torch.Tensor,  # [B, S]
        targets_len: torch.Tensor,  # [B]
        prune_range: int = 5,
        enc_scale: float = 0.1,
        pred_scale: float = 0.25,
        reduction: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import fast_rnnt

        enc, enc_lens = self.forward_encoder(features, features_len)  # [B, T', E], [B]

        context = self._context_from_targets(targets)  # [B, S+1, H]
        pred = self.forward_predictor(context)  # [B, S+1, P]

        boundary = torch.zeros((features.size(0), 4), dtype=torch.int64, device=enc_lens.device)
        boundary[:, 2] = targets_len
        boundary[:, 3] = enc_lens

        # [B]
        simple_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_smoothed(
            lm=pred,
            am=enc,
            symbols=targets,
            termination_symbol=self.blank_idx,
            lm_only_scale=pred_scale,
            am_only_scale=enc_scale,
            boundary=boundary,
            reduction=reduction,
            return_grad=True,
        )

        # [B, T, prune_range]
        ranges = fast_rnnt.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # [B, T, prune_range, C], [B, T, prune_range, C]
        am_pruned, lm_pruned = fast_rnnt.do_rnnt_pruning(am=enc, lm=pred, ranges=ranges)

        # [B, T, prune_range, C]
        logits = self.joint(am_pruned + lm_pruned)

        # [B]
        pruned_loss = fast_rnnt.rnnt_loss_pruned(
            logits=logits,
            symbols=targets,
            ranges=ranges,
            termination_symbol=self.blank_idx,
            boundary=boundary,
            reduction=reduction,
        )
        return simple_loss.sum(), pruned_loss.sum()

    def forward_single(
        self,
        encoder: torch.Tensor,  # [B, C],
        context: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:  # [B, C]
        pred = self.forward_predictor(context)  # [B, C]
        logits = self.joint(encoder + pred)  # [B, C]
        log_probs = torch.log_softmax(logits, dim=1)  # [B, C]

        return log_probs


def get_train_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.train_steps.transducer.train_step", import_as="train_step"),
            # PartialImport(
            #     code_object_path=f"{pytorch_package}.train_steps.transducer.train_step",
            #     hashed_arguments=kwargs,
            #     unhashed_package_root="",
            #     unhashed_arguments={},
            # ),
        ],
    )


def get_viterbi_train_serializer(
    model_config: FFNNTransducerConfig,
) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{pytorch_package}.train_steps.transducer.train_step_viterbi", import_as="train_step"),
        ],
    )


def get_beam_search_serializer(model_config: FFNNTransducerConfig, **kwargs) -> Collection:
    pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{FFNNTransducer.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            PartialImport(
                code_object_path=f"{pytorch_package}.forward.transducer.beam_search_forward_step",
                import_as="forward_step",
                hashed_arguments=kwargs,
                unhashed_package_root="",
                unhashed_arguments={},
            ),
            Import(f"{pytorch_package}.forward.search_callback.SearchCallback", import_as="forward_callback"),
        ],
    )


def get_default_config_v1(num_inputs: int, num_outputs: int) -> FFNNTransducerConfig:
    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentModuleV1,
        cfg=SpecaugmentConfigV1(
            time_min_num_masks=1,
            time_max_num_masks=1,
            time_mask_max_size=15,
            freq_min_num_masks=1,
            freq_max_num_masks=num_inputs // 10,
            freq_mask_max_size=5,
        ),
    )

    frontend_cfg = VGG4LayerActFrontendV1Config(
        in_features=num_inputs,
        conv1_channels=32,
        conv2_channels=32,
        conv3_channels=64,
        conv4_channels=64,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 2),
        pool1_stride=None,
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=None,
        pool2_padding=None,
        activation=torch.nn.SiLU(),
        out_features=512,
    )

    frontend = ModuleFactoryV1(VGG4LayerActFrontendV1, frontend_cfg)

    ff_cfg = conformer_parts_i6.ConformerPositionwiseFeedForwardV1Config(
        input_dim=512,
        hidden_dim=2048,
        dropout=0.1,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = conformer_parts_i6.ConformerMHSAV1Config(
        input_dim=512,
        num_att_heads=8,
        att_weights_dropout=0.1,
        dropout=0.1,
    )

    conv_cfg = conformer_parts_i6.ConformerConvolutionV1Config(
        channels=512,
        kernel_size=31,
        dropout=0.1,
        activation=torch.nn.SiLU(),
        norm=torch.nn.BatchNorm1d(num_features=512, affine=False),
    )

    block_cfg = conformer_i6.ConformerBlockV1Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
    )

    conformer_cfg = conformer_i6.ConformerEncoderV1Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    return FFNNTransducerConfig(
        specaugment=specaugment,
        conformer=ModuleFactoryV1(module_class=conformer_i6.ConformerEncoderV1, cfg=conformer_cfg),
        encoder_dim=512,
        prediction_layers=2,
        prediction_layer_size=640,
        prediction_act=torch.nn.Tanh(),
        prediction_dropout=0.1,
        joint_layer_size=128,
        joint_act=torch.nn.Tanh(),
        context_history_size=1,
        context_embedding_size=256,
        target_size=num_outputs,
        blank_idx=0,
    )
