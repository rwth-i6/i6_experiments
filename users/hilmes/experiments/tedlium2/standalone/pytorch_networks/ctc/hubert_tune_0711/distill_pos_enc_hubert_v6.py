"""
V3 fixes eliminate blank
V4 updates keepsome
V5 adds threshold and random
V6 tries to improve training time through batching
"""

import numpy as np
import torch
from torch import nn
from torch import tensor
from typing import Optional
from transformers import HubertModel, HubertConfig

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosEncoderV1Config
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosBlockV1Config, ConformerRelPosEncoderV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from i6_models.parts.conformer.convolution import ConformerConvolutionV2Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
from i6_models.parts.dropout import BroadcastDropout
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from returnn.torch.context import get_run_ctx

from .distill_pos_enc_hubert_v5_cfg import ModelConfig, DistillConfig


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


class Teacher(torch.nn.Module):
    def __init__(self, hubert, linear):
        super().__init__()
        self.hubert = hubert
        self.final_linear = linear


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, distill_config_dict: Optional, **kwargs):
        super().__init__()
        self.cfg: ModelConfig = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        conformer_config = ConformerRelPosEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerRelPosBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                    input_dim=conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
                ),
                mhsa_cfg=ConformerMHSARelPosV1Config(
                    input_dim=conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    with_bias=self.cfg.mhsa_with_bias,
                    dropout=self.cfg.mhsa_dropout,
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
                    learnable_pos_emb=self.cfg.pos_emb_config.learnable_pos_emb,
                    rel_pos_clip=self.cfg.pos_emb_config.rel_pos_clip,
                    with_linear_pos=self.cfg.pos_emb_config.with_linear_pos,
                    with_pos_bias=self.cfg.pos_emb_config.with_pos_bias,
                    separate_pos_emb_per_head=self.cfg.pos_emb_config.separate_pos_emb_per_head,
                    pos_emb_dropout=self.cfg.pos_emb_config.pos_emb_dropout,
                ),
                conv_cfg=ConformerConvolutionV2Config(
                    channels=conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size),
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
                ),
                modules=self.cfg.module_list,
                scales=self.cfg.module_scales,
            ),
        )
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerRelPosEncoderV1(cfg=conformer_config)
        self.num_output_linears = 1 if self.cfg.aux_ctc_loss_layers is None else len(self.cfg.aux_ctc_loss_layers)
        self.output_linears = nn.ModuleList(
            [
                nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
                for _ in range(self.num_output_linears)
            ]
        )
        self.output_dropout = BroadcastDropout(
            p=self.cfg.final_dropout, dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
        )
        self.return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]
        self.scales = self.cfg.aux_ctc_loss_scales or [1.0]
        if self.training and distill_config_dict is not None:
            self.distill_config: DistillConfig = DistillConfig(**distill_config_dict)
            hubert: HubertModel = HubertModel(
                HubertConfig.from_pretrained(
                    f"facebook/hubert-{self.distill_config.model_name}",
                    cache_dir="/work/asr4/hilmes/debug/whisper/transformers/",
                )
            )
            hubert.eval()
            teacher_linear = nn.Linear(hubert.config.hidden_size, self.cfg.label_target_size + 1)  # + CTC blank
            self.teacher = Teacher(hubert, teacher_linear)
            if self.distill_config.kd_hyps is not None:
                self.kd_hyps = {}
                with open(self.distill_config.kd_hyps, "rt") as f:
                    from torch import tensor

                    for line in f:
                        if line == "{\n" or line == "}\n":
                            continue
                        line = line.replace("tensor(nan)", "tensor(float('nan'))")
                        line = line.split(":")
                        name, dic = line[0], ":".join(line[1:])
                        if "tensor(float('nan'))" in dic:
                            print(name, dic)
                        self.kd_hyps[name] = eval(dic)
                    # content = f.read()
                    # content.replace("nan", "float(nan)")
                    # self.kd_hyps = eval(content)
            else:
                self.kd_hyps = None
            if self.distill_config.prior_file is not None:
                self.prior_file = np.loadtxt(self.distill_config.prior_file, dtype="float32")
                self.prior_scale = self.distill_config.prior_scale
            else:
                self.prior_file = None
                self.prior_scale = None
            if self.distill_config.warmup_loss is not None:
                self.upscale = nn.Conv1d(conformer_size, hubert.config.hidden_size, 1)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch
        self.downsample_teacher = nn.MaxPool2d(
            kernel_size=(2, 1),
            stride=(2, 1),
            padding=(0, 0),
        )
        # No particular weight init!

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: list of logprobs [B, T, #labels + blank], mask [B, T]
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

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        conformer_out_layers, out_mask = self.conformer(conformer_in, mask, return_layers=self.return_layers)
        log_probs_list = []
        logit_ls = []
        for i, (out_layer, scale) in enumerate(zip(conformer_out_layers, self.scales)):
            if scale == 0.0:
                continue
            conformer_out = self.output_dropout(out_layer)
            logits = self.output_linears[i](conformer_out)
            logit_ls.append(logits)
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs_list.append(log_probs)

        if len(log_probs_list) == 1:
            log_probs_list = log_probs_list[0]
        if len(logit_ls) == 1:
            logit_ls = logit_ls[0]

        lengths = torch.sum(out_mask, dim=1)
        teacher_logits = None
        if self.training or run_ctx.stage == "train_step":
            with torch.no_grad():
                teacher_outputs = self.teacher.hubert(input_values=squeezed_features)
                teacher_out = teacher_outputs.last_hidden_state
                teacher_out = self.downsample_teacher(teacher_out)
                teacher_logits = self.teacher.final_linear(teacher_out)
                if self.distill_config.warmup_loss is not None and run_ctx.epoch < self.distill_config.warmup_loss:
                    logit_ls = self.upscale(conformer_out_layers[-1].transpose(1, 2)).transpose(1, 2)
                    teacher_logits = teacher_out[:, : logit_ls.shape[1], :]
                    lengths = out_mask
                else:
                    teacher_logits = teacher_logits[:, : log_probs_list.shape[1], :]

        return log_probs_list, lengths, logit_ls, teacher_logits


def train_step(*, model: Model, data, run_ctx, **kwargs):
    assert False, "This seems to be worse in some cases"
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]
    logprobs_list, audio_features_len, student_logits, teacher_logits = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    if not isinstance(logprobs_list, list):
        logprobs_list = [logprobs_list]
    assert not isinstance(student_logits, list)
    assert student_logits.shape[1] == logprobs_list[0].shape[1]
    assert isinstance(teacher_logits, torch.Tensor), type(teacher_logits)

    for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
        if model.distill_config.warmup_loss is not None and run_ctx.epoch < model.distill_config.warmup_loss:
            continue
        transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
        ctc_loss = nn.functional.ctc_loss(
            transposed_logprobs,
            labels,
            input_lengths=audio_features_len,
            target_lengths=labels_len,
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )
        num_phonemes = torch.sum(labels_len)
        scale = scale * model.distill_config.ctc_scale if scale == 1.0 else scale
        run_ctx.mark_as_loss(
            name=f"ctc_loss_layer{layer_index + 1}", loss=ctc_loss, scale=scale, inv_norm_factor=num_phonemes
        )

    T = model.distill_config.t
    if model.distill_config.warmup_loss is not None and run_ctx.epoch < model.distill_config.warmup_loss:
        if model.distill_config.mask_padding is True:
            audio_mask = audio_features_len.unsqueeze(dim=-1)
            audio_mask = ~audio_mask
            student_logits = torch.masked_fill(student_logits, audio_mask, 0)
            teacher_logits = torch.masked_fill(teacher_logits, audio_mask, 0)
        loss = nn.functional.cosine_embedding_loss(
            student_logits.flatten(1, 2),
            teacher_logits.flatten(1, 2),
            torch.ones(student_logits.size()[0]).to(device="cuda"),
            margin=0,
            reduction="sum",
        )
        # run_ctx.mark_as_loss(name=f"Cosine Warmup loss", loss=loss, scale=1, inv_norm_factor=torch.sum(labels_len))
        loss.requires_grad_(True)
        run_ctx.mark_as_loss(name=f"Cosine Warmup loss", loss=loss, scale=1)
    elif model.distill_config.eliminate_blanks is True or (
        not isinstance(model.distill_config.eliminate_blanks, bool)
        and (
            (0 < model.distill_config.eliminate_blanks < run_ctx.epoch)
            or (0 > model.distill_config.eliminate_blanks > -1 * run_ctx.epoch)
        )
    ):
        assert model.distill_config.eliminate_blanks is not False
        if model.prior_file is not None:
            teacher_log_soft = nn.functional.log_softmax(teacher_logits)
            teacher_log_soft -= torch.tensor(model.prior_scale * model.prior_file).to(device="cuda")
            pos = torch.argmax(teacher_log_soft, dim=-1)
        else:
            pos = torch.argmax(teacher_logits, dim=-1)
        pos_blank: torch.Tensor = pos == model.cfg.label_target_size
        pos_non_blank: torch.Tensor = ~pos_blank
        if model.distill_config.keep_some_blanks is not None:
            tmp = pos_non_blank
            if model.distill_config.keep_some_blanks[0] > 0:
                if model.distill_config.increase_keepsome_epochs is not None:
                    keepsome = model.distill_config.keep_some_blanks[0] * (
                        run_ctx.epoch // model.distill_config.increase_keepsome_epochs
                    )
                else:
                    keepsome = model.distill_config.keep_some_blanks[0]
                shift = tmp
                for _ in range(keepsome):
                    shift = torch.roll(shift, -1, dims=-1)
                    shift[-1] = tmp[-1]
                    pos_non_blank = pos_non_blank + shift
            if model.distill_config.keep_some_blanks[1] > 0:
                if model.distill_config.increase_keepsome_epochs is not None:
                    keepsome = model.distill_config.keep_some_blanks[1] * (
                        run_ctx.epoch // model.distill_config.increase_keepsome_epochs
                    )
                else:
                    keepsome = model.distill_config.keep_some_blanks[1]
                shift = tmp
                for _ in range(keepsome):
                    shift = torch.roll(shift, 1, dims=-1)
                    shift[0] = tmp[0]
                    pos_non_blank = pos_non_blank + shift
        elif model.distill_config.trim_blanks is True:
            idx = torch.arange(pos_non_blank.shape[1], 0, -1).to(device="cuda")
            first_pos = pos_non_blank * idx
            first_pos = torch.argmax(first_pos, 1)
            idx = torch.arange(0, pos_non_blank.shape[1], 1).to(device="cuda")
            last_pos = pos_non_blank * idx
            last_pos = torch.argmax(last_pos, 1)
            pos_non_blank_new = torch.zeros_like(pos_non_blank)
            for i, (first, last) in enumerate(zip(first_pos, last_pos)):
                pos_non_blank_new[i, first : last + 1] = 1
            pos_non_blank_new.to(device="cuda").bool()
            pos_non_blank_new[:, 0] = pos_non_blank[:, 0]
            pos_non_blank = pos_non_blank_new
        pos_non_blank = pos_non_blank.unsqueeze(dim=-1)
        teacher_seq = torch.masked_select(teacher_logits, pos_non_blank)
        student_seq = torch.masked_select(student_logits, pos_non_blank)
        teacher_seq = teacher_seq.view(-1, model.cfg.label_target_size + 1)
        student_seq = student_seq.view(-1, model.cfg.label_target_size + 1)
        soft_targets = nn.functional.log_softmax(teacher_seq / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_seq / T, dim=-1)
        soft_targets_loss = torch.nn.functional.kl_div(
            soft_prob, soft_targets, reduction="batchmean", log_target=True
        ) * (T**2)
        if soft_targets_loss.isnan():
            soft_targets_loss = (soft_prob * 0).sum()
            print("WARNING: empty KD loss")
        run_ctx.mark_as_loss(
            name=f"KL", loss=soft_targets_loss, scale=model.distill_config.distill_scale
        ),  # norm not needed here as torch does it
    elif model.kd_hyps is not None:
        sm = 0
        loss_sum = 0
        num_phonemes = tensor(0)
        for i, name in enumerate(data["seq_tag"]):
            if name in model.kd_hyps:
                for teacher_sample, score in model.kd_hyps[name][0].items():
                    teacher_sample = eval(teacher_sample)
                    loss = nn.functional.ctc_loss(
                        log_probs=logprobs_list[-1][i],
                        targets=teacher_sample,
                        input_lengths=audio_features_len[i],
                        target_lengths=tensor([len(teacher_sample)]),
                        blank=model.cfg.label_target_size,
                        reduction="sum",
                        zero_infinity=True,
                    )
                    loss_sum += loss
                    sm += score * loss
                    num_phonemes += tensor(len(teacher_sample))
        if model.distill_config.normalize_stud:
            sm /= loss_sum
        if not sm == 0:
            run_ctx.mark_as_loss(
                name=f"KL", loss=sm, scale=model.distill_config.distill_scale, inv_norm_factor=num_phonemes
            )
    elif model.distill_config.keep_random is not None:
        if model.prior_file is not None:
            teacher_log_soft = nn.functional.log_softmax(teacher_logits)
            teacher_log_soft -= torch.tensor(model.prior_scale * model.prior_file).to(device="cuda")
            pos = torch.argmax(teacher_log_soft, dim=-1)
        else:
            pos = torch.argmax(teacher_logits, dim=-1)
        pos_blank: torch.Tensor = pos == model.cfg.label_target_size
        pos_non_blank: torch.Tensor = ~pos_blank
        seq_non_blank_ls = []
        for seq_blank, seq_non_blank, labels in zip(pos_blank, pos_non_blank, data["labels"]):
            num_non_blank: torch.Tensor = torch.sum(seq_non_blank)  # [1]
            num_select: torch.Tensor = torch.round(num_non_blank * model.distill_config.keep_random).to(
                dtype=torch.int32
            )
            ls_blank = seq_blank.nonzero()
            x = len(ls_blank)
            positions = torch.randperm(x)
            positions = positions[:num_select]
            positions = ls_blank[positions]
            tmp = torch.zeros_like(seq_blank)
            tmp[positions] = 1
            seq_non_blank = seq_non_blank + tmp
            seq_non_blank_ls.append(seq_non_blank)
        pos_non_blank = torch.stack(seq_non_blank_ls, dim=0)
        pos_non_blank = pos_non_blank.unsqueeze(dim=-1)
        teacher_seq = torch.masked_select(teacher_logits, pos_non_blank)
        student_seq = torch.masked_select(student_logits, pos_non_blank)
        teacher_seq = teacher_seq.view(-1, model.cfg.label_target_size + 1)
        student_seq = student_seq.view(-1, model.cfg.label_target_size + 1)
        soft_targets = nn.functional.log_softmax(teacher_seq / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_seq / T, dim=-1)
        soft_targets_loss_new = torch.nn.functional.kl_div(
            soft_prob, soft_targets, reduction="batchmean", log_target=True
        ) * (T**2)
        if soft_targets_loss_new.isnan():
            soft_targets_loss_new = (soft_prob * 0).sum()
            print("WARNING: empty KD loss")
        run_ctx.mark_as_loss(name=f"KL", loss=soft_targets_loss_new, scale=model.distill_config.distill_scale)

    elif model.distill_config.keep_threshold is not None:
        teacher_soft = nn.functional.softmax(teacher_logits, dim=-1)
        teacher_sums = torch.concat(
            (torch.sum(teacher_soft[:, :, :-1], dim=-1, keepdims=True), teacher_soft[:, :, -1:]), dim=-1
        )
        pos_blank = torch.argmax(teacher_logits, dim=-1) == model.cfg.label_target_size
        pos_blank2 = torch.argmax(teacher_soft, dim=-1) == model.cfg.label_target_size
        assert torch.equal(pos_blank, pos_blank2)
        pos_thresh = teacher_sums[:, :, 0] >= model.distill_config.keep_threshold
        pos_sel = pos_blank * pos_thresh
        pos_non_blank: torch.Tensor = ~pos_blank
        pos_non_blank = pos_non_blank + pos_sel
        pos_non_blank = pos_non_blank.unsqueeze(dim=-1)
        teacher_seq = torch.masked_select(teacher_logits, pos_non_blank)
        student_seq = torch.masked_select(student_logits, pos_non_blank)
        teacher_seq = teacher_seq.view(-1, model.cfg.label_target_size + 1)
        student_seq = student_seq.view(-1, model.cfg.label_target_size + 1)
        soft_targets = nn.functional.log_softmax(teacher_seq / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_seq / T, dim=-1)
        soft_targets_loss = torch.nn.functional.kl_div(
            soft_prob, soft_targets, reduction="batchmean", log_target=True
        ) * (T**2)
        run_ctx.mark_as_loss(name=f"KL", loss=soft_targets_loss, scale=model.distill_config.distill_scale)
    else:
        soft_targets = nn.functional.log_softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
        if model.distill_config.mask_padding is True:
            raise NotImplementedError
            audio_mask = mask_tensor(soft_targets, audio_features_len)
            assert all(torch.sum(audio_mask, dim=1) == audio_features_len)
            audio_mask = ~audio_mask
            audio_mask = audio_mask.unsqueeze(dim=-1)
            soft_targets = torch.masked_fill(soft_targets, audio_mask, 0)
            soft_targets_log = torch.masked_fill(soft_targets_log, audio_mask, 0)
            soft_prob = torch.masked_fill(soft_prob, audio_mask, 0)
        soft_targets_loss = torch.nn.functional.kl_div(
            soft_prob, soft_targets, reduction="batchmean", log_target=True
        ) * (T**2)
        run_ctx.mark_as_loss(
            name=f"KL",
            loss=soft_targets_loss,
            scale=model.distill_config.distill_scale,
            inv_norm_factor=torch.tensor(soft_targets.size(0)),
        )


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
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len, test, test2 = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))


def calc_blank_init_hook(run_ctx, **kwargs):
    run_ctx.seqs = {}


def calc_blank_finish_hook(run_ctx, **kwargs):
    import pickle

    with open("blank_counts.pkl", "wb") as f:
        pickle.dump(run_ctx.seqs, f)


def calc_blank_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len, _, _ = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    for seq, tag in zip(logprobs, data["seq_tag"]):
        pos = torch.argmax(seq, dim=-1)
        pos_blank: torch.Tensor = pos == model.cfg.label_target_size
        pos_non_blank: torch.Tensor = ~pos_blank
        idx = torch.arange(pos_non_blank.shape[0], 0, -1).to(device=seq.device)
        first_pos = pos_non_blank * idx
        first_pos = torch.argmax(first_pos, 0, keepdim=True)
        idx = torch.arange(0, pos_non_blank.shape[0], 1).to(device=seq.device)
        last_pos = pos_non_blank * idx
        last_pos = torch.argmax(last_pos, 0, keepdim=True)
        pos_blank = pos_blank[first_pos:last_pos]
        groups = []
        counter = 0
        for pos in pos_blank:
            if pos == 0:  # found non blank
                if counter > 0:
                    groups.append(counter)
                counter = 0
            elif pos == 1:  # another blank blank
                counter += 1
            else:
                assert False, "Matrix is not boolean for some reason"
        run_ctx.seqs[tag] = groups
