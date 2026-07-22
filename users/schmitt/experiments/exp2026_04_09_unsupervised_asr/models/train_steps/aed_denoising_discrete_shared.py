from abc import abstractmethod
from typing import List, Protocol, Sequence, Tuple, Union, Optional, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence
import numpy as np

import returnn.frontend as rf
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict

from .util import get_random_mask, mask_sequence
from .aed_denoising_discrete import DenoisingAedModel
from . import aed_denoising_discrete


class SharedDenoisingAedModel(DenoisingAedModel):
    mask_idx: int
    bos_idx: int
    eos_idx: int
    embedding: nn.Embedding
    decoder: nn.Module

    audio_mask_idx: int
    audio_bos_idx: int
    audio_eos_idx: int
    audio_embedding: nn.Embedding
    audio_decoder: nn.Module

    text_mask_idx: int
    text_bos_idx: int
    text_eos_idx: int
    text_embedding: nn.Embedding
    text_decoder: nn.Module

    @abstractmethod
    def forward_text(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return:
        """

    @abstractmethod
    def forward_audio(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return:
        """

    @abstractmethod
    def decode_audio_seq(
        self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor
    ) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """

    @abstractmethod
    def decode_text_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """

    @abstractmethod
    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """


def train_step(
    *,
    model: SharedDenoisingAedModel,
    extern_data: TensorDict,
    audio_ce_loss_scale: Optional[float] = None,
    audio_masked_ce_loss_scale: Optional[float] = None,
    text_ce_loss_scale: Optional[float] = None,
    text_masked_ce_loss_scale: Optional[float] = None,
    label_smoothing: float = 0.0,
    label_smoothing_start_epoch: int = 0,
    text_masking_opts: Optional[Dict] = None,
    audio_masking_opts: Optional[Dict] = None,
    text_expansion_opts: Optional[Dict] = None,
    aux_loss_scales: Optional[Sequence[float]] = None,
    codebook_diversity_loss_scale: float = 0.0,
    adv_loss_scale: float = 0.0,
    adv_loss_type: str = "bce",
    grad_penalty_scale: float = 0.0,
    **_kwargs,
):
    """
    :param text_expansion_opts: if given ({"min_dup", "max_dup"}), upsample the text encoder input by
        duplicating tokens so it becomes longer than the (unchanged) text reconstruction target,
        simulating the audio>text length ratio. Only applied to the text modality.
    :param adv_loss_scale: if > 0, add the domain-adversarial GAN loss that pushes the shared
        encoder to produce modality-invariant states (a discriminator tries to tell audio from text
        encoder states, the encoder tries to fool it). Requires the model to have a discriminator
        (``discriminator_type`` one of "mlp" / "mlp_2gram" / "mlp_3gram" / "mlp_4gram" / "lstm").
        text is domain 0, audio is domain 1.
    :param adv_loss_type: "bce" (default, classic domain-adversarial GAN), "wasserstein", or
        "wasserstein_interp". "wasserstein" makes the discriminator a critic (raw scores) with loss
        E[D(fake)] - E[D(real)] (fake=audio, real=text) accumulated over the alternating sub-batches
        (works under alternate_batching). "wasserstein_interp" is the faithful WGAN-GP that compares
        both modalities jointly in one step and interpolates between them for the gradient penalty;
        it therefore requires **mixed batches** (turn alternate_batching OFF). When a batch happens to
        contain only one modality, the interpolation loss is skipped for that batch.
    :param grad_penalty_scale: for the wasserstein variants, if > 0 add the WGAN-GP gradient penalty
        on the discriminator step (typical value 10.0). "wasserstein" uses the real-sample penalty at
        the current modality's states (``_Discriminator.gradient_penalty``); "wasserstein_interp" uses
        the interpolated real<->fake penalty (``_Discriminator.interpolated_gradient_penalty``).
    """
    assert set(extern_data.data.keys()) == {"data", "phon_indices", "seq_tag"} or set(extern_data.data.keys()) == {
        "data",
        "phon_indices",
        "phon_indices_w_sil",
        "seq_tag",
    }
    audio_indices_: ReturnnTensor = extern_data["data"]
    phon_indices_: ReturnnTensor = extern_data["phon_indices"]
    if "phon_indices_w_sil" in extern_data:
        phon_indices_targets = extern_data["phon_indices"]
        phon_indices_ = extern_data["phon_indices_w_sil"]
    else:
        phon_indices_targets = None

    # for the faithful WGAN-GP ("wasserstein_interp"), the per-modality steps only stash their encoder
    # states here; the joint critic loss + interpolated gradient penalty are computed once both
    # modalities' states are available (see the end of this function). A fresh dict per batch means a
    # missing modality (single-modality batch) simply won't be in the stash.
    adv_stash = {} if (adv_loss_scale > 0.0 and adv_loss_type == "wasserstein_interp") else None

    model.decode_seq = model.decode_text_seq
    model.forward = model.forward_text
    model.mask_idx = model.text_mask_idx
    model.bos_idx = model.text_bos_idx
    model.eos_idx = model.text_eos_idx
    model.embedding = model.text_embedding
    model.decoder = model.text_decoder
    aed_denoising_discrete.train_step(
        model=model,
        extern_data=TensorDict(
            {
                "data": phon_indices_,
                "seq_tag": extern_data["seq_tag"],
                **({"target": phon_indices_targets} if phon_indices_targets is not None else {}),
            }
        ),
        ce_loss_scale=text_ce_loss_scale,
        masked_ce_loss_scale=text_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=text_masking_opts,
        input_expansion_opts=text_expansion_opts,
        aux_loss_scales=aux_loss_scales,
        codebook_diversity_loss_scale=codebook_diversity_loss_scale,
        adv_loss_scale=adv_loss_scale,
        adv_loss_type=adv_loss_type,
        grad_penalty_scale=grad_penalty_scale,
        adv_stash=adv_stash,
        true_adv_target=0,  # text = domain 0
        loss_name="text",
    )

    model.decode_seq = model.decode_audio_seq
    model.forward = model.forward_audio
    model.mask_idx = model.audio_mask_idx
    model.bos_idx = model.audio_bos_idx
    model.eos_idx = model.audio_eos_idx
    model.embedding = model.audio_embedding
    model.decoder = model.audio_decoder
    aed_denoising_discrete.train_step(
        model=model,
        extern_data=TensorDict({"data": audio_indices_, "seq_tag": extern_data["seq_tag"]}),
        ce_loss_scale=audio_ce_loss_scale,
        masked_ce_loss_scale=audio_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=audio_masking_opts,
        aux_loss_scales=aux_loss_scales,
        codebook_diversity_loss_scale=codebook_diversity_loss_scale,
        adv_loss_scale=adv_loss_scale,
        adv_loss_type=adv_loss_type,
        grad_penalty_scale=grad_penalty_scale,
        adv_stash=adv_stash,
        true_adv_target=1,  # audio = domain 1
        loss_name="audio",
    )

    # faithful WGAN-GP: now that both modalities' encoder states are available, compute the joint
    # critic/generator loss (+ interpolated gradient penalty) -- but only if this batch actually held
    # both modalities (mixed batch). Single-modality batches are skipped, as requested.
    if adv_stash is not None and "text" in adv_stash and "audio" in adv_stash:
        _wasserstein_interp_adv_loss(model, adv_stash, adv_loss_scale, grad_penalty_scale)


def _wasserstein_interp_adv_loss(model, adv_stash: Dict, adv_loss_scale: float, grad_penalty_scale: float):
    """Joint WGAN(-GP) domain-adversarial loss over both modalities' shared-encoder states.

    text = real (domain 0), audio = fake (domain 1). The critic maximizes E[D(text)] - E[D(audio)];
    equivalently it minimizes ``critic_loss = E[D(audio)] - E[D(text)]``. The generator (shared
    encoder) minimizes the negative, pushing the two modalities' state distributions together.

    Discriminator vs generator phase follows the per-modality steps' schedule (``ctx.step % 4``: 0,1
    -> generator, else discriminator), which also set the freeze state accordingly (encoder frozen /
    discriminator unfrozen on the discriminator step, and vice versa). So on the discriminator step
    we detach the states (train only the critic) and add the gradient penalty; on the generator step
    we backprop through the (frozen) critic into the encoder.
    """
    ctx = rf.get_run_ctx()
    text_states, text_lens = adv_stash["text"]
    audio_states, audio_lens = adv_stash["audio"]
    is_disc_step = int(ctx.step) % 4 not in (0, 1)

    if is_disc_step:
        d_text = model.discriminator(text_states.detach(), text_lens)  # [M_text]
        d_audio = model.discriminator(audio_states.detach(), audio_lens)  # [M_audio]
        # per-frame means so neither modality dominates by sheer sequence length.
        critic_loss = d_audio.mean() - d_text.mean()
        ctx.mark_as_loss(critic_loss, name="adv_disc", scale=adv_loss_scale, dims=[])
        if grad_penalty_scale > 0.0:
            grad_pen = model.discriminator.interpolated_gradient_penalty(
                text_states, text_lens, audio_states, audio_lens
            )
            ctx.mark_as_loss(grad_pen, name="adv_grad_pen", scale=adv_loss_scale * grad_penalty_scale, dims=[])
    else:
        d_text = model.discriminator(text_states, text_lens)  # [M_text]
        d_audio = model.discriminator(audio_states, audio_lens)  # [M_audio]
        gen_loss = d_text.mean() - d_audio.mean()  # = -critic_loss
        ctx.mark_as_loss(gen_loss, name="adv_gen", scale=adv_loss_scale, dims=[])


def train_step_v2(
    *,
    model: SharedDenoisingAedModel,
    extern_data: TensorDict,
    audio_ce_loss_scale: Optional[float] = None,
    audio_masked_ce_loss_scale: Optional[float] = None,
    text_ce_loss_scale: Optional[float] = None,
    text_masked_ce_loss_scale: Optional[float] = None,
    label_smoothing: float = 0.0,
    label_smoothing_start_epoch: int = 0,
    text_masking_opts: Optional[Dict] = None,
    audio_masking_opts: Optional[Dict] = None,
    aux_loss_scales: Optional[Sequence[float]] = None,
    **_kwargs,
):
    assert set(extern_data.data.keys()) == {"data", "phon_indices", "seq_tag"}
    audio_indices_: ReturnnTensor = extern_data["data"]
    audio_indices_lens: Tensor = audio_indices_.dims[1].dyn_size_ext.raw_tensor
    phon_indices_: ReturnnTensor = extern_data["phon_indices"]
    phon_indices_lens: Tensor = phon_indices_.dims[1].dyn_size_ext.raw_tensor

    # if torch.all(phon_indices_lens > 0).item():
    #     assert torch.all(audio_indices_lens == 0).item()
    model.decode_seq = model.decode_text_seq
    model.forward = model.forward_text
    model.mask_idx = model.text_mask_idx
    model.bos_idx = model.text_bos_idx
    model.eos_idx = model.text_eos_idx
    model.embedding = model.text_embedding
    model.decoder = model.text_decoder
    aed_denoising_discrete.train_step_v2(
        model=model,
        extern_data=TensorDict({"data": phon_indices_, "seq_tag": extern_data["seq_tag"]}),
        ce_loss_scale=text_ce_loss_scale,
        masked_ce_loss_scale=text_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=text_masking_opts,
        aux_loss_scales=aux_loss_scales,
        loss_name="text",
    )

    # elif torch.all(audio_indices_lens > 0).item():
    #     assert torch.all(phon_indices_lens == 0).item()
    model.decode_seq = model.decode_audio_seq
    model.forward = model.forward_audio
    model.mask_idx = model.audio_mask_idx
    model.bos_idx = model.audio_bos_idx
    model.eos_idx = model.audio_eos_idx
    model.embedding = model.audio_embedding
    model.decoder = model.audio_decoder
    aed_denoising_discrete.train_step_v2(
        model=model,
        extern_data=TensorDict({"data": audio_indices_, "seq_tag": extern_data["seq_tag"]}),
        ce_loss_scale=audio_ce_loss_scale,
        masked_ce_loss_scale=audio_masked_ce_loss_scale,
        label_smoothing=label_smoothing,
        label_smoothing_start_epoch=label_smoothing_start_epoch,
        masking_opts=audio_masking_opts,
        aux_loss_scales=aux_loss_scales,
        loss_name="audio",
    )
