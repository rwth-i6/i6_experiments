from dataclasses import dataclass
import torch
from torch import nn
import multiprocessing
import math
import os
import soundfile

from IPython import embed

from returnn.datasets.hdf import SimpleHDFWriter

from . import modules
from . import commons
from . import attentions
from .monotonic_align import maximum_path

from .feature_extraction import DbMelFeatureExtraction
from ..feature_config import DbMelFeatureExtractionConfig

class Model(nn.Module):
    """
    Flow-based TTS model based on GlowTTS Structure
    Following the definition from https://arxiv.org/abs/2005.11129
    and code from https://github.com/jaywalnut310/glow-tts
    """

    def __init__(
        self,
        fe_config,
        **kwargs
    ):
        """_summary_

        Args:
            n_vocab (int): vocabulary size
            hidden_channels (int): Number of hidden channels in encoder
            filter_channels (int): Number of filter channels in encoder
            filter_channels_dp (int): Number of filter channels in decoder
            out_channels (int): Number of channels in the output
            kernel_size (int, optional): Size of kernels in the encoder. Defaults to 3.
            n_heads (int, optional): Number of heads in the Multi-Head Attention of the encoder. Defaults to 2.
            n_layers_enc (int, optional): Number of layers in the encoder. Defaults to 6.
            p_dropout (_type_, optional): Dropout probability in the encoder. Defaults to 0..
            n_blocks_dec (int, optional): Number of coupling blocks in the decoder. Defaults to 12.
            kernel_size_dec (int, optional): Kernel size in the decoder. Defaults to 5.
            dilation_rate (int, optional): Dilation rate for CNNs of coupling blocks in decoder. Defaults to 5.
            n_block_layers (int, optional): Number of layers in the CNN of the coupling blocks in decoder. Defaults to 4.
            p_dropout_dec (_type_, optional): Dropout probability in the decoder. Defaults to 0..
            n_speakers (int, optional): Number of speakers. Defaults to 0.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            window_size (int, optional): Window size  in Multi-Head Self-Attention for encoder. Defaults to None.
            block_length (_type_, optional): Block length for optional block masking in Multi-Head Attention for encoder. Defaults to None.
            mean_only (bool, optional): Boolean to only project text encodings to mean values instead of mean and std. Defaults to False.
            hidden_channels_enc (int, optional): Number of hidden channels in encoder. Defaults to hidden_channels.
            hidden_channels_dec (_type_, optional): Number of hidden channels in decodder. Defaults to hidden_channels.
            prenet (bool, optional): Boolean to add ConvReluNorm prenet before encoder . Defaults to False.
        """
        super().__init__()

        fe_config = DbMelFeatureExtractionConfig.from_dict(fe_config)
        self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

    def forward(
        self, x, x_lengths, raw_audio=None, raw_audio_lengths=None, g=None, gen=False, noise_scale=1.0, length_scale=1.0
    ):
        
        with torch.no_grad():
            squeezed_audio = torch.squeeze(raw_audio)
            y, y_lengths = self.feature_extraction(
                squeezed_audio, raw_audio_lengths
            )  # [B, T, F]
            y = y.transpose(1, 2)  # [B, F, T]

        return (y, None, None, None, None, y_lengths), (None, None, None), (None, None, None)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    tags = data["seq_tag"]
    audio_features = data["audio_features"]  # [B, T, F]
    # audio_features = audio_features.transpose(1, 2) # [B, F, T] necessary because glowTTS expects the channels to be in the 2nd dimension
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)
    tags = list(np.array(tags)[indices.detach().cpu().numpy()])

    # print(f"phoneme shape: {phonemes.shape}")
    # print(f"phoneme length: {phonemes_len}")
    # print(f"audio_feature shape: {audio_features.shape}")
    # print(f"audio_feature length: {audio_features_len}")
    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
        phonemes, phonemes_len, audio_features, audio_features_len, speaker_labels
    )
    # embed()

    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    l_dp = commons.duration_loss(logw, logw_, phonemes_len)

    run_ctx.mark_as_loss(name="mle", loss=l_mle)
    run_ctx.mark_as_loss(name="dp", loss=l_dp)


############# FORWARD STUFF ################
import numpy as np

def forward_init_hook_spectrograms(run_ctx, **kwargs):
    run_ctx.hdf_writer = SimpleHDFWriter("output.hdf", dim=80, ndim=2)
    run_ctx.pool = multiprocessing.Pool(8)

def forward_finish_hook_spectrograms(run_ctx, **kwargs):
    run_ctx.hdf_writer.close()


def forward_init_hook(run_ctx, **kwargs):
    import json
    import utils
    from utils import AttrDict
    from inference import load_checkpoint
    from generator import UnivNet as Generator
    import numpy as np

    with open("/u/lukas.rilling/experiments/glow_tts_asr_v2/config_univ.json") as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(run_ctx.device)

    state_dict_g = load_checkpoint("/work/asr3/rossenbach/rilling/vocoder/univnet/glow_finetuning/g_01080000", run_ctx.device)
    generator.load_state_dict(state_dict_g["generator"])

    run_ctx.generator = generator


def forward_finish_hook(run_ctx, **kwargs):
    pass


MAX_WAV_VALUE = 32768.0


def forward_step(*, model: Model, data, run_ctx, **kwargs):
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)
    audio_features = data["audio_features"]
    audio_features_len = data["audio_features:size1"]

    tags = data["seq_tag"]

    (log_mels, z_m, z_logs, logdet, z_mask, y_lengths), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
        phonemes,
        phonemes_len,
        audio_features,
        audio_features_len,
        g=speaker_labels,
        gen=True
    )

    noise = torch.randn([1, 64, log_mels.shape[-1]]).to(device=log_mels.device)
    audios = run_ctx.generator.forward(noise, log_mels)
    audios = audios * MAX_WAV_VALUE
    audios = audios.cpu().numpy().astype("int16")

    # mels_gt = audio_features.transpose(1, 2)
    # noise = torch.randn([1, 64, mels_gt.shape[-1]]).to(device=mels_gt.device)
    # audios_gt = run_ctx.generator.forward(noise, mels_gt)
    # audios_gt = audios_gt * MAX_WAV_VALUE
    # audios_gt = audios_gt.cpu().numpy().astype("int16")

    if not os.path.exists("/var/tmp/lukas.rilling/"):
        os.makedirs("/var/tmp/lukas.rilling/")
    if not os.path.exists("/var/tmp/lukas.rilling/out"):
        os.makedirs("/var/tmp/lukas.rilling/out/", exist_ok=True)
    for audio, tag in zip(audios, tags):
        soundfile.write(f"/var/tmp/lukas.rilling/out/" + tag.replace("/", "_") + ".wav", audio[0], 16000)
        # soundfile.write(f"/var/tmp/lukas.rilling/out/" + tag.replace("/", "_") + "_gt.wav", audio_gt[0], 16000)


def forward_step_spectrograms(*, model: Model, data, run_ctx, **kwargs):
    tags = data["seq_tag"]
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features = audio_features.transpose(
        1, 2
    )  # [B, F, T] necessary because glowTTS expects the channels to be in the 2nd dimension
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)
    tags = list(np.array(tags)[indices.detach().cpu().numpy()])

    (y, z_m, z_logs, logdet, z_mask, y_lengths), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
        phonemes, phonemes_len, audio_features, audio_features_len, g=speaker_labels, gen=True
    ) 
    spectograms = y.transpose(2, 1).detach().cpu().numpy()  # [B, T, F]

    run_ctx.hdf_writer.insert_batch(spectograms, y_lengths.detach().cpu().numpy(), tags)


def forward_step_durations(*, model: Model, data, run_ctx, **kwargs):
    """Forward Step to output durations in HDF file
    Currently unused due to the name. Only "forward_step" is used in ReturnnForwardJob.
    Rename to use it as the forward step function.

    :param Model model: _description_
    :param _type_ data: _description_
    :param _type_ run_ctx: _description_
    """
    model.train()
    tags = data["seq_tag"]
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features = audio_features.transpose(
        1, 2
    )  # [B, F, T] necessary because glowTTS expects the channels to be in the 2nd dimension
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)
    tags = list(np.array(tags)[indices.detach().cpu().numpy()])

    # embed()
    (z, z_m, z_logs, logdet, z_mask, y_lengths), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
        phonemes, phonemes_len, audio_features, audio_features_len, speaker_labels, gen=False
    )
    # embed()
    numpy_logprobs = logw_.detach().cpu().numpy()

    durations_with_pad = np.round(np.exp(numpy_logprobs) * x_mask.detach().cpu().numpy())
    durations = durations_with_pad.squeeze(1)

    for tag, duration, feat_len, phon_len in zip(tags, durations, audio_features_len, phonemes_len):
        d = duration[duration > 0]
        # total_sum = np.sum(duration)
        # assert total_sum == feat_len
        assert len(d) == phon_len
        run_ctx.hdf_writer.insert_batch(np.asarray([d]), [len(d)], [tag])
