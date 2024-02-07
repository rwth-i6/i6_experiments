import numpy as np
import torch
import torchaudio
from ...vocoder.simple_gl.blstm_gl_predictor import Model


def forward_init_hook(run_ctx, **kwargs):
    run_ctx.norm_mean = kwargs["norm_mean"]
    run_ctx.norm_std = kwargs["norm_std_dev"]
    simple_gl_checkpoint = kwargs["gl_net_checkpoint"]
    simple_gl_net_config = kwargs["gl_net_config"]

    assert isinstance(simple_gl_net_config, dict)

    run_ctx.gl_model = Model(config=simple_gl_net_config)
    checkpoint_state = torch.load(
        simple_gl_checkpoint,
        map_location=run_ctx.device,
    )
    run_ctx.gl_model.load_state_dict(checkpoint_state["model"])

    num_freq = 800
    run_ctx.griffin_lim = torchaudio.transforms.GriffinLim(
        num_freq,
        n_iter=32,
        win_length=int(0.05 * 16000),
        hop_length=int(0.0125 * 16000),
        power=1.0,
        momentum=0.99,
    )

    import os
    if not os.path.exists("audio_files"):
        os.mkdir("audio_files")


def forward_finish_hook(run_ctx, **kwargs):
    pass


MAX_WAV_VALUE = 32768.0



def save_wav(wav, path, sr, peak_normalization=True):
    from scipy.io import wavfile
    if peak_normalization:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    else:
        wav *= 32767
    wavfile.write(path, sr, wav.astype(np.int16))



def forward_step(*, model: Model, data, run_ctx, **kwargs):
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)

    tags = data["seq_tag"]

    log_mels, masked_durations = model(
        phonemes=phonemes,
        phonemes_len=phonemes_len,
        speaker_labels=speaker_labels,
        target_durations=None,
    )


    lengths = torch.sum(masked_durations, dim=1).to(torch.int)

    _, linears = run_ctx.gl_model(log_mels, lengths)

    linears = linears.transpose(1, 2)
    log_mels = log_mels.transpose(1, 2)

    from matplotlib import pyplot as plt

    for linear, log_mel, length, tag in zip(linears, log_mels, lengths, tags):
        wave = run_ctx.griffin_lim(linear[:, :length]).cpu().numpy()
        save_wav(wave, f"audio_files/{tag.replace('/', '_')}.wav", 16000, peak_normalization=True)
        plt.imshow(log_mel[:,:length])
        plt.gca().invert_yaxis()
        plt.savefig(f"audio_files/{tag.replace('/', '_')}.png")