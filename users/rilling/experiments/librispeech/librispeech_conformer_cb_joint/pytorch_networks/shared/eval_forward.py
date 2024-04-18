import os
import torch
import torchaudio
import numpy as np
import subprocess
import multiprocessing as mp
from i6_experiments.users.rossenbach.experiments.jaist_project.pytorch_networks.vocoder.simple_gl.blstm_gl_predictor import Model
from i6_experiments.users.rilling.datasets.corpus import Corpus, Recording, Segment

ENVIRON = os.environ.copy()
ENVIRON["OMP_NUM_THREADS"] = "2"

MAX_WAV_VALUE = 32768.0

def save_ogg(args):
    """
    :param args: wav, path and sr
    """
    wav, path, sr = args
    wav = wav.astype(np.float32) * 32767 / max(0.01, np.max(np.abs(wav)))
    p1 = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "%i" % sr, "-i", "pipe:0", "-c:a", "libvorbis", "-q", "3.0", path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=ENVIRON,
    )
    p1.communicate(input=wav.astype(np.int16).tobytes())
    p1.terminate()

def forward_init_hook_corpus_univnet(run_ctx, **kwargs):
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

    state_dict_g = load_checkpoint(
        "/work/asr3/rossenbach/rilling/vocoder/univnet/glow_finetuning/g_01080000", run_ctx.device
    )
    generator.load_state_dict(state_dict_g["generator"])

    run_ctx.generator = generator

    run_ctx.corpus = Corpus()
    run_ctx.corpus.name = None

    run_ctx.noise_scale = kwargs["noise_scale"]
    run_ctx.length_scale = kwargs["length_scale"]

    run_ctx.pool = mp.Pool(processes=kwargs.get("num_pool_processed", 4))

    import os
    if not os.path.exists("audio_files"):
        os.mkdir("audio_files")


def forward_finish_hook_corpus_univnet(run_ctx, **kwargs):
    run_ctx.corpus.dump("out_corpus.xml.gz")

def forward_step_corpus_univnet(*, model, data, run_ctx, **kwargs):
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B]
    audio_features = data["audio_features"]

    if "xvectors" in data.keys():
        g = data["xvectors"]
    else: 
        g = data["speakers"]  # [B, 1] (sparse)

    tags = data["seq_tag"]

    (log_mels, z_m, z_logs, logdet, z_mask, y_lengths), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
        phonemes,
        phonemes_len,
        g=g,
        gen=True,
        noise_scale=run_ctx.noise_scale,
        length_scale=run_ctx.length_scale,
    )

    noise = torch.randn([1, 64, log_mels.shape[-1]]).to(device=log_mels.device)
    audios = run_ctx.generator.forward(noise, log_mels)
    audios = audios * MAX_WAV_VALUE
    audios = audios.cpu().numpy().astype("int16")

    pool_args = []
    for audio, length, tag in zip(audios, y_lengths, tags):
        corpus_name, recording_name, segment_name = tag.split("/")
        if run_ctx.corpus.name is None:
            run_ctx.corpus.name = corpus_name

        wave = audio[0]
        audio_path = f"audio_files/{tag.replace('/', '_')}.ogg"
        pool_args.append((wave, audio_path, 16000))

        segment = Segment()
        segment.name = segment_name
        segment.start = 0
        segment.end = len(wave) / 16000.0

        recording = Recording()
        recording.name = recording_name
        recording.audio = audio_path
        recording.add_segment(segment)

        run_ctx.corpus.add_recording(recording)

    run_ctx.pool.map(save_ogg, pool_args)


def forward_init_hook_corpus_gl(run_ctx, **kwargs):
    simple_gl_checkpoint = kwargs["gl_net_checkpoint"]
    simple_gl_net_config = kwargs["gl_net_config"]
    n_iter = kwargs.get("gl_iter", 32)
    momentum = kwargs.get("gl_momentum", 0.99)
    run_ctx.noise_scale = kwargs.get("noise_scale", 1.0)

    run_ctx.corpus = Corpus()
    run_ctx.corpus.name = None

    run_ctx.pool = mp.Pool(processes=kwargs.get("num_pool_processes", 4))

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
        n_iter=n_iter,
        win_length=int(0.05 * 16000),
        hop_length=int(0.0125 * 16000),
        power=1.0,
        momentum=momentum,
    )

    run_ctx.noise_scale = kwargs["noise_scale"]
    run_ctx.length_scale = kwargs["length_scale"]

    run_ctx.ddi_initialized = False
    
    import os

    if not os.path.exists("audio_files"):
        os.mkdir("audio_files")


def forward_finish_hook_corpus_gl(run_ctx, **kwargs):
    run_ctx.corpus.dump("out_corpus.xml.gz")


def forward_step_corpus_gl(*, model, data, run_ctx, **kwargs):
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B]

    if "xvectors" in data:
        g = data["xvectors"]
    elif "speakers" in data:
        g = data["speakers"]
    else:
        raise Exception("Missing speaker embedding!")

    tags = data["seq_tag"]

    if not run_ctx.ddi_initialized:
        for f in model.decoder.flows:
            if hasattr(f, "set_ddi"):
                f.set_ddi(
                    False
                )  # This sets initialized to True in the ActNorm Layers, which prevents the fresh initialization of the Layer during forwarding.
        run_ctx.ddi_initialized = True

    (log_mels, z_m, z_logs, logdet, z_mask, y_lengths), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
        phonemes,
        phonemes_len,
        g=g,
        gen=True,
        noise_scale=run_ctx.noise_scale,
        length_scale=run_ctx.length_scale,
    )

    _, linears = run_ctx.gl_model(log_mels.transpose(1, 2), y_lengths)
    linears = linears.transpose(1, 2)

    pool_args = []
    for linear, length, tag in zip(linears, y_lengths, tags):
        corpus_name, recording_name, segment_name = tag.split("/")
        if run_ctx.corpus.name is None:
            run_ctx.corpus.name = corpus_name
        wave = run_ctx.griffin_lim(linear[:, :length]).cpu().numpy()
        audio_path = f"audio_files/{tag.replace('/', '_')}.ogg"
        pool_args.append((wave, audio_path, 16000))

        segment = Segment()
        segment.name = segment_name
        segment.start = 0
        segment.end = len(wave) / 16000.0

        recording = Recording()
        recording.name = recording_name
        recording.audio = audio_path
        recording.add_segment(segment)

        run_ctx.corpus.add_recording(recording)

    run_ctx.pool.map(save_ogg, pool_args)
