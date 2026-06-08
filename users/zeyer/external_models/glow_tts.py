"""
GlowTTS as an external pretrained TTS model: phonemes -> log-mel (in DbMel feature space).

The GlowTTS model is from Nick (rossenbach); originally wrapped (together with a simple-GL
vocoder + Griffin-Lim) in ``denoising_lm_2024.sis_recipe.tts_model.TtsModel`` to synthesize
waveforms. Here we use only the GlowTTS part to produce the **log-mel directly** (no GL net,
no Griffin-Lim), so it can be fed straight into an ASR encoder whose feature front-end is the
same DbMel space (Slaney mel, f_max 7600, 12.5ms hop, fixed norm).

Checkpoint / lexicon / phoneme-vocab are referenced by i6_core hash via ``generic_job_output``
(so the artifacts are reused wherever those job dirs are available, e.g. imported into the setup work).
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
from functools import cache

from sisyphus import tk
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim

from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output

__all__ = [
    "GlowTtsLogMel",
    "get_glow_tts_model_config",
    "get_glow_tts_preload_from_files",
    "get_glow_tts_checkpoint",
    "get_glow_tts_lexicon",
    "get_glow_tts_phoneme_vocab",
    "get_glow_tts_phoneme_vocab_size",
    "get_glow_tts_phoneme_vocab_special_symbols",
    "get_glow_tts_phoneme_dataset_dict",
    "get_glow_tts_phoneme_extern_data",
]


@cache
def get_glow_tts_checkpoint() -> tk.Path:
    """GlowTTS model checkpoint (from Nick)."""
    return generic_job_output("i6_core/returnn/training/ReturnnTrainingJob.Jqv1rStK7xWH/output/models/epoch.400.pt")


@cache
def get_glow_tts_lexicon() -> tk.Path:
    """Phoneme lexicon (from Nick), matching the GlowTTS phoneme vocab."""
    return generic_job_output("i6_core/lexicon/modification/MergeLexiconJob.P8go21pxx40e/output/lexicon.xml.gz")


@cache
def get_glow_tts_phoneme_vocab() -> tk.Path:
    """GlowTTS phoneme vocab (44 symbols; special symbols at the end, see special_symbols below)."""
    return generic_job_output(
        "i6_core/returnn/vocabulary/ReturnnVocabFromPhonemeInventory.z2RlZd9Y0jWQ/output/vocab.pkl"
    )


def get_glow_tts_phoneme_vocab_size() -> int:
    """:return: size of the phoneme vocab, matching :func:`get_glow_tts_phoneme_vocab`."""
    return 44


def get_glow_tts_phoneme_vocab_special_symbols() -> Dict[str, Any]:
    """Special symbols in the phoneme vocab (matching :func:`get_glow_tts_phoneme_vocab`)."""
    return {"unknown_label": "[UNKNOWN]", "bos_label": "[start]", "eos_label": "[end]"}


def get_glow_tts_model_config() -> Dict[str, Any]:
    """GlowTTS model config (from Nick).

    The ``feature_extraction_config`` mel params here define the log-mel space the model emits,
    and must match the ASR encoder's DbMel front-end for zero-transform feeding.
    """
    return {
        "feature_extraction_config": {
            "sample_rate": 16000,
            "win_size": 0.05,
            "hop_size": 0.0125,
            "f_min": 0,
            "f_max": 7600,
            "min_amp": 1e-10,
            "num_filters": 80,
            "center": True,
            "norm": (-72.83881497383118, 37.73079669103133),
        },
        "encoder_config": {
            "num_layers": 6,
            "vocab_size": 44,
            "basic_dim": 256,
            "conv_dim": 1024,
            "conv_kernel_size": 3,
            "dropout": 0.1,
            "mhsa_config": {
                "input_dim": 256,
                "num_att_heads": 2,
                "dropout": 0.1,
                "att_weights_dropout": 0.1,
                "window_size": 4,
                "heads_share": True,
                "block_length": None,
                "proximal_bias": False,
                "proximal_init": False,
            },
            "prenet_config": {
                "input_embedding_size": 256,
                "hidden_dimension": 256,
                "kernel_size": 5,
                "output_dimension": 256,
                "num_layers": 3,
                "dropout": 0.5,
            },
        },
        "duration_predictor_config": {"num_convs": 2, "hidden_dim": 384, "kernel_size": 3, "dropout": 0.1},
        "flow_decoder_config": {
            "target_channels": 80,
            "hidden_channels": 256,
            "kernel_size": 5,
            "dilation_rate": 1,
            "num_blocks": 12,
            "num_layers_per_block": 4,
            "num_splits": 4,
            "num_squeeze": 2,
            "dropout": 0.05,
            "use_sigmoid_scale": False,
        },
        "num_speakers": 1172,
        "speaker_embedding_size": 256,
        "mean_only": True,
    }


def get_glow_tts_preload_from_files(prefix: str = "tts.glow_tts_model.") -> Dict[str, Dict[str, Any]]:
    """``preload_from_files`` entry to load the (frozen) GlowTTS params.

    :param prefix: param-name prefix of the GlowTTS submodule in the combined model.
        Default matches attaching ``model.tts = GlowTtsLogMel(...)`` (whose submodule is ``glow_tts_model``).
    """
    return {"glow_tts": {"prefix": prefix, "filename": get_glow_tts_checkpoint()}}


@cache
def get_glow_tts_gl_checkpoint() -> tk.Path:
    """Simple-GL (BLSTM linear-spec predictor) checkpoint (from Nick), for the Griffin-Lim waveform path."""
    return generic_job_output("i6_core/returnn/training/ReturnnTrainingJob.H9EByABag8UN/output/models/epoch.050.pt")


def get_glow_tts_gl_net_config() -> Dict[str, Any]:
    """Config for the BLSTM GL-net (log-mel -> linear spectrogram), matching the GlowTTS DbMel space."""
    return {
        "hidden_size": 512,
        "feature_extraction_config": {
            "sample_rate": 16000,
            "win_size": 0.05,
            "hop_size": 0.0125,
            "f_min": 0,
            "f_max": 7600,
            "min_amp": 1e-10,
            "num_filters": 80,
            "center": True,
            "norm": (-72.83881497383118, 37.73079669103133),
        },
    }


def get_glow_tts_gl_preload_from_files(prefix: str = "tts.gl_model.") -> Dict[str, Dict[str, Any]]:
    """``preload_from_files`` entry for the (frozen) GL-net params, for the waveform path."""
    return {"glow_tts_gl": {"prefix": prefix, "filename": get_glow_tts_gl_checkpoint()}}


def get_glow_tts_phoneme_dataset_dict(
    *,
    corpus_text: tk.Path,
    seq_list_file: Optional[tk.Path] = None,
    fixed_random_seed: Optional[int] = None,
    train: bool,
) -> Dict[str, Any]:
    """LmDataset dict that turns text into GlowTTS phoneme sequences (via the lexicon ``phone_info``)."""
    return {
        "class": "LmDataset",
        "corpus_file": corpus_text,
        "seq_list_file": seq_list_file,
        "use_cache_manager": True,
        "skip_empty_lines": False,
        "dtype": "int32",
        "seq_end_symbol": None,
        "unknown_symbol": None,
        "fixed_random_seed": fixed_random_seed,
        "phone_info": get_glow_tts_phone_info(train=train),
    }


def get_glow_tts_phone_info(*, train: bool) -> Dict[str, Any]:
    """``phone_info`` for RETURNN's ``PhoneSeqGenerator``: text -> GlowTTS phonemes via the lexicon."""
    return {
        "lexicon_file": get_glow_tts_lexicon(),
        "phoneme_vocab_file": get_glow_tts_phoneme_vocab(),
        "allo_num_states": 1,
        "add_silence_beginning": 0.01 if train else 0.0,
        "add_silence_between_words": 0.95 if train else 1.0,
        "add_silence_end": 0.01 if train else 0.0,
        "repetition": 0.01 if train else 0.0,
        "silence_repetition": 0.01 if train else 0.0,
        "silence_lemma_orth": "[space]",
        "extra_begin_lemma": {"phons": [{"phon": "[start]"}]},
        "extra_end_lemma": {"phons": [{"phon": "[end]"}]},
    }


def get_glow_tts_phoneme_extern_data() -> Dict[str, Any]:
    """extern_data entry for the GlowTTS phoneme input (matching :func:`get_glow_tts_phoneme_dataset_dict`)."""
    return {
        "dim_tags": [batch_dim, Dim(None, name="phone_seq", kind=Dim.Types.Spatial)],
        "sparse_dim": Dim(get_glow_tts_phoneme_vocab_size(), name="phone_vocab"),
        "vocab": {
            "class": "Vocabulary",
            "vocab_file": get_glow_tts_phoneme_vocab(),
            **get_glow_tts_phoneme_vocab_special_symbols(),
        },
    }


class GlowTtsLogMel(rf.Module):
    """Frozen GlowTTS used as a phonemes -> log-mel generator (no vocoder / Griffin-Lim).

    ``__call__(phonemes, spatial_dim) -> (log_mel [B, T, F], time_dim)``, where the log-mel is in
    the GlowTTS DbMel space (= the ASR encoder's DbMel front-end), so it feeds straight into the encoder.
    Generation uses random speaker + sampled ``noise_scale`` / ``length_scale`` (the latter scales all
    durations -> the cheap-training lever).
    """

    def __init__(
        self,
        *,
        phoneme_vocab_dim: Dim,
        out_dim: Optional[Dim] = None,
        glow_tts_model_config: Optional[Dict[str, Any]] = None,
        glow_tts_noise_scale_range: Tuple[float, float] = (0.3, 0.9),
        glow_tts_length_scale_range: Tuple[float, float] = (0.7, 1.1),
        return_waveform: bool = False,
        gl_net_config: Optional[Dict[str, Any]] = None,
        gl_iter: int = 32,
        gl_momentum: float = 0.99,
    ):
        super().__init__()
        from i6_experiments.users.zeyer.experiments.nick_ctc_rnnt_standalone_2024.pytorch_networks.glow_tts.glow_tts_v1 import (
            Model as GlowTtsModel,
        )

        if glow_tts_model_config is None:
            glow_tts_model_config = get_glow_tts_model_config()
        self.phoneme_vocab_dim = phoneme_vocab_dim
        self.glow_tts_noise_scale_range = glow_tts_noise_scale_range
        self.glow_tts_length_scale_range = glow_tts_length_scale_range
        self.glow_tts_model = GlowTtsModel(config=glow_tts_model_config)
        # log-mel feature dim emitted by the flow decoder. Pass out_dim to reuse the ASR encoder's in_dim
        # (same DbMel space), so the log-mel feeds straight into the encoder without a dim rename.
        _target_channels = glow_tts_model_config["flow_decoder_config"]["target_channels"]
        self.out_dim = out_dim if out_dim is not None else Dim(_target_channels, name="logmel")
        # Optional waveform mode: GlowTTS log-mel -> BLSTM GL-net (linear spec) -> Griffin-Lim -> waveform,
        # so the synthetic audio re-enters the ASR through its OWN feature front-end (like the offline TTS
        # pipeline) instead of feeding the GlowTTS log-mel directly. Tests whether the Griffin-Lim round-trip's
        # realistic distortion transfers better to real audio.
        self.return_waveform = return_waveform
        self.gl_model = None
        self.griffin_lim = None
        if return_waveform:
            import torchaudio
            from i6_experiments.users.zeyer.experiments.nick_ctc_rnnt_standalone_2024.pytorch_networks.vocoder.simple_gl.blstm_gl_predictor import (
                Model as BlstmGlPredictorModel,
            )

            if gl_net_config is None:
                gl_net_config = get_glow_tts_gl_net_config()
            self.gl_model = BlstmGlPredictorModel(config=gl_net_config)
            self.griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=800,
                n_iter=gl_iter,
                win_length=int(0.05 * 16000),
                hop_length=int(0.0125 * 16000),
                power=1.0,
                momentum=gl_momentum,
            )

    def __call__(self, phonemes: Tensor, *, spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        import torch

        self.glow_tts_model.eval()  # frozen TTS: force eval (RETURNN sets the parent model to train() each step)
        assert phonemes.sparse_dim.dimension == self.phoneme_vocab_dim.dimension, (
            f"phoneme vocab size mismatch: {phonemes.sparse_dim} vs {self.phoneme_vocab_dim}"
        )
        batch_dims = phonemes.remaining_dims(spatial_dim)
        assert batch_dims == [batch_dim], f"only single batch dim supported, got {batch_dims}"

        phonemes_pt = phonemes.copy_compatible_to_dims_raw([batch_dim, spatial_dim])  # [B, T_phon] sparse
        phon_lens_pt = spatial_dim.get_size_tensor(device=phonemes.device).copy_compatible_to_dims_raw(
            [batch_dim]
        )  # [B]
        bs = phonemes_pt.size(0)
        dev = phonemes_pt.device

        speaker_labels = torch.randint(0, self.glow_tts_model.num_speakers, (bs, 1), device=dev)  # [B, 1]
        nlo, nhi = self.glow_tts_noise_scale_range
        llo, lhi = self.glow_tts_length_scale_range
        noise_scale = nlo + torch.rand((bs, 1, 1), device=dev) * (nhi - nlo)  # [B, 1, 1]
        length_scale = llo + torch.rand((bs, 1, 1), device=dev) * (lhi - llo)  # [B, 1, 1]

        # GlowTTS gen uses a flow inverse; force float32 (bf16 autocast can be unstable / unsupported here).
        with torch.autocast(device_type=dev.type, enabled=False):
            (log_mels, _z_m, _z_logs, _logdet, _z_mask, y_lengths), _, _ = self.glow_tts_model(
                phonemes_pt,
                phon_lens_pt,
                g=speaker_labels,
                gen=True,
                noise_scale=noise_scale,
                length_scale=length_scale,
            )
        # log_mels: [B, F_logmel, T_freq] (flow-decoder output, DbMel space)
        if self.return_waveform:
            self.gl_model.eval()
            with torch.autocast(device_type=dev.type, enabled=False):
                _, linears = self.gl_model(log_mels.transpose(1, 2), y_lengths)  # [B, T_freq, F_freq]
                linears = linears.transpose(1, 2)  # [B, F_freq, T_freq]
                wave = self.griffin_lim(linears)  # [B, T_wave], 16kHz
            wave_lens = ((y_lengths - 1) * self.griffin_lim.hop_length).to(torch.int32)  # [B]
            wave_lens_rf = rf.convert_to_tensor(wave_lens.cpu(), dims=[batch_dim])
            wave_spatial_dim = Dim(wave_lens_rf, name="glowtts_wave")
            wave_rf = rf.convert_to_tensor(wave, dims=[batch_dim, wave_spatial_dim])
            return wave_rf, wave_spatial_dim

        log_mels = log_mels.transpose(1, 2).contiguous()  # [B, T_freq, F_logmel]
        feat_lens = y_lengths.to(torch.int32)  # [B]
        feat_lens_rf = rf.convert_to_tensor(feat_lens.cpu(), dims=[batch_dim])
        out_spatial_dim = Dim(feat_lens_rf, name="glowtts_time")
        log_mels_rf = rf.convert_to_tensor(log_mels, dims=[batch_dim, out_spatial_dim, self.out_dim])
        log_mels_rf.feature_dim = self.out_dim
        return log_mels_rf, out_spatial_dim
