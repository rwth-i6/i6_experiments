"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/model/definitions/encoders/wav2vec2.py
(``Wav2Vec2EncoderV1``: construction, ``_zero_mean_unit_var``, ``forward``) and src/speech_llm/sae/build_av_states.py
(``_forward_states`` with ``tap="encoder"``, ``_stats_rows``, ``_per_utt_stats``, ``_RunningStats``, the dump loop).

RETURNN-side code of the layer-15 feature dump (the sisyphus side is :mod:`.features`):

* :func:`get_model` -- the pretrained wav2vec2-large-lv60 exactly as the source's no-SFT
  ``AvStatesJob`` built it inside ``SpeechLmV2`` (``Wav2Vec2EncoderV1(encoder_layer=15,
  trainable=True)``; ``apply_spec_augment=True``; blocks past 15 dropped; conv front end frozen),
  then ``eval()`` + ``requires_grad_(False)``.  No Qwen3 decoder and no adapter are built: the
  encoder tap never touched them.
* :func:`l15_forward_step` -- one utterance per batch (the source's ``_forward_states``): per-utterance
  zero-mean/unit-variance waveform normalisation over the valid samples, ``hidden_states[15]``
  (1024-d @ 50 Hz, pre-final-norm), sliced to ``_get_feat_extract_output_lengths``.
* :func:`L15FeatureCallback` -- writes the float32 states as float16 into ``feats.hdf`` (the source
  pickled ``states.astype(np.float16)``), and from the float32 states the per-utterance
  ``{tag: float32 [2, D]}`` mean/std pickle and the global float32 ``[2, D]`` mean/std over every
  dumped frame in forward order (the source's ``_RunningStats`` over its fit split).  The
  ogg-zip seq tags ``<corpus>/<utt>/<utt>`` are written as the bare utterance id
  (:func:`..data.ogg_zip.utt_id_of_segment`), the tag every banked stream carries.

The waveform arrives from RETURNN's ``OggZipDataset`` with raw features (``[B, T, 1]`` float32, no
peak normalisation, no pre-emphasis; see :mod:`.features`); the source decoded the HF Ogg dataset.

Nothing here is imported at sisyphus graph time; ``torch`` / ``transformers`` / ``returnn`` are
imported inside the functions only.
"""

from __future__ import annotations

__all__ = [
    "AUDIO_KEY",
    "L15_DIM",
    "get_model",
    "l15_forward_step",
    "L15FeatureCallback",
    "stats_rows",
    "per_utt_stats",
    "RunningStats",
]

#: the extern_data / MetaDataset key of the raw 16 kHz waveform (posterior_hmm's ``raw_audio``)
AUDIO_KEY = "raw_audio"
#: wav2vec2-Large-lv60 hidden size
L15_DIM = 1024

_STD_FLOOR = 1e-5  # same floor as build_feats.py


# -------------------------------------------------------------------------------------------------
# statistics (build_av_states.py, numerics unchanged)
# -------------------------------------------------------------------------------------------------
def stats_rows(mean, std):
    """Stack per-dim mean/std into the float32 [2, D] layout of ContinuousFeatsJob's feat_stats.npy."""
    import numpy as np

    return np.stack([np.asarray(mean, dtype=np.float64),
                     np.maximum(np.asarray(std, dtype=np.float64), _STD_FLOOR)], axis=0).astype(np.float32)


def per_utt_stats(states):
    """float32 [2, D] mean/std over one utterance's frames (states: [T, D], T >= 1)."""
    import numpy as np

    x = states.astype(np.float64)
    return stats_rows(x.mean(axis=0), x.std(axis=0))


class RunningStats:
    """Streaming per-dim mean/std (frames arrive utterance-wise)."""

    def __init__(self, dim):
        import numpy as np

        self.n = 0
        self.s = np.zeros(dim, dtype=np.float64)
        self.sq = np.zeros(dim, dtype=np.float64)

    def add(self, x):  # x: [T, D]
        import numpy as np

        x = x.astype(np.float64)
        self.n += x.shape[0]
        self.s += x.sum(axis=0)
        self.sq += (x * x).sum(axis=0)

    def stats(self):
        import numpy as np

        if self.n == 0:
            d = self.s.shape[0]
            return stats_rows(np.zeros(d), np.ones(d))
        mean = self.s / self.n
        var = np.maximum(self.sq / self.n - mean * mean, 0.0)
        return stats_rows(mean, np.sqrt(var))


# -------------------------------------------------------------------------------------------------
# model (Wav2Vec2EncoderV1, frozen encoder tap)
# -------------------------------------------------------------------------------------------------
def _build_l15_module(*, hf_model_dir: str, encoder_layer: int = 15):
    import torch
    from torch import nn
    from transformers import Wav2Vec2Model

    class W2v2LayerTap(nn.Module):
        """wav2vec2-large-lv60 up to ``hidden_states[encoder_layer]``."""

        def __init__(self):
            super().__init__()
            self.model = Wav2Vec2Model.from_pretrained(hf_model_dir)
            self.model.config.apply_spec_augment = True
            num_layers = self.model.config.num_hidden_layers  # 24 for Large
            if not 0 <= encoder_layer <= num_layers:
                raise ValueError(f"encoder_layer {encoder_layer} outside 0..{num_layers} (hidden_states index)")
            self.encoder_layer = encoder_layer
            # Keep encoder_layer+1 blocks: lv60 is do_stable_layer_norm, whose encoder applies a FINAL
            # layer_norm to the last kept block's output; keeping block k pushes that norm onto
            # hidden_states[k+1] (discarded) and leaves hidden_states[k] identical to the full model.
            n_keep = encoder_layer + 1
            if n_keep < self.model.config.num_hidden_layers:
                self.model.encoder.layers = self.model.encoder.layers[:n_keep]
                self.model.config.num_hidden_layers = n_keep
            self.model.feature_extractor._freeze_parameters()
            for name, p in self.model.named_parameters():
                assert p.dtype == torch.float32, f"{name}: {p.dtype}; the source ran the encoder in float32"

        @staticmethod
        def _zero_mean_unit_var(wav, lens):
            """Per-utterance do_normalize over valid samples (== Wav2Vec2FeatureExtractor, batch/pad-safe)."""
            T = wav.shape[1]
            mask = (torch.arange(T, device=wav.device)[None, :] < lens[:, None]).to(wav.dtype)
            n = mask.sum(dim=1).clamp(min=1.0)
            mean = (wav * mask).sum(dim=1) / n
            centered = (wav - mean[:, None]) * mask
            var = (centered**2).sum(dim=1) / n
            return centered / torch.sqrt(var[:, None] + 1e-7)

        def train(self, mode: bool = True):
            """Conv feature-extractor pinned to eval always."""
            super().train(mode)
            self.model.feature_extractor.eval()
            return self

        def forward(self, raw_audio, raw_audio_lens):
            """``[B, T]`` float32 waveform, ``[B]`` lengths -> (``[B, T', 1024]`` states, ``[B]`` lengths)."""
            normed = self._zero_mean_unit_var(raw_audio, raw_audio_lens)
            T = normed.shape[1]
            attention_mask = (torch.arange(T, device=normed.device)[None, :] < raw_audio_lens[:, None]).long()
            out = self.model(normed, attention_mask=attention_mask, output_hidden_states=True)
            states = out.hidden_states[self.encoder_layer]
            out_lens = self.model._get_feat_extract_output_lengths(raw_audio_lens).long()
            return states, out_lens

    module = W2v2LayerTap()
    module.eval()
    module.requires_grad_(False)
    return module


def get_model(*, hf_model_dir: str, encoder_layer: int = 15, **_kwargs):
    """RETURNN ``get_model``: the frozen wav2vec2-large-lv60 layer tap (no checkpoint is loaded over it)."""
    return _build_l15_module(hf_model_dir=hf_model_dir, encoder_layer=encoder_layer)


# -------------------------------------------------------------------------------------------------
# forward step
# -------------------------------------------------------------------------------------------------
def l15_forward_step(*, model, extern_data, audio_key: str = AUDIO_KEY, **_kwargs):
    """RETURNN ``forward_step``: one utterance per batch, mark ``hidden_states[15]`` as ``states``."""
    import torch

    import returnn.frontend as rf
    from returnn.tensor import Dim, Tensor as ReturnnTensor

    audio_ = extern_data[audio_key]
    audio = audio_.raw_tensor
    b_dim = audio_.dims[0]
    lens = audio_.dims[1].dyn_size_ext.raw_tensor.to(device=audio.device, dtype=torch.long)
    # raw-audio datastream: [B, T, 1] -> [B, T]
    assert audio.dim() == 3 and audio.shape[2] == 1, f"expected [B, T, 1] raw audio, got {tuple(audio.shape)}"
    audio = audio[:, :, 0]
    # batch 1, as the source: no padding ever reaches the encoder
    assert audio.shape[0] == 1, f"batch must be exactly one utterance, got {tuple(audio.shape)}"
    assert int(lens[0]) == audio.shape[1], (int(lens[0]), tuple(audio.shape))
    assert audio.dtype == torch.float32, audio.dtype

    with torch.no_grad():
        states, out_lens = model(audio, lens)
    n = int(out_lens[0])
    assert 0 < n <= states.shape[1], (n, tuple(states.shape))
    states = states[:, :n].float()

    t_dim = Dim(
        ReturnnTensor("l15_lens", dims=[b_dim], dtype="int32", raw_tensor=out_lens.to(device="cpu", dtype=torch.int32)),
        name="l15_time",
    )
    f_dim = Dim(int(states.shape[-1]), name="l15_feature")
    rf.get_run_ctx().mark_as_output(
        rf.convert_to_tensor(states, dims=[b_dim, t_dim, f_dim]), "states", dims=[b_dim, t_dim, f_dim]
    )


# -------------------------------------------------------------------------------------------------
# callback
# -------------------------------------------------------------------------------------------------
def L15FeatureCallback(
    *,
    out_hdf_file: str = "feats.hdf",
    out_perutt_file: str = "perutt_stats.pkl",
    out_global_file: str = "global_stats.npy",
    out_stats_file: str = "feats.stats.txt",
    feature_dim: int = L15_DIM,
    expected_num_seqs: int = 0,
    **_kwargs,
):
    """Build the forward callback writing the feature HDF and the two statistics files.

    A factory, not a class: RETURNN asserts the callback is a ``ForwardCallbackIface`` instance and
    calls ``forward_callback()`` when it is callable, so returning an instance from a function keeps
    ``import returnn`` out of module import time.

    :param expected_num_seqs: if > 0, ``finish`` asserts exactly this many utterances were dumped.
    """
    import pickle

    import numpy as np

    from returnn.datasets.hdf import SimpleHDFWriter
    from returnn.forward_iface import ForwardCallbackIface

    from ..data.ogg_zip import utt_id_of_segment

    class _L15FeatureCallback(ForwardCallbackIface):
        def __init__(self):
            self._writer = None
            self._perutt = {}
            self._global = None
            self._lens = []

        def init(self, *args, **kwargs):
            self._writer = SimpleHDFWriter(filename=out_hdf_file, dim=feature_dim, ndim=2)
            self._perutt = {}
            self._global = RunningStats(feature_dim)
            self._lens = []

        def process_seq(self, *, seq_tag, outputs, **kwargs):
            seq_tag = utt_id_of_segment(seq_tag)
            s = np.asarray(outputs["states"].raw_tensor)
            assert s.dtype == np.float32, (seq_tag, s.dtype)
            assert s.ndim == 2 and s.shape[1] == feature_dim, f"{seq_tag}: state shape {s.shape}"
            assert s.shape[0] > 0, f"{seq_tag}: empty state sequence"
            assert seq_tag not in self._perutt, f"duplicate seq_tag {seq_tag!r}"
            x = s.astype(np.float16)
            self._writer.insert_batch(x[None, :, :], [x.shape[0]], [seq_tag])
            self._perutt[seq_tag] = per_utt_stats(s)
            self._global.add(s)
            self._lens.append(int(s.shape[0]))

        def finish(self, **kwargs):
            self._writer.close()
            if expected_num_seqs > 0:
                assert len(self._lens) == expected_num_seqs, (len(self._lens), expected_num_seqs)
            np.save(out_global_file, self._global.stats())
            with open(out_perutt_file, "wb") as fh:
                pickle.dump(self._perutt, fh)
            n, total = len(self._lens), int(sum(self._lens))
            lines = [
                f"wav2vec2-large-lv60 hidden_states[15] (pre-final-norm), D={feature_dim}, 50 Hz, stored float16",
                f"utts = {n}   frames = {total}   mean_frames = {total / max(n, 1):.2f}   "
                f"median_frames = {float(np.median(self._lens)) if self._lens else 0.0:.1f}",
                f"global stats over all {self._global.n} dumped frames (float32 states, forward order)",
            ]
            with open(out_stats_file, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            print("\n".join(lines), flush=True)

    return _L15FeatureCallback()
