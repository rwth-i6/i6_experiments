"""
Pilots: can text injection work with 0 h of paired audio?

The 0 h run of the LS audio ladder (``{abl}-audio0-textP38``, zeyer exp2026_05_28_tts_encoder_fzj) collapses on real
speech (124 / 124): trained only on the pseudo-speech encoder's output, i.e. 44 exact phone-mean log-mel vectors
and the linear glides between them, it learns a lookup that real frames never match.
Each pilot is that exact run (same config, same LR schedule) stopped after a few subepochs, with ONE change to the
text branch's pseudo features (or a combination), and is compared against the audio0 run's own dev curve
(real LibriSpeech dev audio, CE / FER per subepoch):

- ``rand``: within-phone variability and speaker/channel randomization
  (frame noise with the measured pooled within-phone std, per-utterance offset with the measured between-utterance
  std, gain, spectral tilt, frequency warp).
- ``rand-consist``: ``rand`` plus a consistency loss between two randomizations of the same rendering
  (same durations, so frames align) at the outputs of encoder blocks 3 and 4.
- ``pitch``: a harmonic comb at a random per-utterance F0 on voiced frames.
- ``realstats``: pseudo features moment-matched per batch to the real-audio channel mean / std,
  so the feature BatchNorm learns real-audio statistics.
- ``all``: ``rand-consist`` + ``pitch`` + ``realstats``.

The statistics (within-phone std, between-utterance std, real channel mean / std) come from
:class:`ComputeMfaPhoneLogMelVarianceJob`, over a subset of the same MFA-aligned LibriSpeech train audio that the
frozen mean table itself is computed from (no paired data used by the ASR).
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Optional, Sequence

from sisyphus import Job, Task, tk

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

PHONEMES_DATA_KEY = "phonemes"

VOICED = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW B D G V DH Z ZH JH M N NG L R W Y".split())
UNVOICED = set("P T K F TH S SH CH HH".split())


class ComputeMfaPhoneLogMelVarianceJob(Job):
    """
    Per-phone mean / std of the 100 Hz log-mel over a subset of the MFA-aligned LibriSpeech train audio
    (every ``stride``-th utterance of each split), with exactly the front-end and frame-to-phone assignment of
    zeyer's ``ComputeMfaPhoneMeanLogMelJob``. Additionally:

    - ``pooled_within_std`` [F]: sqrt of the frame-weighted mean within-phone variance over the speech phones;
    - ``speaker_std`` [F]: std over utterances of (utterance mean frame - mean of its frames' phone means),
      i.e. the per-utterance offset that the phone content does not explain (speaker, channel, level);
    - ``global_mean`` / ``global_std`` [F]: channel statistics over all frames.
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        returnn_root: tk.Path,
        phoneme_vocab: tk.Path,
        splits: Sequence[str] = ("train_clean_100", "train_clean_360", "train_other_500"),
        stride: int = 50,
        sample_rate: int = 16_000,
        window_len: float = 0.025,
        step_len: float = 0.010,
        num_filters: int = 80,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.returnn_root = returnn_root
        self.phoneme_vocab = phoneme_vocab
        self.splits = tuple(splits)
        self.stride = stride
        self.sample_rate = sample_rate
        self.window_len = window_len
        self.step_len = step_len
        self.num_filters = num_filters
        self.rqmt = {"cpu": 4, "mem": 16, "time": 6}
        self.out_stats_npz = self.output_path("logmel_stats.npz")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        import sys
        import json
        import re
        import time

        sys.path.insert(0, self.returnn_root.get_path())
        import numpy
        import torch
        from datasets import load_from_disk
        import returnn.frontend as rf_
        from returnn.tensor import Tensor as T, Dim as D, batch_dim
        from returnn.datasets.util.vocabulary import Vocabulary

        rf_.select_backend_torch()
        batch_dim.dyn_size_ext = rf_.convert_to_tensor(torch.tensor(1, dtype=torch.int32), dims=[])
        vocab = Vocabulary(self.phoneme_vocab.get_path(), unknown_label="[UNKNOWN]")
        labels = vocab.labels
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        silence_idx, unknown_idx = label_to_idx["[space]"], label_to_idx["[UNKNOWN]"]
        n_lab, dim_f = len(labels), self.num_filters
        out_dim = D(dim_f, name="mel")

        def log_mel(audio_np):
            peak = numpy.max(numpy.abs(audio_np))
            if peak != 0.0:
                audio_np = audio_np / peak
            raw = torch.tensor(audio_np[None, :], dtype=torch.float32)
            time_dim = D(int(raw.shape[1]), name="time")
            src = T("audio", dims=[batch_dim, time_dim], dtype="float32", raw_tensor=raw)
            feats, feats_dim = rf_.audio.log_mel_filterbank_from_raw(
                src,
                in_spatial_dim=time_dim,
                out_dim=out_dim,
                sampling_rate=self.sample_rate,
                window_len=self.window_len,
                step_len=self.step_len,
            )
            return feats.copy_compatible_to_dims_raw([batch_dim, feats_dim, out_dim])[0].numpy().astype(numpy.float64)

        sums = numpy.zeros((n_lab, dim_f))
        sqs = numpy.zeros((n_lab, dim_f))
        counts = numpy.zeros((n_lab,), dtype=numpy.int64)
        utt_means, utt_hists = [], []
        ds_all = load_from_disk(self.dataset_dir.get_path())
        t0 = time.time()
        n_total = sum(len(range(0, len(ds_all[s]), self.stride)) for s in self.splits)
        print(f"start: {n_total} utterances (stride {self.stride}) from {self.splits}", flush=True)
        n = 0
        for split in self.splits:
            ds = ds_all[split]
            for ex in ds.select(range(0, len(ds), self.stride)):
                audio_np = numpy.asarray(ex["audio"]["array"], dtype=numpy.float32)
                feats = log_mel(audio_np)
                t_centers = numpy.arange(feats.shape[0]) * self.step_len + self.window_len / 2
                frame_label = numpy.full((feats.shape[0],), silence_idx, dtype=numpy.int64)
                for ph in ex["phonemes"]:
                    base = re.sub(r"\d+$", "", ph["phoneme"])
                    idx = label_to_idx.get(base, unknown_idx if base == "spn" else None)
                    if idx is None:
                        continue
                    frame_label[(t_centers >= ph["start"]) & (t_centers < ph["end"])] = idx
                numpy.add.at(sums, frame_label, feats)
                numpy.add.at(sqs, frame_label, feats**2)
                numpy.add.at(counts, frame_label, 1)
                utt_means.append(feats.mean(axis=0))
                utt_hists.append(numpy.bincount(frame_label, minlength=n_lab))
                n += 1
                if n % 500 == 0:
                    print(f"{n}/{n_total} utts, {time.time() - t0:.0f}s", flush=True)

        c = numpy.maximum(counts, 1)[:, None]
        means = sums / c
        var = numpy.maximum(sqs / c - means**2, 0.0)
        speech = numpy.array([lab.split(".")[0] in VOICED | UNVOICED for lab in labels])
        w = counts * speech
        pooled_within_std = numpy.sqrt((var * w[:, None]).sum(0) / w.sum())
        hist = numpy.stack(utt_hists).astype(numpy.float64)
        expected = (hist @ means) / hist.sum(1, keepdims=True)
        speaker_std = (numpy.stack(utt_means) - expected).std(axis=0)
        tot = counts.sum()
        global_mean = sums.sum(0) / tot
        global_std = numpy.sqrt(sqs.sum(0) / tot - global_mean**2)
        numpy.savez(
            self.out_stats_npz.get_path(),
            labels=numpy.array(labels, dtype=object),
            means=means.astype(numpy.float32),
            stds=numpy.sqrt(var).astype(numpy.float32),
            counts=counts,
            pooled_within_std=pooled_within_std.astype(numpy.float32),
            speaker_std=speaker_std.astype(numpy.float32),
            global_mean=global_mean.astype(numpy.float32),
            global_std=global_std.astype(numpy.float32),
        )
        summary = {
            "n_utts": n,
            "n_frames": int(tot),
            "stride": self.stride,
            "splits": list(self.splits),
            "mean_pooled_within_std": float(pooled_within_std.mean()),
            "mean_speaker_std": float(speaker_std.mean()),
            "mean_global_std": float(global_std.mean()),
        }
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(summary, f, indent=2)
        print("done", summary, flush=True)


@functools.cache
def get_logmel_variance_stats() -> ComputeMfaPhoneLogMelVarianceJob:
    """:return: the statistics job (subset of the same MFA-aligned LS train audio as the mean table)"""
    from i6_experiments.users.zeyer.datasets.hf_librispeech_mfa_alignments import get_librispeech_mfa_alignments_dir
    from i6_experiments.users.zeyer.external_models.glow_tts import get_glow_tts_phoneme_vocab
    from i6_experiments.users.zeyer import tools_paths

    job = ComputeMfaPhoneLogMelVarianceJob(
        dataset_dir=get_librispeech_mfa_alignments_dir(),
        returnn_root=tools_paths.get_returnn_root(),
        phoneme_vocab=get_glow_tts_phoneme_vocab(),
    )
    job.add_alias("exp2026_09_16_fzj_dlm/pseudo0h/mfa_phone_logmel_variance")
    tk.register_output("exp2026_09_16_fzj_dlm/pseudo0h/mfa_phone_logmel_variance.json", job.out_summary)
    return job


# ---------------------------------------------------------------- augmentation (train time, text branch only)


@functools.cache
def _host_constants(stats_file: str, table_file: str, num_mel: int) -> Dict[str, Any]:
    """numpy constants, read once per process (host only, safe inside a CUDA-graph capture)"""
    import numpy

    st = numpy.load(stats_file, allow_pickle=True)
    tab = numpy.load(table_file, allow_pickle=True)
    means = tab["means"].astype(numpy.float64)
    labs = [str(x).split(".")[0] for x in tab["labels"]]
    low = means[:, :20].mean(1) - means[:, 50:].mean(1)
    energy = means.mean(1)
    voiced = [i for i, lab in enumerate(labs) if lab in VOICED]
    unvoiced = [i for i, lab in enumerate(labs) if lab in UNVOICED]
    speech = voiced + unvoiced
    sil = labs.index("[space]")
    # RETURNN's exact mel matrix [257, num_mel], in numpy (this runs inside the traced train step:
    # no torch here, a FakeTensor has no .numpy())
    from returnn.frontend.audio.mel import _mel_filter_bank_matrix_np

    melmat = _mel_filter_bank_matrix_np(f_min=0, f_max=8000.0, sampling_rate=16000, fft_size=512, nr_of_filters=num_mel)
    return {
        "within_std": st["pooled_within_std"],
        "speaker_std": st["speaker_std"],
        "real_mean": st["global_mean"],
        "real_std": st["global_std"],
        # voicing gate thresholds, midway between the table's voiced and unvoiced (resp. silence and speech) rows
        "voicing_c": float((low[voiced].mean() + low[unvoiced].mean()) / 2),
        "energy_c": float((energy[sil] + energy[speech].mean()) / 2),
        "melmat": melmat.astype(numpy.float32),
    }


def _const(arr, dims, like: Tensor) -> Tensor:
    import torch

    return rf.convert_to_tensor(torch.tensor(arr, device=like.raw_tensor.device), dims=dims, dtype="float32")


def _box_matrix(feat_dim: Dim, width: int, like: Tensor) -> Tensor:
    """[F_in, F_out] moving-average matrix over the channel axis, scaled to keep unit variance of white input"""
    import numpy

    f = feat_dim.dimension
    m = numpy.zeros((f, f), dtype=numpy.float32)
    for o in range(f):
        lo, hi = max(0, o - width // 2), min(f, o + width // 2 + 1)
        m[lo:hi, o] = 1.0 / numpy.sqrt(hi - lo)
    out = Dim(f, name="box_out")
    return _const(m, [feat_dim, out], like), out


def _smooth_channels(x: Tensor, feat_dim: Dim, width: int) -> Tensor:
    mat, out = _box_matrix(feat_dim, width, x)
    y = rf.matmul(x, mat, reduce=feat_dim)
    y, _ = rf.replace_dim(y, in_dim=out, out_dim=feat_dim)
    return y


def _augment(x: Tensor, *, spatial_dim: Dim, feat_dim: Dim, opts: Dict[str, Any], consts: Dict[str, Any]) -> Tensor:
    """x: pseudo log-mel [B..., T, F] (log10 power). Returns the augmented features (train time)."""
    batch_dims = x.remaining_dims([spatial_dim, feat_dim])
    ch = rf.range_over_dim(feat_dim, dtype="float32")
    nf = float(feat_dim.dimension)
    if opts.get("rand"):
        # within-phone frame variability (channel-correlated, white in time)
        noise = _smooth_channels(rf.random_normal(x.dims, dtype="float32"), feat_dim, 5)
        x = x + noise * _const(consts["within_std"], [feat_dim], x) * opts.get("within_scale", 1.0)
        # per-utterance offset (speaker / channel), smooth over channels
        off = _smooth_channels(rf.random_normal(batch_dims + [feat_dim], dtype="float32"), feat_dim, 15)
        x = x + off * _const(consts["speaker_std"], [feat_dim], x)
        # gain and spectral tilt, per utterance (log10 units)
        gain = rf.random_uniform(batch_dims, minval=-0.3, maxval=0.3, dtype="float32")
        tilt = rf.random_uniform(batch_dims, minval=-0.5, maxval=0.5, dtype="float32")
        x = x + gain + tilt * (ch / (nf - 1.0) - 0.5)
        # frequency warp (VTLP-like): out channel c reads in channel c * alpha, linear interpolation
        alpha = rf.random_uniform(batch_dims, minval=0.9, maxval=1.1, dtype="float32")
        out = Dim(feat_dim.dimension, name="warp_out")
        src_pos = rf.clip_by_value(rf.range_over_dim(out, dtype="float32") * alpha, 0.0, nf - 1.0)
        warp = rf.relu(1.0 - rf.abs(ch - src_pos))  # [B, F_in, F_out]
        x = rf.matmul(x, warp, reduce=feat_dim)
        x, _ = rf.replace_dim(x, in_dim=out, out_dim=feat_dim)
    if opts.get("pitch"):
        # harmonic comb at a per-utterance F0, pushed through the mel filterbank, on voiced frames only
        fdim = Dim(257, name="lin_freq")
        freqs = rf.range_over_dim(fdim, dtype="float32") * (16000.0 / 512.0)
        f0 = rf.random_uniform(batch_dims, minval=80.0, maxval=260.0, dtype="float32")
        kappa = 4.0
        comb = rf.exp(kappa * (rf.cos(freqs * (6.283185307 / f0)) - 1.0))  # [B, 257]
        melmat = _const(consts["melmat"], [fdim, feat_dim], x)
        num = rf.matmul(comb, melmat, reduce=fdim)
        den = rf.reduce_sum(melmat, axis=fdim) * rf.reduce_mean(comb, axis=fdim)
        ratio = rf.log(rf.maximum(num / den, 1e-4)) * (1.0 / 2.302585093)  # log10, [B, F]
        low_w = _const([1.0 / 20 if i < 20 else 0.0 for i in range(feat_dim.dimension)], [feat_dim], x)
        high_w = _const([1.0 / 30 if i >= 50 else 0.0 for i in range(feat_dim.dimension)], [feat_dim], x)
        low = rf.reduce_sum(x * low_w, axis=feat_dim) - rf.reduce_sum(x * high_w, axis=feat_dim)
        energy = rf.reduce_mean(x, axis=feat_dim)
        voicing = rf.sigmoid((low - consts["voicing_c"]) / 0.25) * rf.sigmoid((energy - consts["energy_c"]) / 0.25)
        x = x + ratio * voicing
    if opts.get("realstats"):
        # per-batch standardization, then the real-audio channel mean / std (NaN-safe for empty batches)
        axes = batch_dims + [spatial_dim]
        n = rf.maximum(
            rf.cast(rf.reduce_sum(spatial_dim.get_size_tensor(device=x.device), axis=batch_dims), "float32"), 1.0
        )
        mean = rf.reduce_sum(x, axis=axes) / n
        var = rf.maximum(rf.reduce_sum(x * x, axis=axes) / n - mean * mean, 1e-5)
        x = (x - mean) * rf.rsqrt(var) * _const(consts["real_std"], [feat_dim], x) + _const(
            consts["real_mean"], [feat_dim], x
        )
    return x


def pseudo0h_frontend_single_stream_train_step(*, model, extern_data, **_kwargs_unused):
    """zeyer's ``aed_pseudo_enc_frontend_single_stream_train_step`` (exp2026_05_28_tts_encoder_fzj), unchanged except:
    the text branch's pseudo features pass through :func:`_augment` (config ``pseudo0h_aug``) in training,
    and with ``pseudo0h_consistency`` a second randomization of the same rendering is encoded and pulled
    towards the first at the outputs of the given encoder blocks (1 - cosine per frame)."""
    from returnn.config import get_global_config
    from returnn.util.collect_outputs_dict import CollectOutputsDict
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import aed as _aed

    config = get_global_config()  # noqa
    aug_opts = config.typed_value("pseudo0h_aug") or {}
    consist = config.typed_value("pseudo0h_consistency") or {}  # {"blocks": [3, 4], "scale": 0.5}
    consts = _host_constants(
        config.value("pseudo0h_stats_file", None), config.value("pseudo_enc_frozen_table", None), 80
    )

    data = extern_data[config.typed_value("default_input")]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[config.typed_value("target")]
    targets_spatial_dim = targets.get_time_dim_tag()
    phonemes = extern_data[PHONEMES_DATA_KEY]
    phonemes_spatial_dim = phonemes.get_time_dim_tag()

    aux_loss_layers = config.typed_value("aux_loss_layers") or ()
    aux_loss_scales = config.typed_value("aux_loss_scales", [1.0] * len(aux_loss_layers))
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    dec_aux_loss_layers = config.typed_value("dec_aux_loss_layers") or ()
    dec_aux_loss_scales = config.typed_value("dec_aux_loss_scales", [1.0] * len(dec_aux_loss_layers))
    use_normalized_loss = config.typed_value("use_normalized_loss", True)
    if isinstance(use_normalized_loss, bool):
        use_normalized_loss = "frames" if use_normalized_loss else "none"
    normed = {"none": False, "frames": True}[use_normalized_loss]
    label_smoothing = config.float("label_smoothing", 0.1)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)

    base_feats, text_spatial_dim = model.pseudo_enc(phonemes, spatial_dim=phonemes_spatial_dim)
    audio_feats, audio_spatial_dim = model.feature_extraction(data, in_spatial_dim=data_spatial_dim)
    if base_feats.dtype != audio_feats.dtype:
        base_feats = rf.cast(base_feats, audio_feats.dtype)
    feat_dim = base_feats.feature_dim

    train_flag = rf.get_run_ctx().train_flag

    def _text_view():
        v = _augment(base_feats, spatial_dim=text_spatial_dim, feat_dim=feat_dim, opts=aug_opts, consts=consts)
        if isinstance(train_flag, bool):
            return v if train_flag else base_feats
        return rf.where(train_flag, v, base_feats)

    def _join(text_feats):
        feats, feats_spatial_dim = rf.concat(
            (audio_feats, audio_spatial_dim), (text_feats, text_spatial_dim), handle_dynamic_dims=True
        )
        feats.feature_dim = model.in_dim
        _a_cap = _dim_capacity(audio_spatial_dim)
        _t_cap = _dim_capacity(text_spatial_dim)
        if _a_cap and _t_cap:
            feats_spatial_dim.capacity = max(_a_cap, _t_cap)
        return feats, feats_spatial_dim

    feats, feats_spatial_dim = _join(_text_view() if aug_opts else base_feats)

    if config.bool("use_eos_postfix", False):
        ctc_targets, (ctc_targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )
    else:
        ctc_targets, ctc_targets_spatial_dim = targets, targets_spatial_dim

    consist_blocks = list(consist.get("blocks", ()))
    keys = [str(i - 1) for i in aux_loss_layers] + [str(b - 1) for b in consist_blocks]
    collected_outputs = CollectOutputsDict(allowed_key_patterns=keys)
    enc_raw, enc_spatial_dim = model.encode_from_features(
        feats, in_spatial_dim=feats_spatial_dim, collected_outputs=collected_outputs
    )
    enc = model.decoder.transform_encoder(enc_raw, axis=enc_spatial_dim)

    if consist_blocks:
        feats2, feats2_spatial_dim = _join(_text_view())
        coll2 = CollectOutputsDict(allowed_key_patterns=[str(b - 1) for b in consist_blocks])
        _, enc2_spatial_dim = model.encode_from_features(
            feats2, in_spatial_dim=feats2_spatial_dim, collected_outputs=coll2
        )
        for b in consist_blocks:
            h1 = collected_outputs[str(b - 1)]
            h2, _ = rf.replace_dim(coll2[str(b - 1)], in_dim=enc2_spatial_dim, out_dim=enc_spatial_dim)
            d = model.encoder.out_dim
            cos = rf.reduce_sum(h1 * h2, axis=d) * rf.rsqrt(
                rf.maximum(rf.reduce_sum(h1 * h1, axis=d) * rf.reduce_sum(h2 * h2, axis=d), 1e-8)
            )
            (1.0 - cos).mark_as_loss(f"consist_{b}", scale=consist.get("scale", 0.5), use_normalized_loss=normed)

    for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx > len(model.encoder.layers):
            continue
        aux_logits = getattr(model, f"enc_aux_logits_{layer_idx}")(collected_outputs[str(layer_idx - 1)])
        aux_ctc_log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        aux_loss = rf.ctc_loss(
            logits=aux_ctc_log_probs,
            logits_normalized=True,
            targets=ctc_targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=ctc_targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        aux_loss.mark_as_loss(
            f"ctc_{layer_idx}",
            scale=aux_loss_scales[i],
            custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(device=aux_ctc_log_probs.device),
            use_normalized_loss=normed,
        )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )
    dec_collected = CollectOutputsDict(allowed_key_patterns=[str(i - 1) for i in dec_aux_loss_layers])
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
        collected_outputs=dec_collected,
    )
    dec_aux_logits = {}
    for layer_idx in dec_aux_loss_layers:
        norm = getattr(model, f"dec_aux_final_layer_norm_{layer_idx}")
        linear = getattr(model, f"dec_aux_logits_{layer_idx}")
        dec_aux_logits[layer_idx] = linear(norm(dec_collected[str(layer_idx - 1)]))

    targets_packed, pack_dim = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    for postfix, scale, logits_ in [("", aed_loss_scale, logits)] + [
        (f"_{k}", dec_aux_loss_scales[i], dec_aux_logits[k]) for i, k in enumerate(dec_aux_loss_layers)
    ]:
        logits_packed, _ = rf.pack_padded(
            logits_, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )
        if not model.out_eos_separated:
            log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        else:
            log_prob = _aed.log_probs_with_eos_separated(
                logits_packed, target_dim=model.target_dim, eos_idx=model.eos_idx
            )
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, label_smoothing, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.mark_as_loss(f"ce{postfix}", scale=scale, use_normalized_loss=normed)
        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name=f"fer{postfix}", as_error=True)


def _dim_capacity(dim: Dim) -> Optional[int]:
    if dim.dimension is not None:
        return dim.dimension
    # noinspection PyProtectedMember
    return dim.capacity or dim._derived_capacity()


# ---------------------------------------------------------------- pilots

PILOT_EPOCHS = 3
PILOTS: Dict[str, Dict[str, Any]] = {
    "rand": {"aug": {"rand": True}},
    "rand-consist": {"aug": {"rand": True}, "consistency": {"blocks": [3, 4], "scale": 0.5}},
    "pitch": {"aug": {"pitch": True}},
    "realstats": {"aug": {"realstats": True}},
    "all": {"aug": {"rand": True, "pitch": True, "realstats": True}, "consistency": {"blocks": [3, 4], "scale": 0.5}},
}


def register_pseudo0h_pilots(*, prefix: str):
    """The 0 h audio run of the LS ladder, stopped after :data:`PILOT_EPOCHS` subepochs, once per pilot."""
    from i6_experiments.users.zeyer.experiments import exp2026_05_28_tts_encoder_fzj as az
    from i6_experiments.users.zeyer.datasets.hf_librispeech_mfa_alignments import (
        get_mfa_phone_mean_logmel_table,
        get_mfa_phone_duration_table,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon

    stats = get_logmel_variance_stats()
    abl = "pseudo-enc-logmel-mfatable-realdur2-lerp-dur07-packed-single-gumbel-muon-nep38-specaug50-stepcomp"
    for pilot, spec in PILOTS.items():
        # the ablation base of the LS winner (az.py, `_abl_base`), with the audio0 override
        kwargs = dict(
            text_train_epoch_split=38,
            ls_audio_subset=0.0,
            batch_size_audio_frames=70_000,
            batch_size_phon=6_000,
            max_phon_len=300,
            asr_logmel=True,
            pseudo_speech_enc=True,
            pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
            pseudo_enc_duration_table=get_mfa_phone_duration_table().out_duration_table,
            pseudo_enc_duration_sigma=0.45,
            pseudo_enc_duration_scale=0.7,
            pseudo_enc_max_len_factor=10,
            train_seq_ordering="random",
            pseudo_enc_lerp=True,
            pseudo_enc_blank_duration_range=(0, 0),
            pseudo_enc_specaug_max_width=6,
            single_stream=True,
            interleave_gumbel_scale=1.0,
            glow_tts_add_silence_between_words=0.15,
            base_lr=1.0,
            peak_lr=5e-3,
            nep=38,
            behavior_version=29,
            pseudo_enc_frontend_concat=True,
            extra_config_updates={
                "optimizer.class": rf.build_dict(Muon)["class"],
                "packed_tensors": True,
                "torch_distributed": {"reduce_type": "grad_explicit"},
                "batch_size": None,
                "packed_batch_size": {"data": 11_200_000, "classes": 5_000, "phonemes": 6_000},
                "batching": "random",
                "torch_cuda_graph": {
                    "batch_size_bound": 500,
                    "dim_capacity": {"data": 312_000, "classes": 80, "phonemes": 300},
                    "warmup_steps": 0,
                    "compile": True,
                },
                "optimizer.weight_decay": 0.027,
                "specaugment_num_spatial_mask_factor": 50,
                "specaugment_steps": (1850, 5550, 9250),
                # the pilot: same schedule as the 38-subepoch run, stopped early; own text-branch step
                "__num_epochs": PILOT_EPOCHS,
                "train_step": pseudo0h_frontend_single_stream_train_step,
                "pseudo0h_aug": spec["aug"],
                **({"pseudo0h_consistency": spec["consistency"]} if spec.get("consistency") else {}),
                "pseudo0h_stats_file": stats.out_stats_npz,
            },
            extra_config_deletes=["optimizer.epsilon"],
        )
        az._train_tts_encoder(f"{abl}-audio0-textP38-pilot{PILOT_EPOCHS}ep-{pilot}", prefix=prefix, **kwargs)
