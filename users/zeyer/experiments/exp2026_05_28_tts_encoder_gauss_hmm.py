"""
Gaussian-HMM aligner in RETURNN, for the text-injection tables.

One diagonal Gaussian per HMM state on the ASR's own log-mel front-end
(:func:`rf.audio.log_mel_filterbank_from_raw` defaults on peak-normalized audio,
the same features the injection feeds the encoder),
monophone, three left-to-right states per phone, one optional state for the silence unit,
trained with the full-sum (Baum-Welch) loss through the native FastBaumWelch op.
The emission means are then the phone -> log-mel table by construction,
and a Viterbi pass over the training data gives the duration statistics.
Nothing but the training audio, its transcripts and the (G2P-extended) lexicon goes in.

Topology per utterance: the phoneme sequence of the transcript (``PhoneSeqGenerator``, deterministic,
``[space]`` at every word boundary and at both ends) as a linear chain of units;
phone units have 3 sub-states (loop + advance), silence units one state (loop) and are optional
(skip edges around them); a virtual start state and a final state close the chain,
so the chain is walked with exactly one emission per frame.
Transitions are unweighted (all edge weights 0).

Flat start: all states share the same initial parameters (mean 0, log std 0);
the chain topology alone breaks the symmetry (Kaldi's train_mono flat start),
Adam on the full-sum loss then does the EM job.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, Tuple, List
import functools
import math

from sisyphus import tk, Job, Task
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim
from i6_experiments.users.zeyer.model_interfaces import ModelDef

PHONEMES_DATA_KEY = "phonemes"
SILENCE_LABEL = "[space]"


class GaussHmm(rf.Module):
    """
    Diagonal Gaussian emissions per HMM state, on log-mel features.
    States are ``phone_idx * num_sub_states + sub_state``; the silence unit uses the first
    ``silence_num_sub_states`` of them (1 by default).
    """

    def __init__(
        self,
        *,
        feat_dim: Dim,
        phone_dim: Dim,
        silence_idx: int,
        num_sub_states: int = 3,
        mandatory_edge_silence: bool = False,
        tdp: Optional[Dict[str, float]] = None,
        edge_silence_init_epochs: int = 0,
        silence_num_sub_states: int = 1,
    ):
        """
        :param silence_num_sub_states: sub-states (minimum duration in frames) of the silence unit, <= num_sub_states.
            With a single state, silence is attractive for isolated low-energy frames anywhere and drifts into a
            broad garbage state; Kaldi/RASR give silence 3-5 states.
        :param mandatory_edge_silence: the leading and trailing silence units of the chain are not optional
            (utterances start and end with silence); with the flat start this pins the silence state
            to the edge frames instead of letting the first/last phone's sub-states absorb them.
            Diagnosis tool only (AZ): silence stays optional in the aligner, use ``tdp`` instead.
        :param tdp: fixed transition costs (-log), keys ``speech_loop``, ``speech_forward``,
            ``silence_loop``, ``silence_forward``, e.g. RASR's 10 ms defaults 3.0 / 0.0 / 0.0 / 3.0:
            silence loops for free and speech is pushed forward, so pauses are not absorbed by the
            first sub-state of the next phone. None = all 0.
        :param edge_silence_init_epochs: flat-start initialisation: the edge silences are mandatory during
            the first N (sub)epochs only, so the silence Gaussian is fitted on the utterance edges before
            it competes freely (with free silence loops and identical initial Gaussians, the silence state
            otherwise turns into a broad garbage state). The aligner itself keeps silence optional.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.phone_dim = phone_dim
        self.silence_idx = silence_idx
        self.num_sub_states = num_sub_states
        self.mandatory_edge_silence = mandatory_edge_silence
        self.tdp = tdp
        self.edge_silence_init_epochs = edge_silence_init_epochs
        assert 1 <= silence_num_sub_states <= num_sub_states, (silence_num_sub_states, num_sub_states)
        self.silence_num_sub_states = silence_num_sub_states
        self.state_dim = Dim(phone_dim.dimension * num_sub_states, name="hmm_states")
        self.mean = rf.Parameter([self.state_dim, feat_dim])
        self.mean.initial = 0.0
        self.log_std = rf.Parameter([self.state_dim, feat_dim])
        self.log_std.initial = 0.0

    def features(self, raw: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        """
        :param raw: peak-normalized waveform [B, T_raw]
        :return: log-mel [B, T, F], T
        """
        return rf.audio.log_mel_filterbank_from_raw(
            raw, in_spatial_dim=in_spatial_dim, out_dim=self.feat_dim, sampling_rate=16_000
        )

    def log_emission(self, feats: Tensor) -> Tensor:
        """
        :param feats: [B, T, F]
        :return: log N(x_t; mu_s, diag sigma_s^2) [B, T, S].
            The quadratic form is expanded into two matmuls, so no [B, T, S, F] intermediate.
        """
        inv_var = rf.exp(-2.0 * self.log_std)  # [S, F]
        sq = rf.matmul(rf.square(feats), inv_var, reduce=self.feat_dim)  # [B, T, S]: sum_f x^2 / var
        lin = rf.matmul(feats, self.mean * inv_var, reduce=self.feat_dim)  # [B, T, S]: sum_f x mu / var
        const = rf.reduce_sum(rf.square(self.mean) * inv_var, axis=self.feat_dim)  # [S]: sum_f mu^2 / var
        log_det = rf.reduce_sum(self.log_std, axis=self.feat_dim)  # [S]: sum_f log sigma
        return -0.5 * sq + lin - 0.5 * const - log_det - 0.5 * self.feat_dim.dimension * math.log(2 * math.pi)


def gauss_hmm_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> GaussHmm:
    """ModelDef: ``target_dim`` is the phoneme vocab dim (the dataset's default target)"""
    from returnn.config import get_global_config

    del epoch, in_dim  # unused
    config = get_global_config()
    feat_dim = Dim(config.int("gauss_hmm_feat_dim", 80), name="logmel", kind=Dim.Types.Feature)
    labels = list(target_dim.vocab.labels)
    return GaussHmm(
        feat_dim=feat_dim,
        phone_dim=target_dim,
        silence_idx=labels.index(SILENCE_LABEL),
        num_sub_states=config.int("gauss_hmm_num_sub_states", 3),
        mandatory_edge_silence=config.bool("gauss_hmm_mandatory_edge_silence", False),
        tdp=config.typed_value("gauss_hmm_tdp", None),
        edge_silence_init_epochs=config.int("gauss_hmm_edge_silence_init_epochs", 0),
        silence_num_sub_states=config.int("gauss_hmm_silence_num_sub_states", 1),
    )


gauss_hmm_model_def: ModelDef[GaussHmm]
gauss_hmm_model_def.behavior_version = 29
gauss_hmm_model_def.backend = "torch"
gauss_hmm_model_def.batch_size_factor = 160


def build_chain_fsa(
    units: Any,
    lens: Any,
    *,
    silence_idx: int,
    num_sub_states: int,
    mandatory_edge_silence: bool = False,
    tdp: Optional[Dict[str, float]] = None,
    silence_num_sub_states: int = 1,
) -> Tuple[Any, Any, Any, int]:
    """
    The per-utterance chain automata of a batch, in the FastBaumWelch edge format.

    Every edge carries the emission it consumes: the chain edges (loop, advance, skip) emit their
    source state, the edges out of the virtual start state emit their destination state,
    so a path of T edges scores exactly T frames. The final state closes the chain.

    :param units: numpy [B, N] int, phoneme ids (silence = ``silence_idx``)
    :param lens: numpy [B] int
    :param mandatory_edge_silence: see :class:`GaussHmm`
    :param tdp: see :class:`GaussHmm`; loop edges cost ``*_loop``, all edges leaving a state
        (advance, next unit, skip, final) cost ``*_forward`` of the source unit, start edges 0
    :return: edges int32 [4, E] (from, to, emission_idx, seq_idx), weights float32 [E] (costs, -log),
        start_end_states int32 [2, B], num states
    """
    import numpy as np

    tdp = tdp or {}
    cost = {
        (True, True): tdp.get("silence_loop", 0.0),
        (True, False): tdp.get("silence_forward", 0.0),
        (False, True): tdp.get("speech_loop", 0.0),
        (False, False): tdp.get("speech_forward", 0.0),
    }  # (is silence, is loop) -> cost
    edges: List[Tuple[int, int, int, int]] = []
    weights: List[float] = []
    starts = []
    ends = []
    state_off = 0
    for b in range(units.shape[0]):
        u = units[b, : int(lens[b])].astype(np.int64)
        n = len(u)
        assert n > 0, f"seq {b}: empty phoneme sequence"
        optional = u == silence_idx
        if mandatory_edge_silence:
            optional[0] = False
            optional[-1] = False
        n_sub = np.where(u == silence_idx, silence_num_sub_states, num_sub_states)
        s0 = state_off  # virtual start
        first = state_off + 1 + np.concatenate([[0], np.cumsum(n_sub)[:-1]])  # first chain state per unit
        final = int(state_off + 1 + n_sub.sum())
        for i in range(n):
            is_sil = bool(u[i] == silence_idx)
            for j in range(int(n_sub[i])):
                s = int(first[i]) + j
                e = int(u[i]) * num_sub_states + j
                edges.append((s, s, e, b))  # loop
                weights.append(cost[(is_sil, True)])
                if j + 1 < n_sub[i]:
                    edges.append((s, s + 1, e, b))  # advance within the unit
                    weights.append(cost[(is_sil, False)])
                    continue
                # last sub-state of the unit: advance to the next unit / the final state,
                # and past an optional next unit
                nxt = i + 1
                edges.append((s, int(first[nxt]) if nxt < n else final, e, b))
                weights.append(cost[(is_sil, False)])
                if nxt < n and optional[nxt]:
                    edges.append((s, int(first[nxt + 1]) if nxt + 1 < n else final, e, b))
                    weights.append(cost[(is_sil, False)])
        edges.append((s0, int(first[0]), int(u[0]) * num_sub_states, b))
        weights.append(0.0)
        if optional[0] and n > 1:
            edges.append((s0, int(first[1]), int(u[1]) * num_sub_states, b))
            weights.append(0.0)
        starts.append(s0)
        ends.append(final)
        state_off = final + 1
    edges_np = np.array(edges, dtype=np.int32).T  # [4, E]
    weights_np = np.array(weights, dtype=np.float32)
    start_end = np.array([starts, ends], dtype=np.int32)
    return edges_np, weights_np, start_end, state_off


def _fsa_for_batch(model: GaussHmm, phonemes: Tensor, phon_spatial_dim: Dim, batch_dim_: Dim, device):
    """FastBaumWelch inputs on ``device`` for the batch of phoneme sequences"""
    import torch

    units = phonemes.copy_compatible_to_dims_raw([batch_dim_, phon_spatial_dim]).cpu().numpy()
    lens = phon_spatial_dim.get_size_tensor().copy_compatible_to_dims_raw([batch_dim_]).cpu().numpy()
    edge_silence = model.mandatory_edge_silence
    if model.edge_silence_init_epochs and rf.get_run_ctx().train_flag:
        edge_silence = edge_silence or int(rf.get_run_ctx().epoch) <= model.edge_silence_init_epochs
    edges, weights, start_end, n_states = build_chain_fsa(
        units,
        lens,
        silence_idx=model.silence_idx,
        num_sub_states=model.num_sub_states,
        mandatory_edge_silence=edge_silence,
        tdp=model.tdp,
        silence_num_sub_states=model.silence_num_sub_states,
    )
    return (
        torch.from_numpy(edges).to(device),
        torch.from_numpy(weights).to(device),
        torch.from_numpy(start_end).to(device),
        n_states,
    )


def _log_emission_time_major(model: GaussHmm, log_em: Tensor, time_dim: Dim, batch_dim_: Dim):
    """:return: raw (T, B, S) float32, seq lens int32 (B) on device"""
    import torch

    raw = log_em.copy_transpose([time_dim, batch_dim_, model.state_dim]).raw_tensor.to(torch.float32)
    seq_lens = rf.copy_to_device(time_dim.get_size_tensor(), log_em.device)
    seq_lens = seq_lens.copy_compatible_to_dims_raw([batch_dim_]).to(torch.int32)
    return raw, seq_lens


def gauss_hmm_full_sum_loss(
    model: GaussHmm, log_em: Tensor, time_dim: Dim, phonemes: Tensor, phon_spatial_dim: Dim
) -> Tensor:
    """
    :return: -log p(x | phoneme chain) per seq, [B], differentiable w.r.t. the emissions
    """
    from returnn.torch.util.native_op import _FastBaumWelchScoresAutogradFunc
    from returnn.torch.util.array_ import sequence_mask_time_major

    (batch_dim_,) = log_em.remaining_dims((time_dim, model.state_dim))
    raw, seq_lens = _log_emission_time_major(model, log_em, time_dim, batch_dim_)
    edges, weights, start_end, n_states = _fsa_for_batch(model, phonemes, phon_spatial_dim, batch_dim_, raw.device)
    seq_mask = sequence_mask_time_major(seq_lens)  # (T, B)
    loss = _FastBaumWelchScoresAutogradFunc.apply(raw, False, seq_mask, edges, weights, start_end, n_states)
    return rf.convert_to_tensor(loss, dims=[batch_dim_], name="fullsum")


def gauss_hmm_viterbi(
    model: GaussHmm, log_em: Tensor, time_dim: Dim, phonemes: Tensor, phon_spatial_dim: Dim
) -> Tuple[Tensor, Tensor]:
    """
    :return: best-path state per frame [B, T] (sparse state_dim; padding frames 0), path score [B] (+log)
    """
    from returnn.torch.util.native_op import fast_viterbi

    (batch_dim_,) = log_em.remaining_dims((time_dim, model.state_dim))
    raw, seq_lens = _log_emission_time_major(model, log_em, time_dim, batch_dim_)
    edges, weights, start_end, _ = _fsa_for_batch(model, phonemes, phon_spatial_dim, batch_dim_, raw.device)
    alignment, scores = fast_viterbi(
        am_scores=raw, am_seq_len=seq_lens, edges=edges, weights=weights, start_end_states=start_end, mask_idx=0
    )
    align = rf.convert_to_tensor(alignment, dims=[time_dim, batch_dim_], sparse_dim=model.state_dim, name="alignment")
    return align.copy_transpose([batch_dim_, time_dim]), rf.convert_to_tensor(scores, dims=[batch_dim_], name="score")


def _features_and_log_emission(model: GaussHmm, extern_data) -> Tuple[Tensor, Dim, Tensor, Dim]:
    from returnn.config import get_global_config

    config = get_global_config()
    data = extern_data[config.typed_value("default_input")]
    data_spatial_dim = data.get_time_dim_tag()
    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    feats, time_dim = model.features(data, in_spatial_dim=data_spatial_dim)
    log_em = model.log_emission(feats)
    phonemes = extern_data[PHONEMES_DATA_KEY]
    return log_em, time_dim, phonemes, phonemes.get_time_dim_tag()


def gauss_hmm_train_step(*, model: GaussHmm, extern_data, **_kwargs_unused):
    """RETURNN train_step: full-sum loss, normalized per frame"""
    log_em, time_dim, phonemes, phon_spatial_dim = _features_and_log_emission(model, extern_data)
    loss = gauss_hmm_full_sum_loss(model, log_em, time_dim, phonemes, phon_spatial_dim)
    rf.get_run_ctx().mark_as_loss(
        loss, "fullsum", custom_inv_norm_factor=time_dim.get_size_tensor(), use_normalized_loss=True
    )


def gauss_hmm_align_forward_step(*, model: GaussHmm, extern_data, **_kwargs_unused):
    """
    RETURNN forward_step: Viterbi state per frame as ``output``.
    (Only that: forward_to_hdf's HDF writer cannot take a sparse extra output with a vocab.)
    """
    log_em, time_dim, phonemes, phon_spatial_dim = _features_and_log_emission(model, extern_data)
    align, _ = gauss_hmm_viterbi(model, log_em, time_dim, phonemes, phon_spatial_dim)
    (batch_dim_,) = align.remaining_dims(time_dim)
    # the config's model_outputs dims are templates; bind them to the actual feature time dim and
    # state dim (mark_as_output compares the dims by identity)
    expected = rf.get_run_ctx().expected_outputs["output"]
    expected.dims[-1].declare_same_as(time_dim)
    expected.sparse_dim.declare_same_as(align.sparse_dim)
    rf.get_run_ctx().mark_as_output(align, "output", dims=[batch_dim_, time_dim])


_phone_seq_gen_cache = None


def gauss_hmm_map_seq(seq, *, phon_dim: Dim, **_kwargs):
    """
    PostprocessingDataset map_seq: the raw utf8 transcript bytes (``classes``) -> phoneme chain,
    deterministic (config ``gauss_hmm_phone_info``), audio passed through.
    """
    import numpy as np
    from returnn.tensor import TensorDict
    from returnn.config import get_global_config
    from returnn.datasets.lm import PhoneSeqGenerator

    global _phone_seq_gen_cache
    if _phone_seq_gen_cache is None:
        _phone_seq_gen_cache = PhoneSeqGenerator(**get_global_config().typed_value("gauss_hmm_phone_info"))
    seq_gen = _phone_seq_gen_cache
    orth = bytes(np.asarray(seq["classes"].raw_tensor).astype("uint8").tolist()).decode("utf8")
    phon_ids = seq_gen.seq_to_class_idxs(seq_gen.generate_seq(orth), dtype="int32")

    out = TensorDict()
    out.data["data"] = seq["data"]
    out.data[PHONEMES_DATA_KEY] = Tensor(
        PHONEMES_DATA_KEY, dims=[Dim(None, name="phon_seq")], dtype="int32", sparse_dim=phon_dim, raw_tensor=phon_ids
    )
    return out


def get_gauss_hmm_phone_info(lexicon: Optional[tk.Path] = None) -> Dict[str, Any]:
    """
    deterministic phoneme chains with silence at every word boundary and at both ends (optional in the HMM)

    :param lexicon: None = the glow-tts lexicon; PhoneSeqGenerator raises on OOV words,
        so pass a lexicon that covers the transcripts (G2P-extended)
    """
    from i6_experiments.users.zeyer.external_models.glow_tts import get_glow_tts_phone_info

    return {
        **get_glow_tts_phone_info(train=False, with_start_end_lemmas=False),
        **({"lexicon_file": lexicon} if lexicon is not None else {}),
        "add_silence_beginning": 1.0,
        "add_silence_between_words": 1.0,
        "add_silence_end": 1.0,
    }


def _ls_ogg_zip(parts, *, training: bool, partition_epoch: Optional[int] = None, subset: Optional[int] = None):
    from i6_experiments.users.zeyer.datasets.librispeech import _get_librispeech_ogg_zip_dict, _raw_audio_opts

    d: Dict[str, Any] = {
        "class": "OggZipDataset",
        "path": [_get_librispeech_ogg_zip_dict()[p] for p in parts],
        "use_cache_manager": True,
        "audio": dict(_raw_audio_opts),
        "targets": {"class": "Utf8ByteTargets"},
    }
    if training:
        d["partition_epoch"] = partition_epoch
        d["seq_ordering"] = "laplace:.1000"
    else:
        d["fixed_random_seed"] = 1
        d["seq_ordering"] = "sorted_reverse"
    if subset:
        d["fixed_random_subset"] = subset
    return d


def _with_phonemes(ogg_zip: Dict[str, Any], phon_dim: Dim) -> Dict[str, Any]:
    from returnn.tensor import Dim as _Dim

    return {
        "class": "PostprocessingDataset",
        "seq_ordering": "default",
        "dataset": ogg_zip,
        "map_seq": functools.partial(gauss_hmm_map_seq, phon_dim=phon_dim),
        "map_outputs": {
            "data": {"dims": [_Dim(None, name="time"), _Dim(1, name="audio")], "dtype": "float32"},
            PHONEMES_DATA_KEY: {"dims": [_Dim(None, name="phon_seq")], "sparse_dim": phon_dim, "dtype": "int32"},
        },
    }


_LS_TRAIN_PARTS = ("train-clean-100", "train-clean-360", "train-other-500")


def get_gauss_hmm_ls_datasets(*, train_epoch_split: int = 10, eval_subset: int = 3000):
    """
    :return: (train dataset config with dev/devtrain, alignment dataset config = the full train set once)
    """
    from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic
    from i6_experiments.users.zeyer.external_models.glow_tts import get_glow_tts_phoneme_extern_data

    phon_extern = get_glow_tts_phoneme_extern_data()
    phon_dim = phon_extern["sparse_dim"]
    extern_data = {
        "data": {"dim_tags": [batch_dim, Dim(None, name="time", kind=Dim.Types.Spatial), Dim(1, name="audio")]},
        PHONEMES_DATA_KEY: phon_extern,
    }
    train = DatasetConfigStatic(
        main_name="LS train-960 + phoneme chains",
        train_dataset=_with_phonemes(
            _ls_ogg_zip(_LS_TRAIN_PARTS, training=True, partition_epoch=train_epoch_split), phon_dim
        ),
        default_input="data",
        default_target=PHONEMES_DATA_KEY,
        extern_data=extern_data,
        eval_datasets={
            "dev": _with_phonemes(_ls_ogg_zip(("dev-other",), training=False, subset=eval_subset), phon_dim),
            "devtrain": _with_phonemes(_ls_ogg_zip(_LS_TRAIN_PARTS, training=False, subset=eval_subset), phon_dim),
        },
        use_deep_copy=True,
    )
    align = DatasetConfigStatic(
        main_name="train",
        main_dataset=_with_phonemes(_ls_ogg_zip(_LS_TRAIN_PARTS, training=False), phon_dim),
        default_input="data",
        default_target=PHONEMES_DATA_KEY,
        extern_data=extern_data,
        use_deep_copy=True,
    )
    return train, align


class GaussHmmTablesJob(Job):
    """
    The injection tables from a trained Gaussian-HMM and its Viterbi alignment of the training data,
    in the formats of the MFA table jobs (:mod:`hf_librispeech_mfa_alignments`):

    ``out_mean_table``: npz ``means`` [vocab, F] (the emission means of the phone's sub-states,
    weighted by their Viterbi occupancy; labels without frames get the occupancy-weighted global mean)
    and ``labels``; ``out_duration_table``: npz ``medians`` / ``means`` / ``counts`` / ``labels``
    [vocab], durations in frames (10 ms); ``out_stats``: json.
    """

    def __init__(
        self,
        *,
        checkpoint: tk.Path,
        alignment_hdf: tk.Path,
        phoneme_vocab: tk.Path,
        returnn_root: tk.Path,
        num_sub_states: int = 3,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.alignment_hdf = alignment_hdf
        self.phoneme_vocab = phoneme_vocab
        self.returnn_root = returnn_root
        self.num_sub_states = num_sub_states
        self.rqmt = {"cpu": 2, "mem": 8, "time": 4}
        self.out_mean_table = self.output_path("mean_logmel.npz")
        self.out_duration_table = self.output_path("phone_durations.npz")
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sys

        sys.path.insert(0, self.returnn_root.get_path())

        import json
        import numpy as np
        import torch
        from returnn.datasets.hdf import HDFDataset
        from returnn.datasets.util.vocabulary import Vocabulary

        vocab = Vocabulary(self.phoneme_vocab.get_path(), unknown_label="[UNKNOWN]")
        labels = list(vocab.labels)
        num_labels = len(labels)
        k = self.num_sub_states

        ckpt = torch.load(self.checkpoint.get_path(), map_location="cpu")
        params = ckpt["model"] if "model" in ckpt else ckpt
        means = params["mean"].numpy().astype(np.float64)  # [S, F]
        assert means.shape[0] == num_labels * k, (means.shape, num_labels, k)

        occupancy = np.zeros((num_labels * k,), dtype=np.int64)
        durs = [[] for _ in range(num_labels)]
        n_seqs = 0
        ds = HDFDataset([self.alignment_hdf.get_path()])
        ds.init_seq_order(epoch=1)
        seq_idx = 0
        while ds.is_less_than_num_seqs(seq_idx):
            ds.load_seqs(seq_idx, seq_idx + 1)
            states = ds.get_data(seq_idx, "data").astype(np.int64)  # [T]
            np.add.at(occupancy, states, 1)
            label = states // k
            sub = states % k
            # a unit ends where the label changes or the sub-state goes back (adjacent identical phones)
            new_unit = np.ones_like(states, dtype=bool)
            new_unit[1:] = (label[1:] != label[:-1]) | (sub[1:] < sub[:-1])
            starts = np.flatnonzero(new_unit)
            ends = np.append(starts[1:], len(states))
            for s, e in zip(starts, ends):
                durs[int(label[s])].append(int(e - s))
            n_seqs += 1
            seq_idx += 1

        occ = occupancy.reshape(num_labels, k).astype(np.float64)
        global_mean = (means * occupancy[:, None]).sum(axis=0) / max(float(occupancy.sum()), 1.0)
        table = np.zeros((num_labels, means.shape[1]), dtype=np.float32)
        for v in range(num_labels):
            if occ[v].sum() > 0:
                table[v] = (means[v * k : (v + 1) * k] * occ[v][:, None]).sum(axis=0) / occ[v].sum()
            else:
                table[v] = global_mean
        np.savez(self.out_mean_table.get_path(), means=table, labels=np.array(labels, dtype=object))

        medians = np.zeros((num_labels,), dtype=np.float32)
        dmeans = np.zeros((num_labels,), dtype=np.float32)
        counts = np.zeros((num_labels,), dtype=np.int64)
        for v in range(num_labels):
            if durs[v]:
                medians[v] = float(np.median(durs[v]))
                dmeans[v] = float(np.mean(durs[v]))
                counts[v] = len(durs[v])
        all_d = [d for v in range(num_labels) for d in durs[v]]
        fallback = float(np.median(all_d)) if all_d else 1.0
        medians[counts == 0] = fallback
        dmeans[counts == 0] = fallback
        np.savez(
            self.out_duration_table.get_path(),
            medians=medians,
            means=dmeans,
            counts=counts,
            labels=np.array(labels, dtype=object),
        )
        stats = {
            "n_seqs": n_seqs,
            "frame_counts": {labels[v]: int(occ[v].sum()) for v in range(num_labels)},
            "state_occupancy": {labels[v]: [int(c) for c in occ[v]] for v in range(num_labels)},
            "duration_median": {labels[v]: float(medians[v]) for v in range(num_labels)},
            "duration_mean": {labels[v]: float(dmeans[v]) for v in range(num_labels)},
            "duration_count": {labels[v]: int(counts[v]) for v in range(num_labels)},
            "num_sub_states": k,
        }
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(stats, f, indent=2)


def gauss_hmm_ls960(
    prefix: str,
    *,
    lexicon: Optional[tk.Path] = None,
    mandatory_edge_silence: bool = False,
    tdp: Optional[Dict[str, float]] = None,
    edge_silence_init_epochs: int = 0,
    silence_num_sub_states: int = 1,
    name: str = "gauss-hmm-mono1g-ls960",
) -> GaussHmmTablesJob:
    """
    Train the Gaussian-HMM on LS train-960 (4 GPUs), Viterbi-align the training data, build the tables.

    :param lexicon: must cover the train-960 transcripts (see :func:`get_gauss_hmm_phone_info`)
    :param mandatory_edge_silence: see :class:`GaussHmm`; False keeps the hash of the first run
    :param tdp: see :class:`GaussHmm`; None keeps the hash of the first run
    :param edge_silence_init_epochs: see :class:`GaussHmm`; 0 keeps the hash of the first run
    :param silence_num_sub_states: see :class:`GaussHmm`; 1 keeps the hash of the first run
    """
    from i6_experiments.users.zeyer.train_v4 import train
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf
    from i6_experiments.users.zeyer.external_models.glow_tts import get_glow_tts_phoneme_vocab
    from i6_experiments.users.zeyer import tools_paths

    train_ds, align_ds = get_gauss_hmm_ls_datasets(train_epoch_split=10)
    n_ep = 20  # two passes over train-960
    common = {
        "gauss_hmm_phone_info": get_gauss_hmm_phone_info(lexicon),
        "gauss_hmm_feat_dim": 80,
        **({"gauss_hmm_mandatory_edge_silence": True} if mandatory_edge_silence else {}),
        **({"gauss_hmm_tdp": tdp} if tdp else {}),
        **({"gauss_hmm_edge_silence_init_epochs": edge_silence_init_epochs} if edge_silence_init_epochs else {}),
        **({"gauss_hmm_silence_num_sub_states": silence_num_sub_states} if silence_num_sub_states != 1 else {}),
    }
    exp = train(
        f"{prefix}/{name}",
        train_dataset=train_ds,
        train_epoch_split=10,
        config={
            **common,
            "__num_epochs": n_ep,
            "__time_rqmt": 6,
            "batch_size": 200_000 * 160,  # 2000 s of audio per batch
            "max_seqs": 400,
            "optimizer": {"class": "adam", "epsilon": 1e-8},
            "accum_grad_multiple_step": 1,
            "learning_rate": 0.01,
            "learning_rates": [0.01] * 14 + [0.005] * 3 + [0.002, 0.001, 0.001],
            "train_step": gauss_hmm_train_step,
            "learning_rate_control_error_measure": "dev_loss_fullsum",
            "torch_distributed": {},
            "__multi_proc_dataset": 4,
        },
        model_def=gauss_hmm_model_def,
        gpu_mem=96,
        num_processes=4,
    )
    phon_dim = align_ds.get_extern_data()[PHONEMES_DATA_KEY]["sparse_dim"]
    state_dim = Dim(phon_dim.dimension * 3, name="hmm_states")
    align_hdf = forward_to_hdf(
        dataset=align_ds,
        model=exp.get_last_fixed_epoch(),
        forward_step=gauss_hmm_align_forward_step,
        config={
            **common,
            "batch_size": 200_000 * 160,
            "max_seqs": 400,
            "model_outputs": {
                "output": {"dims": [batch_dim, Dim(None, name="time")], "sparse_dim": state_dim, "dtype": "int32"},
            },
        },
        # FZJ bills full nodes and settings.py rejects gpu < 4; the Viterbi pass uses one GPU of the node
        # (~1 h for train-960, data-loading bound). Sharding it over the 4 GPUs is the obvious follow-up.
        forward_rqmt={"time": 8, "gpu": 4, "cpu": 16, "mem": 64},
        forward_device="gpu",
        forward_alias_name=f"{prefix}/{name}/align-train960",
    )
    tables = GaussHmmTablesJob(
        checkpoint=exp.get_last_fixed_epoch().checkpoint.path,
        alignment_hdf=align_hdf,
        phoneme_vocab=get_glow_tts_phoneme_vocab(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    tables.add_alias(f"{prefix}/{name}/tables")
    tk.register_output(f"{prefix}/{name}/mean_logmel.npz", tables.out_mean_table)
    tk.register_output(f"{prefix}/{name}/phone_durations.npz", tables.out_duration_table)
    tk.register_output(f"{prefix}/{name}/tables_stats.json", tables.out_stats)
    return tables
