"""Self-attention alignment matrices for decoder-only speech LLMs.

The analog of whisper's cross-attn DTW:
the queries that predict each transcript token attend over the audio token block;
that [token x audio-frame] matrix is the alignment signal.

:class:`SelectSelfAttnAlignHeadsJob` finds the alignment heads on gold dev data
(no published head masks exist for these models, unlike whisper).
:class:`ExtractSelfAttnPerTokenJob` writes the selected heads' mean matrix
in the SAME HDF schema as :class:`..extract_per_token_grads.ExtractInGradsPerTokenJob`,
so :class:`..word_align_from_per_token_grads.WordAlignFromPerTokenGradsJob`
and all metric jobs consume it unchanged
("same DP, different signal" vs grad-align).
"""

from typing import Any, Dict, List, Optional, Union
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


# The adapters report the audio-position count covering real audio (``n_audio_real``
# in the ``collect_attentions`` dict) -- the padded window's tail positions are dropped
# (e.g. voxtral pads to 30 s -> 375 tokens; whisper encoder to 1500 frames).


class SelectSelfAttnAlignHeadsJob(Job):
    """Rank every (layer, head) by alignment WBE on gold dev data; output the top-k.

    For each scored head, the [token x audio] attention matrix goes through the same
    DP aligner as the production align job (generic opts), the resulting word
    boundaries are compared to the gold ones, and heads are ranked by mean WBE.
    """

    __sis_version__ = 1  # Aligner boundary off-by-one fix (end frame t, was t-1)

    # Default (subword, no upsample) keeps the original hash -> finished subword head-sel
    # jobs reuse; only char rows (True) re-hash and run fresh.
    __sis_hash_exclude__ = {"time_upsample_when_short": False}

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_config: Dict[str, Any],
        num_seqs: int = 50,
        top_k: int = 8,
        time_upsample_when_short: bool = False,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param time_upsample_when_short: if a head's [n_tok, n_audio] matrix has more
            tokens than audio frames (char-level targets on a coarse ~80 ms token grid),
            the monotonic DP (needs S<=T) can't run. When set, repeat the time axis
            k=ceil(S/T)x so the DP runs (boundaries resolve to sec_per_frame/k; the
            underlying 80 ms grid is the real limit). Needed for char-level self-attn.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_config = model_config
        self.num_seqs = num_seqs
        self.top_k = top_k
        self.time_upsample_when_short = time_upsample_when_short
        self.returnn_root = returnn_root

        self.rqmt = {"time": 8, "cpu": 4, "gpu": 1, "mem": 80}
        self.out_heads = self.output_var("heads.txt")  # list of [layer, head]
        self.out_report = self.output_var("report.txt")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import time

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch

        from returnn.util import better_exchook

        better_exchook.install()

        from .models import make_model
        from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import Aligner
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            per_utt_boundary_errors,
        )

        dev = torch.device("cuda")
        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)
        for p in model.parameters():
            p.requires_grad = False

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        n_seqs = min(self.num_seqs, len(ds))
        print(f"Scoring heads on {n_seqs} seqs")

        aligner = Aligner(apply_softmax_over_time=True, blank_score=-5)
        n_align_fail = 0
        last_align_err = None
        wbe_sum: Optional[np.ndarray] = None  # [L, H]
        wbe_cnt = 0

        for seq_idx in range(n_seqs):
            data = ds[seq_idx]
            t0_time = time.time()
            audio = np.asarray(data["audio"]["array"])
            sr = data["audio"]["sampling_rate"]
            words = list(data["word_detail"]["utterance"])
            collect: list = []
            with torch.no_grad():
                fwd = model(
                    raw_inputs=torch.tensor(audio)[None],
                    raw_inputs_sample_rate=sr,
                    raw_input_seq_lens=torch.tensor([len(audio)]),
                    raw_targets=[words],
                    raw_target_seq_lens=torch.tensor([len(words)]),
                    omitted_prev_context=torch.tensor([0]),
                    collect_attentions=collect,
                )
            attns = collect[0]["attns"]  # L x [H, S, n_audio]
            n_real = int(collect[0]["n_audio_real"])
            tse = fwd.target_start_end[0].cpu().numpy()
            del fwd
            n_layers, n_heads = len(attns), attns[0].shape[0]
            if wbe_sum is None:
                wbe_sum = np.zeros((n_layers, n_heads))
            sec_per_frame = (len(audio) / sr) / n_real

            # In-word token rows (mirror the grad extract's token enumeration).
            tok_rows = []
            tok_per_word = []
            for w in range(len(words)):
                a, b = int(tse[w, 0]), int(tse[w, 1])
                tok_rows.extend(range(a, b))
                tok_per_word.append(b - a)
            ref = [(s / sr, e / sr) for s, e in zip(data["word_detail"]["start"], data["word_detail"]["stop"])]

            # Char-level targets can exceed the ~80 ms audio-token frames (S>T) -> the DP
            # can't run. Upsample the time axis k=ceil(S/T)x so it can (boundaries then
            # resolve to sec_per_frame/up_k). Per-seq constant (tok_rows, n_real fixed).
            up_k = 1
            if self.time_upsample_when_short and len(tok_rows) > n_real:
                up_k = -(-len(tok_rows) // n_real)
            sec_per_frame_eff = sec_per_frame / up_k
            for li in range(n_layers):
                mat_all = attns[li].numpy()  # [H, S, n_audio]
                for hi in range(n_heads):
                    mat = mat_all[hi][tok_rows][:, :n_real]  # [n_tok, n_real]
                    if up_k > 1:
                        mat = np.repeat(mat, up_k, axis=1)
                    try:
                        spans = aligner.align(mat + 1e-8)
                    except Exception as exc:
                        # A genuinely degenerate head (flat scores) still aligns -- it
                        # just scores poorly. A RAISE here means a structural problem
                        # (e.g. S>T without upsampling): record it and fail loudly after
                        # the sweep rather than silently selecting a random head.
                        wbe_sum[li, hi] += 10.0
                        n_align_fail += 1
                        last_align_err = repr(exc)
                        continue
                    bounds = []
                    cur = 0
                    for k in tok_per_word:
                        # ends inclusive, same convention as the production align job
                        bounds.append((spans[cur][0] * sec_per_frame_eff, spans[cur + k - 1][1] * sec_per_frame_eff))
                        cur += k
                    errs = per_utt_boundary_errors(bounds, ref)
                    wbe_sum[li, hi] += float(np.mean(errs["wbe"]))
            wbe_cnt += 1
            print(f"seq {seq_idx}: scored ({time.time() - t0_time:.1f}s)", flush=True)

        mean_wbe = wbe_sum / max(wbe_cnt, 1)
        order = np.dstack(np.unravel_index(np.argsort(mean_wbe, axis=None), mean_wbe.shape))[0]
        top = [[int(li), int(hi)] for li, hi in order[: self.top_k]]
        report = {
            "top20": [{"layer": int(li), "head": int(hi), "wbe": float(mean_wbe[li, hi])} for li, hi in order[:20]],
            "median_wbe": float(np.median(mean_wbe)),
            "best_wbe": float(mean_wbe[order[0][0], order[0][1]]),
        }
        report["n_align_fail"] = int(n_align_fail)
        report["last_align_err"] = last_align_err
        print("REPORT:", report)
        # Fail loudly on a degenerate selection instead of silently emitting random heads.
        # A real alignment head scores ~0.05-0.20 s WBE on gold dev; >0.5 s means the DP
        # failed for (nearly) all heads -- e.g. char tokens exceed the audio-token grid
        # (S>T) and time_upsample_when_short was not set. See the upsample note.
        assert report["best_wbe"] < 0.5, (
            f"degenerate head selection: best_wbe={report['best_wbe']:.3f}s (expected ~0.1s); "
            f"{n_align_fail} per-head align failures, last error: {last_align_err}. "
            "Likely S>T (char tokens > audio frames) -- pass time_upsample_when_short=True."
        )
        self.out_heads.set(top)
        self.out_report.set(report)


# ---- shared HDF helpers (grad-extract schema consumed by WordAlignFromPerTokenGradsJob) ----
_ATTN_HDF_EXTRA_TYPE = {
    "audio_frames_start_end": (2, 2, "int32"),
    "num_input_frames": (1, 2, "int32"),
    "num_words": (1, 2, "int32"),
    "num_tokens": (1, 2, "int32"),
    "num_tokens_per_word": (1, 2, "int32"),
    "log_probs_per_token": (1, 2, "float32"),
    "exit_log_probs": (1, 2, "float32"),
}


def _make_attn_hdf_writer(out_path):
    from returnn.datasets.hdf import SimpleHDFWriter

    return SimpleHDFWriter(out_path, dim=1, ndim=2, extra_type=dict(_ATTN_HDF_EXTRA_TYPE))


def _attn_frames_se(audio_len, n_real):
    import numpy as np

    edges = np.arange(n_real + 1, dtype=np.float64) * (audio_len / max(n_real, 1))
    return np.stack([np.round(edges[:-1]), np.round(edges[1:])], axis=-1).astype("int32")


def _write_attn_hdf_seq(writer, seq_idx, grad_mat, n_real, n_words, tok_per_word, frames_se):
    """grad_mat: [n_tokens, n_real]; tok_per_word must sum to n_tokens (WordAlign requirement)."""
    import numpy as np

    n_tokens = grad_mat.shape[0]
    assert sum(tok_per_word) == n_tokens, f"{sum(tok_per_word)=} != {n_tokens=}"
    writer.insert_batch(
        grad_mat.reshape(1, -1, 1).astype("float32"),
        seq_len=[n_tokens * n_real],
        seq_tag=[f"seq-{seq_idx}"],
        extra={
            "audio_frames_start_end": frames_se[None],
            "num_input_frames": np.array([[[n_real]]], dtype="int32"),
            "num_words": np.array([[[n_words]]], dtype="int32"),
            "num_tokens": np.array([[[n_tokens]]], dtype="int32"),
            "num_tokens_per_word": np.array(tok_per_word, dtype="int32")[None, :, None],
            "log_probs_per_token": np.zeros((1, n_tokens, 1), dtype="float32"),
            "exit_log_probs": np.zeros((1, 1, 1), dtype="float32"),
        },
    )


class ExtractSelfAttnPerTokenJob(Job):
    """Mean attention matrix of the selected heads, in the grad-extract HDF schema."""

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_config: Dict[str, Any],
        heads: Union[tk.Variable, List[List[int]]],
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param heads: list of (layer, head) pairs, e.g. ``out_heads`` of
            :class:`SelectSelfAttnAlignHeadsJob`.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_config = model_config
        self.heads = heads
        self.returnn_root = returnn_root

        self.rqmt = {"time": 12, "cpu": 2, "gpu": 1, "mem": 80}
        self.out_hdf = self.output_path("out.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import time

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch

        from returnn.util import better_exchook
        from returnn.datasets.hdf import SimpleHDFWriter

        better_exchook.install()

        from .models import make_model

        heads = self.heads.get() if isinstance(self.heads, tk.Variable) else self.heads
        heads = [(int(li), int(hi)) for li, hi in heads]
        print("Heads:", heads)

        dev = torch.device("cuda")
        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)
        for p in model.parameters():
            p.requires_grad = False

        hdf_writer = _make_attn_hdf_writer(self.out_hdf.get_path())

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        print(f"Num seqs: {len(ds)}")

        for seq_idx, data in enumerate(ds):
            t0_time = time.time()
            audio = np.asarray(data["audio"]["array"])
            sr = data["audio"]["sampling_rate"]
            words = list(data["word_detail"]["utterance"])
            collect: list = []
            with torch.no_grad():
                fwd = model(
                    raw_inputs=torch.tensor(audio)[None],
                    raw_inputs_sample_rate=sr,
                    raw_input_seq_lens=torch.tensor([len(audio)]),
                    raw_targets=[words],
                    raw_target_seq_lens=torch.tensor([len(words)]),
                    omitted_prev_context=torch.tensor([0]),
                    collect_attentions=collect,
                )
            attns = collect[0]["attns"]
            n_real = int(collect[0]["n_audio_real"])
            tse = fwd.target_start_end[0].cpu().numpy()
            del fwd
            mat = np.mean([attns[li][hi].numpy() for li, hi in heads], axis=0)[:, :n_real]  # [S, n_real]

            rows = []
            tok_per_word = []
            for w in range(len(words)):
                a, b = int(tse[w, 0]), int(tse[w, 1])
                rows.append(mat[a:b])
                tok_per_word.append(b - a)
            grad_mat = np.concatenate(rows, axis=0)  # [n_tokens, n_real]
            n_tokens = grad_mat.shape[0]

            edges = np.arange(n_real + 1, dtype=np.float64) * (len(audio) / max(n_real, 1))
            frames_se = np.stack([np.round(edges[:-1]), np.round(edges[1:])], axis=-1).astype("int32")

            hdf_writer.insert_batch(
                grad_mat.reshape(1, -1, 1).astype("float32"),
                seq_len=[n_tokens * n_real],
                seq_tag=[f"seq-{seq_idx}"],
                extra={
                    "audio_frames_start_end": frames_se[None],
                    "num_input_frames": np.array([[[n_real]]], dtype="int32"),
                    "num_words": np.array([[[len(words)]]], dtype="int32"),
                    "num_tokens": np.array([[[n_tokens]]], dtype="int32"),
                    "num_tokens_per_word": np.array(tok_per_word, dtype="int32")[None, :, None],
                    "log_probs_per_token": np.zeros((1, n_tokens, 1), dtype="float32"),
                    "exit_log_probs": np.zeros((1, 1, 1), dtype="float32"),
                },
            )
            if seq_idx % 100 == 0:
                print(f"seq {seq_idx}: {n_tokens} tokens x {n_real} frames ({time.time() - t0_time:.2f}s)", flush=True)

        hdf_writer.close()


class ExtractSelfAttnWhisperJob(Job):
    """openai-whisper cross-attention -> grad-extract HDF (so WordAlignFromPerTokenGradsJob can align it).

    Mirrors openai ``find_alignment``'s matrix exactly
    (this is why it uses ``import whisper`` rather than the HF/rf model):
    per head, softmax over frames -> optional z-norm over tokens -> optional median filter,
    then mean over the selected heads.
    The DP / energy / silence / softmax / boundary read-off are left to WordAlign,
    so each align config is a cheap job on the cached HDF (no re-capture).
    Shares the HDF schema helpers with :class:`ExtractSelfAttnPerTokenJob`;
    a no-transform run should match it (sanity check).
    """

    def __init__(
        self,
        *,
        dataset_dir,
        dataset_key,
        overlay,
        heads,
        whisper_model="base",
        zscore=True,
        median_filter=True,
        num_seqs=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.overlay = overlay
        self.heads = heads
        self.whisper_model = whisper_model
        self.zscore = bool(zscore)
        self.median_filter = bool(median_filter)
        self.num_seqs = num_seqs
        self.rqmt = {"time": 6, "cpu": 2, "gpu": 0, "mem": 64}
        self.out_hdf = self.output_path("out.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import io
        import os
        import sys

        sys.path.insert(0, self.overlay)
        from i6_experiments.users.zeyer.external_models.huggingface import (
            set_hf_offline_mode,
            get_content_dir_from_hub_cache_dir,
        )

        set_hf_offline_mode()
        import i6_experiments

        sys.path.insert(0, os.path.dirname(os.path.dirname(i6_experiments.__file__)))
        import numpy as np
        import soundfile as sf
        import torch
        import whisper
        from whisper.timing import median_filter as _medfilt
        from whisper.model import disable_sdpa
        import datasets
        from datasets import load_dataset

        heads = self.heads.get() if isinstance(self.heads, tk.Variable) else self.heads
        heads = [(int(li), int(hi)) for li, hi in heads]
        dev = torch.device("cpu")
        model = whisper.load_model(self.whisper_model).to(dev).eval()
        tok = whisper.tokenizer.get_tokenizer(
            model.is_multilingual, num_languages=getattr(model, "num_languages", 99), language="en", task="transcribe"
        )
        n_sot = len(tok.sot_sequence)
        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        if isinstance(ds.features.get("audio"), datasets.Audio):
            ds = ds.cast_column("audio", datasets.Audio(decode=False))
        n = len(ds) if self.num_seqs is None else min(self.num_seqs, len(ds))

        writer = _make_attn_hdf_writer(self.out_hdf.get_path())
        for si in range(n):
            d = ds[si]
            a = d["audio"]
            if a.get("array") is not None:
                audio = np.asarray(a["array"], dtype=np.float32)
                sr = int(a["sampling_rate"])
            else:
                audio, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
                sr = int(sr)
                audio = np.asarray(audio, dtype=np.float32)
            words = list(d["word_detail"]["utterance"])
            wav = torch.tensor(audio)
            if sr != 16000:
                import torchaudio

                wav = torchaudio.functional.resample(wav[None], sr, 16000)[0]
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(wav), n_mels=model.dims.n_mels).to(dev)
            nf2 = (int(wav.shape[0]) // 160) // 2
            text_tokens = tok.encode(" " + " ".join(w.lower() for w in words))
            tokens = torch.tensor([*tok.sot_sequence, tok.no_timestamps, *text_tokens, tok.eot]).to(dev)
            QKs = [None] * model.dims.n_text_layer
            hooks = [
                b.cross_attn.register_forward_hook(lambda _, i_, o_, idx=i: QKs.__setitem__(idx, o_[-1][0]))
                for i, b in enumerate(model.decoder.blocks)
            ]
            with torch.no_grad(), disable_sdpa():
                model(mel.unsqueeze(0), tokens.unsqueeze(0))
            for h in hooks:
                h.remove()
            _w, wt_ = tok.split_to_word_tokens(text_tokens + [tok.eot])
            wb = np.pad(np.cumsum([len(t) for t in wt_[:-1]]), (1, 0))
            # openai find_alignment per-head transform, then mean over heads (== monolith matrix())
            w_ = torch.stack([QKs[li][hi] for li, hi in heads])[:, :, :nf2].softmax(dim=-1)
            if self.zscore:
                std, mean = torch.std_mean(w_, dim=-2, keepdim=True, unbiased=False)
                w_ = (w_ - mean) / std
            if self.median_filter:
                w_ = _medfilt(w_, 7)
            m = w_.mean(axis=0)[n_sot:-1].double().numpy()  # [n_sot:-1] rows, like the monolith DP
            # WordAlign needs sum(tok_per_word) == n_tokens:
            # keep the wb word spans,
            # drop the trailing extra row(s) beyond the last word boundary so the per-word grouping is exact.
            n_tok_words = int(wb[-1])
            grad_mat = m[:n_tok_words]
            tok_per_word = [int(wb[k + 1] - wb[k]) for k in range(len(wb) - 1)]
            frames_se = _attn_frames_se(int(wav.shape[0]), nf2)
            _write_attn_hdf_seq(writer, si, grad_mat, nf2, len(words), tok_per_word, frames_se)
            if si % 50 == 0:
                print(f"captured {si}/{n}", flush=True)
        writer.close()
