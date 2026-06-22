"""Per-difference ablation of openai-whisper find_alignment vs our cross-attn aligner, both directions.

Computes word-boundary error (WBE) for a ladder of configs toggling, one at a time,
each difference between ``whisper.timing.find_alignment`` and our method:
alignment-head set (Whisper's curated heads vs our gold-tuned top-k vs the single best head),
token z-norm, median filter, log transform,
the DP (Whisper monotonic DTW with the vertical/up transition vs our monotonic DP forbidding it),
the silence topology, energy weighting, and the boundary read-off.
Forward rows start from faithful Whisper;
reverse rows start from our setup and swap settings back.
All on the genuine openai-whisper attention -- the faithful config reproduces find_alignment exactly.
The DTW path is bit-identical to whisper's dtw_cpu;
the monotonic-DP rows use the real ``Aligner``. CPU-only.
"""

from typing import Optional, Union, List
from sisyphus import Job, Task, tk

# (key, knobs). heads: wh | ours | best1.  dp: dtw | mono.  silence: none | ctc (mono only).
CONFIGS = [
    # forward: Whisper (faithful) -> ours, one toggle at a time
    (
        "faithful",
        dict(heads="wh", zscore=True, medfilt=True, log=False, dp="dtw", energy=False, silence="none", readoff="jump"),
    ),
    (
        "span",
        dict(heads="wh", zscore=True, medfilt=True, log=False, dp="dtw", energy=False, silence="none", readoff="span"),
    ),
    (
        "nomedfilt",
        dict(heads="wh", zscore=True, medfilt=False, log=False, dp="dtw", energy=False, silence="none", readoff="jump"),
    ),
    (
        "noznorm",
        dict(heads="wh", zscore=False, medfilt=True, log=False, dp="dtw", energy=False, silence="none", readoff="jump"),
    ),
    (
        "noznorm_log",
        dict(heads="wh", zscore=False, medfilt=True, log=True, dp="dtw", energy=False, silence="none", readoff="jump"),
    ),
    (
        "ourheads",
        dict(
            heads="ours", zscore=True, medfilt=True, log=False, dp="dtw", energy=False, silence="none", readoff="jump"
        ),
    ),
    (
        "best1head",
        dict(
            heads="best1", zscore=True, medfilt=True, log=False, dp="dtw", energy=False, silence="none", readoff="jump"
        ),
    ),
    # reverse: ours (full, word-topology) -> swap ONE setting at a time back toward Whisper.
    # mono+silence are a coupled axis: DTW has no blank states, so it forces silence=none;
    # the mono(none) -> DTW step isolates just the vertical/up transition.
    (
        "ours_full",
        dict(
            heads="ours", zscore=False, medfilt=False, log=True, dp="mono", energy=True, silence="word", readoff="span"
        ),
    ),
    (
        "ours_ctc",
        dict(
            heads="ours", zscore=False, medfilt=False, log=True, dp="mono", energy=True, silence="ctc", readoff="span"
        ),
    ),
    (
        "ours_none",
        dict(
            heads="ours", zscore=False, medfilt=False, log=True, dp="mono", energy=True, silence="none", readoff="span"
        ),
    ),
    (
        "ours_dtw",
        dict(
            heads="ours", zscore=False, medfilt=False, log=True, dp="dtw", energy=True, silence="none", readoff="span"
        ),
    ),
    (
        "ours_dtw_noen",
        dict(
            heads="ours", zscore=False, medfilt=False, log=True, dp="dtw", energy=False, silence="none", readoff="span"
        ),
    ),
    # faithful-origin toggle: add energy to the faithful DTW
    # (stays on the DTW branch, so the matrix is compatible).
    # NOTE: switching the faithful transform onto our monotonic/blank DP is NOT a clean single toggle --
    # whisper's z-norm-over-tokens transform is a DTW cost, not softmax-able emission scores,
    # so our Aligner degenerates on it.
    # The mono/silence axes are isolated from the OURS end instead
    # (ours_full vs ours_dtw = DP step; ours_full vs ours_none = silence).
    (
        "faithful_energy",
        dict(heads="wh", zscore=True, medfilt=True, log=False, dp="dtw", energy=True, silence="none", readoff="jump"),
    ),
    # From no-z-norm + log (a softmax-compatible matrix) switch to our DP, then add word silence.
    # (z-norm makes the matrix incompatible with our Aligner's softmax-over-time, so we branch from here.)
    (
        "noznorm_log_mono",
        dict(heads="wh", zscore=False, medfilt=True, log=True, dp="mono", energy=False, silence="none", readoff="span"),
    ),
    (
        "noznorm_log_mono_sil",
        dict(heads="wh", zscore=False, medfilt=True, log=True, dp="mono", energy=False, silence="word", readoff="span"),
    ),
    (
        "faithful_mono",
        dict(
            heads="wh",
            zscore=True,
            medfilt=True,
            log=False,
            dp="mono",
            energy=False,
            silence="none",
            readoff="span",
            softmax=False,
        ),
    ),
    (
        "faithful_silence",
        dict(
            heads="wh",
            zscore=True,
            medfilt=True,
            log=False,
            dp="mono",
            energy=False,
            silence="word",
            readoff="span",
            softmax=False,
        ),
    ),
]


def _dtw_path(cost):
    import numpy as np

    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    tr = np.zeros((n + 1, m + 1), dtype=np.int8)
    d[0, 0] = 0.0
    tr[0, :] = 2
    tr[:, 0] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c0, c1, c2 = d[i - 1, j - 1], d[i - 1, j], d[i, j - 1]
            if c0 <= c1 and c0 <= c2:
                d[i, j], tr[i, j] = cost[i - 1, j - 1] + c0, 0
            elif c1 <= c2:
                d[i, j], tr[i, j] = cost[i - 1, j - 1] + c1, 1
            else:
                d[i, j], tr[i, j] = cost[i - 1, j - 1] + c2, 2
    i, j, T, J = n, m, [], []
    while i > 0 or j > 0:
        T.append(i - 1)
        J.append(j - 1)
        t = tr[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return np.array(T[::-1]), np.array(J[::-1])


class WhisperDtwAblationJob(Job):
    __sis_version__ = 3  # Aligner mono rows: word end now exclusive (max+1)

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        overlay: str,
        heads_ours: Union[tk.Variable, List[List[int]]],
        whisper_model: str = "base",
        num_seqs: Optional[int] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.overlay = overlay
        self.heads_ours = heads_ours
        self.whisper_model = whisper_model
        self.num_seqs = num_seqs
        self.returnn_root = returnn_root
        self.rqmt = {"time": 6, "cpu": 2, "gpu": 0, "mem": 64}
        self.out_wbes = {key: self.output_var(f"wbe-{key}.txt") for key, _ in CONFIGS}
        self.out_report = self.output_path("report.txt")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import io

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
        from whisper.timing import median_filter, TOKENS_PER_SECOND
        from whisper.model import disable_sdpa
        import datasets
        from datasets import load_dataset
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            per_utt_boundary_errors,
        )
        from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import Aligner

        spf = 1.0 / TOKENS_PER_SECOND
        heads_ours = self.heads_ours.get() if isinstance(self.heads_ours, tk.Variable) else self.heads_ours
        heads_ours = [[int(a), int(b)] for a, b in heads_ours]

        dev = torch.device("cpu")
        model = whisper.load_model(self.whisper_model).to(dev).eval()
        heads_wh = [[int(a), int(b)] for a, b in model.alignment_heads.indices().T.tolist()]
        tok = whisper.tokenizer.get_tokenizer(
            model.is_multilingual, num_languages=getattr(model, "num_languages", 99), language="en", task="transcribe"
        )
        n_sot = len(tok.sot_sequence)
        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        # TIMIT stores audio as encoded files (cast to non-decoding -> bytes/path);
        # Buckeye-fine stores it as a decoded float-array struct -> no cast, read the array directly.
        if isinstance(ds.features.get("audio"), datasets.Audio):
            ds = ds.cast_column("audio", datasets.Audio(decode=False))
        n = len(ds) if self.num_seqs is None else min(self.num_seqs, len(ds))

        def lsm(m):
            mx = m.max(1, keepdims=True)
            return m - (np.log(np.sum(np.exp(m - mx), 1, keepdims=True)) + mx)

        # Capture per-seq attention + grids (one forward each).
        seqs = []
        # rank single heads by mean WBE on this set, faithful transform, to pick best1 (lowest).
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
            ref = [(s / sr, e / sr) for s, e in zip(d["word_detail"]["start"], d["word_detail"]["stop"])]
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
            aud = np.asarray(wav.numpy(), dtype=np.float64)
            win = max(int(0.025 * 16000), 1)
            hann = np.hanning(win)
            hann = hann / hann.sum()
            env = np.sqrt(np.convolve(aud * aud, hann, mode="same") + 1e-9)
            e = env[((np.arange(nf2) + 0.5) / nf2 * (len(env) - 1)).astype(int)]
            e = e / (e.max() + 1e-9)
            seqs.append(([q.cpu() for q in QKs], nf2, wb, ref, len(text_tokens), e))
            if si % 50 == 0:
                print(f"captured {si}/{n}", flush=True)

        def matrix(QKs, nf2, heads, zscore, medfilt):
            w = torch.stack([QKs[layer][head] for layer, head in heads])[:, :, :nf2].softmax(dim=-1)
            if zscore:
                std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
                w = (w - mean) / std
            if medfilt:
                w = median_filter(w, 7)
            return w.mean(axis=0)[n_sot:-1].double().numpy()

        def jump(ti, tj, wb):
            jm = np.pad(np.diff(ti), (1, 0), constant_values=1).astype(bool)
            jt = tj[jm] / TOKENS_PER_SECOND
            return list(zip(jt[wb[:-1]].tolist(), jt[wb[1:]].tolist()))

        def span_dtw(ti, tj, wb, nt):
            se = []
            for k in range(nt):
                fr = tj[ti == k]
                se.append((int(fr.min()), int(fr.max()) + 1) if len(fr) else (se[-1][1] if se else 0,) * 2)
            return [(se[a][0] / TOKENS_PER_SECOND, se[b - 1][1] / TOKENS_PER_SECOND) for a, b in zip(wb[:-1], wb[1:])]

        # pick best single head (lowest WBE, faithful transform, dtw, jump) over a head-rank pass.
        cand = sorted({tuple(h) for h in heads_wh + heads_ours})
        best1, best1_wbe = None, 1e9
        for hh in cand:
            errs = []
            for QKs, nf2, wb, ref, nt, e in seqs:
                m = matrix(QKs, nf2, [list(hh)], True, True)
                ti, tj = _dtw_path(-m)
                b = jump(ti, tj, wb)
                if len(b) == len(ref):
                    errs.append(float(np.mean(per_utt_boundary_errors(b, ref)["wbe"])))
            w = float(np.mean(errs)) if errs else 1e9
            if w < best1_wbe:
                best1, best1_wbe = [list(hh)], w

        head_map = {"wh": heads_wh, "ours": heads_ours, "best1": best1}

        def evaluate(heads, zscore, medfilt, log, dp, energy, silence, readoff, softmax=True):
            hd = head_map[heads]
            al = Aligner(apply_softmax_over_time=softmax, blank_score=-5)
            errs = []
            for QKs, nf2, wb, ref, nt, e in seqs:
                m = matrix(QKs, nf2, hd, zscore, medfilt)
                if dp == "dtw":
                    # energy-weight the matrix before DTW (with or without the log transform),
                    # so faithful+energy (log off) is meaningful rather than a no-op.
                    p = m * (e[None, :] ** 0.5) if energy else m
                    if log:
                        p = lsm(p)
                    ti, tj = _dtw_path(-p)
                    b = jump(ti, tj, wb) if readoff == "jump" else span_dtw(ti, tj, wb, nt)
                else:  # our DP: monotonic (dp=mono) or DTW-transition (dp=adtw); both carry silence states
                    mm = m.copy()
                    bo, mask = None, None
                    if energy:
                        _s = lsm(np.log(np.maximum(mm, 1e-12)))
                        _ze = (e - e.mean()) / (e.std() + 1e-9)
                        bo = _s.mean(0) - 1.0 * _ze * _s.std(0)
                        mm = mm * (e[None, :] ** 0.5)
                    if silence == "none":
                        R = mm.shape[0]
                        mask = np.array([(i == 0 or i == R) for i in range(R + 1)], dtype=bool)
                        bo = None
                    elif silence == "word":
                        R = mm.shape[0]
                        allowed = {0, R} | {int(x) + 1 for x in wb}
                        mask = np.array([(i in allowed) for i in range(R + 1)], dtype=bool)
                    spans = al.align(mm + 1e-8, blank_override=bo, blank_state_mask=mask)
                    b = [(spans[a][0] * spf, spans[bb - 1][1] * spf) for a, bb in zip(wb[:-1], wb[1:])]
                if len(b) == len(ref):
                    errs.append(float(np.mean(per_utt_boundary_errors(b, ref)["wbe"])))
            return float(np.mean(errs)), len(errs)

        lines = [f"model={self.whisper_model} key={self.dataset_key} n={n}"]
        lines.append(f"heads_wh={heads_wh}")
        lines.append(f"heads_ours={heads_ours}")
        lines.append(f"best1_head={best1} (WBE {best1_wbe * 1000:.1f} ms)")
        for key, cfg in CONFIGS:
            wbe, c = evaluate(**cfg)
            self.out_wbes[key].set(wbe)
            lines.append(f"{key:16s} {wbe * 1000:7.2f} ms  ({c}/{n})  {cfg}")
            print(lines[-1], flush=True)
        with open(self.out_report.get_path(), "w") as f:
            f.write("\n".join(lines) + "\n")
