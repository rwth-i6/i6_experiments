"""Per-difference ablation of openai-whisper find_alignment vs our method.

Computes word-boundary error (WBE) for a ladder of configs that toggle, one at a time, each
difference between ``whisper.timing.find_alignment`` and our cross-attn aligner:
head set (whisper's curated ``alignment_heads`` vs our gold-tuned top-k), token z-norm, median
filter, log transform, DP (whisper true-DTW vs our Viterbi-with-blank), energy/silence, and the
boundary read-off (whisper token-index "jumps" vs our first/last-token span). All on the genuine
openai-whisper attention (so the faithful config reproduces find_alignment exactly). CPU-only.

The DTW path is validated bit-identical to whisper's dtw_cpu; the Viterbi rows use the real
``Aligner`` (so they match our production cross-attn aligner). Outputs one WBE Variable per config.
"""

from typing import Optional, Union, List
from sisyphus import Job, Task, tk

# (key, knobs). heads: "wh" = whisper official alignment_heads, "ours" = gold-tuned (passed in).
# dp: "dtw" (whisper true-DTW) or "viterbi" (our Aligner). readoff applies to dtw only.
CONFIGS = [
    ("faithful", dict(heads="wh", zscore=True, medfilt=True, log=False, dp="dtw", energy=False, readoff="jump")),
    ("span", dict(heads="wh", zscore=True, medfilt=True, log=False, dp="dtw", energy=False, readoff="span")),
    ("nomedfilt", dict(heads="wh", zscore=True, medfilt=False, log=False, dp="dtw", energy=False, readoff="jump")),
    ("noznorm", dict(heads="wh", zscore=False, medfilt=True, log=False, dp="dtw", energy=False, readoff="jump")),
    ("noznorm_log", dict(heads="wh", zscore=False, medfilt=True, log=True, dp="dtw", energy=False, readoff="jump")),
    ("ourheads", dict(heads="ours", zscore=True, medfilt=True, log=False, dp="dtw", energy=False, readoff="jump")),
    ("ourheads_span", dict(heads="ours", zscore=True, medfilt=True, log=False, dp="dtw", energy=False, readoff="span")),
    ("ourdp", dict(heads="ours", zscore=False, medfilt=False, log=True, dp="viterbi", energy=False, readoff="span")),
    ("full", dict(heads="ours", zscore=False, medfilt=False, log=True, dp="viterbi", energy=True, readoff="span")),
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
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 0, "mem": 16}
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
        aligner = Aligner(apply_softmax_over_time=True, blank_score=-5)
        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        ds = ds.cast_column("audio", datasets.Audio(decode=False))
        n = len(ds) if self.num_seqs is None else min(self.num_seqs, len(ds))

        def jump_collapse(ti, tj, wb):
            jm = np.pad(np.diff(ti), (1, 0), constant_values=1).astype(bool)
            jt = tj[jm] / TOKENS_PER_SECOND
            return list(zip(jt[wb[:-1]].tolist(), jt[wb[1:]].tolist()))

        def span_dtw(ti, tj, wb, n_tok):
            se = []
            for k in range(n_tok):
                fr = tj[ti == k]
                if len(fr) == 0:
                    p = se[-1][1] if se else 0
                    se.append((p, p))
                else:
                    se.append((int(fr.min()), int(fr.max()) + 1))
            return [(se[a][0] / TOKENS_PER_SECOND, se[b - 1][1] / TOKENS_PER_SECOND) for a, b in zip(wb[:-1], wb[1:])]

        # Capture per-seq attention + grids.
        seqs = []
        for si in range(n):
            d = ds[si]
            a = d["audio"]
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
            centers = ((np.arange(nf2) + 0.5) / nf2 * (len(env) - 1)).astype(int)
            e = env[centers]
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

        def evaluate(heads, zscore, medfilt, log, dp, energy, readoff):
            hd = heads_wh if heads == "wh" else heads_ours
            errs = []
            for QKs, nf2, wb, ref, n_tok, e in seqs:
                m = matrix(QKs, nf2, hd, zscore, medfilt)
                if dp == "dtw":
                    if log:
                        mx = m.max(1, keepdims=True)
                        m = m - (np.log(np.sum(np.exp(m - mx), 1, keepdims=True)) + mx)
                    ti, tj = _dtw_path(-m)
                    b = jump_collapse(ti, tj, wb) if readoff == "jump" else span_dtw(ti, tj, wb, n_tok)
                else:
                    mm = m.copy()
                    bo = None
                    if energy:
                        _s = np.log(np.maximum(mm, 1e-12))
                        _s = _s - max(float(_s.max()), 0.0)
                        _mx = _s.max(1, keepdims=True)
                        _s = _s - _mx - np.log(np.sum(np.exp(_s - _mx), 1, keepdims=True))
                        _ze = (e - e.mean()) / (e.std() + 1e-9)
                        bo = _s.mean(0) - 1.0 * _ze * _s.std(0)
                        mm = mm * (e[None, :] ** 0.5)
                    spans = aligner.align(mm + 1e-8, blank_override=bo)
                    b = [(spans[a][0] * spf, spans[bb - 1][1] * spf) for a, bb in zip(wb[:-1], wb[1:])]
                if len(b) == len(ref):
                    errs.append(float(np.mean(per_utt_boundary_errors(b, ref)["wbe"])))
            return float(np.mean(errs)), len(errs)

        lines = [f"model={self.whisper_model} key={self.dataset_key} n={n}"]
        lines.append(f"heads_wh={heads_wh}")
        lines.append(f"heads_ours={heads_ours}")
        for key, cfg in CONFIGS:
            wbe, c = evaluate(**cfg)
            self.out_wbes[key].set(wbe)
            lines.append(f"{key:16s} {wbe * 1000:7.2f} ms  ({c}/{n})  {cfg}")
            print(lines[-1], flush=True)
        with open(self.out_report.get_path(), "w") as f:
            f.write("\n".join(lines) + "\n")
