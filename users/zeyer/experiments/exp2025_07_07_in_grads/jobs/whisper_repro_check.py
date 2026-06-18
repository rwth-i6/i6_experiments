"""Reproduction check: does our DP + boundary read-off match openai-whisper find_alignment?

Runs the real ``whisper.timing.find_alignment`` (authoritative whisper word timings) while
monkeypatching ``whisper.timing.dtw`` to capture the EXACT internal ``[token x time]`` matrix it
feeds the DP (softmax-over-time + token z-norm + median-filter-7 + mean over the official
alignment heads). Then runs OUR copy of the same DTW recursion on the identical matrix, with BOTH
boundary read-offs:

* ``jump``  -- whisper's: word start = time index where the word's first token first appears on the
  path (``jumps = diff(text_indices)``), word end = the next word's first-token time.
* ``span``  -- ours: (first-token min time, last-token max time).

So if ``ours-jump == whisper`` the DP is bit-identical and the only difference is the collapse; the
``ours-span`` vs ``whisper`` WBE gap then quantifies the read-off choice. CPU-only debug job.
"""

from typing import Optional
from sisyphus import Job, Task, tk


def _dtw_path(cost):
    """openai-whisper dtw_cpu (diag / up / left), copied. Returns (text_indices, time_indices)."""
    import numpy as np

    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    tr = np.zeros((n + 1, m + 1), dtype=np.int8)
    d[0, 0] = 0.0
    tr[0, :] = 2
    tr[:, 0] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c0, c1, c2 = d[i - 1, j - 1], d[i - 1, j], d[i, j - 1]
            if c0 <= c1 and c0 <= c2:
                d[i, j] = cost[i - 1, j - 1] + c0
                tr[i, j] = 0
            elif c1 <= c2:
                d[i, j] = cost[i - 1, j - 1] + c1
                tr[i, j] = 1
            else:
                d[i, j] = cost[i - 1, j - 1] + c2
                tr[i, j] = 2
    i, j = n, m
    text, time = [], []
    while i > 0 or j > 0:
        text.append(i - 1)
        time.append(j - 1)
        t = tr[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    text.reverse()
    time.reverse()
    return np.array(text), np.array(time)


class WhisperReproCheckJob(Job):
    def __init__(self, *, dataset_dir, dataset_key, overlay, whisper_model="base", num_seqs=10, returnn_root=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.overlay = overlay
        self.whisper_model = whisper_model
        self.num_seqs = num_seqs
        self.returnn_root = returnn_root
        self.rqmt = {"time": 2, "cpu": 2, "gpu": 0, "mem": 16}
        self.out_report = self.output_path("report.txt")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys

        sys.path.insert(0, self.overlay)
        from i6_experiments.users.zeyer.external_models.huggingface import (
            set_hf_offline_mode,
            get_content_dir_from_hub_cache_dir,
        )

        set_hf_offline_mode()
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)
        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch
        import whisper
        import whisper.timing as wt
        from whisper.timing import find_alignment, TOKENS_PER_SECOND
        from datasets import load_dataset
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            per_utt_boundary_errors,
        )

        dev = torch.device("cpu")
        model = whisper.load_model(self.whisper_model).to(dev).eval()
        tokenizer = whisper.tokenizer.get_tokenizer(
            model.is_multilingual,
            num_languages=getattr(model, "num_languages", 99),
            language="en",
            task="transcribe",
        )
        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        n = min(self.num_seqs, len(ds))

        def span_collapse(text_idx, time_idx, word_boundaries, n_tok):
            # ours: per token [min,max] frame, then word = (first-token.start, last-token.end).
            tok_se = []
            for ti in range(n_tok):
                fr = time_idx[text_idx == ti]
                if len(fr) == 0:
                    prev = tok_se[-1][1] if tok_se else 0
                    tok_se.append((prev, prev))
                else:
                    tok_se.append((int(fr.min()), int(fr.max()) + 1))
            out = []
            for a, b in zip(word_boundaries[:-1], word_boundaries[1:]):
                out.append((tok_se[a][0] / TOKENS_PER_SECOND, tok_se[b - 1][1] / TOKENS_PER_SECOND))
            return out

        def jump_collapse(text_idx, time_idx, word_boundaries):
            # whisper: jump times at token-index increments.
            jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
            jump_times = time_idx[jumps] / TOKENS_PER_SECOND
            starts = jump_times[word_boundaries[:-1]]
            ends = jump_times[word_boundaries[1:]]
            return list(zip(starts.tolist(), ends.tolist()))

        lines = []
        agg = {"whisper": [], "ours_jump": [], "ours_span": []}
        path_mismatch = 0
        for si in range(n):
            data = ds[si]
            audio = np.asarray(data["audio"]["array"], dtype=np.float32)
            sr = int(data["audio"]["sampling_rate"])
            words = list(data["word_detail"]["utterance"])
            wav = torch.tensor(audio)
            if sr != 16000:
                import torchaudio

                wav = torchaudio.functional.resample(wav[None], sr, 16000)[0]
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(wav), n_mels=model.dims.n_mels).to(dev)
            num_frames = int(wav.shape[0]) // 160
            text = " " + " ".join(w.lower() for w in words)
            text_tokens = tokenizer.encode(text)
            ref = [(s / sr, e / sr) for s, e in zip(data["word_detail"]["start"], data["word_detail"]["stop"])]

            # capture the exact matrix find_alignment feeds dtw, and its path.
            cap = {}
            orig = wt.dtw

            def _dtw_capture(x, _orig=orig, _cap=cap):
                _cap["neg_matrix"] = np.asarray(x).copy()
                out = _orig(x)
                _cap["path"] = (np.asarray(out[0]).copy(), np.asarray(out[1]).copy())
                return out

            wt.dtw = _dtw_capture
            try:
                alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames)
            finally:
                wt.dtw = orig

            # whisper word grouping (must match find_alignment's internal split).
            _words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
            word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

            matrix = -cap["neg_matrix"]
            n_tok = matrix.shape[0]
            # our DP on the identical matrix (whisper feeds dtw(-matrix); cost = -matrix, no log).
            ti_o, tj_o = _dtw_path(-matrix.astype(np.float64))
            ti_w, tj_w = cap["path"]
            same_path = ti_o.shape == ti_w.shape and bool(np.all(ti_o == ti_w)) and bool(np.all(tj_o == tj_w))
            if not same_path:
                path_mismatch += 1

            b_whisper = [(wt_.start, wt_.end) for wt_ in alignment]
            b_jump = jump_collapse(ti_o, tj_o, word_boundaries)
            b_span = span_collapse(ti_o, tj_o, word_boundaries, n_tok)

            ok = len(b_whisper) == len(ref) and len(b_jump) == len(ref) and len(b_span) == len(ref)
            wstr = ""
            if ok:
                ew = float(np.mean(per_utt_boundary_errors(b_whisper, ref)["wbe"]))
                ej = float(np.mean(per_utt_boundary_errors(b_jump, ref)["wbe"]))
                es = float(np.mean(per_utt_boundary_errors(b_span, ref)["wbe"]))
                agg["whisper"].append(ew)
                agg["ours_jump"].append(ej)
                agg["ours_span"].append(es)
                # max |whisper - ours_jump| boundary disagreement (s) -- should be ~0 if faithful.
                dj = max(abs(a[0] - c[0]) for a, c in zip(b_whisper, b_jump))
                de = max(abs(a[1] - c[1]) for a, c in zip(b_whisper, b_jump))
                wstr = f"WBE whisper={ew * 1000:.1f} ours_jump={ej * 1000:.1f} ours_span={es * 1000:.1f} ms | maxΔ(whisper,jump) start={dj * 1000:.2f} end={de * 1000:.2f} ms"
            lines.append(
                f"seq {si}: {len(words)}w {n_tok}tok same_path={same_path} "
                f"lens(whisper/jump/span/ref)={len(b_whisper)}/{len(b_jump)}/{len(b_span)}/{len(ref)} | {wstr}"
            )
            print(lines[-1], flush=True)

        def mean_ms(k):
            return f"{1000 * np.mean(agg[k]):.1f}" if agg[k] else "n/a"

        summary = (
            f"\n=== SUMMARY ({len(agg['whisper'])}/{n} seqs scored) ===\n"
            f"path_mismatch_seqs: {path_mismatch}\n"
            f"mean WBE: whisper={mean_ms('whisper')}  ours_jump={mean_ms('ours_jump')}  ours_span={mean_ms('ours_span')} ms\n"
            f"(ours_jump should == whisper if our DP+collapse is faithful; ours_span isolates the read-off.)\n"
        )
        print(summary, flush=True)
        with open(self.out_report.get_path(), "w") as f:
            f.write("\n".join(lines) + "\n" + summary)
