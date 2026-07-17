"""SAE §1c — measure the residual-silence rate that p_sil has to match (dev, evaluation-only).

Why this exists: wav2vec-U 2.0 uses p_sil=0.5 and 1.0 uses 0.25, but those are tuned to *their* rVAD
residue. Ours is not theirs -- our measured rVAD recall is 0.73, and strongly length-dependent (0.13
for 1-2 frame gaps, 0.81 for >=520 ms), so the silence surviving the trim is an empirical quantity.
Inheriting 0.5 would be folklore; SAE_PLAN §1.0 asks for a {0.25, 0.5} sweep, and this says which
arm should win and why.

The quantity to match is a **token** rate, not a frame rate: the generator collapses consecutive
equal-argmax frames, so one surviving pause becomes one <SIL> token regardless of length. So the
comparable number is

    sil_token_rate = (# gold silence RUNS surviving the trim) / (# phone tokens + # silence runs)

measured against the text side's `sil_token_rate` (PhonemizeWithSilJob stats).

Gold (MFA) is used strictly as measurement, never for training or selection: this job reports
numbers into SAE_1c.md so a p_sil is chosen for a stated reason. Note the reference phone tokens
come from the *untrimmed* alignment, so frames rVAD wrongly deletes count against us rather than
being hidden -- the same convention the §1c PER eval uses.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from sisyphus import Job, Task, tk

_alias_prefix = "sae/1c"


class ResidualSilenceStatsJob(Job):
    """Gold-silence structure of rVAD-trimmed dev frames -> the p_sil the text side should use."""

    def __init__(self, *, split: str = "validation.other", vad_threshold: float = 0.4,
                 limit: Optional[int] = None):
        super().__init__()
        self.split = split
        self.vad_threshold = vad_threshold
        self.limit = limit
        self.out_json = self.output_path("residual_silence.json")
        self.rqmt = {"cpu": 4, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from rVADfast import rVADfast

        from i6_experiments.users.wu.experiments.ssl.analysis import repr_audit as RA
        from i6_experiments.users.wu.experiments.unsupervised_asr import vad_port

        vad = rVADfast(vad_threshold=self.vad_threshold)
        recs = vad_port._load_gilkeyio(self.split, limit=self.limit)

        n_utt = 0
        tot_phone_tok = tot_sil_runs = 0            # trimmed stream, token level
        tot_frames = tot_kept = tot_sil_frames_kept = 0
        tot_ref_phone_tok = 0                       # untrimmed reference (the PER denominator)

        for wav, phonemes in recs:  # _load_gilkeyio yields (waveform, phonemes) tuples
            # 640 samples = one 40 ms frame at 16 kHz; same convention as vad_port.validate_records.
            n_frames = len(wav) // 640
            if n_frames < 3:
                continue
            gold = RA.frame_phone_labels(phonemes, n_frames)          # [T], SIL = RA.SIL_ID
            sil = vad_port.rvad_silence_25hz(wav, vad=vad)
            sil = sil[:n_frames] if len(sil) >= n_frames else np.concatenate(
                [sil, np.ones(n_frames - len(sil), dtype=bool)])

            kept = gold[~sil]
            if len(kept) == 0:
                continue
            toks = RA.run_length_dedup(kept)
            n_sil_runs = int((toks == RA.SIL_ID).sum())
            n_phone_tok = int((toks != RA.SIL_ID).sum())

            tot_phone_tok += n_phone_tok
            tot_sil_runs += n_sil_runs
            tot_frames += n_frames
            tot_kept += len(kept)
            tot_sil_frames_kept += int((kept == RA.SIL_ID).sum())
            tot_ref_phone_tok += len(RA.gold_phone_tokens(gold, drop_sil=True))
            n_utt += 1

        denom = tot_phone_tok + tot_sil_runs
        out = {
            "split": self.split,
            "utts": n_utt,
            "vad_threshold": self.vad_threshold,
            # the number to compare against the text side's sil_token_rate
            "sil_token_rate_trimmed": tot_sil_runs / max(denom, 1),
            "sil_runs": tot_sil_runs,
            "phone_tokens_trimmed": tot_phone_tok,
            # frame-level context
            "frames_total": tot_frames,
            "frames_kept": tot_kept,
            "vad_dropped_frac": 1.0 - tot_kept / max(tot_frames, 1),
            "residual_sil_frame_rate": tot_sil_frames_kept / max(tot_kept, 1),
            # how many gold phones rVAD destroys outright (trimmed vs untrimmed reference)
            "ref_phone_tokens_untrimmed": tot_ref_phone_tok,
            "phone_token_loss_frac": 1.0 - tot_phone_tok / max(tot_ref_phone_tok, 1),
        }
        with open(self.out_json.get_path(), "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2), flush=True)


def register_residual_silence_stats(splits=("validation.clean", "validation.other"), limit=None):
    for s in splits:
        j = ResidualSilenceStatsJob(split=s, limit=limit)
        j.add_alias(f"{_alias_prefix}/residual_silence/{s}")
        tk.register_output(f"{_alias_prefix}/residual_silence_{s}.json", j.out_json)
