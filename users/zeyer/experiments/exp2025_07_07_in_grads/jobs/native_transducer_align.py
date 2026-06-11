"""Native transducer Viterbi alignment baseline.

The transducer's OWN alignment: Viterbi over the teacher-forced RNN-T (or TDT)
lattice — the best monotonic blank/label path — read out as per-token emit frames.
This is the model's built-in alignment path, quantized to the encoder frame grid
(80 ms for the fastconformer parakeets),
the same-model counterpart of the 10 ms-grid grad-align rows.

Boundaries HDF format matches the other forced-align baselines
(float seconds, [n_words, 2] per seq, tags ``seq-{idx}``)
-> :class:`CalcAlignmentMetricsFromWordBoundariesJob` consumes it unchanged.
"""

from typing import Any, Dict, Optional
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class NativeTransducerAlignJob(Job):
    """Viterbi-align each sequence over the transducer's teacher-forced lattice.

    RNN-T: ``psi[t,u] = max(psi[t-1,u] + blank, psi[t,u-1] + label)``.
    TDT: transitions carry a duration ``d`` from the model's duration set
    (blank requires ``d >= 1``, as in decoding).
    Word boundary = frame span from the first to the last token-emit frame of the word.
    Terminal: best ``psi[t, U]`` over all t
    (TDT paths need not land exactly on the last frame).
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_config: Dict[str, Any],
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_config = model_config
        self.returnn_root = returnn_root

        self.rqmt = {"time": 12, "cpu": 2, "gpu": 1, "mem": 64}
        self.out_word_boundaries_hdf = self.output_path("word_boundaries.hdf")

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

        dev = torch.device("cuda")
        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)
        for p in model.parameters():
            p.requires_grad = False

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        print(f"Num seqs: {len(ds)}")

        writer = SimpleHDFWriter(self.out_word_boundaries_hdf.get_path(), dim=2, ndim=2)

        for seq_idx, data in enumerate(ds):
            t0_time = time.time()
            audio = np.asarray(data["audio"]["array"])
            sr = data["audio"]["sampling_rate"]
            words = list(data["word_detail"]["utterance"])

            lattice: list = []
            with torch.no_grad():
                fwd = model(
                    raw_inputs=torch.tensor(audio)[None],
                    raw_inputs_sample_rate=sr,
                    raw_input_seq_lens=torch.tensor([len(audio)]),
                    raw_targets=[words],
                    raw_target_seq_lens=torch.tensor([len(words)]),
                    omitted_prev_context=torch.tensor([0]),
                    collect_lattice=lattice,
                )
            lat = lattice[0]
            tse = fwd.target_start_end[0].cpu().numpy()  # [n_words+1, 2] (incl exit slot)
            n_tokens = int(tse[len(words) - 1, 1])
            ys = fwd.targets[0, :n_tokens].cpu().numpy()
            del fwd

            emit_frame = self._viterbi_emit_frames(
                lat["log_probs"], lat["log_dur"], lat["durations"], ys, model.blank_idx
            )
            t_enc = lat["log_probs"].shape[0]
            sec_per_frame = (len(audio) / sr) / t_enc
            bounds = []
            for w in range(len(words)):
                a, b = int(tse[w, 0]), int(tse[w, 1])
                bounds.append((emit_frame[a] * sec_per_frame, (emit_frame[b - 1] + 1) * sec_per_frame))
            writer.insert_batch(np.array([bounds], dtype="float32"), [len(words)], [f"seq-{seq_idx}"])
            if seq_idx % 100 == 0:
                print(f"seq {seq_idx}: {len(words)} words T={t_enc} ({time.time() - t0_time:.2f}s)", flush=True)

        writer.close()

    @staticmethod
    def _viterbi_emit_frames(lp, ld, durations, ys, blank_idx):
        """Best path over the lattice; returns per-token emit frame indices [U]."""
        import numpy as np

        neg = -1.0e30
        t_max, u1 = lp.shape[0], lp.shape[1]
        u_len = len(ys)
        assert u1 >= u_len + 1
        logblank = lp[:, :, blank_idx]  # [T, U+1]
        loglab = lp[np.arange(t_max)[:, None], np.arange(u_len)[None, :], ys[None, :]]  # [T, U]

        psi = np.full((t_max, u_len + 1), neg)
        if ld is None:
            # RNN-T: blank (t-1,u) vs label (t,u-1).
            bp = np.zeros((t_max, u_len + 1), dtype=np.int8)  # 0 = blank, 1 = label
            psi[0, 0] = 0.0
            for u in range(1, u_len + 1):
                psi[0, u] = psi[0, u - 1] + loglab[0, u - 1]
                bp[0, u] = 1
            for t in range(1, t_max):
                psi[t, 0] = psi[t - 1, 0] + logblank[t - 1, 0]
                for u in range(1, u_len + 1):
                    a = psi[t - 1, u] + logblank[t - 1, u]
                    b = psi[t, u - 1] + loglab[t, u - 1]
                    if a >= b:
                        psi[t, u] = a
                    else:
                        psi[t, u] = b
                        bp[t, u] = 1
            t = int(t_max - 1)
            u = u_len
            emit = [0] * u_len
            while u > 0:
                if bp[t, u]:
                    u -= 1
                    emit[u] = t
                else:
                    t -= 1
            return emit

        # TDT: every transition carries a duration d (blank: d >= 1; token: any d incl 0).
        bp_kind = np.zeros((t_max, u_len + 1), dtype=np.int8)  # 0 = blank, 1 = label
        bp_d = np.zeros((t_max, u_len + 1), dtype=np.int8)
        psi[0, 0] = 0.0
        for t in range(t_max):
            for u in range(u_len + 1):
                if t == 0 and u == 0:
                    continue
                best = neg
                best_kind = 0
                best_d = 0
                for d_i, d in enumerate(durations):
                    if 1 <= d <= t:
                        c = psi[t - d, u] + logblank[t - d, u] + ld[t - d, u, d_i]
                        if c > best:
                            best, best_kind, best_d = c, 0, d
                        if u > 0:
                            c = psi[t - d, u - 1] + loglab[t - d, u - 1] + ld[t - d, u - 1, d_i]
                            if c > best:
                                best, best_kind, best_d = c, 1, d
                    if d == 0 and u > 0:
                        c = psi[t, u - 1] + loglab[t, u - 1] + ld[t, u - 1, d_i]
                        if c > best:
                            best, best_kind, best_d = c, 1, 0
                psi[t, u] = best
                bp_kind[t, u] = best_kind
                bp_d[t, u] = best_d
        t = int(np.argmax(psi[:, u_len]))
        u = u_len
        emit = [0] * u_len
        while u > 0:
            d = int(bp_d[t, u])
            if bp_kind[t, u]:
                u -= 1
                emit[u] = t - d  # token emitted at the source frame
                t -= d
            else:
                t -= d
        return emit
