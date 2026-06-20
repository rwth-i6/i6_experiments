"""
Reusable CIF-segmentation diagnostic probe for the two-level pretrained model.

Two pieces, generic over arm / checkpoint / split so any future two-level run can be probed in one call:
  * :func:`seg_diag_forward` -- a ``ReturnnForwardJobV2`` that runs ``two_level.two_level_segdiag_v1`` over a
    (labelled) split and dumps ``seg_diag.pkl`` (per-utterance CIF diagnostics);
  * :class:`SegDiagAnalysisJob` -- consumes that pickle and writes ``metrics.json`` + ``summary.txt`` (+ plots
    if matplotlib is available): realised frames/token vs the dialled rate, the CTC-feasibility K/N margin,
    codebook usage / collapse, unmasked code-reconstruction accuracy, and a fire-vs-acoustic-change boundary
    score.
  * :func:`register_seg_diag` -- wires forward -> analysis and registers the outputs for one checkpoint.

Wire it into a Sisyphus config via ``experiments/pretrain_two_level/seg_diag.py``.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Optional, Sequence

from sisyphus import Job, Task, tk

from i6_core.returnn.forward import ReturnnForwardJobV2

from ..config import get_forward_config
from ..data import datasets as ds
from ..default_tools import RETURNN_EXE, RETURNN_ROOT

NETWORK_MODULE = "two_level.two_level_v1"  # the pretrain Model (loaded 1:1 from the pretrained ckpt)
SEGDIAG_MODULE = "two_level.two_level_segdiag_v1"  # forward_step + ForwardCallback
FRAME_MS = 40.0  # 25 Hz lower-encoder frame period (one CIF input frame)


def seg_diag_forward(
    prefix: str,
    *,
    checkpoint: tk.Path,
    model_config_dict: Dict[str, Any],
    codebook,
    num_clusters: int,
    vocab_size: int,
    spm_model,
    split: Sequence[str],
    max_examples: int = 64,
    batch_size_sec: int = 300,
    returnn_exe: tk.Path = RETURNN_EXE,
    returnn_root: tk.Path = RETURNN_ROOT,
) -> tk.Path:
    """Run the segmentation-diagnostic forward over ``split``; returns the ``seg_diag.pkl`` path."""
    forward_config = get_forward_config(
        forward_dataset=ds.labeled_hf_dataset(split, spm_model=spm_model, seq_ordering="default"),
        network_module=NETWORK_MODULE,
        net_args={"model_config_dict": model_config_dict, "codebook": codebook},
        decoder=SEGDIAG_MODULE,
        decoder_args={"max_examples": max_examples, "num_clusters": num_clusters},
        config={
            "behavior_version": 21,
            "extern_data": ds.extern_data_audio_text(vocab_size),
            "batch_size": batch_size_sec * 16000,
            "max_seqs": 200,
        },
    )
    fwd = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=forward_config,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["seg_diag.pkl"],
        device="gpu",
        time_rqmt=2,
        mem_rqmt=16,
        cpu_rqmt=4,
    )
    fwd.rqmt["gpu_mem"] = 24
    fwd.add_alias(f"{prefix}/forward")
    return fwd.out_files["seg_diag.pkl"]


class SegDiagAnalysisJob(Job):
    """Turn a ``seg_diag.pkl`` into ``metrics.json`` + human-readable ``summary.txt`` (+ plots/)."""

    def __init__(
        self,
        seg_diag_pkl: tk.Path,
        *,
        num_clusters: int,
        target_rate_hz: float,
        frame_rate_hz: float = 25.0,
        arm_name: str = "",
        codebook: Optional[tk.Path] = None,
        split: Optional[Sequence[str]] = None,
    ):
        self.seg_diag_pkl = seg_diag_pkl
        self.num_clusters = int(num_clusters)
        self.target_rate_hz = float(target_rate_hz)
        self.frame_rate_hz = float(frame_rate_hz)
        self.arm_name = arm_name
        # When both are given (and the pkl carries per-frame ``h``), the analysis also computes the
        # GOLD-anchored boundary metrics + the codebook quant-error "does placement matter" gate.
        self.codebook = codebook
        self.split = tuple(split) if split is not None else None

        self.out_metrics = self.output_path("metrics.json")
        self.out_summary = self.output_path("summary.txt")
        self.out_plots = self.output_path("plots", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import pickle

        import numpy as np

        with open(self.seg_diag_pkl.get_path(), "rb") as f:
            d = pickle.load(f)

        scalars = d["scalars"]  # list of (seq_tag, T, K, sum_alpha, N)
        T = np.array([s[1] for s in scalars], dtype=np.float64)
        K = np.array([s[2] for s in scalars], dtype=np.float64)
        N = np.array([s[4] for s in scalars], dtype=np.float64)
        n_seqs = len(scalars)

        target_fps = self.frame_rate_hz / self.target_rate_hz  # dialled frames/token
        target_ms = 1000.0 / self.target_rate_hz

        def pct(x, p):
            return float(np.percentile(x, p)) if x.size else float("nan")

        # ---- realised segmentation rate ----
        valid = K > 0
        fps = T[valid] / K[valid]  # per-utterance frames/token
        fps_mean = float(fps.mean()) if fps.size else float("nan")
        rate = {
            "target_frames_per_token": target_fps,
            "target_ms": target_ms,
            "realized_fps_mean": fps_mean,
            "realized_fps_std": float(fps.std()) if fps.size else float("nan"),
            "realized_fps_p10_p50_p90": [pct(fps, 10), pct(fps, 50), pct(fps, 90)],
            "realized_ms_mean": fps_mean * FRAME_MS,
            "over_fire_pct": (target_fps / fps_mean - 1.0) * 100.0 if fps_mean else float("nan"),
        }

        # ---- CTC feasibility: need K >= N (more CIF tokens than SPM labels) ----
        has_n = N > 0
        kn = {}
        if has_n.any():
            ratio = K[has_n] / N[has_n]
            kn = {
                "k_over_n_mean": float(ratio.mean()),
                "k_over_n_p10_p50_p90": [pct(ratio, 10), pct(ratio, 50), pct(ratio, 90)],
                "frac_infeasible_K_lt_N": float((K[has_n] < N[has_n]).mean()),
                "n_with_labels": int(has_n.sum()),
            }

        # ---- codebook usage / collapse ----
        counts = np.asarray(d["code_counts"], dtype=np.float64)
        p = counts / max(counts.sum(), 1.0)
        nz = p[p > 0]
        code = {
            "usage_entropy_norm": float(-(nz * np.log(nz)).sum() / np.log(self.num_clusters)) if nz.size else 0.0,
            "coverage_frac": float((counts > 0).mean()),
            "top1_share": float(p.max()) if p.size else float("nan"),
            "recon_acc_unmasked": float(d["pred_match"] / max(d["tok_total"], 1)),
        }

        # ---- boundary quality: layer-9 flux at fired frames vs overall (>1 => fires land on change points) ----
        ratios = []
        for ex in d["examples"]:
            fired = ex["fired"].astype(bool)
            flux = ex["hflux"]
            if flux.size > 1 and fired.any():
                overall = float(flux[1:].mean())  # frame 0 flux := 0 by construction
                if overall > 0:
                    ratios.append(float(flux[fired].mean()) / overall)
        boundary = {
            "flux_at_fire_over_mean": float(np.mean(ratios)) if ratios else float("nan"),
            "n_examples": len(d["examples"]),
        }

        metrics = {
            "arm": self.arm_name,
            "n_seqs": n_seqs,
            "rate": rate,
            "k_over_n": kn,
            "codebook": code,
            "boundary": boundary,
        }

        # ---- GOLD-anchored boundary metrics + quant-error placement gate (optional) ----
        gold_lines = []
        if self.codebook is not None and self.split is not None and d["examples"] and "h" in d["examples"][0]:
            try:
                gold, gold_lines = self._gold_block(d)
                metrics["gold"] = gold
            except Exception as e:  # never let the optional gate kill the core metrics
                metrics["gold_error"] = repr(e)
                gold_lines = [f"## gold metrics FAILED: {e!r}"]

        with open(self.out_metrics.get_path(), "wt") as f:
            json.dump(metrics, f, indent=2)

        # ---- human-readable summary ----
        lines = [
            f"# CIF segmentation diagnostics: {self.arm_name}",
            f"utterances analysed: {n_seqs}  (heavy examples: {boundary['n_examples']})",
            "",
            "## rate (the segmentation dial)",
            f"  dialled       : {target_fps:.3f} frames/token  ({target_ms:.0f} ms)",
            f"  realised      : {rate['realized_fps_mean']:.3f} frames/token  "
            f"(~{rate['realized_ms_mean']:.0f} ms)   [p10/p50/p90 "
            f"{rate['realized_fps_p10_p50_p90'][0]:.2f}/{rate['realized_fps_p10_p50_p90'][1]:.2f}/"
            f"{rate['realized_fps_p10_p50_p90'][2]:.2f}]",
            f"  over-fire     : {rate['over_fire_pct']:+.1f}%  (>0 = more, finer tokens than dialled)",
        ]
        if kn:
            lines += [
                "",
                "## CTC feasibility (need K >= N labels)",
                f"  K/N mean      : {kn['k_over_n_mean']:.2f}   "
                f"[p10/p50/p90 {kn['k_over_n_p10_p50_p90'][0]:.2f}/"
                f"{kn['k_over_n_p10_p50_p90'][1]:.2f}/{kn['k_over_n_p10_p50_p90'][2]:.2f}]",
                f"  infeasible    : {kn['frac_infeasible_K_lt_N'] * 100:.2f}% of utts have K < N (bad)",
            ]
        lines += [
            "",
            "## codebook (target codes over CIF tokens)",
            f"  usage entropy : {code['usage_entropy_norm']:.3f} (1.0 = uniform; -> 0 = collapse)",
            f"  coverage      : {code['coverage_frac'] * 100:.1f}% of {self.num_clusters} clusters used",
            f"  top-1 share   : {code['top1_share'] * 100:.1f}%",
            f"  recon acc     : {code['recon_acc_unmasked'] * 100:.1f}% (unmasked preds == targets)",
            "",
            "## boundary quality",
            f"  flux@fire/mean: {boundary['flux_at_fire_over_mean']:.2f} "
            f"(>1 = fires concentrate on layer-9 change points)",
        ]
        if gold_lines:
            lines += [""] + gold_lines
        with open(self.out_summary.get_path(), "wt") as f:
            f.write("\n".join(lines) + "\n")

        self._plots(d, fps, target_fps)

    # gilkeyio alignment split-name map (our HF split -> alignment parquet basename)
    _SPLIT_MAP = {
        "validation.clean": "dev_clean", "validation.other": "dev_other",
        "test.clean": "test_clean", "test.other": "test_other",
        "train.clean.100": "train_clean_100", "train.clean.360": "train_clean_360",
        "train.other.500": "train_other_500",
    }
    _SIL = {"sil", "sp", "spn", "", "<unk>", "<eps>", "noise", "nsn", "lau"}

    def _load_gold(self):
        import pandas as pd

        hf = os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home")
        base = os.path.join(hf, "hub", "datasets--gilkeyio--librispeech-alignments", "snapshots")
        gold = {}
        for src in self.split:
            name = self._SPLIT_MAP[src]
            files = sorted(glob.glob(os.path.join(base, "*", "data", f"{name}-*.parquet")))
            assert files, f"no gold alignment parquet for split {src!r} ({name}) under {base}"
            for fp in files:
                df = pd.read_parquet(fp, columns=["id", "phonemes"])
                for r in df.itertuples():
                    gold[r.id] = r.phonemes
        return gold

    def _gold_block(self, d):
        """Join per-frame ``h`` to cached gold phone alignments; compute the boundary metrics + the
        codebook quant-error placement gate (gold vs uniform-at-gold-rate vs CIF, all plain seg-mean)."""
        import numpy as np

        rng = np.random.default_rng(0)
        cb = np.load(self.codebook.get_path()).astype(np.float64)  # [C, D]
        cbn = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-8)
        gold = self._load_gold()

        def seg_mean(h, seg):  # h[T,D], seg[T] contiguous 0..K-1 -> z[K,D] plain mean
            K = int(seg.max()) + 1
            z = np.zeros((K, h.shape[1]), np.float64)
            cnt = np.zeros(K, np.float64)
            np.add.at(z, seg, h)
            np.add.at(cnt, seg, 1.0)
            return z / np.maximum(cnt[:, None], 1.0)

        def quant(z):  # mean (1 - max cosine to nearest centroid) + normalized assigned-code entropy
            zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
            sim = zn @ cbn.T
            code = sim.argmax(axis=1)
            cnts = np.bincount(code, minlength=self.num_clusters).astype(np.float64)
            p = cnts / max(cnts.sum(), 1.0)
            p = p[p > 0]
            ent = float(-(p * np.log(p)).sum() / np.log(self.num_clusters)) if p.size else 0.0
            return float((1.0 - sim.max(axis=1)).mean()), ent

        def f1(pred, goldb, tol):  # greedy 1-1 match within tol frames
            pred = np.array(sorted({int(x) for x in pred}))
            goldb = np.array(sorted({int(x) for x in goldb}))
            if pred.size == 0 or goldb.size == 0:
                return 0.0
            used = np.zeros(goldb.size, bool)
            tp = 0
            for q in pred:
                dd = np.abs(goldb - q)
                j = int(dd.argmin())
                if dd[j] <= tol and not used[j]:
                    used[j] = True
                    tp += 1
            prec, rec = tp / pred.size, tp / goldb.size
            return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        def ratio(flux, idx, denom):
            idx = np.asarray([i for i in idx if 1 <= i < flux.size], dtype=int)
            return float(flux[idx].mean() / denom) if idx.size and denom > 0 else np.nan

        acc = {k: [] for k in (
            "flux_gold", "flux_cif", "flux_rand", "f1_cif0", "f1_rand0", "f1_cif1", "f1_rand1",
            "corr", "qe_gold", "qe_unifG", "qe_cif", "qe_unifC", "ent_gold", "ent_unifG", "ent_cif",
            "gold_fps", "cif_fps")}
        n = 0
        for ex in d["examples"]:
            tag = ex["seq_tag"]
            if tag not in gold:
                continue
            h = np.asarray(ex["h"], dtype=np.float64)
            flux = np.asarray(ex["hflux"], dtype=np.float64)
            T = flux.size
            if h.shape[0] != T or T < 4:  # frame-count guard (h must line up with the flux/fire stream)
                continue
            denom = flux[1:].mean()
            ph = gold[tag]
            gb = sorted({int(round(float(p["start"]) * self.frame_rate_hz)) for p in ph
                         if str(p["phoneme"]).lower() not in self._SIL})
            gb = np.array([x for x in gb if 1 <= x <= T - 1])
            fired = np.asarray(ex["fired"]) > 0.5
            fb = np.where(fired)[0]
            fb = fb[fb >= 1]
            if gb.size < 3 or fb.size < 3 or denom <= 0:
                continue
            n += 1
            # seg ids (contiguous 0..K-1). gold: silence merged via searchsorted on phone onsets.
            gbounds = np.array([0] + list(gb))
            seg_gold = np.clip(np.searchsorted(gbounds, np.arange(T), side="right") - 1, 0, None)
            Kg = int(seg_gold.max()) + 1
            seg_cif = np.cumsum(fired).astype(int)
            seg_cif -= seg_cif.min()
            Kc = int(seg_cif.max()) + 1
            chunkG = max(1, int(round(T / Kg)))
            seg_unifG = np.minimum(np.arange(T) // chunkG, Kg - 1)
            chunkC = max(1, int(round(T / Kc)))
            seg_unifC = np.minimum(np.arange(T) // chunkC, Kc - 1)
            # quant-error: plain seg-mean for ALL schemes so only the BOUNDARIES differ
            qg, eg = quant(seg_mean(h, seg_gold)); acc["qe_gold"].append(qg); acc["ent_gold"].append(eg)
            qug, eug = quant(seg_mean(h, seg_unifG)); acc["qe_unifG"].append(qug); acc["ent_unifG"].append(eug)
            qc, ec = quant(seg_mean(h, seg_cif)); acc["qe_cif"].append(qc); acc["ent_cif"].append(ec)
            quc, _ = quant(seg_mean(h, seg_unifC)); acc["qe_unifC"].append(quc)
            # flux ratios (ceiling=gold, floor=random)
            acc["flux_gold"].append(ratio(flux, gb, denom))
            acc["flux_cif"].append(ratio(flux, fb, denom))
            acc["flux_rand"].append(
                float(np.nanmean([ratio(flux, rng.integers(1, T, gb.size), denom) for _ in range(100)])))
            # boundary F1 vs rate-matched random baseline
            acc["f1_cif0"].append(f1(fb, gb, 0)); acc["f1_cif1"].append(f1(fb, gb, 1))
            r0, r1 = [], []
            for _ in range(30):
                rp = rng.choice(np.arange(1, T), size=min(fb.size, T - 2), replace=False)
                r0.append(f1(rp, gb, 0)); r1.append(f1(rp, gb, 1))
            acc["f1_rand0"].append(float(np.mean(r0))); acc["f1_rand1"].append(float(np.mean(r1)))
            ind = np.zeros(T); ind[gb] = 1.0
            acc["corr"].append(float(np.corrcoef(flux[1:], ind[1:])[0, 1]))
            acc["gold_fps"].append(T / Kg); acc["cif_fps"].append(T / Kc)

        m = lambda k: float(np.nanmean(acc[k])) if acc[k] else float("nan")
        gap = m("qe_unifG") - m("qe_gold")  # >0 => gold cleaner => placement matters
        gold_metrics = {
            "n_matched": n,
            "gold_phone_fps": m("gold_fps"), "cif_fps": m("cif_fps"),
            "flux_ratio": {"gold_ceiling": m("flux_gold"), "cif": m("flux_cif"), "random_floor": m("flux_rand")},
            "boundary_f1_exact": {"cif": m("f1_cif0"), "random": m("f1_rand0"), "lift": m("f1_cif0") - m("f1_rand0")},
            "boundary_f1_tol1": {"cif": m("f1_cif1"), "random": m("f1_rand1"), "lift": m("f1_cif1") - m("f1_rand1")},
            "corr_flux_gold": m("corr"),
            "quant_error_cosdist": {
                "gold": m("qe_gold"), "uniform_at_gold_rate": m("qe_unifG"),
                "cif": m("qe_cif"), "uniform_at_cif_rate": m("qe_unifC"),
                "gold_vs_uniform_gap": gap,
            },
            "code_entropy": {"gold": m("ent_gold"), "uniform_at_gold_rate": m("ent_unifG"), "cif": m("ent_cif")},
        }
        verdict = "gold cleaner -> placement MATTERS" if gap > 0.002 else "uniform>=gold -> placement does NOT help target"
        lines = [
            "## GOLD-anchored boundary metrics (vs cached MFA phones)",
            f"  matched utts  : {n}   gold-phone fps {m('gold_fps'):.2f}  cif fps {m('cif_fps'):.2f}",
            f"  flux ratio    : gold(ceiling) {m('flux_gold'):.3f}  cif {m('flux_cif'):.3f}  random(floor) {m('flux_rand'):.3f}",
            f"  bdry-F1 exact : cif {m('f1_cif0'):.3f}  random {m('f1_rand0'):.3f}  lift {m('f1_cif0')-m('f1_rand0'):+.3f}",
            f"  bdry-F1 +-1   : cif {m('f1_cif1'):.3f}  random {m('f1_rand1'):.3f}  lift {m('f1_cif1')-m('f1_rand1'):+.3f}",
            f"  corr(flux,gold): {m('corr'):+.3f}",
            "",
            "## QUANT-ERROR placement gate (cosine dist to nearest centroid; lower=better; plain seg-mean)",
            f"  gold seg       : {m('qe_gold'):.4f}   (code-entropy {m('ent_gold'):.3f})",
            f"  uniform@goldR  : {m('qe_unifG'):.4f}   (code-entropy {m('ent_unifG'):.3f})",
            f"  cif seg        : {m('qe_cif'):.4f}   (code-entropy {m('ent_cif'):.3f})",
            f"  GATE gold-vs-uniform gap = {gap:+.4f}  ({verdict})",
        ]
        return gold_metrics, lines

    def _plots(self, d, fps, target_fps):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return  # plots are optional; metrics.json / summary.txt are the contract

        if fps.size:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.hist(fps, bins=40, color="#4477aa")
            ax.axvline(target_fps, color="crimson", ls="--", label=f"target {target_fps:.2f}")
            ax.set_xlabel("frames / token")
            ax.set_ylabel("utterances")
            ax.set_title(f"realised CIF rate -- {self.arm_name}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{self.out_plots.get_path()}/fps_hist.png", dpi=110)
            plt.close(fig)

        for i, ex in enumerate(d["examples"][:6]):
            alpha, fired = ex["alpha"], ex["fired"].astype(bool)
            fig, ax = plt.subplots(figsize=(9, 2.4))
            ax.plot(alpha, color="#228833", lw=0.8, label="alpha")
            for x in np.where(fired)[0]:
                ax.axvline(x, color="crimson", lw=0.4, alpha=0.6)
            ax.set_title(ex["seq_tag"])
            ax.set_xlabel("25 Hz frame")
            ax.set_ylabel("alpha")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(f"{self.out_plots.get_path()}/alpha_{i:02d}.png", dpi=110)
            plt.close(fig)


def register_seg_diag(
    prefix: str,
    *,
    checkpoint: tk.Path,
    model_config,
    codebook,
    num_clusters: int,
    target_rate_hz: float,
    vocab_size: int,
    spm_model,
    split: Sequence[str] = ds.DEV_CLEAN,
    max_examples: int = 256,
    returnn_exe: tk.Path = RETURNN_EXE,
    returnn_root: tk.Path = RETURNN_ROOT,
) -> SegDiagAnalysisJob:
    """Wire forward -> analysis for ONE checkpoint and register ``metrics.json`` / ``summary.txt``.

    ``model_config`` is a ``TwoLevelConfig`` dataclass (``asdict`` is taken here). Returns the analysis job.
    """
    from dataclasses import asdict

    pkl = seg_diag_forward(
        prefix,
        checkpoint=checkpoint,
        model_config_dict=asdict(model_config),
        codebook=codebook,
        num_clusters=num_clusters,
        vocab_size=vocab_size,
        spm_model=spm_model,
        split=split,
        max_examples=max_examples,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    job = SegDiagAnalysisJob(
        pkl,
        num_clusters=num_clusters,
        target_rate_hz=target_rate_hz,
        frame_rate_hz=model_config.frame_rate_hz,
        arm_name=prefix,
        codebook=codebook,
        split=split,
    )
    job.add_alias(f"{prefix}/analysis")
    tk.register_output(f"{prefix}/metrics.json", job.out_metrics)
    tk.register_output(f"{prefix}/summary.txt", job.out_summary)
    return job
