"""Thin Sisyphus pipeline helpers for SSL training (and later forward/recog)."""

import os
from typing import Any, Dict, Optional

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob, GetBestPtCheckpointJob
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchWordsDummyTimesToCTMJob
from i6_core.text.convert import TextDictToStmJob
from i6_core.recognition.scoring import ScliteJob


def training(
    prefix: str,
    returnn_config: ReturnnConfig,
    *,
    num_epochs: int,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    num_processes: Optional[int] = None,
    gpu_mem: int = 96,
    time_rqmt: float = 24,
    mem_rqmt: float = 100,
    cpu_rqmt: int = 16,
) -> ReturnnTrainingJob:
    """
    Create a ReturnnTrainingJob.

    :param num_processes: if set, multi-GPU via torchrun (per-step DDP grad sync configured in the
        RETURNN config via ``torch_distributed={"reduce_type": "grad"}``). i6_core multiplies the
        cpu/gpu/mem rqmt by num_processes (single node).
    """
    default_rqmt = {
        "mem_rqmt": mem_rqmt,
        "time_rqmt": time_rqmt,
        "cpu_rqmt": cpu_rqmt,
        "log_verbosity": 5,
        "returnn_python_exe": returnn_exe,
        "returnn_root": returnn_root,
    }
    distributed_rqmt = {}
    if num_processes is not None:
        distributed_rqmt = {"horovod_num_processes": num_processes, "distributed_launch_cmd": "torchrun"}
    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config, num_epochs=num_epochs, **default_rqmt, **distributed_rqmt
    )
    train_job.rqmt["gpu_mem"] = gpu_mem
    train_job.add_alias(prefix + "/training")
    tk.register_output(prefix + "/learning_rates", train_job.out_learning_rates)
    return train_job


def ctc_greedy_recog_score(
    prefix: str,
    *,
    checkpoint,
    model_config_dict: Dict[str, Any],
    vocab_size,
    spm_model,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    decoder_module: str = "ctc.conformer_ctc_v1",
    batch_size_sec: int = 600,
    epoch: Optional[int] = None,
):
    """Greedy-CTC recognition + offline sclite WER on dev/test clean+other for one checkpoint.

    Builds a ReturnnForwardJobV2 (greedy decoder -> search_out.py words) per split, the reference
    text-dict straight from the HF parquet (HFTextDictJob), then CTM/STM + ScliteJob. Registers WER.
    ``epoch`` (if given) namespaces the outputs under ``recog/ep<epoch>`` so multiple checkpoints of
    one experiment do not collide. Returns ``{split: ScliteJob.out_wer}``.
    """
    from .config import get_forward_config
    from .data import datasets as ds
    from .data.score import HFTextDictJob
    from .default_tools import SCTK_BINARY_PATH

    sub = f"recog/ep{epoch}" if epoch is not None else "recog"
    splits = {
        "dev-clean": ds.DEV_CLEAN,
        "dev-other": ds.DEV_OTHER,
        "test-clean": ds.TEST_CLEAN,
        "test-other": ds.TEST_OTHER,
    }
    wers = {}
    for name, src in splits.items():
        forward_config = get_forward_config(
            forward_dataset=ds.audio_hf_dataset(src, seq_ordering="default"),
            network_module=decoder_module,
            net_args={"model_config_dict": model_config_dict, "vocab_size": vocab_size},
            decoder=decoder_module,
            decoder_args={"spm_model_file": spm_model, "blank_idx": vocab_size},
            config={
                "behavior_version": 21,
                "extern_data": ds.extern_data_audio(),
                "batch_size": batch_size_sec * 16000,
                "max_seqs": 200,
            },
        )
        fwd = ReturnnForwardJobV2(
            model_checkpoint=checkpoint,
            returnn_config=forward_config,
            returnn_python_exe=returnn_exe,
            returnn_root=returnn_root,
            output_files=["search_out.py"],
            device="gpu",
            time_rqmt=4,
            mem_rqmt=16,
            cpu_rqmt=4,
        )
        fwd.rqmt["gpu_mem"] = 24
        fwd.add_alias(f"{prefix}/{sub}/{name}/forward")
        search_out = fwd.out_files["search_out.py"]

        ref = HFTextDictJob(src).out_text_dict
        ctm = SearchWordsDummyTimesToCTMJob(
            recog_words_file=search_out, seq_order_file=ref, seg_length_time=1000.0
        ).out_ctm_file
        stm = TextDictToStmJob(text_dict=ref, seg_length_time=1000.0).out_stm_path
        sclite = ScliteJob(ref=stm, hyp=ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
        tk.register_output(f"{prefix}/{sub}/{name}/wer", sclite.out_wer)
        wers[name] = sclite.out_wer
    return wers


# --------------------------- multi-checkpoint recog (speech_llm style) + reports ---------------------------

RECOG_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
_RECOG_SPLITS = ["dev-clean", "dev-other", "test-clean", "test-other"]


def fraction_epochs(num_epochs: int, fracs=RECOG_FRACTIONS):
    """Checkpoint epochs at the given fractions of num_epochs (round, clamp to [1,num_epochs], dedup, sort).

    e.g. num_epochs=200 -> [20,60,100,140,180,200]; 60 -> [6,18,30,42,54,60]. These must be a subset of the
    training config's cleanup_old_models["keep"] list (see config.get_training_config(keep_epochs=...)) or
    RETURNN deletes them before recognition.
    """
    return sorted({min(num_epochs, max(1, round(f * num_epochs))) for f in fracs})


def ctc_recog_all_epochs(
    prefix: str,
    *,
    train_job: ReturnnTrainingJob,
    num_epochs: int,
    model_config_dict: Dict[str, Any],
    vocab_size,
    spm_model,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    fracs=RECOG_FRACTIONS,
    decoder_module: str = "ctc.conformer_ctc_v1",
    batch_size_sec: int = 600,
):
    """Greedy-CTC recog + sclite WER at the fraction-epoch checkpoints (10/30/.../100%), then register a
    live per-experiment WER summary under ``<prefix>/summary.md``. Returns ``{epoch: {split: out_wer}}``."""
    epochs = fraction_epochs(num_epochs, fracs)
    wers: Dict[int, Dict[str, Any]] = {}
    for ep in epochs:
        wers[ep] = ctc_greedy_recog_score(
            prefix,
            checkpoint=train_job.out_checkpoints[ep],
            model_config_dict=model_config_dict,
            vocab_size=vocab_size,
            spm_model=spm_model,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            decoder_module=decoder_module,
            batch_size_sec=batch_size_sec,
            epoch=ep,
        )
    register_wer_summary(prefix, wers, num_epochs)
    return wers


def ctc_recog_best_checkpoint(
    prefix: str,
    *,
    train_job: ReturnnTrainingJob,
    model_config_dict: Dict[str, Any],
    vocab_size,
    spm_model,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    key: str = "dev_loss_ctc",
    decoder_module: str = "ctc.conformer_ctc_v1",
    batch_size_sec: int = 600,
):
    """Greedy-CTC recog + sclite WER on the DEV-LOSS-BEST checkpoint (the one keep_best_n retains),
    selected at job end via GetBestPtCheckpointJob on ``key`` (default dev_loss_ctc, index 0 = lowest).
    Outputs land under ``<prefix>/recog/epbest/<split>/wer`` — this is the number to read when the
    fixed fraction-epoch checkpoints miss the dev optimum. Returns ``{split: out_wer}``."""
    best = GetBestPtCheckpointJob(
        model_dir=train_job.out_model_dir,
        learning_rates=train_job.out_learning_rates,
        key=key,
        index=0,
    )
    best.add_alias(f"{prefix}/recog/epbest/get_best")
    return ctc_greedy_recog_score(
        prefix,
        checkpoint=best.out_checkpoint,
        model_config_dict=model_config_dict,
        vocab_size=vocab_size,
        spm_model=spm_model,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        decoder_module=decoder_module,
        batch_size_sec=batch_size_sec,
        epoch="best",
    )


def _fmt_var(v):
    """Resolve a sisyphus Variable/Path to a float at report-render time, defensively (-> None if pending)."""
    try:
        val = v.get()
        return float(val) if val is not None else None
    except Exception:
        return None


def register_wer_summary(prefix: str, wers: Dict[int, Dict[str, Any]], num_epochs: int):
    """Live markdown WER summary (speech_llm style): rows = checkpoint epoch (+ % of training), columns =
    dev/test clean+other, best per column **bold**. Re-rendered by the manager (required=[] -> never blocks,
    fills in as each sclite finishes). Lands at ``<OUTPUT_DIR>/output/<prefix>/summary.md``."""

    def render() -> str:
        best = {}
        for s in _RECOG_SPLITS:
            vals = [x for x in (_fmt_var(wers[ep].get(s)) for ep in wers) if x is not None]
            best[s] = min(vals) if vals else None
        lines = [
            f"# CTC greedy-WER summary — {prefix}",
            "",
            "| epoch | frac | " + " | ".join(f"`{s}`" for s in _RECOG_SPLITS) + " |",
            "|---:|---:|" + "|".join(["---:"] * len(_RECOG_SPLITS)) + "|",
        ]
        for ep in sorted(wers):
            cells = []
            for s in _RECOG_SPLITS:
                v = _fmt_var(wers[ep].get(s))
                if v is None:
                    cells.append("…")
                else:
                    cells.append(f"**{v:.2f}**" if best[s] is not None and v == best[s] else f"{v:.2f}")
            cells_s = " | ".join(cells)
            lines.append(f"| {ep} | {round(ep / num_epochs * 100)}% | {cells_s} |")
        return "\n".join(lines) + "\n"

    tk.register_report(f"{prefix}/summary.md", render, required=[])


def register_train_summary(prefix: str, train_job: ReturnnTrainingJob, *, title: str = "Training summary"):
    """Live markdown training-score summary for an experiment with no WER recog (e.g. BEST-RQ pretraining):
    parses ``out_learning_rates`` and tabulates per-epoch lr + all train_/dev_ scores. Lands at
    ``<OUTPUT_DIR>/output/<prefix>/summary.md``."""
    lr_path = train_job.out_learning_rates

    def render() -> str:
        ns = {
            "EpochData": lambda learningRate=None, error=None: {"lr": learningRate, "error": error or {}},
            "nan": float("nan"),
            "inf": float("inf"),
        }
        # out_learning_rates only exists once the job finishes; while RUNNING the live file is in the
        # job's work/ subdir. Try the output path, then fall back to the work/ copy.
        p = lr_path.get_path()
        if not os.path.exists(p):
            alt = p.replace("/output/", "/work/")
            if os.path.exists(alt):
                p = alt
        try:
            with open(p) as f:
                data = eval(f.read(), ns)  # noqa: S307  RETURNN learning_rates file (trusted, our own job)
        except Exception as e:  # pending / not yet written
            return f"# {title} — {prefix}\n\n_(no scores yet: {e})_\n"
        if not data:
            return f"# {title} — {prefix}\n\n_(no epochs yet)_\n"
        # union of score keys across epochs (skip RETURNN ':meta:' diagnostics), sorted
        keys = []
        for ep in sorted(data):
            for k in data[ep].get("error") or {}:
                if not k.startswith(":meta:") and k not in keys:
                    keys.append(k)
        keys = sorted(keys)
        lines = [
            f"# {title} — {prefix}",
            "",
            "| epoch | lr | " + " | ".join(f"`{k}`" for k in keys) + " |",
            "|---:|---:|" + "|".join(["---:"] * len(keys)) + "|",
        ]
        for ep in sorted(data):
            err = data[ep].get("error") or {}
            lr = data[ep].get("lr")
            lr_s = "—" if lr is None else f"{lr:.2e}"
            cells = []
            for k in keys:
                v = err.get(k)
                if isinstance(v, (int, float)):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append("…" if v is None else str(v))
            lines.append(f"| {ep} | {lr_s} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    tk.register_report(f"{prefix}/summary.md", render, required=[])
