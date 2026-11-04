"""Earnings22-specific pruning diagnostic; not held-out model selection."""


def run():
    from . import source_version
    from .rnnt_qwen import run as run_rnnt
    from ...data.open_asr import DATASETS, PrepareOpenAsrDatasetJob

    spec = next(s for s in DATASETS if s.name == "earnings22-cleaned")
    prepared = PrepareOpenAsrDatasetJob(spec, code_version=source_version())
    return run_rnnt(data={spec.name: prepared}, tuning_only=False, earnings_pruning=True)
