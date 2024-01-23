from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from sisyphus import tk, gs

from .config_01_conformer_ctc_pt import py as py_01
# from .config_01a_conformer_ctc_pt_tuning import py as py_01a


def main() -> SummaryReport:
    def worker_wrapper(job, task_name, call):
        wrapped_jobs = {
            "MakeJob",
            "ReturnnTrainingJob",
            "ReturnnRasrTrainingJob",
            "OptunaReturnnTrainingJob",
            "CompileTFGraphJob",
            "OptunaCompileTFGraphJob",
            "ReturnnRasrComputePriorJob",
            "ReturnnComputePriorJob",
            "ReturnnComputePriorJobV2",
            "OptunaReturnnComputePriorJob",
            "CompileNativeOpJob",
            "AdvancedTreeSearchJob",
            "AdvancedTreeSearchLmImageAndGlobalCacheJob",
            "GenericSeq2SeqSearchJob",
            "GenericSeq2SeqLmImageAndGlobalCacheJob",
            "LatticeToCtmJob",
            "OptimizeAMandLMScaleJob",
            "AlignmentJob",
            "Seq2SeqAlignmentJob",
            "EstimateMixturesJob",
            "EstimateCMLLRJob",
            "ExportPyTorchModelToOnnxJob",
            "OptunaExportPyTorchModelToOnnxJob",
            "ReturnnForwardJob",
            "ReturnnForwardComputePriorJob",
            "OptunaReturnnForwardComputePriorJob",
        }
        if type(job).__name__ not in wrapped_jobs:
            return call
        binds = ["/work/asr4", "/work/common", "/work/tools/"]
        ts = {t.name(): t for t in job.tasks()}
        t = ts[task_name]

        app_call = [
            "apptainer",
            "exec",
        ]
        if t._rqmt.get("gpu", 0) > 0:
            app_call += ["--nv"]

        for path in binds:
            app_call += ["-B", path]

        app_call += [
            "/work/asr4/berger/apptainer/images/i6_u22_pytorch1.13_onnx.sif",
            "python3",
        ]

        app_call += call[1:]

        return app_call

    gs.worker_wrapper = worker_wrapper

    summary_report = SummaryReport()

    summary_report.merge_report(py_01(), update_structure=True)
    # summary_report.merge_report(py_01a(), update_structure=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
