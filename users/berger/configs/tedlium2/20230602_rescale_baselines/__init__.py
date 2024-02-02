from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from sisyphus import tk, gs

from .config_01_conformer_ctc_pt import py as py_01
from .config_04_conformer_transducer_pt import py as py_04

# from .config_01a_conformer_ctc_pt_tuning import py as py_01a


def main() -> SummaryReport:
    def worker_wrapper(job, task_name, call):
        rasr_jobs = {
            "MakeJob",
            "AdvancedTreeSearchJob",
            "AdvancedTreeSearchLmImageAndGlobalCacheJob",
            "FeatureExtractionJob",
            "GenericSeq2SeqSearchJob",
            "GenericSeq2SeqLmImageAndGlobalCacheJob",
            "LatticeToCtmJob",
            "OptimizeAMandLMScaleJob",
            "AlignmentJob",
            "Seq2SeqAlignmentJob",
            "EstimateMixturesJob",
            "EstimateCMLLRJob",
        }
        torch_jobs = {
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
            "ExportPyTorchModelToOnnxJob",
            "TorchOnnxExportJob",
            "OptunaExportPyTorchModelToOnnxJob",
            "ReturnnForwardJob",
            "ReturnnForwardJobV2",
            "ReturnnForwardComputePriorJob",
            "OptunaReturnnForwardComputePriorJob",
        }
        jobclass = type(job).__name__
        if jobclass in rasr_jobs:
            image = "/work/asr4/berger/apptainer/images/i6_tensorflow-2.8_onnx-1.15.sif"
        elif jobclass in torch_jobs:
            image = "/work/asr4/berger/apptainer/images/i6_torch-2.2_onnx-1.16.sif"
        else:
            print(jobclass)
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

        app_call += [image, "python3"]

        app_call += call[1:]

        return app_call

    gs.worker_wrapper = worker_wrapper

    summary_report = SummaryReport()

    summary_report.merge_report(py_01(), update_structure=True)
    summary_report.merge_report(py_04())
    # summary_report.merge_report(py_01a(), update_structure=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
