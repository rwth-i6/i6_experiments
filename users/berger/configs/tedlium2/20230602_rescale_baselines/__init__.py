import copy
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import SummaryKey
from sisyphus import tk, gs

from .config_01_conformer_ctc import py as py_01

from .config_04a_conformer_transducer_bpe import py as py_04a
from .config_04a_conformer_transducer_bpe_rasr import py as py_04a_rasr
from .config_04b_conformer_transducer_phon import py as py_04b


def main() -> SummaryReport:
    def worker_wrapper(job, task_name, call):
        rasr_jobs = {
            "MakeJob",
            "CompileNativeOpJob",
            "AdvancedTreeSearchJob",
            "AdvancedTreeSearchLmImageAndGlobalCacheJob",
            "FeatureExtractionJob",
            "GenericSeq2SeqSearchJob",
            "GenericSeq2SeqSearchJobV2",
            "GenericSeq2SeqLmImageAndGlobalCacheJob",
            "CreateLmImageJob",
            "BuildGenericSeq2SeqGlobalCacheJob",
            "LatticeToCtmJob",
            "OptimizeAMandLMScaleJob",
            "AlignmentJob",
            "Seq2SeqAlignmentJob",
            "EstimateMixturesJob",
            "EstimateCMLLRJob",
        }
        torch_jobs = {
            "ReturnnTrainingJob",
            "ReturnnRasrTrainingJob",
            "OptunaReturnnTrainingJob",
            "ReturnnDumpHDFJob",
            "CompileTFGraphJob",
            "OptunaCompileTFGraphJob",
            "ReturnnRasrComputePriorJob",
            "ReturnnComputePriorJob",
            "ReturnnComputePriorJobV2",
            "OptunaReturnnComputePriorJob",
            "ReturnnForwardJob",
            "ReturnnForwardJobV2",
            "ReturnnForwardComputePriorJob",
            "OptunaReturnnForwardComputePriorJob",
            "CompileKenLMJob",
        }
        onnx_jobs = {
            "ExportPyTorchModelToOnnxJob",
            "TorchOnnxExportJob",
            "OptunaExportPyTorchModelToOnnxJob",
        }
        jobclass = type(job).__name__
        if jobclass in rasr_jobs:
            image = "/work/asr4/berger/apptainer/images/i6_tensorflow-2.8_onnx-1.15.sif"
        elif jobclass in torch_jobs:
            image = "/work/asr4/berger/apptainer/images/i6_torch-2.2_onnx-1.16.sif"
        elif jobclass in onnx_jobs:
            # use this one because mhsa is not onnx exportable in torch 2 yet
            image = "/work/asr4/berger/apptainer/images/i6_u22_pytorch1.13_onnx.sif"
        else:
            return call

        binds = ["/work/asr4", "/work/common", "/work/tools/", "/work/asr4/rossenbach"]
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

    for subreport in [
        copy.deepcopy(py_01()),
        copy.deepcopy(py_04a()),
        copy.deepcopy(py_04a_rasr()),
        copy.deepcopy(py_04b()),
    ]:
        subreport.collapse([SummaryKey.CORPUS.value], best_selector_key=SummaryKey.ERR.value)
        summary_report.merge_report(subreport, update_structure=True)

    summary_report.set_col_sort_key([SummaryKey.ERR.value, SummaryKey.WER.value, SummaryKey.CORPUS.value])

    tk.register_report("summary.report", summary_report)

    return summary_report
