import getpass
import copy
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import SummaryKey
from sisyphus import tk, gs
from .config_01_conformer_ctc_old import py as py_01_old
from .config_01_conformer_ctc import py as py_01

from .config_02_conformer_transducer_phon_viterbi import py as py_02

# from .config_02a_conformer_transducer_phon_viterbi_tuning import py as py_02a
# from .config_03_conformer_transducer_phon_fullsum import py as py_03
from .config_03b_conformer_transducer_phon_fullsum_scratch import py as py_03b
from .config_05_conformer_transducer_phon_align_restrict import py as py_05
# from .config_05a_conformer_transducer_phon_align_restrict_tuning import py as py_05a

# from .config_04a_conformer_transducer_bpe import py as py_04a
# from .config_04a_conformer_transducer_bpe_rasr import py as py_04a_rasr
# from .config_04b_conformer_transducer_phon import py as py_04b


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
            "OptunaReportIntermediateScoreJob",
            "OptunaReportFinalScoreJob",
        }
        onnx_jobs = {
            "ExportPyTorchModelToOnnxJob",
            "TorchOnnxExportJob",
            "OptunaExportPyTorchModelToOnnxJob",
            "OptunaTorchOnnxExportJob",
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

        app_call += ["--env", f"NUMBA_CACHE_DIR=/var/tmp/numba_cache_{getpass.getuser()}"]

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
        copy.deepcopy(py_01_old()[0]),
        copy.deepcopy(py_01()[0]),
        copy.deepcopy(py_02()[0]),
        # copy.deepcopy(py_02a()),
        # copy.deepcopy(py_03()),
        # copy.deepcopy(py_03b()),
        # copy.deepcopy(py_05()),
        # copy.deepcopy(py_05a()),
    ]:
        subreport.collapse([SummaryKey.CORPUS.value], best_selector_key=SummaryKey.ERR.value)
        summary_report.merge_report(subreport, update_structure=True)

    summary_report.set_col_sort_key([SummaryKey.ERR.value, SummaryKey.WER.value, SummaryKey.CORPUS.value])

    tk.register_report("summary.report", summary_report)

    return summary_report
