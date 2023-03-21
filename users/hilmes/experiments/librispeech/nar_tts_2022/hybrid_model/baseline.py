import copy
from sisyphus import tk, gs

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.report.report import MailJob, GenerateReportStringJob
from i6_experiments.common.setups.rasr.util import RasrSteps
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.baselines.librispeech.default_tools import RASR_BINARY_PATH

from .data import get_corpus_data_inputs
from .baseline_args import get_nn_args, get_feature_extraction_args
from .rc_baseline_config import get_nn_args as get_rc_nn_args
from .default_tools import RETURNN_RC_ROOT

from .data import get_corpus_data_inputs_newcv


def hybrid_report_format(report) -> str:
  out = []
  results = {
      "dev-other": -1,
      "dev-other-focal": -1,
  }
  for step_name, score in report.items():
      if not step_name.startswith("scorer"):
          continue
      if "dev-other" in step_name:
          if "focal" in step_name:
            results["dev-other-focal"] = score
          else:
            results["dev-other"] = score
  out.append(
    f"""
        Name: {report["name"]}
        Dev-other: {results["dev-other"]}
        Dev-ohter-focal: {results["dev-other-focal"]}
"""
  )
  return "\n".join(out)

def run_gmm_system_from_common():
    from i6_experiments.common.baselines.librispeech.ls100.gmm.baseline_config import run_librispeech_100_common_baseline
    system = run_librispeech_100_common_baseline(
        extract_additional_rasr_features=get_feature_extraction_args()
    )
    return system

def run_hybrid_baseline_rc(gmm_system, key):

    alias_prefix = f'experiments/librispeech/librispeech_100_hybrid/{key}'
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    gmm_system = copy.deepcopy(gmm_system)
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)
    rasr_init_args.feature_extraction_args = get_feature_extraction_args()
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    nn_args = get_rc_nn_args(num_epochs=125)
    nn_args.training_args["partition_epochs"] = {"train": 10 if not ("1000" in key or "860" in key) else 100, "dev": 1}
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)

    # ******************** NN System ********************

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so", hash_overwrite="TF23_MKL_BLAS")

    lbs_nn_system = HybridSystem(returnn_root=RETURNN_RC_ROOT, returnn_python_exe=returnn_exe, blas_lib=blas_lib,
                                 rasr_arch="linux-x86_64-standard", rasr_binary_path=RASR_BINARY_PATH)
    lbs_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["train-clean-100.train", "train-clean-100.cv"])],
    )
    lbs_nn_system.run(nn_steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = ""
    scores = {}
    for set in ["dev-other"]:
        for job in lbs_nn_system.jobs[set]:
            if job.startswith("scorer") and "optlm" in job:
                scores[job] = lbs_nn_system.jobs[set][job].out_wer
    import getpass
    user = getpass.getuser()
    scores["user"] = user
    scores["name"] = alias_prefix
    content = GenerateReportStringJob(report_values=scores, report_template=hybrid_report_format).out_report
    report = MailJob(subject=alias_prefix, result=content, send_contents=True)
    tk.register_output(f"reports/{alias_prefix}", report.out_status)
