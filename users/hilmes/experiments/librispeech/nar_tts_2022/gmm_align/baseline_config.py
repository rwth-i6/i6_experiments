"""
Definition of the pipeline in terms of inputs and steps that are executed
"""
from sisyphus import gs, tk

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.gmm_align import (
  baseline_args,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.gmm_align.data import (
  get_corpus_data_inputs,
  get_corpus_data_inputs_no_sil_p,
)

from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.gmm_align.default_tools import (
  RASR_BINARY_PATH,
)
from i6_core.report.report import MailJob, GenerateReportStringJob
from i6_experiments.users.hilmes.tools.report import gmm_example_report_format


def run_librispeech_100_common_tts_baseline(
  alias_prefix="experiments/librispeech/nar_tts_2022/gmm_align/baseline",
):

  stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
  gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

  hybrid_init_args = baseline_args.get_init_args()
  mono_args = baseline_args.get_monophone_args()
  # no unknown question needed when G2P is used
  cart_args = baseline_args.get_cart_args(add_unknown=False)
  tri_args = baseline_args.get_triphone_args()
  vtln_args = baseline_args.get_vtln_args()
  sat_args = baseline_args.get_sat_args()
  vtln_sat_args = baseline_args.get_vtln_sat_args()

  final_output_args = OutputArgs("final")
  final_output_args.define_corpus_type("train-clean-100", "train")
  final_output_args.define_corpus_type("tts_align", "test")
  final_output_args.define_corpus_type("dev-other", "dev")
  final_output_args.define_corpus_type("dev-clean", "dev")

  steps = RasrSteps()
  steps.add_step("extract", hybrid_init_args.feature_extraction_args)
  steps.add_step("mono", mono_args)
  steps.add_step(
    "forced_align_mono",
    {
      "name": "tts_align_mono",
      "target_corpus_key": "tts_align",
      "flow": "mfcc+deriv+norm",
      "feature_scorer": ("train-clean-100", "train_mono"),
      "corpus_keys": ["tts_align"],
    },
  )
  steps.add_step("cart", cart_args)
  steps.add_step("tri", tri_args)
  steps.add_step("vtln", vtln_args)
  steps.add_step("sat", sat_args)
  steps.add_step(
    "forced_align_sat",
    {
      "name": "tts_align_sat",
      "target_corpus_key": "tts_align",
      "flow": sat_args.training_args["feature_flow_key"],
      "feature_scorer": ("train-clean-100", "train_sat"),
      "corpus_keys": ["tts_align"],
    },
  )
  steps.add_step("vtln+sat", vtln_sat_args)
  steps.add_step("output", final_output_args)

  corpus_data = get_corpus_data_inputs()

  system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
  system.init_system(
    rasr_init_args=hybrid_init_args,
    train_data=corpus_data.train_data,
    dev_data=corpus_data.dev_data,
    test_data=corpus_data.test_data,
  )
  system.run(steps)

  gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
  scores = {}
  for set in ["dev-clean", "dev-other"]:
    for job in system.jobs[set]:
      if job.startswith("scorer") and "optlm" in job:
        scores[job] = system.jobs[set][job].out_wer
  import getpass

  user = getpass.getuser()
  scores["user"] = user
  scores["name"] = alias_prefix
  content = GenerateReportStringJob(report_values=scores, report_template=gmm_example_report_format).out_report
  report = MailJob(subject=alias_prefix, result=content, send_contents=True)
  tk.register_output(f"reports/{alias_prefix}", report.out_status)

  alignments = {}
  for align in ["tts_align_mono", "tts_align_sat"]:
    alignments[align] = system.alignments["tts_align"][align].alternatives["bundle"]
  return (
    alignments,
    system.allophone_files["train-clean-100"],
  )


def run_librispeech_100_common_tts_baseline_no_silence_prep(
  alias_prefix="experiments/librispeech/nar_tts_2022/gmm_align/no_sil_prep",
):

  stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
  gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

  hybrid_init_args = baseline_args.get_init_args()
  mono_args = baseline_args.get_monophone_args()
  # no unknown question needed when G2P is used
  cart_args = baseline_args.get_cart_args(add_unknown=False)
  tri_args = baseline_args.get_triphone_args()
  vtln_args = baseline_args.get_vtln_args()
  sat_args = baseline_args.get_sat_args()
  vtln_sat_args = baseline_args.get_vtln_sat_args()

  final_output_args = OutputArgs("final")
  final_output_args.define_corpus_type("train-clean-100", "train")
  final_output_args.define_corpus_type("tts_align", "test")
  final_output_args.define_corpus_type("dev-other", "dev")

  steps = RasrSteps()
  steps.add_step("extract", hybrid_init_args.feature_extraction_args)
  steps.add_step("mono", mono_args)
  steps.add_step(
    "forced_align_mono",
    {
      "name": "tts_align_mono",
      "target_corpus_key": "tts_align",
      "flow": "mfcc+deriv+norm",
      "feature_scorer": ("train-clean-100", "train_mono"),
      "corpus_keys": ["tts_align"],
    },
  )
  steps.add_step("cart", cart_args)
  steps.add_step("tri", tri_args)
  steps.add_step("vtln", vtln_args)
  steps.add_step("sat", sat_args)
  steps.add_step(
    "forced_align_sat",
    {
      "name": "tts_align_sat",
      "target_corpus_key": "tts_align",
      "flow": sat_args.training_args["feature_flow_key"],
      "feature_scorer": ("train-clean-100", "train_sat"),
      "corpus_keys": ["tts_align"],
    },
  )
  steps.add_step("vtln+sat", vtln_sat_args)
  steps.add_step("output", final_output_args)

  corpus_data = get_corpus_data_inputs_no_sil_p()

  system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
  system.init_system(
    rasr_init_args=hybrid_init_args,
    train_data=corpus_data.train_data,
    dev_data=corpus_data.dev_data,
    test_data=corpus_data.test_data,
  )
  system.run(steps)

  gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
  alignments = {}

  scores = {}
  for set in ["dev-clean", "dev-other"]:
    for job in system.jobs[set]:
      if job.startswith("scorer") and "optlm" in job:
        scores[job] = system.jobs[set][job].out_wer
  import getpass

  user = getpass.getuser()
  scores["user"] = user
  scores["name"] = alias_prefix
  content = GenerateReportStringJob(report_values=scores, report_template=gmm_example_report_format).out_report
  report = MailJob(subject=alias_prefix, result=content, send_contents=True)
  tk.register_output(f"reports/{alias_prefix}", report.out_status)

  for align in ["tts_align_sat"]:
    alignments[align + "/no_sil_p"] = system.alignments["tts_align"][align].alternatives["bundle"]
  return (
    alignments,
    system.allophone_files["train-clean-100"],
  )


def run_librispeech_100_common_tts_sil_prep_no_speaker(
  alias_prefix="experiments/librispeech/nar_tts_2022/gmm_align/sil_prep_no_speaker",
):

  stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
  gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

  hybrid_init_args = baseline_args.get_init_args()
  mono_args = baseline_args.get_monophone_args()
  # no unknown question needed when G2P is used
  cart_args = baseline_args.get_cart_args(add_unknown=False)
  tri_args = baseline_args.get_triphone_args()
  vtln_args = baseline_args.get_vtln_args()
  sat_args = baseline_args.get_sat_args()
  vtln_sat_args = baseline_args.get_vtln_sat_args()

  final_output_args = OutputArgs("final")
  final_output_args.define_corpus_type("train-clean-100", "train")
  final_output_args.define_corpus_type("tts_align", "test")
  final_output_args.define_corpus_type("dev-other", "dev")
  final_output_args.define_corpus_type("dev-clean", "dev")

  steps = RasrSteps()
  steps.add_step("extract", hybrid_init_args.feature_extraction_args)
  steps.add_step("mono", mono_args)
  steps.add_step(
    "forced_align_mono",
    {
      "name": "tts_align_mono",
      "target_corpus_key": "tts_align",
      "flow": "mfcc+deriv+norm",
      "feature_scorer": ("train-clean-100", "train_mono"),
      "corpus_keys": ["tts_align"],
    },
  )
  steps.add_step("cart", cart_args)
  steps.add_step("tri", tri_args)
  steps.add_step("vtln", vtln_args)
  steps.add_step("sat", sat_args)
  steps.add_step(
    "forced_align_sat",
    {
      "name": "tts_align_sat",
      "target_corpus_key": "tts_align",
      "flow": sat_args.training_args["feature_flow_key"],
      "feature_scorer": ("train-clean-100", "train_sat"),
      "corpus_keys": ["tts_align"],
    },
  )
  steps.add_step("vtln+sat", vtln_sat_args)
  steps.add_step("output", final_output_args)

  corpus_data = get_corpus_data_inputs(remove_speakers=True)

  system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
  system.init_system(
    rasr_init_args=hybrid_init_args,
    train_data=corpus_data.train_data,
    dev_data=corpus_data.dev_data,
    test_data=corpus_data.test_data,
  )
  system.run(steps)

  gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
  scores = {}
  for set in ["dev-clean", "dev-other"]:
    for job in system.jobs[set]:
      if job.startswith("scorer") and "optlm" in job:
        scores[job] = system.jobs[set][job].out_wer
  import getpass

  user = getpass.getuser()
  scores["user"] = user
  scores["name"] = alias_prefix
  content = GenerateReportStringJob(report_values=scores, report_template=gmm_example_report_format).out_report
  report = MailJob(subject=alias_prefix, result=content, send_contents=True)
  tk.register_output(f"reports/{alias_prefix}", report.out_status)

  alignments = {}
  for align in ["tts_align_sat"]:
    alignments[align] = system.alignments["tts_align"][align].alternatives["bundle"]
  return (
    alignments,
    system.allophone_files["train-clean-100"],
  )
