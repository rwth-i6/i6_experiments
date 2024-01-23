from sisyphus import gs, tk

# global imports for "static" components
import copy

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

# relative imports for files that should be copied for new setups

from i6_experiments.common.baselines.librispeech.data import get_corpus_data_inputs
from i6_experiments.common.baselines.librispeech.ls960.gmm import baseline_args
from i6_experiments.common.baselines.librispeech.default_tools import RASR_BINARY_PATH

def get_gt_args():
    dc_detection: bool = False
    gt_normalization: bool = True
    gt_options_extra_args: Optional[Dict] = None
    samples_options = {
        "audio_format": "wav",
        "dc_detection": dc_detection,
    }

    # init_args = baseline_args.get_init_args()
    feature_extraction_args = {
        "gt": {
            "gt_options": {
                "minfreq": 100,
                "maxfreq": 7500,
                "channels": 50,
                # "warp_freqbreak": 7400,
                "tempint_type": "hanning",
                "tempint_shift": 0.01,
                "tempint_length": 0.025,
                "flush_before_gap": True,
                "do_specint": False,
                "specint_type": "hanning",
                "specint_shift": 4,
                "specint_length": 9,
                "normalize": gt_normalization,
                "preemphasis": True,
                "legacy_scaling": False,
                "without_samples": False,
                "samples_options": samples_options,
                "normalization_options": {},
            }
        }
    }

    return feature_extraction_args

class SavedSystem:
    def __init__(
        self,
        system,
        rasr_binary_path=None
    ):
        self.system = system
        self.rasr_binary_path = rasr_binary_path

        self._saved_crp = None
    
    def __enter__(self):
        self._saved_crp = copy.deepcopy(self.system.crp)

        if self.rasr_binary_path:
            for crp in self.system.crp.values():
                crp.set_executables(self.rasr_binary_path) 
        
        return self.system
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.system.crp = self._saved_crp


def run_custom_baseline(
    alias_prefix="baselines/librispeech/ls960/gmm/common_baseline",
    recognition=True,
):

    # the RASR-System pipelines need global alias and output settings
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    # ******************** GMM parameters ********************

    # rasr_init_args = get_init_args_gt()
    rasr_init_args = baseline_args.get_init_args()
    mono_args = baseline_args.get_monophone_args()
    # no unknown question needed when G2P is used
    cart_args = baseline_args.get_cart_args(add_unknown=False)
    tri_args = baseline_args.get_triphone_args()
    vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    vtln_sat_args = baseline_args.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train-other-960", "train")
    # final_output_args.define_corpus_type("dev-clean", "dev")
    final_output_args.define_corpus_type("dev-other", "dev")
    final_output_args.add_feature_to_extract("gt")

    # **************** GMM step definitions *****************

    steps = RasrSteps()
    # print(rasr_init_args.feature_extraction_args)
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    # ******************** Data ********************

    corpus_data = get_corpus_data_inputs(corpus_key="train-other-960", use_g2p_training=True, use_stress_marker=False)

    # ******************** GMM System ********************

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data if recognition else {},
        test_data=corpus_data.test_data if recognition else {},
    )


    rasr_path_w_new_hash = copy.copy(system.rasr_binary_path)
    rasr_path_w_new_hash.hash_overwrite = "LIBRISPEECH_DEFAULT_APPTAINER_RASR_BINARY_PATH"
    with SavedSystem(system, rasr_path_w_new_hash) as tmp_sys:
        tmp_sys.extract_features(get_gt_args())

    tk.register_output("features/gt/train-other-960", system.feature_bundles["train-other-960"]["gt"])

    # run everything
    system.run(steps)

    # recover alias and output path settings
    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir

    return system
