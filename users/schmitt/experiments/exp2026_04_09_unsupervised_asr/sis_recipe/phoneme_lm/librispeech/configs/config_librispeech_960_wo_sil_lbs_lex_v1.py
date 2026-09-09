"""
Phoneme LM on the train-other-960 transcripts phonemized in the *LibriSpeech GMM phoneme set* (official
LibriSpeech lexicon, stress stripped, aligner-preferred variants, adjacent repeats collapsed; see
`data.librispeech.text.get_lbs_lexicon`) instead of our fairseq/g2p_en set. Vocab ``[SILENCE]=0, AA=1, ...,
ZH=39`` (size 40, index 0 unused). Same model / schedule as `config_librispeech_960_wo_sil_v1`, so the two LMs
differ only in the phonemization. Companion of the unsup
`config_librispeech_960_wo_sil_cheat_seg_lbs_lex_v1` (same lexicon object -> same phoneme HDFs), e.g. for
LM-fused recognition or distribution-matching there. PPL is scored on the full 2864-utt dev-other (every seq is
kept with a lexicon), so it is not comparable to the historical LM's 2712-utt PPL either.
"""

import copy

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.librispeech.text import get_lbs_lexicon
from ... import __setup_base_name__

from .config_librispeech_960_wo_sil_v1 import base_config as base_config_, settings, base_num_epochs, get_keep_epochs


lexicon = get_lbs_lexicon(variant_selection="aligner", collapse_repeats=True)

train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings, lexicon=lexicon)
test_data_dict_wo_sil = build_test_datasets(sil_prob=0.0, surround_w_sil=False, lexicon=lexicon)

base_config = copy.deepcopy(base_config_)
base_config["model_args"]["out_dim"] = train_data.datastreams["data"].vocab_size
assert base_config["model_args"]["out_dim"] == 40, base_config["model_args"]["out_dim"]


# see config_librispeech_960_wo_sil_v1: net_args + checkpoint key for configs fusing with this LM
lm_model_args = base_config["model_args"]
lm_prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"
lm_training_name = f"{lm_prefix_name}/baseline"


def py(checkpoints=None):
    """:param checkpoints: if given, the trained LM checkpoints are registered here under its
    training_name, so other configs (LM-fused recognition) can pick them up."""
    train_job = run_experiment(
        training_name=lm_training_name,
        config=copy.deepcopy(base_config),
        train_data=train_data,
        test_data_dict=test_data_dict_wo_sil,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=True,
        ppl_opts={
            "checkpoints": get_keep_epochs(base_num_epochs),
        },
    )
    if checkpoints is not None:
        checkpoints[lm_training_name] = train_job.out_checkpoints
