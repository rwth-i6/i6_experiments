from sisyphus import *

import os

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem
from recipe.i6_experiments.users.mann.setups import prior
from recipe.i6_experiments.common.datasets import librispeech

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
# s = LibriNNSystem(epochs=[12, 24, 32, 48, 80, 160], num_input=50)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

librispeech.export_all(fname)

corpus_object_dict = librispeech.get_corpus_object_dict()
lm_dict = librispeech.get_arpa_lm_dict()
lexicon_dict = librispeech.get_g2p_augmented_bliss_lexicon_dict()

lbs_rasr_system = RasrSystem()

corpus_key = "train-other-960"
rasr_data_input = RasrDataInput(
    corpus_object_dict[corpus_key],
    {"filename": lexicon_dict[corpus_key], "normalize_pronunciation": False}
)

print(corpus_object_dict[corpus_key])
print(lexicon_dict[corpus_key])

lbs_rasr_system.add_corpus("train", rasr_data_input, add_lm=False)

print(lbs_rasr_system.crp["train"].corpus_config)
print(lbs_rasr_system.crp["train"].lexicon_config)


from collections import namedtuple
from itertools import product
StateTying = namedtuple("StateTying", "path num_states partition silence_idx")

tying_names = [
    "{}State-we-{}".format(p, we) for p, we in product([1, 3], ["yes", "no"])
]

tyings = {
    (1, "yes", 84, 81),
    (1, "no", 42, 40),
    (3, "yes", 252, 241),
    (3, "no", 126, 120)
}

path_template = "/work/asr4/raissi/setups/librispeech/960-ls/2022-03--adapt_pipeline/output/baseline_fh/daniel/monophone-dense-{}State-we-{}"

state_tyings = {}

for p, we, ns, si in tyings:
    path = path_template.format(p, we)
    name = os.path.basename(path)
    print(name)
    state_tyings[name] = StateTying(path, ns, p, si)

priors = {}
for name, state_tying in state_tyings.items():
    priors[name] = prior.get_prior_from_transcription_new(
        lbs_rasr_system,
        output_stats=True,
        num_states=state_tying.num_states,
        silence_idx=state_tying.silence_idx,
        hmm_partition=state_tying.partition,
        state_tying=state_tying.path,
        total_frames=330216469
    )


from recipe.i6_experiments.users.mann.setups.reports import DescValueReport, SimpleValueReport

for name, pr in priors.items():
    # continue
    tk.register_callback(prior.plot_prior, pr, fname, "plot_prior.{}.png".format(name))
    opath = os.path.join(fname, f"prior-{name}")
    tk.register_report(opath, SimpleValueReport(pr))
