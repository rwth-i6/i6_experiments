from sisyphus import *

import os

import recipe.i6_experiments.users.mann.setups.nn_system.tedlium as ted

from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob

from i6_core import (
    rasr, tools
)

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

ted_system = ted.get_tedlium_system(
    full_train_corpus=False,
)

def py():
    in_lexicon = ted_system.crp["train"].lexicon_config.file
    add_eow_job = AddEowPhonemesToLexiconJob(in_lexicon)

    tk.register_output("in_lexicon.txt", in_lexicon)
    tk.register_output("out_lexicon.txt", add_eow_job.out_lexicon)
