import sys
from typing import Dict, Union, List, Tuple

from sisyphus import gs

from i6_core import rasr

from i6_experiments.common.setups.hybrid.rasr_system import RasrSystem
from i6_experiments.common.setups.hybrid.util import RasrInitArgs, RasrDataInput


class CtcSystem(RasrSystem):
    """
    - 3 corpora types: train, dev and test
    - only train corpora will be aligned
    - dev corpora for tuning
    - test corpora for final eval

    to create beforehand:
    - corpora: name and i6_core.meta.system.Corpus
    - lexicon
    - lm

    settings needed:
    - am
    - lm
    - lexicon
    - feature extraction
    """

    def __init__(self):
        super().__init__()

        self.ctc_am_args = None

        self.default_align_keep_values = {
            "default": 5,
            "selected": gs.JOB_DEFAULT_KEEP_VALUE,
        }

    # -------------------- Setup --------------------
    def init_system(
            self,
            rasr_init_args: RasrInitArgs,
            train_data: Dict[str, RasrDataInput],
            dev_data: Dict[str, RasrDataInput],
            test_data: Dict[str, RasrDataInput],
    ):
        self.rasr_init_args = rasr_init_args

        self._init_am(**self.rasr_init_args.am_args)

        self._assert_corpus_name_unique(train_data, dev_data, test_data)

        for name, v in sorted(train_data.items()):
            add_lm = True if v.lm is not None else False
            self.add_corpus(name, data=v, add_lm=add_lm)
            self.train_corpora.append(name)

        for name, v in sorted(dev_data.items()):
            self.add_corpus(name, data=v, add_lm=True)
            self.dev_corpora.append(name)

        for name, v in sorted(test_data.items()):
            self.add_corpus(name, data=v, add_lm=True)
            self.test_corpora.append(name)


    def run(self, steps: Union[List, Tuple] = ("all",)):
        """
        run setup

        :param steps:
        :return:
        """
        assert len(steps) > 0
        if len(steps) == 1 and steps[0] == "all":
            steps = ["extract", "mono", "cart", "tri", "vtln", "sat", "vtln+sat"]

        if "init" in steps:
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        for all_c in self.train_corpora + self.dev_corpora + self.test_corpora:
            self.costa(all_c, prefix="costa/", **self.rasr_init_args.costa_args)

        for trn_c in self.train_corpora:
            self.store_allophones(trn_c)

        for eval_c in self.dev_corpora + self.test_corpora:
            self.create_stm_from_corpus(eval_c)
            self.set_sclite_scorer(eval_c)

        if "extract" in steps:
            self.extract_features(
                feat_args=self.rasr_init_args.feature_extraction_args
            )
