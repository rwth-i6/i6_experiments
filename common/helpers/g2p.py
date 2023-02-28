__all__ = ["G2PBasedOovAugmenter"]

import os
from typing import Optional

from sisyphus import tk

Path = tk.setup_path(__package__)

from i6_core.corpus.stats import ExtractOovWordsFromCorpusJob
from i6_core.g2p.apply import ApplyG2PModelJob
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob, G2POutputToBlissLexiconJob
from i6_core.g2p.train import TrainG2PModelJob


class G2PBasedOovAugmenter:
    """
    This is a class that augment a bliss lexicon with OOV tokens/words for a specific corpus .
    It is possible to train a g2p model for this purpose or give a path to a pre-trained model.
    By default, if no model path is provided, the class will train a g2p model relying on Sequitur.
    Furthermore, it is possible to use a different bliss lexicon for training the g2p model.
    If no train_lexicon is provided, original bliss lexicon is used for training and will be augmented.
    """

    def __init__(
        self,
        original_bliss_lexicon: str,
        train_lexicon: Optional[str] = None,
        g2p_model_path: Optional[Path] = None,
        train_args: Optional[dict] = None,
        apply_args: Optional[dict] = None,
    ):
        """
        :param original_bliss_lexicon: path to the original lexicon that will be augmented with OOVs
        :param train_lexicon: path to the train lexicon in case it differs from the original lexicon
        :param g2p_model_path: path to the g2p model, if None a g2p model is trained
        #######################################
        :param train_args: train_args = {
        "num_ramp_ups"   : 4,
        "min_iter"       : 1,
        "max_iter"       : 60,
        "devel"          : "5%",
        "size_constrains": "0,1,0,1",
        "extra_args": "",
        "g2p_path": Optional[tk.Path] = None,
        "g2p_python": Optional[tk.Path] = None,
        }
        #######################################
        :param apply_args: apply_args = {
        "variants_mass"  : 1.0,
        "variants_number": 1
        "g2p_path": "",
        "g2p_python": "",
        "filter_empty_words": bool,
        "concurrent": int,
        }
        """
        self.original_bliss_lexicon = original_bliss_lexicon
        self.train_lexicon = original_bliss_lexicon if train_lexicon is None else train_lexicon
        self.g2p_model_path = g2p_model_path
        self.train_args = train_args if train_args else {}
        self.apply_args = apply_args if apply_args else {}

    def _train_and_set_g2p_model(self, alias_path: str):
        g2p_lexicon_job = BlissLexiconToG2PLexiconJob(bliss_lexicon=self.train_lexicon)
        g2p_train_job = TrainG2PModelJob(g2p_lexicon=g2p_lexicon_job.out_g2p_lexicon, **self.train_args)
        g2p_lexicon_job.add_alias(os.path.join(alias_path, "convert_bliss_lexicon_to_g2p_lexicon"))
        g2p_train_job.add_alias(os.path.join(alias_path, "train_g2p_model"))
        self.g2p_model_path = g2p_train_job.out_best_model

    def get_g2p_augmented_bliss_lexicon(
        self,
        bliss_corpus: Path,
        corpus_name: str,
        alias_path: str,
        casing: str = "none",
    ):
        extract_oov_job = ExtractOovWordsFromCorpusJob(
            bliss_corpus=bliss_corpus,
            bliss_lexicon=self.original_bliss_lexicon,
            casing=casing,
        )
        extract_oov_job.add_alias(os.path.join(alias_path, "extract-oov-from-{}".format(corpus_name)))

        if self.g2p_model_path is None:
            self._train_and_set_g2p_model(alias_path)

        g2p_apply_job = ApplyG2PModelJob(
            g2p_model=self.g2p_model_path,
            word_list_file=extract_oov_job.out_oov_words,
            **self.apply_args,
        )
        g2p_apply_job.add_alias(os.path.join(alias_path, "apply-g2p-for-{}".format(corpus_name)))

        g2p_final_lex_job = G2POutputToBlissLexiconJob(
            iv_bliss_lexicon=self.original_bliss_lexicon,
            g2p_lexicon=g2p_apply_job.out_g2p_lexicon,
        )
        g2p_final_lex_job.add_alias(os.path.join(alias_path, "g2p-output-to-bliss-{}".format(corpus_name)))

        return g2p_final_lex_job.out_oov_lexicon
