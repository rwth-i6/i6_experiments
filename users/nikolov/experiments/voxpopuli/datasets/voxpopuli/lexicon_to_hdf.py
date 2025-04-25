__all__ = ["lexicon_to_hdf"]

import os
import sys
from typing import Optional
from sisyphus import setup_path,gs,tk

import i6_core.returnn as returnn
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
sys.path.append("/u/kaloyan.nikolov/experiments/multilang_0325/config")
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.lexicon import get_bliss_lexicon
from i6_experiments.users.berger.recipe.corpus.transform import ReplaceUnknownWordsJob
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.default_tools import MINI_RETURNN_ROOT

def lexicon_to_hdf(
    #corpus_path: tk.Path,
    lexicon_path: Optional[tk.Path] = None):
	
    splits = ['train', 'test', 'dev']
    langs = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
    if lexicon_path == None:
        lexicon_path = get_bliss_lexicon(
            subdir_prefix = "vox_full",
            raw_lexicon_path = tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/lex.txt"),
            full_text_path = tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/all.txt"),
            bpe_size=81920)
	
    for lang in langs:
        for split in splits:
				
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.{split}.corpus.xml.gz")
			
			
            j = BlissCorpusToTargetHdfJob(
                            bliss_corpus=ReplaceUnknownWordsJob(corpus, lexicon_file=lexicon_path).out_corpus_file,
							bliss_lexicon=lexicon_path,
							returnn_root=MINI_RETURNN_ROOT,
			)

			
            tk.register_output(
                            f"voxpopuli_asr_lexicon/{lang}/{split}.hdf",
                            j.out_hdf,
            )

