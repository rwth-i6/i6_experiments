__all__ = ["lexicon_to_hdf"]

import os
import sys
from sisyphus import setup_path,gs,tk

import i6_core.returnn as returnn
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
sys.path.append("/u/kaloyan.nikolov/experiments/multilang_0325/config")
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.gen_lexicon import get_bpe_lexicon


def lexicon_to_hdf():
	
    splits = ['train']#, 'test', 'dev']
    langs = ["cs"]#, "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
    lex = get_bpe_lexicon()	
	
    for lang in langs:
        for split in splits:
				
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.{split}.corpus.xml.gz")
			
			
            j = returnn.BlissCorpusToTargetHDFJob(
                            bliss_corpus=corpus,
							bliss_lexicon=lex,
							returnn_root=RETURNN_ROOT,
			)

			
            tk.register_output(
                            f"voxpopuli_asr_lexicon/{lang}/{split}.hdf",
                            j.out_hdf,
            )

