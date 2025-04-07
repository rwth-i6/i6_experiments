__all__ = ["get_bpe_lexicon"]
import os
from sisyphus import setup_path,gs,tk

import i6_core.returnn as returnn
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.bpe import CreateBPELexiconJob
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt



SUBWORD_NMT_REPO = get_returnn_subword_nmt(
    commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
).copy()

def get_bpe_lexicon() -> tk.Path:
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_lt.train.corpus.xml.gz"),
        bpe_codes=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/codes_file.all"),
        bpe_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/all.vocab"),
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label="<unk>",
    ).out_lexicon

    tk.register_output(
                            f"voxpopuli_asr/lexicon",
                            bpe_lexicon,
    ) 
    return bpe_lexicon


def get_word_lexicon() -> tk.Path:
    bliss_lex = get_bpe_lexicon()
    word_lexicon = BlissLexiconToG2PLexiconJob(
            bliss_lex,
            include_pronunciation_variants=True,
            include_orthography_variants=True,
    ).out_g2p_lexicon

    return word_lexicon
