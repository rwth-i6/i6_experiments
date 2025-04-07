import os
from sisyphus import setup_path,gs,tk

import i6_core.returnn as returnn


def py():
    corpus = tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_lt.train.corpus.xml.gz")

    j = returnn.BlissToPcmHDFJob(
                    bliss_corpus=corpus,
    )
    tk.register_output(
                    f"voxpopuli_asr_lt.train.hdf",
                    j.out_hdf,
    ) 
