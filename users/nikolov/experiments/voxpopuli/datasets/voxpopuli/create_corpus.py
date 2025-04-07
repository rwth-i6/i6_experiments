__all__ = ["generate_voxpopuli_corpus"]

import os
from sisyphus import setup_path,gs,tk

import i6_core.returnn as returnn


def generate_voxpopuli_corpus():
    
    splits = ['train', 'test', 'dev']
    langs = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
    #langs = ["lt"]

    for lang in langs:
        for split in splits:
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.{split}.corpus.xml.gz")

            j = returnn.BlissToPcmHDFJob(
                            bliss_corpus=corpus,
            )

            #j.out_hdf=j.output_path(f"voxpopuli_asr.{lang}.{split}.hdf")
            tk.register_output(
                            f"voxpopuli_asr/{lang}/{split}.hdf",
                            j.out_hdf,
            ) 
