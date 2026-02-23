__all__ = ["generate_voxpopuli_ogg_corpus"]

import os
from sisyphus import setup_path,gs,tk

#import i6_core.returnn as returnn
from i6_core.returnn.oggzip import BlissToOggZipJob



def generate_voxpopuli_ogg_corpus(): #corpus_path: tk.Path):
    RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/git/returnn")
    
    splits = ['train', 'test', 'dev']
    langs = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]

    for lang in langs:
        for split in splits:
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/ogg/asr_{lang}.{split}.corpus.xml.gz")

            j = BlissToOggZipJob(
                            bliss_corpus=corpus,
                            returnn_root=RETURNN_ROOT,

            )
            j.rqmt = {"cpu": 1, "mem": 1, "time": 72}

            tk.register_output(
                            f"voxpopuli_asr_ogg/{lang}/{split}.zip",
                            j.out_ogg_zip,
            ) 
