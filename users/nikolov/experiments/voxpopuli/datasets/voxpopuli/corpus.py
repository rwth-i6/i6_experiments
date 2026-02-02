__all__ = ["generate_voxpopuli_corpus", "generate_csfleurs_corpus"]

import os
from sisyphus import setup_path,gs,tk
import returnn
import i6_core.returnn as returnn


def generate_voxpopuli_corpus(): #corpus_path: tk.Path):
    
    splits = ['train', 'test', 'dev']
    langs = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]

    for lang in langs:
        for split in splits:
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.{split}.corpus.xml.gz")

            j = returnn.BlissToPcmHDFJob(
                            bliss_corpus=corpus,
            )

            tk.register_output(
                            f"voxpopuli_asr/{lang}/{split}.hdf",
                            j.out_hdf,
            ) 

def generate_csfleurs_corpus(): #corpus_path: tk.Path):
    
    splits = ['test']
    datasets = ['mms', 'read', 'xtts']

    for dataset in datasets:
        for split in splits:
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/csfleurs/corpus/{dataset}.{split}.corpus.xml.gz")

            j = returnn.BlissToPcmHDFJob(
                            bliss_corpus=corpus,
            )

            tk.register_output(
                            f"csfleurs_asr/{dataset}/{split}.hdf",
                            j.out_hdf,
            ) 

def generate_fleurs_corpus(): #corpus_path: tk.Path):
    
    splits = ['test']
    langs = ['en_us', 'es_419']
    for lang in langs:
        for split in splits:
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/fleurs/corpus/{lang}/{split}.corpus.xml.gz")
            j = returnn.BlissToPcmHDFJob(
                            bliss_corpus=corpus,
            )
            tk.register_output(
                            f"fleurs_asr/{lang}/{split}.hdf",
                            j.out_hdf,
            )
            
def generate_switchlingua_corpus(): #corpus_path: tk.Path):
    
    splits = ['dev']
    for split in splits:
        corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/{split}.corpus.xml.gz")
        j = returnn.BlissToPcmHDFJob(
                        bliss_corpus=corpus,
        )
        tk.register_output(
                        f"switchlingua_asr/{split}.hdf",
                        j.out_hdf,
        )
        
def generate_switchlingua_tts_corpus(): #corpus_path: tk.Path):
    
    #splits = ['dev']
    
    corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/TTS/generated_audio/generated/train.corpus.xml.gz")
    j = returnn.BlissToPcmHDFJob(
                    bliss_corpus=corpus,
    )
    tk.register_output(
                    f"switchlingua_asr_tts/train.hdf",
                    j.out_hdf,
    )
    
def generate_miami_corpus(): #corpus_path: tk.Path):
    
    splits = ['test']
    datasets = ['full', 'eng', 'spa']
    for dataset in datasets:
        for split in splits:
            corpus = tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
            j = returnn.BlissToPcmHDFJob(
                            bliss_corpus=corpus,
            )
            tk.register_output(
                            f"miami_asr/{dataset}/{split}.hdf",
                            j.out_hdf,
            )
