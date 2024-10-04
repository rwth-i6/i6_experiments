from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.returnn.hdf import ReturnnDumpHDFJob
from i6_core.lib.corpus import Corpus

from sisyphus import tk, Job

import resource
import sys

try:
    resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
sys.setrecursionlimit(10**6)
_cf_cache = {}

# def hash_monkey(parsed_args):
#     del parsed_args["bliss_corpus"]
#     return Job.hash(parsed_args)

# SegmentCorpusJob.hash = hash_monkey

def py():
    bliss_path = "/u/minh-nghia.phan/setups/rf_ctc/work/i6_core/corpus/transform/MergeCorporaJob.hlZ8ixhLSaaQ/output/merged.xml.gz"
    bliss_corpus = Corpus()
    bliss_corpus.load(bliss_path) # very heavy file
    segment_job = SegmentCorpusJob(tk.Path(bliss_path), 1000)
    segment_file = segment_job.out_single_segment_files[1]
    train_config = {
        "class": "OggZipDataset",
        "path": [
            "/u/minh-nghia.phan/setups/rf_ctc/work/i6_core/returnn/oggzip/BlissToOggZipJob.Cbboscd6En6A/output/out.ogg.zip"
        ],
        "use_cache_manager": True,
        "audio": {
            "features": "raw",
            "sample_rate": 16000,
            "peak_normalization": True,
            "preemphasis": None,
        },
        "targets": {
            "bpe_file": "/u/minh-nghia.phan/setups/rf_ctc/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes",
            "vocab_file": "/u/minh-nghia.phan/setups/rf_ctc/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab",
            "unknown_label": None,
            "bos_label": 0,
            "eos_label": 0,
        },
        "fixed_random_seed": 1,
        "seq_ordering": "sorted_reverse",
        "segment_file": segment_file
    }
    align_config = {
        "class": "HDFDataset",
        "files": [
            "/work/asr3/zyang/share/mnphan/alignment_data/lbs960/gmm_pseudo_word_alignments.hdf"
        ],
        "use_cache_manager": True,
        "segment_file": segment_file,
    }
    train_hdf = ReturnnDumpHDFJob(train_config).out_hdf
    align_hdf = ReturnnDumpHDFJob(align_config).out_hdf
    tk.register_output("lbs_very_small_subset/train.hdf", train_hdf)
    tk.register_output("lbs_very_small_subset/align.hdf", align_hdf)
