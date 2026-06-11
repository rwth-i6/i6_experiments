"""CrisperWhisper OFFICIAL word-timestamp pipeline (hyp-mode baseline).

CrisperWhisper (Nyra Health; arXiv 2408.16589) is whisper-large-v2 finetuned for
verbatim transcription with a retokenized vocab (spaces as standalone tokens,
uh/um as filled-pause tokens) and alignment heads finetuned with an attention
loss on timestamped data. Word timestamps come from DTW over the L2-normalized
head-averaged cross-attention at DECODING time (no forced mode), followed by a
pause-splitting post-step (each pause split evenly between the neighboring
words, capped at 160 ms).

This job runs THEIR pipeline as published:
their transformers fork (``nyrahealth/transformers@crisper_whisper``, required
for the retokenized word-timestamp extraction) shadows the installed
transformers, and ``adjust_pauses_for_hf_pipeline_output`` is loaded from their
repo's ``utils.py``. Decoding via the HF ASR pipeline with
``return_timestamps="word"``.

Outputs hyp transcripts (TextDict, ``seq-{idx}``) + word boundaries
(float seconds, ``[n_words, 2]`` per seq) for the hyp-mode metric chain:
:class:`..hyp_align.BuildDatasetWithHypTranscriptsJob` ->
:class:`..hyp_align.CalcHypAlignMetricsJob`.
Their alignment heads were selected/trained on timestamped TIMIT, so TIMIT
evals of this row are contaminated; Buckeye is the clean comparison.
"""

import os
import sys

from typing import Optional
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class CrisperWhisperOfficialAlignJob(Job):
    """Decode + word timestamps with the official CrisperWhisper pipeline."""

    def __init__(
        self,
        *,
        model_dir: tk.Path,
        transformers_fork_dir: tk.Path,
        crisper_repo_dir: tk.Path,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param model_dir: hub cache dir for ``nyrahealth/CrisperWhisper``.
        :param transformers_fork_dir: checkout of
            ``nyrahealth/transformers@crisper_whisper`` (``src/`` is put on
            ``sys.path`` ahead of the installed transformers).
        :param crisper_repo_dir: checkout of ``nyrahealth/CrisperWhisper``
            (provides ``utils.adjust_pauses_for_hf_pipeline_output``).
        :param dataset_dir: dataset (hub-cache layout) to decode.
        :param dataset_key: split to process.
        """
        super().__init__()
        self.model_dir = model_dir
        self.transformers_fork_dir = transformers_fork_dir
        self.crisper_repo_dir = crisper_repo_dir
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root

        self.rqmt = {"time": 12, "cpu": 4, "gpu": 1, "mem": 50}
        self.out_hyps_txt = self.output_path("hyps.txt.gz")
        self.out_word_boundaries_hdf = self.output_path("word_boundaries.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import gzip
        import time
        import importlib.util

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        # The fork must shadow the installed transformers; insert LAST so it
        # ends up first on sys.path.
        fork_src = os.path.join(self.transformers_fork_dir.get_path(), "src")
        assert os.path.isdir(fork_src), fork_src
        sys.path.insert(0, fork_src)

        # The nyrahealth fork (transformers 4.37.2) hard-pins tokenizers<0.19, but
        # the env has 0.21.4; the fork's import-time dependency check raises
        # ImportError. Shim importlib.metadata.version("tokenizers") to a
        # satisfying value BEFORE importing transformers (no repo/env edits).
        # Validated on the GPU test node: import + pipeline + word-timestamp
        # decode (with their [UH]/[UM] retokenization + pause-split) all work.
        import importlib.metadata as _md

        _orig_md_version = _md.version

        def _shim_md_version(name, *a, **kw):
            if name == "tokenizers":
                return "0.18.0"
            return _orig_md_version(name, *a, **kw)

        _md.version = _shim_md_version

        import numpy as np
        import torch
        import transformers

        print("transformers:", transformers.__version__, "from", transformers.__file__)
        assert os.path.realpath(transformers.__file__).startswith(os.path.realpath(fork_src)), (
            "the nyrahealth fork did not shadow the installed transformers"
        )

        from returnn.util import better_exchook
        from returnn.datasets.hdf import SimpleHDFWriter

        better_exchook.install()

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        # Their pause-splitting post-step, loaded from the repo's utils.py by
        # explicit file path (a bare top-level ``utils`` import is collision-prone).
        utils_py = os.path.join(self.crisper_repo_dir.get_path(), "utils.py")
        spec = importlib.util.spec_from_file_location("crisper_utils", utils_py)
        crisper_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crisper_utils)
        adjust_pauses = crisper_utils.adjust_pauses_for_hf_pipeline_output

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        model_id = get_content_dir_from_hub_cache_dir(self.model_dir)
        print(f"Loading {model_id} (dtype={dtype})...")
        start_time = time.time()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(dev)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=1,
            return_timestamps="word",
            torch_dtype=dtype,
            device=dev,
        )
        print(f"  ({time.time() - start_time:.1f}s)")

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        print(f"Num seqs: {len(ds)}")

        hyps = {}
        n_empty = 0
        writer = SimpleHDFWriter(self.out_word_boundaries_hdf.get_path(), dim=2, ndim=2)
        for seq_idx, data in enumerate(ds):
            start_time = time.time()
            audio = np.asarray(data["audio"]["array"], dtype=np.float32)
            sr = int(data["audio"]["sampling_rate"])
            out = pipe({"raw": audio, "sampling_rate": sr})
            out = adjust_pauses(out)
            words = []
            times = []
            for ch in out["chunks"]:
                # Inner whitespace removed so the word list survives the
                # TextDict round-trip (BuildDatasetWithHypTranscriptsJob does
                # ``.split()``) with counts matching the HDF rows 1:1.
                w = "".join(str(ch["text"]).split())
                if not w:
                    continue
                s, e = ch["timestamp"]
                if s is None:
                    s = times[-1][1] if times else 0.0
                if e is None:
                    e = s
                words.append(w)
                times.append((float(s), float(e)))
            if not words:
                # Empty hyp: keep the seq; the hyp-dataset job substitutes a
                # 1-word placeholder, so write one dummy boundary row to match.
                n_empty += 1
                times = [(0.0, 0.0)]
            hyps[f"seq-{seq_idx}"] = " ".join(words)
            writer.insert_batch(np.array([times], dtype="float32"), seq_len=[len(times)], seq_tag=[f"seq-{seq_idx}"])
            if seq_idx % 50 == 0:
                print(
                    f"seq {seq_idx}: {len(words)} words ({time.time() - start_time:.2f}s) {' '.join(words[:8])!r}...",
                    flush=True,
                )
        writer.close()
        with gzip.open(self.out_hyps_txt.get_path(), "wt") as f:
            f.write(repr(hyps))
        print(f"done: {len(ds)} seqs ({n_empty} empty hyps)")
