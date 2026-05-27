"""Open recognition (greedy decode, no target prompt) on a HuggingFace audio dataset.

Output is a TextDict (``Dict[seq_tag, str]`` saved as gzipped Python repr) so it
can feed straight into the standard sclite WER pipeline via
:func:`i6_experiments.users.zeyer.datasets.utils.sclite_generic_score.sclite_score_hyps_to_ref`.
"""

from typing import Optional, Any, Dict
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class RecogFromModelJob(Job):
    """Run open recognition with a model that exposes ``recog(audio, ...)``.

    Output ``out_hyps_txt`` is a TextDict (``{seq_tag: text}``, gzipped Python
    repr), compatible with :class:`i6_core.returnn.search.SearchWordsDummyTimesToCTMJob`
    and the zeyer ``sclite_score_hyps_to_ref`` helper.
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        model_config: Dict[str, Any],
        max_new_tokens: int = 100,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.model_config = model_config
        self.max_new_tokens = max_new_tokens

        self.rqmt = {"time": 10, "cpu": 2, "gpu": 1, "mem": 80}
        self.out_hyps_txt = self.output_path("hyps.txt.gz")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import time

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch

        from returnn.util import better_exchook
        from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats

        better_exchook.install()

        from .models import make_model

        dev = torch.device("cuda")
        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)
        for p in model.parameters():
            p.requires_grad = False
        report_dev_memory_stats(dev)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        n = len(ds[self.dataset_key])
        print(f"Num seqs: {n}")

        hyps: Dict[str, str] = {}
        total_time = 0.0
        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = data["audio"]["array"]
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            samplerate = data["audio"]["sampling_rate"]

            t0 = time.time()
            hyp_words_batch = model.recog(
                raw_inputs=torch.tensor(audio)[None],
                raw_inputs_sample_rate=samplerate,
                raw_input_seq_lens=torch.tensor([len(audio)]),
                max_new_tokens=self.max_new_tokens,
            )
            elapsed = time.time() - t0
            total_time += elapsed
            hyp_words = hyp_words_batch[0]
            seq_tag = f"seq-{seq_idx}"
            hyps[seq_tag] = " ".join(hyp_words)
            if seq_idx < 5 or seq_idx % 100 == 0:
                ref_words = list(data["word_detail"]["utterance"])
                print(f"seq {seq_idx} ({elapsed:.2f}s)")
                print(f"  ref: {' '.join(ref_words)!r}")
                print(f"  hyp: {hyps[seq_tag]!r}")

        print(f"Total recog time: {total_time:.1f}s = {total_time / 60:.1f}min")
        print(f"Per-seq avg: {total_time / max(n, 1):.2f}s")

        with util.uopen(self.out_hyps_txt.get_path(), "wt") as f:
            f.write(repr(hyps))
        print(f"Wrote {n} hyps to {self.out_hyps_txt.get_path()}")
