"""Speed-comparison + correctness-control recipe
(torch 2.12 companion to exp2026_05_23_grad_align).

Run under a SEPARATE torch-2.12 Sisyphus manager
(full env path, no py7 alias),
isolated from the main recipe's torch-2.7 manager.
Home for ALL grad-align extraction speed comparisons,
across models and backward strategies,
measured on ONE torch version / GPU so the wall-times are comparable:

  - seq_batch_size: B>1 forward+backward
    (amortizes the launch-bound per-seq kernels);
    applies to every model whose adapter has a B>1 forward.
  - batched_backward: the vmapped per-token VJP (is_grads_batched)
    vs K sequential backwards.
  - split_prefix_backward: CTC-prefix-only;
    loop the compiled-scan prefix per token,
    vmap the encoder backward
    (the only way to use the 3-5x-faster compiled scan, which can't be vmapped).
    Needs torch 2.12.

CORRECTNESS CONTROL: per-grad cosine equivalence isn't enough.
The headline WBEs were computed with different code/settings,
so for each model we run a SEQUENTIAL ground-truth path
(no vmap, no seq-batch, no split)
AND every fast path through to the WBE,
with identical align settings.
All variants of a model must produce the SAME WBE
(the fast paths only reorder fp ops);
the wall-time gap is the speed result.

Add models by extending the loop;
non-prefix models simply omit the split variant.
"""

from __future__ import annotations

import returnn.frontend as rf
from sisyphus import tk

from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJobV2
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.wav2vec2_ctc import Wav2Vec2Ctc
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_per_token_grads import (
    ExtractInGradsPerTokenJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.word_align_from_per_token_grads import (
    WordAlignFromPerTokenGradsJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.buckeye_fine_dataset import (
    BuildBuckeyeFineDatasetJob,
)

# Same align settings as the main recipe's grid[0],
# so the control WBEs are comparable across variants.
_ALIGN_OPTS = {"apply_softmax_over_time": True, "blank_score": -6}


def py():
    """Sisyphus entry."""
    dl = DownloadHuggingFaceRepoJobV2(repo_id="alexwengg/buckeye", repo_type="dataset")
    ds = BuildBuckeyeFineDatasetJob(
        raw_dir=dl.out_hub_cache_dir,
        resegment_gap_s=1.0,
        split_up_to_max_seq_len_s=18.0,
        min_words=2,
        skip_misaligned_wavs=True,
        subsample_target_h=0.25,
        subsample_seed=42,
    )
    sb_dir = ds.out_hub_cache_dir

    def _extract_and_wbe(name: str, model_config, **kwargs):
        ex = ExtractInGradsPerTokenJob(
            dataset_dir=sb_dir,
            dataset_key="test",
            model_config=model_config,
            mult_grad_by_inputs=False,
            attr_reduction="L2",
            **kwargs,
        )
        ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        ex.rqmt = {**ex.rqmt, "time": 4}
        ex.add_alias(f"speedcmp-p212/{name}")
        tk.register_output(f"speedcmp-p212/{name}.hdf", ex.out_hdf)
        al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=ex.out_hdf,
            grad_score_key="data",
            dataset_dir=sb_dir,
            dataset_key="test",
            dataset_offset_factors=1,
            align_opts=_ALIGN_OPTS,
        )
        al.add_alias(f"speedcmp-p212/{name}-wbe")
        tk.register_output(f"speedcmp-p212/{name}-wbe.txt", al.out_wbe)
        return ex, al

    # === wav2vec2-CTC (prefix_fwd) ===
    # Control: a sequential ground-truth path + every fast path, all the way to WBE;
    # all four variants must produce the same WBE.
    _w2v_pf = rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_fwd")
    _extract_and_wbe("wav2vec2ctc-prefixfwd-seq-bb0-sb1", _w2v_pf, batched_backward=False)  # sequential ground truth
    _extract_and_wbe("wav2vec2ctc-prefixfwd-tokbatch-bb1-sb1", _w2v_pf, batched_backward=True)  # vmap over tokens
    _extract_and_wbe("wav2vec2ctc-prefixfwd-seqbatch-bb1-sb16", _w2v_pf, batched_backward=True, seq_batch_size=16)
    _extract_and_wbe("wav2vec2ctc-prefixfwd-split-sb16", _w2v_pf, split_prefix_backward=True, seq_batch_size=16)

    # === Other models go here (same helper, same dataset).
    # Non-prefix models (whisper / voxtral / ...) omit the split variant
    # and just compare seq-bb0-sb1 (ground truth) against bb1
    # (plus seq_batch where the adapter supports B>1). ===
