"""Speed-comparison + correctness-control recipe
(torch 2.12 companion to exp2026_05_23_grad_align).

Run under a SEPARATE torch-2.12 Sisyphus manager
(full env path, no py7 alias),
isolated from the main recipe's torch-2.7 manager.
Home for grad-align extraction speed comparisons,
one representative per model family (CTC / AED / speech-LLM),
measured on ONE torch version / GPU so the wall-times are comparable.

Backward strategies compared:
  - seq_batch_size: B>1 forward+backward
    (amortizes the launch-bound per-seq kernels);
    currently the wav2vec2-CTC adapter only -- whisper/phi4 hard-assert B=1.
  - batched_backward: the vmapped per-token VJP (is_grads_batched)
    vs K sequential backwards;
    the main win for the compute-bound AED/LLM backwards.
  - split_prefix_backward: CTC-prefix-only;
    loop the compiled-scan prefix per token,
    vmap the encoder backward
    (the only way to use the 3-5x-faster compiled scan, which can't be vmapped).
    Needs torch 2.12.

CORRECTNESS CONTROL: per-grad cosine equivalence isn't enough.
The headline WBEs were computed with different code/settings,
so for each family we run a SEQUENTIAL ground-truth path
(no vmap, no seq-batch, no split)
AND every fast path through to the WBE,
with identical align settings.
All variants of a model must produce the SAME WBE
(the fast paths only reorder fp ops);
the wall-time gap is the speed result.

Speech-LLM (phi4) uses eager attention, not FlashAttention:
FlashAttention's backward has no vmap rule, so batched_backward can't traverse it;
eager's forward is a bit slower but the one batched backward more than pays it back.
"""

from __future__ import annotations

import returnn.frontend as rf
from sisyphus import tk

from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJobV2
from i6_experiments.users.zeyer.external_models.phi4multimodal import download_phi4multimodal_model
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.wav2vec2_ctc import Wav2Vec2Ctc
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.whisper import Whisper
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.phi4mm import Phi4MM
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

# transformers 4.x lives in a PYTHONPATH overlay (the 2.12 base env is kept transformers-free / modern);
# only the AED/LLM jobs need it. CTC (torchaudio) runs on the clean base env.
_TF_OVERLAY = "/home/az668407/work/transformers-4x-overlay"


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

    def _extract_and_wbe(
        name,
        model_config,
        *,
        mult_grad_by_inputs=False,
        attr_reduction="L2",
        time=4,
        needs_transformers=False,
        **kwargs,
    ):
        ex = ExtractInGradsPerTokenJob(
            dataset_dir=sb_dir,
            dataset_key="test",
            model_config=model_config,
            mult_grad_by_inputs=mult_grad_by_inputs,
            attr_reduction=attr_reduction,
            **kwargs,
        )
        ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if needs_transformers:
            # whisper/phi4 need transformers; serve it from the 4.x overlay (sis still inserts
            # recipe/returnn/sisyphus on sys.path itself, so only transformers comes from here).
            ex.set_env("PYTHONPATH", _TF_OVERLAY)
        ex.rqmt = {**ex.rqmt, "time": time}
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

    # === CTC family -- wav2vec2-CTC (prefix_fwd) ===
    # The prefix lattice backward is launch-bound at B=1, so this family gets the full ladder:
    # sequential ground truth, token-batch, seq-batch, and the compiled-scan split.
    _w2v = rf.build_dict(Wav2Vec2Ctc, grad_wrt="feat_proj_out", per_token_score="prefix_fwd")
    _extract_and_wbe("ctc-wav2vec2-seq-bb0-sb1", _w2v, batched_backward=False)  # sequential ground truth
    _extract_and_wbe("ctc-wav2vec2-tokbatch-bb1-sb1", _w2v, batched_backward=True)  # vmap over tokens
    _extract_and_wbe("ctc-wav2vec2-seqbatch-bb1-sb16", _w2v, batched_backward=True, seq_batch_size=16)
    _extract_and_wbe("ctc-wav2vec2-split-sb16", _w2v, split_prefix_backward=True, seq_batch_size=16)

    # === AED family -- Whisper-base (encoder-decoder, autoregressive per-token score) ===
    # Compute-bound decoder backward -> token-batching is the win; the adapter is B=1 only.
    _whisper = rf.build_dict(
        Whisper,
        model_dir=DownloadHuggingFaceRepoJobV2(repo_id="openai/whisper-base", repo_type="model").out_hub_cache_dir,
    )
    _extract_and_wbe("aed-whisper-seq-bb0-sb1", _whisper, needs_transformers=True, batched_backward=False)
    _extract_and_wbe("aed-whisper-tokbatch-bb1-sb1", _whisper, needs_transformers=True, batched_backward=True)

    # === Speech-LLM family -- Phi-4-multimodal (eager attention so the batched VJP vmaps) ===
    _phi4 = rf.build_dict(
        Phi4MM,
        model_dir=download_phi4multimodal_model(),
        speech_prompt="Transcribe the audio clip into text.",
        unwrap_checkpoint_wrappers=True,
        target_start_end_to_device=True,
        char_level=True,
        char_level_sep=" ",
        attn_implementation="eager",
    )
    _extract_and_wbe(
        "llm-phi4-seq-bb0-sb1", _phi4, needs_transformers=True, mult_grad_by_inputs=True, time=8, batched_backward=False
    )
    _extract_and_wbe(
        "llm-phi4-tokbatch-bb1-sb1",
        _phi4,
        needs_transformers=True,
        mult_grad_by_inputs=True,
        time=8,
        batched_backward=True,
    )
