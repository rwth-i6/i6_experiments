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
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.extract_self_attn import (
    SelectSelfAttnAlignHeadsJob,
    ExtractSelfAttnPerTokenJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.buckeye_fine_dataset import (
    BuildBuckeyeFineDatasetJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_cost_4way_benchmark import (
    AlignCost4WayBenchmarkJob,
)
from i6_experiments.users.zeyer.external_models.voxtral import download_voxtral_mini_3b_model
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.voxtral import Voxtral
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.forced_align_baseline import (
    ForcedAlignBaselineJob,
)
from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import CalcAlignmentMetricsFromWordBoundariesJob
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.grad_align_tables import (
    build_external_tables,
    build_preview_external_tables,
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
    _extract_and_wbe("ctc-wav2vec2-split-sb1", _w2v, split_prefix_backward=True)  # B=1: compiled-scan + tokbatch

    # === AED family -- Whisper-base (encoder-decoder, autoregressive per-token score) ===
    # Compute-bound decoder backward -> token-batching is the win; the adapter is B=1 only.
    _whisper = rf.build_dict(
        Whisper,
        model_dir=DownloadHuggingFaceRepoJobV2(repo_id="openai/whisper-base", repo_type="model").out_hub_cache_dir,
    )
    # The cost table uses whisper-large-v3 (people expect the large model for a cost number);
    # the correctness-control ladder above stays on whisper-base.
    _whisper_large = rf.build_dict(
        Whisper,
        model_dir=DownloadHuggingFaceRepoJobV2(repo_id="openai/whisper-large-v3", repo_type="model").out_hub_cache_dir,
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

    # === Cost table (4-way breakdown): Forward / Prefix-fwd / Prefix-bwd / Backward, one model per
    # family, under torch 2.12 (TIMIT-test, 40 seqs). CTC (wav2vec2) drives the split_prefix_backward
    # interface so it gets all 4 stages; AED (whisper) / Sp.LLM (phi4) have no lattice -> Forward +
    # Backward only (prefix cols n/a). Transducer deferred (needs NeMo in the 2.12 env). DP excluded.
    # Emitted as the `cost` table by grad_align_tables
    # (build_external_tables + build_preview_external_tables),
    # the SAME data + preview pipeline as every other table;
    # the main 2.7 recipe builds no cost table.
    # The mgr's PYTHONPATH overlay serves transformers.
    dl_timit = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/timit", repo_type="dataset")
    cost_models = {"CTC (wav2vec2)": _w2v, "AED (whisper-large)": _whisper_large, "Sp. LLM (phi4)": _phi4}
    cost = AlignCost4WayBenchmarkJob(
        model_configs=cost_models,
        dataset_dir=dl_timit.out_hub_cache_dir,
        dataset_key="test",
        num_seqs=40,
    )
    cost.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cost.add_alias("speedcmp-p212/cost-4way-benchmark")
    tk.register_output("speedcmp-p212/cost-4way-metrics.txt", cost.out_metrics)

    # === Time-stretch table (moved from the main torch-2.7 recipe). ===
    # The slow wav2vec2-CTC ts cells need the compiled-scan split (torch-2.12 only),
    # so the WHOLE table is built here.
    # Finished 2.7 cells (ts0.5/0.75/1.0 wav2vec2, all voxtral, all MMS-FA)
    # are reconstructed with IDENTICAL hashes -> reused, not rerun (sis job state is per-work-dir).
    # Only wav2vec2 ts{1.2,1.5,2.0,3.0} are new split jobs that run under this 2.12 mgr
    # (~5x; ts2.0/3.0 now fit, were "too slow" on 2.7).
    _xa_ds = BuildBuckeyeFineDatasetJob(
        raw_dir=dl.out_hub_cache_dir,
        resegment_gap_s=1.0,
        split_up_to_max_seq_len_s=18.0,
        min_words=2,
        skip_misaligned_wavs=True,
        subsample_target_h=5.0,
        subsample_seed=42,
    )
    _xa_dir = _xa_ds.out_hub_cache_dir
    _xa_tag = "buckeye-segA-5h"
    _xa_off = 1
    _xa_ao = {"apply_softmax_over_time": True, "blank_score": -5}
    _table_results = {"speedcmp-p212/cost-4way-metrics.txt": cost.out_metrics}
    dl_voxtral = download_voxtral_mini_3b_model()

    def _ts_align(ex, name):
        # the sil2.0-wordtopo align the time-stretch table reads (production default: energy-scaled
        # silence s=2; produce the exact table name directly, skipping the generic twin generator).
        al = WordAlignFromPerTokenGradsJob(
            grad_score_hdf=ex.out_hdf,
            grad_score_key="data",
            dataset_dir=_xa_dir,
            dataset_key="test",
            dataset_offset_factors=_xa_off,
            align_opts=_xa_ao,
            audio_energy_pow=0.5,
            word_topology=True,
            blank_silence_energy_scale=2.0,
        )
        nm = f"align/{name}-asotTrue-bs-5-en0.5-sil2.0-wordtopo"
        al.add_alias(nm)
        _table_results[f"{nm}-wbe.txt"] = al.out_wbe

    dl_ds_timit = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/timit", repo_type="dataset")
    # Voxtral self-att head-selection (TIMIT val, subword) -- identical args to the forced baseline
    # in the main recipe, so the finished head-selection is reused (no re-run).
    _vx_sa_cfg = rf.build_dict(
        Voxtral, model_dir=dl_voxtral, forward_mode="transcription", attn_implementation="eager", version=3
    )
    _vx_sa_sel = SelectSelfAttnAlignHeadsJob(
        dataset_dir=dl_ds_timit.out_hub_cache_dir,
        dataset_key="val",
        model_config=_vx_sa_cfg,
        time_upsample_when_short=False,
    )
    for _tsf in [0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0]:
        _tst = f"ts{_tsf}"
        for _tsm, _tsm_sfx in [("vocoder", ""), ("resample", "-resample")]:
            # factor 1.0 == no stretch: omit the kwargs (reuse the base extract), matching the main recipe.
            if _tsf == 1.0:
                _ts_mkw = {}
                _ts_fakw = {}
            else:
                _ts_mkw = {
                    "audio_time_stretch": _tsf,
                    **({} if _tsm == "vocoder" else {"time_stretch_method": "resample"}),
                }
                _ts_fakw = {"audio_time_stretch": _tsf, "time_stretch_method": _tsm}
            # wav2vec2-CTC: the split cells (ts1.2/1.5/2.0) get split_prefix_backward (B=1; seq-batch
            # forbids time-stretch) and run here on 2.12; ts0.5/0.75/1.0 stay IDENTICAL to the main
            # recipe (non-split) -> reused. ts3.0 is dropped below (does not fit the 24h wall even split).
            _w_split = _tsf in (1.2, 1.5, 2.0, 3.0)
            _tsw_cfg = rf.build_dict(
                Wav2Vec2Ctc,
                grad_wrt="feat_proj_out",
                per_token_score="prefix_fwd",
                **_ts_mkw,
            )
            _tsw_ex = ExtractInGradsPerTokenJob(
                dataset_dir=_xa_dir,
                dataset_key="test",
                model_config=_tsw_cfg,
                mult_grad_by_inputs=False,
                attr_reduction="L2",
                batched_backward=True,
                # seq_batch_size>1 (the batched forward) asserts audio_time_stretch==1.0, so the
                # time-stretched cells stay B=1 (compiled-scan split only) -- still faster, ts2.0/3.0 fit 24h.
                **({"split_prefix_backward": True} if _w_split else {}),
            )
            _tsw_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _tsw_name = f"wav2vec2ctc-fproj_out-prefixfwd-{_xa_tag}-L2_grad-pertoken-{_tst}{_tsm_sfx}"
            _tsw_ex.add_alias(_tsw_name)
            # ts3.0 wav2vec2 grad does not fit the 24h GPU wall even with the compiled-scan split
            # (3x-stretched audio, B=1) -> dropped; the table cell shows ">24h". Skip its extract/align
            # (MMS-FA ts3.0 still runs).
            if _tsf < 3.0:
                tk.register_output(f"{_tsw_name}.hdf", _tsw_ex.out_hdf)
                _ts_align(_tsw_ex, _tsw_name)
            # Voxtral (no prefix lattice -> no split; reuse finished 2.7); capped <2.0 (30s encoder window).
            if _tsf < 2.0:
                _tsv_cfg = rf.build_dict(
                    Voxtral,
                    model_dir=dl_voxtral,
                    forward_mode="transcription",
                    grad_wrt="log_mel",
                    char_level=True,
                    char_level_sep=" ",
                    version=7,
                    **_ts_mkw,
                )
                _tsv_ex = ExtractInGradsPerTokenJob(
                    dataset_dir=_xa_dir,
                    dataset_key="test",
                    model_config=_tsv_cfg,
                    mult_grad_by_inputs=False,
                    attr_reduction="L2",
                )
                _tsv_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                _tsv_ex.rqmt = {**_tsv_ex.rqmt, "time": 24}
                _tsv_name = f"voxtral-charlevlogmel-{_xa_tag}-L2_grad-pertoken-{_tst}{_tsm_sfx}"
                _tsv_ex.add_alias(_tsv_name)
                tk.register_output(f"{_tsv_name}.hdf", _tsv_ex.out_hdf)
                _ts_align(_tsv_ex, _tsv_name)
                # Voxtral self-att (alternative aligner) on the stretched audio; heads reused from the
                # unstretched forced baseline (ts1.0 reuses the finished baseline extract).
                _tsvsa_cfg = rf.build_dict(
                    Voxtral,
                    model_dir=dl_voxtral,
                    forward_mode="transcription",
                    attn_implementation="eager",
                    version=5 if _tsf != 1.0 else 3,  # >=5 for time-stretch; v3 at ts1.0 reuses the baseline
                    **_ts_mkw,
                )
                _tsvsa_ex = ExtractSelfAttnPerTokenJob(
                    dataset_dir=_xa_dir, dataset_key="test", model_config=_tsvsa_cfg, heads=_vx_sa_sel.out_heads
                )
                _tsvsa_ex.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                _tsvsa_ex.add_alias(f"baseline-voxtral-selfattn-{_xa_tag}-{_tst}{_tsm_sfx}-extract")
                _ts_align(_tsvsa_ex, f"baseline-voxtral-selfattn-{_xa_tag}-{_tst}{_tsm_sfx}")
            # MMS-FA forced-align reference (reuse finished 2.7).
            _tsf_fa = ForcedAlignBaselineJob(dataset_dir=_xa_dir, dataset_key="test", **_ts_fakw)
            _tsf_name = f"baseline-mms_fa-{_xa_tag}-{_tst}{_tsm_sfx}"
            _tsf_fa.add_alias(_tsf_name)
            _tsf_m = CalcAlignmentMetricsFromWordBoundariesJob(
                word_boundaries_hdf=_tsf_fa.out_hdf,
                dataset_dir=_xa_dir,
                dataset_key="test",
                dataset_offset_factors=_xa_off,
            )
            _table_results[f"{_tsf_name}-wbe.txt"] = _tsf_m.out_wbe

    build_external_tables(_table_results)
    # Partial-progress preview manifests (same mechanism as the main recipe's tables):
    # the time-stretch and cost tables render with dots for any still-running cell,
    # instead of only appearing once their WriteTableDataJob's full input set has finished.
    build_preview_external_tables(_table_results)
