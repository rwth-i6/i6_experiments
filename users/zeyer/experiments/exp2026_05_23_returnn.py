"""
RETURNN experiments and benchmarks (frontend features etc.).

Currently here:

Packed (ragged) tensor storage in the RETURNN frontend
(:mod:`returnn.frontend._packed_backend`, ``PackedBackend``, ``rf.pack``):
benchmarks packed vs padded training steps (fwd + bwd),
step time and peak GPU memory,
for a Conformer encoder (default rel-pos attention + BatchNorm),
a Transformer AED (with label-wise CE loss),
and ``real``: the noTts LS baseline (Conformer L16 + Transformer decoder, aux CTC + CE),
packing the raw audio so the log-mel front-end runs packed as well.

Seq-len presets:

- ``realistic``: batch 32, 4..32 s feature frames at 100 Hz with one long outlier,
  ~68% padding when padded.
  All lens are multiples of 4 (the total subsampling factor),
  so the strided pool output layout stays expressible in the (lens, gap, align) form.
- ``no_padding``: all seqs equal (not realistic):
  the padded path has no waste to win back,
  so the remaining packed-vs-padded gap is exactly the packed overhead.

AMP (bf16 autocast + f32 weights) by default, consistent to what we use in training.
Dropout 0 everywhere:
FlexAttention has no dropout support (the packed attention path),
and compiled-NJT backward is broken in torch 2.7/2.12,
so with att_dropout the packed attention would run eager NJT (correct but slow).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
from functools import cache

from sisyphus import Job, Task, tk
from i6_core.returnn.config import ReturnnConfig

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module, disable_register_output
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as _aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs as _baseline_configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
    ctc_recog_recomb_labelwise_prior_auto_scale,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    model_recog as _ctc_model_recog,
)
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_task_raw_v2,
    get_loquacious_train_subset_dataset_v2,
)

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

# for get_setup_prefix_for_module (alias/output prefix of the loq experiments declared here)
__setup_root_prefix__ = "exp2026_05_23_returnn"


_SEQ_LENS_PRESETS = {
    "realistic": [3200, 1600, 1544, 1388, 1200, 1112, 988, 924, 876, 812, 768, 700, 644, 592, 512, 456]
    + [1500, 1400, 1300, 1248, 1148, 1048, 1000, 948, 900, 848, 800, 748, 700, 648, 600, 400],
    "no_padding": [1000] * 32,
}

# raw-audio sample counts (16 kHz) for the "real" model, which packs the raw audio and runs
# the log-mel front-end packed too. 16 seqs, 2..17.5 s.
_AUDIO_LENS_PRESETS = {
    "random": [278531, 41017, 95000, 201337, 64001, 156789, 36666, 249999]
    + [55555, 121212, 78123, 180001, 32003, 226667, 49999, 143210],
    "sorted": [278531, 271113, 265002, 258888, 254321, 249999, 245005, 241777]
    + [237500, 233333, 230001, 226667, 223456, 220000, 216789, 213001],
}


def py():
    """Sisyphus entry point."""
    for model in ["conformer", "aed"]:
        for lens_name, lens in _SEQ_LENS_PRESETS.items():
            job = PackedVsPaddedBenchmarkJob(model=model, seq_lens=lens)
            tk.register_output(f"returnn/packed-bench-{model}-{lens_name}.json", job.out_results)
    for lens_name, lens in _AUDIO_LENS_PRESETS.items():
        job = PackedVsPaddedBenchmarkJob(model="real", seq_lens=lens)
        tk.register_output(f"returnn/packed-bench-real-{lens_name}.json", job.out_results)
    py_aed_graphc()
    py_aed_graphc_loquacious()


class PackedVsPaddedBenchmarkJob(Job):
    """
    Benchmark packed vs padded train steps on GPU, see the module docstring.

    Output ``out_results`` (json): per variant (padded / packed)
    ms/step and peak GPU memory (GiB),
    plus the speedup, padding waste, and any packed fallback warnings
    (expected: none).
    """

    def __init__(
        self,
        *,
        model: str,
        seq_lens: Sequence[int],
        amp_dtype: Optional[str] = "bfloat16",
        n_warmup: int = 10,
        n_steps: int = 20,
        expected_attention_path: Optional[Union[str, Sequence[str]]] = None,
    ):
        """
        :param model: "conformer" or "aed"
        :param seq_lens: input seq lens (feature frames for the conformer,
            source tokens = frames/4 and target tokens = frames/30 for the aed,
            raw audio samples at 16 kHz for the real model)
        :param amp_dtype: autocast dtype (weights stay float32), or None for full float32
        :param n_warmup: warmup steps (incl. torch.compile of the attention kernels)
        :param n_steps: timed steps
        :param expected_attention_path: assert that the packed run used ONLY this attention impl
            (see returnn.frontend._packed_backend.attention_path_counts).
            A silent fall-through (e.g. to eager NJT) is functionally correct
            but 10-20x slower per call and invisible in the fallback warnings --
            this catches it. Default per model: "flash" (aed) / "rel_pos_triton" (conformer).
        """
        self.model = model
        self.seq_lens = list(seq_lens)
        self.amp_dtype = amp_dtype
        self.n_warmup = n_warmup
        self.n_steps = n_steps
        if expected_attention_path is None:
            # the real model runs both the encoder rel-pos (triton) and the decoder flash paths
            expected_attention_path = {
                "aed": ["flash"],
                "conformer": ["rel_pos_triton"],
                "real": ["rel_pos_triton", "flash"],
            }[model]
        if isinstance(expected_attention_path, str):
            expected_attention_path = [expected_attention_path]
        self.expected_attention_paths = set(expected_attention_path)
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2}
        self.out_results = self.output_path("results.json")

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """run"""
        import json
        import time
        import contextlib

        import torch

        from returnn.tensor import Dim
        import returnn.frontend as rf
        from returnn.frontend import _packed_backend as packed
        from returnn.util.basic import BehaviorVersion

        BehaviorVersion.set_min_behavior_version(BehaviorVersion._latest_behavior_version)
        rf.select_backend_torch()
        assert torch.cuda.is_available()
        seq_lens = self.seq_lens

        def autocast():
            if self.amp_dtype:
                return torch.autocast(device_type="cuda", dtype=getattr(torch, self.amp_dtype))
            return contextlib.nullcontext()

        with rf.set_default_device_ctx("cuda"):
            rf.set_random_seed(42)
            batch_dim = Dim(len(seq_lens), name="batch")
            if self.model == "conformer":
                step_padded, step_packed = self._make_conformer_steps(batch_dim, autocast)
            elif self.model == "aed":
                step_padded, step_packed = self._make_aed_steps(batch_dim, autocast)
            elif self.model == "real":
                step_padded, step_packed = self._make_real_steps(batch_dim, autocast)
            else:
                raise ValueError(f"unknown model {self.model!r}")

            with rf.get_run_ctx().train_flag_ctx(True):
                res: Dict[str, Any] = {
                    "model": self.model,
                    "seq_lens": seq_lens,
                    "amp_dtype": self.amp_dtype,
                    "padding_waste": 1.0 - sum(seq_lens) / (len(seq_lens) * max(seq_lens)),
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                }
                for variant, step_fn in [("padded", step_padded), ("packed", step_packed)]:
                    warned_before = set(packed._warned_fallback_ops)
                    for _ in range(self.n_warmup):
                        step_fn()
                    torch.cuda.synchronize()
                    packed.attention_path_counts.clear()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    for _ in range(self.n_steps):
                        step_fn()
                    torch.cuda.synchronize()
                    res[variant] = {
                        "ms_per_step": (time.perf_counter() - t0) / self.n_steps * 1000.0,
                        "peak_mem_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "fallback_warnings": sorted(set(packed._warned_fallback_ops) - warned_before),
                        "attention_path_counts": dict(packed.attention_path_counts),
                    }
                res["speedup"] = res["padded"]["ms_per_step"] / res["packed"]["ms_per_step"]
                # Guard against silent fall-through to a slower (but functionally correct)
                # attention impl, see expected_attention_path in __init__.
                counts = res["packed"]["attention_path_counts"]
                assert counts and set(counts) == self.expected_attention_paths, (
                    f"packed attention ran {counts}, expected only {sorted(self.expected_attention_paths)}"
                )
                # the packed path must stay on its fast ops -- no silent unpack fallbacks
                assert not res["packed"]["fallback_warnings"], (
                    f"packed run raised fallback warnings: {res['packed']['fallback_warnings']}"
                )

        with open(self.out_results.get_path(), "w") as f:
            json.dump(res, f, indent=2)
            f.write("\n")

    def _make_conformer_steps(self, batch_dim, autocast):
        import torch

        from returnn.tensor import Tensor, Dim
        import returnn.frontend as rf
        from returnn.frontend import _packed_backend as packed
        from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

        seq_lens = self.seq_lens
        feat_dim = Dim(80, name="feat")
        model = ConformerEncoder(
            feat_dim,
            Dim(512, name="model"),
            ff_dim=Dim(2048, name="ff"),
            input_layer=ConformerConvSubsample(
                feat_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2")],
                filter_sizes=[(3, 3), (3, 3)],
                pool_sizes=[(2, 1), (2, 1)],
            ),
            num_heads=8,
            num_layers=12,
            att_dropout=0.0,
        )
        params = list(model.parameters())
        time_dim = Dim(
            Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(seq_lens, dtype=torch.int32))
        )
        x = Tensor("x", dims=[batch_dim, time_dim, feat_dim], dtype="float32")
        x.raw_tensor = torch.randn(len(seq_lens), max(seq_lens), 80, generator=torch.Generator().manual_seed(1)).to(
            "cuda"
        )

        def _run(x_in):
            # frame-level squared-sum stand-in loss
            # (a real conformer setup would use CTC/transducer on top)
            with autocast():
                out, _ = model(x_in, in_spatial_dim=time_dim)
                loss = rf.reduce_sum(out * out, axis=list(out.dims))
            loss.raw_tensor.backward()
            for p in params:
                if p.raw_tensor.grad is not None:
                    p.raw_tensor.grad = None

        def step_padded():
            _run(x)

        def step_packed():
            # layout derived by hand for this model:
            # align 4 = total downsampling (two stride-2 pools);
            # gap 64 -> exactly 16 left after the two subsample stages,
            # as needed by the depthwise conv kernel 32
            _run(packed.pack(x, gap=64, align=4))

        return step_padded, step_packed

    def _make_aed_steps(self, batch_dim, autocast):
        import torch

        from returnn.tensor import Tensor, Dim
        import returnn.frontend as rf
        from returnn.frontend import _packed_backend as packed
        from returnn.frontend.encoder.transformer import TransformerEncoder
        from returnn.frontend.decoder.transformer import TransformerDecoder

        seq_lens = self.seq_lens
        dec_lens = [max(4, sl // 30) for sl in seq_lens]
        src_vocab = Dim(10_000, name="src_vocab")
        tgt_vocab = Dim(10_000, name="tgt_vocab")
        enc_model_dim = Dim(512, name="enc_model")
        enc = TransformerEncoder(src_vocab, enc_model_dim, num_layers=12, num_heads=8, dropout=0.0, att_dropout=0.0)
        dec = TransformerDecoder(
            enc_model_dim,
            tgt_vocab,
            Dim(512, name="dec_model"),
            num_layers=6,
            num_heads=8,
            dropout=0.0,
            att_dropout=0.0,
        )
        params = list(enc.parameters()) + list(dec.parameters())
        enc_time = Dim(
            Tensor(
                "enc_time",
                dims=[batch_dim],
                dtype="int32",
                raw_tensor=torch.tensor([sl // 4 for sl in seq_lens], dtype=torch.int32),
            )
        )
        dec_time = Dim(
            Tensor("dec_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(dec_lens, dtype=torch.int32))
        )
        gen = torch.Generator().manual_seed(2)
        src = Tensor("src", dims=[batch_dim, enc_time], dtype="int32", sparse_dim=src_vocab)
        src.raw_tensor = torch.randint(
            0, src_vocab.dimension, (len(seq_lens), max(seq_lens) // 4), generator=gen, dtype=torch.int32
        ).to("cuda")
        tgt = Tensor("tgt", dims=[batch_dim, dec_time], dtype="int32", sparse_dim=tgt_vocab)
        tgt.raw_tensor = torch.randint(
            0, tgt_vocab.dimension, (len(seq_lens), max(dec_lens)), generator=gen, dtype=torch.int32
        ).to("cuda")

        def _run(src_in, tgt_in):
            with autocast():
                enc_out = enc(src_in, spatial_dim=enc_time)
                enc_state = dec.transform_encoder(enc_out, axis=enc_time)
                logits, _ = dec(
                    tgt_in,
                    spatial_dim=dec_time,
                    state=dec.default_initial_state(batch_dims=[batch_dim]),
                    encoder=enc_state,
                )
                # the real loss: label-wise CE (the packed run takes the packed CE fast path)
                ce = rf.cross_entropy(estimated=logits, target=tgt_in, axis=tgt_vocab, estimated_type="logits")
                loss = rf.reduce_sum(ce, axis=list(ce.dims))
            loss.raw_tensor.backward()
            for p in params:
                if p.raw_tensor.grad is not None:
                    p.raw_tensor.grad = None

        def step_padded():
            _run(src, tgt)

        def step_packed():
            _run(packed.pack(src), packed.pack(tgt))

        return step_padded, step_packed

    def _make_real_steps(self, batch_dim, autocast):
        # the noTts LS baseline: Conformer EncL16-D1024 subsample 6 relu_square no-bias,
        # TransformerDecoder L6 D1024 RMSNorm + rotary causal + gated FF, aux CTC layer 16, log-mel front-end.
        # packing is injected on the RAW AUDIO, so the log-mel feature extraction (stft -> mel -> log)
        # runs packed too and is part of the timed step. Loss: aux CTC + label-wise CE.
        import torch

        from returnn.tensor import Tensor, Dim
        import returnn.frontend as rf
        from returnn.frontend import _packed_backend as packed
        from returnn.frontend.encoder.conformer import (
            ConformerEncoder,
            ConformerEncoderLayer,
            ConformerConvSubsample,
            ConformerPositionwiseFeedForward,
        )
        from returnn.frontend.decoder.transformer import TransformerDecoder

        # the sis worker resolves i6_experiments (the job's own package) but not sibling
        # recipe packages at runtime; the aed import below needs i6_core
        import os
        import sys

        recipe_root = os.path.abspath(__file__)
        for _ in range(5):
            recipe_root = os.path.dirname(recipe_root)
        if recipe_root not in sys.path:
            sys.path.insert(0, recipe_root)

        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import Model

        # log-mel front-end: 16 kHz, step 10 ms -> frame_step 160 samples.
        # pack the raw audio on a multiple of frame_step so the strided stft re-layouts cleanly:
        # align 960 = 6 * 160 -> feat align 6 (the /6 conv grid);
        # gap 19200 = 120 * 160 -> ~20 enc-frame gap after the subsample, headroom for the depthwise conv
        # (so the conv never needs an in-conv regap, i.e. no warning).
        frame_step = 160
        audio_align = 6 * frame_step
        audio_gap = 120 * frame_step

        audio_lens = self.seq_lens
        target_dim = Dim(10_240, name="spm10k")
        model = Model(
            target_dim=target_dim,
            blank_idx=10_240,
            eos_idx=0,
            bos_idx=1,
            enc_build_dict=rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
                att_dropout=0.1,
            ),
            dec_build_dict=rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            ),
            enc_aux_logits=[16],
        )
        params = list(model.parameters())

        audio_time = Dim(
            Tensor(
                "audio_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(audio_lens, dtype=torch.int32)
            )
        )
        audio = Tensor("audio", dims=[batch_dim, audio_time], dtype="float32")
        audio.raw_tensor = (
            torch.randn(len(audio_lens), max(audio_lens), generator=torch.Generator().manual_seed(1)) * 0.1
        ).to("cuda")
        tgt_lens = [max(6, sl // 3200) for sl in audio_lens]
        tgt_time = Dim(
            Tensor("tgt_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(tgt_lens, dtype=torch.int32))
        )
        targets = Tensor("targets", dims=[batch_dim, tgt_time], dtype="int32", sparse_dim=target_dim)
        targets.raw_tensor = torch.randint(
            2, 10_240, (len(audio_lens), max(tgt_lens)), generator=torch.Generator().manual_seed(2), dtype=torch.int32
        ).to("cuda")

        def pack_audio():
            return packed.pack(audio, gap=audio_gap, align=audio_align)

        def losses(audio_in, targets_in):
            feats_in, feat_time_in = model.feature_extraction(audio_in, in_spatial_dim=audio_time)
            enc_out, enc_spatial = model.encode_from_features(feats_in, in_spatial_dim=feat_time_in)
            enc = enc_out.enc if hasattr(enc_out, "enc") else enc_out
            aux_logits = model.enc_aux_logits_16(enc)
            log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
            ctc = rf.ctc_loss(
                logits=log_probs,
                logits_normalized=True,
                targets=targets,  # ctc targets stay plain (the loss unpacks anyway)
                input_spatial_dim=enc_spatial,
                targets_spatial_dim=tgt_time,
                blank_index=model.blank_idx,
            )
            enc_state = model.decoder.transform_encoder(enc, axis=enc_spatial)
            logits, _ = model.decoder(
                targets_in,
                spatial_dim=tgt_time,
                state=model.decoder.default_initial_state(batch_dims=[batch_dim]),
                encoder=enc_state,
            )
            ce = rf.cross_entropy(estimated=logits, target=targets_in, axis=target_dim, estimated_type="logits")
            return rf.reduce_sum(ctc, axis=list(ctc.dims)) + rf.reduce_sum(ce, axis=list(ce.dims))

        # one-time eval-mode (deterministic) parity check: padded vs packed-from-raw-audio
        with torch.no_grad():
            ref = float(losses(audio, targets).raw_tensor)
            pk = float(losses(pack_audio(), packed.pack(targets)).raw_tensor)
        assert abs(ref - pk) / max(abs(ref), 1e-6) < 1e-3, f"real-model loss parity: padded {ref} vs packed {pk}"

        def _run(audio_in, targets_in):
            with autocast():
                loss = losses(audio_in, targets_in)
            loss.raw_tensor.backward()
            for p in params:
                if p.raw_tensor.grad is not None:
                    p.raw_tensor.grad = None

        def step_padded():
            _run(audio, targets)

        def step_packed():
            _run(pack_audio(), packed.pack(targets))

        return step_padded, step_packed


# --- Graph-replay (packed + compile + capture) CTC+AED training ---------------------------------
# Reproduces the AED baseline 96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-speedpertV2-spm10k-spmSample07
# with the whole train step Inductor-compiled and CUDA-graph captured (packed bound-shape regime).
# Verified end-to-end by the smoke run 2026-07-27 (real aed_model_def + aed_training, bs15k synthetic).

# packing layout for the bound-shape regime (as in the benchmark):
# gap covers the within-batch length spread after length-sorted batching,
# align matches the 960-samples-per-frame downsampling of the encoder frontend
_aed_graphc_packed_gap = 18_240
_aed_graphc_packed_align = 960
_aed_graphc_classes_capacity = 75  # real spm10k max target len (reached)


def py_aed_graphc():
    """
    The graphc AED training.
    """
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import _batch_size_factor

    # the batch size bounds the packed content; add the per-seq gap/align slack, aligned.
    # (the default regap bound -- every seq at full audio capacity -- would nearly double
    # the packed frame count, and the encoder activations scale with it -> OOM on the 80GB c25g GPUs)
    packed_total = 200_000 * _batch_size_factor + 200 * (_aed_graphc_packed_gap + _aed_graphc_packed_align)
    packed_total = -(-packed_total // _aed_graphc_packed_align) * _aed_graphc_packed_align
    # data time capacity (samples, multiple of 960):
    # - first run: 312960, from the benchmark's synthetic distribution.
    #   TOO SMALL: LS train max ~475k samples, speed pert rate 0.7 stretches by 1/0.7 -> ~680k;
    #   from epoch 6 the curriculum admits full-length seqs,
    #   whose tails the capacity-sized masks silently ignored
    #   -> convergence gap vs the padded baseline.
    # - v2: fixed capacity + all packed fallbacks eliminated (cross-att q-packing etc);
    #   __hash_version forces the new job while the reference keeps running.
    # Earlier variants, removed from the graph
    # (job dirs + banked scores stay on disk; removal keeps the manager from resubmitting them):
    # - "" (cap 312_960, job lb68VaQEkEeC): reference; capacity too small (silent truncation),
    #   ended at ep 50 (deterministic NaN at ep 51); errored state would spam the manager
    #   and clearing it would resubmit.
    # - "-v2" (cap 720_000, __hash_version 2, job erqk3HOebeeL): cancelled 2026-07-28
    #   (devtrain ce +18% rel at ep 26);
    #   SpecAugment num-masks scaled with the capacity under static tracing -> over-masked ~3.7x.
    for name_suffix, time_cap, extra_cfg in [
        # v3 = v2 config rerun after the SpecAugment fix;
        # THE convergence validation: per-epoch scores must match the padded baseline up to noise
        ("-v3", 720_000, {"__hash_version": 3}),
    ]:
        _py_aed_graphc_exp(name_suffix, time_cap, packed_total, extra_cfg)
    # same as v3 but with the decoder targets also packed, as a comparison
    pdec_exp = _py_aed_graphc_exp("-v3-pdec", 720_000, packed_total, {"__hash_version": 3}, packed_decoder=True)
    # ep-2 NaN bisect ladder (stages 1-22, 2026-07-29/30) -- RESOLVED, stage jobs retired:
    # eager clean -> compiled-no-capture same NaN (capture exonerated)
    # -> failing batch finite standalone (state-dependent, step nondeterministic)
    # -> nanassert false-positive on masked-lane NaN
    # -> zero-init no change
    # -> copy-in race fixes kept as hardening
    # -> nanreport + operand dumps:
    # flash-varlen BACKWARD NaN from empty-KV filler segments (lse=-inf)
    # and causal cu slack (cu total = the BOUND under static tracing).
    # Fix in _sdpa_varlen_attention:
    # real-total cu, trailing-filler clamp, unified q/k/v slack pre-mask, out-tail zeroing.
    # eager-bound nantrace v7 and compiled nanreport: CLEAN through all 2015 ep-2 steps.
    # Full record: projects/2026-05-23-returnn-paper.md.
    # The stages loaded the ep-1 checkpoint, pruned meanwhile, so they could not rerun anyway.
    pdec_train_job = pdec_exp.get_training_job()
    # fix validation in the production path (Inductor-compiled), from a kept checkpoint
    # (the bench links the checkpoint as its epoch 1, so this still trains "ep 2")
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[10].path,
        config_overrides={"num_epochs": 2},
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-compiled-fix-validation.json", job.out_results)

    # LS counterpart of the loq gap sweep.
    # On loq, audio gap 0 / text gap 0 tied the best step time
    # and had the lowest peak memory of every packed variant.
    # LS differs in ways that could change that:
    # shorter utterances, speed perturbation, a different within-batch length spread,
    # so measure it rather than extrapolate.
    # The 18_240 / 2 point is the current default, carried along as the control.
    for audio_gap, text_gap in [(0, 0), (0, 2), (_aed_graphc_packed_gap, 2)]:
        audio_bound = 200_000 * _batch_size_factor + 200 * (audio_gap + _aed_graphc_packed_align)
        audio_bound = -(-audio_bound // _aed_graphc_packed_align) * _aed_graphc_packed_align
        job = TrainStepBenchmarkJob(
            returnn_config=pdec_train_job.returnn_config,
            mode="packed_graphc",
            num_steps=31,
            config_overrides={
                "packed_tensors": {
                    "per_key": {
                        "data": {"gap": audio_gap, "align": _aed_graphc_packed_align},
                        "classes": {"gap": text_gap, "align": 1},
                    }
                },
                "torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"data": 720_000, "classes": _aed_graphc_classes_capacity},
                    "packed_total_bound": {
                        "data": audio_bound,
                        "classes": 200 * (_aed_graphc_classes_capacity + text_gap),
                    },
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        )
        tk.register_output(f"returnn/aed-graphc-bench-gaps-a{audio_gap}-t{text_gap}.json", job.out_results)
    # padded reference at the same RETURNN version and step count.
    # The LS packed-vs-padded ratio we quote is from 2026-07-28,
    # from before the packed rewrite and both CTC commits, so it says nothing about current code.
    job = TrainStepBenchmarkJob(returnn_config=pdec_train_job.returnn_config, mode="padded_eager", num_steps=31)
    tk.register_output("returnn/aed-graphc-bench-gaps-padded_eager.json", job.out_results)


def _py_aed_graphc_exp(name_suffix, time_cap, packed_total, extra_cfg, *, packed_decoder=False):
    """
    one graphc AED training variant, see :func:`py_aed_graphc`

    :param name_suffix: appended to the experiment name
    :param time_cap: dim capacity of the audio (data) time dim, in samples
    :param packed_total: packed_total_bound of the audio
    :param extra_cfg: extra config_updates
    :param packed_decoder: pack the targets (classes) too,
        with per-seq gap 2 for the BOS/EOS shifts
    """
    from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
        config_96gb_bf16_accgrad1,
        _get_cfg_lrlin_oclr_by_bs_nep_v3,
        _batch_size_factor,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import train_exp as aed_train_exp

    if packed_decoder:
        # gap 2 = per-seq slack for the BOS/EOS shifts (packed left/right pad)
        classes_packed_opts = {"gap": 2, "align": 1}
        packed_total_bound = {"data": packed_total, "classes": 200 * (_aed_graphc_classes_capacity + 2)}
    else:
        # targets stay padded:
        # v3 was launched before the packed decoder existed;
        # any change here would rehash and restart the running convergence validation
        classes_packed_opts = {"packed": False}
        packed_total_bound = {"data": packed_total}

    return aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-speedpertV2-spm10k-spmSample07-graphc{name_suffix}",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            **extra_cfg,
            "optimizer.capturable": True,
            # audio packed in the collate (imported + regapped on device by the graph capture)
            "packed_tensors": {
                "per_key": {
                    "data": {"gap": _aed_graphc_packed_gap, "align": _aed_graphc_packed_align},
                    "classes": classes_packed_opts,
                }
            },
            "torch_cuda_graph": {
                "batch_size_bound": 200,  # == max_seqs
                "dim_capacity": {"data": time_cap, "classes": _aed_graphc_classes_capacity},
                "packed_total_bound": packed_total_bound,
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
            },
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        # bhv 21 recog leaves conv/pool unmasked + length-dependent same-padding:
        # padded batches corrupt the frontend features of padded seqs (EOS runaway),
        # catastrophic for packed-trained models (they never saw padding artifacts in training).
        # Validated: v3 ep-100 dev-other batched 20.08 -> 5.39 with bhv 28 (== bs1 quality).
        search_config={"behavior_version": 28},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1},
    )


# ----------------------------------------------------------------------------
# Below: ``_base_config`` and ``loq_train`` are a verbatim copy of
# ``exp2026_05_26_base_fzj.train`` (itself copied verbatim from
# ``exp2025_10_21_chunked_ctc``); ``get_lm`` is imported from the original.
# **Do not refactor** without re-verifying that all downstream Job hashes
# still match (``hpc-sis-m.py --inspect`` must show no changed jobs).
# ----------------------------------------------------------------------------


_base_config = {
    # ("large", 100),  # 100kh in total, 4 full epochs
    # ("large", 150),  # 150kh in total, 6 full epochs
    # ("large", 200),  # 200kh in total, 8 full epochs
    # ("large", 250),  # 250kh in total, 10 full epochs
    # ("large", 500),  # 500kh in total, 20 full epochs
    "subset": "large",
    "total_k_hours": 100,
    "vocab": "spm10k",
    "model": {
        "behavior_version": 24,
        "__serialization_version": 2,
        "enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
            ),
            num_layers=16,
            out_dim=1024,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
        ),
        # Default AED decoder size: 6 layers, 512 dim
        "dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=6,
            model_dim=1024,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            # When only trained on LS ASR data, keep the default dropout?
            # dropout=0.0,
            # att_dropout=0.0,
        ),
        "feature_batch_norm": True,
    },
    "train_update_func_from_n_ep": lambda n_ep: {
        "train": _baseline_configs._get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5)
    },
    "train": dict_update_deep(
        _baseline_configs.config_96gb_bf16_accgrad1,
        {
            "batch_size": 100_000 * _baseline_configs._batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
    ),
    "train_post": dict_update_deep(
        _baseline_configs.post_config, {"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}}
    ),
    # TODO (for later): Bug: train_vocab_opts/dataset_train_opts are not actually plumbed through aed_train_exp.
    #   Not a quick fix: patching now would silently change every training run and break
    #   comparability with all existing results.
    #   Revisit when starting a fresh batch of experiments where breakage is acceptable.
    "train_vocab_opts": {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    "dataset_train_opts": {"train_epoch_split": 1, "train_epoch_wise_filter": None},
    "env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    "lm_recog_extra": {},
}


def small_model_overrides() -> Dict[str, Any]:
    """
    :return: the config overrides of the SMALL ladder point (~22M: 4-layer/256 conformer encoder,
        2-layer/256 transformer decoder), as ``base-small-v2`` and its backend variants use it.

    Module level so a companion recipe (run by its own sis manager, e.g. the JAX one on the
    torch-2.12 env) uses the very same dict: the backend comparison only means something if the
    model config is identical, and a copy would drift.
    """
    import returnn.frontend as rf
    from returnn.frontend.encoder.conformer import (
        ConformerConvSubsample,
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerPositionwiseFeedForward,
    )
    from returnn.frontend.decoder.transformer import FeedForwardGated, TransformerDecoder

    return {
        "model.enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            num_layers=4,
            out_dim=256,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=4,
            ),
        ),
        "model.dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=2,
            model_dim=256,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
        ),
        "train.aux_loss_layers": [2, 4],
        "train.dec_aux_loss_layers": [1],
    }


def loq_train(
    name: str,
    config: Dict[str, Any],
    config_overrides: Optional[Dict[str, Any]] = None,
    *,
    config_deletes: Optional[Sequence[str]] = None,
    recog_def_ctc_only: bool = True,
    prefix: Optional[str] = None,
):
    """Loquacious AED+CTC training + recog pipeline (see the provenance note above)."""
    if prefix is None:
        prefix = get_setup_prefix_for_module(__name__)

    config = dict_update_deep(_base_config.copy(), config.copy())
    # deletes run before the updates, so a variant can drop an option that does not apply to it
    # (e.g. the torch_* options for a TF run) instead of having to set it to a no-op value
    config = dict_update_deep(config, config_overrides, config_deletes, dict_value_merge=False)

    train_epoch_split_per_subset = {"clean": 13, "small": 1, "medium": 2, "large": 25}
    hours_per_subset = {"clean": 13_000, "small": 250, "medium": 2_500, "large": 25_000}
    subset = config.pop("subset")
    total_k_hours = config.pop("total_k_hours")
    train_epoch_split = train_epoch_split_per_subset[subset]
    num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
    n_ep = round(num_full_ep * train_epoch_split)

    train_update_func_from_n_ep = config.pop("train_update_func_from_n_ep")
    if train_update_func_from_n_ep:
        config = dict_update_deep(config, train_update_func_from_n_ep(n_ep))

    model_config = config.pop("model")
    train_config: Dict[str, Any] = config.pop("train")
    post_config = config.pop("train_post")

    vocab = config.pop("vocab", "spm10k")
    # only passed through when set: the default (laplace:.1000) call stays byte-identical,
    # so existing job hashes are unaffected
    train_seq_ordering = config.pop("train_seq_ordering", None)
    task_extra_kwargs = {}
    if train_seq_ordering is not None:
        task_extra_kwargs["train_seq_ordering"] = train_seq_ordering
    # Same only-when-set passthrough as train_seq_ordering above:
    # unset, the task call stays byte-identical, so existing job hashes are unaffected.
    # Context: the shared train_post carries "__multi_proc_dataset_opts": {"num_workers": 25},
    # but that is the v3 key; train_v4 reads "__multi_proc_dataset" (see the WARNING there),
    # so it never applied, and every run here used the task default of 2 workers --
    # which is why both backends sit at a ~0.25 s/batch loader ceiling on base-small.
    multi_proc = config.pop("multi_proc", None)
    if multi_proc is not None:
        task_extra_kwargs["multi_proc"] = multi_proc
    task = get_loquacious_task_raw_v2(
        vocab=vocab, subset_name=subset, train_epoch_split=train_epoch_split, **task_extra_kwargs
    )

    train_vocab_opts = config.pop("train_vocab_opts")
    dataset_train_opts = config.pop("dataset_train_opts")
    env_updates = config.pop("env_updates")
    lm_recog_extra_config = config.pop("lm_recog_extra")

    assert not config

    aux_ctc_layer = max(
        [i for i in train_config["aux_loss_layers"] if i <= model_config["enc_build_dict"]["num_layers"]]
    )

    exp = _aed_train_exp(
        name,
        train_config,
        prefix=prefix + "/aed/",
        task=task,
        model_config=model_config,
        post_config_updates=post_config,
        vocab=vocab,
        train_vocab_opts=train_vocab_opts,
        dataset_train_opts=dataset_train_opts,
        env_updates=env_updates,
        recog_def=_ctc_model_recog if recog_def_ctc_only else None,
        search_config=dict_update_deep(
            {"aux_loss_layers": [aux_ctc_layer]},
            # TF-engine recogs: the beam-search recog_def is a BUILD-TIME (unrolled) loop,
            # so every dim needs a static bound (tf_static_shapes; TF TensorArray ops).
            # Bounds are generous; the engine asserts WITH the observed sizes when a batch
            # exceeds them -- tighten/raise from those messages.
            {
                "tf_static_shapes": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 576_000, "text": 1024},
                }
            }
            if train_config.get("backend") == "tensorflow"
            else None,
        )
        if recog_def_ctc_only
        else None,
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=aux_ctc_layer,
    )
    if vocab == "spm10k":
        lm_name, lm = get_lm(prefix=prefix, vocab=vocab)
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/aed/{name}/ctc+lm-v2/{lm_name}",
            task=task,
            ctc_model=exp.get_last_fixed_epoch(),
            extra_config={
                "aux_loss_layers": [
                    max(
                        [
                            i
                            for i in train_config["aux_loss_layers"]
                            if i <= model_config["enc_build_dict"]["num_layers"]
                        ]
                    )
                ],
                **lm_recog_extra_config,
            },
            lm=lm,
            prior_dataset=get_loquacious_train_subset_dataset_v2(vocab="spm10k"),
        )

    return exp, task, aux_ctc_layer


@cache
def get_lm(*, prefix: str, vocab: str, num_full_ep: int = 5, split: int = 10) -> Tuple[str, ModelWithCheckpoint]:
    """Loquacious Transformer LM (verbatim copy, see the provenance note above)."""
    from sisyphus import tk
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
        config_96gb_bf16_accgrad1,
        _get_cfg_lrlin_oclr_by_bs_nep_v4,
    )
    from i6_experiments.users.zeyer.decoding.perplexity import (
        get_lm_perplexities_for_task_evals,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import lm_model_def, lm_train_def
    from i6_experiments.users.zeyer.train_v4 import train as _train, ModelDefWithCfg

    from i6_experiments.users.zeyer.datasets.loquacious import (
        get_loquacious_text_only_dataset,
    )

    import returnn.frontend as rf
    from returnn.frontend.decoder.transformer import TransformerDecoder

    n_ep = round(num_full_ep * split)
    # orig name: trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop01-b400_20k-nEp...-spm10k
    name = f"trafo-n32-d1024-nFullEp{num_full_ep}-nEp{n_ep}-{vocab}"
    exp = _train(
        f"{prefix}/lm/{name}",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep),
                "batch_size": 20_000,
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        train_dataset=get_loquacious_text_only_dataset(vocab="spm10k", train_epoch_split=split),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=32,
                    model_dim=1024,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    dropout=0.1,
                    att_dropout=0.1,
                )
            },
        ),
        train_def=lm_train_def,
    )

    task = get_loquacious_task_raw_v2(vocab=vocab)
    perplexities_nlm = get_lm_perplexities_for_task_evals(task, label_level="task", lm=exp.get_last_fixed_epoch())
    for eval_set_name, ppl in perplexities_nlm.items():
        tk.register_output(f"{prefix}/lm/{name}/ppl/{eval_set_name}", ppl)

    return name, exp.get_last_fixed_epoch()


def py_aed_graphc_loquacious():
    """
    Loquacious ~500M CTC+AED base (16L Conformer 1024d + 6L Transformer dec 1024d, spm10k),
    trained with graphc (packed collate + Inductor compile + whole-step CUDA graph),
    via the local :func:`loq_train` pipeline copy above.

    Capacity notes (vs the LS 160M runs):
    - data: max_seq_length_default_input = 19.5s = 312_000 samples, NO speed perturbation
      -> dim_capacity 312_960 (multiple of 960) is provably sufficient.
    - text: no target-len filter in this config,
      so the capacity must cover the measured spm10k target-len max of the large subset
      (a too-small value raises loudly in graph-capture _copy_in).
    """
    gap, align = _aed_graphc_packed_gap, _aed_graphc_packed_align
    # measured on the train shards (spm10k SZcvHsG1gYNM),
    # CONDITIONED on the 19.5s audio filter (8.96M of 9.49M seqs pass):
    # max 246, p99.99 = 86, p99 = 66
    # (the unconditional max 366 has long audio and gets filtered out);
    # 256 = conditional max + headroom.
    classes_cap = 256
    packed_total = 100_000 * _loq_batch_size_factor() + 200 * (gap + align)
    packed_total = -(-packed_total // align) * align
    # get_lm is @cache'd: warm it OUTSIDE the disabled-registration blocks below,
    # so its ppl outputs stay registered for all real experiments.
    get_lm(prefix=get_setup_prefix_for_module(__name__), vocab="spm10k")
    # CONFIG ANCHOR ONLY (benches read the ReturnnConfig OBJECT, no job output):
    # with output registration disabled, neither this collapsed v1 training (superseded by
    # -fixdelta) nor its recogs are scheduled -- nothing to hold, job dirs stay on disk.
    with disable_register_output():
        exp, _, _ = loq_train(
            "base-graphc",
            {},
            config_overrides={
                "train.optimizer.capturable": True,
                # NOTE: the Loquacious task uses extern-data keys "audio"/"text"
                # (not "data"/"classes" like the LS baselines).
                # Fully packed, INCLUDING the decoder targets;
                # text gap 2 = per-seq slack for the BOS/EOS shifts (packed left/right pad)
                "train.packed_tensors": {
                    "per_key": {
                        "audio": {"gap": gap, "align": align},
                        "text": {"gap": 2, "align": 1},
                    }
                },
                "train.torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": packed_total, "text": 200 * (classes_cap + 2)},
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        )
    # smoke/benchmark on the REAL experiment config (the ReturnnConfig object, proper hashing):
    # 31 graphc steps = memory fit + NaN-fix verification before/alongside the training,
    # padded_eager = the equal-hardware comparison cell
    cfg = exp.get_training_job().returnn_config
    for mode in ["packed_graphc", "padded_eager"]:
        job = TrainStepBenchmarkJob(returnn_config=cfg, mode=mode, num_steps=31)
        tk.register_output(f"returnn/loq-base-graphc-bench-{mode}.json", job.out_results)
    # (the gap-960 debug probe at the loose text bound is superseded:
    #  the declared-bound run measures 0.499 s/step at gap 960, see below)
    _loq_cost_decomposition(cfg, classes_cap)
    _loq_text_seq_len_stats()

    # v2 of the graphc training, replacing "base-graphc" (which collapsed at ep 7, 100% WER).
    # Two changes, both from the 2026-08-03/04 investigation:
    # - no input-side gaps: measured best-or-tied on both loq and LS, for step time and peak memory,
    #   and with masked conv-norm statistics the gap no longer perturbs them either.
    # - behavior_version 29 (up from 24): the conv-block BatchNorm now masks its statistics,
    #   which otherwise run over the raw storage and count the packing gap frames.
    #   This also brings 25 (scatter masking), 26 (DistributeFilesDataset sharding, a no-op unsharded),
    #   27 (RF module output keeps the input dtype under autocast) and 28 (per-seq specaugment masks).
    #   For a fresh training the latest semantics are what we want; nothing here needs the old ones.
    # STATUS: collapsed at ep 7 as well; CONFIG ANCHOR ONLY (see base-graphc above):
    # not scheduled, no hold needed; cfg_v2 anchors the benches and probe configs below.
    nogap_total = 100_000 * _loq_batch_size_factor() + 200 * align
    nogap_total = -(-nogap_total // align) * align
    with disable_register_output():
        exp_v2, _, _ = loq_train(
            "base-graphc-v2",
            {},
            config_overrides={
                "train.optimizer.capturable": True,
                "model.behavior_version": 29,
                "train.packed_tensors": {
                    "per_key": {
                        "audio": {"gap": 0, "align": align},
                        "text": {"gap": 0, "align": 1},
                    }
                },
                "train.torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": nogap_total, "text": 18_000},
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        )
    # 31-step fit/parity check on the real config, before the training burns GPU hours
    job = TrainStepBenchmarkJob(
        returnn_config=exp_v2.get_training_job().returnn_config, mode="packed_graphc", num_steps=31
    )
    tk.register_output("returnn/loq-base-graphc-v2-smoke.json", job.out_results)

    # The padded baseline v2 must run at the SAME behavior version as base-graphc-v2,
    # otherwise a packed-vs-padded comparison also carries behavior 24 -> 29:
    # 29 masks the conv-block BatchNorm statistics (padded counted its padding frames until now),
    # 27 keeps the module output dtype under autocast, 28 draws specaugment masks per seq.
    # Everything else stays at the loq base defaults, so this is padded, eager, no graph capture.
    loq_train(
        "base-v2",
        {},
        config_overrides={"model.behavior_version": 29, "train._hash_only_returnn_2026_08_06": True},
    )

    # Packed-batch-size benchmark, all at behavior version 29 on the v2 config.
    # Every earlier number used the padded-derived batch size, so the memory packing frees
    # just sat idle. Three things to separate:
    # - padded @ 100k: the baseline
    # - packed @ 100k: the implementation win at an identical batch
    # - packed @ >100k: what packing actually buys, by spending the freed memory
    # Compare THROUGHPUT (frames/sec), not sec/step: a larger batch makes each step slower
    # while doing more work. Note max_seqs stays 200, so if the seq count binds before the
    # frame count, the larger batch sizes will not actually grow -- visible as a flat throughput.
    cfg_v2 = exp_v2.get_training_job().returnn_config
    # (REMOVED the ("packed_graphc", 200) point, loq-v2-bs200k-packed_graphc:
    #  OOM'd right after warmup -- audio bound 32.2M plus the outdated 200*classes_cap text bound;
    #  the tuned-bound partitioned benches answer the bs200k-fit question instead)
    for mode, bs_k in [
        ("padded_eager", 100),
        ("packed_graphc", 100),
        ("packed_graphc", 125),
        ("packed_graphc", 150),
    ]:
        bs = bs_k * 1000 * _loq_batch_size_factor()
        audio_bound = -(-(bs + 200 * align) // align) * align
        job = TrainStepBenchmarkJob(
            returnn_config=cfg_v2,
            mode=mode,
            num_steps=31,
            config_overrides={
                "batch_size": bs,
                # only the graphc points get the capture config.
                # Forcing it on padded_eager would compile the padded step, and there the CTC loss
                # is aten._ctc_loss, which is untraceable under fake tensors
                # (DynamicOutputShapeException) -- that untraceability is why the packed native op exists.
                **(
                    {
                        "torch_cuda_graph": {
                            "batch_size_bound": 200,
                            "dim_capacity": {"audio": 312_960, "text": classes_cap},
                            "packed_total_bound": {"audio": audio_bound, "text": 200 * classes_cap},
                            "warmup_steps": 2,
                            "capture_optimizer": True,
                            "compile": True,
                        }
                    }
                    if mode == "packed_graphc"
                    else {}
                ),
            },
        )
        tk.register_output(f"returnn/loq-v2-bs{bs_k}k-{mode}.json", job.out_results)

    # bare `packed_tensors = True`: gap 0, align 1, no manual layout tuning at all.
    # The frontend stride then does not divide the align, so the conv auto-realigns
    # (align 1 -> 960) on every step. This measures what that costs against the tuned layout,
    # i.e. whether specifying gap/align is worth it or the default is good enough.
    job = TrainStepBenchmarkJob(
        returnn_config=cfg_v2,
        mode="packed_graphc",
        num_steps=31,
        config_overrides={
            "packed_tensors": True,
            "torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                # same bounds as the tuned run, so this compares layout, not buffer size
                "packed_total_bound": {"audio": nogap_total, "text": 18_000},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
            },
        },
    )
    tk.register_output("returnn/loq-v2-packed_tensors-default.json", job.out_results)

    # The shared v2 packed-graphc config, base of every packed collapse probe below.
    # FROZEN: this is the hashed config base of the RUNNING v2/fixdelta lineage --
    # any change here rehashes (= restarts) those trainings. New experiments use
    # _loq_v3_overrides below instead (packed_tensors True, see the pbs block).
    _loq_v2_packed_overrides = {
        # sis-hash marker (user-approved restart 2026-08-06): restart the whole lineage on the
        # frozen RETURNN (RETURNN_CUDA serial-normalize fix + packed FSA + delta fix all in;
        # the code lives in tools/returnn, invisible to hashing -- hence the marker)
        "train._hash_only_returnn_2026_08_06": True,
        "train.optimizer.capturable": True,
        "model.behavior_version": 29,
        "train.packed_tensors": {
            "per_key": {
                "audio": {"gap": 0, "align": align},
                "text": {"gap": 0, "align": 1},
            }
        },
        "train.torch_cuda_graph": {
            "batch_size_bound": 200,
            "dim_capacity": {"audio": 312_960, "text": classes_cap},
            "packed_total_bound": {"audio": nogap_total, "text": 18_000},
            "warmup_steps": 2,
            "capture_optimizer": True,
            "compile": True,
        },
    }
    # REMOVED base-graphc-v2-s1337 (held): v2 with random_seed 1337; collapsed at ep 7 like v2
    # -> the collapse is systematic, not seed luck;
    # its ep-5..7 checkpoints + opt states are snapshotted in ~/tmp/s1337-snapshots.

    # Scale ladder (small 22M / medium 84M / medium1024 160M), from scratch, packed graphc:
    # located the collapse threshold and provided the fast repro vehicle.
    import returnn.frontend as rf
    from returnn.frontend.encoder.conformer import (
        ConformerConvSubsample,
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerPositionwiseFeedForward,
    )
    from returnn.frontend.decoder.transformer import FeedForwardGated, TransformerDecoder

    small_overrides = small_model_overrides()
    # REMOVED base-graphc-v2-small (pre-fix Triton kernel, held): looked clean for tens of epochs,
    # then degenerate ce~4.5 plateau (visible by ep 50) -> even 64-dim heads at width 256 die; -> -small-fixdelta.
    # the padded-eager counterpart: the convergence control that makes a small-model collapse
    # interpretable (same role as base-v2 for the full size)
    loq_train("base-small-v2", {}, config_overrides={"model.behavior_version": 29, **small_overrides})

    # The same model, data and LR schedule on RETURNN's pure-TF RF backend
    # (backend = "tensorflow" -> returnn/tf/engine_rf.py), as the backend comparison:
    # only the backend differs, so the training curves are directly comparable.
    # The torch_* options have TF counterparts (tf_amp) or no equivalent
    # (torch_dataloader_opts: this engine batches in the main process),
    # and the engine rejects any it would otherwise ignore silently.
    loq_train(
        "base-small-v2-tf",
        {},
        config_overrides={
            "model.behavior_version": 29,
            **small_overrides,
            "train.backend": "tensorflow",
            "train.tf_amp": "bfloat16",
        },
        # only the hashed train config needs an explicit delete here; the torch_* entries in the
        # post config are dropped by train() itself, by backend
        config_deletes=["train.torch_amp"],
    )

    # Same again, but with CTC on RETURNN's native op,
    # the TFBackend.ctc_loss default since the TF 2.20 port of native_op.cpp
    # (https://github.com/rwth-i6/returnn/issues/1833, https://github.com/rwth-i6/returnn/issues/1834).
    # The run above had CTC on TF's CPU-only op, the dominant part of its ~2.1x wall clock vs torch.
    # _hash_only marker: the switch lives in tools/returnn, invisible to sis hashing.
    loq_train(
        "base-small-v2-tf-nativectc",
        {},
        config_overrides={
            "model.behavior_version": 29,
            **small_overrides,
            "train.backend": "tensorflow",
            "train.tf_amp": "bfloat16",
            "train._hash_only_tf_native_ctc": True,
        },
        config_deletes=["train.torch_amp"],
    )

    # DRAFT, DISABLED (uncomment to schedule -- launch is gated on AZ):
    # the packed + static-shapes + XLA TF run, the TF analogue of base-graphc-v2-pbs-randshuf-fixdelta.
    # torch_cuda_graph maps to tf_jit (the XLA compile) + tf_static_shapes (the shape bounds);
    # warmup/capture/compile knobs have no TF meaning (a TF1 graph is built once, static by
    # construction). Same content budgets/bounds as the pbs line (they are data properties,
    # not model properties). Open design decision before launching: which small-model PT arm
    # (eager padded vs graphc packed) is the comparison twin, and whether this runs at small
    # or full model size.
    # loq_train(
    #     "base-small-v2-tf-packed-jit",
    #     {},
    #     config_overrides={
    #         "model.behavior_version": 29,
    #         **small_overrides,
    #         "train.backend": "tensorflow",
    #         "train.tf_amp": "bfloat16",
    #         "train._hash_only_tf_native_ctc": True,
    #         "train_seq_ordering": "random",
    #         "train.packed_tensors": True,
    #         "train.batch_size": None,
    #         "train.packed_batch_size": {"audio": 16_192_320, "text": 4_000},
    #         "train.tf_jit": True,
    #         "train.tf_static_shapes": {
    #             "batch_size_bound": 200,
    #             "dim_capacity": {"audio": 312_960, "text": classes_cap},
    #             "packed_total_bound": {"audio": 16_192_320, "text": 4_000},
    #         },
    #     },
    #     config_deletes=["train.torch_amp"],
    # )

    # Medium ladder point (~84M, enc:dec ratio like the full model; compute-bound, unlike small):
    medium_overrides = {
        "model.enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            num_layers=10,
            out_dim=512,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
        ),
        "model.dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=3,
            model_dim=512,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
        ),
        "train.aux_loss_layers": [3, 6, 10],
        "train.dec_aux_loss_layers": [1],
    }
    # REMOVED base-graphc-v2-medium (pre-fix Triton kernel, held): passed ep 7 clean (which pointed
    # the bisect at width/head-dim) but later degenerated (ce~4.4 flat by ep 42) -> -medium-fixdelta.

    # Width probe at the FULL model dim (~160M): 1024 wide but shallow, incl. the full model's
    # 128-dim heads (small/medium have 64); medium (10L/512) passed ep 7 clean,
    # so this separates width/head-dim from depth/param-count as the scale driver.
    medium1024_overrides = {
        "model.enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            num_layers=4,
            out_dim=1024,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
        ),
        "model.dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=2,
            model_dim=1024,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
        ),
        "train.aux_loss_layers": [2, 4],
        "train.dec_aux_loss_layers": [1],
    }
    # REMOVED base-graphc-v2-medium1024 (pre-fix, held): COLLAPSED ep 7-8 like the full size
    # -> width/head-dim drives the onset, not depth/params; served as the fast repro vehicle (~13.5 min/ep).
    # One-kernel bisects on that vehicle (rf_packed_att_fast_paths list = ops KEEPING their fast
    # paths, the disabled op takes the exact padded unpack fallback), all removed:
    # REMOVED -encpad (Triton -> padded, decoder flash kept): SURVIVED clean (ce 0.72@ep45, stopped)
    # -> the Triton rel-pos kernel implicated.
    # REMOVED -decpad (flash -> padded, Triton kept): COLLAPSED ep 8 (gnorm 192) -> flash exonerated.
    # REMOVED -h16 (16x64-dim heads at width 1024): LATE-collapsed ep 11-12
    # -> smaller head dim only delays the runaway.
    # Root cause = the bwd delta = out.do shortcut in the Triton kernel (bf16-out vs f32-dp bias),
    # fixed by in-kernel f32 delta recompute; -fixdelta below validated it (tracks -encpad exactly).

    # Validation of the Triton rel-pos bwd delta fix (2026-08-06): the pre-fix bwd took
    # delta = out . do from the bf16-stored out while dp stays f32, breaking the sharp-row
    # cancellation in ds = p * (dp - delta); measured as a systematic pos_bias_v grad bias
    # (2.5x MC floor, both head dims) with superlinear growth in attention sharpness.
    # The fix recomputes delta = sum_j p*dp in f32 in-kernel; bias gone to floor.
    # The marker key only distinguishes the sis hash (the fix lives in tools/returnn);
    # this run must pass ep 8-9 where the unfixed medium1024 collapsed.
    loq_train(
        "base-graphc-v2-medium1024-fixdelta",
        {},
        config_overrides={
            **_loq_v2_packed_overrides,
            **medium1024_overrides,
            "train._rel_pos_att_bwd_delta_recompute": True,
        },
    )
    # the padded-eager counterpart: the convergence control that makes the medium1024 collapse
    # interpretable (same role as base-small-v2 for the small size)
    loq_train(
        "base-medium1024-v2",
        {},
        config_overrides={
            "model.behavior_version": 29,
            "train._hash_only_returnn_2026_08_06": True,
            **medium1024_overrides,
        },
    )

    # Production relaunches on the FIXED Triton kernel (delta recompute, see -fixdelta above):
    # the full-size flagship + the two scale-ladder points whose pre-fix runs late-collapsed.
    loq_train(
        "base-graphc-v2-fixdelta",
        {},
        config_overrides={**_loq_v2_packed_overrides, "train._rel_pos_att_bwd_delta_recompute": True},
    )
    loq_train(
        "base-graphc-v2-small-fixdelta",
        {},
        config_overrides={
            **_loq_v2_packed_overrides,
            **small_overrides,
            "train._rel_pos_att_bwd_delta_recompute": True,
        },
    )
    loq_train(
        "base-graphc-v2-medium-fixdelta",
        {},
        config_overrides={
            **_loq_v2_packed_overrides,
            **medium_overrides,
            "train._rel_pos_att_bwd_delta_recompute": True,
        },
    )

    # packed_batch_size regime: batch forming on PACKED sums (batch_size None, else the padded
    # accounting always binds first and the packed budget is a no-op). Budget == buffer bound
    # by construction, for BOTH keys. Audio budget = the previous audio bound (same buffers,
    # now FILLED with content: the ~10% padding slack of padded accounting becomes content).
    # Text 4_000 from the laplace batch simulation on the full train set (durations x text lens,
    # 3 epochs, 239_714 batches at this audio budget): text sums mean 2_910 / p99.9 3_303 /
    # max 3_429 -> 4_000 = max +17%, never closes a batch early; the old 18_000 was 5x oversized
    # (it had to worst-case because nothing GUARANTEED the sum; the budget now does).
    # packed_tensors True (auto layout, conv realigns on the fly): measured FASTER than the
    # tuned per_key layout (0.484 vs 0.499-0.545 s/step, loq-v2-packed_tensors-default bench).
    # The clean base for all NEW experiments (pbs / randshuf / bs200k line); the frozen
    # _loq_v2_packed_overrides above stays only for the running v2/fixdelta lineage.
    _loq_v3_overrides = {
        # same 2026-08-06 frozen-RETURNN restart marker as _loq_v2_packed_overrides above
        "train._hash_only_returnn_2026_08_06": True,
        "train.optimizer.capturable": True,
        "model.behavior_version": 29,
        "train.packed_tensors": True,
        # NOT a real RETURNN option -- a pure sis-hash marker (the delta fix lives in
        # tools/returnn, invisible to hashing). FROZEN: renaming it rehashes every -fixdelta
        # job. Future markers: use a "_hash_only_" prefix so this is obvious from the name.
        "train._rel_pos_att_bwd_delta_recompute": True,
    }
    loq_train(
        "base-graphc-v2-pbs-fixdelta",
        {},
        config_overrides={
            **_loq_v3_overrides,
            "train.batch_size": None,
            "train.packed_batch_size": {"audio": 16_192_320, "text": 4_000},
            "train.torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 16_192_320, "text": 4_000},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
            },
        },
    )

    # RANDOM ordering, the configuration the content budget is actually FOR: a packed_batch_size
    # bounds CONTENT, so it is ordering-independent, whereas a normal batch_size bounds the padded
    # rectangle and only means anything under laplace. Measured on the real trainings, pbs vs
    # batch_size at laplace is 1.144x (31.09h vs 35.56h over 100 ep) -- laplace already wastes
    # little, so most of the content budget's advantage is not available there.
    # Single-variable vs base-graphc-v2-pbs-fixdelta: ONLY train_seq_ordering differs.
    # The text budget stays 4_000: packed_batch_size is enforced BY THE BATCHER, so a budget can
    # never overflow the buffer -- it can only close batches early, which shows up as seqs/batch.
    loq_train(
        "base-graphc-v2-pbs-randshuf-fixdelta",
        {},
        config_overrides={
            **_loq_v3_overrides,
            "train_seq_ordering": "random",
            "train.batch_size": None,
            "train.packed_batch_size": {"audio": 16_192_320, "text": 4_000},
            "train.torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 16_192_320, "text": 4_000},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
            },
        },
    )

    # Throughput optimum from the batch-size sweep: 28M packed content, unpartitioned,
    # 409 seqs/s at 71.6 GB (30M buys nothing: +4.7 GB, -2 seqs/s). The sweep measured 227 seqs
    # at this budget, so batch_size_bound 200 would bind -> 300. Text budget scales with the
    # audio budget (28M/16.19M = 1.73x over a laplace max of 3_429) -> 8_000, still non-binding.
    # warmup_steps 0, exactly as the sweep row that MEASURED this point. The first attempt here
    # used warmup_steps 2 and died with CUDA OOM 1.06 GiB short, inside the EAGER warmup -- which
    # is the very effect the sweep documents: the eager step is the memory peak, so it, not the
    # captured graph, sets the batch-size ceiling, and warmup_steps 0 removes it (explicit
    # optimizer-state init + host constants outside the trace).
    # The other two values are the sweep's own derivation at scale = 28M/16_192_320 = 1.7292:
    # text budget round(4_000*scale) = 6_917 and batch_size_bound round(200*scale) = 346.
    # (My first attempt guessed 8_000 / 300; 300 would additionally have bound the batch.)
    loq_train(
        "base-graphc-v2-pbs28m-randshuf-fixdelta",
        {},
        config_overrides={
            **_loq_v3_overrides,
            "train_seq_ordering": "random",
            "train.batch_size": None,
            "train.packed_batch_size": {"audio": 28_000_000, "text": 6_917},
            # max_seqs MUST track batch_size_bound (the sweep sets it from the same value).
            # The first attempt left it at the base default 200 while the bound was 346, so the
            # batcher closed every batch at 200 seqs and the 28M content budget was never reached:
            # measured 24.3-27.2M content per step instead of 28M, showing up as 3-13% audio bound
            # slack that tracked the sub-epoch's mean seq length. The buffer was still sized (and
            # computed over) for 28M, so that slack was pure waste.
            "train.max_seqs": 346,
            "train.torch_cuda_graph": {
                "batch_size_bound": 346,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 28_000_000, "text": 6_917},
                "warmup_steps": 0,
                "capture_optimizer": True,
                "compile": True,
            },
        },
    )

    # REMOVED loq-v2-branch-s1337-ep6-{packed_eager,packed_graphc}: native resume from s1337's
    # ep-6 checkpoint WITH the real Adam moments; both arms reproduced the grad-norm onset with
    # near-identical per-step losses -> capture is numerically faithful at full scale.
    # REMOVED base-packed-eager-v2 (held): packed ops WITHOUT capture/compile, from scratch;
    # collapsed at ep 7 like all graphc runs (5/5 packed vs padded healthy)
    # -> the cause is in the packed eager regime, capture/compile exonerated;
    # ep-5..8 checkpoints (+ ep-7/8 opt) snapshotted in ~/tmp/packed-eager-v2-snapshots.
    # Component-swap bisects (from scratch, eager, one packed component padded-equivalent), removed:
    # REMOVED base-packed-eager-v2-attpad (all attentions via unpack -> exact padded kernels -> repack,
    # rf_packed_att_fast_paths False): SURVIVED and converged (ce 0.76@ep28, stopped) -> attention implicated.
    # REMOVED base-packed-eager-v2-ctcaten (generic aten CTC instead of the packed fast-BW native op):
    # COLLAPSED ep 7 (held) -> the CTC native op exonerated.
    # (a keep-opt v2 variant for faithful branching was planned and dropped:
    #  the s1337 / packed-eager checkpoint+opt snapshots serve that,
    #  and from-scratch runs are the trusted repro anyway)

    # REMOVED base-graphc-v3 (held): v2 + gradient_clip_global_norm 5.0 -> 1.0 (mitigation probe);
    # collapsed at ep 7 exactly like v2 -> tighter clipping does not fix the collapse,
    # and its pre-clip grad-norm log gave the onset curve (smooth exponential, doubling ~500 steps).

    # REMOVED loq-v2-branch-ep6-{padded_eager,packed_eager,packed_compiled,packed_graphc}:
    # branch from v2's ep-6 checkpoint with FRESH Adam moments (the opt state was pruned), LR pinned 7e-5;
    # ALL four arms collapsed, padded included -> a fresh-Adam artifact, no conclusion;
    # lesson: never bisect from a checkpoint without its optimizer state.


def _loq_text_seq_len_stats():
    """
    Target-length distribution of the loq train set (spm10k), for a TIGHT text bound.

    The current bound assumes the worst case per sequence -- ``max_seqs * (cap + 2)``
    = 200 * 258 = 51_600 -- against a measured mean of ~4_566, i.e. 5.5x over-provisioned,
    and every decoder-side op plus the CTC edge count pays for that.
    A batch holds at most ``max_seqs`` sequences, so the real worst case is the SUM OF THE
    200 LONGEST target sequences in the corpus, which is far smaller.
    (Tighter still if one also exploits the audio budget: the long-text seqs are long-audio ones
    and cannot co-occur -- a knapsack over (labels, duration).)

    Uses the TEXT-ONLY dataset variant, so no audio is decoded (9.5M seqs otherwise).
    """
    from i6_core.returnn.dataset import ExtractSeqLensJob
    from i6_experiments.users.zeyer.datasets.loquacious import get_loquacious_text_only_dataset_v2

    ds = get_loquacious_text_only_dataset_v2(vocab="spm10k", train_epoch_split=1)
    # "py" (seq_tag -> len), not "txt": the train ordering is laplace, so the plain length list
    # cannot be joined with the per-seq audio durations, and the bound that matters is
    # CONDITIONED on the 19.5s audio filter (only those seqs ever reach a batch).
    job = ExtractSeqLensJob(dataset=ds.train_dataset, key="text", output_format="py")
    job.rqmt = {"gpu": 0, "cpu": 2, "mem": 8, "time": 8}  # 9.5M seqs to tokenize
    tk.register_output("returnn/loq-train-text-seq-lens.py", job.out_file)


def _loq_cost_decomposition(cfg, classes_cap):
    """
    Where does the loq packed-graphc step time go?
    padded_eager is 0.594 s/step, partitioned packed_graphc 2.27 s/step at the same batching.
    These two intermediate modes split the difference into
    packed layout (packed_eager, dynamic shapes) vs bound-shaped Inductor compile (packed_compiled).
    """
    packed_overrides = {
        "packed_tensors": {
            "per_key": {
                "audio": {"gap": 960, "align": 960},
                "text": {"gap": 2, "align": 1},
            }
        }
    }
    # config_overrides are appended AFTER the mode overrides, so anything set here WINS:
    # packed_eager must not carry a torch_cuda_graph dict (it would re-enable capture),
    # and packed_compiled must repeat the mode's capture=False, compile=True itself.
    # packed_compiled at the loose text bound OOM'd; packed_eager is the informative half
    # (0.725 s/step, vs padded_eager 0.594 -- the packed LAYOUT alone costs, the win is graphc).
    for mode, overrides in [
        ("packed_eager", packed_overrides),
    ]:
        job = TrainStepBenchmarkJob(returnn_config=cfg, mode=mode, num_steps=31, config_overrides=overrides)
        tk.register_output(f"returnn/loq-base-graphc-bench-{mode}-gap960.json", job.out_results)

    # The packed CTC native fast-BW op was measured at 45% of GPU time
    # (A/B against the unpack + torch/cuDNN path).
    # The packed path always uses the native op now:
    # the alternative unpacks to a padded [B,T,V] intermediate,
    # and aten._ctc_loss is untraceable under fake tensors, so it cannot run compiled/captured.

    # A/B of the normalize frame early-exit (native op change, so a version bump gives it a
    # fresh job while keeping the 2.284 s/step reference): identical to the budget-0.8 baseline.
    job = TrainStepBenchmarkJob(
        returnn_config=cfg,
        mode="packed_graphc",
        num_steps=31,
        version=2,
        config_overrides={
            **packed_overrides,
            "torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 16_384_000, "text": 200 * (classes_cap + 2)},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
                "partitioned": True,
                "activation_memory_budget": 0.8,
                "aggressive_recomputation": True,
            },
        },
    )
    tk.register_output("returnn/loq-base-graphc-bench-partitioned-normfix.json", job.out_results)

    # v9: stop paying the conv re-layout. THE dominant cost of the packed loq step.
    # At gap 960 the packed backend logs
    #   "op 'conv': gap 1 < required 16 for the packed conv -- ... re-packing with the required gap"
    # i.e. _packed_backend regap()s the encoder activation on every conv call (it stays packed, but
    # it is a full copy). required_gap = span//2 of the Conformer depthwise conv, counted in the
    # tensor's OWN units at that point, i.e. encoder frames of 960 samples (NOT 640: the gap 11
    # variant below still warned, which is what pinned the factor down).
    # So the cheapest gap that clears it is 16*960 = 15_360; 18_240 is the cfg value.
    # Measured, partitioned + text bound 18_000:
    #   gap    960 (=1 frame)  -> warns, 1.355 s/step
    #   gap 10_560 (=11)       -> warns, 1.373 s/step   (bound is NOT what moves it)
    #   gap 18_240 (=19)       -> clean, 0.744 s/step, peak 72.2 GB
    # 1.82x, and peak memory DROPS despite the larger audio bound -- the copies cost memory too.
    # gap 960 was introduced to shrink the audio bound during the OOM fight; it cost far more in
    # copies than it saved. This is also why LS never saw this: it is on gap 18_240 throughout.
    for audio_gap, audio_bound in [(10_560, 18_304_320), (18_240, 19_840_320)]:
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=9,
            config_overrides={
                "packed_tensors": {
                    "per_key": {
                        "audio": {"gap": audio_gap, "align": 960},
                        "text": {"gap": 2, "align": 1},
                    }
                },
                "torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": audio_bound, "text": 18_000},
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                    "partitioned": True,
                    "activation_memory_budget": 0.8,
                    "aggressive_recomputation": True,
                },
            },
        )
        tk.register_output(
            f"returnn/loq-base-graphc-bench-partitioned-tight18000-audiogap{audio_gap}.json",
            job.out_results,
        )

    # v10: two follow-ups now that the regap is gone (0.744 s/step, peak 72.2 GB).
    # - partitioned=False: partitioning was added to survive the OOM, and it cost 0.984 -> 1.355 at
    #   gap 960. With the copies gone and peak at 72.2 GB the split may no longer be needed at all;
    #   unpartitioned graphc also skips the min-cut recompute entirely.
    # - gap 15_360 = the exact 16-frame minimum (16*960) instead of the cfg 18_240: same clean conv,
    #   smaller audio bound (16_000_000 + 200*(15_360+960) = 19_264_320), i.e. cheaper if it holds.
    for partitioned, audio_gap, audio_bound in [
        (False, 18_240, 19_840_320),
        (True, 15_360, 19_264_320),
    ]:
        graph_opts = {
            "batch_size_bound": 200,
            "dim_capacity": {"audio": 312_960, "text": classes_cap},
            "packed_total_bound": {"audio": audio_bound, "text": 18_000},
            "warmup_steps": 2,
            "capture_optimizer": True,
            "compile": True,
        }
        if partitioned:
            graph_opts.update({"partitioned": True, "activation_memory_budget": 0.8, "aggressive_recomputation": True})
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=10,
            config_overrides={
                "packed_tensors": {
                    "per_key": {
                        "audio": {"gap": audio_gap, "align": 960},
                        "text": {"gap": 2, "align": 1},
                    }
                },
                "torch_cuda_graph": graph_opts,
            },
        )
        tk.register_output(
            f"returnn/loq-base-graphc-bench-tight18000-audiogap{audio_gap}"
            f"{'-partitioned' if partitioned else '-unpartitioned'}.json",
            job.out_results,
        )

    # v11: compare at a REASONABLE packed batch size, i.e. throughput, not s/step at a fixed batch.
    # Packing removes the padding waste, so the packed run may take a bigger batch for the same
    # memory. Measured on the real durations under the REAL laplace:.1000 order
    # (returnn/datasets/basic.py: global permutation FIRST, then sort by length within ~1000-seq
    # bins -- NOT a global sort): padded/packed = 1.105, i.e. only 9.5% waste to recover.
    # (Fully sorted would be 1.00x and pure random 2.19x; the bin size decides, .200 -> 1.57x.)
    # Counter-pressure: the packed BOUND carries max_seqs*(gap+align) = 200*19_200 = 3.84M samples
    # of gap slack on 16M of content (24%), i.e. more than the padding it saves -- so whether packed
    # can actually run the bigger batch is an empirical question, not an accounting one.
    # Text bounds are the LP bound recomputed per audio budget (it scales with the budget):
    # 1000s -> 17_001, 1100s -> 18_286, 1250s -> 20_156.
    # Throughput = batch_size / sec_per_step; s/step across different batch sizes is meaningless.
    # padded_eager OOMs at bs125k and bs150k (79.0 / 79.1 GiB in use), so its best batch is 100k
    # at 26.9M frames/s; packed sustains 125k at 32.8M, i.e. +22% throughput on the same GPU.
    # Those two OOM points are the result, not a job to keep re-running.
    for bench_mode, bs_k, audio_bound, text_bound in [
        ("packed_graphc", 110, 21_440_640, 19_000),
        ("packed_graphc", 125, 23_840_640, 21_000),
    ]:
        overrides = {"batch_size": bs_k * 1_000 * 160}
        if bench_mode == "packed_graphc":
            overrides.update(
                {
                    "packed_tensors": {
                        "per_key": {
                            "audio": {"gap": 18_240, "align": 960},
                            "text": {"gap": 2, "align": 1},
                        }
                    },
                    "torch_cuda_graph": {
                        "batch_size_bound": 200,
                        "dim_capacity": {"audio": 312_960, "text": classes_cap},
                        "packed_total_bound": {"audio": audio_bound, "text": text_bound},
                        "warmup_steps": 2,
                        "capture_optimizer": True,
                        "compile": True,
                    },
                }
            )
        job = TrainStepBenchmarkJob(
            returnn_config=cfg, mode=bench_mode, num_steps=31, version=11, config_overrides=overrides
        )
        tk.register_output(f"returnn/loq-base-graphc-bench-bs{bs_k}k-{bench_mode}.json", job.out_results)

    # v24 (AZ request 2026-08-08): THROUGHPUT OPTIMUM over (batch size x activation memory budget),
    # under RANDOM seq ordering.
    # Ordering matters more than anything else here: with laplace:.1000 the batches are
    # length-homogeneous and padded/packed is only 1.105, under random it is 2.19 (cf. v11).
    # Random is therefore the setting in which the packed batch size is worth measuring at all,
    # and it makes the padded reference honest instead of flattering.
    # With graphc the batch is memory-bound; "partitioned" + activation_memory_budget trades step
    # time for memory (base config, measured 2026-08-07: graph pool 37.4 / 27.9 / 21.6 GiB at
    # ~0.40 / 0.44 / 0.49 s/step for budget none / 0.8 / 0.5). So a smaller budget should allow a
    # BIGGER batch; whether that nets more data throughput is what this sweep answers.
    # packed_batch_size regime (budget == bound, defaults gap 0 align 1): the batcher fills the
    # budget, so content per step == budget.
    # Metrics come from the job's log_batch_size parsing: seqs/sec and content-frames/sec are the
    # comparable throughput numbers (s/step across different batch sizes is meaningless), plus the
    # peak mem_usage incl. the CUDA-graph pool -- the memory-vs-speed trade is the point of the grid.
    # The grid walks the feasible frontier instead of a full square: unpartitioned cannot hold the
    # large batches (16.2M already sits at ~46GB resident incl. the graph pool), and tiny budgets
    # at small batches only pay the recompute without buying anything. An OOM at the top end is a
    # RESULT (it locates the frontier), not a job to keep re-running.
    # Text budget and seq bound scale with the audio budget (LP bound, cf. v11): 4_000 text and
    # 200 seqs per 16.2M audio, i.e. the same ~2x headroom over the mean seq count as production.
    # v25 (AZ 2026-08-08): the grid above was shaped while the EAGER WARMUP still set the memory
    # ceiling, so every budget over 16.19M was only ever run PARTITIONED -- which confounds batch
    # size with the activation-recompute cost. Like-for-like at budget 0.8 the bigger batch is
    # already better (24M 363 seqs/s vs 16.19M 345), and unpartitioned 16.19M uses 45.6 of 79.2 GB.
    # With warmup_steps 0 the ceiling moved, so the unpartitioned points are now reachable:
    # measured pool scales ~2.30 GB per M of budget over a ~8.4 GB param/grad/moment base,
    # so 24M -> ~63 GB and 28M -> ~73 GB should fit, 32M (~82 GB) should not.
    for audio_budget, mem_budget in [
        (16_192_320, None),  # reference: current production config
        (16_192_320, 0.8),
        (24_000_000, None),  # v25: does the bigger batch pay once partitioning is not forced?
        (28_000_000, None),  # v25: expected ~73 GB, i.e. near the unpartitioned frontier
        # v25: MEASURED 24M -> 63.0 GB / 397 seqs/s and 28M -> 71.6 GB / 409 seqs/s (vs 377 at
        # 16.19M), so the bigger batch pays and 28M is close to the wall. This pins it: predicted
        # ~76 GB of 79.2. An OOM here is the RESULT (the frontier sits between 28M and 30M),
        # not a job to re-run. Beyond it partitioning would be needed, and that costs throughput
        # (48M/0.4 manages only 351 seqs/s), so the unpartitioned frontier IS the optimum.
        (30_000_000, None),
        (24_000_000, 0.8),
        (24_000_000, 0.5),
        (32_000_000, 0.5),
        (32_000_000, 0.4),
        (40_000_000, 0.4),
        (48_000_000, 0.4),
    ]:
        scale = audio_budget / 16_192_320
        text_budget = int(round(4_000 * scale))
        graph_opts = {
            "batch_size_bound": int(round(200 * scale)),
            "dim_capacity": {"audio": 312_960, "text": classes_cap},
            "packed_total_bound": {"audio": audio_budget, "text": text_budget},
            # MEASURED 2026-08-08: with the default 2 eager warmup steps every point above
            # 16.19M died with a real CUDA OOM at 79.1 of 79.2 GiB -- in the WARMUP, before the
            # compile ever ran. The eager step is the memory peak (61.6 GB at 16.19M), so it,
            # not the captured graph, was setting the batch-size ceiling. warmup_steps 0 removes
            # it entirely (explicit optimizer-state init + host constants outside the trace),
            # which is what makes the bigger budgets measurable at all.
            "warmup_steps": 0,
            "capture_optimizer": True,
            "compile": True,
        }
        if mem_budget is not None:
            graph_opts["partitioned"] = True
            graph_opts["activation_memory_budget"] = mem_budget
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=24,
            seq_ordering="random",
            config_overrides={
                "batch_size": None,
                "packed_batch_size": {"audio": audio_budget, "text": text_budget},
                "packed_tensors": True,
                "max_seqs": graph_opts["batch_size_bound"],
                "log_batch_size": True,
                "torch_cuda_graph": graph_opts,
            },
        )
        tk.register_output(
            f"returnn/loq-throughput-sweep-audio{audio_budget // 1_000_000}M"
            f"-membudget{'none' if mem_budget is None else mem_budget}.json",
            job.out_results,
        )

    # Production-as-it-runs-today, at its OWN ordering. A normal `batch_size` budget bounds the
    # PADDED rectangle, so pairing it with random ordering measures nothing but the padding waste
    # (measured: 42% fill, 6.66M content in a 15.8M rectangle) -- laplace is what such a config
    # must run, and the only setting in which it means anything. Only the packed_batch_size rows,
    # whose budget counts content, are ordering-independent and therefore worth running random.
    tk.register_output(
        "returnn/loq-throughput-sweep-production-laplace.json",
        TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=25,
            config_overrides={"log_batch_size": True},
        ).out_results,
    )

    # Isolation row for the production comparison. The sweep rows above declare text bound
    # 4_000 per 16.19M audio, production declares 18_000 -- a 4.5x difference on a dim where the
    # ACTUAL content is ~2_600-3_000 either way. Since production measures 65.8 GB / 0.434 s and
    # the same-audio-bound sweep row measures 45.6 GB / 0.347 s, the text bound is the prime
    # suspect for both gaps, and without this row the pbs-vs-production comparison confounds the
    # regime change with the bound change. Same as the 16.19M/none row except the text bound.
    tk.register_output(
        "returnn/loq-throughput-sweep-audio16M-textbound18000.json",
        TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=25,
            seq_ordering="random",
            config_overrides={
                "batch_size": None,
                "packed_batch_size": {"audio": 16_192_320, "text": 18_000},
                "packed_tensors": True,
                "max_seqs": 200,
                "log_batch_size": True,
                "torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": 16_192_320, "text": 18_000},
                    "warmup_steps": 0,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        ).out_results,
    )

    # Padded reference, at LAPLACE: bs100k is the largest padded batch that fits (125k/150k OOMed,
    # cf. v11). It runs laplace because a padded batch_size budget bounds the rectangle, so random
    # ordering would measure its padding waste rather than the setup anyone would actually run --
    # each side gets its own realistic ordering, and only the packed_batch_size rows, being
    # ordering-independent, are run random.
    # The padded side gets its batch size TUNED TOO, else the comparison is unfair: bs100k came
    # from the v11 measurement where 125k/150k OOMed, but at laplace it measures only 64.4 GB of
    # 79.2, i.e. ~15 GB unused. Scaling from that point (~8.4 GB base + ~0.56 GB per 1k of padded
    # batch) puts the wall near 126k, so 110k and 120k should fit and 125k is marginal --
    # an OOM at the top is the result, the same convention as the packed rows.
    # v26: 100k/110k/120k measured 206.1/196.3/196.9 seqs/sec -- throughput FALLS as the batch
    # grows, so the optimum is at or below the smallest point ever measured. The padded rectangle
    # is bounded by the longest seq in the batch, so a bigger budget buys mostly padding here.
    # Probe downwards until the per-step overhead takes over again.
    for padded_bs_k in [60, 70, 80, 90, 100, 110, 120]:
        tk.register_output(
            f"returnn/loq-throughput-sweep-padded-bs{padded_bs_k}k.json",
            TrainStepBenchmarkJob(
                returnn_config=cfg,
                mode="padded_eager",
                num_steps=31,
                # v25: the first run predates the content logging (the eager path reported only
                # num_seqs and the per-dim max), so it could not say how much of the padded
                # rectangle was actually data -- which is the whole comparison
                version=25,
                config_overrides={"batch_size": padded_bs_k * 1_000 * 160, "log_batch_size": True},
            ).out_results,
        )

    # v13: re-measure the best config after two native-op changes:
    # the CTC edge skip was REVERTED (it assumed the CTC topology inside the generic
    # FastBaumWelchPackedOp, which takes arbitrary edges -- any other automaton would have had
    # live edges silently zeroed), and the all-inactive-frame check in normalize now reads `index`
    # once per block instead of once per thread.
    # Reference: 0.554 s/step, measured WITH the edge skip, so this quantifies what the revert costs.
    for bs_k, audio_bound, text_bound in [(100, 19_840_320, 18_000)]:
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=13,
            config_overrides={
                "batch_size": bs_k * 1_000 * 160,
                "packed_tensors": {
                    "per_key": {
                        "audio": {"gap": 18_240, "align": 960},
                        "text": {"gap": 2, "align": 1},
                    }
                },
                "torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": audio_bound, "text": text_bound},
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        )
        tk.register_output(f"returnn/loq-base-graphc-bench-bs{bs_k}k-ctcgeneric.json", job.out_results)

    # v14: best config after the CTC work landed generically.
    # Dead edges now carry an INF weight, written by GetCtcFsaFastBwOp itself
    # (srel_edge_idx < 0 = the edges it routes into the dummy state),
    # so fast_baum_welch skips them on one coalesced load,
    # with no topology and no state-numbering convention shared across files.
    # 81.7% of the 257k edges are dead at these shapes; the op alone: 94.0 -> 60.2 ms (1.56x).
    # The aux heads also share one FSA now instead of building it three times.
    # References: 0.864 with none of this, 0.554 with the (reverted) hardcoded skip,
    # padded_eager 0.594.
    for bs_k, audio_bound, text_bound in [(100, 19_840_320, 18_000)]:
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=14,
            config_overrides={
                "batch_size": bs_k * 1_000 * 160,
                "packed_tensors": {
                    "per_key": {
                        "audio": {"gap": 18_240, "align": 960},
                        "text": {"gap": 2, "align": 1},
                    }
                },
                "torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": audio_bound, "text": text_bound},
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        )
        tk.register_output(f"returnn/loq-base-graphc-bench-bs{bs_k}k-ctcinfweight.json", job.out_results)

    # v17: do we need any input-side gap at all?
    # audio gap 0 / align 960 packs to 16_192_000 samples (16_867 frames); the conv then regaps to
    # its 16 required frames -> 16_867 + 200*16 = 20_067, the same width gap 960 ends at
    # (17_067 + 200*15), but from a smaller input buffer. text gap 0 means the BOS pad is no longer
    # in-place, so it costs one re-layout -- against 26 attention densifies it should still win.
    # The (0, 0) point is also simply the least the user has to configure.
    # `repeat` only changes the sis hash: two identical configs give the run-to-run spread,
    # which is what says whether the 0.499 / 0.508 / 0.512 above are actually different.
    for audio_gap, text_gap, repeat in [(0, 0, 0), (0, 2, 0), (960, 2, 1), (18_240, 2, 1)]:
        audio_bound = 16_000_000 + 200 * (audio_gap + 960)
        audio_bound = -(-audio_bound // 960) * 960
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=17 + repeat,
            config_overrides={
                "batch_size": 100_000 * 160,
                "packed_tensors": {
                    "per_key": {
                        "audio": {"gap": audio_gap, "align": 960},
                        "text": {"gap": text_gap, "align": 1},
                    }
                },
                "torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": audio_bound, "text": 18_000},
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        )
        tk.register_output(
            f"returnn/loq-base-graphc-bench-gaps-a{audio_gap}-t{text_gap}{'-rep' if repeat else ''}.json",
            job.out_results,
        )

    # v16: text gap 1 instead of 2.
    # The decoder input is a BOS pad of the packed targets: footprint L+gap, content 1+L.
    # With gap 2 that leaves gap 1, and _torch_sdpa_varlen_attention densifies whenever the packing
    # has gap frames (nested/jagged needs contiguous offsets), so the decoder pays
    # 26 re-layouts per step (12 q-in + 12 out + k/v; traced).
    # With gap 1 the pad leaves gap 0 -- the pad stays in place AND the densify disappears.
    # The targets-with-EOS is a separate tensor deriving from the same packing,
    # so one slack frame should serve both; if something needs BOS+labels+EOS in ONE seq it will
    # assert on insufficient gap, which is the informative outcome.
    for text_gap in [1, 2]:
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=16,
            config_overrides={
                "batch_size": 100_000 * 160,
                "packed_tensors": {
                    "per_key": {
                        "audio": {"gap": 18_240, "align": 960},
                        "text": {"gap": text_gap, "align": 1},
                    }
                },
                "torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": 19_840_320, "text": 18_000},
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        )
        tk.register_output(f"returnn/loq-base-graphc-bench-textgap{text_gap}.json", job.out_results)

    # v15: regap now derives its bound from the DECLARED total bound (pack total_bound) instead of
    # the per-seq capacity product, which put every seq at its full length at once
    # (loq: 68_400 frames instead of 20_667, inherited by every op after the conv).
    # gap 960 should now land at 17_067 + 200*15 = 20_067 frames, i.e. slightly BELOW the
    # gap-18_240 layout (20_667), while the buffer before the encoder stays 16.38M samples
    # instead of 19.84M. So the gap becomes a layout detail again, not a speed decision.
    # References: gap 18_240 unpartitioned = 0.556 s/step, padded_eager = 0.594.
    # (the gap-18_240 arm is v16 textgap2; gap 960 is the one this change unlocks)
    for audio_gap, audio_bound, tag in [(960, 16_384_000, "gap960")]:
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode="packed_graphc",
            num_steps=31,
            version=15,
            config_overrides={
                "batch_size": 100_000 * 160,
                "packed_tensors": {
                    "per_key": {
                        "audio": {"gap": audio_gap, "align": 960},
                        "text": {"gap": 2, "align": 1},
                    }
                },
                "torch_cuda_graph": {
                    "batch_size_bound": 200,
                    "dim_capacity": {"audio": 312_960, "text": classes_cap},
                    "packed_total_bound": {"audio": audio_bound, "text": 18_000},
                    "warmup_steps": 2,
                    "capture_optimizer": True,
                    "compile": True,
                },
            },
        )
        tk.register_output(f"returnn/loq-base-graphc-bench-declaredbound-{tag}.json", job.out_results)

    # A gap-960 rerun belongs here once regap derives its bound from the DECLARED total bound
    # instead of the per-seq capacity product.
    # Then the regap lands at 17_067 + 200*15 = 20_067 frames,
    # i.e. slightly below the gap-18_240 layout (20_667),
    # while the audio buffer before the encoder stays at 16.38M samples instead of 19.84M.
    # The first attempt derived the bound per tensor, which broke layout compatibility
    # (two tensors regapped independently disagreed, aten.where: 20400 vs 18000), and was reverted.
    # Reference to beat: gap 18_240 unpartitioned = 0.554 s/step at bs100k.

    # v5: + next_frame_packed skips edges past each seq's OWN target length (derived from
    # start_end_states), i.e. the CTC work follows the actual targets rather than the buffer width.
    job = TrainStepBenchmarkJob(
        returnn_config=cfg,
        mode="packed_graphc",
        num_steps=31,
        version=5,
        config_overrides={
            **packed_overrides,
            "torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 16_384_000, "text": 200 * (classes_cap + 2)},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
                "partitioned": True,
                "activation_memory_budget": 0.8,
                "aggressive_recomputation": True,
            },
        },
    )
    tk.register_output("returnn/loq-base-graphc-bench-partitioned-ctcedgeskip.json", job.out_results)

    # v3: + skip INF (zero-probability) edges in the normalize accumulation.
    # prob_add(x, INF) == x, so this is exact; it drops the contended log-space CAS for the
    # ~80% dummy edges the FSA creates when the targets buffer is wider than the target lengths.
    job = TrainStepBenchmarkJob(
        returnn_config=cfg,
        mode="packed_graphc",
        num_steps=31,
        version=3,
        config_overrides={
            **packed_overrides,
            "torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 16_384_000, "text": 200 * (classes_cap + 2)},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
                "partitioned": True,
                "activation_memory_budget": 0.8,
                "aggressive_recomputation": True,
            },
        },
    )
    tk.register_output("returnn/loq-base-graphc-bench-partitioned-ctcskipinf.json", job.out_results)

    # ... and profiled, to see where normalize actually went (the A/B moved only 1.5%,
    # far less than the empty-frame fraction predicts)
    job = TrainStepBenchmarkJob(
        returnn_config=cfg,
        mode="packed_graphc",
        num_steps=31,
        version=2,
        config_overrides={
            **packed_overrides,
            "torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 16_384_000, "text": 200 * (classes_cap + 2)},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
                "partitioned": True,
                "activation_memory_budget": 0.8,
                "aggressive_recomputation": True,
            },
            "torch_profile": {"schedule": {"skip_first": 12, "wait": 1, "warmup": 2, "active": 3, "repeat": 1}},
        },
    )
    tk.register_output("returnn/loq-base-graphc-bench-partitioned-normfix-profiled.json", job.out_results)

    # how much does the (very loose) TEXT bound cost? classes_cap 256 and a 200*258 = 51_600
    # packed total, against an observed max target len of 83 and a mean text footprint of ~4.6k:
    # every decoder-side op runs at ~11x the content it needs. This variant just tightens the
    # bound to a realistic cap (still above the observed max), everything else identical.
    tight_cap = 96
    job = TrainStepBenchmarkJob(
        returnn_config=cfg,
        mode="packed_graphc",
        num_steps=31,
        config_overrides={
            **packed_overrides,
            "torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": tight_cap},
                "packed_total_bound": {"audio": 16_384_000, "text": 200 * (tight_cap + 2)},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
                "partitioned": True,
                "activation_memory_budget": 0.8,
                "aggressive_recomputation": True,
            },
        },
    )
    tk.register_output("returnn/loq-base-graphc-bench-partitioned-tighttext.json", job.out_results)

    # same profile for packed_eager (0.725 s/step): the graphc step spends 883 ms in the CTC
    # native op and still ~1056 ms on everything else, which alone exceeds the whole padded step.
    # Comparing the two kernel breakdowns says whether the non-CTC part is really slower
    # (bound-shaped compute + recompute) or whether the CTC op is just cheaper at actual sizes.
    job = TrainStepBenchmarkJob(
        returnn_config=cfg,
        mode="packed_eager",
        num_steps=31,
        config_overrides={
            **packed_overrides,
            "torch_profile": {"schedule": {"skip_first": 12, "wait": 1, "warmup": 2, "active": 3, "repeat": 1}},
        },
    )
    tk.register_output("returnn/loq-base-graphc-bench-packed_eager-profiled.json", job.out_results)

    # kernel-level profile of the partitioned graphc step: where do the 2.27 s actually go?
    # (op counts in the traced graph only suggest; the profiler measures)
    job = TrainStepBenchmarkJob(
        returnn_config=cfg,
        mode="packed_graphc",
        num_steps=31,
        config_overrides={
            **packed_overrides,
            "torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 16_384_000, "text": 200 * (classes_cap + 2)},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
                "partitioned": True,
                "activation_memory_budget": 0.8,
                "aggressive_recomputation": True,
            },
            # repeat>0 is what makes RETURNN compute max_step and hence export the chrome trace
            # (with repeat=0 the profiler runs but never writes torch_profile.json)
            "torch_profile": {"schedule": {"skip_first": 12, "wait": 1, "warmup": 2, "active": 3, "repeat": 1}},
        },
    )
    tk.register_output("returnn/loq-base-graphc-bench-partitioned-profiled.json", job.out_results)

    # Post-delta-fix CTC-share verification: kernel-level profiles on the CURRENT code,
    # all three modes at the same bs100k batching (packed at the current best gap 18_240 config).
    # Questions: what fraction is the CTC fast-BW op now (was ~45% before the varlen edge fixes,
    # then partially reverted for generic-automaton correctness), and the padded/eager comparison
    # of that share (padded runs aten CTC, not the native op).
    # No packed_tensors/torch_cuda_graph overrides: cfg is the REAL training config and the
    # mode string adapts it (padded_eager nulls both, packed_eager nulls the graph) --
    # so these profile exactly what the production trainings run.
    # v22: measure the config the v2 TRAININGS actually run. v18-v21 inherited `cfg` = the
    # ORIGINAL graphc experiment (gap 18_240, text bound 200*(cap+2)=51_600, bhv 24); the
    # trainings moved on to the nogap v2 config (audio gap 0, text bound 18_000, bhv 29).
    # The v21 trace proved the packed FSA follows the DECLARED bound exactly
    # (n_edges 259_000 = 5*51_600+5*200), so the loose bench bound overstated the CTC cost
    # ~2.8x vs the real trainings (91_000 edges at text 18_000).
    # v21 (loose 51_600 bound): packed_graphc 0.435 (CTC 3.8%, normalize 27.8ms block-512),
    # packed_eager 0.747 (CTC 0.4%), padded_eager 0.615 (aten CTC 1.6%).
    # v21 vs v18 0.554: the RETURNN_CUDA macro fix -- the torch op builder #undef'ed CUDA
    # before the op fw code, so `#if CUDA` picked norm_block_dim 1: normalize ran fully
    # SERIAL (47ms/launch, block [1,1,1] in the v20 trace) in every torch training before.
    # v19/v20 raced/were masked by that serial normalize.
    # v23 (v22 BUG: config_overrides are appended AFTER the mode overrides, so re-declaring
    # packed_tensors/torch_cuda_graph in the overrides resurrected full graphc in ALL THREE
    # modes -- v22's trio came out identical, 0.358-0.362. That graphc measurement itself is
    # valid: tight 18_000 text bound confirmed in-trace, 28_971 blocks = 91_000 edges,
    # normalize 27.8 -> 5.8 ms, CTC 3.5%. v23 tailors the overrides per mode.)
    _v2_align = 960
    _v2_audio_bound = 100_000 * _loq_batch_size_factor() + 200 * _v2_align
    _v2_audio_bound = -(-_v2_audio_bound // _v2_align) * _v2_align
    _v2_packed_tensors = {
        "per_key": {
            "audio": {"gap": 0, "align": _v2_align},
            "text": {"gap": 0, "align": 1},
        }
    }
    _v2_graph_opts = {
        "batch_size_bound": 200,
        "dim_capacity": {"audio": 312_960, "text": classes_cap},
        "packed_total_bound": {"audio": _v2_audio_bound, "text": 18_000},
        "warmup_steps": 2,
        "capture_optimizer": True,
        "compile": True,
    }
    for profiled_mode in ["packed_graphc", "packed_eager", "padded_eager"]:
        overrides = {
            "behavior_version": 29,
            "torch_profile": {"schedule": {"skip_first": 12, "wait": 1, "warmup": 2, "active": 3, "repeat": 1}},
        }
        if profiled_mode != "padded_eager":
            overrides["packed_tensors"] = _v2_packed_tensors
        if profiled_mode == "packed_graphc":
            overrides["torch_cuda_graph"] = _v2_graph_opts
        job = TrainStepBenchmarkJob(
            returnn_config=cfg,
            mode=profiled_mode,
            num_steps=31,
            version=23,
            config_overrides=overrides,
        )
        tk.register_output(f"returnn/loq-bench-profiled-fixdelta-{profiled_mode}.json", job.out_results)


def _loq_batch_size_factor():
    """the raw-sample batch-size factor of the baseline configs"""
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import _batch_size_factor

    return _batch_size_factor


class TrainStepBenchmarkJob(Job):
    """
    Run a real training config for a bounded number of train steps in one mode
    (``padded_eager`` / ``packed_eager`` / ``packed_compiled`` / ``packed_graphc``),
    on the real train dataset,
    and parse the per-step train losses and sec/step from the log.

    Fixed random seed and identical data order across modes:
    the per-step losses of two modes must match up to bf16 noise,
    so diffing these outputs is a numerical parity check
    of the packed / compiled / graph-captured train step.
    The sec/step stats give the realistic speed comparison.

    The curriculum (``epoch_wise_filter``) is stripped:
    it would keep long seqs out of the first epoch,
    and the interesting behavior (capacity bounds, packing) needs them.
    """

    __sis_hash_exclude__ = {
        "config_overrides": None,
        "returnn_config": None,
        "returnn_config_file": None,
        "load_checkpoint": None,
        "seq_ordering": None,
        "version": 1,
    }

    def __init__(
        self,
        *,
        returnn_config_file: Optional[tk.Path] = None,
        returnn_config: Optional[ReturnnConfig] = None,
        mode: str,
        num_steps: int = 300,
        random_seed: int = 42,
        config_overrides: Optional[Dict[str, Any]] = None,
        load_checkpoint: Optional[tk.Path] = None,
        seq_ordering: Optional[str] = None,
        version: int = 1,
    ):
        """
        :param returnn_config_file: serialized RETURNN config file of a training job (older jobs)
        :param returnn_config: the config object (preferred; e.g. from the training job of an experiment)
        :param mode: "padded_eager", "packed_eager", "packed_compiled" or "packed_graphc"
        :param num_steps: stop (kill) the training once this many train steps are logged
        :param random_seed: fixed seed, same across modes
        :param config_overrides: appended (repr) after the mode overrides,
            e.g. ``{"specaugment_steps": (0, 0, 0)}`` to force the specaugment schedule on from step 0
        :param load_checkpoint: init the params from this checkpoint (e.g. for ep-2 repro benches)
        :param seq_ordering: rewrite the seq ordering of every train (sub-)dataset that has one,
            e.g. "random" instead of the config's "laplace:.1000".
            Laplace batches are length-homogeneous, which hides most of the padding waste --
            exactly the effect a packed-vs-padded throughput measurement is about
            (measured on the real durations: padded/packed = 1.105 under laplace:.1000,
            2.19 under random).
        :param version: behavior version, bump to force a re-run (hash-neutral at the default)
        """
        assert mode in (
            "padded_eager",
            "packed_eager",
            "packed_compiled",
            "packed_compiled_nandump",
            "packed_compiled_nandump_zeroinit",
            "packed_compiled_nanreport",
            "packed_compiled_nanassert",
            "packed_compiled_nofuse",
            "packed_aot_eager",
            "packed_eager_bound",
            "packed_graphc",
        )
        assert (returnn_config is None) != (returnn_config_file is None), "specify exactly one config source"
        self.returnn_config_file = returnn_config_file
        self.returnn_config = returnn_config
        self.load_checkpoint = load_checkpoint
        self.seq_ordering = seq_ordering
        self.mode = mode
        self.num_steps = num_steps
        self.random_seed = random_seed
        self.config_overrides = config_overrides
        self.version = version
        self.rqmt = {"gpu": 1, "cpu": 24, "mem": 100, "time": 2}
        self.out_results = self.output_path("results.json")
        self.out_log = self.output_path("returnn.log")

    def tasks(self):
        """tasks"""
        # run() is idempotent (rewrites config, reruns, reparses),
        # so an interrupted run can simply restart
        yield Task("run", rqmt=self.rqmt, resume="run")

    _mode_overrides = {
        "padded_eager": "packed_tensors = None\ntorch_cuda_graph = None\n",
        "packed_eager": "torch_cuda_graph = None\n",
        "packed_compiled": "torch_cuda_graph = dict(torch_cuda_graph, capture=False, compile=True)\n",
        # like packed_compiled, but dumps the first non-finite step's batch (debug, see graph_capture)
        "packed_compiled_nandump": (
            "torch_cuda_graph = dict(torch_cuda_graph, capture=False, compile=True, debug_nan_dump_inputs=True)\n"
        ),
        # nandump with zero-filled generated buffers (no asserts):
        # if the ep-2 NaN vanishes, it came from uninitialized reads; if it stays, it is computed
        "packed_compiled_nandump_zeroinit": (
            "torch_cuda_graph = dict(torch_cuda_graph, capture=False, compile=True,"
            " debug_nan_dump_inputs=True, debug_zero_init_buffers=True)\n"
        ),
        # NaN-count REPORT per buffer per call (no abort): the culprit is the buffer whose
        # NaN pattern changes at the failing call (pre-guard/masked-lane NaNs stay constant)
        "packed_compiled_nanreport": (
            "torch_cuda_graph = dict(torch_cuda_graph, capture=False, compile=True,"
            " inductor_nan_asserts=True, debug_nan_report=True,"
            " debug_nan_dump_inputs=True, debug_zero_init_buffers=True,"
            # operands of the failing cross-att flash backward (nanassert-v2 module L24992;
            # buffer numbering is stable per graph): grad_out, q, k, v, out, lse,
            # cu_q, cu_k, philox offset/seed, grad_k -- dumped per call, last = failing call
            " debug_dump_buffer_names=('buf1977', 'buf1882', 'buf1890', 'buf1898',"
            " 'buf1900', 'buf1901', 'buf1281', 'buf1355', 'buf1902', 'buf1903', 'buf1980'))\n"
        ),
        # plus Inductor nan-asserts: the generated code checks every buffer,
        # the first failing assert names the producing op
        "packed_compiled_nanassert": (
            "torch_cuda_graph = dict(torch_cuda_graph, capture=False, compile=True,"
            " inductor_nan_asserts=True, debug_nan_dump_inputs=True,"
            # zero-fill: kills the benign garbage tails of bound-sized buffers,
            # so any remaining nan-assert names a REAL compute defect;
            # a fully clean run instead = the program reads unwritten regions
            " debug_zero_init_buffers=True)\n"
        ),
        # step_core UNTRACED, plain eager, on the same bound buffers:
        # NaN = pure bound-regime RF semantics bug (pdb-able); clean = the AOT trace differs
        "packed_eager_bound": (
            "torch_cuda_graph = dict(torch_cuda_graph, capture=False, compile=True,"
            " debug_eager_bound=True, debug_nan_dump_inputs=True)\n"
        ),
        # the traced AOT graph run with EAGER kernels (no Inductor):
        # NaN persisting = decomposition-semantics issue; clean = Inductor codegen at fault
        "packed_aot_eager": (
            "torch_cuda_graph = dict(torch_cuda_graph, capture=False, compile=True,"
            " debug_aot_eager=True, debug_nan_dump_inputs=True)\n"
        ),
        # conservative Inductor codegen: no epilogue fusion / pattern matcher --
        # discriminator for miscompiled-fusion NaNs (failing call moves with codegen layout)
        "packed_compiled_nofuse": (
            "torch_cuda_graph = dict(torch_cuda_graph, capture=False, compile=True,"
            " inductor_conservative=True, debug_nan_dump_inputs=True)\n"
        ),
        "packed_graphc": "",
    }

    def run(self):
        """run"""
        import glob
        import json
        import os
        import re
        import shutil
        import subprocess
        import sys
        import time

        import returnn
        import i6_experiments

        returnn_root = os.path.dirname(os.path.dirname(os.path.abspath(returnn.__file__)))
        recipe_root = os.path.dirname(os.path.dirname(os.path.abspath(i6_experiments.__file__)))
        if self.returnn_config is not None:
            base_cfg_path = "returnn.base.config"
            # no black formatting (cosmetic only):
            # this runs on the GPU node, where the pickled black path is unavailable
            # (c25g does not mount /work)
            self.returnn_config.black_formatting = False
            self.returnn_config.write(base_cfg_path)
            with open(base_cfg_path, "rt", encoding="utf-8") as f:
                cfg = f.read()
        else:
            with open(self.returnn_config_file.get_path(), "rt", encoding="utf-8") as f:
                cfg = f.read()
        work = os.path.abspath("train-work")
        os.makedirs(work + "/models", exist_ok=True)
        import_model_line = ""
        if self.load_checkpoint is not None:
            src = self.load_checkpoint.get_path()
            opt_src = src[: -len(".pt")] + ".opt.pt"
            if os.path.exists(opt_src):
                # continue-training repro: place the checkpoint (+ optimizer state) as epoch 1
                # of the local model dir, so the engine natively resumes at epoch 2
                # (a bare `load` makes the engine look for the optimizer state in the local dir)
                for src_path, link_name in [(src, "epoch.001.pt"), (opt_src, "epoch.001.opt.pt")]:
                    link_path = work + "/models/" + link_name
                    if not os.path.exists(link_path):
                        os.symlink(src_path, link_path)
                # the resume also needs the epoch-1 entry of the learning_rates file
                # (the engine refuses to resume with the last epoch's scores missing).
                # COPY it (RETURNN rewrites the file every epoch -- a symlink would
                # write into the source training job).
                # A finished training has it in output/, a running one still in work/
                job_dir = os.path.join(os.path.dirname(src), "..", "..")
                dst_lr = work + "/learning_rates"
                for sub in ["output", "work"]:
                    src_lr = os.path.join(job_dir, sub, "learning_rates")
                    if os.path.exists(src_lr) and not os.path.exists(dst_lr):
                        shutil.copyfile(src_lr, dst_lr)
                        break
            else:
                # no optimizer state (cleaned up): the resume path CANNOT be used --
                # get_existing_models skips a checkpoint without .opt.pt for training,
                # so the engine would silently start from scratch (that happened).
                # Import the weights instead; the caller must pin the LR
                # (the import restarts at global step 0, i.e. warmup).
                import_model_line = f"import_model_train_epoch1 = {src!r}\n"
        cfg += (
            "\n\n# ---- TrainStepBenchmarkJob overrides ----\n"
            f"model = {work + '/models/epoch'!r}\n"
            + import_model_line
            + f"learning_rate_file = {work + '/learning_rates'!r}\n"
            "use_train_proc_manager = False\n"
            f"random_seed = {self.random_seed}\n"
            + self._mode_overrides[self.mode]
            + "".join(f"{k} = {v!r}\n" for k, v in (self.config_overrides or {}).items())
            + "def _strip_epoch_wise_filter(d):\n"
            "    if isinstance(d, dict):\n"
            "        d.pop('epoch_wise_filter', None)\n"
            "        for v in d.values():\n"
            "            _strip_epoch_wise_filter(v)\n"
            "    elif isinstance(d, (list, tuple)):\n"
            "        for v in d:\n"
            "            _strip_epoch_wise_filter(v)\n"
            "\n"
            "_strip_epoch_wise_filter(train)\n"
            + (
                ""
                if self.seq_ordering is None
                else (
                    "import functools as _functools\n"
                    "def _set_seq_ordering(d):\n"
                    # the real sub-dataset dict is a KEYWORD of a functools.partial
                    # (DistributeFilesDataset get_sub_epoch_dataset=partial(..., base_dict={...})),
                    # so a plain dict/list walk never reaches the ordering that matters
                    "    if isinstance(d, _functools.partial):\n"
                    "        _set_seq_ordering(d.keywords)\n"
                    "        _set_seq_ordering(list(d.args))\n"
                    "    elif isinstance(d, dict):\n"
                    # 'default' means unordered; only real orderings (laplace, sorted, random) are rewritten
                    "        if d.get('seq_ordering', 'default') != 'default':\n"
                    f"            d['seq_ordering'] = {self.seq_ordering!r}\n"
                    "        for v in d.values():\n"
                    "            _set_seq_ordering(v)\n"
                    "    elif isinstance(d, (list, tuple)):\n"
                    "        for v in d:\n"
                    "            _set_seq_ordering(v)\n"
                    "\n"
                    "_set_seq_ordering(train)\n"
                    # guard the partial path above: a silently unchanged ordering would make the
                    # whole measurement meaningless (laplace batches hide the padding waste)
                    f"assert 'laplace' not in repr(train), 'seq_ordering {self.seq_ordering} did not reach every dataset'\n"
                )
            )
        )
        cfg_path = work + "/returnn.config"
        with open(cfg_path, "wt", encoding="utf-8") as f:
            f.write(cfg)
        log_path = self.out_log.get_path()
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(p for p in [recipe_root, env.get("PYTHONPATH")] if p)
        # like the real trainings (env_updates in the base configs);
        # the phase transitions of graph capture (warmup / compile / capture)
        # fragment badly without it
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # log the Inductor peak-memory reorder pass (baseline/method estimates, failures)
        env.setdefault("TORCH_LOGS", "+torch._inductor.memory")
        # persistent Inductor cache:
        # the generated kernels survive the job (node-local tmp does not),
        # needed to map buffers in nan-assert failures to ops
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", work + "/inductor-cache")

        with open(log_path, "wt", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                [sys.executable, returnn_root + "/rnn.py", cfg_path],
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=work,
                env=env,
            )
            # leave slack before the slurm time limit so parsing + output still happen
            deadline = time.monotonic() + self.rqmt["time"] * 3600 - 900
            while proc.poll() is None:
                time.sleep(10)
                with open(log_path, "rt", encoding="utf-8", errors="replace") as f:
                    n_steps_logged = sum(1 for line in f if re.search(r"ep \d+ train, step \d+,", line))
                if n_steps_logged >= self.num_steps or time.monotonic() > deadline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

        # keep any nan-batch dumps (debug_nan_dump_inputs writes them into the train cwd,
        # which sisyphus deletes once the job finishes)
        for fn in glob.glob(os.path.join(work, "nan-*.pt")):
            shutil.copy(fn, os.path.join(os.path.dirname(self.out_results.get_path()), os.path.basename(fn)))
        # same for the Torch profiler outputs (torch_profile option)
        for fn in glob.glob(os.path.join(work, "torch_profile*.json")) + glob.glob(
            os.path.join(work, "torch_memory_profile*.html")
        ):
            shutil.copy(fn, os.path.join(os.path.dirname(self.out_results.get_path()), os.path.basename(fn)))

        # "ep 1 train, step 3, <losses>[, num_seqs N, max_size:k N, sum_size:k N (log_batch_size)],
        #  mem_usage:cuda 46.9GB[, mem_graph_pool:cuda 38.3GB][, 0.398 sec/step], ..."
        step_re = re.compile(r"ep \d+ train, step (\d+), (.*?), mem_usage:cuda ([0-9.]+)([KMGT]?B)(.*)")
        pool_re = re.compile(r"mem_graph_pool:cuda ([0-9.]+)([KMGT]?B)")
        sec_re = re.compile(r"([0-9.]+) sec/step")
        gb_per = {"B": 1 / 1024**3, "KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}
        steps = []
        with open(log_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = step_re.search(line)
                if not m:
                    continue
                losses = {}
                sizes = {}
                for part in m.group(2).split(", "):
                    name, _, value = part.rpartition(" ")
                    # batch-size info (log_batch_size), kept apart from the losses
                    # so the cross-mode loss diff stays a pure parity check
                    if name == "num_seqs" or name.startswith(("max_size:", "sum_size:")):
                        sizes[name] = int(value)
                        continue
                    try:
                        losses[name] = float(value)
                    except ValueError:
                        pass
                tail = m.group(5)
                m_pool = pool_re.search(tail)
                m_sec = sec_re.search(tail)
                steps.append(
                    {
                        "step": int(m.group(1)),
                        "losses": losses,
                        "sizes": sizes,
                        "mem_usage_gb": float(m.group(3)) * gb_per[m.group(4)],
                        "graph_pool_gb": float(m_pool.group(1)) * gb_per[m_pool.group(2)] if m_pool else None,
                        "sec_per_step": float(m_sec.group(1)) if m_sec else None,
                    }
                )
        # a crashed run can still leave a few parsed steps (e.g. OOM after warmup);
        # a deadline-killed healthy run has close to num_steps
        assert len(steps) >= min(self.num_steps, 10), f"only {len(steps)} train steps parsed from {log_path}"
        # from step 5 on: warmup, compile and capture are done, the rest are steady-state steps
        steady = [s for s in steps if s["step"] >= 5]
        times = sorted(s["sec_per_step"] for s in steady if s["sec_per_step"] is not None)
        median_sec = times[len(times) // 2] if times else None

        def _mean_size(name):
            """mean over the steady steps of one batch-size field (None if not logged)"""
            vals = [s["sizes"][name] for s in steady if name in s["sizes"]]
            return sum(vals) / len(vals) if vals else None

        mean_seqs = _mean_size("num_seqs")
        # Content frames actually computed on. The key is named after the DATA KEY under graph
        # capture and after the DIM in the eager path, so pick the largest sum_size instead of
        # guessing a name: audio content outweighs the label content by orders of magnitude.
        content_keys = {k for s in steady for k in s["sizes"] if k.startswith("sum_size:")}
        mean_content = max((_mean_size(k) for k in content_keys), default=None)
        res = {
            "mode": self.mode,
            "num_steps_parsed": len(steps),
            "median_sec_per_step": median_sec,
            "mean_num_seqs": mean_seqs,
            "mean_content_frames": mean_content,
            # the throughput numbers: what a bigger batch actually buys.
            # sec/step alone is meaningless across different batch sizes.
            "seqs_per_sec": mean_seqs / median_sec if (mean_seqs and median_sec) else None,
            "content_frames_per_sec": mean_content / median_sec if (mean_content and median_sec) else None,
            # peak GPU need INCL the CUDA-graph private pool (the engine reports the sum)
            "max_mem_usage_gb": max((s["mem_usage_gb"] for s in steps), default=None),
            "graph_pool_gb": next((s["graph_pool_gb"] for s in reversed(steps) if s["graph_pool_gb"]), None),
            "steps": steps,
        }
        with open(self.out_results.get_path(), "wt", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
