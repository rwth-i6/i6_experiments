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

from typing import Any, Dict, Optional, Sequence, Union

from sisyphus import Job, Task, tk
from i6_core.returnn.config import ReturnnConfig


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
    py_aed_graphc_bench()
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
    # ep-2 NaN bisect (graphc replay NaN'd at ep 2 step 171, ctc exactly 0 + ce nan):
    # rerun ep 2 from the ep-1 checkpoint in packed EAGER mode --
    # NaN there too = data/model numerics, clean = captured-graph state bug
    pdec_train_job = pdec_exp.get_training_job()
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_eager",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-eager-bisect.json", job.out_results)
    # eager: CLEAN through all 2015 ep-2 steps (2026-07-29) => replay-path bug.
    # Stage 2: compiled WITHOUT capture -- splits traced-program vs capture/replay
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-bisect.json", job.out_results)
    # compiled-no-capture: SAME nan fingerprint at ep 2 step 4 (2026-07-29)
    # => the bug is in the traced program, capture exonerated.
    # Stage 3: rerun with the batch dump -- hands over the exact failing batch
    # for offline eager-vs-compiled diffing
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nandump",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
        # v2: the first run finished without preserving the dump (train cwd cleaned) -- rerun
        version=2,
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-nandump.json", job.out_results)
    # nandump verdict (2026-07-29): trap fired at compiled call 19, batch preserved;
    # single-batch repro from the ep-1 ckpt is FINITE eager AND compiled (4 calls)
    # => the batch alone does not trigger it; state/RNG-dependent
    # (also the failing step varies across identical runs: step 4 vs 19).
    # Stage 4: Inductor nan-asserts -- the first failing buffer assert names the op
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nanassert",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
        # v2: with the flash empty-row guard + zero-init in place,
        # the next nan-assert names the first REMAINING non-finite buffer
        # (iterate: guard benign masked-lane sources until the real call-19 producer is named)
        version=2,
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-nanassert.json", job.out_results)
    # nanassert verdict (2026-07-30): fires on buf2058 = flash-varlen decoder self-att output,
    # NaN rows for empty segments (fillers/gaps) = EXPECTED masked-lane NaN, false positive;
    # zero-init did not change it (computed, not garbage).
    # Stage 5: full nandump run WITH zero-init, WITHOUT asserts --
    # ep-2 NaN gone = uninitialized reads; still there = a masked-lane NaN escapes a fusion
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nandump_zeroinit",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-nandump-zeroinit.json", job.out_results)
    # zeroinit verdict (2026-07-30): NaN STILL at exactly call 19 -> deterministic, computed,
    # not uninitialized reads. => masked-lane NaN escape (flash empty-row NaN x fusion).
    # Stage 6: validate the source guard (nan_to_num on the flash varlen output under
    # static traceable, see _sdpa_varlen_attention): full run must now be clean
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nandump",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
        version=3,
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-nanguard.json", job.out_results)
    # nanguard verdict (2026-07-30): STILL non-finite at call 19 -> flash tail exonerated.
    # _copy_in audit found a pinned-staging reuse RACE (next host write vs in-flight async H2D)
    # + stale buffer regions when batch extents shrink (sorted_reverse!).
    # Stage 8: validate the event-guard + stale-zeroing fixes -- full run must be clean
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nandump",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
        version=4,
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-copyfix.json", job.out_results)
    # copyfix verdict: STILL call 19 (fixes kept as hardening); nanassert-v2 blocked:
    # the assert checks the raw flash output BEFORE the nan_to_num guard node runs.
    # Stage 9: NaN-count report per buffer per call, run through call 19,
    # diff the reports call-18-vs-19 -> the changing buffer is the culprit
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nanreport",
        num_steps=25,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
        # v2: with the failing flash-backward operand dumps (see the mode string)
        version=2,
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-nanreport.json", job.out_results)
    # ROOT CAUSE (from the operand dumps): 92 filler segments with q>=1 (BOS/EOS row)
    # but k=0 in cross-att -> flash lse=-inf -> backward exp(qk - -inf) = NaN,
    # sporadically poisoning the layer's grads. Fix: clamp the trailing cu_q to the
    # real total so filler segments are zero-length (see _sdpa_varlen_attention).
    # Stage 10: full ep-2 validation with the fix -- must run all ~2015 steps clean
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nandump",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
        # v6: v5 exposed that the ORIGINAL cu_q assigned the whole buffer slack to the
        # last segment (flash wrote those rows); with the clamp they stay unwritten
        # garbage (finite, nan_to_num-immune) -> zero all rows beyond the covered region
        version=6,
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-emptykvfix.json", job.out_results)
    # v6 verdict: NaN at call 4 -- the failing call MOVES with every codegen layout
    # (19 / 1 / 4) = miscompiled-fusion signature, not a semantics issue.
    # Stage 12: conservative codegen (no epilogue fusion / pattern matcher):
    # clean full run = miscompiled fusion confirmed (and a usable mitigation)
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nofuse",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-nofuse.json", job.out_results)
    # nofuse verdict: STILL NaN (call 2) -- deeper than epilogue fusion / pattern matcher.
    # Stage 13: aot-eager (traced graph, eager kernels): splits decomposition-vs-codegen
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_aot_eager",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-aot-eager.json", job.out_results)
    # aot-eager verdict: NaN at call 40 WITHOUT Inductor -> codegen exonerated;
    # the defect is in the bound regime and/or the AOT trace semantics.
    # Stage 14: untraced eager on bound buffers -- the final split
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_eager_bound",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-eager-bound.json", job.out_results)
    # eager-bound verdict: NaN at call 133 UNTRACED PLAIN EAGER on the bound buffers
    # -> RF bound-regime semantics bug, all torch.compile machinery exonerated.
    # Stage 15: same + per-module non-finite forward hooks -> names the RF module
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_eager_bound",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2, "debug_module_nan_hooks": True},
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-eager-bound-hooks.json", job.out_results)
    # hooks verdict: PT-module forward hooks are INERT for RF models (call path bypasses
    # the wrappers, 0 hits); failing call also NON-deterministic across identical runs
    # (133 vs 273: cuDNN autotune / atomic-scatter ordering chaos picks the step).
    # Stage 16: on a non-finite loss, restore params+RNG and REPLAY the step under a
    # TorchDispatchMode NaN tracer -> raises at the exact producing aten op
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_eager_bound",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
        # v3: the tracer tripped on aten.empty (uninitialized by design) -- allowlisted
        # v4: v3 revealed the poisoning happens ONE STEP EARLIER (NaN grads -> in-step
        # optimizer writes NaN params; loss check fires a step late; replay traced a
        # poisoned conv-1 bias cast) -- now ALL outs (incl grads) are checked
        # v5: v4's replay tripped on detach of the benign filler-NaN flash rows AND the
        # flash fallback swallowed the tracer -> alias/copy ops allowlisted, tracer re-raised
        # v6: v5 TRACED IT: NaN grads out of the flash cross-att BACKWARD (dq [15200,8,64],
        # first checked op aten.abs inside autograd.grad) = the empty-KV/lse mechanism.
        # Fix: cu_q tail clamp + out-row zeroing + Q INPUT PRE-MASK (the where's backward
        # zeroes the uncovered dq rows whatever flash leaves in its internal buffer).
        # This run must go clean through the full ep 2.
        # v7: v6 STILL fired (same abs signature) -> the CAUSAL self-att slack:
        # cu_seqlens padded its last entry to the BOUND (get_dim_value = capacity under
        # static tracing), so the last segment attends the garbage slack rows.
        # Root fix: cu total = real device lens sum; unified q/k/v slack pre-mask for
        # causal AND cross att.
        # v7 VERDICT: CLEAN, all 2015 ep-2 steps (was: NaN within <300 every prior run).
        version=7,
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-eager-bound-nantrace.json", job.out_results)
    # Stage 22: the same fix validated in the PRODUCTION path (Inductor-compiled)
    job = TrainStepBenchmarkJob(
        returnn_config=pdec_train_job.returnn_config,
        mode="packed_compiled_nandump",
        num_steps=2200,
        load_checkpoint=pdec_train_job.out_checkpoints[1].path,
        config_overrides={"num_epochs": 2},
        version=8,
    )
    tk.register_output("returnn/aed-graphc-v3-pdec-ep2-compiled-slackfix.json", job.out_results)


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


def py_aed_graphc_loquacious():
    """
    Loquacious ~500M CTC+AED base (16L Conformer 1024d + 6L Transformer dec 1024d, spm10k),
    trained with graphc (packed collate + Inductor compile + whole-step CUDA graph),
    cloned from :func:`i6_experiments.users.zeyer.experiments.exp2026_05_26_base_fzj.train` ("base").

    Capacity notes (vs the LS 160M runs):
    - data: max_seq_length_default_input = 19.5s = 312_000 samples, NO speed perturbation
      -> dim_capacity 312_960 (multiple of 960) is provably sufficient.
    - text: no target-len filter in this config,
      so the capacity must cover the measured spm10k target-len max of the large subset
      (a too-small value raises loudly in graph-capture _copy_in).
    """
    from i6_experiments.users.zeyer.experiments.exp2026_05_26_base_fzj import train as loq_train

    gap, align = _aed_graphc_packed_gap, _aed_graphc_packed_align
    # measured on the train shards (spm10k SZcvHsG1gYNM),
    # CONDITIONED on the 19.5s audio filter (8.96M of 9.49M seqs pass):
    # max 246, p99.99 = 86, p99 = 66
    # (the unconditional max 366 has long audio and gets filtered out);
    # 256 = conditional max + headroom.
    classes_cap = 256
    packed_total = 100_000 * _loq_batch_size_factor() + 200 * (gap + align)
    packed_total = -(-packed_total // align) * align
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
    # tighter-audio-gap variant, probing the fix for the 808MiB OOM
    # (FastBaumWelchPackedOp CTC buffer at the bound [20667, 10241] f32):
    # gap 18240 is LS-sized (within-batch length spread there),
    # gap 960 shrinks the audio bound 19.84M -> 16.38M samples,
    # i.e. -17% on every bound-sized encoder activation;
    # the bench also shows whether a 1-frame gap breaks anything (losses / asserts)
    job = TrainStepBenchmarkJob(
        returnn_config=cfg,
        mode="packed_graphc",
        num_steps=31,
        config_overrides={
            "packed_tensors": {
                "per_key": {
                    "audio": {"gap": 960, "align": 960},
                    "text": {"gap": 2, "align": 1},
                }
            },
            "torch_cuda_graph": {
                "batch_size_bound": 200,
                "dim_capacity": {"audio": 312_960, "text": classes_cap},
                "packed_total_bound": {"audio": 16_384_000, "text": 200 * (classes_cap + 2)},
                "warmup_steps": 2,
                "capture_optimizer": True,
                "compile": True,
                # the -17% audio bound moved the OOM margin only ~0.4GB,
                # so bound-sized activations do NOT dominate the ~78GB compile-run peak;
                # the snapshot (dumped on OOM) names every allocation at the peak
                "dump_memory_snapshot": True,
                # the aten-level FX dump names the producer of the surviving
                # [text_bound, vocab] scatter (scatter_add_53/54 in the wrapper);
                # the generated-code node names alone were ambiguous
                "dump_fx_dir": "fx-dump",
            },
        },
    )
    tk.register_output("returnn/loq-base-graphc-bench-packed_graphc-gap960.json", job.out_results)


def _loq_batch_size_factor():
    """the raw-sample batch-size factor of the baseline configs"""
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import _batch_size_factor

    return _batch_size_factor


def py_aed_graphc_bench():
    """
    Realistic per-mode train-step benchmark + numerical parity check
    on the graphc-v2 training config.
    """
    # The config file is referenced by absolute path, not via the training job output:
    # the checks (finished jobs) ran against the exact config of the then-running v2 job,
    # without pulling that job into the dependency graph.
    v2_config = tk.Path(
        "/home/az668407/setups/combined/2021-05-31/work/i6_core/returnn/training"
        "/ReturnnTrainingJob.erqk3HOebeeL/output/returnn.config"
    )
    for mode in ["padded_eager", "packed_eager", "packed_compiled", "packed_graphc"]:
        job = TrainStepBenchmarkJob(returnn_config_file=v2_config, mode=mode)
        tk.register_output(f"returnn/aed-graphc-train-bench-{mode}.json", job.out_results)
        # specaugment forced on from step 0:
        # the static-traceable specaugment num-masks path was the convergence-regression cause
        # (capacity-scaled masking);
        # the normal schedule keeps specaugment off within the 300 bench steps,
        # so these variants are the ones that verify the fix (all modes must match)
        job = TrainStepBenchmarkJob(
            returnn_config_file=v2_config, mode=mode, config_overrides={"specaugment_steps": (0, 0, 0)}
        )
        tk.register_output(f"returnn/aed-graphc-train-bench-{mode}-specaugOn.json", job.out_results)
    # The Loquacious memory/NaN diagnostic jobs that used to be registered here
    # (referencing a generated config file via a raw tk.Path -- wrong, never do that)
    # are removed; their result evidence stays in the work dir and the project notes.
    # The proper benchmark set for the Loquacious experiment
    # gets wired via the ReturnnConfig object of the registered experiment
    # (see py_aed_graphc_loquacious).


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
        yield Task("run", rqmt=self.rqmt)

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
        if self.load_checkpoint is not None:
            # continue-training repro: place the checkpoint (+ optimizer state) as epoch 1
            # of the local model dir, so the engine natively resumes at epoch 2
            # (a bare `load` makes the engine look for the optimizer state in the local dir)
            src = self.load_checkpoint.get_path()
            for src_path, link_name in [(src, "epoch.001.pt"), (src[: -len(".pt")] + ".opt.pt", "epoch.001.opt.pt")]:
                link_path = work + "/models/" + link_name
                if os.path.exists(src_path) and not os.path.exists(link_path):
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
        cfg += (
            "\n\n# ---- TrainStepBenchmarkJob overrides ----\n"
            f"model = {work + '/models/epoch'!r}\n"
            f"learning_rate_file = {work + '/learning_rates'!r}\n"
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

        step_re = re.compile(r"ep \d+ train, step (\d+), (.*?), mem_usage:cuda [0-9.]+GB(?:, ([0-9.]+) sec/step)?")
        steps = []
        with open(log_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = step_re.search(line)
                if not m:
                    continue
                losses = {}
                for part in m.group(2).split(", "):
                    name, _, value = part.rpartition(" ")
                    try:
                        losses[name] = float(value)
                    except ValueError:
                        pass
                steps.append(
                    {
                        "step": int(m.group(1)),
                        "losses": losses,
                        "sec_per_step": float(m.group(3)) if m.group(3) else None,
                    }
                )
        # a crashed run can still leave a few parsed steps (e.g. OOM after warmup);
        # a deadline-killed healthy run has close to num_steps
        assert len(steps) >= min(self.num_steps, 10), f"only {len(steps)} train steps parsed from {log_path}"
        times = sorted(s["sec_per_step"] for s in steps if s["sec_per_step"] is not None and s["step"] >= 5)
        res = {
            "mode": self.mode,
            "num_steps_parsed": len(steps),
            "median_sec_per_step": times[len(times) // 2] if times else None,
            "steps": steps,
        }
        with open(self.out_results.get_path(), "wt", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
