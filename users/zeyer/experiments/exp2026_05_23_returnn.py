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


_SEQ_LENS_PRESETS = {
    "realistic": (
        [3200, 1600, 1544, 1388, 1200, 1112, 988, 924, 876, 812, 768, 700, 644, 592, 512, 456]
        + [1500, 1400, 1300, 1248, 1148, 1048, 1000, 948, 900, 848, 800, 748, 700, 648, 600, 400]
    ),
    "no_padding": [1000] * 32,
}

# raw-audio sample counts (16 kHz) for the "real" model, which packs the raw audio and runs
# the log-mel front-end packed too. 16 seqs, 2..17.5 s.
_AUDIO_LENS_PRESETS = {
    "random": [278531, 41017, 95000, 201337, 64001, 156789, 36666, 249999,
               55555, 121212, 78123, 180001, 32003, 226667, 49999, 143210],
    "sorted": [278531, 271113, 265002, 258888, 254321, 249999, 245005, 241777,
               237500, 233333, 230001, 226667, 223456, 220000, 216789, 213001],
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
            this catches it. Default per model: "flash" (aed) / "rel_pos_flex" (conformer).
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
                "conformer": ["rel_pos_flex"],
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
            Tensor("audio_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(audio_lens, dtype=torch.int32))
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
