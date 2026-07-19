"""
RETURNN experiments and benchmarks (frontend features etc.).

Currently here:

Packed (ragged) tensor storage in the RETURNN frontend
(:mod:`returnn.frontend._packed_backend`, ``PackedBackend``, ``rf.pack``):
benchmarks packed vs padded training steps (fwd + bwd),
step time and peak GPU memory,
for a Conformer encoder (default rel-pos attention + BatchNorm)
and a Transformer AED (with label-wise CE loss).

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

from typing import Any, Dict, Optional, Sequence

from sisyphus import Job, Task, tk


_SEQ_LENS_PRESETS = {
    "realistic": (
        [3200, 1600, 1544, 1388, 1200, 1112, 988, 924, 876, 812, 768, 700, 644, 592, 512, 456]
        + [1500, 1400, 1300, 1248, 1148, 1048, 1000, 948, 900, 848, 800, 748, 700, 648, 600, 400]
    ),
    "no_padding": [1000] * 32,
}


def py():
    """Sisyphus entry point."""
    for model in ["conformer", "aed"]:
        for lens_name, lens in _SEQ_LENS_PRESETS.items():
            job = PackedVsPaddedBenchmarkJob(model=model, seq_lens=lens)
            tk.register_output(f"returnn/packed-bench-{model}-{lens_name}.json", job.out_results)


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
        expected_attention_path: Optional[str] = None,
    ):
        """
        :param model: "conformer" or "aed"
        :param seq_lens: input seq lens (feature frames for the conformer,
            source tokens = frames/4 and target tokens = frames/30 for the aed)
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
            expected_attention_path = {"aed": "flash", "conformer": "rel_pos_flex"}[model]
        self.expected_attention_path = expected_attention_path
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

        from returnn.tensor import Tensor, Dim
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
                assert counts and set(counts) == {self.expected_attention_path}, (
                    f"packed attention ran {counts}, expected only {self.expected_attention_path!r}"
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
