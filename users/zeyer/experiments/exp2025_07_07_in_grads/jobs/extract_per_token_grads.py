"""Per-token variant of :class:`ExtractInGradsFromModelJob`."""

from typing import Optional, List
from sisyphus import Task
from .extract_in_grad_scores import ExtractInGradsFromModelJob


import contextlib

# Set once per run() to self.amp_attn_fp32 (process-local; one job = one config).
_AMP_ATTN_FP32 = {"v": False}


@contextlib.contextmanager
def _amp_ctx(amp_dtype):
    """autocast(amp_dtype) for the forward graph. If _AMP_ATTN_FP32 is set, additionally force the
    scaled-dot-product attention core (Q*K, softmax, attn*V) to f32 while projections / FFN / conv
    stay in amp_dtype -- a probe for whether the bf16 attention matmuls are what cost WBE."""
    import torch

    if amp_dtype is None:
        yield
        return
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[amp_dtype]
    import torch.nn.functional as F

    _orig_sdpa = F.scaled_dot_product_attention
    if _AMP_ATTN_FP32["v"]:

        def _sdpa_f32(q, k, v, *a, **k_):
            odt = q.dtype
            with torch.autocast("cuda", enabled=False):
                o = _orig_sdpa(q.float(), k.float(), v.float(), *a, **k_)
            return o.to(odt)

        F.scaled_dot_product_attention = _sdpa_f32
    try:
        with torch.autocast("cuda", dtype=dt):
            yield
    finally:
        F.scaled_dot_product_attention = _orig_sdpa


def _amp_forward(model, amp_dtype, **kw):
    """Forward under AMP (see :func:`_amp_ctx`). The per-token ``autograd.grad`` is taken OUTSIDE
    this context (backward replays the recorded forward casts)."""
    with _amp_ctx(amp_dtype):
        return model(**kw)


class ExtractInGradsPerTokenJob(ExtractInGradsFromModelJob):
    """Per-token version of :class:`ExtractInGradsFromModelJob`.

    The base class computes one backward per word, where the per-word loss is
    ``sum_{t in word} log p(target_t)``. This subclass instead computes one
    backward per **token** (= subword position within a word), yielding K grad
    maps per word where K is the number of subwords/tokens that word spans.

    Output HDF schema differs from the base:

    - ``inputs`` is a flat ``[total_tokens * num_input_frames]`` array (one row
      per subword position), instead of ``[total_words * num_input_frames]``.
    - new stream ``num_tokens_per_word``: per chunk per word, the count of
      tokens that word contains. Lets downstream code regroup tokens to words.
    - ``log_probs_per_word`` is now stored per token, not per word, so its
      length is also ``total_tokens`` (was ``total_words``).

    Same constructor as the base; same ``mult_grad_by_inputs`` /
    ``attr_reduction`` semantics, applied per token. Cost is roughly K x more
    backward passes per word (forward pass is shared via ``retain_graph=True``).
    """

    # noise_n_samples=1 / noise_std=0.0 reproduce the pre-SmoothGrad single-pass behaviour exactly,
    # so exclude them at their defaults: existing jobs keep their hash.
    # Do NOT bump __sis_version__ for a behaviour-preserving optional kwarg --
    # that would needlessly re-run every existing job.
    # Only the SmoothGrad variants (noise_std != 0) differ from the defaults, and thus get a distinct hash.
    __sis_hash_exclude__ = {
        "noise_n_samples": 1,
        "noise_std": 0.0,
        "ig_steps": 1,
        "vargrad": False,
        "eg_steps": 1,
        "eg_baseline_std": 0.0,
        "target_source": "word_detail",
        "batched_backward": False,
        "seq_batch_size": 1,
        "amp_dtype": None,
        "amp_attn_fp32": False,
    }

    def __init__(
        self,
        *,
        noise_n_samples: int = 1,
        noise_std: float = 0.0,
        ig_steps: int = 1,
        vargrad: bool = False,
        eg_steps: int = 1,
        eg_baseline_std: float = 0.0,
        target_source: str = "word_detail",
        batched_backward: bool = False,
        seq_batch_size: int = 1,
        amp_dtype: Optional[str] = None,
        amp_attn_fp32: bool = False,
        **kwargs,
    ):
        """Extends the base with SmoothGrad-style noise averaging.

        :param noise_n_samples: number of noisy forward+backward passes per seq
            to average. 1 = standard (no noise averaging).
        :param noise_std: standard deviation of Gaussian noise added to the
            raw audio waveform before each forward pass. 0.0 = no noise.
            Typical useful range: 0.01-0.1 (audio waveforms are in [-1, 1]).
        :param batched_backward: compute the per-token grads of a word in one vmapped backward
            (autograd.grad with is_grads_batched=True) instead of K sequential backwards.
            Mathematically identical; faster where the shared graph dominates. Off by default
            because vmap-incompatible backward kernels (some fused/custom ops) would raise.
        :param seq_batch_size: forward this many sequences per model call
            and compute each backward for one token of every sequence simultaneously
            (their grads live in separate batch rows; the models have no cross-seq interaction).
            Mathematically identical to the default per-seq loop; amortizes both passes.
            Requires a model adapter with B>1 forward support (e.g. Wav2Vec2Ctc).
            Composes with ``batched_backward`` (vmap over tokens on the batched graph).
        """
        super().__init__(**kwargs)
        self.noise_n_samples = int(noise_n_samples)
        self.noise_std = float(noise_std)
        self.ig_steps = int(ig_steps)
        self.vargrad = bool(vargrad)
        self.eg_steps = int(eg_steps)
        self.eg_baseline_std = float(eg_baseline_std)
        self.target_source = str(target_source)
        self.batched_backward = bool(batched_backward)
        self.seq_batch_size = int(seq_batch_size)
        assert self.seq_batch_size >= 1
        self.amp_dtype = amp_dtype
        assert self.amp_dtype in (None, "bfloat16", "float16"), self.amp_dtype
        self.amp_attn_fp32 = bool(amp_attn_fp32)
        assert self.target_source in ("word_detail", "phonetic_detail"), self.target_source
        assert sum(x > 1 for x in (self.ig_steps, self.eg_steps)) <= 1, "ig_steps / eg_steps exclusive"
        if self.eg_steps > 1:
            self.rqmt = dict(self.rqmt, time=self.rqmt["time"] * self.eg_steps)
        assert self.noise_n_samples >= 1
        assert self.noise_std >= 0.0
        assert self.ig_steps >= 1
        assert not (self.ig_steps > 1 and (self.noise_n_samples > 1 or self.noise_std > 0.0)), (
            "ig_steps is mutually exclusive with SmoothGrad noise"
        )
        if self.noise_n_samples > 1:
            self.rqmt = dict(self.rqmt, time=self.rqmt["time"] * self.noise_n_samples)
        if self.ig_steps > 1:
            self.rqmt = dict(self.rqmt, time=self.rqmt["time"] * self.ig_steps)

    rqmt = {"time": 100, "cpu": 2, "gpu": 1, "mem": 125}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        # Back-compat: instances pickled before these optional attrs existed lack them on unpickle
        # (e.g. the IG job was created before `vargrad` was added).
        for _a, _d in (
            ("noise_n_samples", 1),
            ("noise_std", 0.0),
            ("ig_steps", 1),
            ("vargrad", False),
            ("eg_steps", 1),
            ("eg_baseline_std", 0.0),
            ("target_source", "word_detail"),
            ("batched_backward", False),
            ("seq_batch_size", 1),
            ("amp_dtype", None),
            ("amp_attn_fp32", False),
        ):
            if not hasattr(self, _a):
                setattr(self, _a, _d)
        import os
        import sys
        import time
        import gc

        from i6_experiments.users.zeyer.external_models.huggingface import (
            set_hf_offline_mode,
            get_content_dir_from_hub_cache_dir,
        )
        from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        print("Import Torch, Numpy...")
        start_time = time.time()

        import numpy as np
        import torch

        print(f"({time.time() - start_time} secs)")

        from returnn.torch.util import diagnose_gpu

        # Probe torch.cuda.is_available() early, under a timeout.
        # On a wedged-GPU node it otherwise hangs for hours
        # (e.g. inside transformers' flash-attn check during make_model),
        # holding a GPU slot for nothing.
        # Crash fast (SIGABRT) on timeout instead. Mirrors returnn.__main__.
        with diagnose_gpu.timeout("torch.cuda.is_available()"):
            torch.cuda.is_available()

        from returnn.util import better_exchook
        from returnn.datasets.hdf import SimpleHDFWriter
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
        from i6_experiments.users.zeyer.torch.batch_gather import batches_gather
        from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats

        better_exchook.install()

        try:
            # noinspection PyUnresolvedReferences
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            pass

        from .attr_reduction import get_attr_reduce_func

        attr_reduce_func = get_attr_reduce_func(self.attr_reduction)

        from .models import make_model, ForwardOutput

        device_str = "cuda"
        dev = torch.device(device_str)

        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)

        for p in model.parameters():
            p.requires_grad = False

        # AMP: log_probs recomputes logits from saved hidden -> wrap it too so the score path
        # matches the forward. Instance-patch persists into _run_seq_batched (same model object).
        _AMP_ATTN_FP32["v"] = self.amp_attn_fp32
        if self.amp_dtype is not None:
            _orig_log_probs = model.log_probs
            _amp_dtype = self.amp_dtype

            def _amp_log_probs(*a, **k):
                with _amp_ctx(_amp_dtype):
                    return _orig_log_probs(*a, **k)

            model.log_probs = _amp_log_probs

        report_dev_memory_stats(dev)

        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(),
            # grads: [num_chunks * ~chunk_num_tokens * ~chunk_num_input_frames, 1]
            dim=1,
            ndim=2,
            extra_type={
                "audio_frames_start_end": (2, 2, "int32"),  # [num_chunks * ~chunk_num_input_frames, 2]
                "num_input_frames": (1, 2, "int32"),  # [num_chunks, 1]
                "num_words": (1, 2, "int32"),  # [num_chunks, 1]
                "num_tokens": (1, 2, "int32"),  # [num_chunks, 1]
                # per chunk per word, how many tokens that word contains
                "num_tokens_per_word": (1, 2, "int32"),  # [num_chunks * ~chunk_num_words, 1]
                # For debugging / verification. Per-token now, not per-word.
                "log_probs_per_token": (1, 2, "float32"),  # [num_chunks * ~chunk_num_tokens, 1]
                "exit_log_probs": (1, 2, "float32"),  # [num_chunks, 1]
            },
        )

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Dataset keys:", ds.keys())
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

        from returnn.datasets.hdf import HDFDataset

        chunk_segmentation_hdf_ds: Optional[HDFDataset] = None
        if self.chunk_segmentation_hdf is not None:
            chunk_segmentation_hdf_ds = HDFDataset([self.chunk_segmentation_hdf.get_path()])
            chunk_segmentation_hdf_ds.initialize()
            chunk_segmentation_hdf_ds.init_seq_order(epoch=1)

        if self.seq_batch_size > 1:
            assert chunk_segmentation_hdf_ds is None, "seq_batch_size: chunking not supported"
            assert self.noise_n_samples == 1 and self.noise_std == 0.0, "seq_batch_size: SmoothGrad not supported"
            assert self.ig_steps == 1 and self.eg_steps == 1, "seq_batch_size: IG/EG not supported"
            self._run_seq_batched(
                model=model,
                ds=ds[self.dataset_key],
                hdf_writer=hdf_writer,
                attr_reduce_func=attr_reduce_func,
                dev=dev,
            )
            hdf_writer.close()
            return

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = data["audio"]["array"]
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            samplerate = data["audio"]["sampling_rate"]
            words = data[self.target_source]["utterance"]
            transcription = " ".join(words)
            print(f"seq {seq_idx}, {audio.shape=}, {samplerate=}, {transcription!r}")
            num_words = len(words)
            if num_words == 0:
                # Empty recog (0 words): there is nothing to extract per-token.
                # We must NEVER skip a seq -- every seq gets an entry, so the HDF stays
                # index-aligned and no seq is ever missing downstream (a missing entry is a bug).
                # Write a valid 0-token entry: 0 grad rows, 0 chunks, every stream empty.
                # WordAlignFromPerTokenGradsJob detects num_tokens.size==0 and emits a 0-row
                # boundary for it. (Forced-mode references are never empty; this is a hyp-mode path.)
                hdf_writer.insert_batch(
                    np.zeros((1, 0, 1), dtype="float32"),
                    seq_len=[0],
                    seq_tag=[f"seq-{seq_idx}"],
                    extra={
                        "audio_frames_start_end": np.zeros((1, 0, 2), dtype="int32"),
                        "num_input_frames": np.zeros((1, 0, 1), dtype="int32"),
                        "num_words": np.zeros((1, 0, 1), dtype="int32"),
                        "num_tokens": np.zeros((1, 0, 1), dtype="int32"),
                        "num_tokens_per_word": np.zeros((1, 0, 1), dtype="int32"),
                        "log_probs_per_token": np.zeros((1, 0, 1), dtype="float32"),
                        "exit_log_probs": np.zeros((1, 0, 1), dtype="float32"),
                    },
                )
                continue
            assert len(transcription.split(" ")) == num_words

            if seq_idx == 0:
                print("data keys:", data.keys())

            if chunk_segmentation_hdf_ds is not None:
                chunk_segmentation_hdf_ds.load_seqs(seq_idx, seq_idx + 1)
                chunk_audio_start_end = chunk_segmentation_hdf_ds.get_data(seq_idx, "audio_chunk_start_end")
                chunk_words_indices_start_end = chunk_segmentation_hdf_ds.get_data(seq_idx, "data")
                num_chunks = chunk_audio_start_end.shape[0]
                assert num_chunks == chunk_words_indices_start_end.shape[0]
                assert chunk_words_indices_start_end[:, 1].max() == len(words)
            else:
                num_chunks = 1
                chunk_audio_start_end = np.array([[0, len(audio)]], dtype=np.int32)
                chunk_words_indices_start_end = np.array([[0, num_words]], dtype=np.int32)

            # Per chunk we collect:
            grad_mat: List[torch.Tensor] = []  # flattened per-token grads
            audio_frames_start_end: List[torch.Tensor] = []
            num_input_frames: List[int] = []
            num_words_: List[int] = []
            num_tokens_: List[int] = []  # total tokens per chunk
            num_tokens_per_word_: List[int] = []  # per word, flat across chunks
            log_probs_per_token: List[torch.Tensor] = []  # per token, flat
            exit_log_probs: List[torch.Tensor] = []

            for chunk_idx in range(num_chunks):
                start_time = time.time()
                audio_start, audio_end = chunk_audio_start_end[chunk_idx]
                audio_start, audio_end = max(audio_start, 0), min(audio_end, len(audio))
                words_start, words_end = chunk_words_indices_start_end[chunk_idx]
                words_start, words_end = max(words_start, 0), min(words_end, num_words)
                print(
                    f"** Forwarding chunk {chunk_idx + 1}/{num_chunks}"
                    f" audio {audio_start / samplerate}-{audio_end / samplerate}"
                    f" words {words_start}-{words_end} ({words[words_start:words_end]!r})"
                )

                # noinspection PyShadowingNames
                def _calc_log_probs_and_per_token_grads(
                    w: int,
                    *,
                    report_mem: bool = False,
                    forward_output: ForwardOutput,
                    no_grad: bool = False,
                    raw_grad: bool = False,
                ):
                    t0, t1 = forward_output.target_start_end[:, w].unbind(1)  # [B], [B]
                    loss = model.log_probs(forward_output=forward_output, start=t0, end=t1)  # [B, t1-t0, V]
                    targets = batch_slice(forward_output.targets, (t0, t1))  # [B, t1-t0]
                    loss = batches_gather(loss, indices=targets, num_batch_dims=2)  # [B, t1-t0]
                    # Mask out beyond word boundary.
                    mask = torch.arange(loss.shape[1], device=loss.device)[None, :] >= (t1 - t0)[:, None]
                    loss.masked_fill_(mask, 0.0)
                    num_tokens = int((t1 - t0).max())  # B=1 in practice
                    if no_grad:
                        return loss.detach(), num_tokens, None

                    # K backwards: one per token. Forward graph is shared via retain_graph=True.
                    # With batched_backward, the K vector-Jacobian products are computed in a single
                    # vmapped autograd.grad (is_grads_batched=True): mathematically identical to the K
                    # sequential backwards, but the shared graph is traversed once. grad_outputs[k] is
                    # a one-hot over loss's token dim, selecting token k (matching loss[:, k].sum()).
                    attrs_per_token: List[torch.Tensor] = []

                    def _reduce_grad(grad):
                        with torch.no_grad():
                            attr = batch_slice(grad.float(), forward_output.input_slice_start_end)  # [B, T, F]
                            if not raw_grad:
                                if self.mult_grad_by_inputs:
                                    e = batch_slice(forward_output.inputs.float(), forward_output.input_slice_start_end)
                                    attr = attr * e
                                attr = attr_reduce_func(attr)  # [B, T]
                        return attr  # [B,T,F] if raw_grad else [B,T]

                    if self.batched_backward and num_tokens > 1:
                        v = torch.zeros((num_tokens,) + loss.shape, device=loss.device, dtype=loss.dtype)
                        for k in range(num_tokens):
                            v[k, :, k] = 1.0  # one-hot per token, summed over B like loss[:, k].sum()
                        (grads,) = torch.autograd.grad(
                            loss, forward_output.inputs, grad_outputs=v, is_grads_batched=True, retain_graph=True
                        )  # [num_tokens, *inputs.shape]
                        for k in range(num_tokens):
                            attrs_per_token.append(_reduce_grad(grads[k]))
                    else:
                        for k in range(num_tokens):
                            single_loss = loss[:, k].sum()  # scalar
                            (grad,) = torch.autograd.grad(single_loss, forward_output.inputs, retain_graph=True)
                            attrs_per_token.append(_reduce_grad(grad))

                    if report_mem:
                        report_dev_memory_stats(dev)

                    return loss.detach(), num_tokens, attrs_per_token

                # SmoothGrad: run noise_n_samples forward+backward passes,
                # accumulate per-token grad attrs, average at the end.
                # When noise_n_samples==1 and noise_std==0.0 this is identical
                # to the original single-pass behaviour.
                _raw_audio_chunk = torch.tensor(audio[audio_start:audio_end])
                _noise_n = self.noise_n_samples
                _noise_std = self.noise_std
                # We store attrs accumulated across samples for each word/token.
                # attrs_accum[w][k] = sum of attr tensors for word w, token k.
                _attrs_accum: List[Optional[List[Optional[torch.Tensor]]]] = []
                _attrs_sq: List[Optional[List[Optional[torch.Tensor]]]] = []  # sum of squares, for VarGrad
                for _sample_idx in range(_noise_n):
                    if _noise_std > 0.0:
                        _noisy = _raw_audio_chunk + torch.randn_like(_raw_audio_chunk) * _noise_std
                    else:
                        _noisy = _raw_audio_chunk
                    _fwd: ForwardOutput = _amp_forward(
                        model,
                        self.amp_dtype,
                        raw_inputs=_noisy[None],
                        raw_inputs_sample_rate=samplerate,
                        raw_input_seq_lens=torch.tensor([audio_end - audio_start]),
                        raw_targets=[words[words_start:words_end]],
                        raw_target_seq_lens=torch.tensor([words_end - words_start]),
                        omitted_prev_context=torch.tensor([words_start]),
                    )
                    # On first sample, record frame count and structure.
                    if _sample_idx == 0:
                        forward_output = _fwd
                    for _w in range(words_end - words_start):
                        _, _n_tok, _attrs = _calc_log_probs_and_per_token_grads(
                            _w,
                            forward_output=_fwd,
                            report_mem=(_w == 0 and _sample_idx == 0),
                        )
                        if _sample_idx == 0:
                            _attrs_accum.append([_attrs[_k].clone() if _attrs else None for _k in range(_n_tok)])
                            if self.vargrad:
                                _attrs_sq.append(
                                    [(_attrs[_k] ** 2).clone() if _attrs else None for _k in range(_n_tok)]
                                )
                        else:
                            for _k in range(_n_tok):
                                if _attrs_accum[_w][_k] is not None and _attrs:
                                    _attrs_accum[_w][_k] = _attrs_accum[_w][_k] + _attrs[_k]
                                    if self.vargrad:
                                        _attrs_sq[_w][_k] = _attrs_sq[_w][_k] + _attrs[_k] ** 2
                # Average accumulated attrs.
                if _noise_n > 1:
                    for _w in range(len(_attrs_accum)):
                        for _k in range(len(_attrs_accum[_w])):
                            if _attrs_accum[_w][_k] is not None:
                                if self.vargrad:
                                    # VarGrad: per-frame STD of the saliency across the noise samples.
                                    _mean = _attrs_accum[_w][_k] / _noise_n
                                    _var = _attrs_sq[_w][_k] / _noise_n - _mean**2
                                    _attrs_accum[_w][_k] = torch.sqrt(torch.clamp(_var, min=0.0) + 1e-12)
                                else:
                                    _attrs_accum[_w][_k] = _attrs_accum[_w][_k] / _noise_n
                # Wrap accumulated attrs so existing code can consume them.
                _attrs_accum_flat = _attrs_accum  # indexed [w][k]

                if self.ig_steps > 1:
                    # Integrated Gradients: integrate the per-token RAW grad along the audio-amplitude
                    # path baseline(0)->x (alpha=k/N), average, multiply by the leaf input at x (baseline
                    # 0), then reduce. forward_output above is the full-x forward, reused as the leaf.
                    # Exact IG for grad_wrt='raw_waveform'; amplitude-path saliency integral otherwise.
                    _ig_n = self.ig_steps
                    # detach: forward_output.inputs requires grad (it's the grad leaf); the IG attr is a
                    # value, not part of any graph, so detach before it flows into the stored saliency.
                    _leaf_x = batch_slice(forward_output.inputs.float(), forward_output.input_slice_start_end).detach()
                    _grad_sum = {}  # (w, k) -> [B, T, F]
                    for _step in range(_ig_n):
                        _alpha = (_step + 1) / _ig_n
                        _fwd_ig = _amp_forward(
                            model,
                            self.amp_dtype,
                            raw_inputs=(_raw_audio_chunk * _alpha)[None],
                            raw_inputs_sample_rate=samplerate,
                            raw_input_seq_lens=torch.tensor([audio_end - audio_start]),
                            raw_targets=[words[words_start:words_end]],
                            raw_target_seq_lens=torch.tensor([words_end - words_start]),
                            omitted_prev_context=torch.tensor([words_start]),
                        )
                        for _w in range(words_end - words_start):
                            _, _n_tok, _raw = _calc_log_probs_and_per_token_grads(
                                _w, forward_output=_fwd_ig, raw_grad=True, report_mem=(_w == 0 and _step == 0)
                            )
                            for _k in range(_n_tok):
                                _key = (_w, _k)
                                _grad_sum[_key] = _raw[_k] if _key not in _grad_sum else _grad_sum[_key] + _raw[_k]
                    _attrs_accum_flat = []
                    for _w in range(words_end - words_start):
                        _, _n_tok, _ = _calc_log_probs_and_per_token_grads(
                            _w, forward_output=forward_output, no_grad=True
                        )
                        _attrs_accum_flat.append(
                            [attr_reduce_func((_grad_sum[(_w, _k)] / _ig_n) * _leaf_x) for _k in range(_n_tok)]
                        )

                if self.eg_steps > 1:
                    # Expected Gradients: per step a RANDOM noise baseline x' (std eg_baseline_std) and a
                    # random alpha~U(0,1); integrate the raw grad over x'+alpha*(x-x'), avg, x leaf-at-x,
                    # reduce. Amplitude-path approximation for intermediate grad targets (exact raw_waveform).
                    _eg_n = self.eg_steps
                    _leaf_x = batch_slice(forward_output.inputs.float(), forward_output.input_slice_start_end).detach()
                    _grad_sum = {}
                    for _step in range(_eg_n):
                        _xp = torch.randn_like(_raw_audio_chunk) * self.eg_baseline_std
                        _alpha = float(torch.rand(1).item())
                        _x_point = _xp + _alpha * (_raw_audio_chunk - _xp)
                        _fwd_eg = _amp_forward(
                            model,
                            self.amp_dtype,
                            raw_inputs=_x_point[None],
                            raw_inputs_sample_rate=samplerate,
                            raw_input_seq_lens=torch.tensor([audio_end - audio_start]),
                            raw_targets=[words[words_start:words_end]],
                            raw_target_seq_lens=torch.tensor([words_end - words_start]),
                            omitted_prev_context=torch.tensor([words_start]),
                        )
                        for _w in range(words_end - words_start):
                            _, _n_tok, _raw = _calc_log_probs_and_per_token_grads(
                                _w, forward_output=_fwd_eg, raw_grad=True, report_mem=(_w == 0 and _step == 0)
                            )
                            for _k in range(_n_tok):
                                _key = (_w, _k)
                                _grad_sum[_key] = _raw[_k] if _key not in _grad_sum else _grad_sum[_key] + _raw[_k]
                    _attrs_accum_flat = []
                    for _w in range(words_end - words_start):
                        _, _n_tok, _ = _calc_log_probs_and_per_token_grads(
                            _w, forward_output=forward_output, no_grad=True
                        )
                        _attrs_accum_flat.append(
                            [attr_reduce_func((_grad_sum[(_w, _k)] / _eg_n) * _leaf_x) for _k in range(_n_tok)]
                        )

                print("** Calculating grads (or using SmoothGrad-averaged grads)")
                chunk_num_input_frames = forward_output.get_inputs_seq_lens_sliced()[0].item()
                num_input_frames.append(chunk_num_input_frames)
                num_words_.append(words_end - words_start)
                chunk_total_tokens = 0
                for w in range(words_end - words_start):
                    if self.ig_steps > 1 or self.eg_steps > 1 or _noise_n > 1 or _noise_std > 0.0:
                        # Use pre-accumulated (averaged) attrs from the noise loop.
                        word_log_probs, n_tok, _ = _calc_log_probs_and_per_token_grads(
                            w,
                            forward_output=forward_output,
                            no_grad=True,
                            report_mem=w in {0, words_end - words_start - 1},
                        )
                        attrs = _attrs_accum_flat[w]
                    else:
                        word_log_probs, n_tok, attrs = _calc_log_probs_and_per_token_grads(
                            w,
                            forward_output=forward_output,
                            report_mem=w in {0, words_end - words_start - 1},
                        )
                    assert word_log_probs.shape == (1, n_tok), f"got {word_log_probs.shape=} {n_tok=}"
                    for k in range(n_tok):
                        assert attrs[k].shape == (1, chunk_num_input_frames)
                        grad_mat.append(attrs[k][0])  # [T]
                    log_probs_per_token.append(word_log_probs[0, :n_tok])  # [n_tok]
                    num_tokens_per_word_.append(n_tok)
                    chunk_total_tokens += n_tok
                num_tokens_.append(chunk_total_tokens)

                audio_frames_start_end.append(forward_output.input_raw_start_end[0])

                with torch.no_grad():
                    chunk_exit_log_prob, _, _ = _calc_log_probs_and_per_token_grads(
                        w=words_end - words_start, forward_output=forward_output, no_grad=True
                    )
                    # exit is one token; take the first element.
                    chunk_exit_log_prob = chunk_exit_log_prob[0, 0:1]  # [1]
                exit_log_probs.append(chunk_exit_log_prob[0])

                print("** Freeing")
                del forward_output
                gc.collect()
                report_dev_memory_stats(dev)
                print(f"({time.time() - start_time} secs for the seq)")

            # Stack/concat. grad_mat is a list of [T] tensors of length total_tokens.
            grad_mat_ = torch.stack(grad_mat).flatten()  # [total_tokens * T]
            audio_frames_start_end_ = torch.concat(audio_frames_start_end)
            num_input_frames_ = torch.tensor(num_input_frames)
            num_words__ = torch.tensor(num_words_)
            num_tokens__ = torch.tensor(num_tokens_)
            num_tokens_per_word__ = torch.tensor(num_tokens_per_word_)
            log_probs_per_token_ = torch.concat(log_probs_per_token)  # [total_tokens]
            exit_log_probs_ = torch.stack(exit_log_probs)

            print("** Storing to HDF")
            hdf_writer.insert_batch(
                # [1, total_tokens * T, 1]
                grad_mat_.cpu().numpy()[None, :, None],
                seq_len=[len(grad_mat_)],
                seq_tag=[f"seq-{seq_idx}"],
                extra={
                    "audio_frames_start_end": audio_frames_start_end_.cpu().numpy()[None],
                    "num_input_frames": num_input_frames_.cpu().numpy()[None, :, None],
                    "num_words": num_words__.cpu().numpy()[None, :, None],
                    "num_tokens": num_tokens__.cpu().numpy()[None, :, None],
                    "num_tokens_per_word": num_tokens_per_word__.cpu().numpy()[None, :, None],
                    "log_probs_per_token": log_probs_per_token_.cpu().numpy()[None, :, None],
                    "exit_log_probs": exit_log_probs_.cpu().numpy()[None, :, None],
                },
            )

        hdf_writer.close()

    def _run_seq_batched(self, *, model, ds, hdf_writer, attr_reduce_func, dev):
        """Seq-batched (seq_batch_size>1) extraction loop:
        one forward per batch of consecutive seqs,
        each backward computes one token of EVERY sequence simultaneously
        (one-hot grad_outputs; grads land in separate batch rows,
        exact because the models have no cross-seq interaction).
        Same per-seq math and HDF layout as the default loop.
        """
        import time
        import gc
        import numpy as np
        import torch
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
        from i6_experiments.users.zeyer.torch.batch_gather import batches_gather

        def _process(batch):
            start_time = time.time()
            nb = len(batch)
            sr = batch[0][2]
            assert all(b[2] == sr for b in batch)
            lens = [len(b[1]) for b in batch]
            raw = torch.zeros((nb, max(lens)))
            for i, (_, audio, _, _) in enumerate(batch):
                raw[i, : len(audio)] = torch.tensor(audio)
            fwd = _amp_forward(
                model,
                self.amp_dtype,
                raw_inputs=raw,
                raw_inputs_sample_rate=sr,
                raw_input_seq_lens=torch.tensor(lens),
                raw_targets=[b[3] for b in batch],
                raw_target_seq_lens=torch.tensor([len(b[3]) for b in batch]),
                omitted_prev_context=torch.zeros(nb, dtype=torch.int64),
            )
            tse = fwd.target_start_end.cpu()  # [B, W_max+1, 2]
            # Per seq: in-word token positions (word order; separators excluded) + per-word counts.
            tok_pos: list = []
            toks_per_word: list = []
            for i, (_, _, _, words) in enumerate(batch):
                pos: list = []
                counts: list = []
                for w in range(len(words)):
                    t0, t1 = int(tse[i, w, 0]), int(tse[i, w, 1])
                    pos.extend(range(t0, t1))
                    counts.append(t1 - t0)
                tok_pos.append(pos)
                toks_per_word.append(counts)
            # All token scores (incl. each seq's exit slot) from ONE log_probs call over [0, S_b+1).
            n_targets = torch.tensor([int(tse[i, len(b[3]), 1]) for i, b in enumerate(batch)])
            loss = model.log_probs(
                forward_output=fwd, start=torch.zeros(nb, dtype=torch.int64), end=n_targets
            )  # [B, n_max, V]
            targets_sl = batch_slice(
                fwd.targets,
                (torch.zeros(nb, dtype=torch.int64, device=fwd.targets.device), n_targets.to(fwd.targets.device)),
            )
            loss = batches_gather(loss, indices=targets_sl, num_batch_dims=2)  # [B, n_max]
            mask = torch.arange(loss.shape[1], device=loss.device)[None, :] >= n_targets.to(loss.device)[:, None]
            loss.masked_fill_(mask, 0.0)

            def _reduce(grad):  # [B, T_max, F] -> [B, T_sliced_max]
                with torch.no_grad():
                    attr = batch_slice(grad.float(), fwd.input_slice_start_end)
                    if self.mult_grad_by_inputs:
                        e = batch_slice(fwd.inputs.float(), fwd.input_slice_start_end)
                        attr = attr * e
                    return attr_reduce_func(attr)

            m_max = max(len(p) for p in tok_pos)
            attr_rows = [[None] * len(p) for p in tok_pos]
            if self.batched_backward:
                # vmap in chunks: the one-hot bank over all tokens at once would hold
                # m_max grad buffers of the whole batched graph.
                chunk = 32
                for j0 in range(0, m_max, chunk):
                    js = range(j0, min(j0 + chunk, m_max))
                    v = loss.new_zeros((len(js),) + tuple(loss.shape))
                    for jj, j in enumerate(js):
                        for i in range(nb):
                            if j < len(tok_pos[i]):
                                v[jj, i, tok_pos[i][j]] = 1.0
                    (grads,) = torch.autograd.grad(
                        loss, fwd.inputs, grad_outputs=v, is_grads_batched=True, retain_graph=True
                    )
                    for jj, j in enumerate(js):
                        attr = _reduce(grads[jj])
                        for i in range(nb):
                            if j < len(tok_pos[i]):
                                attr_rows[i][j] = attr[i]
            else:
                for j in range(m_max):
                    v = loss.new_zeros(tuple(loss.shape))
                    for i in range(nb):
                        if j < len(tok_pos[i]):
                            v[i, tok_pos[i][j]] = 1.0
                    (grad,) = torch.autograd.grad(loss, fwd.inputs, grad_outputs=v, retain_graph=True)
                    attr = _reduce(grad)
                    for i in range(nb):
                        if j < len(tok_pos[i]):
                            attr_rows[i][j] = attr[i]

            in_lens = fwd.get_inputs_seq_lens_sliced()
            loss_det = loss.detach().cpu()
            for i, (seq_idx, _, _, words) in enumerate(batch):
                t_b = int(in_lens[i])
                pos = tok_pos[i]
                grad_mat_ = torch.stack([attr_rows[i][j][:t_b] for j in range(len(pos))]).flatten()
                hdf_writer.insert_batch(
                    grad_mat_.cpu().numpy()[None, :, None],
                    seq_len=[len(grad_mat_)],
                    seq_tag=[f"seq-{seq_idx}"],
                    extra={
                        "audio_frames_start_end": fwd.input_raw_start_end[i, :t_b].cpu().numpy()[None],
                        "num_input_frames": np.array([[[t_b]]], dtype="int32"),
                        "num_words": np.array([[[len(words)]]], dtype="int32"),
                        "num_tokens": np.array([[[len(pos)]]], dtype="int32"),
                        "num_tokens_per_word": np.array(toks_per_word[i], dtype="int32")[None, :, None],
                        "log_probs_per_token": loss_det[i, pos].numpy().astype("float32")[None, :, None],
                        "exit_log_probs": np.array(
                            [[[float(loss_det[i, int(tse[i, len(words), 0])])]]], dtype="float32"
                        ),
                    },
                )
            gc.collect()
            print(f"({time.time() - start_time} secs for the batch of {nb})", flush=True)

        buf: list = []
        for seq_idx, data in enumerate(ds):
            audio = data["audio"]["array"]
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            words = list(data[self.target_source]["utterance"])
            print(f"seq {seq_idx}, {audio.shape=}, {' '.join(words)!r}", flush=True)
            buf.append((seq_idx, audio, data["audio"]["sampling_rate"], words))
            if len(buf) == self.seq_batch_size:
                _process(buf)
                buf = []
        if buf:
            _process(buf)
