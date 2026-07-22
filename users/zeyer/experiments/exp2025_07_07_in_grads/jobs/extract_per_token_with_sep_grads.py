"""Per-token grad extract that also backwards through inter-word separator subwords.

Counterpart to :class:`ExtractInGradsPerTokenJob` for the char_level + char_level_sep
(or char_level_brackets) case where the model's tokenized target contains subwords
*between* words (the separator or BOW/EOW markers). The base PerToken job iterates
``for w in range(num_words)`` and only backwards through subwords inside each word's
``target_start_end[:, w]`` range -- it never touches the inter-word subwords.

This subclass identifies those inter-word subwords as the gap
``[target_start_end[:, w].end, target_start_end[:, w+1].start)`` and does one
backward per separator subword too. The resulting per-token grad maps for the
separator should attribute to the silence between words and act as anchors
for the per-subword Aligner.

HDF schema differs from :class:`ExtractInGradsPerTokenJob`:

- ``num_subwords``: replaces ``num_tokens``. Total subwords per chunk
  (chars-per-word + separator-subwords between consecutive words).
- ``num_tokens_per_word``: unchanged (chars per word).
- ``num_inter_word_tokens``: new, ``[total_inter_word_gaps_across_chunks, 1]``,
  count of separator subwords between word w and w+1. Has length
  ``num_words - 1`` per chunk (can be 0 if no separator between two words).
- Row order in ``inputs``: word_0_chars, inter_0_1_seps, word_1_chars,
  inter_1_2_seps, ..., word_{N-1}_chars.

Requires model to be ``char_level=True`` and have at least one separator-emitting
flag (``char_level_sep`` or ``char_level_brackets``); otherwise the gaps are
empty and this degenerates to the base PerToken behavior.
"""

from typing import Optional, List
from sisyphus import Task
from .extract_in_grad_scores import ExtractInGradsFromModelJob


class ExtractInGradsPerTokenWithSepGradsJob(ExtractInGradsFromModelJob):
    """See module docstring."""

    rqmt = {"time": 100, "cpu": 2, "gpu": 1, "mem": 125}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
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

        import numpy as np
        import torch

        from returnn.util import better_exchook
        from returnn.datasets.hdf import SimpleHDFWriter
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
        from i6_experiments.users.zeyer.torch.batch_gather import batches_gather
        from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats

        better_exchook.install()

        try:
            import lovely_tensors
            lovely_tensors.monkey_patch()
        except ImportError:
            pass

        from .attr_reduction import get_attr_reduce_func

        attr_reduce_func = get_attr_reduce_func(self.attr_reduction)

        from .models import make_model, ForwardOutput

        dev = torch.device("cuda")
        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)
        for p in model.parameters():
            p.requires_grad = False
        report_dev_memory_stats(dev)

        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(),
            dim=1,
            ndim=2,
            extra_type={
                "audio_frames_start_end": (2, 2, "int32"),
                "num_input_frames": (1, 2, "int32"),
                "num_words": (1, 2, "int32"),
                "num_subwords": (1, 2, "int32"),
                "num_tokens_per_word": (1, 2, "int32"),
                "num_inter_word_tokens": (1, 2, "int32"),
                "log_probs_per_subword": (1, 2, "float32"),
                "exit_log_probs": (1, 2, "float32"),
            },
        )

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}; key={self.dataset_key}; num_seqs={len(ds[self.dataset_key])}")

        from returnn.datasets.hdf import HDFDataset

        chunk_segmentation_hdf_ds: Optional[HDFDataset] = None
        if self.chunk_segmentation_hdf is not None:
            chunk_segmentation_hdf_ds = HDFDataset([self.chunk_segmentation_hdf.get_path()])
            chunk_segmentation_hdf_ds.initialize()
            chunk_segmentation_hdf_ds.init_seq_order(epoch=1)

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = data["audio"]["array"]
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            samplerate = data["audio"]["sampling_rate"]
            words = data["word_detail"]["utterance"]
            num_words = len(words)

            if chunk_segmentation_hdf_ds is not None:
                chunk_segmentation_hdf_ds.load_seqs(seq_idx, seq_idx + 1)
                chunk_audio_start_end = chunk_segmentation_hdf_ds.get_data(seq_idx, "audio_chunk_start_end")
                chunk_words_indices_start_end = chunk_segmentation_hdf_ds.get_data(seq_idx, "data")
                num_chunks = chunk_audio_start_end.shape[0]
            else:
                num_chunks = 1
                chunk_audio_start_end = np.array([[0, len(audio)]], dtype=np.int32)
                chunk_words_indices_start_end = np.array([[0, num_words]], dtype=np.int32)

            grad_mat: List[torch.Tensor] = []
            audio_frames_start_end: List[torch.Tensor] = []
            num_input_frames: List[int] = []
            num_words_: List[int] = []
            num_subwords_: List[int] = []
            num_tokens_per_word_: List[int] = []  # chars per word
            num_inter_word_tokens_: List[int] = []  # seps between consecutive words
            log_probs_per_subword: List[torch.Tensor] = []
            exit_log_probs: List[torch.Tensor] = []

            for chunk_idx in range(num_chunks):
                start_time = time.time()
                audio_start, audio_end = chunk_audio_start_end[chunk_idx]
                audio_start, audio_end = max(audio_start, 0), min(audio_end, len(audio))
                words_start, words_end = chunk_words_indices_start_end[chunk_idx]
                words_start, words_end = max(words_start, 0), min(words_end, num_words)
                chunk_num_words = words_end - words_start
                print(
                    f"** Forwarding chunk {chunk_idx + 1}/{num_chunks}"
                    f" audio {audio_start / samplerate}-{audio_end / samplerate}"
                    f" words {words_start}-{words_end} ({words[words_start:words_end]!r})"
                )

                forward_output: ForwardOutput = model(
                    raw_inputs=torch.tensor(audio[audio_start:audio_end])[None],
                    raw_inputs_sample_rate=samplerate,
                    raw_input_seq_lens=torch.tensor([audio_end - audio_start]),
                    raw_targets=[words[words_start:words_end]],
                    raw_target_seq_lens=torch.tensor([chunk_num_words]),
                    omitted_prev_context=torch.tensor([words_start]),
                )

                tse = forward_output.target_start_end  # [B=1, num_words+1, 2]

                def _backward_subword_range(t0_int: int, t1_int: int, *, no_grad: bool = False):
                    """Backward through subword positions [t0_int, t1_int). Returns (log_probs[t1-t0], attrs[t1-t0])."""
                    assert t1_int > t0_int, f"empty range [{t0_int}, {t1_int})"
                    t0 = torch.tensor([t0_int], device=tse.device)
                    t1 = torch.tensor([t1_int], device=tse.device)
                    loss = model.log_probs(forward_output=forward_output, start=t0, end=t1)  # [B, t1-t0, V]
                    targets = batch_slice(forward_output.targets, (t0, t1))
                    loss = batches_gather(loss, indices=targets, num_batch_dims=2)  # [B, t1-t0]
                    n_tok = t1_int - t0_int
                    if no_grad:
                        return loss.detach(), None
                    attrs: List[torch.Tensor] = []
                    for k in range(n_tok):
                        single_loss = loss[:, k].sum()
                        (grad,) = torch.autograd.grad(single_loss, forward_output.inputs, retain_graph=True)
                        with torch.no_grad():
                            attr = batch_slice(grad.float(), forward_output.input_slice_start_end)
                            if self.mult_grad_by_inputs:
                                e = batch_slice(forward_output.inputs.float(), forward_output.input_slice_start_end)
                                attr = attr * e
                            attr = attr_reduce_func(attr)  # [B, T]
                        attrs.append(attr)
                    return loss.detach(), attrs

                print("** Calculating grads (per char + per separator)")
                chunk_num_input_frames = forward_output.get_inputs_seq_lens_sliced()[0].item()
                num_input_frames.append(chunk_num_input_frames)
                num_words_.append(chunk_num_words)
                chunk_total_subwords = 0

                for w in range(chunk_num_words):
                    t0_w = int(tse[0, w, 0])
                    t1_w = int(tse[0, w, 1])
                    word_lp, word_attrs = _backward_subword_range(t0_w, t1_w)
                    n_chars = t1_w - t0_w
                    for k in range(n_chars):
                        assert word_attrs[k].shape == (1, chunk_num_input_frames)
                        grad_mat.append(word_attrs[k][0])
                    log_probs_per_subword.append(word_lp[0, :n_chars])
                    num_tokens_per_word_.append(n_chars)
                    chunk_total_subwords += n_chars

                    if w + 1 < chunk_num_words:
                        t0_next = int(tse[0, w + 1, 0])
                        n_sep = t0_next - t1_w
                        assert n_sep >= 0, f"negative gap between word {w} and {w + 1}: {t1_w=} {t0_next=}"
                        if n_sep > 0:
                            sep_lp, sep_attrs = _backward_subword_range(t1_w, t0_next)
                            for k in range(n_sep):
                                grad_mat.append(sep_attrs[k][0])
                            log_probs_per_subword.append(sep_lp[0, :n_sep])
                            chunk_total_subwords += n_sep
                        num_inter_word_tokens_.append(n_sep)

                num_subwords_.append(chunk_total_subwords)

                audio_frames_start_end.append(forward_output.input_raw_start_end[0])

                with torch.no_grad():
                    t0_exit = int(tse[0, chunk_num_words, 0])
                    t1_exit = int(tse[0, chunk_num_words, 1])
                    exit_lp, _ = _backward_subword_range(t0_exit, t1_exit, no_grad=True)
                exit_log_probs.append(exit_lp[0, 0])

                print("** Freeing")
                del forward_output
                gc.collect()
                report_dev_memory_stats(dev)
                print(f"({time.time() - start_time} secs for the seq)")

            grad_mat_ = torch.stack(grad_mat).flatten()
            audio_frames_start_end_ = torch.concat(audio_frames_start_end)
            num_input_frames_ = torch.tensor(num_input_frames)
            num_words__ = torch.tensor(num_words_)
            num_subwords__ = torch.tensor(num_subwords_)
            num_tokens_per_word__ = torch.tensor(num_tokens_per_word_)
            num_inter_word_tokens__ = torch.tensor(num_inter_word_tokens_) if num_inter_word_tokens_ else torch.zeros(0, dtype=torch.int64)
            log_probs_per_subword_ = torch.concat(log_probs_per_subword)
            exit_log_probs_ = torch.stack(exit_log_probs)

            print("** Storing to HDF")
            hdf_writer.insert_batch(
                grad_mat_.cpu().numpy()[None, :, None],
                seq_len=[len(grad_mat_)],
                seq_tag=[f"seq-{seq_idx}"],
                extra={
                    "audio_frames_start_end": audio_frames_start_end_.cpu().numpy()[None],
                    "num_input_frames": num_input_frames_.cpu().numpy()[None, :, None],
                    "num_words": num_words__.cpu().numpy()[None, :, None],
                    "num_subwords": num_subwords__.cpu().numpy()[None, :, None],
                    "num_tokens_per_word": num_tokens_per_word__.cpu().numpy()[None, :, None],
                    "num_inter_word_tokens": num_inter_word_tokens__.cpu().numpy()[None, :, None],
                    "log_probs_per_subword": log_probs_per_subword_.cpu().numpy()[None, :, None],
                    "exit_log_probs": exit_log_probs_.cpu().numpy()[None, :, None],
                },
            )

        hdf_writer.close()
