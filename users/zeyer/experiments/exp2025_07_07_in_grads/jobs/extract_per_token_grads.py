"""Per-token variant of :class:`ExtractInGradsFromModelJob`."""

from typing import Optional, List
from sisyphus import Task
from .extract_in_grad_scores import ExtractInGradsFromModelJob


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

        print("Import Torch, Numpy...")
        start_time = time.time()

        import numpy as np
        import torch

        print(f"({time.time() - start_time} secs)")

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

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = data["audio"]["array"]
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            samplerate = data["audio"]["sampling_rate"]
            words = data["word_detail"]["utterance"]
            transcription = " ".join(words)
            print(f"seq {seq_idx}, {audio.shape=}, {samplerate=}, {transcription!r}")
            num_words = len(words)
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

                forward_output: ForwardOutput = model(
                    raw_inputs=torch.tensor(audio[audio_start:audio_end])[None],
                    raw_inputs_sample_rate=samplerate,
                    raw_input_seq_lens=torch.tensor([audio_end - audio_start]),
                    raw_targets=[words[words_start:words_end]],
                    raw_target_seq_lens=torch.tensor([words_end - words_start]),
                    omitted_prev_context=torch.tensor([words_start]),
                )

                # noinspection PyShadowingNames
                def _calc_log_probs_and_per_token_grads(
                    w: int,
                    *,
                    report_mem: bool = False,
                    forward_output: ForwardOutput,
                    no_grad: bool = False,
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
                    attrs_per_token: List[torch.Tensor] = []
                    for k in range(num_tokens):
                        single_loss = loss[:, k].sum()  # scalar
                        (grad,) = torch.autograd.grad(single_loss, forward_output.inputs, retain_graph=True)
                        with torch.no_grad():
                            attr = batch_slice(grad.float(), forward_output.input_slice_start_end)  # [B, T, F]
                            if self.mult_grad_by_inputs:
                                e = batch_slice(forward_output.inputs.float(), forward_output.input_slice_start_end)
                                attr = attr * e
                            attr = attr_reduce_func(attr)  # [B, T]
                        attrs_per_token.append(attr)

                    if report_mem:
                        report_dev_memory_stats(dev)

                    return loss.detach(), num_tokens, attrs_per_token

                print("** Calculating grads")
                chunk_num_input_frames = forward_output.get_inputs_seq_lens_sliced()[0].item()
                num_input_frames.append(chunk_num_input_frames)
                num_words_.append(words_end - words_start)
                chunk_total_tokens = 0
                for w in range(words_end - words_start):
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
