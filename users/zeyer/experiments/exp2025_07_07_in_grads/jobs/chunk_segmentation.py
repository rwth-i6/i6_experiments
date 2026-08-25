from __future__ import annotations
from typing import Optional, Union, Any, Dict, List, Tuple
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class ChunkSegmentationFromModelJob(Job):
    """
    For chunked long-form recognition:
    This job finds an approximate optimal segmentation of what words to assign to which chunk.
    The chunks are fixed in size and amount of overlap.

    The output HDF contains the word start/end indices per chunk,
    and the audio sample start/end indices per chunk (although they are redundant, as they are fixed).
    """

    # excluded values = pre-flag behavior
    __sis_hash_exclude__ = {
        "pass_omitted_prev_words": False,
        "max_words_per_chunk": None,
        "word_start_completion_norm": False,
        "exit_scale": 1.0,
    }

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        model_config: Dict[str, Any],
        chunk_size_secs: float = 30.0,
        chunk_overlap_secs: float = 5.0,
        empty_exit_penalty: float = -5.0,
        word_start_heuristic: bool = True,
        pass_omitted_prev_words: bool = False,
        max_words_per_chunk: Optional[int] = None,
        word_start_completion_norm: Union[bool, str] = False,
        exit_scale: float = 1.0,
        dump_wav_first_n_seqs: int = 0,
    ):
        """
        :param dataset_dir: hub cache dir, e.g. via DownloadHuggingFaceRepoJobV2. for load_dataset
        :param dataset_key: e.g. "train", "test", whatever the dataset provides
        :param returnn_root:
        :param model_config:
        :param chunk_size_secs: chunk size in seconds
        :param chunk_overlap_secs:
        :param empty_exit_penalty:
        :param word_start_heuristic:
        :param pass_omitted_prev_words: pass the actual omitted previous words to the model forward
            (`omitted_prev_words`), for wrappers with a native prev-text mechanism
            (e.g. Whisper `<|startofprev|>`). The wrapper must declare the parameter.
        :param max_words_per_chunk: cap the candidate words passed per chunk forward
            (default: all remaining words of the sequence).
            Needed for limited-context models
            (Whisper: 448 decoder positions; transducers: the T x U joint memory).
            Semantically free when the cap exceeds what a chunk can contain
            (~100 words at 30s); the final chunk asserts the cap did not bite.
        :param word_start_completion_norm: length-fair word-start selection:
            the plain argmax over ``accum_exit(w)`` compares exits at different word counts
            and ignores the remaining ``S - w`` words' future cost --
            with weak per-word scores (several nats each, e.g. Whisper on conversational
            speech) it degenerates to "consume almost nothing" (extreme lag).
            With this on, each remaining word is credited with a mean per-word score c_bar
            (optimistic completion estimate, A*-style):
            ``argmax accum_exit(w) - w * c_bar``.
            Exactly neutral when per-word scores are ~0 (confident models).
            ``True``: c_bar = mean over ALL scored words of the just-scored chunk.
            KNOWN BAD for audio-anchored scorers (CTC): the far-future words
            (audio not in the chunk) score catastrophically and drag c_bar down,
            over-rewarding progress (collapses to max consumption). Kept for hash
            stability of finished jobs only.
            ``"consumed"``: c_bar = running per-seq mean over the words up to the
            plain-argmax boundary of each chunk (in-chunk by construction) --
            prices remaining words at the observed in-their-right-chunk rate.
        :param dump_wav_first_n_seqs: for debugging
        """
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.model_config = model_config

        self.chunk_size_secs = chunk_size_secs
        self.chunk_overlap_secs = chunk_overlap_secs
        self.empty_exit_penalty = empty_exit_penalty
        self.word_start_heuristic = word_start_heuristic
        self.pass_omitted_prev_words = pass_omitted_prev_words
        self.max_words_per_chunk = max_words_per_chunk
        if max_words_per_chunk is not None:
            # the exact DP cannot move its word window, only the heuristic advances the start
            assert word_start_heuristic, "max_words_per_chunk requires word_start_heuristic"
        assert word_start_completion_norm in (False, True, "consumed"), f"{word_start_completion_norm=}"
        self.word_start_completion_norm = word_start_completion_norm
        if word_start_completion_norm:
            assert word_start_heuristic, "word_start_completion_norm only affects the heuristic"
        # exit_scale: weight on the raw exit log-prob (before empty_exit_penalty);
        # 0 = ignore the exit score entirely (pure word-score monotone assignment).
        self.exit_scale = exit_scale
        self.dump_wav_first_n_seqs = dump_wav_first_n_seqs

        self.rqmt = {"time": 40, "cpu": 2, "gpu": 1, "mem": 125}

        self.out_hdf = self.output_path("out.hdf")

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["dump_wav_first_n_seqs"]
        return super().hash(parsed_args)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import time
        import math
        from dataclasses import dataclass

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
        from i6_experiments.users.zeyer.numpy.wave import write_wave_file
        from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
        from i6_experiments.users.zeyer.torch.batch_gather import batches_gather

        # os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
        better_exchook.install()

        try:
            # noinspection PyUnresolvedReferences
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            pass

        from .models import make_model, ForwardOutput

        device_str = "cuda"
        dev = torch.device(device_str)

        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)

        for p in model.parameters():
            p.requires_grad = False

        # Pure inference: the DP only reads log-prob values, never grads.
        # Without this, the _Node accum tensors keep each chunk's autograd graph alive
        # (the model wrappers create grad-enabled input leaves for the grad-extract jobs),
        # leaking ~5 GB GPU memory per chunk -> OOM after ~12 chunks.
        torch.set_grad_enabled(False)

        report_dev_memory_stats(dev)

        # Write word start/end ranges per chunk, and the chunk audio sample start/end ranges.
        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(), dim=2, ndim=2, extra_type={"audio_chunk_start_end": (2, 2, "int32")}
        )

        # Iter over data

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Dataset keys:", ds.keys())
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = data["audio"]["array"]
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            samplerate = data["audio"]["sampling_rate"]
            chunk_size_samples = math.ceil(self.chunk_size_secs * samplerate)
            words: List[str] = data["word_detail"]["utterance"]
            transcription = " ".join(words)
            print(f"* Seq {seq_idx}, {audio.shape=}, {len(audio) / samplerate} secs, {samplerate=}, {transcription!r}")
            assert len(transcription.split(" ")) == len(words)

            if seq_idx == 0:
                print("  data keys:", data.keys())

            # First a loop to determine the corse-chunkwise segmentation:
            # For fixed chunks (partially overlapping), assign the most likely words.
            # Dyn programming, outer loop over chunks.

            print("* Chunkwise segmenting...")

            # Prepare.
            # Collect chunk_start_end.
            chunk_start_end: List[Tuple[int, int]] = []  # in samples
            cur_audio_start = 0  # in samples
            while True:  # while not ended
                cur_audio_end = cur_audio_start + chunk_size_samples
                if cur_audio_end > len(audio):
                    cur_audio_end = len(audio)
                if len(audio) - cur_audio_end <= 128 and self.chunk_overlap_secs == 0:
                    # Skip to end. Avoids potential problems with too short chunks.
                    cur_audio_end = len(audio)
                assert cur_audio_end > cur_audio_start
                assert cur_audio_end - cur_audio_start > 1  # require some min len
                chunk_start_end.append((cur_audio_start, cur_audio_end))
                if cur_audio_end >= len(audio):
                    break  # only break point here
                cur_audio_start = cur_audio_end - math.ceil(self.chunk_overlap_secs * samplerate)
                assert cur_audio_start >= 0

            # The dyn prog array, to be filled.
            array: List[List[_Node]] = []  # [chunk_idx][rel word_idx]
            # running per-seq stats for word_start_completion_norm == "consumed"
            _cn_sum, _cn_cnt = 0.0, 0

            # In the (S+1)*C grid (RNN-T style), but we might not fill all S+1 entries per chunk idx.
            @dataclass
            class _Node:
                chunk_idx: int  # 0 <= c < C. the chunk we are in.
                word_idx: int  # 0 <= s <= S. we have seen this many words so far (incl. this chunk), words[:s]
                accum_in_log_prob: torch.Tensor  # []. log prob of this node (accumulated all the way from the start)
                # []. accum_in_log_prob+exit (end_token_id). horizontal transition to next chunk
                accum_exit_log_prob: torch.Tensor
                # []. accum_in_log_prob+word (one or more labels). vertical transition to next word. (None if s==S)
                accum_word_log_prob: Optional[torch.Tensor]
                backpointer: Optional[_Node]  # prev chunk, or prev word

            # Chunk-synchronous search.
            for cur_chunk_idx, (cur_audio_start, cur_audio_end) in enumerate(chunk_start_end):
                # Determine cur_word_start/cur_word_end, that is what we feed through the model
                # (that is not the actual chunk; we take subsets of that for the dyn prog score).
                if cur_chunk_idx == 0 or not self.word_start_heuristic:
                    prev_array_word_idx = 0
                    cur_word_start = 0
                else:
                    # Heuristic. Look through last chunk, look out for best exit_log_prob
                    prev_nodes_sel = array[cur_chunk_idx - 1]
                    prev_array_word_idx = int(
                        torch.stack([node.accum_exit_log_prob for node in prev_nodes_sel]).argmax().item()
                    )
                    if self.word_start_completion_norm:
                        # Length-fair selection (see __init__ docstring):
                        # credit each remaining word with a mean per-word score c_bar.
                        if self.word_start_completion_norm == "consumed":
                            # in-chunk rate: the words up to the plain-argmax boundary,
                            # accumulated per seq (bootstrap: the lag-argmax range IS in-chunk)
                            _wl = [
                                float(n.accum_word_log_prob - n.accum_in_log_prob)
                                for n in prev_nodes_sel[:prev_array_word_idx]
                                if n.accum_word_log_prob is not None
                            ]
                            _cn_sum += sum(_wl)
                            _cn_cnt += len(_wl)
                            _c_bar = (_cn_sum / _cn_cnt) if _cn_cnt else 0.0
                        else:  # True: naive all-scored-words mean (see docstring caveat)
                            _wl = [
                                float(n.accum_word_log_prob - n.accum_in_log_prob)
                                for n in prev_nodes_sel
                                if n.accum_word_log_prob is not None
                            ]
                            _c_bar = (sum(_wl) / len(_wl)) if _wl else 0.0
                        prev_array_word_idx = int(
                            np.argmax([float(n.accum_exit_log_prob) - n.word_idx * _c_bar for n in prev_nodes_sel])
                        )
                    cur_word_start = prev_nodes_sel[prev_array_word_idx].word_idx
                if self.max_words_per_chunk is not None:
                    # Feasibility clamp: the remaining chunks (incl. this one) must be able to
                    # absorb the remaining words (<= max_words_per_chunk each), so the DP always
                    # reaches the last word. If the heuristic's best exit lags behind that
                    # frontier, take the best exit among the feasible nodes instead.
                    # By induction the feasible set is non-empty whenever the whole seq is
                    # feasible (len(words) <= num_chunks * cap); otherwise assert (invalid seq).
                    min_start = len(words) - (len(chunk_start_end) - cur_chunk_idx) * self.max_words_per_chunk
                    if cur_word_start < min_start:
                        assert cur_chunk_idx > 0, (
                            f"seq {seq_idx}: infeasible: {len(words)} words in "
                            f"{len(chunk_start_end)} chunks x cap {self.max_words_per_chunk}"
                        )
                        prev_nodes = array[cur_chunk_idx - 1]
                        cand = [i for i, n in enumerate(prev_nodes) if n.word_idx >= min_start]
                        assert cand, (
                            f"seq {seq_idx} chunk {cur_chunk_idx}: no feasible word start >= {min_start} "
                            f"(prev chunk covers {prev_nodes[0].word_idx}..{prev_nodes[-1].word_idx}); "
                            f"{len(words)} words in {len(chunk_start_end)} chunks "
                            f"x cap {self.max_words_per_chunk} is likely infeasible"
                        )
                        prev_array_word_idx = max(cand, key=lambda i: float(prev_nodes[i].accum_exit_log_prob))
                        cur_word_start = prev_nodes[prev_array_word_idx].word_idx
                cur_word_end = len(words)  # Go to the end. Not so expensive...
                if self.max_words_per_chunk is not None:
                    # Limited-context models (see __init__ docstring).
                    cur_word_end = min(cur_word_end, cur_word_start + self.max_words_per_chunk)
                print(
                    f"** Forwarding chunk {cur_chunk_idx} (out of {len(chunk_start_end)}),"
                    f" {cur_audio_start / samplerate}:{cur_audio_end / samplerate} secs,"
                    f" words {cur_word_start}:{cur_word_end} (out of {len(words)})",
                    flush=True,
                )
                if cur_word_start >= cur_word_end:
                    # word_start_heuristic advanced to the last word already:
                    # a previous chunk's best exit consumed all words,
                    # so this and any later chunks are trailing (silence, no words to assign).
                    # Stop extending the grid and drop the trailing chunks --
                    # they carry no word assignments anyway,
                    # and forwarding an empty target range crashes the model.
                    # Reached by small chunk sizes (e.g. 10s) on recordings with trailing silence.
                    assert self.word_start_heuristic  # only reachable via the heuristic branch
                    chunk_start_end = chunk_start_end[:cur_chunk_idx]
                    break
                if cur_audio_end >= len(audio):
                    assert cur_word_end == len(words)  # need to overthink approx if this fails...

                forward_output: ForwardOutput = model(
                    raw_inputs=torch.tensor(audio[cur_audio_start:cur_audio_end]).unsqueeze(0),
                    raw_inputs_sample_rate=samplerate,
                    raw_input_seq_lens=torch.tensor([cur_audio_end - cur_audio_start]),
                    raw_targets=[words[cur_word_start:cur_word_end]],
                    raw_target_seq_lens=torch.tensor([cur_word_end - cur_word_start]),
                    omitted_prev_context=torch.tensor([cur_word_start]),
                    # wrappers with a native prev-text mechanism also get the actual words (explicit flag)
                    **({"omitted_prev_words": [words[:cur_word_start]]} if self.pass_omitted_prev_words else {}),
                )

                # Calculate log probs
                # logits = model.lm_head(last_out[:, dst_text_start - 1 :])  # [B,T-dst_text_start+1,V]
                # logits = logits.float()
                # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B,T-dst_text_start,V]

                array.append([])
                assert len(array) == cur_chunk_idx + 1
                for w in range(cur_word_end - cur_word_start + 1):
                    t0, t1 = forward_output.target_start_end[:, w].unbind(1)  # [B], [B]
                    log_probs = model.log_probs(forward_output=forward_output, start=t0, end=t1)  # [B,t1-t0,V]
                    word_idx = cur_word_start + w
                    if word_idx < cur_word_end:
                        targets = batch_slice(forward_output.targets, (t0, t1))  # [B,t1-t0]->V
                        word_log_prob = batches_gather(log_probs, indices=targets, num_batch_dims=2)  # [B,t1-t0]
                        word_log_prob.masked_fill_(
                            torch.arange(word_log_prob.shape[1], device=word_log_prob.device)[None, :]
                            # target_start_end (and thus t0/t1) can be on CPU (e.g. Phi4MM),
                            # word_log_prob is on the model device.
                            >= (t1 - t0).to(word_log_prob.device)[:, None],
                            0.0,
                        )
                        word_log_prob = word_log_prob.sum()  # []. assume single batch
                    else:
                        word_log_prob = None
                    # log_probs is relative to start=t0 (shape [B,t1-t0,V]),
                    # so position 0 is the distribution predicting the token at t0.
                    exit_log_prob = self.exit_scale * log_probs[0, 0, model.assistant_end_token_id]  # []. single batch
                    if w == 0:
                        # Add some penalty. For empty chunks, the prob is often overestimated.
                        exit_log_prob += self.empty_exit_penalty
                    prev_node_left, prev_node_below = None, None
                    if w > 0:
                        prev_node_below = array[cur_chunk_idx][-1]
                        assert prev_node_below.word_idx == word_idx - 1
                    if cur_chunk_idx > 0 and prev_array_word_idx + w < len(array[cur_chunk_idx - 1]):
                        prev_node_left = array[cur_chunk_idx - 1][prev_array_word_idx + w]
                        assert prev_node_left.word_idx == word_idx
                    if prev_node_below and not prev_node_left:
                        prev_node = prev_node_below
                        accum_in_log_prob = prev_node_below.accum_word_log_prob
                    elif not prev_node_below and prev_node_left:
                        prev_node = prev_node_left
                        accum_in_log_prob = prev_node_left.accum_exit_log_prob
                    elif prev_node_below and prev_node_left:
                        # This is the only place for recombination.
                        # Use maximum approximation.
                        if prev_node_below.accum_word_log_prob >= prev_node_left.accum_exit_log_prob:
                            prev_node = prev_node_below
                            accum_in_log_prob = prev_node_below.accum_word_log_prob
                        else:
                            prev_node = prev_node_left
                            accum_in_log_prob = prev_node_left.accum_exit_log_prob
                    else:
                        assert cur_chunk_idx == word_idx == 0
                        prev_node = None
                        accum_in_log_prob = torch.zeros(())
                    array[cur_chunk_idx].append(
                        _Node(
                            chunk_idx=cur_chunk_idx,
                            word_idx=word_idx,
                            accum_in_log_prob=accum_in_log_prob,
                            backpointer=prev_node,
                            accum_word_log_prob=(accum_in_log_prob + word_log_prob)
                            if word_idx < cur_word_end
                            else None,
                            accum_exit_log_prob=accum_in_log_prob + exit_log_prob,
                        )
                    )
                assert (
                    len(array[cur_chunk_idx]) == cur_word_end - cur_word_start + 1
                    and array[cur_chunk_idx][0].word_idx == cur_word_start
                    and array[cur_chunk_idx][-1].word_idx == cur_word_end
                )

                del forward_output, log_probs  # not needed anymore now

            # Backtrack
            nodes_alignment: List[_Node] = []
            node = array[-1][-1]  # remember: RNN-T like grid, last chunk (frame), last entry covers all words
            assert node.word_idx == len(words)  # has seen all words
            while node:
                nodes_alignment.append(node)
                node = node.backpointer
            nodes_alignment.reverse()

            # Collect words per chunk
            words_per_chunks: List[List[int]] = [[] for _ in range(len(chunk_start_end))]
            words_covered = 0
            for node in nodes_alignment[1:]:
                if node.backpointer.chunk_idx == node.chunk_idx:
                    assert node.word_idx == node.backpointer.word_idx + 1
                    words_per_chunks[node.chunk_idx].append(node.word_idx - 1)
                    assert words_covered == node.word_idx - 1
                    words_covered += 1
                else:
                    assert node.chunk_idx == node.backpointer.chunk_idx + 1
                    assert node.word_idx == node.backpointer.word_idx
            assert words_covered == len(words)
            words_indices_start_end = [(ws[0], ws[-1] + 1) if ws else (-1, -1) for ws in words_per_chunks]
            print("  Words per chunks:", words_indices_start_end)

            assert len(words_indices_start_end) == len(chunk_start_end)
            hdf_writer.insert_batch(
                np.array(words_indices_start_end)[None],
                seq_len=[len(chunk_start_end)],
                seq_tag=[f"seq-{seq_idx}"],
                extra={"audio_chunk_start_end": np.array(chunk_start_end)[None]},
            )

            if seq_idx < self.dump_wav_first_n_seqs:
                for cur_chunk_idx, ((cur_audio_start, cur_audio_end), ws) in enumerate(
                    zip(chunk_start_end, words_per_chunks)
                ):
                    write_wave_file(
                        f"seq{seq_idx}-chunk{cur_chunk_idx}.wav",
                        samples=audio[cur_audio_start:cur_audio_end],
                        sr=samplerate,
                    )
                    with open(f"seq{seq_idx}-chunk{cur_chunk_idx}.txt", "w") as f:
                        f.write(" ".join(words[w] for w in ws))

        hdf_writer.close()

        # better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())


class ChunkSegmentationFromModelBatchedJob(Job):
    """Cross-sequence batched version of :class:`ChunkSegmentationFromModelJob`.

    Identical DP and output HDF,
    but processes up to ``max_batch_size`` sequences in lockstep,
    batching the model forward across their current chunks via ``model.forward_batched``
    (one HF forward per step instead of one per (seq, chunk)).
    Requires a model wrapper providing ``forward_batched`` and configured with ``grad_wrt=None``
    (currently Phi4MM).
    In fp32 the batched forward reproduces the single-sequence job's log-probs (~1e-3),
    hence identical assignments;
    in bf16 it may differ by matmul non-determinism (set ``model_dtype`` in the model config).
    """

    # Bumped to 2 when sort_by_length / batched_logprobs were added:
    # those flags are part of the hash, so every value combination gets its own hash.
    __sis_version__ = 2
    # Algorithm variants added later.
    # The excluded value for each is the one that REPRODUCES the behavior from before the param existed
    # (no beam / no bias / summed word scores),
    # so a job left at the defaults hashes like a pre-variant job and the finished sweeps stay reused;
    # a non-default value still changes the hash.
    # Polarity matters:
    # excluding the *new* value instead would make a variant job collide with the old hash and not rerun.
    __sis_hash_exclude__ = {
        "word_start_beam": None,
        "exit_bias": 0.0,
        "length_norm": False,
        "dump_word_scores": False,
        "chunk_offset_secs": 0.0,
        "exit_scale": 1.0,
    }

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        model_config: Dict[str, Any],
        chunk_size_secs: float = 30.0,
        chunk_overlap_secs: float = 5.0,
        empty_exit_penalty: float = -5.0,
        word_start_heuristic: bool = True,
        max_batch_size: int = 8,
        sort_by_length: bool = True,
        batched_logprobs: bool = True,
        word_start_beam: Optional[float] = None,
        exit_bias: float = 0.0,
        length_norm: bool = False,
        dump_word_scores: bool = False,
        chunk_offset_secs: float = 0.0,
        exit_scale: float = 1.0,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.model_config = model_config
        self.chunk_size_secs = chunk_size_secs
        self.chunk_overlap_secs = chunk_overlap_secs
        self.empty_exit_penalty = empty_exit_penalty
        self.word_start_heuristic = word_start_heuristic
        self.max_batch_size = max_batch_size
        # sort_by_length: order seqs by word count so a batch groups similar-length prompts,
        # cutting padding waste. batched_logprobs: one lm_head over the whole chunk transcript
        # instead of one per word (the DP's real bottleneck). Both default on (verified equivalent).
        self.sort_by_length = sort_by_length
        self.batched_logprobs = batched_logprobs
        # word_start_beam: keep every previous-chunk exit within this log-prob margin of the best,
        # and start from the earliest survivor.
        # None = plain argmax (the original heuristic);
        # a wider beam prunes less and approaches word_start_heuristic=False (exact).
        # One knob interpolating the two, since the argmax end measurably beats the exact end.
        # exit_bias: added to EVERY exit log-prob,
        # unlike empty_exit_penalty which only applies to a chunk exiting with zero words --
        # a global words-per-chunk knob (cf. word insertion penalty).
        # length_norm: score a word by its per-token MEAN log-prob instead of the sum,
        # so emitting a multi-token word is not systematically dearer than the single exit token.
        self.word_start_beam = word_start_beam
        self.exit_bias = exit_bias
        self.length_norm = length_norm
        # dump_word_scores: additionally write per-word (score, num_tokens) of the winning path
        # to a second HDF -- the word's log-prob in its assigned chunk,
        # the raw material for confidence flagging / drift detection.
        self.dump_word_scores = dump_word_scores
        # stagger the grid: the first chunk is shortened to this (0 = regular grid)
        self.chunk_offset_secs = chunk_offset_secs
        # exit_scale: weight on the raw exit log-prob (before exit_bias / empty_exit_penalty);
        # 0 = ignore the exit score entirely (pure word-score monotone assignment).
        self.exit_scale = exit_scale

        self.rqmt = {"time": 40, "cpu": 2, "gpu": 1, "mem": 125}
        self.out_hdf = self.output_path("out.hdf")
        if dump_word_scores:
            self.out_word_scores_hdf = self.output_path("word_scores.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import math
        from dataclasses import dataclass

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
        from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
        from i6_experiments.users.zeyer.torch.batch_gather import batches_gather

        better_exchook.install()

        from .models import make_model, ForwardOutput

        dev = torch.device("cuda")
        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)
        for p in model.parameters():
            p.requires_grad = False
        torch.set_grad_enabled(False)
        report_dev_memory_stats(dev)

        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(), dim=2, ndim=2, extra_type={"audio_chunk_start_end": (2, 2, "int32")}
        )
        scores_writer = None
        if self.dump_word_scores:
            # per word: (log-prob score, num subword tokens)
            scores_writer = SimpleHDFWriter(self.out_word_scores_hdf.get_path(), dim=2, ndim=2)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]))
        split = ds[self.dataset_key]

        @dataclass
        class _Node:
            chunk_idx: int
            word_idx: int
            accum_in_log_prob: torch.Tensor
            accum_exit_log_prob: torch.Tensor
            accum_word_log_prob: Optional[torch.Tensor]
            backpointer: Optional["_Node"]
            # (word log-prob score, num subword tokens) of the word this node would emit,
            # as plain floats (not tensors) for the optional dump; None at the last word index.
            word_score: Optional[Tuple[float, int]] = None

        def _chunk_bounds(audio, samplerate):
            chunk_size_samples = math.ceil(self.chunk_size_secs * samplerate)
            cse: List[Tuple[int, int]] = []
            cur = 0
            first = True
            while True:
                # chunk_offset_secs staggers the whole grid by shortening only the FIRST chunk
                # (e.g. offset L/2 -> chunks [0,L/2), [L/2,3L/2), ...): a second, offset chunking
                # whose boundaries fall mid-chunk of the default grid, for agreement/combination.
                if first and self.chunk_offset_secs:
                    end = cur + math.ceil(self.chunk_offset_secs * samplerate)
                else:
                    end = cur + chunk_size_samples
                first = False
                if end > len(audio):
                    end = len(audio)
                if len(audio) - end <= 128 and self.chunk_overlap_secs == 0:
                    end = len(audio)
                assert end > cur and end - cur > 1
                cse.append((cur, end))
                if end >= len(audio):
                    break
                cur = end - math.ceil(self.chunk_overlap_secs * samplerate)
                assert cur >= 0
            return cse

        num_seqs = len(split)
        rows = [split[i] for i in range(num_seqs)]  # preload once (needed for length sort)
        order = list(range(num_seqs))
        if self.sort_by_length:
            # descending word count -> each batch groups similar-length prompts, less padding waste
            order = sorted(order, key=lambda i: -len(rows[i]["word_detail"]["utterance"]))
        results: Dict[int, Any] = {}  # orig seq_idx -> (words_indices_start_end, chunk_start_end)
        for group_start in range(0, num_seqs, self.max_batch_size):
            group = order[group_start : group_start + self.max_batch_size]  # orig seq indices in this batch
            st = {}  # orig seq idx -> per-seq DP state
            for i in group:
                data = rows[i]
                audio = data["audio"]["array"]
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                else:
                    audio = audio.astype(np.float32)
                sr = data["audio"]["sampling_rate"]
                words = data["word_detail"]["utterance"]
                st[i] = dict(
                    audio=audio,
                    sr=sr,
                    words=words,
                    cse=_chunk_bounds(audio, sr),
                    array=[],  # [chunk_idx][rel word_idx] of _Node
                    cci=0,  # current chunk index
                    done=False,
                )
            print(f"* group {group}: chunks per seq = {[len(st[i]['cse']) for i in group]}", flush=True)

            while True:
                # gather active sequences' current-chunk forward inputs
                active, fwd_audio, fwd_words, fwd_omitted, ws_start = [], [], [], [], {}
                for i in group:  # i: orig seq idx
                    s = st[i]  # this seq's DP state
                    if s["done"] or s["cci"] >= len(s["cse"]):
                        s["done"] = True
                        continue
                    cci = s["cci"]  # current chunk index
                    if cci == 0 or not self.word_start_heuristic:
                        prev_idx, cws = 0, 0  # rel node idx in prev chunk's array; chunk word start
                    else:
                        exits = torch.stack([n.accum_exit_log_prob for n in s["array"][cci - 1]])  # [num prev nodes]
                        if self.word_start_beam is None:
                            prev_idx = int(exits.argmax().item())
                        else:
                            # Beam prune:
                            # keep every exit within the margin of the best, start from the earliest survivor.
                            # Wider beam -> prunes less -> approaches the exact DP.
                            keep = (exits >= exits.max() - self.word_start_beam).nonzero()
                            prev_idx = int(keep.min().item())
                        cws = s["array"][cci - 1][prev_idx].word_idx
                    cwe = len(s["words"])  # chunk word end (exclusive; word window = words[cws:cwe])
                    if cws >= cwe:
                        # trailing silence: drop remaining chunks (see single-seq job for rationale)
                        assert self.word_start_heuristic
                        s["cse"] = s["cse"][:cci]
                        s["done"] = True
                        continue
                    a0, a1 = s["cse"][cci]
                    active.append(i)
                    ws_start[i] = (prev_idx, cws, cwe)  # word-window info for this chunk step
                    fwd_audio.append(torch.tensor(s["audio"][a0:a1]))
                    fwd_words.append(s["words"][cws:cwe])
                    fwd_omitted.append(cws)
                if not active:
                    break

                # One flushed line per chunk step, so the log keeps updating
                # and a slow run is distinguishable from a real hang:
                # small chunk sizes mean ~1000+ chunks per seq,
                # and the DP is otherwise silent for a whole group (only group start / seq end print).
                print(
                    f"** chunk step: {len(active)} active seqs,"
                    f" chunk {[st[i]['cci'] for i in active]} of {[len(st[i]['cse']) for i in active]},"
                    f" word start {[ws_start[i][1] for i in active]}"
                    f" of {[len(st[i]['words']) for i in active]}",
                    flush=True,
                )

                fwd_outputs = model.forward_batched(
                    raw_inputs_list=fwd_audio,
                    raw_inputs_sample_rate=st[active[0]]["sr"],
                    raw_targets_list=fwd_words,
                    omitted_prev_context_list=fwd_omitted,
                )

                for j, i in enumerate(active):  # j: batch slot in fwd_outputs, i: orig seq idx
                    s = st[i]  # this seq's DP state
                    cci = s["cci"]  # current chunk index
                    prev_idx, cws, cwe = ws_start[i]
                    fo: ForwardOutput = fwd_outputs[j]
                    s["array"].append([])
                    assert len(s["array"]) == cci + 1
                    all_lp = None
                    if self.batched_logprobs:
                        # one lm_head over the whole transcript, then slice per word (vs one lm_head
                        # per word). Same values (up to fp noise); the lm_head is the DP bottleneck.
                        max_end = int(fo.target_start_end[0, -1, 1])
                        all_lp = model.log_probs(
                            forward_output=fo, start=torch.tensor([0]), end=torch.tensor([max_end])
                        )  # [1, max_end, V]
                    for w in range(cwe - cws + 1):  # w: rel word idx; w == cwe-cws = exit-only row
                        t0, t1 = fo.target_start_end[:, w].unbind(1)  # [1], [1]: word w's token span
                        if all_lp is not None:
                            log_probs = all_lp[:, int(t0) : int(t1)]  # [1, t1-t0, V]
                        else:
                            log_probs = model.log_probs(forward_output=fo, start=t0, end=t1)  # [1, t1-t0, V]
                        word_idx = cws + w  # absolute word idx
                        if word_idx < cwe:
                            targets = batch_slice(fo.targets, (t0, t1))  # [1, t1-t0]
                            wlp = batches_gather(log_probs, indices=targets, num_batch_dims=2)  # [1, t1-t0]
                            wlp.masked_fill_(
                                torch.arange(wlp.shape[1], device=wlp.device)[None, :]
                                >= (t1 - t0).to(wlp.device)[:, None],
                                0.0,
                            )
                            wlp = wlp.sum()  # scalar: the word's summed log-prob
                            if self.length_norm:
                                wlp = wlp / max(1, int(t1 - t0))
                        else:
                            wlp = None
                        exit_lp = self.exit_scale * log_probs[0, 0, model.assistant_end_token_id]  # scalar exit score
                        if self.exit_bias:
                            exit_lp = exit_lp + self.exit_bias
                        if w == 0:
                            exit_lp = exit_lp + self.empty_exit_penalty
                        prev_left, prev_below = None, None  # below: emit word here; left: exit prev chunk
                        if w > 0:
                            prev_below = s["array"][cci][-1]
                            assert prev_below.word_idx == word_idx - 1
                        if cci > 0 and prev_idx + w < len(s["array"][cci - 1]):
                            prev_left = s["array"][cci - 1][prev_idx + w]
                            assert prev_left.word_idx == word_idx
                        if prev_below and not prev_left:
                            prev_node, accum_in = prev_below, prev_below.accum_word_log_prob
                        elif not prev_below and prev_left:
                            prev_node, accum_in = prev_left, prev_left.accum_exit_log_prob
                        elif prev_below and prev_left:
                            if prev_below.accum_word_log_prob >= prev_left.accum_exit_log_prob:
                                prev_node, accum_in = prev_below, prev_below.accum_word_log_prob
                            else:
                                prev_node, accum_in = prev_left, prev_left.accum_exit_log_prob
                        else:
                            assert cci == word_idx == 0
                            prev_node, accum_in = None, torch.zeros(())
                        s["array"][cci].append(
                            _Node(
                                chunk_idx=cci,
                                word_idx=word_idx,
                                accum_in_log_prob=accum_in,
                                backpointer=prev_node,
                                accum_word_log_prob=(accum_in + wlp) if word_idx < cwe else None,
                                accum_exit_log_prob=accum_in + exit_lp,
                                word_score=(float(wlp), int(t1 - t0)) if word_idx < cwe else None,
                            )
                        )
                    assert (
                        len(s["array"][cci]) == cwe - cws + 1
                        and s["array"][cci][0].word_idx == cws
                        and s["array"][cci][-1].word_idx == cwe
                    )
                    s["cci"] += 1
                del fwd_outputs

            # backtrack + write per seq
            for i in group:
                s = st[i]
                words, cse = s["words"], s["cse"]
                nodes_alignment: List[_Node] = []
                node = s["array"][-1][-1]
                assert node.word_idx == len(words)
                while node:
                    nodes_alignment.append(node)
                    node = node.backpointer
                nodes_alignment.reverse()
                words_per_chunks: List[List[int]] = [[] for _ in range(len(cse))]  # [chunk_idx] -> word indices
                # per word (in word order, since the path is monotone): (score, num_tokens)
                wscores: List[Tuple[float, int]] = []
                covered = 0
                for node in nodes_alignment[1:]:
                    if node.backpointer.chunk_idx == node.chunk_idx:
                        assert node.word_idx == node.backpointer.word_idx + 1
                        words_per_chunks[node.chunk_idx].append(node.word_idx - 1)
                        # the emitting (vertical) transition starts at the backpointer node,
                        # which carries the emitted word's score.
                        wscores.append(node.backpointer.word_score)
                        assert covered == node.word_idx - 1
                        covered += 1
                    else:
                        assert node.chunk_idx == node.backpointer.chunk_idx + 1
                        assert node.word_idx == node.backpointer.word_idx
                assert covered == len(words) == len(wscores)
                wise = [(ws[0], ws[-1] + 1) if ws else (-1, -1) for ws in words_per_chunks]  # (-1, -1) = empty chunk
                assert len(wise) == len(cse)
                results[i] = (wise, cse, wscores)
                print(f"  seq {i}: words per chunks = {wise}", flush=True)

        # Write in original dataset order (the metric job reads seqs positionally); groups may have
        # been reordered by sort_by_length.
        for i in range(num_seqs):
            wise, cse, wscores = results[i]
            hdf_writer.insert_batch(
                np.array(wise)[None],
                seq_len=[len(cse)],
                seq_tag=[f"seq-{i}"],
                extra={"audio_chunk_start_end": np.array(cse)[None]},
            )
            if scores_writer is not None:
                scores_writer.insert_batch(
                    np.array(wscores, dtype="float32")[None],
                    seq_len=[len(wscores)],
                    seq_tag=[f"seq-{i}"],
                )
        hdf_writer.close()
        if scores_writer is not None:
            scores_writer.close()


class CalcChunkAssignmentMetricsJob(Job):
    """
    Word-to-chunk assignment quality
    for the DP chunk assignment produced by :class:`ChunkSegmentationFromModelJob`.

    The DP only decides which chunk each word belongs to
    (not a frame-level position within it),
    so the natural metric is whether that decision was right,
    not a word-boundary-error (WBE)
    derived from some invented within-chunk placement (e.g. uniform split) --
    such a WBE would conflate the DP's actual decision
    with an arbitrary placement heuristic the algorithm never made.
    Instead:

    - ``accuracy``: fraction of words assigned to a chunk
      whose audio span actually contains the word's reference center time
      (with overlapping chunks, any covering chunk counts as correct).
    - placement error in SECONDS: time distance from the word's reference center
      to the assigned chunk's audio span (0 if the center is inside it),
      summarized as median / p95 / mean / max and the fraction beyond 0.5s and 1s.
      This is unit-consistent across chunk sizes (unlike a chunk-index error),
      and its tail is the robustness signal that matters:
      the median is ~0 while a small fraction of words drift many seconds.
    - ``chunk_idx_mae``: kept as a legacy softer signal
      (mean absolute error to the nearest covering chunk index).

    The seconds error has resolution = chunk size,
    so it measures coarse localization, NOT a word boundary:
    the method assigns words to chunks, it never decides a within-chunk position.
    A real per-frame word boundary (and thus WBE, comparable to WhisperX etc.)
    belongs in a separate downstream step
    once there's an actual within-chunk decode or forced alignment.
    """

    # Bumped to 2 when the seconds-based placement-error stats were added.
    __sis_version__ = 2

    def __init__(
        self,
        *,
        chunk_seg_hdf: tk.Path,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        dataset_offset_factors: int,
        aggregation: str = "micro",
        samplerate: Optional[int] = None,
    ):
        """
        :param chunk_seg_hdf: from :class:`ChunkSegmentationFromModelJob`.
        :param dataset_dir:
        :param dataset_key:
        :param returnn_root:
        :param dataset_offset_factors:
            see :class:`ChunkSegmentationFromModelJob`'s dataset conventions
            (also used by the grad-align metric jobs in exp2025_05_05_align.py).
        :param aggregation: "micro" (mean over all words) or "macro" (per-utt mean then over utts).
        :param samplerate: audio sampling rate for the seconds error; None reads it from the dataset.
        """
        super().__init__()
        self.chunk_seg_hdf = chunk_seg_hdf
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.dataset_offset_factors = dataset_offset_factors
        self.aggregation = aggregation
        # None -> read the audio sampling rate from the dataset feature (no audio decode);
        # pass an int to override.
        self.samplerate = samplerate

        self.out_accuracy = self.output_var("accuracy.txt")
        self.out_chunk_idx_mae = self.output_var("chunk_idx_mae.txt")
        # placement error in seconds (see class docstring): the robustness headline.
        self.out_error_median_sec = self.output_var("error_median_sec.txt")
        self.out_error_p95_sec = self.output_var("error_p95_sec.txt")
        self.out_frac_gt_1s = self.output_var("frac_gt_1s.txt")
        self.out_metrics = self.output_var("metrics.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 10, "time": 5, "engine": "short"})

    def run(self):
        import os
        import sys
        import numpy as np

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset

        chunk_seg_hdf_ds = HDFDataset([self.chunk_seg_hdf.get_path()])
        chunk_seg_hdf_ds.initialize()
        chunk_seg_hdf_ds.init_seq_order(epoch=1)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

        # audio sampling rate, to report the placement error in seconds.
        # prefer the dataset feature (no audio decode); fall back to decoding one seq.
        samplerate = self.samplerate
        if not samplerate:
            samplerate = getattr(ds[self.dataset_key].features["audio"], "sampling_rate", None)
        if not samplerate:
            samplerate = ds[self.dataset_key][0]["audio"]["sampling_rate"]
        assert samplerate, "could not determine samplerate; pass samplerate=..."
        print("Samplerate:", samplerate)

        utt_accs: List[List[float]] = []  # per utt, per word: 1.0 correct / 0.0 wrong
        utt_idx_errs: List[List[int]] = []  # per utt, per word: chunk-index error
        utt_errs_sec: List[List[float]] = []  # per utt, per word: seconds from center to assigned chunk span

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            chunk_seg_hdf_ds.load_seqs(seq_idx, seq_idx + 1)

            words = data["word_detail"]["utterance"]
            ref_word_starts = data["word_detail"]["start"]
            ref_word_ends = data["word_detail"]["stop"]
            assert len(words) == len(ref_word_starts) == len(ref_word_ends)
            # In samples, same units as chunk_start_end (dataset_offset_factors converts the raw
            # annotation unit to samples; see ChunkSegmentationFromModelJob's dataset conventions).
            ref_word_center_samples = [
                0.5 * (s + e) * self.dataset_offset_factors for s, e in zip(ref_word_starts, ref_word_ends)
            ]

            words_indices_start_end = chunk_seg_hdf_ds.get_data(seq_idx, "data")
            chunk_start_end = chunk_seg_hdf_ds.get_data(seq_idx, "audio_chunk_start_end")
            assert len(words_indices_start_end) == len(chunk_start_end)

            pred_chunk_idx = [None] * len(words)
            for chunk_idx, (word_start, word_end) in enumerate(words_indices_start_end):
                for w in range(int(word_start), int(word_end)):
                    pred_chunk_idx[w] = chunk_idx
            assert all(p is not None for p in pred_chunk_idx), "not all words got assigned to a chunk"

            accs, idx_errs, errs_sec = [], [], []
            for w in range(len(words)):
                center = ref_word_center_samples[w]
                covering = [ci for ci, (a0, a1) in enumerate(chunk_start_end) if a0 <= center < a1]
                if not covering:
                    # Center falls outside all chunk spans (e.g. trailing silence beyond the last
                    # chunk edge due to rounding) -- fall back to the nearest chunk by edge distance.
                    covering = [
                        min(
                            range(len(chunk_start_end)),
                            key=lambda ci: min(
                                abs(center - chunk_start_end[ci][0]), abs(center - chunk_start_end[ci][1])
                            ),
                        )
                    ]
                accs.append(1.0 if pred_chunk_idx[w] in covering else 0.0)
                idx_errs.append(min(abs(pred_chunk_idx[w] - ci) for ci in covering))
                # seconds error: time distance from the center to the ASSIGNED chunk's span,
                # 0 if the center is inside it. no covering fallback needed -- the assigned chunk always exists.
                a0p, a1p = chunk_start_end[pred_chunk_idx[w]]
                d_samp = 0.0 if a0p <= center < a1p else min(abs(center - a0p), abs(center - a1p))
                errs_sec.append(float(d_samp) / samplerate)

            print(f"** seq {seq_idx}, {len(words)=}, acc={float(np.mean(accs)):.3f}")
            utt_accs.append(accs)
            utt_idx_errs.append(idx_errs)
            utt_errs_sec.append(errs_sec)

        if self.aggregation == "micro":
            accuracy = float(np.mean([a for accs in utt_accs for a in accs]))
            chunk_idx_mae = float(np.mean([e for errs in utt_idx_errs for e in errs]))
        elif self.aggregation == "macro":
            accuracy = float(np.mean([np.mean(accs) for accs in utt_accs]))
            chunk_idx_mae = float(np.mean([np.mean(errs) for errs in utt_idx_errs]))
        else:
            raise ValueError(f"invalid aggregation {self.aggregation!r}")

        # Placement-error stats are pooled over all words (percentiles-of-percentiles is not meaningful),
        # so they are the same regardless of `aggregation`.
        flat_errs_sec = np.array([e for errs in utt_errs_sec for e in errs])
        error_stats = {
            "error_mean_sec": float(flat_errs_sec.mean()),
            "error_median_sec": float(np.percentile(flat_errs_sec, 50)),
            "error_p90_sec": float(np.percentile(flat_errs_sec, 90)),
            "error_p95_sec": float(np.percentile(flat_errs_sec, 95)),
            "error_p99_sec": float(np.percentile(flat_errs_sec, 99)),
            "error_max_sec": float(flat_errs_sec.max()),
            "frac_gt_0.5s": float((flat_errs_sec > 0.5).mean()),
            "frac_gt_1s": float((flat_errs_sec > 1.0).mean()),
        }
        metrics = {
            "accuracy": accuracy,
            "chunk_idx_mae": chunk_idx_mae,
            "samplerate": int(samplerate),
            "aggregation": self.aggregation,
            **error_stats,
        }
        print("CORPUS METRICS:", metrics)
        self.out_accuracy.set(accuracy)
        self.out_chunk_idx_mae.set(chunk_idx_mae)
        self.out_error_median_sec.set(error_stats["error_median_sec"])
        self.out_error_p95_sec.set(error_stats["error_p95_sec"])
        self.out_frac_gt_1s.set(error_stats["frac_gt_1s"])
        self.out_metrics.set(metrics)


class ChunkBoundaryReverifyJob(Job):
    """
    Local repair pass for a chunk assignment
    (from :class:`ChunkSegmentationFromModelBatchedJob`):
    re-places the words near every chunk boundary
    by comparing each word's acoustic score in the two adjacent chunks directly.

    Motivation (measured at cs30):
    almost all misplaced words sit in a NEIGHBORING chunk,
    nearly always too early,
    because the LLM can emit a word from text context alone
    before its audio arrives,
    and the DP path objective (word scores plus exit scores) accepts that.
    Re-running the same DP objective would reproduce the same boundary,
    so this pass uses a different, purely acoustic criterion:
    for the words within ``boundary_window`` of a boundary,
    score each word both in the left chunk and in the right chunk
    (same forced-decoding scores as the DP, but no exit scores),
    and pick the boundary position maximizing
    the summed left-scores before it plus the summed right-scores after it.
    """

    __sis_version__ = 1
    # excluded values = pre-gate behavior (no confidence gate)
    __sis_hash_exclude__ = {"word_scores_hdf": None, "min_window_conf": -2.0, "min_move_margin": 0.0}

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        model_config: Dict[str, Any],
        chunk_seg_hdf: tk.Path,
        boundary_window: int = 10,
        max_batch_size: int = 8,
        word_scores_hdf: Optional[tk.Path] = None,
        min_window_conf: float = -2.0,
        min_move_margin: float = 0.0,
    ):
        """
        :param dataset_dir: hub cache dir, like :class:`ChunkSegmentationFromModelBatchedJob`.
        :param dataset_key:
        :param returnn_root:
        :param model_config: same convention as the segmentation jobs.
        :param chunk_seg_hdf: assignment to refine (out_hdf of a segmentation job).
        :param boundary_window: words considered on each side of a boundary.
        :param max_batch_size: chunk forwards batched together.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.model_config = model_config
        self.chunk_seg_hdf = chunk_seg_hdf
        self.boundary_window = boundary_window
        self.max_batch_size = max_batch_size
        # Confidence gate: with a word-scores HDF (dump_word_scores of the segmentation job),
        # a boundary is SKIPPED when the windowed (+-boundary_window words) mean per-token
        # log-prob at the boundary is below min_window_conf --
        # inside such drifted regions both adjacent chunks are wrong,
        # so the local left-vs-right comparison can push words deeper (seen: max 45s -> 77s).
        self.word_scores_hdf = word_scores_hdf
        self.min_window_conf = min_window_conf
        # required log-prob advantage PER MOVED WORD to accept a boundary shift (0 = plain argmax)
        self.min_move_margin = min_move_margin

        self.rqmt = {"time": 40, "cpu": 2, "gpu": 1, "mem": 125}
        self.out_hdf = self.output_path("out.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys

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
        from returnn.datasets.hdf import HDFDataset, SimpleHDFWriter
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
        from i6_experiments.users.zeyer.torch.batch_gather import batches_gather

        better_exchook.install()

        from .models import make_model, ForwardOutput

        dev = torch.device("cuda")
        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)
        for p in model.parameters():
            p.requires_grad = False
        torch.set_grad_enabled(False)

        seg_ds = HDFDataset([self.chunk_seg_hdf.get_path()])
        seg_ds.initialize()
        seg_ds.init_seq_order(epoch=1)
        sc_ds = None
        if self.word_scores_hdf is not None:
            sc_ds = HDFDataset([self.word_scores_hdf.get_path()])
            sc_ds.initialize()
            sc_ds.init_seq_order(epoch=1)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        split = ds[self.dataset_key]
        print("Using key:", self.dataset_key, "num seqs:", len(split), flush=True)

        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(), dim=2, ndim=2, extra_type={"audio_chunk_start_end": (2, 2, "int32")}
        )

        def word_scores(fo: ForwardOutput, num_words: int) -> List[float]:
            """Per-word summed log-prob, like the DP's word score (no exit scores)."""
            max_end = int(fo.target_start_end[0, -1, 1])
            all_lp = model.log_probs(forward_output=fo, start=torch.tensor([0]), end=torch.tensor([max_end]))
            out = []
            for w in range(num_words):
                t0, t1 = fo.target_start_end[:, w].unbind(1)
                lp = all_lp[:, int(t0) : int(t1)]
                targets = batch_slice(fo.targets, (t0, t1))
                wlp = batches_gather(lp, indices=targets, num_batch_dims=2)
                wlp.masked_fill_(
                    torch.arange(wlp.shape[1], device=wlp.device)[None, :] >= (t1 - t0).to(wlp.device)[:, None],
                    0.0,
                )
                out.append(float(wlp.sum()))
            return out

        K = self.boundary_window
        n_moved_total = 0
        for seq_idx in range(seg_ds.num_seqs):
            seg_ds.load_seqs(seq_idx, seq_idx + 1)
            wise = [list(x) for x in np.asarray(seg_ds.get_data(seq_idx, "data"))]
            cse = np.asarray(seg_ds.get_data(seq_idx, "audio_chunk_start_end"))
            data = split[seq_idx]
            audio = np.asarray(data["audio"]["array"], dtype=np.float32)
            sr = data["audio"]["sampling_rate"]
            words = data["word_detail"]["utterance"]

            # boundaries between adjacent non-empty chunks, using the ORIGINAL assignment
            # (windows may not overlap chunk cores for the usual chunk sizes)
            wconf = None
            if sc_ds is not None:
                sc_ds.load_seqs(seq_idx, seq_idx + 1)
                sc = np.asarray(sc_ds.get_data(seq_idx, "data"))
                conf = sc[:, 0] / np.maximum(sc[:, 1], 1)
                kern = np.ones(2 * K + 1)
                wconf = np.convolve(conf, kern, mode="same") / np.convolve(np.ones(len(conf)), kern, mode="same")

            n_gated = 0
            bounds = []  # (c_left, c_right, lo, s, hi): candidate range [lo, hi], orig boundary s
            for c in range(len(wise) - 1):
                if wise[c][0] < 0 or wise[c + 1][0] < 0:
                    continue
                s = int(wise[c][1])
                if wconf is not None and wconf[min(s, len(wconf) - 1)] < self.min_window_conf:
                    n_gated += 1
                    continue
                lo = max(s - K, int(wise[c][0]))
                hi = min(s + K, int(wise[c + 1][1]))
                if lo < hi:
                    bounds.append((c, c + 1, lo, s, hi))
            if n_gated:
                print(f"seq {seq_idx}: {n_gated} boundaries gated (low windowed confidence)", flush=True)

            # batched forwards: per boundary two requests
            # (left chunk with words up to hi, right chunk with words from lo)
            reqs = []  # (bound_idx, side, cws, cwe, chunk_idx)
            for bi, (cl, cr, lo, s, hi) in enumerate(bounds):
                reqs.append((bi, "L", int(wise[cl][0]), hi, cl))
                reqs.append((bi, "R", lo, int(wise[cr][1]), cr))
            scores = {}
            for g0 in range(0, len(reqs), self.max_batch_size):
                group = reqs[g0 : g0 + self.max_batch_size]
                fwd_outputs = model.forward_batched(
                    raw_inputs_list=[torch.tensor(audio[cse[c][0] : cse[c][1]]) for (_, _, _, _, c) in group],
                    raw_inputs_sample_rate=sr,
                    raw_targets_list=[words[cws:cwe] for (_, _, cws, cwe, _) in group],
                    omitted_prev_context_list=[cws for (_, _, cws, _, _) in group],
                )
                for (bi, side, cws, cwe, _), fo in zip(group, fwd_outputs):
                    sc = word_scores(fo, cwe - cws)
                    scores[(bi, side)] = (cws, sc)

            n_moved = 0
            for bi, (cl, cr, lo, s, hi) in enumerate(bounds):
                if wise[cl][0] < 0 or wise[cr][0] < 0:
                    # a previous boundary emptied one side; skip to avoid conflicting updates
                    continue
                cws_l, sc_l = scores[(bi, "L")]
                cws_r, sc_r = scores[(bi, "R")]
                # candidate boundary s' in [lo2, hi]: words < s' in left chunk, >= s' in right chunk.
                # lo2 respects updates from the PREVIOUS boundary (windows can overlap when a chunk
                # holds < 2*K words): without it, conflicting sequential updates can orphan words
                # (seen: a word ended up outside every chunk range, reported as chunk 0, err 77s).
                lo2 = max(lo, int(wise[cl][0]))
                best_s, best_v, v_orig = s, None, None
                for s2 in range(lo2, hi + 1):
                    v = sum(sc_l[w - cws_l] for w in range(lo, s2)) + sum(sc_r[w - cws_r] for w in range(s2, hi))
                    if s2 == s:
                        v_orig = v
                    if best_v is None or v > best_v:
                        best_s, best_v = s2, v
                # min_move_margin: only accept a move when it is decisive
                # (per moved word), since near-ties are noise and moving a correct
                # but poorly-scored (mumbled) word breaks it.
                if best_s != s and v_orig is not None and best_v < v_orig + self.min_move_margin * abs(best_s - s):
                    best_s = s
                if best_s != s:
                    n_moved += abs(best_s - s)
                    wise[cl][1] = best_s
                    wise[cr][0] = best_s
                    if wise[cl][1] <= wise[cl][0]:
                        wise[cl] = [-1, -1]
                    if wise[cr][1] <= wise[cr][0]:
                        wise[cr] = [-1, -1]
            n_moved_total += n_moved
            print(f"seq {seq_idx}: {len(bounds)} boundaries, {n_moved} words moved", flush=True)

            hdf_writer.insert_batch(
                np.array(wise)[None],
                seq_len=[len(cse)],
                seq_tag=[f"seq-{seq_idx}"],
                extra={"audio_chunk_start_end": np.array(cse)[None]},
            )
        print(f"total words moved: {n_moved_total}", flush=True)
        hdf_writer.close()


class DriftSpanRepairJob(Job):
    """
    Repair pass for DRIFTED spans in a chunk assignment
    (from :class:`ChunkSegmentationFromModelBatchedJob`).

    Detection: a span of words whose windowed mean per-token log-prob
    (from the segmentation job's ``dump_word_scores`` HDF)
    falls below a threshold --
    the signature of a DP path that dragged a whole run of words
    into too-early chunks
    (measured at cs30: one seq with a 97-word drifted run,
    detector recall 0.55 at 1% flagged).

    Repair: per flagged span,
    a small anchored DP re-distributes the span's words
    over the span's chunk range (plus ``pad_chunks`` on each side):
    words outside the range keep their chunks (anchors),
    each chunk in the range gets a contiguous (possibly empty) word run,
    maximizing purely acoustic word scores
    (each chunk forwarded once with the whole span transcript,
    same forced-decoding scores as the DP but no exit scores,
    like :class:`ChunkBoundaryReverifyJob`).
    The repaired assignment is only accepted when it beats
    the original assignment's score under the SAME score matrix
    by ``min_move_margin`` per moved word.

    Differs from :class:`ChunkBoundaryReverifyJob`:
    that one shifts single boundaries within +-K words
    and is gated OFF inside drifted regions
    (where both adjacent chunks are wrong);
    this one targets exactly those regions
    and can move a whole run across several chunks at once.
    """

    __sis_version__ = 1

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        model_config: Dict[str, Any],
        chunk_seg_hdf: tk.Path,
        word_scores_hdf: tk.Path,
        detect_window: int = 10,
        flag_window_conf: float = -2.0,
        pad_chunks: int = 1,
        max_batch_size: int = 8,
        min_move_margin: float = 2.0,
    ):
        """
        :param dataset_dir: hub cache dir, like :class:`ChunkSegmentationFromModelBatchedJob`.
        :param dataset_key:
        :param returnn_root:
        :param model_config: same convention as the segmentation jobs.
        :param chunk_seg_hdf: assignment to repair (out_hdf of a segmentation job).
        :param word_scores_hdf: dump_word_scores HDF of the same segmentation job.
        :param detect_window: half-width (in words) of the confidence window;
            also the max gap (in words) when merging flagged words into spans.
        :param flag_window_conf: flag a word when its windowed conf is below this.
        :param pad_chunks: chunks added on each side of a flagged span's chunk range.
        :param max_batch_size: chunk forwards batched together.
        :param min_move_margin: required log-prob advantage PER MOVED WORD
            to accept a repaired span (near-ties are noise, see the reverify job).
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.model_config = model_config
        self.chunk_seg_hdf = chunk_seg_hdf
        self.word_scores_hdf = word_scores_hdf
        self.detect_window = detect_window
        self.flag_window_conf = flag_window_conf
        self.pad_chunks = pad_chunks
        self.max_batch_size = max_batch_size
        self.min_move_margin = min_move_margin

        self.rqmt = {"time": 40, "cpu": 2, "gpu": 1, "mem": 125}
        self.out_hdf = self.output_path("out.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys

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
        from returnn.datasets.hdf import HDFDataset, SimpleHDFWriter
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
        from i6_experiments.users.zeyer.torch.batch_gather import batches_gather

        better_exchook.install()

        from .models import make_model, ForwardOutput

        dev = torch.device("cuda")
        model_config = instanciate_delayed_copy(self.model_config)
        model = make_model(**model_config, device=dev)
        for p in model.parameters():
            p.requires_grad = False
        torch.set_grad_enabled(False)

        seg_ds = HDFDataset([self.chunk_seg_hdf.get_path()])
        seg_ds.initialize()
        seg_ds.init_seq_order(epoch=1)
        sc_ds = HDFDataset([self.word_scores_hdf.get_path()])
        sc_ds.initialize()
        sc_ds.init_seq_order(epoch=1)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        split = ds[self.dataset_key]
        print("Using key:", self.dataset_key, "num seqs:", len(split), flush=True)

        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(), dim=2, ndim=2, extra_type={"audio_chunk_start_end": (2, 2, "int32")}
        )

        def word_scores(fo: ForwardOutput, num_words: int) -> List[float]:
            """Per-word summed log-prob, like the DP's word score (no exit scores)."""
            max_end = int(fo.target_start_end[0, -1, 1])
            all_lp = model.log_probs(forward_output=fo, start=torch.tensor([0]), end=torch.tensor([max_end]))
            out = []
            for w in range(num_words):
                t0, t1 = fo.target_start_end[:, w].unbind(1)
                lp = all_lp[:, int(t0) : int(t1)]
                targets = batch_slice(fo.targets, (t0, t1))
                wlp = batches_gather(lp, indices=targets, num_batch_dims=2)
                wlp.masked_fill_(
                    torch.arange(wlp.shape[1], device=wlp.device)[None, :] >= (t1 - t0).to(wlp.device)[:, None],
                    0.0,
                )
                out.append(float(wlp.sum()))
            return out

        K = self.detect_window
        n_moved_total = 0
        for seq_idx in range(seg_ds.num_seqs):
            seg_ds.load_seqs(seq_idx, seq_idx + 1)
            wise = [list(x) for x in np.asarray(seg_ds.get_data(seq_idx, "data"))]
            cse = np.asarray(seg_ds.get_data(seq_idx, "audio_chunk_start_end"))
            sc_ds.load_seqs(seq_idx, seq_idx + 1)
            sc = np.asarray(sc_ds.get_data(seq_idx, "data"))
            data = split[seq_idx]
            audio = np.asarray(data["audio"]["array"], dtype=np.float32)
            sr = data["audio"]["sampling_rate"]
            words = data["word_detail"]["utterance"]
            num_words = len(words)

            conf = sc[:, 0] / np.maximum(sc[:, 1], 1)
            kern = np.ones(2 * K + 1)
            wconf = np.convolve(conf, kern, mode="same") / np.convolve(np.ones(len(conf)), kern, mode="same")
            flagged = np.nonzero(wconf < self.flag_window_conf)[0]

            # word -> chunk of the original assignment
            w2c = np.full(num_words, -1, dtype=int)
            for c, (a, b) in enumerate(wise):
                if a >= 0:
                    w2c[int(a) : int(b)] = c

            # flagged words -> spans (merge gaps <= K) -> repair ranges (chunk + word range)
            spans = []
            for i in flagged:
                if spans and i - spans[-1][1] <= K:
                    spans[-1][1] = i
                else:
                    spans.append([i, i])
            repairs = []  # [c_lo, c_hi, w_lo, w_hi): anchored DP range
            for f0, f1 in spans:
                cs = [c for c in w2c[f0 : f1 + 1] if c >= 0]
                if not cs:
                    continue
                c_lo = max(0, min(cs) - self.pad_chunks)
                c_hi = min(len(wise) - 1, max(cs) + self.pad_chunks)
                # monotone assignment -> the words of a chunk range are contiguous
                starts = [int(wise[c][0]) for c in range(c_lo, c_hi + 1) if wise[c][0] >= 0]
                ends = [int(wise[c][1]) for c in range(c_lo, c_hi + 1) if wise[c][0] >= 0]
                if not starts:
                    continue
                repairs.append([c_lo, c_hi, min(starts), max(ends)])
            # merge overlapping repair ranges
            repairs.sort()
            merged = []
            for r in repairs:
                if merged and r[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], r[1])
                    merged[-1][3] = max(merged[-1][3], r[3])
                else:
                    merged.append(r)
            repairs = merged

            # batched forwards: per repair one request per chunk in its range
            # (chunk audio + the whole span transcript)
            reqs = [(ri, c) for ri, (c_lo, c_hi, w_lo, w_hi) in enumerate(repairs) for c in range(c_lo, c_hi + 1)]
            scores = {}  # (ri, c) -> per-word scores of words[w_lo:w_hi] in chunk c
            for g0 in range(0, len(reqs), self.max_batch_size):
                group = reqs[g0 : g0 + self.max_batch_size]
                fwd_outputs = model.forward_batched(
                    raw_inputs_list=[torch.tensor(audio[cse[c][0] : cse[c][1]]) for (_, c) in group],
                    raw_inputs_sample_rate=sr,
                    raw_targets_list=[words[repairs[ri][2] : repairs[ri][3]] for (ri, _) in group],
                    omitted_prev_context_list=[repairs[ri][2] for (ri, _) in group],
                )
                for (ri, c), fo in zip(group, fwd_outputs):
                    scores[(ri, c)] = word_scores(fo, repairs[ri][3] - repairs[ri][2])

            n_moved = 0
            for ri, (c_lo, c_hi, w_lo, w_hi) in enumerate(repairs):
                n_c = c_hi - c_lo + 1
                n_s = w_hi - w_lo
                m = np.array([scores[(ri, c)] for c in range(c_lo, c_hi + 1)])  # [n_c, n_s]
                # anchored monotone DP: chunk ci gets span words [t_{ci-1}, t_ci)
                neg = -1e30
                dp = np.full((n_c, n_s + 1), neg)
                dp[0, 0] = 0.0
                for s in range(1, n_s + 1):
                    dp[0, s] = dp[0, s - 1] + m[0, s - 1]
                for ci in range(1, n_c):
                    dp[ci, 0] = dp[ci - 1, 0]
                    for s in range(1, n_s + 1):
                        dp[ci, s] = max(dp[ci - 1, s], dp[ci, s - 1] + m[ci, s - 1])
                best_v = dp[n_c - 1, n_s]
                # backtrack -> new chunk per span word
                new_c = np.empty(n_s, dtype=int)
                ci, s = n_c - 1, n_s
                while s > 0:
                    if ci > 0 and dp[ci, s] == dp[ci - 1, s]:
                        ci -= 1
                    else:
                        s -= 1
                        new_c[s] = c_lo + ci
                # score of the ORIGINAL assignment under the same matrix + margin acceptance
                v_orig = sum(m[w2c[w] - c_lo, w - w_lo] for w in range(w_lo, w_hi))
                moved = int(np.sum(new_c != w2c[w_lo:w_hi]))
                if moved == 0 or best_v < v_orig + self.min_move_margin * moved:
                    print(
                        f"seq {seq_idx}: span words [{w_lo},{w_hi}) chunks [{c_lo},{c_hi}]:"
                        f" keep (moved {moved}, gain {best_v - v_orig:.2f})",
                        flush=True,
                    )
                    continue
                n_moved += moved
                for c in range(c_lo, c_hi + 1):
                    idx = np.nonzero(new_c == c)[0]
                    wise[c] = [w_lo + int(idx[0]), w_lo + int(idx[-1]) + 1] if len(idx) else [-1, -1]
                w2c[w_lo:w_hi] = new_c
                print(
                    f"seq {seq_idx}: span words [{w_lo},{w_hi}) chunks [{c_lo},{c_hi}]:"
                    f" moved {moved}, gain {best_v - v_orig:.2f}",
                    flush=True,
                )
            n_moved_total += n_moved
            print(f"seq {seq_idx}: {len(flagged)} flagged words, {len(repairs)} spans, {n_moved} moved", flush=True)

            hdf_writer.insert_batch(
                np.array(wise)[None],
                seq_len=[len(cse)],
                seq_tag=[f"seq-{seq_idx}"],
                extra={"audio_chunk_start_end": np.array(cse)[None]},
            )
        print(f"total words moved: {n_moved_total}", flush=True)
        hdf_writer.close()
