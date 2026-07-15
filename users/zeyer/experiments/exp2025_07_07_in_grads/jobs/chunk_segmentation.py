from __future__ import annotations
from typing import Optional, Any, Dict, List, Tuple
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

            array: List[List[_Node]] = []  # [chunk_idx][rel word_idx]

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

            for cur_chunk_idx, (cur_audio_start, cur_audio_end) in enumerate(chunk_start_end):
                if cur_chunk_idx == 0 or not self.word_start_heuristic:
                    prev_array_word_idx = 0
                    cur_word_start = 0
                else:
                    # Heuristic. Look through last chunk, look out for best exit_log_prob
                    prev_array_word_idx = int(
                        torch.stack([node.accum_exit_log_prob for node in array[cur_chunk_idx - 1]]).argmax().item()
                    )
                    cur_word_start = array[cur_chunk_idx - 1][prev_array_word_idx].word_idx
                cur_word_end = len(words)  # Go to the end. Not so expensive...
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
                    exit_log_prob = log_probs[0, 0, model.assistant_end_token_id]  # []. assume single batch
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

    # Bumped to 2 when sort_by_length / batched_logprobs were added: the flags are part of the hash
    # (no __sis_hash_exclude__), so every value combination gets its own hash.
    __sis_version__ = 2

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

        self.rqmt = {"time": 40, "cpu": 2, "gpu": 1, "mem": 125}
        self.out_hdf = self.output_path("out.hdf")

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

        def _chunk_bounds(audio, samplerate):
            chunk_size_samples = math.ceil(self.chunk_size_secs * samplerate)
            cse: List[Tuple[int, int]] = []
            cur = 0
            while True:
                end = cur + chunk_size_samples
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
            group = order[group_start : group_start + self.max_batch_size]
            # per-seq state
            st = {}
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
                for i in group:
                    s = st[i]
                    if s["done"] or s["cci"] >= len(s["cse"]):
                        s["done"] = True
                        continue
                    cci = s["cci"]
                    if cci == 0 or not self.word_start_heuristic:
                        prev_idx, cws = 0, 0
                    else:
                        prev_idx = int(
                            torch.stack([n.accum_exit_log_prob for n in s["array"][cci - 1]]).argmax().item()
                        )
                        cws = s["array"][cci - 1][prev_idx].word_idx
                    cwe = len(s["words"])
                    if cws >= cwe:
                        # trailing silence: drop remaining chunks (see single-seq job for rationale)
                        assert self.word_start_heuristic
                        s["cse"] = s["cse"][:cci]
                        s["done"] = True
                        continue
                    a0, a1 = s["cse"][cci]
                    active.append(i)
                    ws_start[i] = (prev_idx, cws, cwe)
                    fwd_audio.append(torch.tensor(s["audio"][a0:a1]))
                    fwd_words.append(s["words"][cws:cwe])
                    fwd_omitted.append(cws)
                if not active:
                    break

                # One flushed line per chunk step, so the log keeps updating and a slow run is
                # distinguishable from a real hang: small chunk sizes mean ~1000+ chunks per seq, and
                # the DP is otherwise silent for a whole group (only group start / seq end print).
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

                for j, i in enumerate(active):
                    s = st[i]
                    cci = s["cci"]
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
                    for w in range(cwe - cws + 1):
                        t0, t1 = fo.target_start_end[:, w].unbind(1)
                        if all_lp is not None:
                            log_probs = all_lp[:, int(t0) : int(t1)]
                        else:
                            log_probs = model.log_probs(forward_output=fo, start=t0, end=t1)
                        word_idx = cws + w
                        if word_idx < cwe:
                            targets = batch_slice(fo.targets, (t0, t1))
                            wlp = batches_gather(log_probs, indices=targets, num_batch_dims=2)
                            wlp.masked_fill_(
                                torch.arange(wlp.shape[1], device=wlp.device)[None, :]
                                >= (t1 - t0).to(wlp.device)[:, None],
                                0.0,
                            )
                            wlp = wlp.sum()
                        else:
                            wlp = None
                        exit_lp = log_probs[0, 0, model.assistant_end_token_id]
                        if w == 0:
                            exit_lp = exit_lp + self.empty_exit_penalty
                        prev_left, prev_below = None, None
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
                words_per_chunks: List[List[int]] = [[] for _ in range(len(cse))]
                covered = 0
                for node in nodes_alignment[1:]:
                    if node.backpointer.chunk_idx == node.chunk_idx:
                        assert node.word_idx == node.backpointer.word_idx + 1
                        words_per_chunks[node.chunk_idx].append(node.word_idx - 1)
                        assert covered == node.word_idx - 1
                        covered += 1
                    else:
                        assert node.chunk_idx == node.backpointer.chunk_idx + 1
                        assert node.word_idx == node.backpointer.word_idx
                assert covered == len(words)
                wise = [(ws[0], ws[-1] + 1) if ws else (-1, -1) for ws in words_per_chunks]
                assert len(wise) == len(cse)
                results[i] = (wise, cse)
                print(f"  seq {i}: words per chunks = {wise}", flush=True)

        # Write in original dataset order (the metric job reads seqs positionally); groups may have
        # been reordered by sort_by_length.
        for i in range(num_seqs):
            wise, cse = results[i]
            hdf_writer.insert_batch(
                np.array(wise)[None],
                seq_len=[len(cse)],
                seq_tag=[f"seq-{i}"],
                extra={"audio_chunk_start_end": np.array(cse)[None]},
            )
        hdf_writer.close()


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
    - ``chunk_idx_mae``: mean absolute error
      between the assigned chunk index and the nearest covering chunk index --
      a softer signal than accuracy,
      distinguishing near-misses (off by one chunk, near a boundary) from far misses.

    A real per-frame word boundary (and thus WBE, comparable to WhisperX etc.)
    belongs in a separate downstream step
    once there's an actual within-chunk decode or forced alignment.
    """

    __sis_version__ = 1

    def __init__(
        self,
        *,
        chunk_seg_hdf: tk.Path,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        dataset_offset_factors: int,
        aggregation: str = "micro",
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
        """
        super().__init__()
        self.chunk_seg_hdf = chunk_seg_hdf
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.dataset_offset_factors = dataset_offset_factors
        self.aggregation = aggregation

        self.out_accuracy = self.output_var("accuracy.txt")
        self.out_chunk_idx_mae = self.output_var("chunk_idx_mae.txt")
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

        utt_accs: List[List[float]] = []  # per utt, per word: 1.0 correct / 0.0 wrong
        utt_idx_errs: List[List[int]] = []  # per utt, per word: chunk-index error

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

            accs, idx_errs = [], []
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

            print(f"** seq {seq_idx}, {len(words)=}, acc={float(np.mean(accs)):.3f}")
            utt_accs.append(accs)
            utt_idx_errs.append(idx_errs)

        if self.aggregation == "micro":
            accuracy = float(np.mean([a for accs in utt_accs for a in accs]))
            chunk_idx_mae = float(np.mean([e for errs in utt_idx_errs for e in errs]))
        elif self.aggregation == "macro":
            accuracy = float(np.mean([np.mean(accs) for accs in utt_accs]))
            chunk_idx_mae = float(np.mean([np.mean(errs) for errs in utt_idx_errs]))
        else:
            raise ValueError(f"invalid aggregation {self.aggregation!r}")

        metrics = {"accuracy": accuracy, "chunk_idx_mae": chunk_idx_mae, "aggregation": self.aggregation}
        print("CORPUS METRICS:", metrics)
        self.out_accuracy.set(accuracy)
        self.out_chunk_idx_mae.set(chunk_idx_mae)
        self.out_metrics.set(metrics)
