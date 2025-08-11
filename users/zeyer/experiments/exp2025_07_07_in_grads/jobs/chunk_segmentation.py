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
                    f" words {cur_word_start}:{cur_word_end} (out of {len(words)})"
                )
                assert cur_word_end > cur_word_start  # need to fix heuristic if this fails...
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
                            >= (t1 - t0)[:, None],
                            0.0,
                        )
                        word_log_prob = word_log_prob.sum()  # []. assume single batch
                    else:
                        word_log_prob = None
                    exit_log_prob = log_probs[0, t0[0], model.assistant_end_token_idx]  # []. assume single batch
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
