from typing import Optional, Any, Dict, List, Tuple
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class ExtractInGradsFromModelJob(Job):
    """
    Extract grads to input for each output word with some model

    This was derived/generalized from
    :class:`i6_experiments.users.zeyer.experiments.exp2025_05_05_align.ExtractInGradsFromPhi4MultimodalInstructJob`
    and
    :class:`i6_experiments.users.zeyer.experiments.exp2025_05_05_align.ExtractInGradsFromPhi4MultimodalInstructLongFormJob`
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        chunk_segmentation_hdf: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
        model_config: Dict[str, Any],
        mult_grad_by_inputs: bool,
        attr_reduction: str,
    ):
        """
        :param dataset_dir: hub cache dir, e.g. via DownloadHuggingFaceRepoJobV2. for load_dataset
        :param dataset_key: e.g. "train", "test", whatever the dataset provides
        :param chunk_segmentation_hdf: via :class:`ChunkSegmentationFromModelJob`
        :param returnn_root: for some utils. version of RETURNN should not really matter
        :param model_config:
        :param mult_grad_by_inputs:
        :param attr_reduction: "sum", "L2", "L1", ... see :func:`get_attr_reduce_func`

        TODO We could extend this by `attr_method` or so, to support IntegratedGradients or others
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.chunk_segmentation_hdf = chunk_segmentation_hdf
        self.returnn_root = returnn_root
        self.model_config = model_config
        self.mult_grad_by_inputs = mult_grad_by_inputs
        self.attr_reduction = attr_reduction

        self.rqmt = {"time": 40, "cpu": 2, "gpu": 1, "mem": 125}

        self.out_hdf = self.output_path("out.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import time
        import gc

        os.environ["HF_HUB_CACHE"] = "/<on_purpose_invalid_hf_hub_cache_dir>"

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

        # os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
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
            # grads: [num_chunks * ~chunk_num_words * ~chunk_num_input_frames,1]
            dim=1,
            ndim=2,
            extra_type={
                "audio_frames_start_end": (2, 2, "int32"),  # [num_chunks * ~chunk_num_input_frames,2]
                "num_input_frames": (1, 2, "int32"),  # [num_chunks,1]
                "num_words": (1, 2, "int32"),  # [num_chunks,1]
                # For debugging/verification.
                "log_probs_per_word": (1, 2, "float32"),  # [num_chunks * ~chunk_num_words,1]
                "exit_log_probs": (1, 2, "float32"),  # [num_chunks,1]
            },
        )

        # Iter over data

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
            # For TIMIT: but not used currently...
            def _tag(i, d):
                return f"timit-{self.dataset_key}-{i}-{d['dialect_region']}-{d['speaker_id']}-{d['id']}"

            # Buckeye:
            # In [59]: len(ds_buckeye["val"][0]["audio"]["array"])
            # Out[59]: 9969854
            #
            # In [60]: ds_buckeye["val"][0]["word_detail"]["stop"][-1]
            # Out[60]: 9969

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
                chunk_audio_start_end = chunk_segmentation_hdf_ds.get_data(seq_idx, "audio_chunk_start_end")  # [C,2]
                chunk_words_indices_start_end = chunk_segmentation_hdf_ds.get_data(seq_idx, "data")  # [C,2]
                num_chunks = chunk_audio_start_end.shape[0]
                assert num_chunks == chunk_words_indices_start_end.shape[0]
                assert chunk_words_indices_start_end[:, 1].max() == len(words)
            else:
                num_chunks = 1
                chunk_audio_start_end = np.array([[0, len(audio)]], dtype=np.int32)  # [C,2]
                chunk_words_indices_start_end = np.array([[0, num_words]], dtype=np.int32)  # [C,2]

            # Per chunk, we will have:
            grad_mat: List[torch.Tensor] = []
            audio_frames_start_end: List[torch.Tensor] = []
            num_input_frames: List[int] = []
            num_words_: List[int] = []
            # also store log probs for debugging/verification
            log_probs: List[torch.Tensor] = []  # per word
            exit_log_probs: List[torch.Tensor] = []  # exit log prob, for EOS

            for chunk_idx in range(num_chunks):
                start_time = time.time()
                audio_start, audio_end = chunk_audio_start_end[chunk_idx]
                audio_start, audio_end = max(audio_start, 0), min(audio_end, len(audio))  # just in case
                words_start, words_end = chunk_words_indices_start_end[chunk_idx]
                words_start, words_end = max(words_start, 0), min(words_end, num_words)  # just in case
                print(
                    f"** Forwarding chunk {chunk_idx + 1}/{num_chunks}"
                    f" audio {audio_start / samplerate}-{audio_end / samplerate}"
                    f" words {words_start}-{words_end} ({words[words_start:words_end]!r})"
                )

                forward_output: ForwardOutput = model(
                    raw_inputs=torch.tensor(audio[audio_start:audio_end])[None],  # [B=1,T]
                    raw_inputs_sample_rate=samplerate,
                    raw_input_seq_lens=torch.tensor([audio_end - audio_start]),  # [B=1]
                    raw_targets=[words[words_start:words_end]],  # [B=1,num_words]
                    raw_target_seq_lens=torch.tensor([words_end - words_start]),  # [B=1]
                    omitted_prev_context=torch.tensor([words_start]),
                )

                # noinspection PyShadowingNames
                def _calc_log_probs_and_input_grads(
                    w: int, *, report_mem: bool = False, forward_output: ForwardOutput, no_grad: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    t0, t1 = forward_output.target_start_end[:, w].unbind(1)  # [B], [B]
                    loss = model.log_probs(forward_output=forward_output, start=t0, end=t1)  # [B,t1-t0,V]
                    targets = batch_slice(forward_output.targets, (t0, t1))  # [B,t1-t0]->V
                    loss = batches_gather(loss, indices=targets, num_batch_dims=2)  # [B,t1-t0]
                    loss.masked_fill_(
                        torch.arange(loss.shape[1], device=loss.device)[None, :] >= (t1 - t0)[:, None], 0.0
                    )
                    loss = loss.sum(dim=-1)  # [B]
                    if no_grad:
                        return loss, None
                    (grad,) = torch.autograd.grad(loss.sum(), forward_output.inputs, retain_graph=True)
                    loss = loss.detach()  # detach to avoid memory leak. we return this here below

                    if report_mem:
                        report_dev_memory_stats(dev)

                    with torch.no_grad():
                        attr = batch_slice(grad.float(), forward_output.input_slice_start_end)  # [B,T,F]
                        if self.mult_grad_by_inputs:
                            e = batch_slice(forward_output.inputs.float(), forward_output.input_slice_start_end)
                            attr *= e
                        attr = attr_reduce_func(attr)  # [B,T]
                        return loss, attr

                print("** Calculating grads")
                chunk_num_input_frames = forward_output.get_inputs_seq_lens_sliced()[0].item()
                num_input_frames.append(chunk_num_input_frames)
                num_words_.append(words_end - words_start)
                chunk_grad_mat: List[torch.Tensor] = []
                chunk_log_probs: List[torch.Tensor] = []  # also store log probs for debugging/verification
                for w in range(words_end - words_start):
                    word_log_probs, grads = _calc_log_probs_and_input_grads(
                        w, forward_output=forward_output, report_mem=w in {0, words_end - words_start - 1}
                    )
                    assert grads.shape == (1, chunk_num_input_frames) and word_log_probs.shape == (1,)
                    chunk_grad_mat.append(grads[0])
                    chunk_log_probs.append(word_log_probs[0])
                chunk_grad_mat_ = torch.stack(chunk_grad_mat)  # [chunk_num_words,chunk_num_input_frames]
                chunk_grad_mat_ = chunk_grad_mat_.flatten()  # [chunk_num_words * chunk_num_input_frames]
                grad_mat.append(chunk_grad_mat_)
                log_probs.append(torch.stack(chunk_log_probs))  # [chunk_num_words]

                # collect each [chunk_num_input_frames, 2] -> sample start/end
                audio_frames_start_end.append(forward_output.input_raw_start_end[0])

                # Get exit (EOS) log prob. Assume one behind last word is EOS.
                with torch.no_grad():
                    chunk_exit_log_prob, _ = _calc_log_probs_and_input_grads(
                        w=words_end - words_start, forward_output=forward_output, no_grad=True
                    )
                    assert chunk_exit_log_prob.shape == (1,)
                exit_log_probs.append(chunk_exit_log_prob[0])

                print("** Freeing")
                del forward_output  # not needed anymore now
                gc.collect()
                report_dev_memory_stats(dev)
                print(f"({time.time() - start_time} secs for the seq)")

            grad_mat_ = torch.concat(grad_mat)  # [num_chunks * ~chunk_num_words * ~chunk_num_input_frames]
            audio_frames_start_end_ = torch.concat(audio_frames_start_end)  # [num_chunks * chunk_num_input_frames,2]
            num_input_frames_ = torch.tensor(num_input_frames)  # [num_chunks]
            num_words__ = torch.tensor(num_words_)  # [num_chunks]
            log_probs_ = torch.concat(log_probs)  # [num_chunks * ~chunk_num_words]
            exit_log_probs_ = torch.stack(exit_log_probs)  # [num_chunks]

            print("** Storing to HDF")
            hdf_writer.insert_batch(
                # Convert to Numpy and add dummy dim at the end to have it compatible for the HDF.
                # Also add dummy batch dim in the beginning (for insert_batch).
                # [1,num_chunks * ~chunk_num_words * ~chunk_num_input_frames,1]
                grad_mat_.cpu().numpy()[None, :, None],
                seq_len=[len(grad_mat_)],
                seq_tag=[f"seq-{seq_idx}"],
                extra={
                    # Mapping the input frames to audio samples (start/end for each input frame).
                    # [1,num_chunks * ~chunk_num_input_frames,2]
                    "audio_frames_start_end": audio_frames_start_end_.cpu().numpy()[None],
                    "num_input_frames": num_input_frames_.cpu().numpy()[None, :, None],  # [1,num_chunks,1]
                    "num_words": num_words__[None, :, None],  # [1,num_chunks,1]
                    # Some extra info, e.g. for debugging/verification.
                    # [1,num_chunks * ~chunk_num_words,1]
                    "log_probs_per_word": log_probs_.cpu().numpy()[None, :, None],
                    "exit_log_probs": exit_log_probs_.cpu().numpy()[None, :, None],  # [1,num_chunks,1]
                },
            )

        hdf_writer.close()

        # better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())
