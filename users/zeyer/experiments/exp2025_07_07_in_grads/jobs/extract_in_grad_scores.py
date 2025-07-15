from typing import Optional, Any, Dict, List
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class ExtractInGradsFromModelJob(Job):
    """
    Extract grads to input for each output word with some model

    This was derived/generalized from
    :class:`i6_experiments.users.zeyer.experiments.exp2025_05_05_align.ExtractInGradsFromPhi4MultimodalInstructJob`
    and
    TODO
    :class:`i6_experiments.users.zeyer.experiments.exp2025_05_05_align.ExtractInGradsFromPhi4MultimodalInstructLongFormJob`
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        model_config: Dict[str, Any],
        mult_grad_by_inputs: bool,
        attr_reduction: str,  # "sum", "L2", "L1", ...
    ):
        """
        :param dataset_dir: hub cache dir, e.g. via DownloadHuggingFaceRepoJobV2. for load_dataset
        :param dataset_key: e.g. "train", "test", whatever the dataset provides
        :param returnn_root: for some utils. version of RETURNN should not really matter
        :param model_config:
        :param mult_grad_by_inputs:
        :param attr_reduction:

        TODO We could extend this by `attr_method` or so, to support IntegratedGradients or others
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
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

        hdf_writer = SimpleHDFWriter(self.out_hdf.get_path(), dim=1, ndim=2, extra_type={"sizes": (2, 2, "int32")})

        # Iter over data

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Dataset keys:", ds.keys())
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

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
            transcription = " ".join(data["word_detail"]["utterance"])
            print(f"seq {seq_idx}, {audio.shape=}, {samplerate=}, {transcription!r}")

            if seq_idx == 0:
                print("data keys:", data.keys())

            start_time = time.time()
            print("** Forwarding")
            num_words = len(data["word_detail"]["utterance"])
            assert len(transcription.split(" ")) == num_words

            forward_output: ForwardOutput = model(
                raw_inputs=audio,
                raw_inputs_sample_rate=samplerate,
                raw_input_seq_lens=torch.tensor([len(audio)]),
                raw_targets=[data["word_detail"]["utterance"]],
                raw_target_seq_lens=torch.tensor([len(data["word_detail"]["utterance"])]),
            )

            # noinspection PyShadowingNames
            def _calc_input_grads(w: int, *, report_mem: bool = False, forward_output: ForwardOutput) -> torch.Tensor:
                t0, t1 = forward_output.target_start_end[:, w].unbind(1)  # [B], [B]
                loss = model.log_probs(forward_output=forward_output, start=t0, end=t1)  # [B,t1-t0,V]
                targets = batch_slice(forward_output.targets, (t0, t1))  # [B,t1-t0]->V
                loss = batches_gather(loss, indices=targets, num_batch_dims=2)  # [B,t1-t0]
                loss.masked_fill_(torch.arange(loss.shape[1], device=loss.device)[None, :] >= (t1 - t0)[:, None], 0.0)
                (grad,) = torch.autograd.grad(loss.sum(), forward_output.inputs, retain_graph=True)
                del loss

                if report_mem:
                    report_dev_memory_stats(dev)

                with torch.no_grad():
                    attr = batch_slice(grad.float(), forward_output.input_slice_start_end)  # [B,T,F]
                    if self.mult_grad_by_inputs:
                        e = batch_slice(forward_output.inputs.float(), forward_output.input_slice_start_end)
                        attr *= e
                    attr = attr_reduce_func(attr)  # [B,T]
                    return attr

            print("** Calculating grads")
            num_input_frames = forward_output.get_inputs_seq_lens_sliced()[0].item()
            grad_mat: List[torch.Tensor] = []
            for w in range(num_words):
                grads = _calc_input_grads(w, report_mem=w in {0, num_words - 1}, forward_output=forward_output)
                assert grads.shape == (1, num_input_frames)
                grad_mat.append(grads[0])
            grad_mat_ = torch.stack(grad_mat)  # [num_words,num_input_frames]
            # Convert to Numpy and flatten and add dummy dim at the end to have it compatible for the HDF.
            # Also add dummy batch dim in the beginning (for insert_batch).
            grad_mat__ = grad_mat_.detach().cpu().numpy().flatten()[None, :, None]

            print("** Storing to HDF")
            hdf_writer.insert_batch(
                grad_mat__,
                seq_len=[num_words * num_input_frames],
                seq_tag=[f"seq-{seq_idx}"],
                extra={
                    "sizes": np.array([num_words, num_input_frames])[None, None],
                    # Mapping the input frames to audio samples (start/end for each input frame).
                    "audio_start_end": forward_output.input_raw_start_end.detach().cpu().numpy(),
                },
                # TODO chunk
            )

            print("** Freeing")
            del forward_output  # not needed anymore now
            gc.collect()
            report_dev_memory_stats(dev)
            print(f"({time.time() - start_time} secs for the seq)")

        hdf_writer.close()

        # better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())
