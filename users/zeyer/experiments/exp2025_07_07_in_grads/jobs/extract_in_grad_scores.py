from typing import Optional, Union, Any, Dict, List
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class ExtractInGradsFromPhi4MultimodalInstructJob(Job):
    """
    ... ()
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        model_config: Dict[str, Any],
        attr_method: Union[str, Dict[str, Any]],  # "IntegratedGradients" or {"type": "IntegratedGradients"}
        attr_reduction: str,  # "sum", "L2", "L1", ...
    ):
        """
        :param dataset_dir: hub cache dir, e.g. via DownloadHuggingFaceRepoJobV2. for load_dataset
        :param dataset_key: e.g. "train", "test", whatever the dataset provides
        :param returnn_root: for some utils. version of RETURNN should not really matter
        :param model_config:
        :param attr_method:
        :param attr_reduction:
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.model_config = model_config
        self.attr_method = attr_method
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
        from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats

        # os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
        better_exchook.install()

        try:
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            pass

        from .models import make_model
        from .grad_score_types import get_grad_score_func

        grad_score_func = get_grad_score_func(self.grad_score_type)

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
            assert len(transcription.split(" ")) == len(data["word_detail"]["utterance"])

            # Not needed here, as we already have only the selected audio embedding part.
            src_start, src_end = None, None

            def _calc_input_grads(t0, t1, *, report_mem: bool = False) -> torch.Tensor:
                logits = model.lm_head(last_out[:, t0 - 1 : t1 - 1])
                logits = logits.float()
                if logits.shape[0] > 1:
                    logits = logits.mean(dim=0, keepdim=True)
                fake_logits = logits + (-logits).detach()  # zero, but grads will go to logits

                loss = torch.nn.functional.cross_entropy(
                    fake_logits[0], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
                )
                loss.backward(retain_graph=True)
                (grad,) = torch.autograd.grad(loss, inputs_embeds, retain_graph=True)
                if report_mem:
                    _report_dev_memory_stats()
                del fake_logits, logits
                grad, inputs_embeds.grad = inputs_embeds.grad, None
                with torch.no_grad():
                    e = inputs_embeds.float()[0, src_start:src_end]
                    grad = grad.float()[0, src_start:src_end]
                    return grad_score_func(e, grad)

            print("** Calculating grads")
            num_input_frames = inputs_embeds[0, src_start:src_end].shape[0]
            num_words = len(words_start_end)
            grad_mat: List[torch.Tensor] = []
            for w, (t0, t1) in enumerate(words_start_end):
                grads = _calc_input_grads(t0, t1, report_mem=w in {0, num_words - 1})
                assert grads.shape == (num_input_frames,)
                grad_mat.append(grads)
            grad_mat_ = torch.stack(grad_mat)  # [num_words,num_input_frames]
            # Convert to Numpy and flatten and add dummy dim at the end to have it compatible for the HDF.
            # Also add dummy batch dim in the beginning (for insert_batch).
            grad_mat__ = grad_mat_.detach().cpu().numpy().flatten()[None, :, None]

            print("** Freeing")
            del last_out, inputs_embeds, inputs  # not needed anymore now
            gc.collect()
            _report_dev_memory_stats()
            print(f"({time.time() - start_time} secs for the seq)")

            hdf_writer.insert_batch(
                grad_mat__,
                seq_len=[num_words * num_input_frames],
                seq_tag=[f"seq-{seq_idx}"],
                extra={"sizes": np.array([num_words, num_input_frames])[None, None]},
            )

        hdf_writer.close()

        # better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())
