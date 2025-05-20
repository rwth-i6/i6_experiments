import functools
from typing import Optional, Any, Dict
from sisyphus import Job, Task, tk
from i6_core.util import uopen
from .huggingface import get_content_dir_from_hub_cache_dir


def phi4mi_recog_score_wer(
    *, dataset_dir: Optional[tk.Path] = None, dataset_name: str = "tedlium", dataset_split: str = "test"
):
    from i6_experiments.users.zeyer.datasets.huggingface.extract_text import ExtractTextFromHuggingFaceDatasetJob
    from i6_experiments.users.zeyer.datasets.huggingface.open_asr_leaderboard import (
        text_dict_normalize_file,
        download_esb_datasets_test_only_sorted,
    )
    from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import sclite_score_hyps_to_ref

    dl_phi4mi_model_dir = download_phi4multimodal_model()
    if dataset_dir is None:
        dataset_dir = download_esb_datasets_test_only_sorted()

    recog_job = Phi4MultimodalRecognitionJob(
        model_dir=dl_phi4mi_model_dir,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    tk.register_output(f"phi4mi.{dataset_name}.{dataset_split}.recog.txt.py.gz", recog_job.out_recog)
    ref_text_job = ExtractTextFromHuggingFaceDatasetJob(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    tk.register_output(f"{dataset_name}.{dataset_split}.ref.txt.py.gz", ref_text_job.out_text)
    # Should get: 2.89% WER with Phi4MI on Tedlium (test).
    # Without normalization: 6.25%
    # With hyp normalization: 11.41%
    # With ref+hyp normalization: 2.88%
    tk.register_output(
        f"phi4mi.{dataset_name}.{dataset_split}.wer.txt",
        sclite_score_hyps_to_ref(
            text_dict_normalize_file(recog_job.out_recog), ref_text_dict=text_dict_normalize_file(ref_text_job.out_text)
        ).main_measure_value,
    )


@functools.cache
def download_phi4multimodal_model() -> tk.Path:
    from .huggingface import DownloadHuggingFaceRepoJob

    dl_phi4mi = DownloadHuggingFaceRepoJob(model_id="microsoft/Phi-4-multimodal-instruct")
    tk.register_output("phi4mi-model", dl_phi4mi.out_hub_cache_dir)
    return dl_phi4mi.out_hub_cache_dir


class Phi4MultimodalRecognitionJob(Job):
    """
    Do recognition with Phi4Multimodal.
    Store in our common TextDict format.

    https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
    https://huggingface.co/microsoft/Phi-4-multimodal-instruct
    https://github.com/huggingface/open_asr_leaderboard/blob/main/phi/run_phi4_multimodal.sh
    https://github.com/huggingface/open_asr_leaderboard/blob/main/phi/run_eval.py
    """

    def __init__(
        self,
        *,
        model_dir: tk.Path,
        speech_prompt: str = "Transcribe the audio clip into text.",
        max_new_tokens: Optional[int] = 512,
        num_beams: int = 1,
        dataset_dir: tk.Path,
        dataset_name: str,
        dataset_split: str,
        returnn_root: Optional[tk.Path] = None,
        batch_size: int = 16,
    ):
        """
        :param model_dir:
        :param speech_prompt:
        :param max_new_tokens:
        :param num_beams:
        :param dataset_dir: e.g. via DownloadHuggingFaceRepoJobV2 or DownloadAndPrepareHuggingFaceDatasetJob
            of the esb-datasets, which is also used for the OpenASRLeaderboard
            (or anything compatible to that).
        :param dataset_name:
        :param dataset_split:
        :param returnn_root:
        :param batch_size:
        """
        super().__init__()
        self.model_dir = model_dir
        self.speech_prompt = speech_prompt
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.returnn_root = returnn_root
        self.batch_size = batch_size

        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 125}

        self.out_recog = self.output_path("recog.txt.py.gz")  # gzipped textdict

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import time

        os.environ["HF_HUB_CACHE"] = "/<on_purpose_invalid_hf_hub_cache_dir>"

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        print("Import transformers / other libs...")
        start_time = time.time()

        import torch
        import returnn.util.basic as util
        from returnn.util import better_exchook

        # os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
        better_exchook.install()

        try:
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            pass

        device_str = "cuda"
        dev = torch.device(device_str)

        # done also in https://github.com/huggingface/open_asr_leaderboard/blob/main/phi/run_eval.py
        torch.set_float32_matmul_precision("high")

        def _report_dev_memory_stats():
            dev = torch.device(device_str)
            if dev.type == "cuda":
                stats = [
                    f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                    f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                    f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                    f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
                ]
                print(f"Memory usage ({device_str}):", " ".join(stats))

        from datasets import load_dataset, Audio
        from transformers import AutoProcessor, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

        # from https://github.com/huggingface/open_asr_leaderboard/blob/main/phi/run_eval.py
        class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
            """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

            def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
                """Initialize the multiple token batch stopping criteria.

                Args:
                    stop_tokens: Stop-tokens.
                    batch_size: Batch size.

                """

                self.stop_tokens = stop_tokens
                self.max_stop_tokens = stop_tokens.shape[-1]
                self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                # Only gather the maximum number of inputs compatible with stop tokens
                # and checks whether generated inputs are equal to `stop_tokens`
                generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
                equal_generated_inputs = torch.all(generated_inputs, dim=2)

                # Mark the position where a stop token has been produced for each input in the batch,
                # but only if the corresponding entry is not already set
                sequence_idx = torch.any(equal_generated_inputs, dim=1)
                sequence_set_mask = self.stop_tokens_idx == 0
                self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

                return torch.all(self.stop_tokens_idx)

        print(f"({time.time() - start_time} secs)")
        print("Loading model...")
        start_time = time.time()
        model_dir = get_content_dir_from_hub_cache_dir(self.model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype="auto", trust_remote_code=True, device_map=device_str
        ).to(dev)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalForCausalLM

        model: Phi4MultimodalForCausalLM  # just as an example...
        print(model)
        print("model.dtype:", model.dtype)
        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        # print("\n--- AUDIO PROCESSING ---")
        # https://huggingface.co/microsoft/Phi-4-multimodal-instruct
        # https://github.com/huggingface/open_asr_leaderboard/blob/main/phi/run_eval.py

        prompt = f"<|user|><|audio_1|>{self.speech_prompt}<|end|><|assistant|>"
        print(f">>> Prompt: {prompt!r}")

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            # We don't need the logits here. There is currently no way to not compute them,
            # so num_logits_to_keep=1 is the best we can do.
            "num_logits_to_keep": 1,
        }

        stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
        stop_tokens_ids = processor.tokenizer(
            stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
        )["input_ids"]
        stop_tokens_ids = stop_tokens_ids.to(model.device)

        def benchmark(batch, min_new_tokens=None):
            # Load audio inputs
            audios = [(audio["array"], audio["sampling_rate"]) for audio in batch["audio"]]
            minibatch_size = len(audios)
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=self.num_beams * minibatch_size)]
            )

            # START TIMING
            start_time = time.time()

            with torch.autocast(model.device.type, enabled=True):
                inputs = processor(text=[prompt] * minibatch_size, audios=audios, return_tensors="pt").to(dev)

                # Model Inference
                pred_ids = model.generate(
                    **inputs,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    **gen_kwargs,
                    min_new_tokens=min_new_tokens,
                )

            # Gather the sequence index of the stop token
            stop_tokens_idx = gen_kwargs["stopping_criteria"][0].stop_tokens_idx.reshape(minibatch_size, -1)[:, 0]

            # If a stop token was produced, we need to remove its length from the found index,
            # however there might be a chance that the stop token was not produced and the index
            # returned is the length of the generated sequence
            stop_tokens_idx = torch.where(
                stop_tokens_idx > 0,
                stop_tokens_idx - stop_tokens_ids.shape[-1],
                pred_ids.shape[-1],
            )

            # Convert token ids to text transcription
            pred_text = [
                processor.decode(
                    _pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for _pred_ids, _stop_tokens_idx in zip(pred_ids, stop_tokens_idx)
            ]

            # END TIMING
            runtime = time.time() - start_time
            print("Runtime:", runtime, "secs, or", runtime / minibatch_size, "secs per sample")

            batch["predictions"] = pred_text  # without normalizer here
            # batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
            return batch

        # https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/data_utils.py
        # data_utils.load_data(args) is just load_dataset, nothing else
        dataset = load_dataset(
            get_content_dir_from_hub_cache_dir(self.dataset_dir),
            name=self.dataset_name,
            split=self.dataset_split,
            token=True,
        )
        print(f"Dataset: {dataset}")

        # This is dataset = data_utils.prepare_data(dataset):
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))  # Re-sample to 16kHz

        dataset = dataset.filter(
            lambda ref: ref.strip() not in {"", "ignore time segment in scoring"}, input_columns=["text"]
        )

        dataset = dataset.map(benchmark, batch_size=self.batch_size, batched=True, remove_columns=["audio"])

        # See SearchOutputRawReplaceJob and co.
        with uopen(self.out_recog.get_path(), "wt") as out:
            out.write("{\n")
            for result in dataset:
                # https://huggingface.co/datasets/esb/datasets
                seq_tag = f"{self.dataset_name}/{self.dataset_split}/{result['id']}"
                pred = result["predictions"]
                assert isinstance(pred, str)
                out.write(f"{seq_tag!r}: {pred!r},\n")
            out.write("}\n")
