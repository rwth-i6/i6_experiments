"""
See :class:`CrisperWhisperRecognitionJob`.

https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
https://huggingface.co/nyrahealth/CrisperWhisper
https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh
https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_eval.py
https://github.com/nyrahealth/CrisperWhisper

"""

from typing import Optional, Any, Dict
import functools
from sisyphus import Job, Task, tk
from i6_core.util import uopen
from .huggingface import get_content_dir_from_hub_cache_dir


def crisper_whisper_recog_score_wer(
    *, dataset_dir: Optional[tk.Path] = None, dataset_name: str = "tedlium", dataset_split: str = "test"
):
    """
    Do recognition with CrisperWhisper, mostly consistent to OpenASRLeaderboard.
    See :class:`CrisperWhisperRecognitionJob` below.
    """
    from i6_experiments.users.zeyer.datasets.huggingface.extract_text import ExtractTextFromHuggingFaceDatasetJob
    from i6_experiments.users.zeyer.datasets.huggingface.open_asr_leaderboard import (
        text_dict_normalize_file,
        download_esb_datasets_test_only_sorted,
    )
    from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import sclite_score_hyps_to_ref

    model_dir = download_crisper_whisper_model()
    if dataset_dir is None:
        dataset_dir = download_esb_datasets_test_only_sorted()

    recog_job = CrisperWhisperRecognitionJob(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        batch_size=1,  # TODO test...
        dtype="bfloat16",  # TODO test...
    )
    tk.register_output(f"crisper_whisper.{dataset_name}.{dataset_split}.recog.txt.py.gz", recog_job.out_recog)
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
        f"crisper_whisper.{dataset_name}.{dataset_split}.wer.txt",
        sclite_score_hyps_to_ref(
            text_dict_normalize_file(recog_job.out_recog), ref_text_dict=text_dict_normalize_file(ref_text_job.out_text)
        ).main_measure_value,
    )


@functools.cache
def download_crisper_whisper_model() -> tk.Path:
    from .huggingface import DownloadHuggingFaceRepoJobV2

    dl_model = DownloadHuggingFaceRepoJobV2(repo_id="nyrahealth/CrisperWhisper", repo_type="model")
    tk.register_output("crisper-whisper-model", dl_model.out_hub_cache_dir)
    return dl_model.out_hub_cache_dir


class CrisperWhisperRecognitionJob(Job):
    """
    Do recognition with CrisperWhisper.
    Store in our common TextDict format.

    https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
    https://huggingface.co/nyrahealth/CrisperWhisper
    https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_whisper.sh
    https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_eval.py
    https://github.com/nyrahealth/CrisperWhisper
    """

    __sis_version__ = 3

    def __init__(
        self,
        *,
        model_dir: tk.Path,
        max_new_tokens: Optional[int] = None,
        language: str = "en",
        dataset_dir: tk.Path,
        dataset_name: str,
        dataset_split: str,
        returnn_root: Optional[tk.Path] = None,
        batch_size: int = 32,
        dtype: str = "auto",
    ):
        """
        :param model_dir:
        :param max_new_tokens:
        :param language:
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
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.returnn_root = returnn_root
        self.batch_size = batch_size
        self.dtype = dtype

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

        # done also in https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_eval.py
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

        import datasets

        print("Datasets version:", datasets.__version__)

        import transformers

        print("Transformers version:", transformers.__version__)

        from datasets import load_dataset, Audio
        from transformers import (
            AutoConfig,
            AutoModelForSpeechSeq2Seq,
            AutoModelForCTC,
            AutoProcessor,
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
        )

        print(f"({time.time() - start_time} secs)")
        print("Loading model...")
        start_time = time.time()
        model_dir = get_content_dir_from_hub_cache_dir(self.model_dir)

        config = AutoConfig.from_pretrained(model_dir)
        cls_model = AutoModelForSpeechSeq2Seq if type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING else AutoModelForCTC
        model = cls_model.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=self.dtype, device_map=device_str
        ).to(dev)
        processor = AutoProcessor.from_pretrained(model_dir)
        model_input_name = processor.model_input_names[0]

        gen_kwargs: Optional[Dict[str, Any]] = None
        if model.can_generate():
            gen_kwargs = {"max_new_tokens": self.max_new_tokens}
            # for multilingual Whisper-checkpoints we see a definitive WER boost by setting the language and task args
            if getattr(model.generation_config, "is_multilingual"):
                gen_kwargs["language"] = self.language
                gen_kwargs["task"] = "transcribe"
        elif self.max_new_tokens:
            raise ValueError("`max_new_tokens` should only be set for auto-regressive models, but got a CTC model.")

        from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
        from transformers.models.whisper.processing_whisper import WhisperProcessor

        model: WhisperForConditionalGeneration  # just as an example...
        processor: WhisperProcessor
        print(model)
        print("model.dtype:", model.dtype)
        print("model.device:", model.device)
        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        def benchmark(batch):
            # Load audio inputs
            audios = [audio["array"] for audio in batch["audio"]]
            minibatch_size = len(audios)

            # START TIMING
            start_time = time.time()

            if not model.can_generate():  # or len(audios[0]) > processor.feature_extractor.n_samples:
                # 1.2 Either CTC pre-processing (normalize to mean 0, std 1), or long-form Whisper processing
                inputs = processor(
                    audios,
                    sampling_rate=16_000,
                    truncation=False,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                )
            else:
                # 1.3 Standard Whisper processing: pad audios to 30-seconds and converted to log-mel
                inputs = processor(audios, sampling_rate=16_000, return_tensors="pt", device=dev)

            inputs = inputs.to(dev)
            inputs[model_input_name] = inputs[model_input_name].to(model.dtype)

            # TODO currently i get:
            #   You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50360]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
            #   The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

            # 2. Model Inference
            if model.can_generate():
                # 2.1 Auto-regressive generation for encoder-decoder models
                pred_ids = model.generate(**inputs, **gen_kwargs)
            else:
                # 2.2. Single forward pass for CTC
                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred_ids = logits.argmax(-1)

            # 3.2 Convert token ids to text transcription
            pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

            # END TIMING
            runtime = time.time() - start_time
            print("Runtime:", runtime, "secs, or", runtime / minibatch_size, "secs per sample")

            batch["predictions"] = pred_text
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
