"""
Continuation of :mod:`exp24_09_16_grad_align`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Dict, List, Tuple
from sisyphus import tk, Job, Task
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    DownloadHuggingFaceRepoJobV2,
    get_model_dir_from_hub_cache_dir,
)

if TYPE_CHECKING:
    import torch
    import numpy as np


def py():
    dl_aya = DownloadHuggingFaceRepoJob(model_id="CohereLabs/aya-expanse-32b")
    tk.register_output("aya-model", dl_aya.out_hub_cache_dir)

    # gen_aya = GenAya(hub_cache_dir=dl_aya.out_hub_cache_dir)
    # tk.register_output("aya-gen", gen_aya.out)

    dl_phi4mi = DownloadHuggingFaceRepoJob(model_id="microsoft/Phi-4-multimodal-instruct")
    tk.register_output("phi4mi-model", dl_phi4mi.out_hub_cache_dir)

    dl_ds_buckeye = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/buckeye", repo_type="dataset")
    tk.register_output("buckeye-dataset", dl_ds_buckeye.out_hub_cache_dir)

    dl_ds_timit = DownloadHuggingFaceRepoJobV2(repo_id="nh0znoisung/timit", repo_type="dataset")
    tk.register_output("timit-dataset", dl_ds_timit.out_hub_cache_dir)

    gen_phi4mi = GenPhi4MultimodalInstruct(
        model_dir=dl_phi4mi.out_hub_cache_dir,
        datasets={"buckeye": dl_ds_buckeye.out_hub_cache_dir, "timit": dl_ds_timit.out_hub_cache_dir},
    )
    tk.register_output("aya-gen", gen_phi4mi.out)


class GenAya(Job):
    def __init__(self, *, hub_cache_dir: tk.Path):
        """
        :param hub_cache_dir: e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        """
        super().__init__()
        self.hub_cache_dir = hub_cache_dir
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 125}
        self.out = self.output_path("out")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import time

        os.environ["HF_HUB_CACHE"] = "/<on_purpose_invalid_hf_hub_cache_dir>"

        print("Import transformers / other libs...")
        start_time = time.time()

        import torch
        import returnn.util.basic as util
        from returnn.util import better_exchook

        os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
        better_exchook.install()

        try:
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            pass

        device_str = "cuda"

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

        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"({time.time() - start_time} secs)")
        print("Loading model...")
        start_time = time.time()
        model_dir = get_model_dir_from_hub_cache_dir(self.hub_cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, torch_dtype="auto", device_map=device_str
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype="auto", device_map=device_str
        )
        from transformers.models.cohere.modeling_cohere import CohereForCausalLM

        model: CohereForCausalLM  # just as an example...
        # CohereForCausalLM
        print(model)
        print("model.dtype:", model.dtype)
        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        print("Generate text...")
        start_time = time.time()
        # Format message with the chat template
        messages = [{"role": "user", "content": "Translate from English into German: This is a multilingual model."}]
        input_s = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("input:", input_s)  # for debugging
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>...<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
        print(input_ids)
        input_ids = input_ids.to(device_str)

        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.3,
        )

        gen_text = tokenizer.decode(gen_tokens[0])
        print(gen_text)
        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        def _gen_text(*, src_lang: str, dst_lang: str, src_text: str, dst_text: str):
            input_ids_parts = []
            src_text_mask_parts = []
            dst_text_mask_parts = []
            for input_s_, src_text_mask_, dst_text_mask_ in [
                (
                    f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate from {src_lang} into {dst_lang}:\n",
                    False,
                    False,
                ),
                (src_text, True, False),
                (f"<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", False, False),
                (dst_text, False, True),
                ("<|END_OF_TURN_TOKEN|>", False, False),
            ]:
                input_ids_ = tokenizer.encode(input_s_, add_special_tokens=False, return_tensors="pt").to(device_str)
                input_ids_parts.append(input_ids_)
                src_text_mask_parts.append(torch.full(input_ids_.shape, fill_value=src_text_mask_, device=device_str))
                dst_text_mask_parts.append(torch.full(input_ids_.shape, fill_value=dst_text_mask_, device=device_str))
            input_ids = torch.cat(input_ids_parts, dim=-1)
            src_text_mask = torch.cat(src_text_mask_parts, dim=-1)
            dst_text_mask = torch.cat(dst_text_mask_parts, dim=-1)
            return input_ids, src_text_mask, dst_text_mask

        for p in model.parameters():
            p.requires_grad = False

        input_embeddings = model.get_input_embeddings()

        input_ids, src_text_mask, dst_text_mask = _gen_text(
            src_lang="English",
            dst_lang="German",
            src_text="This is a multilingual model.",
            dst_text="Dies ist ein mehrsprachiges Modell.",
        )
        src_text_start = int(src_text_mask[0].nonzero().squeeze().min())
        src_text_end = int(src_text_mask[0].nonzero().squeeze().max()) + 1
        dst_text_start = int(dst_text_mask[0].nonzero().squeeze().min())
        dst_text_end = int(dst_text_mask[0].nonzero().squeeze().max()) + 1

        words = [[dst_text_start, dst_text_start + 1]]
        for t in range(dst_text_start + 1, dst_text_end):
            s = tokenizer.decode(input_ids[0, t : t + 1])
            if s.startswith(" ") or not s[:1].isalpha():  # new word
                words[-1][1] = t
                words.append([t, t + 1])
            else:
                words[-1][1] = t + 1
        print("words:", words)

        inputs_embeds = input_embeddings(input_ids[:, :-1])
        inputs_embeds = inputs_embeds.detach()
        inputs_embeds.requires_grad = True
        inputs_embeds.retain_grad()

        # TODO try some more...
        # SmoothGrad:
        # Add noise for better gradients.
        # inputs_embeds_ = inputs_embeds + torch.where(
        #     src_text_mask[:, :-1, None],
        #     torch.randn((5,) + inputs_embeds.shape[1:], device=inputs_embeds.device, dtype=inputs_embeds.dtype) * 1.0,
        #     0.0,
        # )
        # nf = torch.linspace(0.0, 0.5, steps=10, device=inputs_embeds.device, dtype=inputs_embeds.dtype)[:, None, None]
        # inputs_embeds_ = torch.where(
        #     src_text_mask[:, :-1, None],
        #     inputs_embeds * (1.0 - nf)
        #     + nf
        #     * torch.randn(
        #         nf.shape[:1] + inputs_embeds.shape[1:], device=inputs_embeds.device, dtype=inputs_embeds.dtype
        #     ),
        #     inputs_embeds,
        # )
        inputs_embeds_ = inputs_embeds

        res = model(inputs_embeds=inputs_embeds_)
        print(res)
        logits = res.logits.float()
        if logits.shape[0] > 1:
            logits = logits.mean(dim=0, keepdim=True)
        fake_logits = logits + (-logits).detach()  # zero, but grads will go to logits
        logits = logits + (logits * (0.5 + -1.0)).detach()  # smoothed, but grads will go to logits

        def _calc_input_grads(*, ref_norm: Optional[torch.Tensor] = None, i: Optional[int] = None):
            loss.backward(retain_graph=True)
            grad, inputs_embeds.grad = inputs_embeds.grad, None
            with torch.no_grad():
                e = inputs_embeds.float()
                grad = grad.float()
                ls = [
                    ("e*grad", (e * grad)[0, src_text_start:src_text_end].sum(dim=-1)),
                    ("L10(e*grad)", torch.norm((e * grad)[0, src_text_start:src_text_end], p=10, dim=-1)),
                    ("L1(e*grad)", torch.norm((e * grad)[0, src_text_start:src_text_end], p=1, dim=-1)),
                    ("L0.1(e*grad)", torch.norm((e * grad)[0, src_text_start:src_text_end], p=0.1, dim=-1)),
                    ("L1(grad)", torch.norm(grad[0, src_text_start:src_text_end], p=1, dim=-1)),
                    ("L0.1(grad)", torch.norm(grad[0, src_text_start:src_text_end], p=0.1, dim=-1)),
                ]
                if ref_norm is not None:
                    std, mean = torch.std_mean(ref_norm, dim=0)
                    std0 = torch.norm(ref_norm, p=2, dim=0)
                    ls.append(
                        (
                            "e*grad/absmean",
                            (e * grad)[0, src_text_start:src_text_end].sum(dim=-1) / ref_norm.abs().mean(dim=0),
                        )
                    )
                    ls.append(
                        ("e*grad-mean/std", ((e * grad)[0, src_text_start:src_text_end].sum(dim=-1) - mean) / std)
                    )
                    ls.append(("e*grad/std0", (e * grad)[0, src_text_start:src_text_end].sum(dim=-1) / std0))
                    ls.append(("log_sm", ref_norm.log_softmax(dim=0)[i]))
                for name, v in ls:
                    print(name, int(v.argmax()), v)
                return (e * grad)[0, src_text_start:src_text_end].sum(dim=-1)

        grad_mat = []
        grad_mat_fake = []
        for t0, t1 in words:
            loss = torch.nn.functional.cross_entropy(
                logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
            )
            grad_mat.append(_calc_input_grads())
            loss = torch.nn.functional.cross_entropy(
                fake_logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
            )
            grad_mat_fake.append(_calc_input_grads())
        grad_mat = torch.stack(grad_mat)
        grad_mat_fake = torch.stack(grad_mat_fake)

        for i, (t0, t1) in enumerate(words):
            print(f"*** {t0=} {t1=} {input_ids[0, t0:t1]=} {tokenizer.decode(input_ids[0, t0:t1])=}")
            loss = torch.nn.functional.cross_entropy(
                logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
            )
            _calc_input_grads(ref_norm=grad_mat, i=i)
            loss = torch.nn.functional.cross_entropy(
                fake_logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
            )
            _calc_input_grads(ref_norm=grad_mat_fake, i=i)

        better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())


class GenPhi4MultimodalInstruct(Job):
    def __init__(self, *, model_dir: tk.Path, datasets: Dict[str, tk.Path]):
        """
        :param model_dir: hub cache dir of model e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        :param datasets: hub cache dirs, e.g. via DownloadHuggingFaceRepoJobV2
        """
        super().__init__()
        self.model_dir = model_dir
        self.datasets = datasets
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 125}
        self.out = self.output_path("out")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import time

        os.environ["HF_HUB_CACHE"] = "/<on_purpose_invalid_hf_hub_cache_dir>"

        print("Import transformers / other libs...")
        start_time = time.time()

        import torch
        import returnn.util.basic as util
        from returnn.util import better_exchook

        os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
        better_exchook.install()

        try:
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            pass

        device_str = "cuda"
        dev = torch.device(device_str)

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

        from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

        print(f"({time.time() - start_time} secs)")
        print("Loading model...")
        start_time = time.time()
        model_dir = get_model_dir_from_hub_cache_dir(self.model_dir)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype="auto", trust_remote_code=True, device_map=device_str
        ).to(dev)
        generation_config = GenerationConfig.from_pretrained(model_dir)

        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalForCausalLM

        model: Phi4MultimodalForCausalLM  # just as an example...
        print(model)
        print("model.dtype:", model.dtype)
        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        # Part 2: Audio Processing
        print("\n--- AUDIO PROCESSING ---")

        from urllib.request import urlopen
        import io
        import soundfile as sf

        # Download and open audio file
        audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
        audio, samplerate = sf.read(io.BytesIO(urlopen(audio_url).read()))

        speech_prompt = "Transcribe the audio clip into text."
        prompt = f"<|user|><|audio_1|>{speech_prompt}<|end|><|assistant|>"
        print(f">>> Prompt\n{prompt}")

        # Process with the model
        inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors="pt").to(dev)

        generate_ids = model.generate(
            **inputs,
            num_logits_to_keep=0,  # bug to have this?
            max_new_tokens=1000,
            generation_config=generation_config,
        )
        # generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        print(f">>> Response\n{response}")

        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        for p in model.parameters():
            p.requires_grad = False

        tokenizer = processor.tokenizer
        (assistant_token_id,) = tokenizer.convert_tokens_to_ids(["<|assistant|>"])
        (end_token_id,) = tokenizer.convert_tokens_to_ids(["<|end|>"])

        transcription = "what we do as a society we have to think about where we're moving to i frequently talk to students about cognitive enhancing drugs and a lot of students take them for studying and exams but other students feel angry about this they feel those students are cheating and we have no long-term health and safety studies in healthy people and we really need those before people start taking them"
        prompt = f"<|user|><|audio_1|>{speech_prompt}<|end|><|assistant|>{transcription}<|end|>"
        inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors="pt")
        input_ids = inputs["input_ids"]
        (dst_text_start,) = torch.nonzero(input_ids[0] == assistant_token_id).squeeze(dim=1)
        dst_text_start = int(dst_text_start) + 1  # one past the assistant token
        dst_text_end = input_ids.shape[-1] - 1  # right before the <end> token. excluding.
        inputs = inputs.to(dev)
        input_ids = inputs["input_ids"]
        inputs_embeds = inputs["input_audio_embeds"]
        inputs_embeds.requires_grad = True
        inputs_embeds.retain_grad()
        res = model(**inputs)
        logits = res.logits
        assert logits.shape[:2] == input_ids.shape

        words = [[dst_text_start, dst_text_start + 1]]
        for t in range(dst_text_start + 1, dst_text_end):
            s = tokenizer.decode(input_ids[0, t : t + 1])
            if s.startswith(" ") or not s[:1].isalpha():  # new word
                words[-1][1] = t
                words.append([t, t + 1])
            else:
                words[-1][1] = t + 1
        print("words:", words)

        # Naming wrong... it's no "text" but audio.
        # Also not needed here, as we already have only the selected audio embedding part.
        src_text_start, src_text_end = None, None

        # t0 = assistant_start_frame  # first token for test here
        # t1 = t0 + 1
        # loss = torch.nn.functional.cross_entropy(
        #     res.logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
        # )

        logits = logits.float()
        if logits.shape[0] > 1:
            logits = logits.mean(dim=0, keepdim=True)
        fake_logits = logits + (-logits).detach()  # zero, but grads will go to logits
        # logits = logits + (logits * (0.5 + -1.0)).detach()  # smoothed, but grads will go to logits

        def _calc_input_grads(t0, t1) -> torch.Tensor:
            loss = torch.nn.functional.cross_entropy(
                fake_logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
            )
            loss.backward(retain_graph=True)
            grad, inputs_embeds.grad = inputs_embeds.grad, None
            with torch.no_grad():
                e = inputs_embeds.float()
                grad = grad.float()
                return (e * grad)[0, src_text_start:src_text_end].sum(dim=-1)

        grad_mat = []
        for t0, t1 in words:
            grad_mat.append(_calc_input_grads(t0, t1))
        grad_mat = torch.stack(grad_mat)  # [num_words,num_input_frames]
        absmean = grad_mat.abs().mean(dim=0, keepdim=True)  # [1,num_input_frames]
        grad_mat /= absmean

        for w, (t0, t1) in enumerate(words):
            v = grad_mat[w]
            print(f"*** {w=} {t0=} {t1=} {tokenizer.decode(input_ids[0, t0:t1])!r} -> {int(v.argmax())} {v}")

        better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())


class Aligner:
    """
    Create forced alignment given some arbitrary score matrix of shape [S,T],
    where T are the time frames, and S are the labels (or words).
    The alignment is of shape [T] pointing into 2*S+1 states,
    where all the even states correspond to blank/silence states.

    Derived from :class:`i6_experiments.users.zeyer.experiments.exp2024_09_09_grad_align.ForcedAlignOnScoreMatrixJob`.
    """

    def __init__(
        self,
        *,
        cut_off_eos: bool = True,
        norm_scores: bool = False,
        apply_log: bool = True,
        substract: Optional[Union[str, float]] = "max_gt0",
        apply_softmax_over_time: bool = False,
        apply_softmax_over_time_est_blank: bool = True,
        apply_softmax_over_labels: bool = False,
        blank_score: Union[float, str] = 0.0,  # or "calc"
        blank_score_est: str = "neg_prob",
        non_blank_score_reduce: str = "mean",
        blank_score_flipped_percentile: int = 0,
        num_seqs: int = -1,
        num_labels: Optional[int] = None,
        blank_idx: int,
        out_align_hdf_filename: str,
    ):
        self.cut_off_eos = cut_off_eos
        self.norm_scores = norm_scores
        self.apply_log = apply_log
        self.substract = substract
        self.apply_softmax_over_time = apply_softmax_over_time
        self.apply_softmax_over_time_est_blank = apply_softmax_over_time_est_blank
        self.apply_softmax_over_labels = apply_softmax_over_labels
        self.blank_score = blank_score
        self.blank_score_est = blank_score_est
        self.non_blank_score_reduce = non_blank_score_reduce
        self.blank_score_flipped_percentile = blank_score_flipped_percentile
        self.num_seqs = num_seqs
        self.num_labels = num_labels
        self.blank_idx = blank_idx

        from returnn.datasets.hdf import SimpleHDFWriter

        self.hdf_writer = SimpleHDFWriter(
            out_align_hdf_filename, dim=self.num_labels, ndim=1, extra_type={"states": (1, 1, "int32")}
        )

    def align(self, *, seq_tag: str, labels: List[int], score_matrix: np.ndarray, plot_dir: Optional[str] = None):
        """
        :param score_matrix: [S,T]
        :param plot_dir: if given, plots the scores and alignment as PDF into this dir
        """
        import numpy as np
        import os

        print("seq tag:", seq_tag)
        print("labels:", labels, f"(len {len(labels)})")

        print("score matrix shape (S x T):", score_matrix.shape)
        if self.cut_off_eos:
            # Last row is EOS, remove it.
            score_matrix = score_matrix[:-1]
        assert len(score_matrix) == len(labels), f"score_matrix.shape {score_matrix.shape} vs len labels {len(labels)}"
        T = score_matrix.shape[1]  # noqa
        S = score_matrix.shape[0]  # noqa

        if self.norm_scores:  # norm such that sum over whole matrix is 1
            score_matrix = score_matrix / np.sum(score_matrix)

        non_blank_score = np.max(score_matrix, axis=0)  # [T]
        blank_score = np.max(score_matrix) - non_blank_score

        # Note: We are going to search the alignment path with the highest score.
        if self.apply_log:
            # Assuming L2 norm scores (i.e. >0).
            score_matrix = np.log(score_matrix)
            blank_score = np.log(blank_score)
        # Otherwise assume already in log space.
        # Make sure they are all negative or zero max.
        m = np.max(score_matrix)
        print("score matrix max:", m)
        if self.substract == "max_gt0":
            score_matrix = score_matrix - max(m, 0.0)
            blank_score = blank_score - max(m, 0.0)
        elif isinstance(self.substract, float):
            score_matrix = score_matrix - self.substract
            blank_score = blank_score - self.substract
        elif not self.substract:
            pass
        else:
            raise ValueError(f"invalid substract {self.substract!r}")
        if self.apply_softmax_over_time:
            score_matrix = _log_softmax(score_matrix, axis=1)
            non_blank_score = np.max(np.exp(score_matrix), axis=0)  # [T]
            if self.apply_softmax_over_time_est_blank:
                blank_score = 1.0 - non_blank_score
                blank_score = np.log(blank_score)
        if self.blank_score_est == "flipped_after_softmax_over_time":
            # mean or max, both seem ok. optimal percentile changes.
            reduce_func = {
                "max": np.max,
                "mean": np.mean,
                "log_mean_exp": lambda x, axis: np.log(np.mean(np.exp(x), axis=axis)),
            }
            log_non_blank_score = reduce_func[self.non_blank_score_reduce](score_matrix, axis=0)  # [T]
            # for max, 10 enough. for mean: 30 or so.
            flip_point = np.percentile(log_non_blank_score, self.blank_score_flipped_percentile)
            blank_score = 2 * flip_point - log_non_blank_score  # [T]
        elif self.blank_score_est == "neg_prob":
            pass  # that's what we did above
        else:
            raise ValueError(f"invalid blank_score_est {self.blank_score_est!r}")
        if self.apply_softmax_over_labels:
            # Concat blank score to the end, to include it in the softmax.
            score_matrix = np.concatenate([score_matrix, blank_score[None, :]], axis=0)  # [S+1, T]
            score_matrix = _log_softmax(score_matrix, axis=0)
            score_matrix, blank_score = score_matrix[:-1], score_matrix[-1]

        # scores/backpointers over the states and time steps.
        # states = blank/sil + labels. whether we give scores to blank (and what score) or not is to be configured.
        # [T, S*2+1]
        backpointers = np.full(
            (T, S * 2 + 1), 3, dtype=np.int32
        )  # 0: diagonal-skip, 1: diagonal, 2: left, 3: undefined
        align_scores = np.full((T, S * 2 + 1), -np.infty, dtype=np.float32)

        score_matrix_ = np.zeros((T, S * 2 + 1), dtype=np.float32)  # [T, S*2+1]
        score_matrix_[:, 1::2] = score_matrix.T
        if isinstance(self.blank_score, (int, float)):
            score_matrix_[:, 0::2] = self.blank_score  # blank score
        elif self.blank_score == "calc":
            score_matrix_[:, 0::2] = blank_score[:, None]
        else:
            raise ValueError(f"invalid blank_score {self.blank_score!r} setting")

        # The first two states are valid start states.
        align_scores[0, :2] = score_matrix_[0, :2]
        backpointers[0, :] = 0  # doesn't really matter

        # calculate align_scores and backpointers
        for t in range(1, T):
            scores_diagonal_skip = np.full([2 * S + 1], -np.infty)
            scores_diagonal_skip[2:] = align_scores[t - 1, :-2] + score_matrix_[t, 2:]  # [2*S-1]
            scores_diagonal_skip[::2] = -np.infty  # diagonal skip is not allowed in blank
            scores_diagonal = np.full([2 * S + 1], -np.infty)
            scores_diagonal[1:] = align_scores[t - 1, :-1] + score_matrix_[t, 1:]  # [2*S]
            scores_horizontal = align_scores[t - 1, :] + score_matrix_[t, :]  # [2*S+1]

            score_cases = np.stack([scores_diagonal_skip, scores_diagonal, scores_horizontal], axis=0)  # [3, 2*S+1]
            backpointers[t] = np.argmax(score_cases, axis=0)  # [2*S+1]->[0,1,2]
            align_scores[t : t + 1] = np.take_along_axis(score_cases, backpointers[t : t + 1], axis=0)  # [1,2*S+1]

        # All but the last two states are not valid final states.
        align_scores[-1, :-2] = -np.infty

        # backtrace
        best_final = np.argmax(align_scores[-1])  # scalar, S*2 or S*2-1
        s = best_final
        t = T - 1
        alignment: List[Tuple[int, int]] = []
        while True:
            assert 0 <= s < S * 2 + 1 and 0 <= t < T
            alignment.append((t, s))
            if t == 0 and s <= 1:  # we reached some start state
                break

            b = backpointers[t, s]
            if b == 0:
                s -= 2
                t -= 1
            elif b == 1:
                s -= 1
                t -= 1
            elif b == 2:
                t -= 1
            else:
                raise ValueError(f"invalid backpointer {b} at s={s}, t={t}")

        assert len(alignment) == T
        alignment.reverse()
        alignment_ = []
        for t, s in alignment:
            if s % 2 == 0:
                alignment_.append(self.blank_idx)
            else:
                alignment_.append(labels[s // 2])
        alignment_ = np.array(alignment_, dtype=np.int32)  # [T]
        assert len(alignment_) == T

        self.hdf_writer.insert_batch(
            alignment_[None, :], seq_len=[T], seq_tag=[seq_tag], extra={"states": np.array(alignment)[None, :, 1]}
        )

        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)

            from matplotlib import pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            alignment_map = np.zeros([T, S], dtype=np.int32)  # [T, S]
            for t, s in alignment:
                if s % 2 == 1:
                    alignment_map[t, s // 2] = 1

            rows = [
                ("log(gradients) (local scores d)", score_matrix.T),
                ("Partial scores D", -1 * align_scores),
                ("backpointers", -1 * backpointers),
                ("alignment", alignment_map),
                ("blank scores", blank_score),
            ]
            fig, ax = plt.subplots(nrows=len(rows), ncols=1, figsize=(20, 10 * len(rows)))
            for i, (alias, mat) in enumerate(rows):
                if mat.ndim == 1:
                    mat = _y_to_mat(mat)  # [T,Y]
                # mat is [T,S*2+1] or [T,S]
                mat_ = ax[i].matshow(mat.T, cmap="Blues", aspect="auto")
                ax[i].set_title(f"{alias} for seq {seq_tag}")
                ax[i].set_xlabel("time")
                ax[i].set_ylabel("labels")
                ax[i].set_ylim(ax[i].get_ylim()[::-1])

                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                if alias == "backpointers":
                    cbar = fig.colorbar(mat_, cax=cax, orientation="vertical", ticks=[0, -1, -2, -3])
                    cbar.ax.set_yticklabels(["diagonal-skip", "diagonal", "left", "unreachable"])
                elif alias == "alignment":
                    cbar = fig.colorbar(mat_, cax=cax, orientation="vertical", ticks=[0, 1])
                    cbar.ax.set_yticklabels(["", "label"])
                else:
                    fig.colorbar(mat_, cax=cax, orientation="vertical")

            plt.tight_layout()
            plt.savefig(f"{plot_dir}/alignment_{seq_tag.replace('/', '_')}.pdf")

    def close(self):
        self.hdf_writer.close()


def _log_softmax(x: np.ndarray, *, axis: Optional[int]) -> np.ndarray:
    max_score = np.max(x, axis=axis, keepdims=True)
    x = x - max_score
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def _y_to_mat(y, y_num_pixels=100):  # only for visualization
    x_num_pixels = len(y)
    y_min, y_max = np.min(y), np.max(y)
    mat = np.full((x_num_pixels, y_num_pixels), y_min)
    for x_, y_ in enumerate(y):
        y__ = int((y_ - y_min) / max(y_max - y_min, 1) * (y_num_pixels - 1))
        mat[x_, y__] = y_
    return mat  # [T,Y]


def _debug_grad_score_types(
    *,
    loss: torch.Tensor,
    inputs_embeds: torch.Tensor,
    words: List[Tuple[int, int]],
    tokenizer,
    src_text_start: Optional[int] = None,
    src_text_end: Optional[int] = None,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    fake_logits: torch.Tensor,
):
    import torch

    def _calc_input_grads(*, ref_norm: Optional[torch.Tensor] = None, i: Optional[int] = None):
        loss.backward(retain_graph=True)
        grad, inputs_embeds.grad = inputs_embeds.grad, None
        with torch.no_grad():
            e = inputs_embeds.float()
            grad = grad.float()
            ls = [
                ("e*grad", (e * grad)[0, src_text_start:src_text_end].sum(dim=-1)),
                ("L10(e*grad)", torch.norm((e * grad)[0, src_text_start:src_text_end], p=10, dim=-1)),
                ("L1(e*grad)", torch.norm((e * grad)[0, src_text_start:src_text_end], p=1, dim=-1)),
                ("L0.1(e*grad)", torch.norm((e * grad)[0, src_text_start:src_text_end], p=0.1, dim=-1)),
                ("L1(grad)", torch.norm(grad[0, src_text_start:src_text_end], p=1, dim=-1)),
                ("L0.1(grad)", torch.norm(grad[0, src_text_start:src_text_end], p=0.1, dim=-1)),
            ]
            if ref_norm is not None:
                std, mean = torch.std_mean(ref_norm, dim=0)
                std0 = torch.norm(ref_norm, p=2, dim=0)
                ls.append(
                    (
                        "e*grad/absmean",
                        (e * grad)[0, src_text_start:src_text_end].sum(dim=-1) / ref_norm.abs().mean(dim=0),
                    )
                )
                ls.append(("e*grad-mean/std", ((e * grad)[0, src_text_start:src_text_end].sum(dim=-1) - mean) / std))
                ls.append(("e*grad/std0", (e * grad)[0, src_text_start:src_text_end].sum(dim=-1) / std0))
                ls.append(("log_sm", ref_norm.log_softmax(dim=0)[i]))
            for name, v in ls:
                print(name, int(v.argmax()), v)
            return (e * grad)[0, src_text_start:src_text_end].sum(dim=-1)

    grad_mat = []
    grad_mat_fake = []
    for t0, t1 in words:
        loss = torch.nn.functional.cross_entropy(
            logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
        )
        grad_mat.append(_calc_input_grads())
        loss = torch.nn.functional.cross_entropy(
            fake_logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
        )
        grad_mat_fake.append(_calc_input_grads())
    grad_mat = torch.stack(grad_mat)
    grad_mat_fake = torch.stack(grad_mat_fake)

    for i, (t0, t1) in enumerate(words):
        print(f"*** {t0=} {t1=} {input_ids[0, t0:t1]=} {tokenizer.decode(input_ids[0, t0:t1])=}")
        loss = torch.nn.functional.cross_entropy(
            logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
        )
        _calc_input_grads(ref_norm=grad_mat, i=i)
        loss = torch.nn.functional.cross_entropy(
            fake_logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
        )
        _calc_input_grads(ref_norm=grad_mat_fake, i=i)
