"""
Continuation of :mod:`exp24_09_16_grad_align`.
"""

from typing import TYPE_CHECKING, Optional, Dict, List, Tuple
from sisyphus import tk, Job, Task
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    DownloadHuggingFaceRepoJobV2,
    get_model_dir_from_hub_cache_dir,
)

if TYPE_CHECKING:
    import torch


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
