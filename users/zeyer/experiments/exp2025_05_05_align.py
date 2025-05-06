"""
Continuation of :mod:`exp24_09_16_grad_align`.
"""

from typing import Optional
from sisyphus import tk, Job, Task
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    get_model_dir_from_hub_cache_dir,
)


def py():
    dl = DownloadHuggingFaceRepoJob(model_id="CohereLabs/aya-expanse-32b")
    tk.register_output("aya", dl.out_hub_cache_dir)

    gen = Gen(hub_cache_dir=dl.out_hub_cache_dir)
    tk.register_output("aya-gen", gen.out)


class Gen(Job):
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
                    f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate from {src_lang} into {dst_lang}: ",
                    False,
                    False,
                ),
                (src_text, True, False),
                ("<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", False, False),
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
        inputs_embeds_ = inputs_embeds

        res = model(inputs_embeds=inputs_embeds_)
        print(res)
        logits = res.logits.float()
        if logits.shape[0] > 1:
            logits = logits.mean(dim=0, keepdim=True)
        fake_logits = logits + (-logits).detach()  # zero, but grads will go to logits
        # logits = logits + (logits * (0.9 + -1.0)).detach()  # smoothed, but grads will go to logits

        def _calc_input_grads(*, ref_norm: Optional[torch.Tensor] = None, i: Optional[int] = None):
            loss.backward(retain_graph=True)
            grad, inputs_embeds.grad = inputs_embeds.grad, None
            with torch.no_grad():
                e = inputs_embeds.float()
                grad = grad.float()
                ls = [
                    (e * grad)[0, src_text_start:src_text_end].sum(dim=-1),
                    torch.norm((e * grad)[0, src_text_start:src_text_end], p=10, dim=-1),
                    torch.norm((e * grad)[0, src_text_start:src_text_end], p=1, dim=-1),
                    torch.norm((e * grad)[0, src_text_start:src_text_end], p=0.1, dim=-1),
                    torch.norm(grad[0, src_text_start:src_text_end], p=1, dim=-1),
                    torch.norm(grad[0, src_text_start:src_text_end], p=0.1, dim=-1),
                ]
                if ref_norm is not None:
                    ls.append(torch.norm(grad[0, src_text_start:src_text_end], p=1, dim=-1) / ref_norm.sum(dim=0))
                    ls.append(ref_norm.log_softmax(dim=0)[i])
                for v in ls:
                    print(v, int(v.argmax()))
                return torch.norm(grad[0, src_text_start:src_text_end], p=1, dim=-1)

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
            print(f"{t0=} {t1=} {input_ids[0, t0:t1]=} {tokenizer.decode(input_ids[0, t0:t1])=}")
            loss = torch.nn.functional.cross_entropy(
                logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
            )
            _calc_input_grads(ref_norm=grad_mat, i=i)
            loss = torch.nn.functional.cross_entropy(
                fake_logits[0, t0 - 1 : t1 - 1], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
            )
            _calc_input_grads(ref_norm=grad_mat_fake, i=i)

        better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())
