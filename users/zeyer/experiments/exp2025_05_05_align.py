"""
Continuation of :mod:`exp24_09_16_grad_align`.
"""

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
        messages = [{"role": "user", "content": "Translate from English into German:\nThis is a multilingual model"}]
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

        input_s = (
            "<BOS_TOKEN>"
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate from English into German:\n"
            "This is a multilingual model<|END_OF_TURN_TOKEN|>"
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            "Dies ist ein mehrsprachiges Modell.<|END_OF_TURN_TOKEN|>"
        )
        input_ids = tokenizer.encode(input_s, return_tensors="pt")
        input_ids = input_ids.to(device_str)

        target_ids = tokenizer.encode(["<EOS_TOKEN>"], return_tensors="pt").to(device_str)
        target_ids = torch.cat([input_ids[:, 1:], target_ids], dim=1)

        res = model(input_ids, labels=target_ids)
        print(res)

        better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())
