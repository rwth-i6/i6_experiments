"""
Continuation of :mod:`exp24_09_16_grad_align`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Any, Dict, List, Tuple
from sisyphus import tk, Job, Task
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    DownloadHuggingFaceRepoJobV2,
    get_content_dir_from_hub_cache_dir,
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

    for ds_name, ds_dir in {"timit": dl_ds_timit, "buckeye": dl_ds_buckeye}.items():
        for key in ["val", "test"]:  # does not need train...
            gen_phi4mi = ExtractInGradsFromPhi4MultimodalInstructJob(
                model_dir=dl_phi4mi.out_hub_cache_dir,
                dataset_dir=ds_dir.out_hub_cache_dir,
                dataset_key=key,
            )
            gen_phi4mi.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            name = f"phi4mi-{ds_name}-{key}-grads"
            gen_phi4mi.add_alias(name)
            tk.register_output(f"{name}.hdf", gen_phi4mi.out_hdf)

            for grad_type in ["dot_e_grad", "L01_e_grad", "L1_e_grad", "L2_e_grad", "L01_grad", "L1_grad", "L2_grad"]:
                align_name = f"align/{name}-{grad_type}"
                align = CalcAlignmentMetricsJob(
                    grad_score_hdf=gen_phi4mi.out_hdf,
                    grad_score_key={"dot_e_grad": "data"}.get(grad_type, grad_type),
                    dataset_dir=ds_dir.out_hub_cache_dir,
                    dataset_key=key,
                    dataset_offset_factors={"timit": 1, "buckeye": 1000}[ds_name],
                    align_opts={"apply_softmax_over_time": True, "blank_score": -6},
                )
                align.add_alias(align_name)
                tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # Long-form experiment on Buckeye with Phi4.
    name_ = "phi4mi-buckeye-val-grads-longform"
    for chunk_opts in [
        {},
        {"chunk_overlap_secs": 1.0, "empty_exit_penalty": 0.0},
        {"chunk_size_secs": 20.0, "chunk_overlap_secs": 1.0, "empty_exit_penalty": 0.0},
        {"chunk_size_secs": 10.0, "chunk_overlap_secs": 1.0, "empty_exit_penalty": 0.0},
    ]:
        name = name_ + "-" + "-".join(f"{k}={v}" for k, v in chunk_opts.items())
        j = ChunkSegmentationFromPhi4MultimodalInstructLongFormJob(
            model_dir=dl_phi4mi.out_hub_cache_dir,
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            speech_prompt="Transcribe the audio clip into text.",
            **chunk_opts,
            dump_wav_first_n_seqs=5,  # debugging
        )
        j.add_alias(f"align/{name}-seg")
        tk.register_output("align/phi4mi-buckeye-val-grads-L2_e_grad-longform-seg.hdf", j.out_hdf)
        j = ExtractInGradsFromPhi4MultimodalInstructLongFormJob(
            model_dir=dl_phi4mi.out_hub_cache_dir,
            dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
            dataset_key="val",
            speech_prompt="Transcribe the audio clip into text.",
            chunk_segmentation_hdf=j.out_hdf,
        )
        j.add_alias(f"align/{name}")
        tk.register_output(f"align/{name}.hdf", j.out_hdf)
        for grad_type in ["dot_e_grad", "L01_e_grad", "L1_e_grad", "L2_e_grad", "L01_grad", "L1_grad", "L2_grad"]:
            align_name = f"align/{name}-{grad_type}"
            align = CalcChunkedAlignmentMetricsJob(
                grad_score_hdf=j.out_hdf,
                grad_score_key={"dot_e_grad": "data"}.get(grad_type, grad_type),
                dataset_dir=dl_ds_buckeye.out_hub_cache_dir,
                dataset_key="val",
                dataset_offset_factors={"timit": 1, "buckeye": 1000}["buckeye"],
                align_opts={"apply_softmax_over_time": True, "blank_score": -6},
            )
            align.add_alias(align_name)
            tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)

    # Test different prompts
    for i, prompt in enumerate(
        [
            "Transcribe the audio clip into text.",
            "Based on the attached audio, generate a comprehensive text transcription of the spoken content.",
            "Transcribe the audio clip into text, but please very accurately.",
        ]
    ):
        ds_name, ds_dir, key = "timit", dl_ds_timit, "val"
        gen_phi4mi = ExtractInGradsFromPhi4MultimodalInstructJob(
            model_dir=dl_phi4mi.out_hub_cache_dir,
            dataset_dir=ds_dir.out_hub_cache_dir,
            dataset_key=key,
            speech_prompt=prompt,
        )
        gen_phi4mi.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        name = f"phi4mi-{ds_name}-{key}-prompt{i}-grads"
        gen_phi4mi.add_alias(name)
        tk.register_output(f"{name}.hdf", gen_phi4mi.out_hdf)

        grad_type = "L2_e_grad"
        align_name = f"align/{name}-{grad_type}"
        align = CalcAlignmentMetricsJob(
            grad_score_hdf=gen_phi4mi.out_hdf,
            grad_score_key={"dot_e_grad": "data"}.get(grad_type, grad_type),
            dataset_dir=ds_dir.out_hub_cache_dir,
            dataset_key=key,
            dataset_offset_factors={"timit": 1, "buckeye": 1000}[ds_name],
            align_opts={"apply_softmax_over_time": True, "blank_score": -6},
        )
        align.add_alias(align_name)
        tk.register_output(f"{align_name}-wbe.txt", align.out_wbe)


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
        model_dir = get_content_dir_from_hub_cache_dir(self.hub_cache_dir)
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


class ExtractInGradsFromPhi4MultimodalInstructJob(Job):
    __sis_hash_exclude__ = {"speech_prompt": "Transcribe the audio clip into text."}

    def __init__(
        self,
        *,
        model_dir: tk.Path,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        speech_prompt: str = "Transcribe the audio clip into text.",
    ):
        """
        :param model_dir: hub cache dir of model e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        :param dataset_dir: hub cache dir, e.g. via DownloadHuggingFaceRepoJobV2. for load_dataset
        :param dataset_key: e.g. "train", "test", whatever the dataset provides
        """
        super().__init__()
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.speech_prompt = speech_prompt

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

        print("Import transformers / other libs...")
        start_time = time.time()

        import numpy as np
        import torch
        import returnn.util.basic as util
        from returnn.util import better_exchook
        from returnn.datasets.hdf import SimpleHDFWriter

        # os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
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

        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"({time.time() - start_time} secs)")
        print("Loading model...")
        start_time = time.time()
        model_dir = get_content_dir_from_hub_cache_dir(self.model_dir)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype="auto", trust_remote_code=True, device_map=device_str
        ).to(dev)

        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalForCausalLM

        model: Phi4MultimodalForCausalLM  # just as an example...
        print(model)
        print("model.dtype:", model.dtype)
        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        # print("\n--- AUDIO PROCESSING ---")

        # from urllib.request import urlopen
        # import io
        # import soundfile as sf

        # Download and open audio file
        # audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
        # audio, samplerate = sf.read(io.BytesIO(urlopen(audio_url).read()))

        speech_prompt = self.speech_prompt
        # prompt = f"<|user|><|audio_1|>{speech_prompt}<|end|><|assistant|>"
        # print(f">>> Prompt\n{prompt}")
        #
        # # Process with the model
        # inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors="pt").to(dev)
        #
        # generate_ids = model.generate(
        #     **inputs,
        #     num_logits_to_keep=0,  # bug to have this?
        #     max_new_tokens=1000,
        #     generation_config=generation_config,
        # )
        # # generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        # # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # response = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # print(f">>> Response\n{response}")
        #
        # _report_dev_memory_stats()
        # print(f"({time.time() - start_time} secs)")

        for p in model.parameters():
            p.requires_grad = False

        # TODO maybe monkey patch some modules, e.g. Phi4MMRMSNorm,
        #   via liger_kernel.transformers.monkey_patch._patch_rms_norm_module?

        tokenizer = processor.tokenizer
        (assistant_token_id,) = tokenizer.convert_tokens_to_ids(["<|assistant|>"])
        (end_token_id,) = tokenizer.convert_tokens_to_ids(["<|end|>"])

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
            # We don't need the logits here. There is currently no way to not compute them,
            # so num_logits_to_keep=1 is the best we can do.
            # We then will compute only the needed logits below,
            # and for that, we need the last layer output, thus output_hidden_states=True.
            res = model(**inputs, output_hidden_states=True, num_logits_to_keep=1)
            last_out = res.hidden_states[-1]  # [B,T,D]
            del res
            assert last_out.shape[:2] == input_ids.shape
            _report_dev_memory_stats()

            words_start_end = [[dst_text_start, dst_text_start + 1]]
            tokens = []
            for t in range(dst_text_start + 1, dst_text_end):
                s = tokenizer.decode(input_ids[0, t : t + 1])
                tokens.append(s)
                if s.startswith(" "):  # new word
                    words_start_end[-1][1] = t
                    words_start_end.append([t, t + 1])
                else:
                    words_start_end[-1][1] = t + 1
            assert len(words_start_end) == len(data["word_detail"]["utterance"]), f"got {tokens=}"

            # Not needed here, as we already have only the selected audio embedding part.
            src_start, src_end = None, None

            def _calc_input_grads(t0, t1, *, report_mem: bool = False) -> Dict[str, torch.Tensor]:
                logits = model.lm_head(last_out[:, t0 - 1 : t1 - 1])
                logits = logits.float()
                if logits.shape[0] > 1:
                    logits = logits.mean(dim=0, keepdim=True)
                fake_logits = logits + (-logits).detach()  # zero, but grads will go to logits

                loss = torch.nn.functional.cross_entropy(
                    fake_logits[0], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
                )
                loss.backward(retain_graph=True)
                if report_mem:
                    _report_dev_memory_stats()
                del fake_logits, logits
                grad, inputs_embeds.grad = inputs_embeds.grad, None
                with torch.no_grad():
                    e = inputs_embeds.float()
                    grad = grad.float()
                    return {
                        "dot_e_grad": (e * grad)[0, src_start:src_end].sum(dim=-1),
                        "L01_grad": torch.norm(grad[0, src_start:src_end], p=0.1, dim=-1),
                        "L1_grad": torch.norm(grad[0, src_start:src_end], p=1, dim=-1),
                        "L2_grad": torch.norm(grad[0, src_start:src_end], p=2, dim=-1),
                        "L01_e_grad": torch.norm((e * grad)[0, src_start:src_end], p=0.1, dim=-1),
                        "L1_e_grad": torch.norm((e * grad)[0, src_start:src_end], p=1, dim=-1),
                        "L2_e_grad": torch.norm((e * grad)[0, src_start:src_end], p=2, dim=-1),
                    }

            print("** Calculating grads")
            num_input_frames = inputs_embeds[0, src_start:src_end].shape[0]
            num_words = len(words_start_end)
            grad_mats: Dict[str, List[torch.Tensor]] = {}
            for w, (t0, t1) in enumerate(words_start_end):
                for name, grads in _calc_input_grads(t0, t1, report_mem=w in {0, num_words - 1}).items():
                    assert grads.shape == (num_input_frames,)
                    grad_mats.setdefault(name, []).append(grads)
            # each mat is [num_words,num_input_frames]
            grad_mats_: Dict[str, torch.Tensor] = {name: torch.stack(grad_mat) for name, grad_mat in grad_mats.items()}
            # Convert to Numpy and flatten and add dummy dim at the end to have it compatible for the HDF.
            # Also add dummy batch dim in the beginning (for insert_batch).
            grad_mats__ = {k: v.detach().cpu().numpy().flatten()[None, :, None] for k, v in grad_mats_.items()}

            print("** Freeing")
            del last_out, inputs_embeds, inputs  # not needed anymore now
            gc.collect()
            _report_dev_memory_stats()
            print(f"({time.time() - start_time} secs for the seq)")

            first_key = next(iter(grad_mats_.keys()))
            hdf_writer.insert_batch(
                grad_mats__[first_key],
                seq_len=[num_words * num_input_frames],
                seq_tag=[f"seq-{seq_idx}"],
                extra={
                    "sizes": np.array([num_words, num_input_frames])[None, None],
                    **{k: v for k, v in grad_mats__.items() if k != first_key},
                },
            )

        hdf_writer.close()

        # better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())


class ChunkSegmentationFromPhi4MultimodalInstructLongFormJob(Job):
    """
    Long-form variant
    """

    __sis_version__ = 3

    def __init__(
        self,
        *,
        model_dir: tk.Path,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        speech_prompt: str = "Transcribe the audio clip into text.",
        chunk_size_secs: float = 30.0,
        chunk_overlap_secs: float = 5.0,
        empty_exit_penalty: float = -5.0,
        dump_wav_first_n_seqs: int = 0,
    ):
        """
        :param model_dir: hub cache dir of model e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        :param dataset_dir: hub cache dir, e.g. via DownloadHuggingFaceRepoJobV2. for load_dataset
        :param dataset_key: e.g. "train", "test", whatever the dataset provides
        :param returnn_root:
        :param speech_prompt: prompt to use for the audio
        :param chunk_size_secs: chunk size in seconds
        :param chunk_overlap_secs:
        :param dump_wav_first_n_seqs: for debugging
        """
        super().__init__()
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.speech_prompt = speech_prompt
        self.chunk_size_secs = chunk_size_secs
        self.chunk_overlap_secs = chunk_overlap_secs
        self.empty_exit_penalty = empty_exit_penalty
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

        os.environ["HF_HUB_CACHE"] = "/<on_purpose_invalid_hf_hub_cache_dir>"

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        print("Import transformers / other libs...")
        start_time = time.time()

        import numpy as np
        import torch
        import returnn.util.basic as util
        from returnn.util import better_exchook
        from returnn.datasets.hdf import SimpleHDFWriter

        # os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
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

        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"({time.time() - start_time} secs)")
        print("Loading model...")
        start_time = time.time()
        model_dir = get_content_dir_from_hub_cache_dir(self.model_dir)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map=device_str,
        ).to(dev)

        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalForCausalLM
        from transformers.models.phi4_multimodal.processing_phi4_multimodal import Phi4MultimodalProcessor

        processor: Phi4MultimodalProcessor  # just as an example...
        model: Phi4MultimodalForCausalLM  # just as an example...
        print(model)
        print("model.dtype:", model.dtype)
        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        speech_prompt = self.speech_prompt

        for p in model.parameters():
            p.requires_grad = False

        tokenizer = processor.tokenizer
        (assistant_token_id,) = tokenizer.convert_tokens_to_ids(["<|assistant|>"])
        (end_token_id,) = tokenizer.convert_tokens_to_ids(["<|end|>"])

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
                assert cur_audio_end > cur_audio_start
                chunk_start_end.append((cur_audio_start, cur_audio_end))
                if cur_audio_end >= len(audio):
                    break  # only break point here
                cur_audio_start = cur_audio_end - math.ceil(self.chunk_overlap_secs * samplerate)
                assert cur_audio_start >= 0

            array: List[List[_Node]] = []  # [chunk_idx][rel word_idx]

            # In the (S+1)*C grid (RNN-T style), but we are not filling all S+1 entries per chunk.
            @dataclass
            class _Node:
                chunk_idx: int  # 0 <= c < C. the chunk we are in.
                word_idx: int  # 0 <= s <= S. we have seen this many words so far, words[:s]
                log_prob: torch.Tensor  # []. log prob of this node
                exit_log_prob: torch.Tensor  # []. log_prob+exit (end_token_id). horizontal transition to next chunk
                word_log_prob: Optional[
                    torch.Tensor
                ]  # []. log_prob+word (one or more labels). vertical transition to next word. (None if s==S)
                backpointer: Optional[_Node]  # prev chunk, or prev word

            for cur_chunk_idx, (cur_audio_start, cur_audio_end) in enumerate(chunk_start_end):
                if cur_chunk_idx == 0:
                    prev_array_word_idx = 0
                    cur_word_start = 0
                else:
                    # Heuristic. Look through last chunk, look out for best exit_log_prob
                    prev_array_word_idx = int(
                        torch.stack([node.exit_log_prob for node in array[cur_chunk_idx - 1]]).argmax().item()
                    )
                    cur_word_start = array[cur_chunk_idx - 1][prev_array_word_idx].word_idx
                # cur_word_end = cur_word_start + math.ceil(self.max_words_per_min * self.chunk_size_secs / 60.0)
                cur_word_end = len(words)  # Go to the end. Not so expensive...
                if cur_word_end > len(words):
                    cur_word_end = len(words)
                print(
                    f"** Forwarding chunk {cur_chunk_idx} (out of {len(chunk_start_end)}),"
                    f" {cur_audio_start / samplerate}:{cur_audio_end / samplerate} secs,"
                    f" words {cur_word_start}:{cur_word_end} (out of {len(words)})"
                )
                assert cur_word_end > cur_word_start  # need to fix heuristic if this fails...
                if cur_audio_end >= len(audio):
                    assert cur_word_end == len(words)  # need to overthink approx if this fails...
                transcription = " ".join(words[cur_word_start:cur_word_end])
                if cur_word_start > 0:
                    transcription = "... " + transcription
                prompt = f"<|user|><|audio_1|>{speech_prompt}<|end|><|assistant|>{transcription}<|end|>"
                with torch.no_grad():  # no grad here for segmentation, audio embeddings
                    inputs = processor(
                        text=prompt, audios=[(audio[cur_audio_start:cur_audio_end], samplerate)], return_tensors="pt"
                    )
                input_ids = inputs["input_ids"]
                (dst_text_start,) = torch.nonzero(input_ids[0] == assistant_token_id).squeeze(dim=1)
                dst_text_start = int(dst_text_start) + 1  # one past the assistant token
                dst_text_end = input_ids.shape[-1] - 1  # pos of the <end> token
                assert input_ids[0, dst_text_end] == end_token_id
                inputs = inputs.to(dev)
                input_ids = inputs["input_ids"]
                with torch.no_grad():  # no grad here for segmentation
                    # We don't need the logits here. There is currently no way to not compute them,
                    # so num_logits_to_keep=1 is the best we can do.
                    # We then will compute the needed logits below.
                    res = model(**inputs, output_hidden_states=True, num_logits_to_keep=1)
                last_out = res.hidden_states[-1]  # [B,T,D]
                del res
                assert last_out.shape[:2] == input_ids.shape

                words_start_end = [[dst_text_start, dst_text_start + 1]]
                tokens = [tokenizer.decode(input_ids[0, dst_text_start : dst_text_start + 1])]
                words_ = [tokens[-1]]
                for t in range(dst_text_start + 1, dst_text_end):
                    s = tokenizer.decode(input_ids[0, t : t + 1])
                    tokens.append(s)
                    if s.startswith(" "):  # new word
                        words_.append(s[1:])
                        words_start_end[-1][1] = t
                        words_start_end.append([t, t + 1])
                    else:
                        words_[-1] += s
                        words_start_end[-1][1] = t + 1
                if cur_word_start > 0:
                    assert words_[0] == "..."
                    words_start_end = words_start_end[1:]
                    words_ = words_[1:]
                assert len(words_start_end) == len(words_) == cur_word_end - cur_word_start, f"got {tokens=}"
                assert words_ == words[cur_word_start:cur_word_end], f"got {tokens=}"

                # Calculate log probs
                logits = model.lm_head(last_out[:, dst_text_start - 1 :])  # [B,T-dst_text_start+1,V]
                logits = logits.float()
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B,T-dst_text_start,V]
                array.append([])
                assert len(array) == cur_chunk_idx + 1
                for w, (t0, t1) in enumerate(words_start_end + [(dst_text_end, dst_text_end + 1)]):
                    word_idx = cur_word_start + w
                    if word_idx < cur_word_end:
                        word_log_prob = torch.sum(
                            torch.stack([log_probs[0, t - dst_text_start][input_ids[0, t]] for t in range(t0, t1)])
                        )  # []
                    else:
                        word_log_prob = None
                    exit_log_prob = log_probs[0, t0 - dst_text_start][end_token_id]  # []
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
                        log_prob = prev_node_below.word_log_prob
                    elif not prev_node_below and prev_node_left:
                        prev_node = prev_node_left
                        log_prob = prev_node_left.exit_log_prob
                    elif prev_node_below and prev_node_left:
                        if prev_node_below.word_log_prob >= prev_node_left.exit_log_prob:
                            prev_node = prev_node_below
                            log_prob = prev_node_below.word_log_prob
                        else:
                            prev_node = prev_node_left
                            log_prob = prev_node_left.exit_log_prob
                    else:
                        assert cur_chunk_idx == word_idx == 0
                        prev_node = None
                        log_prob = torch.zeros(())
                    array[cur_chunk_idx].append(
                        _Node(
                            chunk_idx=cur_chunk_idx,
                            word_idx=word_idx,
                            log_prob=log_prob,
                            backpointer=prev_node,
                            word_log_prob=(log_prob + word_log_prob) if word_idx < cur_word_end else None,
                            exit_log_prob=log_prob + exit_log_prob,
                        )
                    )
                assert (
                    len(array[cur_chunk_idx]) == len(words_start_end) + 1
                    and array[cur_chunk_idx][0].word_idx == cur_word_start
                    and array[cur_chunk_idx][-1].word_idx == cur_word_end
                )

                del last_out, inputs, logits, log_probs  # not needed anymore now

            # Backtrack
            nodes_alignment: List[_Node] = []
            node = array[-1][-1]
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
                    _write_wave_file(
                        f"seq{seq_idx}-chunk{cur_chunk_idx}.wav",
                        samples=audio[cur_audio_start:cur_audio_end],
                        sr=samplerate,
                    )
                    with open(f"seq{seq_idx}-chunk{cur_chunk_idx}.txt", "w") as f:
                        f.write(" ".join(words[w] for w in ws))

        hdf_writer.close()

        # better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())


class ExtractInGradsFromPhi4MultimodalInstructLongFormJob(Job):
    """
    Long-form variant
    """

    def __init__(
        self,
        *,
        model_dir: tk.Path,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        speech_prompt: str = "Transcribe the audio clip into text.",
        chunk_segmentation_hdf: tk.Path,
    ):
        """
        :param model_dir: hub cache dir of model e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        :param dataset_dir: hub cache dir, e.g. via DownloadHuggingFaceRepoJobV2. for load_dataset
        :param dataset_key: e.g. "train", "test", whatever the dataset provides
        :param returnn_root:
        :param speech_prompt: prompt to use for the audio
        :param chunk_segmentation_hdf: via ExtractInGradsFromPhi4MultimodalInstructLongFormDumpChunkSegmentationJob
        """
        super().__init__()
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.speech_prompt = speech_prompt
        self.chunk_segmentation_hdf = chunk_segmentation_hdf

        self.rqmt = {"time": 40, "cpu": 2, "gpu": 1, "mem": 125}

        self.out_hdf = self.output_path("out.hdf")

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

        import numpy as np
        import torch
        import returnn.util.basic as util
        from returnn.util import better_exchook
        from returnn.datasets.hdf import SimpleHDFWriter

        # os.environ["DEBUG"] = "1"  # for better_exchook to use debug shell on error
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

        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"({time.time() - start_time} secs)")
        print("Loading model...")
        start_time = time.time()
        model_dir = get_content_dir_from_hub_cache_dir(self.model_dir)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map=device_str,
        ).to(dev)

        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalForCausalLM
        from transformers.models.phi4_multimodal.processing_phi4_multimodal import Phi4MultimodalProcessor

        processor: Phi4MultimodalProcessor  # just as an example...
        model: Phi4MultimodalForCausalLM  # just as an example...
        print(model)
        print("model.dtype:", model.dtype)
        _report_dev_memory_stats()
        print(f"({time.time() - start_time} secs)")

        speech_prompt = self.speech_prompt

        for p in model.parameters():
            p.requires_grad = False

        tokenizer = processor.tokenizer
        (assistant_token_id,) = tokenizer.convert_tokens_to_ids(["<|assistant|>"])
        (end_token_id,) = tokenizer.convert_tokens_to_ids(["<|end|>"])

        from returnn.datasets.hdf import HDFDataset

        chunk_segmentation_hdf_ds = HDFDataset([self.chunk_segmentation_hdf.get_path()])
        chunk_segmentation_hdf_ds.initialize()
        chunk_segmentation_hdf_ds.init_seq_order(epoch=1)

        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(),
            dim=1,
            ndim=2,
            extra_type={
                "audio_chunk_start_end": (2, 2, "int32"),
                "words_indices_start_end": (2, 2, "int32"),
                "num_input_frames_per_chunk": (1, 2, "int32"),
            },
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
            words: List[str] = data["word_detail"]["utterance"]
            transcription = " ".join(words)
            print(f"* Seq {seq_idx}, {audio.shape=}, {len(audio) / samplerate} secs, {samplerate=}, {transcription!r}")
            assert len(transcription.split(" ")) == len(words)

            if seq_idx == 0:
                print("  data keys:", data.keys())

            chunk_segmentation_hdf_ds.load_seqs(seq_idx, seq_idx + 1)
            chunk_start_end = chunk_segmentation_hdf_ds.get_data(seq_idx, "audio_chunk_start_end")
            words_indices_start_end = chunk_segmentation_hdf_ds.get_data(seq_idx, "data")
            assert words_indices_start_end[:, 1].max() == len(words)

            num_input_frames_per_chunk: List[int] = []
            grad_mats: Dict[str, List[torch.Tensor]] = {}

            for chunk_idx, ((cur_audio_start, cur_audio_end), (cur_word_start, cur_word_end)) in enumerate(
                zip(chunk_start_end, words_indices_start_end)
            ):
                if cur_word_start == cur_word_end:
                    print(
                        f"** Skipping empty chunk {chunk_idx} (out of {len(chunk_start_end)}),"
                        f" {cur_audio_start / samplerate}:{cur_audio_end / samplerate} secs"
                    )
                    num_input_frames_per_chunk.append(0)
                    continue
                start_time = time.time()
                print(
                    f"** Forwarding chunk {chunk_idx} (out of {len(chunk_start_end)}),"
                    f" {cur_audio_start / samplerate}:{cur_audio_end / samplerate} secs,"
                    f" words {cur_word_start}:{cur_word_end}"
                )
                transcription = " ".join(words[cur_word_start:cur_word_end])
                if cur_word_start > 0:
                    transcription = "... " + transcription
                prompt = f"<|user|><|audio_1|>{speech_prompt}<|end|><|assistant|>{transcription}<|end|>"
                inputs = processor(
                    text=prompt, audios=[(audio[cur_audio_start:cur_audio_end], samplerate)], return_tensors="pt"
                )
                input_ids = inputs["input_ids"]
                (dst_text_start,) = torch.nonzero(input_ids[0] == assistant_token_id).squeeze(dim=1)
                dst_text_start = int(dst_text_start) + 1  # one past the assistant token
                dst_text_end = input_ids.shape[-1] - 1  # right before the <end> token. excluding.
                inputs = inputs.to(dev)
                input_ids = inputs["input_ids"]
                inputs_embeds = inputs["input_audio_embeds"]
                inputs_embeds.requires_grad = True
                inputs_embeds.retain_grad()
                # We don't need the logits here. There is currently no way to not compute them,
                # so num_logits_to_keep=1 is the best we can do.
                # We then will compute only the needed logits below,
                # and for that, we need the last layer output, thus output_hidden_states=True.
                res = model(**inputs, output_hidden_states=True, num_logits_to_keep=1)
                last_out = res.hidden_states[-1]  # [B,T,D]
                del res
                assert last_out.shape[:2] == input_ids.shape

                words_start_end = [[dst_text_start, dst_text_start + 1]]
                tokens = [tokenizer.decode(input_ids[0, dst_text_start : dst_text_start + 1])]
                words_ = [tokens[-1]]
                for t in range(dst_text_start + 1, dst_text_end):
                    s = tokenizer.decode(input_ids[0, t : t + 1])
                    tokens.append(s)
                    if s.startswith(" "):  # new word
                        words_.append(s[1:])
                        words_start_end[-1][1] = t
                        words_start_end.append([t, t + 1])
                    else:
                        words_[-1] += s
                        words_start_end[-1][1] = t + 1
                if cur_word_start > 0:
                    assert words_[0] == "..."
                    words_start_end = words_start_end[1:]
                    words_ = words_[1:]
                assert len(words_start_end) == len(words_) == cur_word_end - cur_word_start, f"got {tokens=}"
                assert words_ == words[cur_word_start:cur_word_end], f"got {tokens=}"

                # Not needed here, as we already have only the selected audio embedding part.
                src_start, src_end = None, None

                def _calc_input_grads(t0, t1, *, report_mem: bool = False) -> Dict[str, torch.Tensor]:
                    logits = model.lm_head(last_out[:, t0 - 1 : t1 - 1])
                    logits = logits.float()
                    if logits.shape[0] > 1:
                        logits = logits.mean(dim=0, keepdim=True)
                    fake_logits = logits + (-logits).detach()  # zero, but grads will go to logits

                    loss = torch.nn.functional.cross_entropy(
                        fake_logits[0], input_ids[0, t0:t1], ignore_index=-100, reduction="sum"
                    )
                    loss.backward(retain_graph=True)
                    if report_mem:
                        _report_dev_memory_stats()
                    del fake_logits, logits
                    grad, inputs_embeds.grad = inputs_embeds.grad, None
                    with torch.no_grad():
                        e = inputs_embeds.float()
                        grad = grad.float()
                        return {
                            "dot_e_grad": (e * grad)[0, src_start:src_end].sum(dim=-1),
                            "L01_grad": torch.norm(grad[0, src_start:src_end], p=0.1, dim=-1),
                            "L1_grad": torch.norm(grad[0, src_start:src_end], p=1, dim=-1),
                            "L2_grad": torch.norm(grad[0, src_start:src_end], p=2, dim=-1),
                            "L01_e_grad": torch.norm((e * grad)[0, src_start:src_end], p=0.1, dim=-1),
                            "L1_e_grad": torch.norm((e * grad)[0, src_start:src_end], p=1, dim=-1),
                            "L2_e_grad": torch.norm((e * grad)[0, src_start:src_end], p=2, dim=-1),
                        }

                print("** Calculating grads")
                num_input_frames = inputs_embeds[0, src_start:src_end].shape[0]
                num_input_frames_per_chunk.append(num_input_frames)
                for w, (t0, t1) in enumerate(words_start_end):
                    for name, grads in _calc_input_grads(t0, t1, report_mem=w in {0, len(words_start_end) - 1}).items():
                        assert grads.shape == (num_input_frames,)
                        grad_mats.setdefault(name, []).append(grads)

                print("** Freeing")
                del last_out, inputs_embeds, inputs  # not needed anymore now
                _report_dev_memory_stats()
                print(f"({time.time() - start_time} secs for the seq)")

            assert len(num_input_frames_per_chunk) == len(chunk_start_end)
            # each mat is [num_words,num_input_frames_per_chunk[...]].
            # All concatenated (flattened).
            grad_mats_: Dict[str, torch.Tensor] = {name: torch.concat(grad_mat) for name, grad_mat in grad_mats.items()}
            # Convert to Numpy and add dummy dim at the end to have it compatible for the HDF.
            # Also add dummy batch dim in the beginning (for insert_batch).
            grad_mats__ = {k: v.detach().cpu().numpy()[None, :, None] for k, v in grad_mats_.items()}

            first_key = next(iter(grad_mats_.keys()))
            hdf_writer.insert_batch(
                grad_mats__[first_key],
                seq_len=[len(grad_mats_[first_key])],
                seq_tag=[f"seq-{seq_idx}"],
                extra={
                    "audio_chunk_start_end": np.array(chunk_start_end)[None],
                    "words_indices_start_end": np.array(words_indices_start_end)[None],
                    "num_input_frames_per_chunk": np.array(num_input_frames_per_chunk)[None, :, None],
                    **{k: v for k, v in grad_mats__.items() if k != first_key},
                },
            )

        hdf_writer.close()

        # better_exchook.debug_shell(user_ns=locals(), user_global_ns=locals())


class CalcAlignmentMetricsJob(Job):
    def __init__(
        self,
        *,
        grad_score_hdf: tk.Path,
        grad_score_key: str,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        dataset_offset_factors: int,
        align_opts: Dict[str, Any],
    ):
        """
        :param grad_score_hdf:
        :param grad_score_key: in the grad_score_hdf, e.g. "data" (for dot_e_grad), "L2_grad" or so.
            Likely the HDF comes from :class:`ExtractInGradsFromPhi4MultimodalInstructJob`,
            so see that code for reference.
        :param dataset_dir:
        :param dataset_key:
        :param returnn_root:
        :param dataset_offset_factors:
            For TIMIT, the start/end offsets are directly on sample level.
            For Buckeye, there is factor 1000?
                len(ds_buckeye["val"][0]["audio"]["array"]) = 9969854,
                ds_buckeye["val"][0]["word_detail"]["stop"][-1] = 9969
        :param align_opts: see :class:`Aligner`
        """
        super().__init__()
        self.grad_score_hdf = grad_score_hdf
        self.grad_score_key = grad_score_key
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.dataset_offset_factors = dataset_offset_factors
        self.align_opts = align_opts

        self.out_wbe = self.output_var("wbe.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 10, "time": 5})

    def run(self):
        import os
        import sys
        import numpy as np

        os.environ["HF_HUB_CACHE"] = "/<on_purpose_invalid_hf_hub_cache_dir>"

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset

        grad_score_hdf_ds = HDFDataset([self.grad_score_hdf.get_path()])
        grad_score_hdf_ds.initialize()
        grad_score_hdf_ds.init_seq_order(epoch=1)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Dataset keys:", ds.keys())
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

        aligner = Aligner(**self.align_opts)

        # Follow https://arxiv.org/pdf/2406.02560, normalize first per utterance, then over the utterances.
        # Word boundary error (WBE) (we also called this time-stamp error (TSE))
        wbe_utts = []

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            grad_score_hdf_ds.load_seqs(seq_idx, seq_idx + 1)

            samplerate = data["audio"]["sampling_rate"]
            num_audio_samples = len(data["audio"]["array"])
            audio_len_secs = num_audio_samples / samplerate

            # We expect that the HDF comes from ExtractInGradsFromPhi4MultimodalInstructJob,
            # or is formatted exactly like ExtractInGradsFromPhi4MultimodalInstructJob would do.
            sizes = grad_score_hdf_ds.get_data(seq_idx, "sizes")
            assert sizes.shape == (1, 2)
            num_words, num_timeframes = [int(i) for i in sizes[0]]
            grad_mat = grad_score_hdf_ds.get_data(seq_idx, self.grad_score_key)
            assert grad_mat.shape == (num_words * num_timeframes, 1)
            grad_mat = grad_mat.reshape(num_words, num_timeframes)
            secs_per_timeframe = audio_len_secs / num_timeframes
            align_word_start_ends = aligner.align(grad_mat)
            align_word_start_ends = [tuple(t * secs_per_timeframe for t in ts) for ts in align_word_start_ends]
            assert num_words == len(align_word_start_ends)

            print(f"** seq {seq_idx}, {num_words=} {audio_len_secs=} {num_audio_samples=} {secs_per_timeframe=}")

            words: List[str] = data["word_detail"]["utterance"]
            ref_word_starts: List[float] = data["word_detail"]["start"]
            ref_word_ends: List[float] = data["word_detail"]["stop"]
            assert num_words == len(words) == len(ref_word_starts) == len(ref_word_ends)
            ref_word_start_ends = [
                tuple(t * self.dataset_offset_factors / samplerate for t in ts)
                for ts in zip(ref_word_starts, ref_word_ends)
            ]
            assert num_words == len(ref_word_start_ends) == len(align_word_start_ends)

            wbe_utt = np.mean(
                [
                    0.5
                    * (
                        abs(ref_word_start_ends[w][0] - align_word_start_ends[w][0])
                        + abs(ref_word_start_ends[w][1] - align_word_start_ends[w][1])
                    )
                    for w in range(num_words)
                ]
            )
            print("  WBE:", float(wbe_utt))
            wbe_utts.append(wbe_utt)

        wbe = float(np.mean(wbe_utts))
        self.out_wbe.set(wbe)


class CalcChunkedAlignmentMetricsJob(Job):
    def __init__(
        self,
        *,
        returnn_root: Optional[tk.Path] = None,
        grad_score_hdf: tk.Path,
        grad_score_key: str,
        dataset_dir: tk.Path,
        dataset_key: str,
        dataset_offset_factors: int,
        align_opts: Dict[str, Any],
    ):
        """
        :param returnn_root:
        :param grad_score_hdf:
        :param grad_score_key: in the grad_score_hdf, e.g. "data" (for dot_e_grad), "L2_grad" or so.
            Likely the HDF comes from :class:`ExtractInGradsFromPhi4MultimodalInstructJob`,
            so see that code for reference.
        :param dataset_dir:
        :param dataset_key:
        :param dataset_offset_factors:
            For TIMIT, the start/end offsets are directly on sample level.
            For Buckeye, there is factor 1000?
                len(ds_buckeye["val"][0]["audio"]["array"]) = 9969854,
                ds_buckeye["val"][0]["word_detail"]["stop"][-1] = 9969
        :param align_opts: see :class:`Aligner`
        """
        super().__init__()
        self.returnn_root = returnn_root
        self.grad_score_hdf = grad_score_hdf
        self.grad_score_key = grad_score_key
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.dataset_offset_factors = dataset_offset_factors
        self.align_opts = align_opts

        self.out_wbe = self.output_var("wbe.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 10, "time": 5})

    def run(self):
        import os
        import sys
        import numpy as np

        os.environ["HF_HUB_CACHE"] = "/<on_purpose_invalid_hf_hub_cache_dir>"

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset

        grad_score_hdf_ds = HDFDataset([self.grad_score_hdf.get_path()])
        grad_score_hdf_ds.initialize()
        grad_score_hdf_ds.init_seq_order(epoch=1)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Dataset keys:", ds.keys())
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

        aligner = Aligner(**self.align_opts)

        # Follow https://arxiv.org/pdf/2406.02560, normalize first per utterance, then over the utterances.
        # Word boundary error (WBE) (we also called this time-stamp error (TSE))
        wbe_utts = []

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            grad_score_hdf_ds.load_seqs(seq_idx, seq_idx + 1)

            samplerate = data["audio"]["sampling_rate"]
            num_audio_samples = len(data["audio"]["array"])
            audio_len_secs = num_audio_samples / samplerate
            words: List[str] = data["word_detail"]["utterance"]

            print(f"** seq {seq_idx}, {len(words)=} {audio_len_secs=} {num_audio_samples=}")

            ref_word_starts: List[float] = data["word_detail"]["start"]
            ref_word_ends: List[float] = data["word_detail"]["stop"]
            assert len(words) == len(ref_word_starts) == len(ref_word_ends)
            ref_word_start_ends = [
                tuple(t * self.dataset_offset_factors / samplerate for t in ts)
                for ts in zip(ref_word_starts, ref_word_ends)
            ]
            assert len(words) == len(ref_word_start_ends)

            chunk_start_end = grad_score_hdf_ds.get_data(seq_idx, "audio_chunk_start_end")
            words_indices_start_end = grad_score_hdf_ds.get_data(seq_idx, "words_indices_start_end")
            assert words_indices_start_end[:, 1].max() == len(words)
            num_input_frames_per_chunk = grad_score_hdf_ds.get_data(seq_idx, "num_input_frames_per_chunk")
            assert len(num_input_frames_per_chunk) == len(chunk_start_end) == len(words_indices_start_end)
            assert num_input_frames_per_chunk.shape == (len(chunk_start_end), 1)
            num_input_frames_per_chunk = num_input_frames_per_chunk.squeeze(1)

            grad_mat = grad_score_hdf_ds.get_data(seq_idx, self.grad_score_key)
            assert grad_mat.ndim == 2 and grad_mat.shape[1] == 1  # HDF restriction
            grad_mat = grad_mat.squeeze(1)

            align_word_start_ends = []
            grad_mat_offset = 0
            for chunk_idx, (
                (cur_audio_start, cur_audio_end),
                (cur_word_start, cur_word_end),
                num_input_frames,
            ) in enumerate(zip(chunk_start_end, words_indices_start_end, num_input_frames_per_chunk)):
                if cur_word_start == cur_word_end:  # empty chunk
                    continue

                num_words_ = cur_word_end - cur_word_start
                grad_mat_ = grad_mat[grad_mat_offset : grad_mat_offset + num_words_ * num_input_frames]
                grad_mat_offset += num_words_ * num_input_frames
                grad_mat_ = grad_mat_.reshape(num_words_, num_input_frames)

                cur_align_word_start_ends = aligner.align(grad_mat_)
                cur_align_word_start_ends = [
                    tuple(
                        (cur_audio_start + (t / num_input_frames) * (cur_audio_end - cur_audio_start)) / samplerate
                        for t in ts
                    )
                    for ts in cur_align_word_start_ends
                ]
                assert len(cur_align_word_start_ends) == num_words_
                align_word_start_ends += cur_align_word_start_ends

            assert grad_mat_offset == grad_mat.shape[0]
            assert len(words) == len(align_word_start_ends)

            wbe_utt = np.mean(
                [
                    0.5
                    * (
                        abs(ref_word_start_ends[w][0] - align_word_start_ends[w][0])
                        + abs(ref_word_start_ends[w][1] - align_word_start_ends[w][1])
                    )
                    for w in range(len(words))
                ]
            )
            print("  WBE:", float(wbe_utt))
            wbe_utts.append(wbe_utt)

        wbe = float(np.mean(wbe_utts))
        self.out_wbe.set(wbe)


class CalcAlignmentMetricsFromWordBoundariesJob(Job):
    """
    Calc metrics from given wound boundaries.
    (I.e. no alignment is happening here; that is already given via word boundaries.)
    """

    def __init__(
        self,
        *,
        word_boundaries_hdf: tk.Path,
        dataset_dir: tk.Path,
        dataset_key: str,
        returnn_root: Optional[tk.Path] = None,
        dataset_offset_factors: int,
    ):
        """
        :param word_boundaries_hdf: e.g. from ExtractInGradsFromPhi4MultimodalInstructLongFormJob.
            Assumes shape [num_words,2], where each frame represents [word_start,word_end] (in samples).
        :param dataset_dir:
        :param dataset_key:
        :param returnn_root:
        :param dataset_offset_factors:
            For TIMIT, the start/end offsets are directly on sample level.
            For Buckeye, there is factor 1000?
                len(ds_buckeye["val"][0]["audio"]["array"]) = 9969854,
                ds_buckeye["val"][0]["word_detail"]["stop"][-1] = 9969
        """
        super().__init__()
        self.word_boundaries_hdf = word_boundaries_hdf
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.dataset_offset_factors = dataset_offset_factors

        self.out_wbe = self.output_var("wbe.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 10, "time": 5})

    def run(self):
        import os
        import sys
        import numpy as np

        os.environ["HF_HUB_CACHE"] = "/<on_purpose_invalid_hf_hub_cache_dir>"

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset

        word_boundaries_hdf_ds = HDFDataset([self.word_boundaries_hdf.get_path()])
        word_boundaries_hdf_ds.initialize()
        word_boundaries_hdf_ds.init_seq_order(epoch=1)

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset: {ds}")
        print("Dataset keys:", ds.keys())
        print("Using key:", self.dataset_key)
        print("Num seqs:", len(ds[self.dataset_key]))

        # Follow https://arxiv.org/pdf/2406.02560, normalize first per utterance, then over the utterances.
        # Word boundary error (WBE) (we also called this time-stamp error (TSE))
        wbe_utts = []

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            word_boundaries_hdf_ds.load_seqs(seq_idx, seq_idx + 1)

            samplerate = data["audio"]["sampling_rate"]
            num_audio_samples = len(data["audio"]["array"])
            audio_len_secs = num_audio_samples / samplerate
            words: List[str] = data["word_detail"]["utterance"]

            align_word_start_ends = word_boundaries_hdf_ds.get_data(seq_idx, "data")
            assert align_word_start_ends.shape == (len(words), 2)

            print(f"** seq {seq_idx}, {len(words)=} {audio_len_secs=} {num_audio_samples=}")

            ref_word_starts: List[float] = data["word_detail"]["start"]
            ref_word_ends: List[float] = data["word_detail"]["stop"]
            assert len(words) == len(ref_word_starts) == len(ref_word_ends)
            ref_word_start_ends = [
                tuple(t * self.dataset_offset_factors / samplerate for t in ts)
                for ts in zip(ref_word_starts, ref_word_ends)
            ]
            assert len(words) == len(ref_word_start_ends) == len(align_word_start_ends)

            wbe_utt = np.mean(
                [
                    0.5
                    * (
                        abs(ref_word_start_ends[w][0] - align_word_start_ends[w][0])
                        + abs(ref_word_start_ends[w][1] - align_word_start_ends[w][1])
                    )
                    for w in range(len(words))
                ]
            )
            print("  WBE:", float(wbe_utt))
            wbe_utts.append(wbe_utt)

        wbe = float(np.mean(wbe_utts))
        self.out_wbe.set(wbe)


class Aligner:
    """
    Create forced alignment given some arbitrary score matrix of shape [S,T],
    where T are the time frames, and S are the labels (or words).
    The alignment is of shape [T] pointing into 2*S+1 states,
    where all the even states correspond to blank/silence states.

    Derived from :class:`i6_experiments.users.zeyer.experiments.exp2024_09_09_grad_align.ForcedAlignOnScoreMatrixJob`.

    Some good options before:

                {"apply_softmax_over_time": True, "blank_score": -6},
                {
                    "apply_softmax_over_time": True,
                    "blank_score": "calc",
                    "blank_score_est": "flipped_after_softmax_over_time",
                    "non_blank_score_reduce": "log_mean_exp",
                    "blank_score_flipped_percentile": 80,
                    "apply_softmax_over_labels": True,
                },
                {
                    "apply_softmax_over_time": True,
                    "blank_score": "calc",
                    "blank_score_est": "flipped_after_softmax_over_time",
                    "non_blank_score_reduce": "log_mean_exp",
                    "blank_score_flipped_percentile": 60,
                    "apply_softmax_over_labels": True,
                },
                {
                    "apply_softmax_over_time": True,
                    "blank_score": "calc",
                    "blank_score_est": "flipped_after_softmax_over_time",
                    "non_blank_score_reduce": "log_mean_exp",
                    "blank_score_flipped_percentile": 40,
                    "apply_softmax_over_labels": True,
                },
    """

    def __init__(
        self,
        *,
        cut_off_eos: bool = False,
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

    def align(
        self,
        score_matrix: np.ndarray,
        *,
        plot_filename: Optional[str] = None,
        num_final_words: int = 1,
    ) -> List[Tuple[int, int]]:
        """
        :param score_matrix: [S,T]
        :param plot_filename: if given, plots the scores and alignment as PDF into this file
        :param num_final_words: if 1, the last (S-1) is the only allowed last word.
        :return: list of start/end offsets, both are including. len is S
        """
        import numpy as np
        import os

        if self.cut_off_eos:
            # Last row is EOS, remove it.
            score_matrix = score_matrix[:-1]
        S, T = score_matrix.shape
        inf = np.inf

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
        align_scores = np.full((T, S * 2 + 1), -inf, dtype=np.float32)

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
            scores_diagonal_skip = np.full([2 * S + 1], -inf)
            scores_diagonal_skip[2:] = align_scores[t - 1, :-2] + score_matrix_[t, 2:]  # [2*S-1]
            scores_diagonal_skip[::2] = -inf  # diagonal skip is not allowed in blank
            scores_diagonal = np.full([2 * S + 1], -inf)
            scores_diagonal[1:] = align_scores[t - 1, :-1] + score_matrix_[t, 1:]  # [2*S]
            scores_horizontal = align_scores[t - 1, :] + score_matrix_[t, :]  # [2*S+1]

            score_cases = np.stack([scores_diagonal_skip, scores_diagonal, scores_horizontal], axis=0)  # [3, 2*S+1]
            backpointers[t] = np.argmax(score_cases, axis=0)  # [2*S+1]->[0,1,2]
            align_scores[t : t + 1] = np.take_along_axis(score_cases, backpointers[t : t + 1], axis=0)  # [1,2*S+1]

        # All but the last two (* num_final_words) states are not valid final states.
        align_scores[-1, : -2 * num_final_words] = -inf

        # backtrace
        best_final = np.argmax(align_scores[-1])  # scalar, S*2 or S*2-1
        s = best_final
        t = T - 1
        # Get (t,s) tuples (t not really needed for now; depends on topology).
        # s is the state with 0 <= s < S * 2 + 1, and 0 <= t < T.
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

        labels_start_end: List[Tuple[int, int]] = []  # start/end offsets, both are including
        prev_s = 0
        for t, s in alignment:
            if prev_s != s:  # new state
                if prev_s % 2 == 0:  # sil before
                    assert s % 2 == 1  # expect that we get a non-sil label now
                if s % 2 != 0:  # non-sil new label
                    labels_start_end.append((t, t))
            if s % 2 != 0:  # in non-sil label
                labels_start_end[-1] = (labels_start_end[-1][0], t - 1)  # update end
            prev_s = s
        assert S - (num_final_words - 1) <= len(labels_start_end) <= S

        if plot_filename is not None:
            assert plot_filename.endswith(".pdf")
            os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

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
                ax[i].set_title(f"{alias} for seq {os.path.splitext(os.path.basename(plot_filename))[0]}")
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
            plt.savefig(plot_filename)

        return labels_start_end


def _log_softmax(x: np.ndarray, *, axis: Optional[int]) -> np.ndarray:
    import numpy as np

    max_score = np.max(x, axis=axis, keepdims=True)
    x = x - max_score
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def _y_to_mat(y, y_num_pixels=100):  # only for visualization
    import numpy as np

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


def _write_wave_file(filename: str, samples: np.ndarray, *, sr: int = 16_000, w: int = 2):
    """
    Write a wave file to disk

    :param filename:
    :param samples: 1D, float, -1 to 1
    :param sr: sample rate
    :param w: sample width in bytes
    """
    import wave

    assert samples.ndim == 1
    samples = samples.clip(-1, 1)
    with wave.open(filename, "w") as f:
        f.setnchannels(1)
        f.setframerate(sr)
        f.setsampwidth(w)
        samples_int = (samples * (2 ** (8 * w - 1) - 1)).astype({1: "int8", 2: "int16", 4: "int32"}[w])
        f.writeframes(samples_int.tobytes())
        f.close()
