from typing import Optional, List
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir


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
        grad_score_type: str,
    ):
        """
        :param model_dir: hub cache dir of model e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        :param dataset_dir: hub cache dir, e.g. via DownloadHuggingFaceRepoJobV2. for load_dataset
        :param dataset_key: e.g. "train", "test", whatever the dataset provides
        :param returnn_root: for some utils. version of RETURNN should not really matter
        :param speech_prompt: text-only part of the prompt
        :param grad_score_type: see :func:`get_grad_score_func`. e.g. "dot_e_grad" or "L1_e_grad"
        """
        super().__init__()
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.returnn_root = returnn_root
        self.speech_prompt = speech_prompt
        self.grad_score_type = grad_score_type

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

        from .grad_score_types import get_grad_score_func

        grad_score_func = get_grad_score_func(self.grad_score_type)

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
