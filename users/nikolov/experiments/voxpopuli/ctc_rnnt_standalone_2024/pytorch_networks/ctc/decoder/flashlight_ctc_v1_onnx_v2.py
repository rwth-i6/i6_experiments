"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
"""

from dataclasses import dataclass
import time
import numpy as np
from typing import Any, Dict, Optional
from collections import Counter

@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    beam_size_token: int
    beam_threshold: float

    # needed files
    lexicon: str
    returnn_vocab: str

    # additional search options
    lm_weight: float = 0.0
    sil_score: float = 0.0
    word_score: float = 0.0

    # prior correction
    blank_log_penalty: Optional[float] = None
    prior_scale: float = 0.0
    prior_file: Optional[str] = None

    arpa_lm: Optional[str] = None

    use_torch_compile: bool = False
    torch_compile_options: Optional[Dict[str, Any]] = None


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


def forward_init_hook(run_ctx, **kwargs):
    """

    :param run_ctx:
    :param kwargs:
    :return:
    """

    import os
    import torch
    from torchaudio.models.decoder import ctc_decoder

    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    config.prior_scale = 0.2
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    if config.arpa_lm is not None:
        lm = cf(config.arpa_lm)
    else:
        lm = None

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, vocab_as_list=False, unknown_label="UNK")
    #labels = [item[1] for item in vocab.labels]
    labels = vocab.labels

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=config.lexicon,
        tokens=labels + ["[blank]"],
        blank_token="[blank]",
        sil_token="[blank]",
        unk_word="UNK",
        nbest=1,
        beam_size=config.beam_size,
        beam_size_token=config.beam_size_token,
        beam_threshold=config.beam_threshold,
    )

    run_ctx.labels = labels
    run_ctx.blank_log_penalty = config.blank_log_penalty

    if config.prior_file:
        run_ctx.prior = np.loadtxt(config.prior_file, dtype="float32")
        run_ctx.prior_scale = config.prior_scale
    else:
        run_ctx.prior = None

    import onnxruntime as ort
    from torch.onnx import export as onnx_export
    from torch import nn
    model = run_ctx.engine._model

    dummy_data = torch.rand(3,16000, device='cpu')
    dummy_data_len = torch.ones((3,), dtype=torch.int32)*16000
    
    print(f'Torch version: {torch.__version__}')
    print(config.lexicon)
    print(labels)
    onnx_export(
            model.eval(),
            (dummy_data, dummy_data_len),
            f='model.onnx',
            input_names=['data', 'data_len'],
            output_names=['classes', 'classes_len'],
            dynamic_axes={
                'data': {0: 'batch', 1: 'time'},
                'data_len': {0: 'batch'},
                'classes': {0: 'batch', 1: 'time'}, 
                'classes_len': {0: 'batch'},
                },
            )
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    sess_options.inter_op_num_threads = int(os.getenv('SLURM_CPUS_PER_TASK', 1))

    run_ctx.onnx_sess = ort.InferenceSession(
            'model.onnx', providers=['CPUExecutionProvider'], sess_options=sess_options)

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_am_time = 0
        run_ctx.total_search_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.print_rtf:
        print(
            "Total-AM-Time: %.2fs, AM-RTF: %.3f"
            % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s)
        )
        print(
            "Total-Search-Time: %.2fs, Search-RTF: %.3f"
            % (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
        )
        total_proc_time = run_ctx.total_am_time + run_ctx.total_search_time
        print("Total-time: %.2f, Batch-RTF: %.3f" % (total_proc_time, total_proc_time / run_ctx.running_audio_len_s))


def forward_step(*, model, data, run_ctx, **kwargs):
    import torch

    raw_audio = data["data"]  # [B, T', F]
    raw_audio_len = data["data:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

    am_start = time.time()
    raw_audio=np.squeeze(raw_audio.numpy())
    raw_audio_float = raw_audio.astype(np.float32)
    #print(raw_audio.astype(np.float32).shape)
    if raw_audio_float.ndim == 1:
        raw_audio_float = raw_audio_float.reshape(1, -1)
    logprobs, audio_features_len = run_ctx.onnx_sess.run(
            None,
            {
                'data': raw_audio_float,
                'data_len': raw_audio_len.numpy().astype(np.int32),
            } )

    tags = data["seq_tag"]

    logprobs_cpu = torch.from_numpy(logprobs).float().cpu()
    audio_features_len = torch.from_numpy(audio_features_len).float().cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior

    #np.set_printoptions(threshold=np.inf)
    #print(len(logprobs_cpu))
    #print(np.argmax(np.array(logprobs_cpu), axis=2))
    #sprint(audio_features_len)
    
    #hpy = ""
    #for char in np.argmax(logprobs_cpu, axis=1):
     #   if hpy[-1] != char:
     #       if char[-2:] == "@@":
     #          char = char[:-2]
    #        hyp = hyp + char
    #print(char)
    
    am_time = time.time() - am_start
    run_ctx.total_am_time += am_time

    search_start = time.time()
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len)
 
    #from ..decoder.returnn_ctc_multilang import ctc_decoder as ctc_decoder_greedy
    #print(audio_features_len)
    #hypothesis_greedy = ctc_decoder_greedy(logprobs_cpu, audio_features_len, run_ctx.labels)
    #print(hypothesis_greedy)

    search_time = time.time() - search_start
    run_ctx.total_search_time += search_time

    if run_ctx.print_rtf:
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))
 
    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])

        #prefixes = [run_ctx.labels[idx][:2] for idx in hyp[0].tokens]
        #recognition = [run_ctx.labels[idx][3:] for idx in hyp[0].tokens]
            
        #prefix_counts = Counter(prefixes)
        #correct_prefix, _ = prefix_counts.most_common(1)[0]

        #total_words = len(words)
        #correct_count = prefix_counts[correct_prefix]
        #percentage = (1 - ( correct_count / total_words)) * 100
        if run_ctx.print_hypothesis:
            print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
