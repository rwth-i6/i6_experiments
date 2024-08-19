"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
"""

from dataclasses import dataclass
import time
import numpy as np
from typing import Any, Dict, Optional


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
    import sys
    sys.path.insert(0, "/u/kaloyan.nikolov/git/returnn")
    del sys.path[3]
    import torch
    from torchaudio.models.decoder import ctc_decoder

    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    if config.arpa_lm is not None:
        lm = cf(config.arpa_lm)
    else:
        lm = None

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    labels = vocab.labels

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=config.lexicon,
        lm=lm,
        lm_weight=config.lm_weight,
        tokens=labels + ["[blank]"],
        blank_token="[blank]",
        sil_token="[blank]",
        unk_word="[unknown]",
        nbest=1,
        beam_size=config.beam_size,
        beam_size_token=config.beam_size_token,
        beam_threshold=config.beam_threshold,
        sil_score=config.sil_score,
        word_score=config.word_score,
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
    sys.path.insert(0, '/u/kaloyan.nikolov/git/returnn')
    sys.path.insert(1, "/u/kaloyan.nikolov/experiments/on_device_asr_24/recipe/i6_experiments/users/nikolov/experiments/tedlium2/asr_2023/hybrid/distil_hubert/pytorch_networks/")

    from distill_hubert_v2 import Model

    dummy_data = torch.rand(3,16000, device='cpu')
    dummy_data_len = torch.ones((3,), dtype=torch.int32)*16000
    
    #assert False, (model)
    model_kwargs = {
    "hubert_dict": {"model_name": "base-ls960", "distill_scale": 0.0},
    "conformer_dict": {
        "hidden_d": 384,
        "conv_kernel_size": 7,
        "att_heads": 6,
        "ff_dim": 1536,
        "spec_num_time": 20,
        "spec_max_time": 20,
        "spec_num_feat": 5,
        "spec_max_feat": 16,
        "pool_1_stride": (3, 1),
        "pool_1_kernel_size": (1, 2),
        "pool_1_padding": None,
        "pool_2_stride": None,
        "pool_2_kernel_size": (1, 2),
        "pool_2_padding": None,
        "num_layers": 12,
        "upsample_kernel": 3,
        "upsample_stride": 3,
        "upsample_padding": 0,
        "upsample_out_padding": 0,
        "dropout": 0.2,
        "feat_extr": True,
    },
    }
    model = Model(**model_kwargs)

    checkpoint_state = torch.load('/work/asr4/hilmes/sis_work_folder/asr_2023/i6_core/returnn/training/GetBestPtCheckpointJob.6TVqYk7TaGje/output/checkpoint.pt', map_location=run_ctx.device)
   
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state["model"], strict=False)

    
    print(f'Torch version: {torch.__version__}')
    onnx_export(
            model.eval(),
            (dummy_data, dummy_data_len),
            f='model.onnx',
            verbose=True,
            input_names=['data', 'data_len'],
            output_names=['classes', 'classes_len'],
            opset_version=17,
            dynamic_axes={
                'data': {0: 'batch', 1: 'time'},
                'data_len': {0: 'batch'},
                'classes': {0: 'batch', 1: 'time'}, 
                'classes_len': {0: 'batch'},
                }           
            )
    sess_options = ort.SessionOptions()
    sess_option.intra_op_num_threads = int(os.getenv('SLURM_CPUS_PER_TASK', 4))


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

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

    am_start = time.time()
    logprobs, audio_features_len = run_ctx.onnx_sess.run(
            None,
            {
                'data': np.squeeze(raw_audio.numpy()),
                'data_len': raw_audio_len.numpy()
            } )

    tags = data["seq_tag"]

    logprobs_cpu = torch.from_numpy(logprobs).float().cpu()
    audio_features_len = torch.from_numpy(audio_features_len).float().cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior

    am_time = time.time() - am_start
    run_ctx.total_am_time += am_time

    search_start = time.time()
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len)
    search_time = time.time() - search_start
    run_ctx.total_search_time += search_time

    if run_ctx.print_rtf:
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        if run_ctx.print_hypothesis:
            print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
