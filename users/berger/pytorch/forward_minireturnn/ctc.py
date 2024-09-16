from typing import Optional
from returnn.torch.context import RunCtx
import torch
import numpy as np
from returnn.datasets.util.vocabulary import Vocabulary
from sisyphus import tk


def flashlight_ctc_decoder_init_hook(
    run_ctx: RunCtx,
    lexicon_file: tk.Path,
    vocab_file: tk.Path,
    lm_file: Optional[tk.Path] = None,
    prior_file: Optional[tk.Path] = None,
    beam_size: int = 50,
    beam_size_token: Optional[int] = None,
    beam_threshold: float = 50.0,
    lm_scale: float = 0.0,
    prior_scale: float = 0.0,
    word_score: float = 0.0,
    unk_score: float = float("-inf"),
    sil_score: float = 0.0,
    blank_token: str = "<blank>",
    silence_token: str = "[SILENCE]",
    unk_word: str = "<unk>",
    **kwargs,
):
    from torchaudio.models.decoder import ctc_decoder

    vocab = Vocabulary.create_vocab(vocab_file=vocab_file, unknown_label=None)
    labels = vocab.labels
    assert isinstance(labels, list)

    labels = list({value: key for key, value in vocab._vocab.items()}.values())

    if blank_token not in labels:
        labels.append(blank_token)

    if silence_token not in labels:
        labels.append(silence_token)
    print(f"labels: {labels}")

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=lexicon_file,
        tokens=labels,
        lm=lm_file,
        nbest=1,
        beam_size=beam_size,
        beam_threshold=beam_threshold,
        beam_size_token=beam_size_token,
        lm_weight=lm_scale,
        word_score=word_score,
        unk_score=unk_score,
        sil_score=sil_score,
        blank_token=blank_token,
        sil_token=silence_token,
        unk_word=unk_word,
    )

    run_ctx.running_audio_len_s = 0
    run_ctx.total_am_time = 0
    run_ctx.total_search_time = 0

    run_ctx.recognition_file = open("search_out.py", "w")
    run_ctx.recognition_file.write("{\n")

    run_ctx.prior_scale = prior_scale
    if prior_file is not None and prior_scale != 0:
        run_ctx.priors = np.loadtxt(prior_file, dtype=np.float32)
    else:
        run_ctx.priors = None


def flashlight_ctc_decoder_finish_hook(run_ctx: RunCtx, **kwargs):
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

    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()


def flashlight_ctc_decoder_forward_step(
    *,
    model: torch.nn.Module,
    data: dict,
    run_ctx: RunCtx,
    features_per_second: float = 16000,
    **_,
):
    import time

    audio_features = data["data"].float()
    audio_features_len = data["data:size1"]
    seq_tags = data["seq_tag"]

    audio_len_batch = torch.sum(audio_features_len).detach().cpu().numpy() / features_per_second
    run_ctx.running_audio_len_s += audio_len_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_features = audio_features.to(device)
    audio_features_len = audio_features_len.to(device)
    model = model.to(device)

    am_start = time.perf_counter()
    log_probs, output_lengths = model(audio_features, audio_features_len)
    am_time = time.perf_counter() - am_start
    run_ctx.total_am_time += am_time

    log_probs = log_probs.to("cpu")
    output_lengths = output_lengths.to("cpu")

    if run_ctx.priors is not None:
        log_probs = log_probs - run_ctx.prior_scale * run_ctx.priors

    search_start = time.perf_counter()
    hypotheses = run_ctx.ctc_decoder(log_probs, output_lengths)
    search_time = time.perf_counter() - search_start
    run_ctx.total_search_time += search_time
    hypotheses = [nbest[0] for nbest in hypotheses]

    print(f"Batch-AM-Time: {am_time:.3f} seconds, AM-RTF: {am_time / audio_len_batch:.3f}")
    print(f"Batch-Search-Time: {search_time:.3f} seconds, Search-RTF: {search_time / audio_len_batch:.3f}")
    print(
        f"Batch-Time: {am_time + search_time:.3f} seconds, Batch-RTF: {(am_time + search_time) / audio_len_batch:.3f}"
    )

    for seq_tag, hyp in zip(seq_tags, hypotheses):
        print(f"Recognized sequence {repr(seq_tag)}:")
        print(f"   Words: {hyp.words}")
        print(f"   Tokens: {hyp.tokens}")
        print(f"   Score: {hyp.score}")
        print()

        recog_result = " ".join([word for word in hyp.words if not word.startswith("[") or word.startswith("<")])

        run_ctx.recognition_file.write(f"{repr(seq_tag)}: {repr(recog_result)},\n")


def greedy_ctc_decoder_init_hook(
    run_ctx: RunCtx,
    vocab_file: tk.Path,
    prior_file: Optional[tk.Path] = None,
    prior_scale: float = 0.0,
    blank_token: str = "<blank>",
    **kwargs,
):
    vocab = Vocabulary.create_vocab(vocab_file=vocab_file, unknown_label=None)
    labels = list({value: key for key, value in vocab._vocab.items()}.values())

    if blank_token not in labels:
        run_ctx.blank_idx = len(labels)
        labels.append(blank_token)
    else:
        run_ctx.blank_idx = labels.index(blank_token)

    print(f"labels: {labels}")

    run_ctx.labels = labels

    run_ctx.running_audio_len_s = 0
    run_ctx.total_am_time = 0
    run_ctx.total_search_time = 0

    run_ctx.recognition_file = open("search_out.py", "w")
    run_ctx.recognition_file.write("{\n")

    run_ctx.prior_scale = prior_scale
    if prior_file is not None and prior_scale != 0:
        run_ctx.priors = np.loadtxt(prior_file, dtype=np.float32)
    else:
        run_ctx.priors = None


def greedy_ctc_decoder_finish_hook(run_ctx: RunCtx, **kwargs):
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

    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()


def greedy_ctc_decoder_forward_step(
    *,
    model: torch.nn.Module,
    data: dict,
    run_ctx: RunCtx,
    features_per_second: float = 16000,
    **_,
):
    import time

    audio_features = data["data"].float()
    audio_features_len = data["data:size1"]
    seq_tags = data["seq_tag"]

    audio_len_batch = torch.sum(audio_features_len).detach().cpu().numpy() / features_per_second
    run_ctx.running_audio_len_s += audio_len_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_features = audio_features.to(device)
    audio_features_len = audio_features_len.to(device)
    model = model.to(device)

    am_start = time.perf_counter()
    log_probs, output_lengths = model(audio_features, audio_features_len)
    am_time = time.perf_counter() - am_start
    run_ctx.total_am_time += am_time

    log_probs = log_probs.to("cpu")
    output_lengths = output_lengths.to("cpu")

    if run_ctx.priors is not None:
        log_probs = log_probs - run_ctx.prior_scale * run_ctx.priors

    search_start = time.perf_counter()
    argmax = torch.argmax(log_probs, dim=2)  # [B, T]
    hyps = []
    for b in range(len(argmax)):
        hyp_indices = torch.unique_consecutive(argmax[b])  # [T]
        hyp_indices = [idx for idx in hyp_indices if idx != run_ctx.blank_idx]
        hyps.append(hyp_indices)
    search_time = time.perf_counter() - search_start
    run_ctx.total_search_time += search_time

    print(f"Batch-AM-Time: {am_time:.3f} seconds, AM-RTF: {am_time / audio_len_batch:.3f}")
    print(f"Batch-Search-Time: {search_time:.3f} seconds, Search-RTF: {search_time / audio_len_batch:.3f}")
    print(
        f"Batch-Time: {am_time + search_time:.3f} seconds, Batch-RTF: {(am_time + search_time) / audio_len_batch:.3f}"
    )

    for seq_tag, hyp in zip(seq_tags, hyps):
        symbols = [run_ctx.labels[idx] for idx in hyp]
        print(f"Recognized sequence {repr(seq_tag)}:")
        print(f"   Tokens: {symbols}")
        recog_result = " ".join([symbol for symbol in symbols if not symbol.startswith("[") or symbol.startswith("<")])
        print()

        run_ctx.recognition_file.write(f"{repr(seq_tag)}: {repr(recog_result)},\n")
