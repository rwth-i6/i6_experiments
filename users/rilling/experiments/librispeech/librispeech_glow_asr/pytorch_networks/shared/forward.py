import numpy as np

def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    from torchaudio.models.decoder import ctc_decoder
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")
    import subprocess
    if kwargs["arpa_lm"] is not None:
        lm = subprocess.check_output(["cf", kwargs["arpa_lm"]]).decode().strip()
    else:
        lm = None
    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=kwargs["returnn_vocab"], unknown_label=None)
    labels = vocab.labels
    print(f"labels from vocab:{labels}")
    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=kwargs["lexicon"],
        lm=lm,
        lm_weight=kwargs["lm_weight"],
        # Add unnecessary silence because it is part of the lexicon
        # tokens=labels + ["[SILENCE]"],
        tokens=labels,
        blank_token="[blank]",
        sil_token="[space]",  # [space] is our actual silence
        unk_word="[UNKNOWN]",
        nbest=1,
        beam_size=kwargs["beam_size"],
        beam_threshold=kwargs["beam_threshold"],
        sil_score=kwargs.get("sil_score", 0.0),
        word_score=kwargs.get("word_score", 0.0),
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = kwargs.get("blank_log_penalty", None)

    if kwargs.get("prior_file", None):
        run_ctx.prior = np.loadtxt(kwargs["prior_file"], dtype="float32")
        run_ctx.prior_scale = kwargs["prior_scale"]
    else:
        run_ctx.prior = None


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    tags = data["seq_tag"]

    logprobs_cpu = logprobs.cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))