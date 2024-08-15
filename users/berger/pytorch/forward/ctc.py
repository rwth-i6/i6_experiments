from typing import Optional
import torch
import numpy as np
from returnn.frontend import Tensor
from returnn.tensor.tensor_dict import TensorDict
from returnn.datasets.util.vocabulary import Vocabulary
from sisyphus import tk


def flashlight_ctc_decoder_forward_step(
    *,
    model: torch.nn.Module,
    extern_data: TensorDict,
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
    unk_word: str = "[UNKNOWN]",
    **_,
):
    from torchaudio.models.decoder import ctc_decoder

    audio_features = extern_data["data"].raw_tensor
    assert audio_features is not None

    assert extern_data["data"].dims[1].dyn_size_ext is not None
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    seq_tags = extern_data["seq_tag"].raw_tensor
    assert seq_tags is not None

    vocab = Vocabulary.create_vocab(vocab_file=vocab_file, unknown_label=None)
    labels = vocab.labels
    assert isinstance(labels, list)

    if blank_token not in labels:
        labels.append(blank_token)

    if silence_token not in labels:
        labels.append(silence_token)

    dec = ctc_decoder(
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_features = audio_features.to(device)
    audio_features_len = audio_features_len.to(device)
    model = model.to(device)

    log_probs, output_lengths = model(audio_features, audio_features_len)
    log_probs = log_probs.to("cpu")
    output_lengths = output_lengths.to("cpu")

    if prior_file is not None and prior_scale != 0:
        priors = np.loadtxt(prior_file, dtype=np.float32)
        log_probs = log_probs - prior_scale * priors

    hypotheses = dec(log_probs, output_lengths)
    hypotheses = [nbest[0] for nbest in hypotheses]

    word_arrays = []
    word_array_lengths = []

    for seq_tag, hyp in zip(seq_tags, hypotheses):
        print(f"Recognized sequence {repr(seq_tag)}:")
        print(f"   Words: {hyp.words}")
        print(f"   Tokens: {hyp.tokens}")
        print(f"   Score: {hyp.score}")
        print()
        word_arrays.append(np.array(hyp.words, dtype="U"))  # "U": unicode string
        word_array_lengths.append(len(hyp.words))

    # Need to pad word lists to the same length so that they are stackable
    max_len = np.max(word_array_lengths)
    word_arrays_padded = [
        np.pad(word_array, pad_width=(0, max_len - length))
        for word_array, length in zip(word_arrays, word_array_lengths)
    ]

    word_tensor = Tensor(
        name="tokens", dtype="string", raw_tensor=np.stack(word_arrays_padded, axis=0), feature_dim_axis=None
    )
    length_array = np.array(word_array_lengths, dtype=np.int32)

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        assert run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext is not None
        run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext.raw_tensor = length_array

    run_ctx.mark_as_output(word_tensor, name="tokens")
