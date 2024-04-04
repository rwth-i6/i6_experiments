import torch
import numpy as np
from torchaudio.models.rnnt import RNNT
from returnn.frontend import Tensor
from returnn.tensor.tensor_dict import TensorDict
from i6_experiments.users.berger.pytorch.forward.transducer_beam_search import beam_search
from sisyphus import tk
from i6_core.lib.lexicon import Lexicon


def flashlight_ctc_decoder_forward_step(
    *, model: torch.nn.Module, extern_data: TensorDict, lexicon_file: tk.Path, beam_size: int, **kwargs
):
    from torchaudio.models.decoder import ctc_decoder

    audio_features = extern_data["data"].raw_tensor
    assert extern_data["data"].dims[1].dyn_size_ext is not None
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    seq_tags = extern_data["seq_tag"].raw_tensor
    assert seq_tags is not None

    assert audio_features is not None
    assert audio_features_len is not None

    lexicon = Lexicon()
    lexicon.load(lexicon_file)
    label_list = list(lexicon.phonemes)

    dec = ctc_decoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens_arrays = []
    tokens_len = []

    for b in range(audio_features.size(0)):
        seq_feature_len = audio_features_len[b : b + 1]
        seq_features = audio_features[b : b + 1, : seq_feature_len[0]]
        token_indices = beam_search(
            model=model.to(device=device),
            features=seq_features.to(device=device),
            features_len=seq_feature_len.to(device=device),
            beam_size=beam_size,
        )
        tokens_array = np.array([label_list[token_idx] for token_idx in token_indices], dtype="<U3")
        print(f"Recognized sequence {repr(seq_tags[b])}: {tokens_array}")
        tokens_arrays.append(tokens_array)
        tokens_len.append(len(tokens_array))

    max_len = np.max(tokens_len)
    tokens_arrays_padded = [
        np.pad(tokens_array, pad_width=(0, max_len - len(tokens_array))) for tokens_array in tokens_arrays
    ]

    tokens_tensor = Tensor(
        name="tokens", dtype="string", raw_tensor=np.stack(tokens_arrays_padded, axis=0), feature_dim_axis=None
    )
    tokens_len_array = np.array(tokens_len, dtype=np.int32)

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        assert run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext is not None
        run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext.raw_tensor = tokens_len_array

    run_ctx.mark_as_output(tokens_tensor, name="tokens")

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")
    import subprocess

    if kwargs["arpa_lm"] is not None:
        lm = subprocess.check_output(["cf", kwargs["arpa_lm"]]).decode().strip()
    else:
        lm = None
    from returnn.datasets.util.vocabulary import Vocabulary

    vocab = Vocabulary.create_vocab(vocab_file=kwargs["returnn_vocab"], unknown_label=None)
    labels = vocab.labels
    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=kwargs["lexicon"],
        lm=lm,
        lm_weight=kwargs["lm_weight"],
        tokens=labels + ["[blank]", "[SILENCE]", "[UNK]"],
        # "[SILENCE]" and "[UNK]" are not actually part of the vocab,
        # but the decoder is happy as long they are defined in the token list
        # even if they do not exist as label index in the softmax output,
        blank_token="[blank]",
        sil_token="[SILENCE]",
        unk_word="[unknown]",
        nbest=1,
        beam_size=kwargs["beam_size"],
        beam_size_token=kwargs.get("beam_size_token", None),
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

    run_ctx.running_audio_len_s = 0
    run_ctx.total_am_time = 0
    run_ctx.total_search_time = 0
