import torch
from torchaudio.models.rnnt import RNNT
from torchaudio.models.rnnt_decoder import RNNTBeamSearch, _get_hypo_tokens, _get_hypo_score
import numpy as np
from returnn.frontend import Tensor
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk
from i6_core.lib.lexicon import Lexicon


def beam_search_forward_step(
    *, model: RNNT, extern_data: TensorDict, lexicon_file: tk.Path, beam_size: int, blank_id: int, **kwargs
):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beam_search = RNNTBeamSearch(model=model, blank=blank_id, step_max_tokens=1)

    tokens_arrays = []
    tokens_len = []

    for b in range(audio_features.size(0)):
        top_hypotheses = beam_search.forward(
            input=audio_features[b : b + 1, : audio_features_len[b]].to(device=device),
            length=audio_features_len[b : b + 1].to(device=device),
            beam_width=beam_size,
        )
        top_hypothesis = top_hypotheses[0]
        tokens = _get_hypo_tokens(top_hypothesis)
        score = _get_hypo_score(top_hypothesis)

        clean_tokens = filter(lambda idx: idx != blank_id, tokens)
        clean_tokens_str = [label_list[token] for token in clean_tokens]
        print(f"Recognized sequence {repr(seq_tags[b])}. Top hypothesis (score {score}, avg: {score / len(tokens)}):")
        print(f"    {repr(clean_tokens_str)}")

        tokens_array = np.array(clean_tokens_str, dtype="<U3")
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
