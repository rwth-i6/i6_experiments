import torch
import numpy as np
from torchaudio.models.rnnt import RNNT
from returnn.frontend import Tensor
from returnn.tensor.tensor_dict import TensorDict
from i6_experiments.users.berger.pytorch.forward.transducer_beam_search import beam_search
from sisyphus import tk
from i6_core.lib.lexicon import Lexicon


def beam_search_forward_step(*, model: RNNT, extern_data: TensorDict, lexicon_file: tk.Path, beam_size: int, **kwargs):
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
