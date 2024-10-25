from dataclasses import dataclass
import time
import numpy as np
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable
import returnn.frontend as rf

if TYPE_CHECKING:
    from returnn.tensor import TensorDict

# flashlight decoder for 4-gram LM combined with CTC using different labels
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
    log_add: bool = False # use log-add when merging hypothese

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


# initialization done in get_model function, it's hacky.

def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

    model_def = config.typed_value("_model_def")
    model_args = config.typed_value("model_args")
    search_args = config.typed_value("search_args")
    # add the flashlight decoder to search_args, hacky
    decoder_config_args = config.typed_value("decoder_config_args")
    decoder_config = DecoderConfig(**decoder_config_args)
    extra_config_args = config.typed_value("extra_config_args", {})
    extra_config = ExtraConfig(**extra_config_args)

    import torch
    from torchaudio.models.decoder import ctc_decoder

    from returnn.util.basic import cf
    from returnn.datasets.util.vocabulary import Vocabulary
    if decoder_config.arpa_lm is not None:
        lm = cf(decoder_config.arpa_lm)
    else:
        lm = None
    vocab = Vocabulary.create_vocab(vocab_file=decoder_config.returnn_vocab, unknown_label=None)
    labels = vocab.labels
    flashlight_decoder = ctc_decoder(
        lexicon=decoder_config.lexicon,
        lm=lm,
        lm_weight=decoder_config.lm_weight,
        tokens=labels + ["[blank]"],
        blank_token="[blank]",
        sil_token="[blank]",
        unk_word="[unknown]",
        nbest=1,
        beam_size=decoder_config.beam_size,
        beam_size_token=decoder_config.beam_size_token,
        beam_threshold=decoder_config.beam_threshold,
        sil_score=decoder_config.sil_score,
        word_score=decoder_config.word_score,
        log_add=decoder_config.log_add,
    )
    blank_log_penalty = decoder_config.blank_log_penalty
    prior = np.loadtxt(decoder_config.prior_file, dtype="float32")
    prior_scale = decoder_config.prior_scale
    # hacky
    search_args.update({
        'flashlight_decoder': {'decoder': flashlight_decoder,
                               'prior': prior,
                               'prior_scale': prior_scale,
                               'blank_log_penalty': blank_log_penalty,
    }
    })

    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim, model_args=model_args, search_args=search_args)
    return model

def model_recog_flashlight_ctc(
    *,
    model,
    data,
    data_spatial_dim,
    max_seq_len: Optional[int] = None,
):

    import torch
    from returnn.config import get_global_config

    config = get_global_config()
    search_args = config.typed_value("search_args", {})
    # this part maybe not useful?
    mask_eos = search_args.get("mask_eos_output", True)
    add_eos_to_blank = search_args.get("add_eos_to_blank", False)
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))

    # sepcify encoder output name, and how to compute the ctc output
    collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    input_length_cpu = enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()

    assert hasattr(model, "ctc_output_layer")
    #assert model.enc_aux_logits_12, "Expected final ctc logits in enc_aux_logits_12"

    #enc_ctc = model.enc_aux_logits_12(enc_args["enc"])
    #enc_ctc = model.ctc_output_layer(collected_outputs["11"])
    ctc_enc_layer_id = model.ctc_enc_layer_id
    if isinstance(ctc_enc_layer_id, int):
        ctc_enc_layer_id = str(ctc_enc_layer_id)
    else:
        assert isinstance(ctc_enc_layer_id, str)
    enc_ctc = model.ctc_output_layer(collected_outputs[ctc_enc_layer_id])
    batch_size_dim = batch_dims[0]


    ctc_logit = enc_ctc.copy_transpose(
        (batch_size_dim, enc_spatial_dim, model.target_dim_w_blank)
    ) # [B,T,V+1]
    torch_ctc_log_prob = torch.nn.functional.log_softmax(ctc_logit.raw_tensor, dim=-1)
    assert 'flashlight_decoder' in model.search_args.keys(), 'provide a flashlight decoder'
    flashlight_decoder = model.search_args['flashlight_decoder']['decoder']
    blank_log_penalty = model.search_args['flashlight_decoder']['blank_log_penalty']
    prior = model.search_args['flashlight_decoder']['prior']
    prior_scale = model.search_args['flashlight_decoder']['prior_scale']
    log_prob_cpu = torch_ctc_log_prob.cpu()
    if blank_log_penalty is not None:
        # assume blank is the last
        log_prob_cpu[:, :, -1] -= blank_log_penalty
    if prior is not None:
        log_prob_cpu -= prior_scale * prior

    hyp_list = flashlight_decoder(log_prob_cpu, input_length_cpu)
    # List[List[CTCHypothesis]]
    # hypotheses in shape:
    # tokens: torch.LongTensor
    # """Predicted sequence of token IDs. Shape `(L, )`, where `L` is the length of the output sequence"""
    #
    # words: List[str]
    # """List of predicted words.
    #
    # Note:
    #     This attribute is only applicable if a lexicon is provided to the decoder. If
    #     decoding without a lexicon, it will be blank. Please refer to :attr:`tokens` and
    #     :func:`~torchaudio.models.decoder.CTCDecoder.idxs_to_tokens` instead.
    # """
    #
    # score: float
    # """Score corresponding to hypothesis"""
    #
    # timesteps: torch.IntTensor
    # """Timesteps corresponding to the tokens. Shape `(L, )`, where `L` is the length of the output sequence"""

    # when using phonemes this can be problematic, as the tokens need the lexicon to map phonemes to words
    # for chars no need to worry yet, but in the end this should be solved
    # get hyp lengths, pad hyps
    # convert to rf.Tensor
    #token_lists = [[hyp.tokens for hyp in beam] for beam in hyp_list]
    # for simplicity, create a torch tensor List(T,b) then B then use pad_sequence
    import torch.nn.utils.rnn.pad_sequence as pad_sequence
    token_batch = []
    for hyp_batch in hyp_list:
        beam_tokens = [hyp.tokens for hyp in hyp_batch] # list of torch tensors
        beam_tokens = pad_sequence(beam_tokens,batch_first=False, padding_value=0) # (T,beam)
        token_batch.append(beam_tokens)
    torch_tokens = pad_sequence(token_batch, batch_first=True, padding_value=0) #(B,T, beam)
    torch_tokens = torch_tokens.transpose(1,2) # (B,beam, T)

    hyp_lengths = [[hyp.tokens.shape[0] for hyp in beam] for beam in hyp_list]

    # or modify returnn to allow direct word outputs? seq-tag?


    return hypotheses

def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import batch_dim, Tensor, Dim
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    recog_def = config.typed_value("_recog_def")
    recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim)

    # print sequence tag
    # print(extern_data["seq_tag"].raw_tensor)

    # recog results including beam {batch, beam, out_spatial},
    # log probs {batch, beam},
    # out_spatial_dim,
    # final beam_dim

    out_spatial_dims = Dim(hyp_lengths)
    hyps, scores, out_spatial_dim, beam_dim = recog_out


    rf.get_run_ctx().mark_as_output(hyps, "hyps", dims=[batch_dim, beam_dim, out_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, beam_dim])










    return None
