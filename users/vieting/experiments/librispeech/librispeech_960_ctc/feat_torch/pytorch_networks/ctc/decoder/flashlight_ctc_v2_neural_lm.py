"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
"""

from dataclasses import dataclass
import importlib
import time
import numpy as np
from typing import Any, Dict, Optional
import torch
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState

@dataclass
class DecoderConfig():
    # search related options:
    beam_size: int
    beam_size_token: int
    beam_threshold: float

    # needed files
    lexicon: str
    returnn_vocab: str

    # lm stuff
    lm_module: str
    lm_args: Dict[str, Any]
    lm_checkpoint: str
    lm_vocab: str
    lm_bpe_codes: str
    lm_is_bpe: bool

    # additional search options
    lm_weight: float = 0.0
    sil_score: float = 0.0
    word_score: float = 0.0

    # prior correction
    blank_log_penalty: Optional[float] = None
    prior_scale: float = 0.0
    prior_file: Optional[str] = None




@dataclass
class ExtraConfig():
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True



@dataclass
class LMStateVector:
    score: float
    states: Any


class CustomLM(CTCDecoderLM):
    """Create a Python wrapper around `language_model` to feed to the decoder."""

    def __init__(self, language_model: torch.nn.Module, bpe_vocab, lm_vocab):
        CTCDecoderLM.__init__(self)
        self.language_model = language_model
        self.bpe_vocab = bpe_vocab
        self.lm_vocab = lm_vocab
        self.sil = -1  # index for silent token in the language model
        self.states: Dict[CTCDecoderLMState, LMStateVector] = {}
        self.start_state = None
        language_model.eval()

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        print("start")
        with torch.no_grad():
            # SOS is index 0
            inp = torch.tensor([0], dtype=torch.int64)  # [S]
            inp = torch.unsqueeze(inp, 0)  # [1, 1, 1]
            score, lm_states = self.language_model(inp, None)
            state_vector = LMStateVector(score, lm_states)
        self.states[state] = state_vector
        self.start_state = state
        self.num_lm_ping = 0
        self.states[state] = state_vector
        return state

    def score(self, state: CTCDecoderLMState, token_index: int):
        outstate = state.child(token_index)
        if state not in self.states:
            assert False, "Tried to access a deleted state"
        if token_index == len(self.lm_vocab):
            return LMStateVector(0.0, None), 0.0
        if outstate not in self.states:
            with torch.no_grad():
                word = self.lm_vocab[token_index]
                bpe_idxs = self.bpe_vocab.get_seq(word)
                print("computing word %s with indices %s in LM step %i" % (word, str(bpe_idxs), self.num_lm_ping))
                inp = torch.tensor(bpe_idxs, dtype=torch.int64)  # [S]
                inp = torch.unsqueeze(inp, 0)  # [1, S]
                score, lm_states = self.language_model(inp, self.states[state].states)
            if state != self.start_state:
                print(state)
                print(token_index)
                self.num_lm_ping += 1
            self.states[outstate] = LMStateVector(score, lm_states)
        score = self.states[outstate].score

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        print("NUM LM PING")
        print(self.num_lm_ping)
        return self.score(state, self.sil)


def load_lm(config, run_ctx):
    checkpoint_state = torch.load(
        config.lm_checkpoint,
        map_location=run_ctx.device,
    )
    step = checkpoint_state["step"]
    epoch = checkpoint_state["epoch"]

    lm_module = importlib.import_module(
        "..." + config.lm_module, package=__package__,
    )
    language_model = lm_module.Model(epoch=epoch, step=step, **config.lm_args)

    missing_keys, unexpected_keys = language_model.load_state_dict(checkpoint_state["model"], strict=False)
    if missing_keys:
        raise Exception(
            "\n".join(
                [
                    f"While loading model {config.lm_checkpoint}:",
                    "Unexpected key(s) in state_dict: " + ", ".join(map(repr, unexpected_keys)),
                    "Missing key(s) in state_dict: " + ", ".join(map(repr, missing_keys)),
                    "Any missing key is an error.",
                ]
            )
        )
    if unexpected_keys:
        print(
            f"Note: While loading {config.lm_checkpoint}, unexpected key(s) in state_dict: "
            + ", ".join(map(repr, unexpected_keys)),
        )
    return language_model.to(device=run_ctx.device)


def forward_init_hook(run_ctx, **kwargs):
    """

    :param run_ctx:
    :param kwargs:
    :return:
    """
    import torch
    from torchaudio.models.decoder import ctc_decoder

    from returnn.datasets.util.vocabulary import Vocabulary, BytePairEncoding
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    vocab = Vocabulary.create_vocab(
        vocab_file=config.returnn_vocab, unknown_label=None)
    labels = vocab.labels

    if config.lm_is_bpe:
        bpe_vocab = Vocabulary.create_vocab(
            vocab_file=config.returnn_vocab,
            bpe_file=config.lm_bpe_codes,
            unknown_label=None,
        )
    else:
        raise NotImplementedError("Only BPE LM supported for now")

    lm_index_to_word = {}
    with open(config.lexicon) as f:
        for i, line in enumerate(f.readlines()):
            lm_index_to_word[i] = line.split(" ")[0].strip()


    language_model = load_lm(config, run_ctx)
    neural_lm = CustomLM(language_model=language_model, bpe_vocab=bpe_vocab, lm_vocab=lm_index_to_word)

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=config.lexicon,
        lm=neural_lm,
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

    #if config.use_torch_compile:
    #    options = config.torch_compile_options or {}
    #    run_ctx.engine._model = torch.compile(run_ctx.engine._model, **options)

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
        print("Total-AM-Time: %.2fs, AM-RTF: %.3f" %
              (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s))
        print("Total-Search-Time: %.2fs, Search-RTF: %.3f" %
              (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s))
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

    am_time = time.time() - am_start
    run_ctx.total_am_time += am_time

    search_start = time.time()
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())
    assert False
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
