"""
Prototype greedy AED decoder
"""
import importlib
from dataclasses import dataclass
import time
import torch
import numpy
from typing import Dict, Tuple, Any

from .shared.interface import LabelScorerIntf
from .shared.beam_search_v1 import BeamSearchOpts, beam_search

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..conformer_zoneout_prototype_0624.aed_prototype_v2 import Model

@dataclass
class DecoderConfig:
    returnn_vocab: str
    beam_search_opts: BeamSearchOpts
    lm_module: str
    lm_args: Dict[str, Any]
    lm_scale: float
    lm_checkpoint: str

    @staticmethod
    def from_dict(d):
        d = d.copy()
        d["beam_search_opts"] = BeamSearchOpts(**d["beam_search_opts"])
        return DecoderConfig(**d)


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


class AEDLabelScorer(LabelScorerIntf):
    
    def __init__(self, model: "Model", lm_model: torch.nn.Module, lm_scale: float, encoder_outputs, audio_features_len):
        """

        :param model:
        :param encoder_outputs: as [beam, T, D] with repetition over beam
        """
        super().__init__()
        self.model = model
        self.lm_model = lm_model
        self.lm_scale = lm_scale
        self.encoder_outputs = encoder_outputs
        self.audio_features_len = audio_features_len

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
        """
        :param batch_size:
        :param device:
        :return: state. all tensors are expected to have shape [Batch, Beam=1, ...].
        """
        return None

    def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """
        :param prev_state: state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape [Batch, Beam, ...].
        :param prev_label: shape [Batch, Beam] -> index in [0...Label-1]
        :return: (scores, state).
            scores: shape [Batch, Beam, Label], log-prob-like scores.
                Broadcasting is allowed for any of the dims (e.g. think of :class:`LengthRewardScorer`).
            state: all tensors are expected to have shape [Batch, Beam, ...].
        """
        # remove batch axis of size 1 as the decoder can not work with that
        if prev_state is not None:
            lstm_state_1, lstm_state_2, att_context, accum_att_weights, lm_h, lm_c = tuple([state.squeeze(0) for state in prev_state])
            # pack lstm state
            decoder_state = (lstm_state_1, lstm_state_2), att_context, accum_att_weights
            lm_in_state = lm_h, lm_c
        else:
            # initialize LM state
            _, lm_in_state = self.lm_model(torch.zeros_like(prev_label, device=prev_label.device), None)


        # just treat batch_axis which is forced 1 as time axis forced 1
        single_labels = torch.transpose(prev_label, 0, 1)  # [1, beam] -> [Beam, 1]

        # in the first step we have beam size 1, ugly workaround for now
        if single_labels.shape[0] != self.encoder_outputs.shape[0]:
            encoder_outputs = self.encoder_outputs[0].unsqueeze(0)
            audio_features_len = self.audio_features_len[0].unsqueeze(0)
        else:
            encoder_outputs = self.encoder_outputs
            audio_features_len = self.audio_features_len

        decoder_logits, states = self.model.decoder(
            encoder_outputs, single_labels, audio_features_len, shift_embeddings=prev_state is None, state=prev_state
        )  # [B,1,Vocab] and state of [B, *]
        decoder_log_softmax = torch.nn.functional.log_softmax(decoder_logits, dim=-1)  # [beam, 1, #vocab]
        decoder_log_softmax = torch.transpose(decoder_log_softmax, 0, 1)  # [1, beam, #vocab]

        lm_logits, lm_out_state = self.lm_model.forward(single_labels, lm_in_state)
        lm_log_softmax = torch.nn.functional.log_softmax(lm_logits, dim=-1)  # [beam, 1, #vocab]
        lm_log_softmax = torch.transpose(lm_log_softmax, 0, 1)  # [1, beam, #vocab]

        # log prob combination
        log_softmax = decoder_log_softmax + self.lm_scale * lm_log_softmax

        # first unpack decoder lstm and lm lstm
        states = states[0][0], states[0][1], states[1], states[2], lm_out_state[0], lm_out_state[1]
        # [Batch, beam, ... required, so expand]
        expanded_states = [state.unsqueeze(0) for state in states]
        return log_softmax, expanded_states


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    config = DecoderConfig.from_dict(kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=config.returnn_vocab, unknown_label=None)
    run_ctx.labels = vocab.labels

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis
    run_ctx.config = config

    checkpoint_state = torch.load(
        config.lm_checkpoint,
        map_location=run_ctx.device,
    )
    step = checkpoint_state["step"]
    epoch = checkpoint_state["epoch"]

    lm_module = importlib.import_module(
        "..." + config.lm_module, package="pytorch_networks.decoder"
    )
    run_ctx.language_model = lm_module.Model(epoch=epoch, step=step, **config.lm_args)
    run_ctx.lm_scale = config.lm_scale

    missing_keys, unexpected_keys = run_ctx.language_model.load_state_dict(checkpoint_state["model"], strict=False)
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


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    print("Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s))

def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000

    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s += audio_len_batch
        am_start = time.time()

    conformer_out, encoder_seq_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        bpe_labels=None,
        do_search=False,
        encoder_only=True,
    )

    # run search for each example individually
    hyps = []
    for encoder_sequence, encoder_seq_len in zip(conformer_out, encoder_seq_len):
        encoder_sequence = torch.unsqueeze(encoder_sequence, 0).expand(
            (run_ctx.config.beam_search_opts.beam_size, encoder_sequence.shape[0], encoder_sequence.shape[1])
        )
        encoder_seq_len = torch.unsqueeze(encoder_seq_len, 0).expand(
            (run_ctx.config.beam_search_opts.beam_size,)
        )
        print("base shapes")
        print(encoder_sequence.shape)
        print(encoder_seq_len.shape)
        print("build scorer")
        label_scorer = AEDLabelScorer(
            model=model,
            encoder_outputs=encoder_sequence,
            audio_features_len=encoder_seq_len,
            lm_model=run_ctx.language_model,
            lm_scale=run_ctx.lm_scale,
        )
        seq_targets, seq_log_prob, out_seq_len = beam_search(
            label_scorer=label_scorer,
            batch_size=1,
            max_seq_len=torch.max(encoder_seq_len).unsqueeze(0),
            device=encoder_sequence.device,
            opts=run_ctx.config.beam_search_opts
        )

        # first entry in batch (sized 1 anyway), first sequence in beam
        hyp = seq_targets[0][0][:out_seq_len[0][0]]
        hyps.append(hyp.cpu().numpy())


    tags = data["seq_tag"]

    if run_ctx.print_rtf:
        am_time = time.time() - am_start
        run_ctx.total_time += am_time
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time, am_time / audio_len_batch))

    # hypothesis = run_ctx.ctc_decoder(logprobs.cpu(), audio_features_len.cpu())
    for hyp, tag in zip (hyps, tags):
        sequence = " ".join([run_ctx.labels[i] for i in hyp])
        text = sequence.replace("@@ ", "")
        print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))