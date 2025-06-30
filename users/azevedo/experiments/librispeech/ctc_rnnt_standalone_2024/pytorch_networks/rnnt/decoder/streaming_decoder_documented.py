"""
Experimental RNNT decoder
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch
from torch import nn
from torchaudio.models import RNNT
from functools import reduce

from .documented_rnnt_beam_search import RNNTBeamSearch, Hypothesis
from .chunk_handler import AudioStreamer, StreamingASRContextManager
from ..auxil.functional import Mode

@dataclass
class DecoderConfig:
    # files
    returnn_vocab: Union[str, Any]

    # search related options:
    beam_size: int

    # LM vars e.g. "lm.lstm.some_lstm_variant_file.Model"
    lm_module: Optional[str]
    lm_model_args: Optional[Dict[str, Any]]
    lm_checkpoint: Optional[str]
    lm_scale: float = 0.0
    zero_ilm_scale: float = 0.0

    # streaming definitions if mode == Mode.STREAMING
    mode: Union[Mode, str] = None
    chunk_size: Optional[int] = None
    carry_over_size: Optional[float] = None
    lookahead_size: Optional[int] = None
    stride: Optional[int] = None

    pad_value: Optional[float] = None

    # for new hash
    test_version: Optional[float] = None

    eos_penalty: Optional[float] = None

    # prior correction
    blank_log_penalty: Optional[float] = None

    # batched encoder config
    batched_encoder = False

    # extra compile options
    use_torch_compile: bool = False
    torch_compile_options: Optional[Dict[str, Any]] = None


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True

    # LM model package path
    lm_package: Optional[str] = None


class Transcriber(nn.Module):
    def __init__(
            self,
            feature_extraction: nn.Module,
            encoder: nn.Module,
            mapping: nn.Module,
            chunk_size: Optional[int] = None,
            lookahead_size: Optional[int] = None,
            carry_over_size: Optional[int] = None
    ) -> None:
        super().__init__()

        self.feature_extraction = feature_extraction
        self.encoder = encoder
        self.mapping = mapping

        self.chunk_size = chunk_size
        self.lookahead_size = lookahead_size
        self.carry_over_size = carry_over_size

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param input:
        :param lengths:
        :return:
        """
        squeezed_features = torch.squeeze(input)
        with torch.no_grad():
            self.feature_extraction.set_mode(Mode.OFFLINE)
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, lengths)

        # mask = mask_tensor(audio_features, audio_features_len)

        self.encoder.set_mode_cascaded(Mode.OFFLINE)
        print(f"{audio_features.shape = }, {audio_features_len.shape = }")
        encoder_out, out_mask = self.encoder(
            audio_features, audio_features_len,
            lookahead_size=self.lookahead_size, carry_over_size=self.carry_over_size,
        )
        encoder_out = self.mapping(encoder_out)
        encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        return encoder_out, encoder_out_lengths

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:

        if self.chunk_size is None:
            output, out_lengths = self.forward(input, lengths)
            return output, out_lengths, [[]]
                
        with torch.no_grad():
            # TODO: TEST!
            chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(self.chunk_size))
            audio_features, audio_features_len = self.feature_extraction.infer(input, lengths, chunk_size_frames)

            encoder_out, out_mask, state = self.encoder.infer(
                audio_features, audio_features_len,
                states,
                chunk_size=chunk_size_frames,
                lookahead_size=self.lookahead_size
            )

            encoder_out = self.mapping(encoder_out)  # [1, C', F'']
            encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        return encoder_out, encoder_out_lengths, [state]


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    config = DecoderConfig(**kwargs["config"])
    config.mode = {str(m): m for m in Mode}[config.mode]
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.config = config

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    run_ctx.blank_log_penalty = config.blank_log_penalty

    from returnn.datasets.util.vocabulary import Vocabulary

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    run_ctx.labels = vocab.labels

    print("create RNNT model...")
    model = run_ctx.engine._model
    model.joiner.set_mode(config.mode)

    rnnt_model = RNNT(
        transcriber=Transcriber(
            feature_extraction=model.feature_extraction,
            encoder=model.conformer,
            mapping=model.encoder_out_linear,
            chunk_size=config.chunk_size,
            lookahead_size=model.lookahead_size,
            carry_over_size=config.carry_over_size,
        ),
        predictor=model.predictor,
        joiner=model.joiner,
    )

    lm_model = None
    if config.lm_module is not None:
        # load LM
        assert extra_config.lm_package is not None
        lm_module_prefix = ".".join(config.lm_module.split(".")[:-1])
        lm_module_class = config.lm_module.split(".")[-1]

        LmModule = __import__(
            ".".join([extra_config.lm_package, lm_module_prefix]),
            fromlist=[lm_module_class],
        )
        LmClass = getattr(LmModule, lm_module_class)

        lm_model = LmClass(**config.lm_model_args)
        checkpoint_state = torch.load(
            config.lm_checkpoint,
            map_location=run_ctx.device,
        )
        lm_model.load_state_dict(checkpoint_state["model"])
        lm_model.to(device=run_ctx.device)
        lm_model.eval()


        print("loaded external LM")
        print()

    run_ctx.rnnt_decoder = RNNTBeamSearch(
        model=rnnt_model,
        blank=model.cfg.label_target_size,
        blank_penalty=run_ctx.blank_log_penalty,
        device=run_ctx.device,
        lm_model=lm_model,
        lm_scale=config.lm_scale,
        zero_ilm_scale=config.zero_ilm_scale,
        lm_sos_token_index=0,
    )
    print("done!")

    run_ctx.beam_size = config.beam_size

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.print_rtf:
        print(
            "Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s)
        )


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', 1]
    raw_audio_len = data["raw_audio:size1"].cpu()  # [B]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

        start = time.time()

    config: DecoderConfig = run_ctx.config

    hyps, p_hyps = [], []
    for i in range(raw_audio.shape[0]):
        if config.mode == Mode.OFFLINE:
            hyp = process_offline_sample(
                raw_audio=raw_audio[i], raw_audio_len=raw_audio_len[i], run_ctx=run_ctx
            )
        else:
            hyp, hypo_trie = process_streaming_sample(
                raw_audio=raw_audio[i], raw_audio_len=raw_audio_len[i], 
                config=config, run_ctx=run_ctx
            )
            p_hyps.append(hypo_trie)
        hyps.append(hyp)

    if run_ctx.print_rtf:
        total_time = time.time() - start
        run_ctx.total_time += total_time
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (total_time, total_time / audio_len_batch))

    for i, (hyp, tag) in enumerate(zip(hyps, data["seq_tag"])):
        sequence = [run_ctx.labels[idx] for idx in hyp if idx < len(run_ctx.labels)]
        text = " ".join(sequence).replace("@@ ", "")
        if run_ctx.print_hypothesis:
            if p_hyps:
                hypo_trie = p_hyps[i]
                _, _, path = follow_trie(hypo_trie, hyp, ret_path=True)
                path_seq = [[run_ctx.labels[idx] for idx in subp if idx < len(run_ctx.labels)] for subp in path]
                path_seq = map(lambda x: " ".join(x), path_seq)
                path_seq = "|".join(list(path_seq)).replace("@@ ", "")  # @@ at boundaries are ignored intentionally
                print(path_seq)
            else:
                print(text)

        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))


def process_offline_sample(
        raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, run_ctx
) -> torch.Tensor:
    hypothesis: List[Hypothesis] = None

    hypothesis, _ = run_ctx.rnnt_decoder.infer(
        input=raw_audio,
        length=raw_audio_len,
        beam_width=run_ctx.beam_size,
    )
    return hypothesis[0].tokens[:-1]  # remove <S> token

def process_streaming_sample(
        raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, config, run_ctx
) -> torch.Tensor:
    chunk_streamer = AudioStreamer(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        left_size=config.chunk_size,
        right_size=config.lookahead_size,
        stride=config.stride,
        pad_value=config.pad_value,
    )

    hypo_trie = {}
    hypothesis: List[Hypothesis] = None
    # context manager for handling states of streaming inference (only need to pass chunk and its effective size)
    with StreamingASRContextManager(run_ctx.rnnt_decoder, carryover_sz=config.carry_over_size, beam_sz=run_ctx.beam_size) as cm:
        for chunk, eff_chunk_sz in chunk_streamer:
            _, hypothesis = cm.process_chunk(ext_chunk=chunk, eff_chunk_sz=eff_chunk_sz) 

            # extend trie dict
            for hypo in hypothesis:
                hypo_trie, cur = insert_trie(hypo_trie, hypo.tokens)

    return hypothesis[0].tokens[:-1], hypo_trie


def follow_trie(trie: Dict, tokens: List, ret_path: bool = False) -> Union[Tuple[dict, int], Tuple[Dict, int, List]]:
    cur = trie
    path = []
    cur_subpath = []
    for i, token in enumerate(tokens):
        if ret_path:
            if "end" in cur:
                path.append(cur_subpath)
                cur_subpath = []
            cur_subpath.append(token)

        if token not in cur:
            return ValueError("Hypo not in trie")
        
        cur = cur[token]

    if "end" in cur:
        path.append(cur_subpath)
    if ret_path:
        return cur, i, path

    return cur, i

def insert_trie(trie: dict, tokens: list) -> dict:
    cur = trie
    for token in tokens:
        cur = cur.setdefault(token, {})
    # indicate that partial hypo ended here
    cur["end"] = True

    return trie, cur