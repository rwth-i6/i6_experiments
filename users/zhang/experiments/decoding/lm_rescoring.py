from __future__ import annotations
import functools
from typing import Optional, Union, Any, Dict, List, Callable, TYPE_CHECKING
from sisyphus import Job, Task, tk

from i6_experiments.common.datasets.librispeech.language_model import (
    get_librispeech_normalized_lm_data,
)
from sisyphus.delayed_ops import DelayedBase

from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, ModelDefWithCfg, ModelDef, RescoreDef
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    get_content_dir_from_hub_cache_dir,
)

from .rescoring import combine_scores, rescore
from .prior_rescoring import prior_score, Prior

from i6_experiments.users.zhang.datasets.librispeech import (
    get_train_corpus_text,
    _get_test_corpus_text,
    get_test_corpus_text,
)

if TYPE_CHECKING:
    from i6_experiments.users.zhang.experiments.lm.ffnn import FeedForwardLm
    from returnn.frontend.decoder.transformer import TransformerDecoder
from i6_core.util import uopen

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import abc
from typing import List, Dict, Type, Any
import torch
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim

class LMScorer(abc.ABC):
    """
    Abstract interface for language model scorers.
    """

    @abc.abstractmethod
    def score(self, sequence: str) -> float:
        """Compute score or log-probability for a single token sequence."""
        pass

    def batch_score(self, sequences: Optional[List[str], Tensor]) -> Optional[List[str], Tensor]:
        """Optional batch scoring; default to sequential calls to `score`."""
        return [self.score(seq) for seq in sequences]

    def finish(self):
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'LMScorer':
        """Create an instance from a config dict."""
        pass


class ScorerFactory:
    """
    Factory/registry for LMScorer implementations.
    """
    registry: Dict[str, Type[LMScorer]] = {}

    @classmethod
    def register(cls, key: str):
        def decorator(model_cls: Type[LMScorer]) -> Type[LMScorer]:
            cls.registry[key] = model_cls
            return model_cls
        return decorator

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> LMScorer:
        lm_type = cfg.get("lm_type")
        if lm_type not in cls.registry:
            raise ValueError(f"Unknown lm_type '{lm_type}' in config")
        model_cls = cls.registry[lm_type]
        return model_cls.from_config(cfg)

# TODO: this is not going to work straightforward for RETURNN models, the returnn engine need configured for any returnn models.
#  But it is possible to be rewritten as a rescor_def that goes to forwardJob
# Can also be a downstream job that take the top1 of reordered n-best list. Just use a dummy LM_scorer
class LmRescoringJob(Job):
    """
    Generic rescoring job that uses any LMScorer. The LM must implement the interface
    return new scores with given hyps
    Takes input: recog_out_file and lm_config.
    """
    def __init__(self, *, lm_cfg: dict, recog_out_file: tk.Path, version:int = 0):
        super().__init__()
        #self.name = f"HFLM-RESCO-{llm_name}"
        self.scorer = None
        self.lm_cfg = lm_cfg
        self.n_best_file = recog_out_file
        self.out_file = self.output_path(_v2_forward_out_filename)
        self.rqmt = {"time": 4, "cpu": 3, "mem": 8, "gpu": 1, "gpu_mem": 23}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import math
        self.scorer: LMScorer = ScorerFactory.build(self.lm_cfg)
        prev_one_ctx = self.lm_cfg.get("prev_one_ctx",False)

        import returnn.util.basic as util

        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        def _report_dev_memory_stats():
            dev = torch.device(device_str)
            if dev.type == "cuda":
                stats = [
                    f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                    f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                    f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                    f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
                ]
                print(f"Memory usage ({device_str}):", " ".join(stats))

        def write_multi_entries(file, scores, hyps):
            assert len(scores) == len(hyps)
            for score, hyp in zip(scores, hyps):
                file.write(f"  ({score!r}, {hyp!r}),\n")

        _report_dev_memory_stats()
        import gzip
        import i6_core.util as cutil
        d_rec = eval(cutil.uopen(self.n_best_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        out_file = gzip.open(self.out_file.get_path(), "wt")
        out_file.write("{\n")
        # Iterate records
        lines_seen = 0
        total_lines = sum(len(n_best) for _, n_best in d_rec.items())
        log_every = 1000  # print a message every 1k lines

        for seq_tag, n_best in d_rec.items():
            out_file.write(f"{seq_tag!r}: [\n")
            hyps = [x[1] for x in n_best]
            lines_seen += len(hyps)
            #am_scores = [x[0] for x in n_best]
            lm_scores = self.scorer.batch_score(hyps)
            # reorder and select top
            if lines_seen % log_every == 0:
                print(f"[Line {lines_seen:,}/{total_lines}] {100 * lines_seen / total_lines:.2f}% processed…")
                _report_dev_memory_stats()
            write_multi_entries(out_file, lm_scores, hyps)
            # Should not do reordering here, there is an existing takeBestJob in downstream part
            # This is for add context
            if prev_one_ctx:
                reorder = list(zip(lm_scores, hyps))
                reorder.sort(key=lambda x: x[0], reverse=True)
                self.scorer.prompt = raw_text_from_bpe_seq(reorder[0][1].split())
            # out_file.write(f"  ({reorder[0][1]!r}, {reorder[0][2]!r}),\n")
            out_file.write("],\n")

        # cleanup
        self.scorer.finish()
        torch.cuda.empty_cache()
        import gc;gc.collect()
        print("Finished and cleaned up.")

        out_file.write("}\n")
        out_file.close()


def raw_text_from_bpe_seq(seq:list):
    return " ".join(seq).replace("@@ ","").replace(" <s>", "")

_v2_forward_out_filename = "output.py.gz"
_v2_forward_ext_out_filename = "output_ext.py.gz"

@ScorerFactory.register("Dummy")
class DummyLmScorer(LMScorer):
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'DummyLmScorer':
        return cls()

    def score(self, sequence: str) -> float:
        pass

    def batch_score(self, sequence: List[str]) -> List[float]:
        return [0.0 for _ in sequence]

    def finish(self):
        # cleanup
        pass

@ScorerFactory.register("HuggingFaceLm")
class HuggingFaceLmScorer(LMScorer):
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'HuggingFaceLmScorer':
        model_dir = cfg["model_dir"]
        batch_size = cfg.get("batch_size", 1)
        prompt = cfg.get("prompt", None)
        eos_symbol = cfg.get("eos_symbol", "")
        lower_case = cfg.get("lower_case", False)
        # dummy init
        instance = cls()
        instance.batch_size = batch_size
        instance.model_dir = model_dir
        delimiter = " " if not eos_symbol else (eos_symbol + " ")  # Not sure
        instance.prompt = None
        if isinstance(prompt, tk.Path):
            with open(prompt.get_path(), "r", encoding="utf-8") as f:
                prompt = [line.strip() for line in f.readlines()]
        if prompt:
            prompt +=  [""]  # +[""] So that for last prompt(or only one prompt) it also has eos
            instance.prompt = delimiter.join(prompt) # TODO:

        instance.lower_case = lower_case
        instance.eos_symbol = eos_symbol

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import time, torch
        start_time = time.time()
        print("Loading model...")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = get_content_dir_from_hub_cache_dir(instance.model_dir)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        print(f"\nTokenizer_max_length:{instance.tokenizer.model_max_length}\n")

        instance.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        instance.model.eval()
        instance.device = torch.device(device_str)
        instance.model.to(instance.device)

        print(f"({time.time() - start_time} secs)")
        #instance.n_best_file = recog_out_file
        #instance.batch_size = {"Llama-3.2-1B": 8, "Llama-3.1-8B": 3}.get(llm_name,1) if batch_size is None else batch_size
        #instance.lower_case = lower_case
        #instance.eos_symbol = eos_symbol
        return instance

    def score(self, sequence: str) -> float:
        pass

    def batch_score(self, sequence: List[str]) -> List[float]:
        # Given a list of sequences for scoring, return a list of LM score
        # Batching depends on initialised LM_scorer

        # Helper to process a batch of lines
        def _process_batch(batch_lines, batch_prompt, scores_buffer):
            enc_hyp = self.tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                add_special_tokens=False if self.prompt else True,
            )
            hyp_input_ids = enc_hyp["input_ids"].to(self.device)

            # Prepare inputs
            if self.prompt:
                enc_prompt = self.tokenizer(
                    batch_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                )
                input_ids = torch.cat([enc_prompt["input_ids"], enc_hyp["input_ids"]], dim=1).to(self.device)
                attention_mask = torch.cat([enc_prompt["attention_mask"], enc_hyp["attention_mask"]], dim=1).to(self.device)
            else:
                input_ids = enc_hyp["input_ids"].to(self.device)
                attention_mask = enc_hyp["attention_mask"].to(self.device)

            # Compute logits and log-probs
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask).logits
                gather_ids = hyp_input_ids[:, 1:].unsqueeze(-1)
                scores_mask = enc_hyp["attention_mask"][..., 1:].to(self.device)
                if self.prompt:
                    gather_ids = hyp_input_ids.unsqueeze(-1)
                    scores_mask = enc_hyp["attention_mask"].to(self.device)
                    logits = logits[:, -hyp_input_ids.shape[1] - 1:-1, :]

                log_probs = torch.log_softmax(logits, dim=-1)
                llm_scores = torch.gather(log_probs, dim=-1, index=gather_ids).squeeze()
                llm_scores = llm_scores * scores_mask
                llm_scores = llm_scores.sum(dim=1)

            scores_buffer.extend(llm_scores.tolist())

        batch_lines, batch_prompt, scores_buffer = [], [], []
        # Iterate records
        for hyp in sequence:
            hyp = raw_text_from_bpe_seq(hyp.split())
            line = hyp.strip().lower() if self.lower_case else hyp.strip()
            if not line: # Encounter an empty hyp
                if len(batch_lines)>0: # Process accumulated batch and append -1e30 as score for empty
                    _process_batch(batch_lines, batch_prompt, scores_buffer)
                    batch_lines, batch_prompt = [], []
                scores_buffer.append(-1e30)
                continue
            eos_symbol = (" " + self.tokenizer.eos_token) if self.eos_symbol == "eos" else self.eos_symbol
            batch_lines.append(line + eos_symbol)
            if self.prompt:
                batch_prompt.append(self.prompt.lower() if self.lower_case else self.prompt)

            if len(batch_lines) == self.batch_size:
                if len(batch_prompt)>1:
                    if batch_prompt[0] == batch_prompt[1]:
                        batch_prompt = [batch_prompt[0]] * len(batch_lines)
                _process_batch(batch_lines, batch_prompt, scores_buffer)
                batch_lines, batch_prompt = [], []

            # leftover
        if batch_lines:
            _process_batch(batch_lines, batch_prompt, scores_buffer)
        return scores_buffer


    def finish(self):
        # cleanup
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("Finished and cleaned up.")


'''RETURNN models, LMScorer Implementations
for rescoring seems not necessary but might be used to simplify the code in recog_def'''
@ScorerFactory.register("FeedForward")
class FeedForwardLmScorer(LMScorer):
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'FeedForwardLmScorer':
        lm_config = cfg["lm_config"]
        ckpt_file = cfg["filepath"]
        batch_size = cfg.get("batch_size", 1)
        weight = cfg.get("weight", 1)
        # dummy init
        instance = cls()
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        instance.device = torch.device(device_str)
        instance.batch_size = batch_size
        instance.scale = weight
        from i6_experiments.users.zhang.experiments.lm.ffnn import FeedForwardLm
        instance.model = FeedForwardLm(**lm_config)
        checkpoint_state = torch.load(ckpt_file, map_location=instance.device)
        instance.model.load_state_dict(
            checkpoint_state, strict=False
        )
        instance.model.eval()
        instance.model.to(instance.device)
        #instance.n_best_file = recog_out_file
        #instance.batch_size = {"Llama-3.2-1B": 8, "Llama-3.1-8B": 3}.get(llm_name,1) if batch_size is None else batch_size
        #instance.lower_case = lower_case
        #instance.eos_symbol = eos_symbol
        return instance

    def score(self, sequence: str) -> float:
        pass

    def batch_score(self, sequence: Tensor) -> Tensor:
        # Given a list of sequences for scoring, return a list of LM score
        # Batching depends on initialised LM_scorer

        # Helper to process a batch of lines
        def _process_batch(targets: Tensor, targets_spatial_dim: Dim, scores_buffer):
            vocab = self.model.vocab_dim.vocab
            assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

            targets_w_bos, (targets_w_eos_spatial_dim,) = rf.pad(
                targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
            )
            targets_w_eos, _ = rf.pad(
                targets,
                axes=[targets_spatial_dim],
                padding=[(0, 1)],
                value=vocab.eos_label_id,
                out_dims=[targets_w_eos_spatial_dim],
            )
            batch_dims = targets.remaining_dims(targets_spatial_dim)

            logits, _ = self.model(
                targets,
                spatial_dim=targets_spatial_dim,
                out_spatial_dim=targets_w_eos_spatial_dim,
                state=self.model.default_initial_state(batch_dims=batch_dims),
            )
            # import pdb; pdb.set_trace()
            log_prob = rf.log_softmax(logits, axis=self.model.vocab_dim)
            log_prob_targets = rf.gather(log_prob, indices=targets_w_eos,
                                         axis=self.model.vocab_dim)  # Why before it is indices = targets_w_eos?
            log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
            assert log_prob_targets_seq.dims_set == set(batch_dims)

            scores_buffer.extend(log_prob_targets_seq.raw_tensor.tolist())

        # Helper to prepare a batch input for returnn usage
        def _prepare_batch(batch_lines, lengths):
            from torch.nn.utils.rnn import pad_sequence
            padded_lines = pad_sequence(batch_lines, batch_first=True, padding_value=self.model.eos_idx).to(self.device)
            enc_dyn_lengths = rf.Tensor(
                name="data_dyn_lengths",
                dims=[batch_dim],
                dtype="int32",
                raw_tensor=torch.tensor(lengths).to(self.device),
            )
            data_spatial_dim = Dim(name="data_seq", dimension=enc_dyn_lengths, kind=Dim.Types.Spatial)
            hyps_r = rf.convert_to_tensor(padded_lines, dims=(batch_dim, data_spatial_dim), )
            return hyps_r, data_spatial_dim

        batch_lines, batch_prompt, scores_buffer, lengths = [], [], [], []
        # Iterate records
        for hyp in sequence:
            batch_lines.append([torch.tensor(self.model.target_dim.vocab.label_to_id(label)) for label in hyp.split()])
            lengths.append(len(hyp.split()))
            if len(batch_lines) == self.batch_size:
                hyps_r, data_spatial_dim = _prepare_batch(batch_lines, lengths)
                _process_batch(hyps_r, data_spatial_dim, scores_buffer)
                batch_lines, batch_prompt = [], []

            # leftover
        if batch_lines:
            hyps_r, data_spatial_dim = _prepare_batch(batch_lines, lengths)
            _process_batch(hyps_r, data_spatial_dim, scores_buffer)
        return scores_buffer


    def finish(self):
        # cleanup
        del self.model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("Finished and cleaned up.")


@ScorerFactory.register("Transformer")
class TransformerLmScorer(LMScorer): #TODO: for returnn model, the difference is minor and may not need a separate class
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> 'TransformerLmScorer':
        model_dir = cfg["model_dir"]
        lm_config = cfg["lm_config"]
        ckpt_file = cfg["filename"]
        batch_size = cfg.get("batch_size", 1)
        weight = cfg.get("weight", 1)
        # dummy init
        instance = cls()
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        instance.device = torch.device(device_str)
        instance.batch_size = batch_size
        instance.model_dir = model_dir
        instance.scale = weight
        from returnn.frontend.decoder.transformer import TransformerDecoder
        instance.model = TransformerDecoder(**lm_config)
        checkpoint_state = torch.load(ckpt_file, map_location=instance.device)
        instance.model.load_state_dict(
            checkpoint_state, strict=False
        )
        instance.model.eval()
        instance.model.to(instance.device)
        #instance.n_best_file = recog_out_file
        #instance.batch_size = {"Llama-3.2-1B": 8, "Llama-3.1-8B": 3}.get(llm_name,1) if batch_size is None else batch_size
        #instance.lower_case = lower_case
        #instance.eos_symbol = eos_symbol
        return instance

    def score(self, sequence: str) -> float:
        pass

    def batch_score(self, sequence: Tensor) -> Tensor:
        # Given a list of sequences for scoring, return a list of LM score
        # Batching depends on initialised LM_scorer

        # Helper to process a batch of lines
        def _process_batch(targets, targets_spatial_dim, scores_buffer):
            vocab = self.model.vocab_dim.vocab
            assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

            targets_w_bos, (targets_w_eos_spatial_dim,) = rf.pad(
                targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
            )
            targets_w_eos, _ = rf.pad(
                targets,
                axes=[targets_spatial_dim],
                padding=[(0, 1)],
                value=vocab.eos_label_id,
                out_dims=[targets_w_eos_spatial_dim],
            )
            batch_dims = targets.remaining_dims(targets_spatial_dim)
            logits, _ = self.model(
                    targets_w_bos,
                    spatial_dim=targets_w_eos_spatial_dim,
                    state=self.model.default_initial_state(batch_dims=batch_dims),
                )
                # import pdb;pdb.set_trace()
                # logits, pack_dim = rf.pack_padded(
                #     logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
                # )
                #  We need to mask out the padding..? Yeah, but before feeds to the model..? Isnt it already specified in w_eos_spatial_dim?
                # Transformerdecoder seems already use causal attention by default.

            # import pdb; pdb.set_trace()
            log_prob = rf.log_softmax(logits, axis=self.model.vocab_dim)
            log_prob_targets = rf.gather(log_prob, indices=targets_w_eos,
                                         axis=self.model.vocab_dim)  # Why before it is indices = targets_w_eos?
            log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
            assert log_prob_targets_seq.dims_set == set(batch_dims)
            scores_buffer.extend(log_prob_targets_seq.raw_tensor.tolist())

        batch_lines, batch_prompt, scores_buffer = [], [], []
        # Iterate records
        for hyp in sequence:
            batch_lines.append(hyp)
            if len(batch_lines) == self.batch_size:
                _process_batch(batch_lines, batch_prompt, scores_buffer)
                batch_lines, batch_prompt = [], []

            # leftover
        if batch_lines:
            _process_batch(batch_lines, batch_prompt, scores_buffer)
        return scores_buffer


    def finish(self):
        # cleanup
        del self.model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("Finished and cleaned up.")


def trafo_lm_rescore_def(*, model: rf.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    import returnn.frontend as rf

    targets_beam_dim  # noqa  # unused here

    # noinspection PyTypeChecker
    model: TransformerDecoder
    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=vocab.eos_label_id,
        out_dims=[targets_w_eos_spatial_dim],
    )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    logits, _ = model(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=None,
        state=model.default_initial_state(batch_dims=batch_dims),
    )

    log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
    log_prob_targets = rf.gather(
        log_prob, indices=targets_w_eos, axis=model.vocab_dim
    )  # [batch,beam,targets_spatial_w_eos]
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq

trafo_lm_rescore_def: RescoreDef


def ffnn_rescore_def(*, model: rf.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    import returnn.frontend as rf

    targets_beam_dim  # noqa  # unused here

    # noinspection PyTypeChecker
    model: FeedForwardLm
    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    targets_w_bos, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=vocab.eos_label_id,
        out_dims=[targets_w_eos_spatial_dim],
    )
    batch_dims = targets.remaining_dims(targets_spatial_dim)

    logits, _ = model(
        targets,
        spatial_dim=targets_spatial_dim,
        out_spatial_dim=targets_w_eos_spatial_dim,
        state=model.default_initial_state(batch_dims=batch_dims),
    )
    # import pdb; pdb.set_trace()
    log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
    log_prob_targets = rf.gather(log_prob, indices=targets_w_eos,
                                 axis=model.vocab_dim)  # Why before it is indices = targets_w_eos?
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq

ffnn_rescore_def: RescoreDef

def ngram_score(
    recog_output: RecogOutput,
    *,
    lm: tk.Path,
    lm_rescore_def: RescoreDef,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    alias_name: Optional[str] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param recog_output:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
        We ignore the scores here and just use the text of the hyps.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param rescore_rqmt:
    """
    vocab_opts_file # noqa
    assert lm_rescore_def == ngram_rescore_def
    return rescore(
        recog_output=recog_output,
        model=ModelWithCheckpoint(
            definition=ModelDefWithCfg(model_def=ngram_model_def, config={"_lm_file": lm}), checkpoint=None
        ),
        vocab=vocab,
        rescore_def=lm_rescore_def, # assert ngram_rescore_def,
        forward_rqmt=rescore_rqmt,
        forward_device="cpu",
        forward_alias_name=alias_name,
    )


def ngram_model_def(**_other):
    import torch
    from returnn.config import get_global_config
    import kenlm  # pip install kenlm

    config = get_global_config()

    class _NGramModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self._lm_file = config.typed_value("_lm_file")
            self.lm = kenlm.LanguageModel(self._lm_file)

    return _NGramModel()


ngram_model_def: ModelDef
ngram_model_def.behavior_version = 22
ngram_model_def.backend = "torch"
ngram_model_def.batch_size_factor = 1


def ngram_rescore_def(*, model: rf.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    import torch
    import kenlm
    import returnn.frontend as rf
    from returnn.tensor import batch_dim

    # noinspection PyUnresolvedReferences
    lm: kenlm.LanguageModel = model.lm
    vocab = targets.sparse_dim.vocab

    # https://github.com/kpu/kenlm/blob/master/python/example.py
    # https://github.com/kpu/kenlm/blob/master/python/kenlm.pyx

    assert targets.dims_set == {batch_dim, targets_beam_dim, targets_spatial_dim}
    targets = targets.copy_transpose((batch_dim, targets_beam_dim, targets_spatial_dim))

    res_raw = torch.zeros((batch_dim.get_dim_value(), targets_beam_dim.get_dim_value()))
    for i in range(batch_dim.get_dim_value()):
        targets_beam_size = targets_beam_dim.dyn_size_ext
        if batch_dim in targets_beam_size.dims:
            targets_beam_size = rf.gather(targets_beam_size, axis=batch_dim, indices=i)
        for j in range(targets_beam_size.raw_tensor.item()):
            seq_len = targets_spatial_dim.dyn_size_ext
            seq_len = rf.gather(seq_len, axis=targets_beam_dim, indices=j)
            seq_len = rf.gather(seq_len, axis=batch_dim, indices=i)
            assert seq_len.dims == ()
            targets_raw = targets.raw_tensor[i, j, : seq_len.raw_tensor]
            targets_str = vocab.get_seq_labels(targets_raw.numpy())
            res_raw[i, j] = lm.score(targets_str)

    # KenLM returns score in +log10 space.
    # We want to return in (natural) +log space.
    # 10 ** x = e ** (x * log(10))
    res_raw *= torch.log(torch.tensor(10.0))

    res = rf.convert_to_tensor(res_raw, dims=(batch_dim, targets_beam_dim))
    return res


ngram_rescore_def: RescoreDef

def lm_score(
    recog_output: RecogOutput,
    *,
    lm: ModelWithCheckpoint,
    lm_rescore_def: RescoreDef,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    alias_name: Optional[str] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param recog_output:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
        We ignore the scores here and just use the text of the hyps.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param rescore_rqmt:
    """
    return rescore(
        recog_output=recog_output,
        model=lm,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        rescore_def=lm_rescore_def,
        forward_rqmt=rescore_rqmt,
        forward_alias_name=alias_name,
    )

# TODO: consider to use RETURNN Job, LLM can be wrapped, see ngram_model_def
def HF_lm_score(
    recog_output: RecogOutput,
    *,
    lm: Dict[str, Any],
    lm_rescore_def: RescoreDef,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    alias_name: Optional[str] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param recog_output:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
        We ignore the scores here and just use the text of the hyps.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param rescore_rqmt:
    """
    lm_rescore_def, vocab, vocab_opts_file # noqa
    resco_job = LmRescoringJob(
        recog_out_file=recog_output.output,
        lm_cfg=lm,
        # {
        #     "lm_type": "HuggingFaceLm",
        #     "model_dir": dl_model.out_hub_cache_dir,
        #     "batch_size": 8,
        #     "weight": llm_scale,
        # }
    )
    resco_job.rqmt.update(rescore_rqmt)
    alias_ext = "HF_LM"
    if lm["lm_type"] == "Dummy":
        resco_job.rqmt = {"time": 1, "cpu": 1, "mem": 8}
        alias_ext = "/NoLM"
    if alias_name:
        resco_job.add_alias(alias_name + alias_ext)
    #resco_job.rqmt["gpu_mem"] = 48
    #print(f"HF_lm_score: LM:{lm}")
    #resco_job.add_alias("lm/" + lm["name"] + "/rescoring")
    # tk.register_output(
    #     lm["name"].replace(".","_") + "/rescoring/" + os.path.basename(recog_output.output.path).replace(".","_"),
    #     resco_job.out_file)
    return RecogOutput(output=resco_job.out_file)


def lm_am_framewise_prior_rescore(
    res: RecogOutput,
    *,
    dataset: DatasetConfig,
    raw_res_search_labels: RecogOutput,
    raw_res_labels: RecogOutput,
    am: Optional[ModelWithCheckpoint] = None,
    am_rescore_def: Optional[RescoreDef] = None,
    am_rescore_rqmt: Optional[Dict[str, Any]] = None,
    am_scale: Union[float, tk.Variable, DelayedBase] = 1.0,
    lm: Optional[ModelWithCheckpoint, dict],
    lm_rescore_def: RescoreDef,
    lm_scale: Union[float, tk.Variable, DelayedBase],
    lm_rescore_rqmt: Optional[Dict[str, Any]] = None,
    vocab: tk.Path,
    vocab_opts_file: tk.Path,
    prior: Optional[Prior] = None,
    prior_scale: Union[float, tk.Variable, DelayedBase] = 0.0,
    search_labels_to_labels: Optional[Callable[[RecogOutput], RecogOutput]] = None,
    alias_name: Optional[str] = None,
) -> RecogOutput:
    """
    With functools.partial, you can use this for ``recog_post_proc_funcs`` in :func:`recog_model` and co.

    If you also want to combine a prior, e.g. for CTC, you might want to use :func:`prior_rescore` first.

    :param res:
        The format of the JSON is: {"<seq_tag>": [(score, "<text>"), ...], ...},
        i.e. the standard RETURNN search output with beam.
    :param dataset: the orig data which was used to generate res
    :param raw_res_search_labels:
    :param raw_res_labels:
    :param am:
    :param am_rescore_def:
    :param am_rescore_rqmt:
    :param am_scale: scale for the new AM scores
    :param lm: language model, if a Dict is given-> external model like HF
    :param lm_scale: scale for the LM scores
    :param lm_rescore_rqmt:
    :param vocab: for LM labels in res / raw_res_labels
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param prior:
    :param prior_scale: scale for the prior scores. this is used as the negative weight
    :param search_labels_to_labels: function to convert the search labels to the labels
    """
    res, raw_res_search_labels, search_labels_to_labels  # noqa  # unused here
    if isinstance(lm, ModelWithCheckpoint): # RETURNN LM
        lm_scorer = lm_score
    elif isinstance(lm, tk.Path): # assert use ngram lm here
        lm_scorer = ngram_score
    elif lm is None or lm_scale == 0:
        if lm is not None: # This only for float case of the scale
            print(f"\nScale for {lm} is set to 0, not likely a tuned value! Use Dummy rescorer, set prior scale to 0 too\n")
        lm = {"lm_type": "Dummy"}
        prior_scale = 0.0
        lm_scorer = HF_lm_score
    else:
        lm_scorer = HF_lm_score
    def get_generic_alias_name(alias):
        # parts = alias.strip().split("/")
        # if len(parts) >= 2:
        #     return "/".join(parts[:2] + [parts[-1]])
        # else:
        #     return alias
        return alias

    alias_name = get_generic_alias_name(alias_name)
    res_labels_lm_scores = lm_scorer(
        raw_res_labels, lm=lm, lm_rescore_def=lm_rescore_def ,vocab=vocab, vocab_opts_file=vocab_opts_file, rescore_rqmt=lm_rescore_rqmt,
        alias_name=alias_name + "/rescoring",
    )
    res_labels_am_scores = rescore(
        recog_output=raw_res_labels,
        dataset=dataset,
        model=am,
        vocab=vocab,
        vocab_opts_file=vocab_opts_file,
        rescore_def=am_rescore_def,
        forward_rqmt=am_rescore_rqmt,
        forward_alias_name=alias_name + "/AMrescoring",
    )
    scores = [(am_scale, res_labels_am_scores), (lm_scale, res_labels_lm_scores)]
    if prior and prior_scale:
        assert search_labels_to_labels
        res_search_labels_prior_scores = prior_score(raw_res_search_labels, prior=prior)
        res_labels_prior_scores = search_labels_to_labels(res_search_labels_prior_scores)
        scores.append((prior_scale * (-1), res_labels_prior_scores))
    else:
        assert isinstance(prior_scale, (int, float)) and prior_scale == 0.0
    combine_scores_alias = alias_name + f"/combine{f'pr{prior_scale}'.replace('.','_') + f'_lm{lm_scale}'.replace('.','_')}"
    return combine_scores(scores, combine_scores_alias)

# "work/i6_core/returnn/search/SearchOutputRawReplaceJob.a5gb36CLt4N6/output/search_results.py.gz"
# "work/i6_core/returnn/search/SearchRemoveLabelJob.HxKTed4GQc38/output/search_results.py.gz"
'''
Trafo:
"work/i6_core/returnn/training/ReturnnTrainingJob.dbF2UjypTWBI/output/models/epoch.050.pt"
{
    "num_layers": 12,
    "model_dim": 512,
    "dropout": 0.0,
}
'''

def py():
    from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str
    lm_name = "ffnn_test_bpe128"
    lm_scale = 0.5
    resco_job = LmRescoringJob(
        recog_out_file=tk.Path(
            "work/i6_core/returnn/search/SearchRemoveLabelJob.HxKTed4GQc38/output/search_results.py.gz"),
        lm_cfg={
            "lm_type": "FeedForward",
            "filepath": "work/i6_core/returnn/training/ReturnnTrainingJob.lio07GWKTIob/output/models/epoch.050.pt",
            "lm_config":{
                "vocab_dim": Dim(name="vocab_dim",dimension=get_vocab_by_str("bpe128").dim), "context_size": 8, "num_layers": 2, "ff_hidden_dim": 2048, "dropout": 0.1
            },
            "batch_size": 20,
            "weight": lm_scale,
        }
    )
    resco_job.add_alias(
        "lm/" + lm_name + "/rescoring_" + f"w{lm_scale}".replace(".", "") )
    tk.register_output(
        lm_name + "/librispeech-rescoring_test" + f"w{lm_scale}".replace(".", ""),
        resco_job.out_file)
