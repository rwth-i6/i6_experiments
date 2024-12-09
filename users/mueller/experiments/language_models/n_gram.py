from __future__ import annotations

# from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection, Dict
# import tree
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import hashlib
# import contextlib
# import functools

# from returnn.tensor import Tensor, Dim, single_step_dim
# import returnn.frontend as rf
# from returnn.frontend.tensor_array import TensorArray
# from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

# from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
#     RFModelWithMakeLabelScorer,
# )

# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import (
#     trafo_lm_kazuki_import,
# )
# from i6_experiments.users.yang.torch.utils.tensor_ops import mask_eos_label
# from i6_experiments.users.yang.torch.utils.masking import get_seq_mask_v2
# from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import ctc_prefix_posterior_v3
# from i6_experiments.users.yang.torch.lm.network.lstm_lm import LSTMLM, LSTMLMConfig

# if TYPE_CHECKING:
#     from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
#     from i6_experiments.users.zeyer.model_with_checkpoints import (
#         ModelWithCheckpoints,
#         ModelWithCheckpoint,
#     )

# _log_mel_feature_dim = 80


# class BigramLMRF(rf.Module):
#     r"""Recurrent neural network transducer (RNN-T) prediction network. NOTE: This is feed-forward not RNN-T.

#     From torchaudio, modified to be used with rf
#     """

#     def __init__(
#         self,
#         # cfg: PredictorConfig,
#         label_target_size: Dim,
#         output_dim: Dim,
#         symbol_embedding_dim: int = 128,
#         emebdding_dropout: float = 0.0,
#         num_ff_layers: int = 2,
#         ff_hidden_dim: int = 1000,
#         ff_dropout: float = 0.0,
#         use_bottleneck: bool = False,
#         bottleneck_dim: Optional[int] = 512,
#     ) -> None:
#         """

#         :param cfg: model configuration for the predictor
#         :param label_target_size: shared value from model
#         :param output_dim: Note that this output dim is for 1(!) single lstm.
#             The actual output dim is 2*output_dim since forward and backward lstm
#             are concatenated.
#         """
#         super().__init__()

#         self.label_target_size = label_target_size
#         self.output_dim = output_dim
#         self.embedding_dropout = emebdding_dropout
#         self.ff_dropout = ff_dropout
#         self.num_ff_layers = num_ff_layers
#         self.use_bottleneck = use_bottleneck

#         self.symbol_embedding_dim = Dim(
#             name="symbol_embedding", dimension=symbol_embedding_dim
#         )
#         self.ff_hidden_dim = Dim(name="ff_hidden", dimension=ff_hidden_dim)

#         self.embedding = rf.Embedding(label_target_size, self.symbol_embedding_dim)
#         # self.input_layer_norm = rf.LayerNorm(self.symbol_embedding_dim)

#         self.layers = rf.Sequential(
#             rf.Linear(
#                 self.symbol_embedding_dim if idx == 0 else self.ff_hidden_dim,
#                 self.ff_hidden_dim,
#             )
#             for idx in range(self.num_ff_layers)
#         )
#         if self.use_bottleneck:
#             self.bottleneck_dim = Dim(name="bottleneck", dimension=bottleneck_dim)
#             self.bottleneck = rf.Linear(self.ff_hidden_dim, self.bottleneck_dim)
#             self.final_linear = rf.Linear(self.bottleneck_dim, output_dim)
#         else:
#             self.final_linear = rf.Linear(self.ff_hidden_dim, output_dim)
#         # self.output_layer_norm = rf.LayerNorm(output_dim)

#         for name, param in self.named_parameters():
#             param.initial = rf.init.Normal(stddev=0.1) # mean already 0.0

#     def default_initial_state(
#         self, *, batch_dims: Sequence[Dim], use_batch_dims_for_pos: bool = False
#     ) -> rf.State:
#         """
#         all states are None. Need this to be (maybe) compatible
#         with some LM interfaces or RF
#         """
#         state = rf.State(
#             {
#                 k: None for k, v in self.layers.items()
#             }
#         )

#         return state

#     def select_state(self, state: rf.State, backrefs) -> rf.State:
#         state = tree.map_structure(
#             lambda s: rf.gather(s, indices=backrefs), state
#         )
#         return state

#     def __call__(
#         self,
#         input: rf.Tensor,
#         spatial_dim: Optional[Dim] = None,
#         # lengths: torch.Tensor,
#         state: Optional[rf.State] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
#         r"""Forward pass.

#         All states are None. Need states to be (maybe) compatible
#         with some LM interfaces or RF

#         B: batch size;
#         U: maximum sequence length in batch;
#         D: feature dimension of each input sequence element.

#         Args:
#             input (torch.Tensor): target sequences, with shape `(B, U)` and each element
#                 mapping to a target symbol, i.e. in range `[0, num_symbols)`.
#             lengths (torch.Tensor): with shape `(B,)` and i-th element representing
#                 number of valid frames for i-th batch element in ``input``.
#             state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
#                 representing internal state generated in preceding invocation
#                 of ``forward``. (Default: ``None``)

#         Returns:
#             (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
#                 torch.Tensor
#                     output encoding sequences, with shape `(B, U, output_dim)`
#                 torch.Tensor
#                     output lengths, with shape `(B,)` and i-th element representing
#                     number of valid elements for i-th batch element in output encoding sequences.
#                 List[List[torch.Tensor]]
#                     output states; list of lists of tensors
#                     representing internal state generated in current invocation of ``forward``.
#         """
#         embedding_out = self.embedding(input)
#         embedding_out = rf.dropout(
#             embedding_out,
#             drop_prob=self.embedding_dropout,
#             axis=embedding_out.feature_dim,
#         )
#         # input_layer_norm_out = self.input_layer_norm(embedding_out)

#         # lstm_out = input_layer_norm_out
#         ff_out = embedding_out

#         if state is None:
#             state = self.default_initial_state(batch_dims=input.dims[:-1])

#         new_state = rf.State()

#         for layer_name, layer in self.layers.items():
#             layer: rf.Linear  # or similar
#             # if layer_name in ["0"]: # "0"
#             #     breakpoint()
#             ff_out = layer(ff_out)
#             new_state[layer_name] = None
#             # if collected_outputs is not None:
#             #     collected_outputs[layer_name] = decoded
#         if self.use_bottleneck:
#             bottleneck_out = self.bottleneck(ff_out)
#             final_linear_out = self.final_linear(bottleneck_out)
#         else:
#             final_linear_out = self.final_linear(ff_out)
#         # output_layer_norm_out = self.output_layer_norm(linear_out)
#         return {
#             "output": final_linear_out,
#             "state": new_state,
#         }



# def get_model(*, epoch: int, **_kwargs_unused):
#     from returnn.config import get_global_config

#     config = get_global_config()
#     lm_cfg = config.typed_value("lm_cfg", {})
#     lm_cls = lm_cfg.pop("class")
#     assert lm_cls == "BigramLMRF", "Only Bigram LM are supported"
#     extern_data_dict = config.typed_value("extern_data")
#     data = Tensor(name="data", **extern_data_dict["data"])
#     return BigramLMRF(
#         label_target_size=data.sparse_dim,
#         output_dim=data.sparse_dim,
#         **lm_cfg,
#     )

# from returnn.tensor import batch_dim
# from i6_experiments.users.phan.utils.masking import get_seq_mask
# def train_step(*, model: BigramLMRF, extern_data: TensorDict, **_kwargs_unused):
#     targets = extern_data["data"]
#     delayed = extern_data["delayed"]
#     spatial_dim = delayed.get_time_dim_tag()
#     targets_len_rf = spatial_dim.dyn_size_ext
#     targets_len = targets_len_rf.raw_tensor
#     out = model(delayed)
#     logits: Tensor = out["output"]
#     logits_raw = logits.raw_tensor # (B, T, V)
#     targets_raw = targets.raw_tensor.long() # (B, T)
#     ce = torch.nn.functional.cross_entropy(
#         input = logits_raw.transpose(1, 2),
#         target = targets_raw,
#         reduction="none",
#     )
#     seq_mask = get_seq_mask(seq_lens=targets_len, max_seq_len=targets_len.max(), device=logits_raw.device)
#     loss = (ce*seq_mask).sum()
#     ppl = torch.exp(loss/targets_len.sum())
#     rf.get_run_ctx().mark_as_loss(
#         name="log_ppl", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
#     )
#     rf.get_run_ctx().mark_as_loss(
#         name="ppl", loss=ppl, as_error=True,
#     )

#     # doesn't work wtf ?????
#     # ce = rf.loss.cross_entropy(
#     #     estimated=logits,
#     #     target=targets,
#     #     axis=targets.sparse_dim,
#     #     estimated_type="logits",
#     # )
#     # rf.get_run_ctx().mark_as_loss(
#     #     name="log_ppl",
#     #     loss=ce,
#     #     custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim),
#     # )
#     # log_ppl = rf.reduce_mean(ce, axis=spatial_dim) #(B,)
#     # ppl = rf.exp(log_ppl) # (B, )
#     # ppl = rf.reduce_mean(ppl, axis=batch_dim) # scalar
#     # rf.get_run_ctx().mark_as_loss(
#     #     name="ppl",
#     #     loss=ppl,
#     #     as_error=True,
#     # )

# # intended for Albert setup
# # for transcription setup use get_model and train_step
# def from_scratch_training(
#     *,
#     model: Model,
#     data: rf.Tensor,
#     data_spatial_dim: Dim,
#     targets: rf.Tensor,
#     targets_spatial_dim: Dim,
# ):
#     """Function is run within RETURNN."""
#     from returnn.config import get_global_config

#     # import for training only, will fail on CPU servers
#     # from i6_native_ops import warp_rnnt

#     config = get_global_config()  # noqa
#     input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
#         targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
#     )
#     targets_w_eos, _ = rf.pad(
#         targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx,
#         out_dims=[targets_w_eos_spatial_dim]
#     )
#     out = model(input_labels)
#     estimated_type = "logits" if not model.log_prob_output else "log-probs"
#     ce = rf.cross_entropy(out, targets_w_eos, targets_w_eos_spatial_dim, estimated_type)
#     ce.mark_as_loss(
#         "ce",
#         custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
#         use_normalized_loss=True,
#     )
#     # calculate PPL



# from_scratch_training: TrainDef[Model]
# from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"




#-----------------------------------------
import torch
from typing import Optional, Union, List
import os
import shutil
import sys
import tempfile
import subprocess as sp


from i6_core.lm.kenlm import CompileKenLMJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.util import uopen
from i6_core.lib.lm import Lm
from i6_core.lm.srilm import ComputeNgramLmPerplexityJob
import i6_core.util as util

from sisyphus import Job, Task, tk, gs

from i6_experiments.users.mueller.datasets.librispeech import get_librispeech_lm_combined_txt
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH

def get_count_based_n_gram(vocab: Bpe, N_order: int) -> list[tk.Path]:
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    
    subword_nmt = get_returnn_subword_nmt()
    
    lm_data = get_librispeech_lm_combined_txt()
    
    bpe_text = ApplyBPEToTextJob(
        text_file=lm_data,
        bpe_codes=vocab.codes,
        bpe_vocab=tk.Path(vocab.vocab.get_path()[:-5] + "dummy_count.vocab"),
        subword_nmt_repo=subword_nmt,
        gzip_output=True,
        mini_task=False,
    ).out_bpe_text
    
    lm_arpa = KenLMplzJob(
        text=[bpe_text],
        order=N_order,
        interpolate_unigrams=False,
        use_discount_fallback=True,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        pruning=None,
        vocabulary=None
    ).out_lm
    
    ppl_job = ComputeNgramLmPerplexityJob(
        ngram_order=N_order,
        lm = lm_arpa,
        eval_data=bpe_text,
        ngram_exe=SRILM_PATH.join_right("ngram"),
        mem_rqmt=4,
        time_rqmt=1,
    )
    
    tk.register_output(f"datasets/LibriSpeech/lm/count_based_{N_order}-gram", ppl_job.out_ppl_score)
    
    res = []
    for N in range(2, N_order+1):
        conversion_job = ConvertARPAtoTensor(
            lm=lm_arpa,
            bpe_vocab=vocab.vocab,
            N_order=N,
        )
        
        conversion_job.add_alias(f"datasets/LibriSpeech/lm/count_based_{N}-gram")
        
        res.append(conversion_job.out_lm_tensor)
    
    return res


class ConvertARPAtoTensor(Job):
    def __init__(
        self,
        lm: tk.Path,
        bpe_vocab: tk.Path,
        N_order: int,
    ):
        self.lm = lm
        self.bpe_vocab = bpe_vocab
        self.N_order = N_order
        self.out_lm_tensor = self.output_path("lm.pt")

        # self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lm_loader = Lm(self.lm)
        n_grams = list(lm_loader.get_ngrams(self.N_order))
        
        vocab = eval(uopen(self.bpe_vocab, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(vocab, dict), "Has to be a dict containing the vocab!"
        
        # Read out the words and probabilities and turn into indexes of vocab
        n_grams = [list(map(lambda x: vocab[x], words.split(" "))) + [probs[0]] for words, probs in n_grams]
        n_grams = list(map(list, zip(*n_grams)))
        
        assert len(n_grams) - 1 == self.N_order, f"The conversion into a list failed ({len(n_grams) - 1} != {self.N_order})!"
        
        vocab_n = len(vocab) - 1 # we combine eos and bos
        tensor = torch.full((vocab_n,)*self.N_order, float("-inf"), dtype=torch.float32)
        # Set the probabilites by using N indexes
        tensor[n_grams[:-1]] = torch.tensor(n_grams[-1], dtype=torch.float32)
        # The probs are in logs base 10
        tensor = torch.pow(10, tensor)
        
        atol = 0.005
        if self.N_order == 2:
            s = tensor.sum(1)
            assert s[0].allclose(torch.tensor(1.0), atol=atol), f"The next word probabilities for <s> do not sum to 1! {s[0]}"
            assert s[1].allclose(torch.tensor(0.0)), f"Prob of <unk> should be 0! (1) {s[1]}"
            assert s[2:].allclose(torch.tensor(1.0), atol=atol), f"The next word probabilities do not sum to 1! {s[2:]}"
        else:
            assert (tensor.sum(-1) < 1.0).all(), f"The next word probabilities are not smaller than 1! {tensor.sum(-1)}"
        assert tensor.sum(tuple(range(0, self.N_order-1)))[1].allclose(torch.tensor(0.0)), f"Prob of <unk> should be 0! (2) {tensor.sum(tuple(range(0, self.N_order-1)))[1]}"
        
        tensor = tensor.log()
        
        with uopen(self.out_lm_tensor, "wb") as f:
            torch.save(tensor, f)
            
            
class ApplyBPEToTextJob(Job):
    """
    Apply BPE codes on a text file
    """

    __sis_hash_exclude__ = {"gzip_output": False}

    def __init__(
        self,
        text_file: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: Optional[tk.Path] = None,
        subword_nmt_repo: Optional[tk.Path] = None,
        gzip_output: bool = False,
        mini_task=True,
    ):
        """
        :param text_file: words text file to convert to bpe
        :param bpe_codes: bpe codes file, e.g. ReturnnTrainBpeJob.out_bpe_codes
        :param bpe_vocab: if provided, then merge operations that produce OOV are reverted,
            use e.g. ReturnnTrainBpeJob.out_bpe_dummy_count_vocab
        :param subword_nmt_repo: subword nmt repository path. see also `CloneGitRepositoryJob`
        :param gzip_output: use gzip on the output text
        :param mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """
        self.text_file = text_file
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)
        self.gzip_output = gzip_output

        self.out_bpe_text = self.output_path("words_to_bpe.txt.gz" if gzip_output else "words_to_bpe.txt")

        self.mini_task = mini_task
        self.rqmt = {"cpu": 2, "mem": 4, "time": 12}

    def tasks(self):
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            input_file = self.text_file.get_path()
            tmp_infile = os.path.join(tmp, "in_text.txt")
            tmp_outfile = os.path.join(tmp, "out_text.txt")
            with util.uopen(tmp_infile, "wt") as out:
                sp.call(["zcat", "-f", input_file], stdout=out)
            cmd = [
                sys.executable,
                os.path.join(self.subword_nmt_repo.get_path(), "apply_bpe.py"),
                "--input",
                tmp_infile,
                "--codes",
                self.bpe_codes.get_path(),
                "--output",
                tmp_outfile,
            ]

            if self.bpe_vocab:
                cmd += ["--vocabulary", self.bpe_vocab.get_path()]
                
            util.create_executable("apply_bpe.sh", cmd)
            sp.run(cmd, check=True)

            if self.gzip_output:
                with util.uopen(tmp_outfile, "rt") as fin, util.uopen(self.out_bpe_text, "wb") as fout:
                    sp.call(["gzip"], stdin=fin, stdout=fout)
            else:
                shutil.copy(tmp_outfile, self.out_bpe_text.get_path())

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)
    
class KenLMplzJob(Job):
    """
    Run the lmplz command of the KenLM toolkit to create a gzip compressed ARPA-LM file
    """

    def __init__(
        self,
        *,
        text: Union[tk.Path, List[tk.Path]],
        order: int,
        interpolate_unigrams: bool,
        use_discount_fallback: bool,
        pruning: Optional[List[int]],
        vocabulary: Optional[tk.Path],
        kenlm_binary_folder: tk.Path,
        mem: float = 4.0,
        time: float = 1.0,
    ):
        """

        :param text: training text data
        :param order: "N"-order of the "N"-gram LM
        :param interpolate_unigrams: Set True for KenLM default, and False for SRILM-compatibility.
            Having this as False will increase the share of the unknown probability
        :param pruning: absolute pruning threshold for each order,
            e.g. to remove 3-gram and 4-gram singletons in a 4th order model use [0, 0, 1, 1]
        :param vocabulary: a "single word per line" file to determine valid words,
            everything else will be treated as unknown
        :param kenlm_binary_folder: output of the CompileKenLMJob, or a direct link to the build
            dir of the KenLM repo
        :param mem: memory rqmt, needs adjustment for large training corpora
        :param time: time rqmt, might adjustment for very large training corpora and slow machines
        """
        self.text = text
        self.order = order
        self.interpolate_unigrams = interpolate_unigrams
        self.pruning = pruning
        self.vocabulary = vocabulary
        self.kenlm_binary_folder = kenlm_binary_folder
        self.use_discount_fallback = use_discount_fallback

        self.out_lm = self.output_path("lm.gz")

        self.rqmt = {"cpu": 1, "mem": mem, "time": time}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            lmplz_command = [
                os.path.join(self.kenlm_binary_folder.get_path(), "lmplz"),
                "-o",
                str(self.order),
                "--interpolate_unigrams",
                str(self.interpolate_unigrams),
                "-S",
                "%dG" % int(self.rqmt["mem"]),
                "-T",
                tmp,
            ]
            if self.use_discount_fallback:
                lmplz_command += ["--discount_fallback"]
            if self.pruning is not None:
                lmplz_command += ["--prune"] + [str(p) for p in self.pruning]
            if self.vocabulary is not None:
                lmplz_command += ["--limit_vocab_file", self.vocabulary.get_path()]

            zcat_command = ["zcat", "-f"] + [text.get_path() for text in self.text]
            with uopen(self.out_lm, "wb") as lm_file:
                p1 = sp.Popen(zcat_command, stdout=sp.PIPE)
                p2 = sp.Popen(lmplz_command, stdin=p1.stdout, stdout=sp.PIPE)
                sp.check_call("gzip", stdin=p2.stdout, stdout=lm_file)
                if p2.returncode:
                    raise sp.CalledProcessError(p2.returncode, cmd=lmplz_command)

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mem"]
        del parsed_args["time"]
        return super().hash(parsed_args)