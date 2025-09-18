__all__ = ["train_sentence_piece_model"]

from functools import cache
import random
from typing import Optional, Sequence, Tuple

from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from sisyphus import Job, Path, Task, tk

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.label.sentencepiece.train import SentencePieceType, TrainSentencePieceJob
from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from i6_core.util import uopen


@cache
def train_sentence_piece_model(
    bliss_corpus: tk.Path,
    dim: int,
    ty: SentencePieceType,
    alias_prefix: Optional[str] = None,
    user_defined_symbols: Optional[Sequence[str]] = None,
    limit_to_num_sentences: Optional[int] = None,
    spm_normalization: Optional[str] = None,
) -> Tuple[SentencePieceModel, tk.Path]:
    text_lines = CorpusToTxtJob(bliss_corpus, gzip=True).out_txt
    spm_opts = {
        "split_digits": True,
        "unk_id": 2,  # default is 0
        "bos_id": 1,  # default is 1
        "eos_id": 0,  # default is 2
    }
    if limit_to_num_sentences is not None:
        spm_opts.update(
            {
                "input_sentence_size": limit_to_num_sentences,
                "shuffle_input_sentence": True,
            }
        )
    if spm_normalization is not None:
        spm_opts["normalization_rule_name"] = spm_normalization
    if user_defined_symbols is not None:
        dim += len(user_defined_symbols)
        spm_opts["user_defined_symbols"] = ",".join(user_defined_symbols)
    train_job = TrainSentencePieceJob(
        training_text=text_lines, vocab_size=dim, model_type=ty, additional_options=spm_opts
    )
    base_name = f"{ty.value}-{dim}"
    if user_defined_symbols is not None:
        base_name += "-" + ",".join(user_defined_symbols)
    if alias_prefix is not None:
        train_job.add_alias(f"{alias_prefix}/spm/{base_name}")
        tk.register_output(f"{alias_prefix}/spm/{base_name}.model", train_job.out_model)
    vocab_job = ExtractSentencePieceVocabJob(train_job.out_model)
    if alias_prefix is not None:
        vocab_job.add_alias(f"{alias_prefix}/vocab/{base_name}")
        tk.register_output(f"{alias_prefix}/vocab/{base_name}.vocab", vocab_job.out_vocab)
    spm = SentencePieceModel(
        dim=dim,
        model_file=train_job.out_model,
        unknown_label="<unk>",
        bos_idx=1,
        eos_idx=0,
    )
    return spm, vocab_job.out_vocab


class ShuffleAndTakeLinesJob(Job):
    def __init__(self, file: Path, num_lines: int, *, gzip: bool = True, seed: int = 42):
        self.file = file
        assert num_lines >= 0
        self.num_lines = num_lines
        self.seed = seed

        self.out = self.output_path("out.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.file, "rt") as in_f:
            lines = in_f.readlines()
        random.seed(self.seed)
        random.shuffle(lines)
        with uopen(self.out, "wt") as out_f:
            out_f.writelines(lines[: self.num_lines])
