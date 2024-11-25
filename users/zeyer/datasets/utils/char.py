"""
Character vocab

Also see: :mod:`bytes` (utf8 bytes), or consider small BPE/SPM vocab.
"""


from __future__ import annotations
from typing import Optional, Sequence
from sisyphus import tk, Job, Task
from sisyphus.job_path import NoBackup
from returnn_common.datasets_old_2022_10.interface import VocabConfigStatic
from i6_core.util import uopen


def get_char_vocab(
    txt: tk.Path, *, num_classes: int, unknown_label: Optional[str] = None, extra_labels: Sequence[str] = (), **other
) -> VocabConfigStatic:
    """
    get char vocab

    :param txt: line-based text file
    :param num_classes: number of classes. Unfortunately you must know this in advance currently.
        So you could run the pipeline with some dummy value, and then it will crash,
        but then you will see the correct value,
        and fix this.
        Later, we would make this a Sisyphus Variable, but currently our pipeline does not really allow this.
    :param unknown_label: None (default) means there is no unknown label
    :param extra_labels: additional labels to add to the beginning, e.g. BOS/EOS.
        (currently I tend to use "\n" as EOS).
    :param other: passed to :class:`VocabConfigStatic` opts
    """
    job = GetCharacterSetJob(txt, extra_labels=extra_labels)

    return VocabConfigStatic(
        num_classes=num_classes,
        opts={"class": "CharacterTargets", "vocab_file": job.out_vocab_file, "unknown_label": unknown_label, **other},
    )


class GetCharacterSetJob(Job):
    def __init__(
        self, txt: tk.Path, *, include_newline: bool = False, extra_labels: Sequence[str] = (), out_format: str = "py"
    ):
        super().__init__()
        self.txt = txt
        self.out_format = out_format

        self.out_vocab_file = self.output_path({"txt": "chars.txt", "py": "chars.vocab.py"}[out_format])
        self.out_num_labels = self.output_var("num_labels.txt", backup=NoBackup)

        self.include_newline = include_newline
        self.extra_labels = extra_labels

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 24, "gpu": 0})

    def run(self):
        chars = set()

        with uopen(self.txt.get_path(), "rt") as txt_file:
            for line in txt_file:
                line: str
                if not self.include_newline:
                    line = line.rstrip("\n")
                for c in line.strip():
                    chars.add(c)

        chars_list = list(sorted(chars))
        if self.extra_labels:
            for c in self.extra_labels:
                assert c not in chars  # we expect that these are not in the input
            assert len(set(self.extra_labels)) == len(self.extra_labels)  # assume unique
            # add them to beginning
            chars_list = list(self.extra_labels) + chars_list

        with uopen(self.out_vocab_file.get_path(), "wt") as out_file:
            if self.out_format == "txt":
                assert "\n" not in chars_list
                for c in chars_list:
                    out_file.write(c + "\n")
            elif self.out_format == "py":
                out_file.write("{\n")
                for i, c in enumerate(chars_list):
                    out_file.write(f"{c!r}: {i},\n")
                out_file.write("}\n")
            else:
                raise ValueError(f"invalid out_format: {self.out_format}")
        self.out_num_labels.set(len(chars_list))
