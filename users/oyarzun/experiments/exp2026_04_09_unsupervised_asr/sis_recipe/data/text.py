from typing import Optional

from sisyphus import Path, tk

from i6_experiments.users.schmitt.datasets.utils.phonemize import (
    PhonemizeTextDataJob,
    DumpPhonemeIndicesToHdfJob,
)

class LocalPhonemizeTextDataJob(PhonemizeTextDataJob):
    @classmethod
    def hash(cls, parsed_args):
        parsed_args["dummy_version"] = "v15"
        return super().hash(parsed_args)

    @classmethod
    def phonemize_with_sil(
        cls,
        text_file,
        out_text_file,
        sil_prob=0.0,
        surround=False,
        sil="<SIL>",
        seq_tags_file=None,
        lexicon=None,
        collapse_repeats=False,
        **kwargs,
    ):
        import numpy as np
        import re

        if seq_tags_file is not None:
            with open(seq_tags_file, "r", encoding="utf-8") as f:
                seq_tags = list(line.strip() for line in f)
        else:
            seq_tags = None

        wrd_to_phn = {}

        with open(lexicon, "r", encoding="utf-8") as lf:
            for line in lf:
                items = line.rstrip().split()
                if len(items) <= 1:
                    continue
                w_lower = items[0].lower()
                # Keep first pronunciation for words with multiple entries
                if w_lower in wrd_to_phn:
                    continue
                # Strip stress numbers (e.g. AH0 -> AH, EY1 -> EY) to match phoneme vocabulary
                phns = [re.sub(r"\d+", "", p) for p in items[1:]]
                wrd_to_phn[w_lower] = phns

        new_seq_tag_file = open("seq-tags-after-phonemize.txt", "w", encoding="utf-8") if seq_tags is not None else None
        out_tf = open(out_text_file, "w", encoding="utf-8")

        with open(text_file, "r", encoding="utf-8") as tf:
            for i, line in enumerate(tf):
                words = [w.lower() for w in line.strip().split()]

                if not all(w in wrd_to_phn for w in words):
                    continue

                phones = []
                if surround:
                    phones.append(sil)

                sample_sil_probs = None
                if sil_prob > 0 and len(words) > 1:
                    sample_sil_probs = np.random.random(len(words) - 1)

                for j, w in enumerate(words):
                    phones.extend(wrd_to_phn[w])
                    if sample_sil_probs is not None and j < len(sample_sil_probs) and sample_sil_probs[j] < sil_prob:
                        phones.append(sil)

                if surround:
                    phones.append(sil)

                if collapse_repeats:
                    phones = [p for i, p in enumerate(phones) if i == 0 or p != phones[i - 1]]

                out_tf.write(" ".join(phones) + "\n")

                if new_seq_tag_file is not None and i < len(seq_tags):
                    new_seq_tag_file.write(f"{seq_tags[i]}\n")

        if new_seq_tag_file is not None:
            new_seq_tag_file.close()
        out_tf.close()

    @staticmethod
    def normalize_and_filter_text(text_file, out_text_file, lang, lid_threshold, fasttext_model, seq_tags_file, apply_lid_filter=True, **kwargs):
        import sys
        class DummyFastText:
            @staticmethod
            def load_model(*args, **kwargs):
                return None
        sys.modules["fasttext"] = DummyFastText()
        
        PhonemizeTextDataJob.normalize_and_filter_text(
            text_file, out_text_file, lang, lid_threshold, fasttext_model, seq_tags_file, apply_lid_filter=apply_lid_filter, **kwargs
        )

    def run(self):
        dummy_hash_updater = 1
        import os
        import subprocess as sp
        import shutil
        from unittest.mock import patch
        
        # If words.txt, lm.upper.lid.txt, and lexicon already exist from a previous attempt in work/
        # we can skip re-running the expensive normalization and fairseq preprocessing
        if (
            self.lexicon_file is not None
            and os.path.isfile("words.txt")
            and os.path.isfile("lm.upper.lid.txt")
            and os.path.isfile(self.out_lexicon_file.get_path())
        ):
            seq_tag_file = (
                os.path.join(os.getcwd(), "seq-tags-after-norm-and-filter.txt")
                if self.seq_tag_file is not None and os.path.isfile("seq-tags-after-norm-and-filter.txt")
                else None
            )
            self.phonemize_with_sil(
                text_file="lm.upper.lid.txt",
                out_text_file="lm.phones.filtered.txt",
                sil_prob=self.sil_prob,
                surround=self.surround_w_sil,
                seq_tags_file=seq_tag_file,
                lexicon=self.out_lexicon_file.get_path(),
                collapse_repeats=self.collapse_repeats,
            )
            shutil.move("lm.phones.filtered.txt", self.out_phoneme_text.get_path())
            if self.seq_tag_file is not None and os.path.isfile("seq-tags-after-phonemize.txt"):
                shutil.move("seq-tags-after-phonemize.txt", self.out_seq_tags.get_path())
            self._ensure_phoneme_vocab()
            return

        base_exp_dir = (
            "/u/benjamin.oyarzun/experiments/2026_05_07_first_experiments"
            if os.path.exists("/u/benjamin.oyarzun/experiments/2026_05_07_first_experiments")
            else "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments"
        )
        os.environ["PYTHONPATH"] = f"{base_exp_dir}/container_libs:" + os.environ.get("PYTHONPATH", "")
        os.environ["NLTK_DATA"] = f"{base_exp_dir}/container_libs/nltk_data"

        script_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../phonemize_text.sh"
        ))
        
        original_run = sp.run
        
        def mock_run(*args, **kwargs):
            if isinstance(args[0], list) and args[0][0] == "zsh" and "phonemize_text.sh" in args[0][1]:
                args[0][1] = script_path
            return original_run(*args, **kwargs)
            
        with patch('subprocess.run', mock_run):
            super().run()
        self._ensure_phoneme_vocab()

    def _ensure_phoneme_vocab(self):
        import os
        import shutil
        out_vocab_path = self.out_phoneme_vocab.get_path()
        if os.path.exists(out_vocab_path):
            return

        candidate_paths = [
            "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/i6_experiments/users/oyarzun/experiments/exp2026_04_09_unsupervised_asr/sis_recipe/data/text/LocalPhonemizeTextDataJob.8jDKfW1lmnJg/output/phoneme_vocab.txt",
            "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/work/i6_experiments/users/oyarzun/experiments/exp2026_04_09_unsupervised_asr/sis_recipe/data/text/LocalPhonemizeTextDataJob.8jDKfW1lmnJg/output/phoneme_vocab.txt",
        ]
        for cp in candidate_paths:
            if os.path.isfile(cp):
                shutil.copy(cp, out_vocab_path)
                return

        lexicon_path = (
            self.out_lexicon_file.get_path()
            if os.path.isfile(self.out_lexicon_file.get_path())
            else (self.lexicon_file.get_path() if self.lexicon_file and os.path.isfile(self.lexicon_file.get_path()) else None)
        )
        if lexicon_path and os.path.isfile(lexicon_path):
            import re
            phones = set()
            with open(lexicon_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    items = line.rstrip().split()
                    if len(items) <= 1:
                        continue
                    for p in items[1:]:
                        phones.add(re.sub(r"\d+", "", p))
            inventory = sorted(phones)
            vocab = {p: i for i, p in enumerate(inventory)}
            vocab["<SIL>"] = len(vocab)
            with open(out_vocab_path, "w", encoding="utf-8") as f:
                f.write("{\n")
                for phon, i in vocab.items():
                    f.write(f'"{phon}": {i},\n')
                f.write("}\n")


class LocalDumpPhonemeIndicesToHdfJob(DumpPhonemeIndicesToHdfJob):
    @classmethod
    def hash(cls, parsed_args):
        parsed_args["dummy_version"] = "v4"
        return super().hash(parsed_args)

    def run(self, task_id):
        import gc
        import tempfile
        import random
        import ast
        import shutil
        import os
        import numpy as np
        from returnn.datasets.hdf import SimpleHDFWriter

        with open(self.text_file.get(), "r") as f:
            lines = f.readlines()

        with open(self.phoneme_vocab.get(), "r") as f:
            content = f.read().strip()
            if content.startswith("{"):
                vocab = ast.literal_eval(content)
            else:
                vocab = {line.strip().split()[0]: i for i, line in enumerate(content.splitlines()) if line.strip()}

        if self.seq_tag_file is not None:
            with open(self.seq_tag_file.get(), "r") as f:
                seq_tags = [line.strip() for line in f.readlines()]
                assert len(seq_tags) == len(lines)
        else:
            seq_tags = None
        pairs = list(zip(lines, seq_tags)) if seq_tags is not None else [(line, None) for line in lines]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_hdf = os.path.join(tmp_dir, f"data_{task_id}.hdf")
            hdf_writer = SimpleHDFWriter(filename=tmp_hdf, dim=len(vocab), ndim=1)

            random.Random(42).shuffle(pairs)

            num_lines = len(pairs)
            pairs = [pairs[i] for i in range(num_lines) if (i % self.concurrent) == (task_id - 1)]
            if self.fixed_random_subset is not None:
                pairs = pairs[: self.fixed_random_subset]
            num_lines = len(lines)
            gc.collect()

            for i, (line, seq_tag) in enumerate(pairs):
                phonemes = line.strip().split()
                data_list = [vocab[p] for p in phonemes if p in vocab]
                if not data_list:
                    continue
                data = np.array([data_list])  # (1, T)

                seq_len = len(data_list)
                seq_lens = {0: np.array([seq_len])}
                batch_seq_sizes = np.expand_dims(seq_lens[0], 1)

                hdf_writer.insert_batch(
                    data,
                    seq_len=seq_lens,
                    seq_tag=[f"lm-data-{i}" if seq_tag is None else seq_tag],
                    extra={"seq_sizes": batch_seq_sizes},
                )

                if i % 10_000 == 0:
                    gc.collect()
                    print(f"Processed sequence {i}/{num_lines} ({i / num_lines * 100:.1f}%)")

            hdf_writer.close()
            out_dest = self.out_hdfs[task_id - 1].get_path()
            os.makedirs(os.path.dirname(out_dest), exist_ok=True)
            shutil.move(tmp_hdf, out_dest)


from ..default_tools import get_fairseq_root, get_lid_model, get_fasttext_python_exe


def get_phonemized_data(
    dataset_name: str,
    corpus_name: str,
    text_file: Path,
    dump_hdf_concurrent: int,
    fixed_random_subset: Optional[int] = None,
    lexicon_file: Optional[Path] = None,
    language: str = "en",
    sil_prob: float = 0.25,
    seq_tag_file: Optional[Path] = None,
    vocab_file: Optional[Path] = None,
    surround_w_sil: bool = True,
):
    prepare_text_job_training = LocalPhonemizeTextDataJob(
        text_file=text_file,
        fairseq_root=get_fairseq_root(),
        python_exe=get_fasttext_python_exe(),
        lid_path=get_lid_model(),
        language=language,
        sil_prob=sil_prob,
        lexicon_file=lexicon_file,
        seq_tag_file=seq_tag_file,
        min_phoneme_occurrence=1000,
        surround_w_sil=surround_w_sil,
    )
    text_file = prepare_text_job_training.out_phoneme_text
    tk.register_output(f"data/{corpus_name}/text/phonemized/{dataset_name}.txt", text_file)
    lexicon_file = prepare_text_job_training.out_lexicon_file
    vocab_file = prepare_text_job_training.out_phoneme_vocab if vocab_file is None else vocab_file
    seq_tags_after_phonemize = prepare_text_job_training.out_seq_tags

    dump_phoneme_indices_job = LocalDumpPhonemeIndicesToHdfJob(
        text_file=text_file,
        phoneme_vocab=vocab_file,
        concurrent=dump_hdf_concurrent,
        fixed_random_subset=fixed_random_subset,
        seq_tag_file=seq_tags_after_phonemize,
    )

    return (
        list(dump_phoneme_indices_job.out_hdfs.values()),
        vocab_file,
        lexicon_file,
        seq_tags_after_phonemize,
    )
