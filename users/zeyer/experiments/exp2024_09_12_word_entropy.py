"""
Word entropy calculation
"""

from sisyphus import Job, Task, Path, tk
from typing import Dict, Any, Optional, Sequence
import os
import sys


def py():
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2, LibrispeechOggZip
    from i6_experiments.users.zeyer.datasets.librispeech import LibrispeechOggZip, Bpe

    bpe1k_num_labels_with_blank = 1057  # incl blank
    bpe1k_blank_idx = 1056  # at the end
    bpe1k_vocab = Path(
        "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"
    )
    returnn_dataset = LibrispeechOggZip(
        vocab=Bpe(
            codes=Path(
                "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.codes"
            ),
            vocab=bpe1k_vocab,
            dim=1056,
        ),
        train_epoch_split=1,
    ).get_dataset("train")

    prefix = "exp2024_09_12_word_entropy/"
    name = "entropies"
    job = CalcWordEntropies(
        seq_list=Path(
            "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
        ),
        returnn_dataset=returnn_dataset,
        returnn_dataset_key="classes",
    )
    job.add_alias(prefix + name)
    tk.register_output(prefix + name + "/per_pos.json", job.out_entropies_per_pos)


class CalcWordEntropies(Job):
    """Calculate word entropies over dataset"""

    def __init__(
        self,
        *,
        seq_list: Optional[tk.Path] = None,
        returnn_dataset: Dict[str, Any],  # for BPE labels
        returnn_dataset_key: str = "classes",
        returnn_root: Optional[tk.Path] = None,
        pos_keys: Sequence[int] = (0, -1, 1, -2),
    ):
        self.seq_list = seq_list
        self.returnn_dataset = returnn_dataset
        self.returnn_dataset_key = returnn_dataset_key
        self.returnn_root = returnn_root
        self.pos_keys = pos_keys

        self.out_entropies_per_pos = self.output_path("out_entropies_per_pos.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        from typing import List, Tuple
        import numpy as np
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        from i6_experiments.users.schmitt.hdf import load_hdf_data
        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)

        sys.path.insert(0, returnn_root.get_path())

        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset
        from returnn.log import log

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 4
        log.init_by_config(config)

        import tree

        dataset_dict = self.returnn_dataset
        dataset_dict = tree.map_structure(lambda x: x.get_path() if isinstance(x, Path) else x, dataset_dict)
        print("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)
        labels = dataset.labels[self.returnn_dataset_key]

        seq_list_ = None
        if self.seq_list:
            seq_list = open(self.seq_list).read().splitlines()

            # We might want "train-other-960/1034-121119-0049/1034-121119-0049",
            # but it's actually "train-clean-100/1034-121119-0049/1034-121119-0049" in the RETURNN dataset.
            # Transform the seq tags for the RETURNN dataset.
            all_tags = set(dataset.get_all_tags())
            all_tags_wo_prefix = {}
            for tag in all_tags:
                tag_wo_prefix = tag.split("/", 2)[-1]
                assert tag_wo_prefix not in all_tags_wo_prefix
                all_tags_wo_prefix[tag_wo_prefix] = tag
            seq_list_ = []
            for seq_tag in seq_list:
                tag_wo_prefix = seq_tag.split("/", 2)[-1]
                if seq_tag in all_tags:
                    seq_list_.append(seq_tag)
                elif tag_wo_prefix in all_tags_wo_prefix:
                    seq_list_.append(all_tags_wo_prefix[tag_wo_prefix])
                else:
                    print(f"seq tag {seq_tag} not found in dataset")

        dataset.init_seq_order(epoch=1, seq_list=seq_list_)

        from collections import Counter

        word_counts = Counter()
        total_num_words = 0

        seq_idx = 0
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            seq = dataset.get_data(seq_idx, self.returnn_dataset_key)

            for word in seq:
                word_counts[word] += 1
                total_num_words += 1

            seq_idx += 1

        total_num_seqs = seq_idx
        word_probs = {word: count / total_num_words for word, count in word_counts.items()}
        word_entropies = {word: -np.log2(prob) for word, prob in word_probs.items()}
        print("Total num seqs:", total_num_seqs)
        print("Total num words:", total_num_words)
        print("Total num unique words:", len(word_counts))
        print("Top 10 words:", [(labels[w], c) for w, c in word_counts.most_common(10)])
        print(
            "Top 10 entropies:",
            [(labels[w], ent) for w, ent in sorted(word_entropies.items(), key=lambda x: x[1], reverse=True)[:10]],
        )

        # Now reiterate through the dataset to get some stats on the word entropies per position.
        dataset.finish_epoch()
        dataset.init_seq_order(epoch=1, seq_list=seq_list_)

        # Order is relevant, duplicate positions are not counted.
        total_entropies_per_pos = {k: 0.0 for k in self.pos_keys}
        total_entropies_per_pos_count = {k: 0 for k in total_entropies_per_pos}
        seq_idx = 0
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            seq = dataset.get_data(seq_idx, self.returnn_dataset_key)

            covered_abs_pos = set()
            for k in self.pos_keys:
                abs_pos = k
                if abs_pos < 0:
                    abs_pos = len(seq) + abs_pos
                if not 0 <= abs_pos < len(seq):
                    continue
                if abs_pos in covered_abs_pos:
                    continue
                covered_abs_pos.add(abs_pos)

                total_entropies_per_pos[k] += word_entropies[seq[abs_pos]]
                total_entropies_per_pos_count[k] += 1

            seq_idx += 1

        avg_entropies_per_pos = {k: v / total_entropies_per_pos_count[k] for k, v in total_entropies_per_pos.items()}

        with open(self.out_entropies_per_pos, "w") as f:
            import json

            json.dump(avg_entropies_per_pos, f)
