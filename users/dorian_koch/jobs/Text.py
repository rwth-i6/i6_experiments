import collections
from typing import Dict, List, Tuple
from sisyphus import Job, Task, tk
from i6_core.util import uopen


class WordFrequencyJob(Job):
    def __init__(
        self,
        *,
        text: tk.Path,
    ):
        self.text = text  # file with text on each line, i.e. LmDataset

        self.out_freq_dict = self.output_path("word_frequency.py.gz")
        self.out_lines = self.output_var("num_lines")
        self.out_num_unique_words = self.output_var("num_unique_words")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        word_freq = {}
        num_lines = 0
        with uopen(self.text, "rt") as f:
            word_counter = collections.Counter()
            for line in f:
                word_counter.update(line.split())
                num_lines += 1
            word_freq = dict(word_counter)
        print(f"Total lines processed: {num_lines}")

        with uopen(self.out_freq_dict, "wt") as out:
            out.write(repr(word_freq))
            out.write("\n")
        self.out_lines.set(num_lines)
        self.out_num_unique_words.set(len(word_freq))
        print(f"Done writing")


class CategorizeWordsByFreq(Job):
    def __init__(self, word_freq: tk.Path, percentiles: List[Tuple[str, float]]):
        self.word_freq = word_freq
        self.out_category_to_word = self.output_path("category_to_words.py.gz")
        self.out_word_to_category = self.output_path("word_to_category.py.gz")
        self.percentiles = percentiles
        # sort by percentiles
        self.percentiles.sort(key=lambda x: x[1])
        assert self.percentiles[-1][1] == 100.0, "Last percentile must be 100.0"

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 2
        return super().hash(d)

    def run(self):
        with uopen(self.word_freq, "rt") as f:
            word_freq = eval(f.read())
        print(f"Loaded {len(word_freq)} words")
        categorized = {key: [] for key, _ in self.percentiles}
        word_to_cat = {}
        # sort word freq desc
        word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

        total_freq = sum(word_freq.values())
        print(f"Total frequency:\t\t{total_freq}")
        cur_freq = 0
        cur_perc_index = 0
        for word, freq in word_freq.items():
            cur_freq += freq
            percentile_cat_name = self.percentiles[cur_perc_index][0]
            categorized[percentile_cat_name].append(word)
            word_to_cat[word] = percentile_cat_name

            if cur_perc_index < len(self.percentiles) - 1:
                if cur_freq / total_freq >= self.percentiles[cur_perc_index][1] / 100.0:
                    cur_perc_index += 1
                    print(
                        f"Threshold of {percentile_cat_name} at\t{cur_freq} ({cur_freq / total_freq * 100:.2f}) frequency overstepped (last frequency: {freq} for word '{word}')"
                    )
                    print(f"{len(categorized[percentile_cat_name])} words in category {percentile_cat_name}")

        print(
            f"{len(categorized[self.percentiles[cur_perc_index][0]])} words in category {self.percentiles[cur_perc_index][0]}"
        )
        print("Writing...")
        with uopen(self.out_category_to_word, "wt") as out:
            out.write(repr(categorized))
            out.write("\n")
        print(f"Categorized words written to {self.out_category_to_word}")
        with uopen(self.out_word_to_category, "wt") as out:
            out.write(repr(word_to_cat))
            out.write("\n")


class CategorizeWordsByPOS(Job):
    def __init__(self, word_freq: tk.Path):
        self.word_freq = word_freq
        self.out_category_to_word = self.output_path("category_to_words.py.gz")
        self.out_word_to_category = self.output_path("word_to_category.py.gz")
        self.out_category_counts = self.output_var("category_counts")
        import spacy

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    # @classmethod
    # def hash(cls, parsed_args):
    #    d = dict(**parsed_args)
    #    d["__version"] = 2
    #    return super().hash(d)

    def run(self):
        import spacy

        # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
        print("Loading spaCy model for POS tagging...")
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model for POS tagging")

        with uopen(self.word_freq, "rt") as f:
            word_freq: Dict[str, int] = eval(f.read())
        print(f"Loaded {len(word_freq)} words")
        categorized: Dict[str, List[str]] = {}
        word_to_cat: Dict[str, str] = {}
        # sort word freq desc

        total_num = len(word_freq)
        cur_num = 0
        for word in word_freq.keys():
            doc = nlp(word)
            if len(doc) > 1 and "'" not in word:
                print(f"Word '{word}' has more than one token: {doc}")
            if len(doc) == 0:
                print(f"Word '{word}' could not be processed by spaCy, skipping.")
                raise ValueError(f"Word '{word}' could not be processed by spaCy, skipping.")
            pos = doc[0].pos_
            categorized.setdefault(pos, []).append(word)
            word_to_cat[word] = pos
            cur_num += 1
            if cur_num % 50000 == 0:
                print(f"Processed {cur_num}/{total_num} words ({cur_num / total_num * 100:.2f}%)")
        print("Writing...")
        with uopen(self.out_category_to_word, "wt") as out:
            out.write(repr(categorized))
            out.write("\n")
        print(f"Categorized words written to {self.out_category_to_word}")
        with uopen(self.out_word_to_category, "wt") as out:
            out.write(repr(word_to_cat))
            out.write("\n")

        # count categories
        category_counts = {cat: len(words) for cat, words in categorized.items()}
        self.out_category_counts.set(category_counts)
        print(f"Category counts: {category_counts}")


class ExtractWordStatsFromSclite(Job):
    def __init__(self, report_dir: tk.Path):
        self.report_dir = report_dir
        self.out_word_stats = self.output_path("word_stats.py.gz")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        # get word frequency from sclite.pra files

        print("Reading sclite.pra")
        word_counter = collections.Counter()
        error_counter = collections.Counter()  # this is just for debugging and checking
        cur_seq_id = None
        with open(f"{self.report_dir}/sclite.pra", errors="ignore") as f:
            for line in f:
                if line.startswith("id:"):
                    # id: (dev-clean/422-122949-0034/422-122949-0034-000)
                    cur_seq_id = line.split()[1][1:-1]
                    assert len(cur_seq_id) > 0
                elif line.startswith("REF:"):
                    assert cur_seq_id is not None
                    words = line.strip().split()
                    word_counter.update(words)
                    error_words = [w for w in words if all(c.isupper() or c == "'" for c in w)]
                    error_counter.update(error_words)

        print(f"Counted all the words")

        # get substitutions and deletions from sclite.dtl
        with open(f"{self.report_dir}/sclite.dtl", errors="ignore") as f:
            part = None
            for line in f:  # TODO implement this
                if part is None:
                    if line.startswith("INSERTIONS"):
                        part = "insertions"
                elif part == "insertions":
                    if line.startswith("DELETIONS"):
                        part = "deletions"
                    elif len(line.strip()) == 0:
                        continue
                    else:
                        pass
                elif part == "deletions":
                    if line.startswith("SUBSTITUTIONS"):
                        part = "substitutions"
                    elif len(line.strip()) == 0:
                        continue
                    else:
                        pass
                elif part == "substitutions":
                    if line.startswith("FALSELY RECOGNIZED"):
                        part = "falsely_recognized"
                    elif len(line.strip()) == 0:
                        continue
                elif part == "falsely_recognized":
                    if len(line.strip()) == 0:
                        part = None
                    else:
                        pass

        # with uopen(self.out_word_stats, "wt") as out:
        #    out.write(repr(word_stats))
        #    out.write("\n")
        print(f"Word stats written to {self.out_word_stats}")
