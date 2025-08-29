import collections
import random
from typing import Any, Dict, List, Tuple
from sisyphus import Job, Task, tk
from i6_core.util import uopen
import re
from returnn.datasets.util.vocabulary import Vocabulary


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


class AddWordFrequencies(Job):
    def __init__(self, word_freqs: List[tk.Path]):
        self.word_freqs = word_freqs
        self.out_freqs = self.output_path("word_freqs_combined.py.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        combined_freq = collections.Counter()
        for word_freq_file in self.word_freqs:
            with uopen(word_freq_file, "rt") as f:
                word_freq = eval(f.read())
                assert isinstance(word_freq, dict), f"Expected word_freq to be a dict, got {type(word_freq)}"
                print(f"Loaded {len(word_freq)} words from {word_freq_file}")
                combined_freq += collections.Counter(word_freq)
        print(f"Combined {len(self.word_freqs)} word frequency files into {len(combined_freq)} unique words")
        with uopen(self.out_freqs, "wt") as out:
            out.write(repr(dict(combined_freq)))
            out.write("\n")


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


categorize_words_by_pos_nlp = None


def init_spacy():
    import spacy

    """Ensure each process loads its own copy of the model."""
    global categorize_words_by_pos_nlp
    print("Loading spaCy model in worker...")
    categorize_words_by_pos_nlp = spacy.load("en_core_web_sm")


def process_word(word: str) -> Tuple[str, str]:
    """Process a single word and return (word, POS) or raise an error if it fails."""
    doc = categorize_words_by_pos_nlp(word)
    if len(doc) > 1 and "'" not in word:
        print(f"Word '{word}' has more than one token: {doc}")
    if len(doc) == 0:
        print(f"Word '{word}' could not be processed by spaCy, skipping.")
        raise ValueError(f"Word '{word}' could not be processed by spaCy, skipping.")
    pos = doc[0].pos_
    return word, pos


def fix_sclite_pra(f):
    it = iter(f)
    while True:
        line: str = next(it, None)
        if line is None:
            break
        while line.endswith("%.2f\n"):
            line = line[: -len("%.2f\n")]
            line += "* "
            line += next(it, "")

        assert "%.2f" not in line, f"Line '{line}' still contains '%.2f' after fixing"
        yield line


class CategorizeWordsByPOS(Job):
    def __init__(self, word_freq: tk.Path, model_name: str = "en_core_web_sm"):
        self.word_freq = word_freq
        self.out_category_to_word = self.output_path("category_to_words.py.gz")
        self.out_word_to_category = self.output_path("word_to_category.py.gz")
        self.out_category_counts = self.output_var("category_counts")
        self.model_name = model_name
        self.num_cpu = 16
        self.nlp = None

    def tasks(self):
        yield Task("run", rqmt={"cpu": self.num_cpu, "mem": self.num_cpu + 4, "time": 2})

    def run(self):
        from multiprocessing import Pool

        print("Import spacy")
        import spacy

        # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
        # or similar for other models
        print(f"Loading spaCy model '{self.model_name}' for POS tagging...")
        self.nlp = spacy.load(self.model_name)
        print("Loaded spaCy model for POS tagging", flush=True)

        with uopen(self.word_freq, "rt") as f:
            word_freq: Dict[str, int] = eval(f.read())
        print(f"Loaded {len(word_freq)} words", flush=True)

        words = list(word_freq.keys())
        total_num = len(word_freq)

        # Multiprocessing
        print("Processing words in parallel...")
        results = []
        with Pool(processes=self.num_cpu, initializer=init_spacy) as pool:
            # results = pool.map(process_word, words)
            for result in pool.imap_unordered(process_word, words):
                results.append(result)
                if len(results) % 10000 == 0:
                    print(
                        f"Processed {len(results)}/{total_num} words ({len(results) / total_num * 100:.2f}%)",
                        flush=True,
                    )
        print("Finished processing words", flush=True)

        categorized: Dict[str, List[str]] = {}
        word_to_cat: Dict[str, str] = {}
        for word, pos in results:
            categorized.setdefault(pos, []).append(word)
            word_to_cat[word] = pos
        # sort word freq desc

        #
        # cur_num = 0
        # for word in word_freq.keys():
        #     doc = nlp(word)
        #     if len(doc) > 1 and "'" not in word:
        #         print(f"Word '{word}' has more than one token: {doc}")
        #     if len(doc) == 0:
        #         print(f"Word '{word}' could not be processed by spaCy, skipping.")
        #         raise ValueError(f"Word '{word}' could not be processed by spaCy, skipping.")
        #     pos = doc[0].pos_
        #     categorized.setdefault(pos, []).append(word)
        #     word_to_cat[word] = pos
        #     cur_num += 1
        #     if cur_num % 10000 == 0:
        #         print(f"Processed {cur_num}/{total_num} words ({cur_num / total_num * 100:.2f}%)", flush=True)

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
        category_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))
        self.out_category_counts.set(category_counts)
        print(f"Category counts: {category_counts}")


class ExtractWordStatsFromSclite(Job):
    def __init__(self, report_dir: tk.Path):
        self.report_dir = report_dir
        self.out_word_stats = self.output_path("word_stats.py.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 6
        return super().hash(d)

    def run(self):
        # get word frequency from sclite.pra files

        print("Reading sclite.pra")
        word_counter = collections.Counter()
        error_counter = collections.Counter()  # this is just for debugging and checking
        cur_seq_id = None
        numbers = []
        total_ref = []
        with open(f"{self.report_dir}/sclite.pra", errors="ignore") as f:
            for line in fix_sclite_pra(f):
                if line.startswith("id:"):
                    # id: (dev-clean/422-122949-0034/422-122949-0034-000)
                    cur_seq_id = line.split()[1][1:-1]
                    assert len(cur_seq_id) > 0
                elif line.startswith("Scores:"):
                    assert cur_seq_id is not None
                    # Scores: (#C #S #D #I) 79 2 2 0
                    parts = line.split()
                    assert len(parts) == 9, line
                    assert len(numbers) == 0, f"{cur_seq_id}, {line}"
                    numbers = [int(p) for p in parts[-4:]]

                elif line.startswith("REF:") or line.startswith(">> REF:"):
                    assert cur_seq_id is not None
                    words = line[line.index(":") + 1 :].strip().split()
                    # print(f"Processing REF line for sequence {cur_seq_id}: {words}")
                    # remove empty words
                    words = [w.strip() for w in words if len(w.strip()) > 0]
                    total_ref += words
                elif line.strip() == "":
                    if len(total_ref) > 0 and len(total_ref) >= numbers[0] + numbers[1] + numbers[2] + numbers[3]:
                        words = [
                            w for w in total_ref if not all([c == "*" for c in w])
                        ]  # filter out *** for insertions
                        word_counter.update([w.lower() for w in words])  # normalize to lowercase
                        error_words = [w for w in words if all(c.isupper() or c == "'" for c in w)]
                        error_counter.update([w.lower() for w in error_words])  # normalize to lowercase

                        assert len(words) == numbers[0] + numbers[1] + numbers[2], (
                            f"Expected {numbers[0]} words, {numbers[1]} substitutions and {numbers[2]} deletions, but found {len(words)} words in REF line for sequence {cur_seq_id}"
                        )

                        assert len(error_words) == numbers[1] + numbers[2], (
                            f"Expected {numbers[1]} substitutions and {numbers[2]} deletions, but found {len(error_words)} error words in REF line for sequence {cur_seq_id}"
                        )
                        total_ref = []
                        numbers = []
                    else:
                        print(
                            f"Empty line, but total_ref = {len(total_ref)} and numbers = {numbers} for sequence {cur_seq_id}, skipping for now"
                        )

        print(f"Counted all the words")
        print(f"Total words counted: {len(word_counter)}")
        print(f"Sum: {sum(word_counter.values())}")

        match_pattern = re.compile(r"^\s*(\d+):\s*(\d+)\s*->\s*(.+)$")

        stats = {
            "insertions": {},
            "deletions": {},
            "substitutions": {},
            "falsely_recognized": {},
            "word_counter": dict(word_counter),
        }

        # get substitutions and deletions from sclite.dtl
        with open(f"{self.report_dir}/sclite.dtl", errors="ignore") as f:
            part = None
            for line in f:
                if len(line.strip()) == 0:
                    continue

                if part is None:
                    if line.startswith("INSERTIONS"):
                        part = "insertions"
                        continue
                elif part == "insertions":
                    if line.startswith("DELETIONS"):
                        part = "deletions"
                        continue
                elif part == "deletions":
                    if line.startswith("SUBSTITUTIONS"):
                        part = "substitutions"
                        continue
                elif part == "substitutions":
                    if line.startswith("FALSELY RECOGNIZED"):
                        part = "falsely_recognized"
                        continue
                elif part == "falsely_recognized":
                    pass

                if part is not None and (extract_word := match_pattern.match(line)):
                    id_, count, word = extract_word.groups()

                    count = int(count)
                    word = word.strip()
                    word = word.lower()
                    assert word not in stats[part], f"Word '{word}' already exists in {part} stats"
                    stats[part][word] = count

        with uopen(self.out_word_stats, "wt") as out:
            out.write(repr(stats))
            out.write("\n")
        print(f"Word stats written to {self.out_word_stats}")

        # do some sanity checks
        for word, count in error_counter.items():
            assert count == stats["deletions"].get(word, 0) + stats["substitutions"].get(word, 0), (
                f"Word '{word}' has {count} errors, but {stats['deletions'].get(word, 0)} deletions and {stats['substitutions'].get(word, 0)} substitutions"
            )

        for word, _ in stats["substitutions"].items():
            assert error_counter[word] == stats["deletions"].get(word, 0) + stats["substitutions"].get(word, 0), (
                f"Word '{word}' has {error_counter[word]} error, but {stats['deletions'].get(word, 0)} deletions and {stats['substitutions'].get(word, 0)} substitutions"
            )


class CategorizeWordStats(Job):
    def __init__(self, word_stats: tk.Path, word_to_category: tk.Path):
        self.word_stats = word_stats
        self.word_to_category = word_to_category
        self.out_categorized_word_stats = self.output_path("categorized_word_stats.py.gz")
        self.out_categorized_word_stats_relative = self.output_path("categorized_word_stats-relative.py.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 2
        return super().hash(d)

    def run(self):
        with uopen(self.word_stats, "rt") as f:
            word_stats = eval(f.read())
        with uopen(self.word_to_category, "rt") as f:
            word_to_category = eval(f.read())

        assert isinstance(word_stats, dict), f"Expected word_stats to be a dict, got {type(word_stats)}"
        assert isinstance(word_to_category, dict), (
            f"Expected word_to_category to be a dict, got {type(word_to_category)}"
        )

        # normalize word_to_category to lowercase
        word_to_category = {word.lower(): category for word, category in word_to_category.items()}

        out_stats = {}

        for stat, counters in word_stats.items():
            out_stat = {}
            assert isinstance(counters, dict), f"Expected counters to be a dict, got {type(counters)}"
            for word, count in counters.items():
                category = word_to_category.get(word.lower(), "unknown")
                assert isinstance(category, str)
                out_stat.setdefault(category, 0)
                out_stat[category] += count
            out_stats[stat] = out_stat

        out_stats = dict(sorted(out_stats.items(), key=lambda item: item[0]))
        with uopen(self.out_categorized_word_stats, "wt") as out:
            out.write(repr(out_stats))
            out.write("\n")

        if "word_counter" in out_stats:
            out_stats_relative = {}
            for stat, counters in out_stats.items():
                if "unknown" in counters:
                    print(f"Warning: 'unknown' category found in {stat}, skipping relative calculation for this stat")
                    continue
                if stat == "word_counter":
                    continue
                out_stat = {cat: val / out_stats["word_counter"].get(cat, 1) for cat, val in counters.items()}
                out_stats_relative[stat] = out_stat
            out_stats_relative = dict(sorted(out_stats_relative.items(), key=lambda item: item[0]))
            with uopen(self.out_categorized_word_stats_relative, "wt") as out:
                out.write(repr(out_stats_relative))
                out.write("\n")
        else:
            print("No 'word_counter' found in word stats, skipping relative calculation")
            with uopen(self.out_categorized_word_stats_relative, "wt") as out:
                out.write(repr({}))
                out.write("\n")


class FigureOutHowManyCorrectWordsAreBungedUp(Job):
    def __init__(self, before: tk.Path, after: tk.Path):
        self.before = before  # sclite report dirs
        self.after = after
        self.out_stats_seqs = self.output_path("bungup_stats_sequence_level.py.gz")
        self.out_stats_total = self.output_path("bungup_stats_total.py")
        self.examples_bungled = self.output_path("bungled_examples.txt.gz")
        self.examples_corrected = self.output_path("corrected_examples.txt.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 5
        return super().hash(d)

    def read_report_dir(self, report: tk.Path) -> Dict[str, Tuple[List[bool], int, str, str]]:
        cur_seq_id = None
        ret = {}
        total_ref = []
        total_hyp = []
        numbers = []
        with open(f"{report}/sclite.pra", errors="ignore") as f:
            for line in fix_sclite_pra(f):
                if line.startswith("id:"):
                    # id: (dev-clean/422-122949-0034/422-122949-0034-000)
                    cur_seq_id = line.split()[1][1:-1]
                    assert len(cur_seq_id) > 0
                elif line.startswith("Scores:"):
                    assert cur_seq_id is not None
                    # Scores: (#C #S #D #I) 79 2 2 0
                    parts = line.split()
                    assert len(parts) == 9
                    assert len(numbers) == 0, f"{cur_seq_id}, {line}"
                    numbers = [int(p) for p in parts[-4:]]

                elif line.startswith("REF:") or line.startswith(">> REF:"):
                    assert cur_seq_id is not None
                    words = line[line.index(":") + 1 :].strip().split()
                    words = [w.strip() for w in words if len(w.strip()) > 0]
                    total_ref += words
                elif line.startswith("HYP:") or line.startswith(">> HYP:"):
                    assert cur_seq_id is not None
                    words = line[line.index(":") + 1 :].strip().split()
                    words = [w.strip() for w in words if len(w.strip()) > 0]
                    total_hyp += words
                elif line.strip() == "":
                    if len(total_ref) > 0 and len(total_ref) >= numbers[0] + numbers[1] + numbers[2] + numbers[3]:
                        words = [
                            w for w in total_ref if not all([c == "*" for c in w])
                        ]  # filter out *** for insertions
                        insertions = [w for w in total_ref if all(c == "*" for c in w)]
                        error_list = [not all(c.isupper() or c == "'" for c in w) for w in words]

                        ret[cur_seq_id] = (
                            error_list,  # True if correct
                            len(insertions),
                            " ".join(total_ref),
                            " ".join(total_hyp),
                        )

                        total_ref = []
                        total_hyp = []
                        numbers = []
                    else:
                        print(
                            f"Empty line, but total_ref = {len(total_ref)} and numbers = {numbers} for sequence {cur_seq_id}, skipping for now"
                        )
        return ret

    def run(self):
        before = self.read_report_dir(self.before)
        after = self.read_report_dir(self.after)

        assert len(before) == len(after), "Before and after reports must have the same number of sequences"
        assert set(before.keys()) == set(after.keys()), "Before and after reports must have the same sequence IDs"

        stats = {}
        total_correct_made_incorrect = 0
        total_incorrect_made_correct = 0
        total_correct_before = 0
        total_incorrect_before = 0
        total_before_insertions = 0
        total_after_insertions = 0

        out_bungled = ""
        out_corrected = ""

        for seq_id in before.keys():
            before_correct, before_insertions, before_refs, before_hyps = before[seq_id]
            after_correct, after_insertions, after_refs, after_hyps = after[seq_id]

            assert len(before_correct) == len(after_correct), (
                f"Sequence {seq_id} has different number of errors before and after"
            )

            # count how many words were bunged up
            correct_made_incorrect = sum(1 for b, a in zip(before_correct, after_correct) if b and not a)
            incorrect_made_correct = sum(1 for b, a in zip(before_correct, after_correct) if not b and a)
            # same = sum(1 for b, a in zip(before_correct, after_correct) if b == a)

            if correct_made_incorrect == 0 and incorrect_made_correct > 0:
                out_corrected += f"{seq_id}\n"
                out_corrected += f"RefB: {before_refs}\n"
                out_corrected += f"RefA: {after_refs}\n"
                out_corrected += f"Before: {before_hyps}\n"
                out_corrected += f"After: {after_hyps}\n"
                out_corrected += "\n"
            elif incorrect_made_correct == 0 and correct_made_incorrect > 0:
                out_bungled += f"{seq_id}\n"
                out_bungled += f"RefB: {before_refs}\n"
                out_bungled += f"RefA: {after_refs}\n"
                out_bungled += f"Before: {before_hyps}\n"
                out_bungled += f"After: {after_hyps}\n"
                out_bungled += "\n"

            num_correct_before = sum(1 for b in before_correct if b)
            num_incorrect_before = len(before_correct) - num_correct_before

            stats[seq_id] = {
                "correct_made_incorrect": correct_made_incorrect,
                "incorrect_made_correct": incorrect_made_correct,
                "num_correct_before": num_correct_before,
                "num_incorrect_before": num_incorrect_before,
                "before_insertions": before_insertions,
                "after_insertions": after_insertions,
            }
            total_correct_made_incorrect += correct_made_incorrect
            total_incorrect_made_correct += incorrect_made_correct
            total_correct_before += num_correct_before
            total_incorrect_before += num_incorrect_before
            total_before_insertions += before_insertions
            total_after_insertions += after_insertions

        print(f"Total correct words made incorrect: {total_correct_made_incorrect}")
        print(f"Total incorrect words made correct: {total_incorrect_made_correct}")
        print(f"Total correct words before: {total_correct_before}")
        print(f"Total incorrect words before: {total_incorrect_before}")
        print(f"Total before insertions: {total_before_insertions}")
        print(f"Total after insertions: {total_after_insertions}")

        with uopen(self.out_stats_seqs, "wt") as out:
            out.write(repr(stats))
            out.write("\n")

        print(f"Stats per sequence written to {self.out_stats_seqs}")

        with uopen(self.out_stats_total, "w") as out:
            out.write(
                repr(
                    {
                        "total_correct_made_incorrect": total_correct_made_incorrect,
                        "total_incorrect_made_correct": total_incorrect_made_correct,
                        "total_correct_before": total_correct_before,
                        "total_incorrect_before": total_incorrect_before,
                        "total_before_insertions": total_before_insertions,
                        "total_after_insertions": total_after_insertions,
                    }
                )
            )
            out.write("\n")

        print(f"Total stats written to {self.out_stats_total}")

        with uopen(self.examples_bungled, "wt") as out:
            out.write(out_bungled)
            out.write("\n")

        with uopen(self.examples_corrected, "wt") as out:
            out.write(out_corrected)
            out.write("\n")


class SimulateTokenSubstitution(Job):
    def __init__(self, text_file: tk.Path, vocab_opts: Dict[str, Any], swapout_prob: Tuple[float, float]):
        self.text_file = text_file
        self.vocab_opts = vocab_opts
        self.swapout_prob = swapout_prob  # (min, max) probabilities for swapping out tokens
        self.text_file_out = self.output_path("text_file_out.txt.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 2
        return super().hash(d)

    def random_swapout(self, line: str, vocab: Vocabulary) -> str:
        words = line.strip()
        word_ids = vocab.get_seq(words)
        if not word_ids:
            print(f"No valid words to swap out in line: {line}")
            return words  # No valid words to swap out
        prob = random.uniform(*self.swapout_prob)
        for i in range(len(word_ids)):
            if random.uniform(0, 1) < prob:  # Randomly decide to swap out
                word_ids[i] = random.randint(0, vocab.num_labels - 1)  # Swap with a random word ID
        return vocab.get_seq_labels(word_ids)

    def run(self):
        import sys
        import os
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(None)  # self.returnn_root
        sys.path.insert(1, returnn_root.get_path())

        from returnn.datasets.util.vocabulary import Vocabulary

        vocab = self.vocab_opts
        vocab = util.instanciate_delayed(vocab)
        print("RETURNN vocab opts:", vocab)
        vocab = Vocabulary.create_vocab(**vocab)
        print("Vocab:", vocab)
        print("num labels:", vocab.num_labels)
        assert vocab.num_labels == len(vocab.labels)

        with uopen(self.text_file, "rt") as f, uopen(self.text_file_out, "wt") as out_f:
            for line in f:
                line: str
                out_f.write(self.random_swapout(line, vocab) + "\n")
