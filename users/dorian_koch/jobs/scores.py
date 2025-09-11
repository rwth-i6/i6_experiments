import collections
from typing import Dict, List, Sequence, Tuple, Any
from i6_experiments.users.zeyer.recog import RecogOutput
from i6_experiments.users.zeyer.datasets.score_results import ScoreResult
from sisyphus import Job, Task, tk
from i6_core.util import uopen
import re


def read_scores(file_path, item_format):
    with uopen(file_path) as f:
        txt = f.read()

    from returnn.util.literal_py_to_pickle import literal_eval

    # Note: literal_py_to_pickle.literal_eval is quite efficient.
    # However, currently, it does not support inf/nan literals,
    # so it might break for some input.
    # We might want to put a simple fallback to eval here if needed.
    # Or maybe extend literal_py_to_pickle.literal_eval to support inf/nan literals.
    try:
        data: Dict[str, Any] = literal_eval(txt)
    except Exception as exc:
        print(f": Warning: literal_py_to_pickle.literal_eval failed:")
        print(f"  {type(exc).__name__}: {exc}")
        print("  Fallback to eval...")
        data: Dict[str, Any] = eval(txt)

    assert data is not None
    assert isinstance(data, dict)
    assert len(data) > 0

    # Check some data.
    key, value = next(iter(data.items()))
    assert isinstance(key, str), f"expected seq tag as keys, got {key!r} ({type(key)})"  # seq tag
    if item_format == "single":
        assert isinstance(value, str), f" expected str ({item_format}), got {value!r} ({type(value)})"
    elif item_format == "list_with_scores":
        assert isinstance(value, list), f": expected list ({item_format}), got {value!r} ({type(value)})"
        assert len(value) > 0, f" expected non-empty list ({item_format}), got {value!r} for seq {key}"
        value0 = value[0]
        assert (
            isinstance(value0, tuple)
            and len(value0) == 2
            and isinstance(value0[0], float)
            and isinstance(value0[1], str)
        ), f"expected (score,text) tuples ({item_format}), got {value0!r} ({type(value0)})"
    else:
        raise ValueError(f"invalid item_format {item_format!r}")

    return data


class TextDictToScoresTextDictJob(Job):
    def __init__(self, *, text_dict: RecogOutput):
        self.text_dict = text_dict

        self.out_scores = self.output_path("scores.txt.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 2
        return super().hash(d)

    def get_recog_output(self) -> RecogOutput:
        return RecogOutput(output=self.out_scores)

    def run(self):
        """run"""
        data = read_scores(self.text_dict.output, item_format="single")

        with uopen(self.out_scores, "wt") as f:
            f.write("{\n")
            for key, value in data.items():
                assert isinstance(value, str), f"{self}: expected str text, got {value!r} ({type(value)})"
                f.write(f"{key!r}: [(0.0, {value!r})],\n")
            f.write("}\n")


class AverageScores(Job):
    def __init__(self, *, scores: RecogOutput, length_normalize_with_vocab: Dict[str, Any] | None = None):
        self.scores = scores
        self.length_normalize_with_vocab = length_normalize_with_vocab

        self.out_avg = self.output_var("avg")
        self.out_median = self.output_var("median")

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        # d["__version"] = 3
        return super().hash(d)

    def tasks(self):
        # yield Task("run", rqmt={"cpu": 16, "mem": 8, "time": 1})
        yield Task("run", mini_task=True)

    def run(self):
        # See TextDictDataset
        data = read_scores(self.scores.output, item_format="list_with_scores")

        for key, value in data.items():
            assert len(value) == 1  # just one score
            score, text = value[0]
            assert isinstance(score, float), f"{self}: expected float score, got {score!r} ({type(score)})"
            assert isinstance(text, str), f"{self}: expected str text, got {text!r} ({type(text)})"

        # Calculate average and median.
        if self.length_normalize_with_vocab is not None:
            from returnn.datasets.util.vocabulary import Vocabulary
            import i6_core.util as util

            vocab = self.length_normalize_with_vocab
            vocab = util.instanciate_delayed(vocab)
            print("RETURNN vocab opts:", vocab)
            vocab = Vocabulary.create_vocab(**vocab)
            print("Vocab:", vocab)
            print("num labels:", vocab.num_labels)
            assert vocab.num_labels == len(vocab.labels)
            scores = []
            for _, value in data.items():
                score, text = value[0]
                splitted_text = vocab.get_seq(text.strip())
                # +1 because end of sentence
                scores.append(score / (1 + len(splitted_text)))
        else:
            scores = [value[0][0] for value in data.values()]
        avg_score = sum(scores) / len(scores)
        median_score = sorted(scores)[len(scores) // 2]

        self.out_avg.set(avg_score)
        self.out_median.set(median_score)


class CalcSearchErrors(Job):
    def __init__(self, *, ref_scores: RecogOutput, hyp_scores: RecogOutput):
        import Levenshtein

        self.ref_scores = ref_scores
        self.hyp_scores = hyp_scores

        self.out_search_errors = self.output_var("search_errors")
        self.out_model_error = self.output_var(
            "model_error"
        )  # number of times the model didnt get the right answer, and was scored lower
        self.out_oracle_wer = self.output_var("oracle_wer")

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 6
        return super().hash(d)

    def get_search_error(self, corpus_name: str):
        return ScoreResult(dataset_name=corpus_name, main_measure_value=self.out_search_errors)

    def get_model_error(self, corpus_name: str):
        return ScoreResult(dataset_name=corpus_name, main_measure_value=self.out_model_error)

    def get_oracle_wer(self, corpus_name: str):
        return ScoreResult(dataset_name=corpus_name, main_measure_value=self.out_oracle_wer)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import Levenshtein

        # See TextDictDataset
        ref_scores = read_scores(self.ref_scores.output, item_format="list_with_scores")
        hyp_scores = read_scores(self.hyp_scores.output, item_format="list_with_scores")

        assert len(ref_scores) == len(hyp_scores), f"{self}: ref and hyp scores must have the same number of items"
        assert set(ref_scores.keys()) == set(hyp_scores.keys())

        num_errors = 0
        num_model_error = 0
        num_total = len(ref_scores)
        total_words = 0
        total_oracle_edit_distance = 0

        for key in ref_scores:
            ref_value = ref_scores[key]
            hyp_value = hyp_scores[key]

            assert len(ref_value) == 1
            assert len(hyp_value) > 0, f"{self}: expected non-empty hyp value list, got {hyp_value!r}"
            ref_score, ref_text = ref_value[0]
            ref_line = ref_text.strip().split()

            lowest_edit_distance = 9999999999
            for hyp_score, hyp_text in hyp_value:
                assert isinstance(hyp_score, float), (
                    f"{self}: expected float score, got {hyp_score!r} ({type(hyp_score)})"
                )
                assert isinstance(hyp_text, str), f"{self}: expected str text, got {hyp_text!r} ({type(hyp_text)})"

                hyp_line = hyp_text.strip().split()

                lowest_edit_distance = min(lowest_edit_distance, Levenshtein.distance(ref_line, hyp_line))
            total_words += len(ref_line)
            total_oracle_edit_distance += lowest_edit_distance
            hyp_score, hyp_text = max(hyp_value)

            has_error = ref_score > hyp_score and ref_text.strip() != hyp_text.strip()
            if has_error:
                num_errors += 1
            if ref_text.strip() != hyp_text.strip() and ref_score < hyp_score:
                num_model_error += 1

            print(f"{has_error}: ref={ref_score}, hyp={hyp_score}")
            print(f" ref={ref_text!r}")
            print(f" hyp={hyp_text!r}")

        search_error_rate = 100 * num_errors / num_total if num_total > 0 else 0.0
        model_error_rate = 100 * num_model_error / num_total if num_total > 0 else 0.0
        oracle_wer = 100 * total_oracle_edit_distance / total_words if total_words > 0 else 0.0

        self.out_search_errors.set(search_error_rate)
        self.out_model_error.set(model_error_rate)
        self.out_oracle_wer.set(oracle_wer)

        print(f"{self}: Search error rate: {search_error_rate:.4f} ({num_errors}/{num_total})")
        print(f"{self}: Model error rate: {model_error_rate:.4f} ({num_model_error}/{num_total})")
        print(f"{self}: Oracle WER: {oracle_wer:.4f} ({total_oracle_edit_distance}/{total_words})")
