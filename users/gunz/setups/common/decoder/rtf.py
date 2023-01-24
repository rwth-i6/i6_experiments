__all__ = ["ExtractSearchStatisticsJob"]

import collections
import gzip
import typing
import xml.etree.ElementTree as ET

from sisyphus import tk, Job, Task


Path = tk.setup_path(__package__)


class ExtractSearchStatisticsJob(Job):
    def __init__(
        self,
        search_logs: typing.List[typing.Union[str, tk.Path]],
        corpus_duration_hours: float,
    ):
        self.corpus_duration = corpus_duration_hours
        self.search_logs = search_logs

        self.elapsed_time = self.output_var("elapsed_time")
        self.user_time = self.output_var("user_time")
        self.system_time = self.output_var("system_time")
        self.elapsed_rtf = self.output_var("elapsed_rtf")
        self.user_rtf = self.output_var("user_rtf")
        self.system_rtf = self.output_var("system_rtf")
        self.avg_word_ends = self.output_var("avg_word_ends")
        self.avg_trees = self.output_var("avg_trees")
        self.avg_states = self.output_var("avg_states")
        self.recognizer_time = self.output_var("recognizer_time")
        self.recognizer_rtf = self.output_var("recognizer_rtf")
        self.rescoring_time = self.output_var("rescoring_time")
        self.rescoring_rtf = self.output_var("rescoring_rtf")
        self.tf_lm_time = self.output_var("tf_lm_time")
        self.tf_lm_rtf = self.output_var("tf_lm_rtf")
        self.decoding_rtf = self.output_var("decoding_rtf")
        self.ss_statistics = self.output_var("ss_statistics")
        self.seq_ss_statistics = self.output_var("seq_ss_statistics")
        self.eval_statistics = self.output_var("eval_statistics")

        self.rqmt = {"cpu": 1, "mem": 2.0, "time": 0.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        queue_stats = collections.defaultdict(list)

        total_elapsed = 0.0
        total_user = 0.0
        total_system = 0.0
        total_frames = 0
        total_word_ends = 0.0
        total_trees = 0.0
        total_states = 0.0
        recognizer_time = 0.0
        rescoring_time = 0.0
        lm_time = 0.0
        ss_statistics = collections.defaultdict(lambda: (0.0, 0))
        seq_ss_statistics = {}
        eval_statistics = {}

        for path in self.search_logs:
            with gzip.open(tk.uncached_path(path), "rt") as f:
                root = ET.fromstring(f.read())
            host = root.findall("./system-information/name")[0].text
            elapsed = float(root.findall("./timer/elapsed")[0].text)
            user = float(root.findall("./timer/user")[0].text)
            system = float(root.findall("./timer/system")[0].text)
            total_elapsed += elapsed
            total_user += user
            total_system += system

            for layer in root.findall('.//segment/layer[@name="recognizer"]'):
                frames = int(
                    layer.findall('./statistics/frames[@port="features"]')[0].attrib[
                        "number"
                    ]
                )
                total_frames += frames
                total_word_ends += frames * float(
                    layer.findall(
                        './search-space-statistics/statistic[@name="ending words after pruning"]/avg'
                    )[0].text
                )
                total_trees += frames * float(
                    layer.findall(
                        './search-space-statistics/statistic[@name="trees after  pruning"]/avg'
                    )[0].text
                )
                total_states += frames * float(
                    layer.findall(
                        './search-space-statistics/statistic[@name="states after pruning"]/avg'
                    )[0].text
                )

                recognizer_time += float(layer.findall("./flf-recognizer-time")[0].text)

            for rescore in root.findall(".//segment/flf-push-forward-rescoring-time"):
                rescoring_time += float(rescore.text)

            for lm_total in root.findall(".//fwd-summary/total-run-time"):
                lm_time += float(lm_total.text)

            for seg in root.findall(".//segment"):
                seg_stats = {}
                full_name = seg.attrib["full-name"]
                for sss in seg.findall(
                    './layer[@name="recognizer"]/search-space-statistics/statistic[@type="scalar"]'
                ):
                    min_val = float(sss.findtext("./min", default="0"))
                    avg_val = float(sss.findtext("./avg", default="0"))
                    max_val = float(sss.findtext("./max", default="0"))
                    seg_stats[sss.attrib["name"]] = (min_val, avg_val, max_val)
                seq_ss_statistics[full_name] = seg_stats

                features = seg.find(
                    './layer[@name="recognizer"]/statistics/frames[@port="features"]'
                )
                seq_ss_statistics[full_name]["frames"] = int(features.attrib["number"])

                tf_fwd = seg.find(
                    './layer[@name="recognizer"]/information[@component="flf-lattice-tool.network.recognizer.feature-extraction.tf-fwd"]'
                )
                if tf_fwd is not None:
                    seq_ss_statistics[full_name]["tf_fwd"] = float(
                        tf_fwd.text.strip().split()[-1]
                    )
                else:
                    seq_ss_statistics[full_name]["tf_fwd"] = 0.0

                eval_statistics[full_name] = {}
                for evaluation in seg.findall(".//evaluation"):
                    stat_name = evaluation.attrib["name"]
                    alignment = evaluation.find('statistic[@type="alignment"]')
                    eval_statistics[full_name][stat_name] = {
                        "errors": int(alignment.findtext("edit-operations")),
                        "ref-tokens": int(
                            alignment.findtext(
                                'count[@event="token"][@source="reference"]'
                            )
                        ),
                        "score": float(alignment.findtext('score[@source="best"]')),
                    }

        for s in seq_ss_statistics.values():
            frames = s["frames"]
            for stat, val in s.items():
                if stat == "frames":
                    pass
                elif stat == "tf_fwd":
                    prev_count, prev_frames = ss_statistics[stat]
                    ss_statistics[stat] = (
                        prev_count + val,
                        (3600.0 * 1000.0 * self.corpus_duration),
                    )
                else:
                    prev_count, prev_frames = ss_statistics[stat]
                    ss_statistics[stat] = (
                        prev_count + val[1] * frames,
                        prev_frames + frames,
                    )
        for s in ss_statistics:
            count, frames = ss_statistics[s]
            ss_statistics[s] = count / frames

        self.elapsed_time.set(total_elapsed / 3600.0)
        self.user_time.set(total_user / 3600.0)
        self.system_time.set(total_system / 3600.0)
        self.elapsed_rtf.set(total_elapsed / (3600.0 * self.corpus_duration))
        self.user_rtf.set(total_user / (3600.0 * self.corpus_duration))
        self.system_rtf.set(total_system / (3600.0 * self.corpus_duration))
        self.avg_word_ends.set(total_word_ends / total_frames)
        self.avg_trees.set(total_trees / total_frames)
        self.avg_states.set(total_states / total_frames)
        self.recognizer_time.set(recognizer_time / (3600.0 * 1000.0))
        self.recognizer_rtf.set(
            recognizer_time / (3600.0 * 1000.0 * self.corpus_duration)
        )
        self.rescoring_time.set(rescoring_time / (3600.0 * 1000.0))
        self.rescoring_rtf.set(
            rescoring_time / (3600.0 * 1000.0 * self.corpus_duration)
        )
        self.tf_lm_time.set(lm_time / (3600.0 * 1000.0))
        self.tf_lm_rtf.set(lm_time / (3600.0 * 1000.0 * self.corpus_duration))
        self.decoding_rtf.set(
            (recognizer_time + rescoring_time)
            / (3600.0 * 1000.0 * self.corpus_duration)
        )
        self.ss_statistics.set(dict(ss_statistics.items()))
        self.seq_ss_statistics.set(seq_ss_statistics)
        self.eval_statistics.set(eval_statistics)
