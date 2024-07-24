__all__ = ["LatticeToNBestListJob", "NBestListToHDFDatasetJob"]

import os
import numpy as np
import re
import shutil
import sys
from typing import List, Optional

from sisyphus import tk, Job, Task

import i6_core.lib.corpus as corpus
import i6_core.rasr as rasr
from i6_core.lib.hdf import get_returnn_simple_hdf_writer


class LatticeToNBestListJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp: rasr.CommonRasrParameters,
        lattice_cache: tk.Path,
        n: int,
        *,
        parallelize: bool = False,
        remove_duplicates: bool = True,
        ignore_non_words: bool = True,
        word_level: bool = True,
        extra_config: Optional[rasr.RasrConfig] = None,
        extra_post_config: Optional[rasr.RasrConfig] = None,
    ):
        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = self.create_config(**kwargs)
        self.concurrent = crp.concurrent if parallelize else 1
        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")
        self.lattice_cache = lattice_cache

        self.out_log_file = self.log_file_output_path("nbest_from_lattice", crp, self.concurrent > 1)
        self.out_nbest_file = self.output_path("nbest")

        self.rqmt = {
            "time": max(crp.corpus_duration / (5.0 * self.concurrent), 168),
            "cpu": 1,
            "mem": 4,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        if self.concurrent > 1:
            yield Task("merge", mini_task=True, tries=2)  # 2 tries to catch fs problems crashing the job

    def create_files(self):
        self.write_config(self.config, self.post_config, "nbest_from_lattice.config")
        self.write_run_script(self.exe, "nbest_from_lattice.config")

    def run(self, task_id):
        log_file = self.out_log_file if self.concurrent <= 1 else self.out_log_file[task_id]
        self.run_script(task_id, log_file)
        if self.concurrent <= 1:
            shutil.move("nbest.1", self.out_nbest_file.get_path())

    def merge(self):
        with open(self.out_nbest_file.get_path(), "wt") as out:
            for t in range(1, self.concurrent + 1):
                assert os.path.getsize("nbest.%d" % t) > 0, "Empty File nbest.%d, maybe restart merge" % t
                with open("nbest.%d" % t, "rt") as f:
                    shutil.copyfileobj(f, out)

    @classmethod
    def create_config(
        cls,
        crp,
        lattice_cache,
        n,
        parallelize,
        remove_duplicates,
        ignore_non_words,
        word_level,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "flf-lattice-tool.corpus",
                "lexicon": "flf-lattice-tool.lexicon",
            },
            parallelize=parallelize,
        )
        # segment
        config.flf_lattice_tool.network.initial_nodes = "segment"
        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = "0->archive-reader:1 0->dump-n-best:1"

        # read lattice
        config.flf_lattice_tool.network.archive_reader.type = "archive-reader"
        config.flf_lattice_tool.network.archive_reader.links = "to-lemma"
        config.flf_lattice_tool.network.archive_reader.format = "flf"
        config.flf_lattice_tool.network.archive_reader.path = lattice_cache

        # map alphabet
        config.flf_lattice_tool.network.to_lemma.type = "map-alphabet"
        config.flf_lattice_tool.network.to_lemma.map_input = "to-lemma" if word_level else "to-lemma-pron"
        config.flf_lattice_tool.network.to_lemma.project_input = True
        config.flf_lattice_tool.network.to_lemma.links = "n-best"

        # dump n-best
        config.flf_lattice_tool.network.n_best.type = "n-best"
        config.flf_lattice_tool.network.n_best.n = n
        config.flf_lattice_tool.network.n_best.remove_duplicates = remove_duplicates
        config.flf_lattice_tool.network.n_best.ignore_non_words = ignore_non_words
        config.flf_lattice_tool.network.n_best.links = "dump-n-best"

        config.flf_lattice_tool.network.dump_n_best.type = "dump-n-best"
        config.flf_lattice_tool.network.dump_n_best.links = "sink:0"
        config.flf_lattice_tool.network.dump_n_best.dump.channel = "nbest.$(TASK)"

        # sink
        config.flf_lattice_tool.network.sink.type = "sink"
        config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config


class NBestListToHDFDatasetJob(Job):
    def __init__(
        self,
        n: int,
        nbest_path: tk.Path,
        corpus_file: tk.Path,
        state_tying: tk.Path,
        returnn_root: Optional[tk.Path] = None,
        risk_phoneme_level: bool = True,
        phoneme_alignment: Optional[List[tk.Path]] = None,
        lm_scores: Optional[tk.Path] = None,
        ignore_non_words: bool = True,
        concurrent: int = 1,
    ):
        self.n = n
        self.nbest_path = nbest_path
        self.corpus_file = corpus_file
        self.state_tying = state_tying
        self.returnn_root = returnn_root
        self.risk_phoneme_level = risk_phoneme_level
        self.phoneme_alignment = phoneme_alignment
        self.lm_scores = lm_scores
        self.ignore_non_words = ignore_non_words
        self._non_words = []
        self.concurrent = concurrent

        self.out_classes_hdfs = {idx: self.output_path(f"classes.{idx}.hdf") for idx in range(1, concurrent + 1)}
        self.out_classes_lens_hdfs = {
            idx: self.output_path(f"classes_lens.{idx}.hdf") for idx in range(1, concurrent + 1)
        }
        self.out_score_hdfs = {idx: self.output_path(f"score.{idx}.hdf") for idx in range(1, concurrent + 1)}
        self.out_risk_hdfs = {idx: self.output_path(f"risk.{idx}.hdf") for idx in range(1, concurrent + 1)}

        self.rqmt = {
            "time": 168,
            "cpu": 1,
            "mem": 32,
        }

    def tasks(self):
        if self.concurrent > 1:
            yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        else:
            yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self, task_id=None):
        returnn_root = None if self.returnn_root is None else self.returnn_root.get_path()

        # load corpus
        c = corpus.Corpus()
        c.load(self.corpus_file.get())

        # load state tying
        state_tying = {}
        with open(self.state_tying.get()) as f:
            for line in f.read().splitlines():
                phone = line.split("{")[0]
                idx = int(line.split()[-1])
                if phone in state_tying:
                    assert state_tying[phone] == idx, f"phone {phone}: {idx} vs. existing {state_tying[phone]}"
                else:
                    state_tying[phone] = idx
                    if "[" in phone:
                        self._non_words.append(idx)
                        self._non_words.append(phone.lower())

        # load alignment for phoneme level risk
        phoneme_alignment = None
        if self.risk_phoneme_level:
            assert self.phoneme_alignment is not None, "Phoneme alignment is needed to compute risk on phoneme level"
            if returnn_root is not None:
                sys.path.append(returnn_root)
            from returnn.datasets.hdf import HDFDataset
            phoneme_alignment = HDFDataset(self.phoneme_alignment, use_cache_manager=True)

        # load reference LM scores
        if self.lm_scores is not None:
            with open(self.lm_scores.get()) as f:
                lm_scores = eval(f.read())

        # create datasets
        hdf_writer = get_returnn_simple_hdf_writer(returnn_root)
        classes_hdf = hdf_writer(
            filename=self.out_classes_hdfs[task_id].get(),
            dim=self.n,
            ndim=2,
        )
        classes_lens_hdf = hdf_writer(
            filename=self.out_classes_lens_hdfs[task_id].get(),
            dim=self.n,
            ndim=1,
        )
        score_hdf = hdf_writer(
            filename=self.out_score_hdfs[task_id].get(),
            dim=self.n,
            ndim=1,
        )
        risk_hdf = hdf_writer(
            filename=self.out_risk_hdfs[task_id].get(),
            dim=self.n,
            ndim=1,
        )

        with open(self.nbest_path.get()) as f:
            file_content = f.read()
        num_seqs_dumped = 0
        # utterances are separated by an empty line
        for utterance_lines in file_content.split("\n\n")[task_id - 1::self.concurrent]:
            if len(utterance_lines) == 0:
                continue
            utterance_n_hyps = 0
            ref_in_hyps = False
            ref_in_hyps_with_mismatching_pronunciation = False
            classes = []
            classes_lens = []
            score = []
            risk_word_level = []
            for idx, line in enumerate(utterance_lines.splitlines()):
                if utterance_n_hyps == self.n and ref_in_hyps:
                    break  # do not read further since we already have N
                if idx == 0:
                    assert line.startswith("# "), f"Could not parse line {idx}: {line}"
                    segment = line.split()[1]
                    ref_words = c.get_segment_by_name(segment).orth.lower().split()
                    if phoneme_alignment is not None and segment in phoneme_alignment.get_all_tags():
                        ref_phonemes = phoneme_alignment.get_data_by_seq_tag(segment, "data")
                        ref_phonemes = ref_phonemes[ref_phonemes > 0]
                    else:
                        ref_phonemes = None
                elif idx == 1:
                    pass  # contains information about N (how long is N-best list for this utterance)
                elif idx == 2:
                    assert line.startswith("# "), f"Could not parse line {idx}: {line}"
                    col_headers = line.split()[1:]
                    scales = {col.split("/")[0]: float(col.split("/")[1]) for col in col_headers if "/" in col}
                    col_headers = [col.split("/")[0] for col in col_headers]
                else:
                    hyp = re.search(r"<s>(.*)</s>", line).group(1)
                    if hyp.strip():
                        assert "/" in hyp, f"hypothesis does not seem to contain pronunciation: '{hyp}' in line {idx + 1}"
                    hyp_words = [word.split("  /")[0].strip().lower() for word in hyp.split("/ ") if word.strip()]
                    hyp_pronunciations = [word.split("  /")[1].strip() for word in hyp.split("/ ") if word.strip()]
                    hyp_labels = " ".join(hyp_pronunciations).split()
                    classes_n = [state_tying[hyp_label] for hyp_label in hyp_labels]
                    if (
                        len(classes) == 0 or
                        min(self.levenshtein_distance(classes_n, clss) for clss in classes) > 0 and  # not in nbest list
                        len(classes) < self.n
                    ):
                        classes.append(classes_n)
                        classes_lens.append(len(classes_n))
                        scores = {
                            col_headers[col_idx]: float(line.split()[col_idx])
                            for col_idx in range(len(col_headers))
                        }
                        score.append(-sum(scores[col] * scales[col] for col in ["lm"]))
                        risk_word_level.append(self.levenshtein_distance(ref_words, hyp_words))
                        utterance_n_hyps += 1
                        if self.levenshtein_distance(ref_words, hyp_words) == 0:
                            ref_in_hyps = True
                    elif not ref_in_hyps and self.levenshtein_distance(ref_words, hyp_words) == 0:
                        classes[-1] = classes_n
                        classes_lens[-1] = len(classes_n)
                        scores = {
                            col_headers[col_idx]: float(line.split()[col_idx])
                            for col_idx in range(len(col_headers))
                        }
                        score[-1] = -sum(scores[col] * scales[col] for col in scales)
                        risk_word_level[-1] = self.levenshtein_distance(ref_words, hyp_words)
                        ref_in_hyps = True
                        if ref_phonemes is not None:
                            if self.levenshtein_distance(ref_phonemes, classes_n) > 0:
                                ref_in_hyps_with_mismatching_pronunciation = True

            ref_in_hyps = ref_in_hyps or ref_in_hyps_with_mismatching_pronunciation
            if not ref_in_hyps:
                if self.lm_scores is not None and segment in lm_scores and ref_phonemes is not None:
                    classes[-1] = ref_phonemes.tolist()
                    classes_lens[-1] = len(ref_phonemes)
                    score[-1] = lm_scores[segment] * scales["lm"]
                    risk_word_level[-1] = 0
                    print(f"Force ref into hyps for segment {segment}")
                else:
                    print(f"WARNING: No ref or LM score and alignment found for segment {segment}")
            if ref_in_hyps_with_mismatching_pronunciation:
                print(
                    f"Use hypothesis that is correct on word level "
                    f"but does not match phoneme level alignment in segment {segment}"
                )

            if self.risk_phoneme_level:
                if ref_phonemes is not None:
                    risk = [self.levenshtein_distance(ref_phonemes, classes[idx]) for idx in range(len(classes))]
                else:
                    print(f"WARNING: segment not found in alignment: {segment}, use heuristics instead")
                    ref_idx = np.argmin(risk_word_level)
                    # this is not correct if the reference is not in the hypotheses, and it might use a different
                    # pronunciation variant than the alignment would
                    risk = [self.levenshtein_distance(classes[ref_idx], classes[idx]) for idx in range(len(classes))]
            else:
                risk = risk_word_level

            # pad classes to max length
            max_classes = max(len(classes[n_idx]) for n_idx in range(min(self.n, len(classes))))
            if max_classes == 0:
                classes[0] = [state_tying["[SILENCE]"]]
            for n_idx in range(len(classes)):
                classes[n_idx] += [state_tying["[SILENCE]"]] * (max_classes - len(classes[n_idx]))

            # pad if less than self.n hypotheses exist
            for _ in range(self.n - len(classes)):
                classes.append([state_tying["[SILENCE]"]] * len(classes[0]))
                classes_lens.append(0)
                score.append(0.0)
                risk.append(0)

            # write to datasets
            classes_hdf.insert_batch(
                inputs=np.array(classes).T.reshape(1, -1, self.n),
                seq_len=[np.array(classes).shape[1]],
                seq_tag=[segment],
            )
            classes_lens_hdf.insert_batch(
                inputs=np.array(classes_lens).reshape(1, self.n),
                seq_len=[self.n],
                seq_tag=[segment],
            )
            score_hdf.insert_batch(
                inputs=np.array(score).reshape(1, self.n),
                seq_len=[self.n],
                seq_tag=[segment],
            )
            risk_hdf.insert_batch(
                inputs=np.array(risk).reshape(1, self.n),
                seq_len=[self.n],
                seq_tag=[segment],
            )
            num_seqs_dumped += 1
            if num_seqs_dumped % 100 == 0 or num_seqs_dumped == 1:
                print(f"Dumped {num_seqs_dumped} seqs")

        classes_hdf.close()
        classes_lens_hdf.close()
        score_hdf.close()
        risk_hdf.close()

    def levenshtein_distance(self, s1, s2):
        if self.ignore_non_words:
            s1 = [s for s in s1 if s not in self._non_words]
            s2 = [s for s in s2 if s not in self._non_words]

        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]
