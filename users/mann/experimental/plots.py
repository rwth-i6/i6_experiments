from sisyphus import Job, Task, tk

import pickle
import numpy as np
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter

import i6_core.lib.rasr_cache as rc
from i6_core.lib import corpus, lexicon
from i6_experiments.users.mann.setups.state_tying import Allophone
from typing import Tuple

class OccurrenceCounter:
    
    def __init__(self, occurrence_thresholds: Tuple[float, float], duration_threshold: int = 10):
        self.counter = 0
        self.on = False
        self.duration = 0

        self.duration_threshold = 10
        self.enter_threshold, self.exit_threshold = occurrence_thresholds

    def feed_prob(self, prob: float):
        if self.on and prob < self.exit_threshold and self.duration > self.duration_threshold:
            self.on = False
            self.counter += 1
            self.duration = 0
        elif not self.on and prob > self.enter_threshold:
            self.on = True
        if self.on:
            self.duration += 1
    
    def __str__(self):
        return f"OccurrenceCounter(counter={self.counter}, on={self.on}, duration={self.duration})"


class AllophoneSequencer:
    def __init__(self, corpus, lexicon, state_tying, hmm_partition):
        self.corpus = corpus
        self.lexicon = lexicon
        self.state_tying = state_tying
        self.states_per_phone = hmm_partition

        self.lexicon_dict = {}
        self.corpus_dict = {}
        self.state_tying_dict = {}

        self.init_lexicon_dict()
        self.init_corpus_dict()
        self.init_state_tying_dict()
    
    def init_lexicon_dict(self):
        lex = lexicon.Lexicon()
        lex.load(self.lexicon.get_path())
        # build lookup dict
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if orth:
                    self.lexicon_dict[orth] = [phon.split(" ") for phon in lemma.phon]
        return

    def init_state_tying_dict(self):
        state_tying = self.state_tying.get_path()
        with open(state_tying, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                allo = Allophone.parse_line(line)
                self.state_tying_dict[allo.write(omit_idx=True)] = allo.idx
        return

    def init_corpus_dict(self):
        c = corpus.Corpus()
        c.load(self.corpus.get_path())
        for segment in c.segments():
            words = segment.orth.split(" ")
            self.corpus_dict[segment.fullname()] = words
        return

    def get_allophone_sequence(self, seq_tag: str) -> List[str]:
        word_seq = self.corpus_dict[seq_tag]
        allo_seq = []
        for word in word_seq:
            allo_seq.append(
                Allophone("[SILENCE]", "#", "#", initial=True, final=True, state=0)
            )
            allo_seq.append(
                [self.get_lemma(phon_seq) for phon_seq in self.lexicon_dict[word][:1]]
            )
        allo_seq.append(
            Allophone("[SILENCE]", "#", "#", initial=True, final=True, state=0)
        )

        # flatten
        allo_seq_flat = []
        for allo in allo_seq:
            if isinstance(allo, list):
                for sub_allo in allo:
                    allo_seq_flat.extend(sub_allo)
            else:
                allo_seq_flat.append(allo)
        # write allophones
        allo_seq_flat = [allo.write(omit_idx=True) for allo in allo_seq_flat]
        return allo_seq_flat
    
    def get_label_sequence(self, allo_seq: List[str]):
        return [ self.state_tying_dict[allo] for allo in allo_seq ]
    
    def get_lemma(self, phon_seq):
        allo_seq = []
        ext_phon_seq = ["#"] + phon_seq + ["#"]
        for left, center, right in zip(
            ext_phon_seq,
            ext_phon_seq[1:],
            ext_phon_seq[2:]
        ):
            allo_seq += [
                Allophone(
                    center, left, right,
                    initial=(left == "#"),
                    final=(right == "#"),
                    state=i
                )
                for i in range(self.states_per_phone)
            ]
        return allo_seq

class PlotSoftAlignmentJob(Job):
    __sis_hash_exclude__ = { "hmm_partition": 3 }

    def __init__(self,
        dumps: tk.Path,
        alignment: tk.Path,
        allophones: tk.Path,
        # bliss_lexicon: tk.Path,
        # bliss_corpus: tk.Path,
        state_tying: tk.Path,
        occurrence_thresholds: float,
        segments: List[str],
        hmm_partition: int=3,
    ):
        self.dumps = dumps
        self.alignment = alignment
        self.allophones = allophones
        self.segments = segments
        self.occurrence_thresholds = occurrence_thresholds
        # self.bliss_lexicon = bliss_lexicon
        # self.bliss_corpus = bliss_corpus
        self.state_tying = state_tying
        self.hmm_partition = hmm_partition

        self.out_plots = {
            segment: self.output_path('plot.{}.png'.format(segment.replace("/", "_")))
            for segment in segments
        }

        self.out_plots_pkl = {
            segment: self.output_path('plot.dump.{}.pkl'.format(segment.replace("/", "_")))
            for segment in segments
        }

        self.out_plots_raw = {
            segment: self.output_path('plot.raw.{}.png'.format(segment.replace("/", "_")))
            for segment in segments
        }
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        # init alignment
        alignments = rc.open_file_archive(self.alignment.get_path())
        alignments.setAllophones(self.allophones.get_path())

        state_tying_dict = {}
        with open(self.state_tying.get_path(), "r") as f:
            for line in f:
                allo_str, idx_str = line.split(" ")
                state_tying_dict[allo_str] = int(idx_str)
            
        # init dumps
        import h5py
        with h5py.File(self.dumps.get_path(), 'r') as f:
            end = 0
            for seq_tag, (seq_length, _) in zip(f["seqTags"], f["seqLengths"]):
                end += seq_length
                seq_tag = seq_tag.decode("utf-8")
                print(seq_tag)
                if self.segments is not None and seq_tag not in self.segments:
                    continue
                begin = end - seq_length
                bw_scores = f["inputs"][begin:end]
                allophone_sequence, aligned_labels = self.get_allophones(alignments, seq_tag)
                label_sequence = self.get_label_sequence(allophone_sequence, state_tying_dict)
                bw_image = self.make_bw_image(
                    bw_scores,
                    allophone_sequence,
                    label_sequence,
                )
                viterbi_image = self.make_viterbi_image(
                    aligned_labels,
                )

                self.plot(
                    bw_image,
                    viterbi_image,
                    allophone_sequence,
                    seq_tag,
                )

                plt.figure()
                plt.imshow(bw_scores.T)
                plt.savefig(self.out_plots_raw[seq_tag].get_path())
    
    def get_allophones(self, alignments, seq_tag) -> List[str]:
        """Code due to Simon Berger."""
        alignment = alignments.read(seq_tag, "align")
        start = 0
        label_idx_sequence = []
        allophone_sequence = []
        max_idx = 1
        while start < len(alignment):
            try:
                allophone = alignments.allophones[alignment[start][1]]  # phon{#+#}@i@f.state
            except AttributeError:
                # filename = alignments._short_seg_names[seq_tag]
                allophone = alignments.files[seq_tag].allophones[alignment[start][1]]
            state = alignment[start][2]
            # if state > self.hmm_partition - 1:
            #     continue
            allophone_sequence.append(".".join((allophone, str(state))))
            end = self.get_next_segment(alignment, start, collapse_3state=(self.hmm_partition == 1))
            length = end - start
            if "[SILENCE]" in allophone:
                label_idx_sequence += [0] * length
            else:
                label_idx_sequence += [max_idx] * length
                max_idx += 1
            start = end
        return allophone_sequence, label_idx_sequence
    
    def get_label_sequence(self, allophone_sequence, state_tying_dict):
        return [ state_tying_dict[allo] for allo in allophone_sequence ]
    
    def make_bw_image(self,
        bw_scores,
        allophone_sequence,
        label_idx_sequence,
        sum_consecutive=[0]
    ):
        """Implementation due to Simon Berger."""
        T, C = np.shape(bw_scores)
        occurrence_counters = {
            label_idx: OccurrenceCounter(self.occurrence_thresholds, duration_threshold=2)
            for label_idx in set(label_idx_sequence)
        }

        sil_idx = 0
        for allo, label in zip(allophone_sequence, label_idx_sequence):
            if "[SILENCE]" in allo:
                sil_idx = label
                break
        
        non_sil_label_indices = [label for allo, label in zip(allophone_sequence, label_idx_sequence) if "[SILENCE]" not in allo]
        non_sil_allophones = [allo for allo in allophone_sequence if "[SILENCE]" not in allo]

        non_sil_label_indices = [sil_idx] + non_sil_label_indices
        non_sil_allophones = ["[SILENCE]{#+#}@i@f.0"] + non_sil_allophones

        image = np.empty((len(non_sil_label_indices), T), dtype=np.float64)
        for t in range(T):
            label_idx_counters = Counter()
            for i, (allophone, label_idx) in enumerate(zip(non_sil_allophones, non_sil_label_indices)):
                if "[SILENCE]" in allophone:
                    # Silence/blank is not occurrence-counted and does not need to sum up states
                    value = bw_scores[t, label_idx]
                else:
                    value = sum([bw_scores[t, label_idx + offset] for offset in sum_consecutive])

                    if label_idx_counters[label_idx] == occurrence_counters[label_idx].counter:
                        occurrence_counters[label_idx].feed_prob(value)
                    else:
                        value = 0
                    label_idx_counters.update([label_idx])
                image[i, t] = value
        return image
    
    def make_viterbi_image(self, label_idx_seq):
        T = len(label_idx_seq)
        C = max(label_idx_seq) + 1
        image = np.zeros((C, T), dtype=np.float32)
        for t, idx in enumerate(label_idx_seq):
            image[idx, t] = 1.0
        return image
    
    def plot(self, bw_image, viterbi_image, allophone_sequence, seq_tag, show_labels=True):
        C, T = np.shape(bw_image)

        fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
        for ax, image in zip(axes, [bw_image, viterbi_image]):
            ax.set_xlabel("Frame")
            ax.xaxis.set_label_coords(0.98, -0.03)
            ax.set_xbound(0, T - 1)
            ax.set_ybound(-0.5, C - 0.5)

            if show_labels:
                if self.hmm_partition == 3:
                    silence_first_allo_seq = ["[SILENCE]"] + [
                        allo.split("{")[0] if allo.split(".")[-1] == "1" else "" for allo in allophone_sequence if "[SILENCE]" not in allo
                    ]
                else:
                    silence_first_allo_seq = ["[SILENCE]"] + [
                        allo.split("{")[0] for allo in allophone_sequence if "[SILENCE]" not in allo
                    ]
                ax.set_yticks(np.arange(C))
                ax.set_yticklabels(silence_first_allo_seq)

            ax.imshow(image, cmap="Blues", interpolation="nearest", aspect="auto", origin="lower")
        pickle.dump(fig, open(self.out_plots_pkl[seq_tag].get_path(), "wb+"))
        fig.savefig(self.out_plots[seq_tag].get_path())


    @staticmethod
    def get_next_segment(align, start, collapse_3state=False):
        assert start < len(align)
        allo_idx = align[start][1]
        state = align[start][2]
        end = start
        while end < len(align) and align[end][1] == allo_idx and (collapse_3state or align[end][2] == state):
            end += 1
        return end
