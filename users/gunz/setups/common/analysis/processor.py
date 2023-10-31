import gzip
import logging
import numpy as np
import random
import subprocess
import sys
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

from i6_core.lib.rasr_cache import FileArchiveBundle

from .allophone_state import AllophoneState


class AlignmentProcessor:
    def __init__(
        self,
        alignment_bundle_path: str,
        allophones_path: str,
        sil_allophone: str = "[SILENCE]",
        monophone: bool = True,
    ):
        alignment_bundle_path = (
            subprocess.check_output(["cf", alignment_bundle_path]).decode(sys.stdout.encoding).strip()
        )
        allophones_path = subprocess.check_output(["cf", allophones_path]).decode(sys.stdout.encoding).strip()

        logging.info(f"bundle={alignment_bundle_path}")
        logging.info(f"allos={allophones_path}")

        self.alignment_bundle = FileArchiveBundle(alignment_bundle_path)
        self.alignment_bundle.setAllophones(allophones_path)
        self.monophone = monophone
        self.segments = [s for s in self.alignment_bundle.file_list() if not s.endswith(".attribs")]
        self.sil_allophone = sil_allophone

        logging.info(f"files={', '.join(self.segments[:3])}")

    def get_alignment_states(self, seg_name: str) -> str:
        return " ".join(self._get_alignment_states(seg_name))

    def get_raw_alignment_ids(self, seg_name: str) -> List[int]:
        alignment = self.alignment_bundle.read(seg_name, "align")
        return [t[1] for t in alignment]

    def percent_silence(self, sample: int = 5000):
        segments = random.sample(self.segments, sample) if sample > 0 else self.segments
        sil_per_s = [self.percent_silence_in(seg) for seg in segments]
        return sum(sil_per_s) / len(sil_per_s)

    def percent_silence_in(self, seg_name: str):
        alignment_states = self._get_alignment_states(seg_name)
        if len(alignment_states) == 0:
            return 0.0
        num_sil = sum((1 for st in alignment_states if self.sil_allophone in st.upper()))
        return num_sil / len(alignment_states)

    def plot_segment(self, seg_name: str, show_labels: bool, font_size: int = 10, show_title: bool = True):
        def get_next_segment(align, start, collapse_3state=False):
            assert start < len(align)
            allo_idx = align[start][1]
            state = align[start][2]
            end = start
            while end < len(align) and align[end][1] == allo_idx and (collapse_3state or align[end][2] == state):
                end += 1
            return end

        def get_allophones(alignments: FileArchiveBundle, seq_tag) -> Tuple[List[str], List[int]]:
            """Code from Simon Berger."""
            alignment = alignments.read(seq_tag, "align")
            start = 0
            label_idx_sequence = []
            allophone_sequence = []
            max_idx = 1
            while start < len(alignment):
                allophone = alignments.files[seq_tag].allophones[alignment[start][1]]
                state = alignment[start][2]
                allophone_sequence.append(".".join((allophone, str(state))))
                end = get_next_segment(alignment, start, collapse_3state=True)
                length = end - start
                if self.sil_allophone in allophone:
                    label_idx_sequence += [0] * length
                else:
                    label_idx_sequence += [max_idx] * length
                    max_idx += 1
                start = end

            return allophone_sequence, label_idx_sequence

        def make_viterbi_image(label_idx_seq):
            T = len(label_idx_seq)
            C = max(label_idx_seq) + 1
            image = np.zeros((C, T), dtype=np.float32)
            for t, idx in enumerate(label_idx_seq):
                image[idx, t] = 1.0
            return image

        def plot(viterbi_image, allophone_sequence, seq_tag, show_labels=True):
            import matplotlib.pyplot as plt

            plt.rc("font", family="Latin Modern Roman", size=font_size)

            C, T = np.shape(viterbi_image)

            fig, ax = plt.subplots()
            if show_title:
                ax.set_title(f"Viterbi alignment of\n{seq_tag}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("State")
            # ax.xaxis.set_label_coords(0.98, -0.03)
            ax.set_xbound(0, T - 1)
            ax.set_ybound(-0.5, C - 0.5)

            if show_labels:
                silence_first_allo_seq = [self.sil_allophone] + [
                    allo.split("{")[0] for allo in allophone_sequence if self.sil_allophone not in allo
                ]
                ax.set_yticks(np.arange(C))
                ax.set_yticklabels(silence_first_allo_seq)

            ax.imshow(
                viterbi_image,
                cmap="Blues",
                interpolation="nearest",
                aspect="auto",
                origin="lower",
            )
            return fig, ax

        sequence, aligned_indices = get_allophones(self.alignment_bundle, seg_name)
        img = make_viterbi_image(aligned_indices)
        fig, ax = plot(img, sequence, seg_name, show_labels=show_labels)

        return fig, ax, img, sequence, seg_name

    def _get_alignment_states(self, seg_name: str) -> List[str]:
        alignment = self.alignment_bundle.read(seg_name, "align")
        alignment_states = [f"{self.alignment_bundle.files[seg_name].allophones[t[1]]}.{t[2]:d}" for t in alignment]
        if self.monophone:
            alignment_states = [AllophoneState.from_alignment_state(st).to_mono() for st in alignment_states]
        return alignment_states


class CorpusProcessor:
    def __init__(self, corpus_file_path: str):
        corpus_file_path = subprocess.check_output(["cf", corpus_file_path]).decode(sys.stdout.encoding).strip()

        if corpus_file_path.endswith("gz"):
            with gzip.open(corpus_file_path, "rt") as corpus_file:
                self.corpus = ET.parse(corpus_file)
        else:
            with open(corpus_file_path, "rt") as corpus_file:
                self.corpus = ET.parse(corpus_file)

    def get_audio(self, seg_name: str) -> Optional[str]:
        try:
            corpus_name, recording, segment = seg_name.split("/")

            el = self.corpus.find(f".//recording[@name='{recording}']")
            return el.attrib["audio"].strip()
        except:
            return None

    def get_transcription(self, seg_name: str) -> Optional[str]:
        try:
            corpus_name, recording, segment = seg_name.split("/")

            el = self.corpus.find(f".//recording[@name='{recording}']/segment[@name='{segment}']/orth")
            return el.text.strip()
        except:
            return None
