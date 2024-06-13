__all__ = ["ComputeSilenceRatioJob", "ComputeAveragePhonemeLengthJob", "ComputeLabelStatisticsJob"]

from collections import Counter, defaultdict
import itertools
import logging
from typing import List, Union

from sisyphus import tk, Job, Task

import i6_core.lib.rasr_cache as rasr_cache


class ComputeSilenceRatioJob(Job):
    def __init__(
        self,
        allophone_file: tk.Path,
        alignment_files: Union[tk.Path, List[tk.Path]],
        silence_label: str,
    ):
        self.allophone_file = allophone_file
        self.alignment_files = alignment_files
        self.silence_label = silence_label

        self.out_silence_frames = self.output_var("silence_frames.txt")
        self.out_total_frames = self.output_var("total_frames.txt")
        self.out_silence_ratio = self.output_var("silence_ratio.txt")

        self.rqmt = {"cpu": 1, "mem": 24, "time": 24, "sbatch_args": ["-p", "cpu_slow"]}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def load_alignment(allophones_file: tk.Path, alignment_file: Union[tk.Path, rasr_cache.FileArchive]):
        if isinstance(alignment_file, tk.Path):
            f = rasr_cache.FileArchive(alignment_file.get_path())
            f.setAllophones(allophones_file.get_path())
        elif isinstance(alignment_file, rasr_cache.FileArchive):
            f = alignment_file
        else:
            raise NotImplementedError

        alignments = []
        for i, k in enumerate(f.ft):
            finfo = f.ft[k]
            if "attrib" not in finfo.name:
                alignment = [(f.allophones[mix], state) for time, mix, state, _ in f.read(finfo.name, "align")]
                alignments.append(alignment)
        return alignments

    @staticmethod
    def count(alignment, silence_symbol):
        silence_frames = list()
        total_frames = list()
        for cur_label, val in itertools.groupby(alignment):
            v_len = len(list(val))
            if cur_label[0] == silence_symbol:
                print("silence:", cur_label, v_len)
                silence_frames.append(v_len)

            print("frame:", cur_label, v_len)
            total_frames.append(v_len)

        return sum(silence_frames), sum(total_frames)

    def run(self):
        logging.info("Collection statistics...")
        if isinstance(self.alignment_files, List):
            num_alignments = len(self.alignment_files)
            alignment_files = self.alignment_files
        elif isinstance(self.alignment_files, tk.Path):
            if self.alignment_files.get_path().endswith(".bundle"):
                bundle_file = rasr_cache.FileArchiveBundle(self.alignment_files.get_path())
                bundle_file.setAllophones(self.allophone_file.get_path())
                alignment_files = list(bundle_file.archives.values())
                num_alignments = len(alignment_files)
            else:
                alignment_files = [self.alignment_files]
                num_alignments = 1
        else:
            raise NotImplementedError

        silence_frames = []
        total_frames = []

        for i, ap in enumerate(alignment_files, start=1):
            logging.info(f"Alignment: {i}/{num_alignments}")
            alignment_content = self.load_alignment(self.allophone_file, ap)
            for align in alignment_content:
                seg_silence_frames, seg_total_frames = self.count(align, self.silence_label)
                silence_frames.append(seg_silence_frames)
                total_frames.append(seg_total_frames)

        sum_silence_frames = sum(silence_frames)
        sum_total_frames = sum(total_frames)

        self.out_silence_frames.set(sum_silence_frames)
        self.out_total_frames.set(sum_total_frames)
        self.out_silence_ratio.set(sum_silence_frames / sum_total_frames)


class ComputeAveragePhonemeLengthJob(Job):
    def __init__(
        self,
        allophone_file: tk.Path,
        alignment_files: Union[tk.Path, List[tk.Path]],
        silence_label: str,
        non_speech_labels: Union[str, List[str]],
    ):
        self.allophone_file = allophone_file
        self.alignment_files = alignment_files
        self.silence_label = silence_label
        self.non_speech_labels = non_speech_labels

        self.out_average_phoneme_length = self.output_var("average_phoneme_length.txt")
        self.out_average_phoneme_length_state_0 = self.output_var("average_phoneme_length_state_0.txt")
        self.out_average_phoneme_length_state_1 = self.output_var("average_phoneme_length_state_1.txt")
        self.out_average_phoneme_length_state_2 = self.output_var("average_phoneme_length_state_2.txt")
        self.out_average_phoneme_length_with_non_speech = self.output_var("average_phoneme_length_with_non_speech.txt")
        self.out_total_frames = self.output_var("total_frames.txt")
        self.out_num_seqs = self.output_var("num_seqs.txt")

        self.rqmt = {"cpu": 1, "mem": 24, "time": 24, "sbatch_args": ["-p", "cpu_slow"]}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def load_alignment(allophones_file: tk.Path, alignment_file: Union[tk.Path, rasr_cache.FileArchive]):
        if isinstance(alignment_file, tk.Path):
            f = rasr_cache.FileArchive(alignment_file.get_path())
            f.setAllophones(allophones_file.get_path())
        elif isinstance(alignment_file, rasr_cache.FileArchive):
            f = alignment_file
        else:
            raise NotImplementedError

        alignments = []
        for i, k in enumerate(f.ft):
            finfo = f.ft[k]
            if "attrib" not in finfo.name:
                alignment = [(f.allophones[mix], state) for time, mix, state, _ in f.read(finfo.name, "align")]
                alignments.append(alignment)
        return alignments

    @staticmethod
    def count(alignment, silence_symbol, non_speech_symbols):
        hmm_lengths = list()
        hmm_unk_lengths = list()
        hmm_0_lengths = list()
        hmm_1_lengths = list()
        hmm_2_lengths = list()
        hmm_num = 0
        hmm_unk_num = 0
        hmm_0_num = 0
        hmm_1_num = 0
        hmm_2_num = 0
        total_frames = list()

        for cur_label, val in itertools.groupby(alignment):
            v_len = len(list(val))
            total_frames.append(v_len)

            if cur_label[0].strip() == silence_symbol:
                continue

            hmm_unk_lengths.append(v_len)
            hmm_unk_num += 1

            if cur_label[0].strip() in non_speech_symbols:
                continue

            hmm_lengths.append(v_len)
            hmm_num += 1

            if cur_label[1] == 0:
                hmm_0_lengths.append(v_len)
                hmm_0_num += 1
            if cur_label[1] == 1:
                hmm_1_lengths.append(v_len)
                hmm_1_num += 1
            if cur_label[1] == 2:
                hmm_2_lengths.append(v_len)
                hmm_2_num += 1

        return (
            hmm_lengths,
            hmm_num,
            hmm_0_lengths,
            hmm_0_num,
            hmm_1_lengths,
            hmm_1_num,
            hmm_2_lengths,
            hmm_2_num,
            total_frames,
            hmm_unk_lengths,
            hmm_unk_num,
        )

    def run(self):
        logging.info("Collection statistics...")
        if isinstance(self.alignment_files, List):
            num_alignments = len(self.alignment_files)
            alignment_files = self.alignment_files
        elif isinstance(self.alignment_files, tk.Path):
            if self.alignment_files.get_path().endswith(".bundle"):
                bundle_file = rasr_cache.FileArchiveBundle(self.alignment_files.get_path())
                bundle_file.setAllophones(self.allophone_file.get_path())
                alignment_files = list(bundle_file.archives.values())
                num_alignments = len(alignment_files)
            else:
                alignment_files = [self.alignment_files]
                num_alignments = 1
        else:
            raise NotImplementedError

        num_seqs = 0
        hmm_lengths = 0
        hmm_unk_lengths = 0
        hmm_0_lengths = 0
        hmm_1_lengths = 0
        hmm_2_lengths = 0
        hmm_num = 0
        hmm_unk_num = 0
        hmm_0_num = 0
        hmm_1_num = 0
        hmm_2_num = 0
        total_frames = 0

        for i, ap in enumerate(alignment_files, start=1):
            logging.info(f"Alignment: {i}/{num_alignments}")
            alignment_content = self.load_alignment(self.allophone_file, ap)
            for align in alignment_content:
                num_seqs += 1
                (
                    seg_hmm_lengths,
                    seg_hmm_num,
                    seg_hmm_0_lengths,
                    seg_hmm_0_num,
                    seg_hmm_1_lengths,
                    seq_hmm_1_num,
                    seg_hmm_2_lengths,
                    seg_hmm_2_num,
                    seg_total_frames,
                    seg_hmm_unk_lengths,
                    seg_hmm_unk_num,
                ) = self.count(
                    align,
                    self.silence_label,
                    self.non_speech_labels,
                )
                hmm_lengths += sum(seg_hmm_lengths)
                hmm_0_lengths += sum(seg_hmm_0_lengths)
                hmm_1_lengths += sum(seg_hmm_1_lengths)
                hmm_2_lengths += sum(seg_hmm_2_lengths)
                hmm_unk_lengths += sum(seg_hmm_unk_lengths)
                hmm_num += seg_hmm_num
                hmm_0_num += seg_hmm_0_num
                hmm_1_num += seq_hmm_1_num
                hmm_2_num += seg_hmm_2_num
                hmm_unk_num += seg_hmm_unk_num
                total_frames += sum(seg_total_frames)

        self.out_average_phoneme_length.set(hmm_lengths / hmm_num)
        self.out_average_phoneme_length_with_non_speech.set(hmm_unk_lengths / hmm_unk_num)
        self.out_average_phoneme_length_state_0.set(hmm_0_lengths / hmm_0_num)
        self.out_average_phoneme_length_state_1.set(hmm_1_lengths / hmm_1_num if hmm_1_num > 0 else 0)
        self.out_average_phoneme_length_state_2.set(hmm_2_lengths / hmm_2_num if hmm_2_num > 0 else 0)
        self.out_total_frames.set(total_frames)
        self.out_num_seqs.set(num_seqs)


class ComputeLabelStatisticsJob(Job):
    def __init__(
        self,
        allophone_file: tk.Path,
        alignment_files: Union[tk.Path, List[tk.Path]],
        silence_label: str,
        non_speech_labels: Union[str, List[str]],
    ):
        self.allophone_file = allophone_file
        self.alignment_files = alignment_files
        self.silence_label = silence_label
        self.non_speech_labels = non_speech_labels

        self.out_counts = self.output_var("counts.pickle", pickle=True)
        self.out_label_lengths = self.output_var("label_lengths.txt")
        self.out_statistics = self.output_path("statistics.txt")
        self.out_silence_begin_histogram = self.output_path("silence_begin_histogram.png")
        self.out_silence_intra_histogram = self.output_path("silence_intra_histogram.png")
        self.out_silence_end_histogram = self.output_path("silence_end_histogram.png")
        self.out_hmm_histogram = self.output_path("hmm_label_histogram.png")
        self.out_hmm_0_histogram = self.output_path("hmm_0_histogram.png")
        self.out_hmm_1_histogram = self.output_path("hmm_1_histogram.png")
        self.out_hmm_2_histogram = self.output_path("hmm_2_histogram.png")
        self.out_phoneme_histogram = self.output_path("label_histogram.png")

        self.rqmt = {"cpu": 1, "mem": 24, "time": 24, "sbatch_args": ["-p", "cpu_slow"]}

    def tasks(self):
        yield Task("count", rqmt=self.rqmt)
        yield Task("plot", rqmt=self.rqmt)

    @staticmethod
    def load_alignment(allophones_file: tk.Path, alignment_file: Union[tk.Path, rasr_cache.FileArchive]):
        if isinstance(alignment_file, tk.Path):
            f = rasr_cache.FileArchive(alignment_file.get_path())
            f.setAllophones(allophones_file.get_path())
        elif isinstance(alignment_file, rasr_cache.FileArchive):
            f = alignment_file
        else:
            raise NotImplementedError

        alignments = []
        for i, k in enumerate(f.ft):
            finfo = f.ft[k]
            if "attrib" not in finfo.name:
                alignment = [(f.allophones[mix], state) for time, mix, state, _ in f.read(finfo.name, "align")]
                alignments.append(alignment)
        return alignments

    @staticmethod
    def count_hmm_states(alignment):
        label_lengths = list()
        for cur_label, val in itertools.groupby(alignment):
            label_lengths.append(tuple([cur_label, len(list(val))]))
        return label_lengths

    @staticmethod
    def count_merge_hmm_states(alignment):
        label_lengths = list()
        for cur_label, val in itertools.groupby(alignment, key=lambda t: t[0]):
            label_lengths.append(tuple([cur_label, len(list(val))]))
        return label_lengths

    @staticmethod
    def count_hmm_lengths(count, silence_symbol):
        hmm_lengths = []
        hmm0_lengths = []
        hmm1_lengths = []
        hmm2_lengths = []
        for label, length in count:
            if isinstance(silence_symbol, str):
                if label[0].strip().find(silence_symbol) == 0:
                    continue
            elif isinstance(silence_symbol, List):
                for s in silence_symbol:
                    if label[0].strip().find(s) == 0:
                        continue
            else:
                raise NotImplementedError

            hmm_lengths.append(length)
            if label[1] == 0:
                hmm0_lengths.append(length)
            if label[1] == 1:
                hmm1_lengths.append(length)
            if label[1] == 2:
                hmm2_lengths.append(length)
        return hmm_lengths, hmm0_lengths, hmm1_lengths, hmm2_lengths

    @staticmethod
    def count_phon_lengths(count, silence_symbol):
        phon_lengths = []
        for label, length in count:
            if isinstance(silence_symbol, str):
                if label.strip().find(silence_symbol) == 0:
                    continue
            elif isinstance(silence_symbol, List):
                for s in silence_symbol:
                    if label.strip().find(s) == 0:
                        continue
            else:
                raise NotImplementedError

            phon_lengths.append(length)
        return phon_lengths

    @staticmethod
    def count_silence(count, silence_symbol):
        if count[0][0][0].strip().find(silence_symbol) == 0:
            sil_begin_length = count.pop(0)[1]
        else:
            sil_begin_length = 0
        if count[-1][0][0].strip().find(silence_symbol) == 0:
            sil_end_length = count.pop(-1)[1]
        else:
            sil_end_length = 0
        sil_middle_length = []
        for label, length in count:
            if label[0].strip().find(silence_symbol) == 0:
                sil_middle_length.append(length)
        return [sil_begin_length], sil_middle_length, [sil_end_length]

    @staticmethod
    def hist_data_to_dataframe(x_label, y_label, data_dict):
        import pandas

        d_t = defaultdict(list)
        for k, v in sorted(data_dict.items()):
            d_t[x_label].append(k)
            d_t[y_label].append(v)

        df = pandas.DataFrame(data=d_t)

        return df

    def count(self):
        logging.info("Collection statistics...")
        if isinstance(self.alignment_files, List):
            num_alignments = len(self.alignment_files)
            alignment_files = self.alignment_files
        elif isinstance(self.alignment_files, tk.Path):
            if self.alignment_files.get_path().endswith(".bundle"):
                bundle_file = rasr_cache.FileArchiveBundle(self.alignment_files.get_path())
                bundle_file.setAllophones(self.allophone_file.get_path())
                alignment_files = list(bundle_file.archives.values())
                num_alignments = len(alignment_files)
            else:
                alignment_files = [self.alignment_files]
                num_alignments = 1
        else:
            raise NotImplementedError

        labels_counter = defaultdict(list)
        labels_merged_counter = defaultdict(list)
        silence_begin_counter = list()
        silence_intra_counter = list()
        silence_end_counter = list()
        silence_begin_histogram = Counter()
        silence_end_histogram = Counter()
        silence_intra_histogram = Counter()
        hmm_histogram = Counter()
        hmm_0_histogram = Counter()
        hmm_1_histogram = Counter()
        hmm_2_histogram = Counter()
        phoneme_histogram = Counter()
        phoneme_counter = list()
        hmm_counter = list()
        hmm_0_counter = list()
        hmm_1_counter = list()
        hmm_2_counter = list()

        idx = 0
        for i, ap in enumerate(alignment_files, start=1):
            logging.info(f"Alignment: {i}/{num_alignments}")
            alignment_content = self.load_alignment(self.allophone_file, ap)
            for align in alignment_content:
                labels_counter[idx] = count = self.count_hmm_states(align)
                labels_merged_counter[idx] = count_merge = self.count_merge_hmm_states(align)

                sil_begin, sil_intra, sil_end = self.count_silence(count, self.silence_label)

                silence_begin_counter.extend(sil_begin)
                silence_intra_counter.extend(sil_intra)
                silence_end_counter.extend(sil_end)
                silence_begin_histogram[sil_begin[0]] += 1
                silence_end_histogram[sil_end[0]] += 1
                for ii in sil_intra:
                    silence_intra_histogram[i] += 1

                hmm_lengths, hmm0_lengths, hmm1_lengths, hmm2_lengths = self.count_hmm_lengths(
                    count, self.non_speech_labels
                )
                for hmm in hmm_lengths:
                    hmm_histogram[hmm] += 1
                for h0 in hmm0_lengths:
                    hmm_0_histogram[h0] += 1
                for h1 in hmm1_lengths:
                    hmm_1_histogram[h1] += 1
                for h2 in hmm2_lengths:
                    hmm_2_histogram[h2] += 1

                phon_lengths = self.count_phon_lengths(count_merge, self.non_speech_labels)
                for phon in phon_lengths:
                    phoneme_histogram[phon] += 1

                phoneme_counter.extend(phon_lengths)
                hmm_counter.extend(hmm_lengths)
                hmm_0_counter.extend(hmm0_lengths)
                hmm_1_counter.extend(hmm1_lengths)
                hmm_2_counter.extend(hmm2_lengths)

                idx += 1

        del alignment_content, alignment_files

        results = [
            labels_counter,
            labels_merged_counter,
            silence_begin_counter,
            silence_intra_counter,
            silence_end_counter,
            silence_begin_histogram,
            silence_intra_histogram,
            silence_end_histogram,
            hmm_counter,
            hmm_0_counter,
            hmm_1_counter,
            hmm_2_counter,
            phoneme_counter,
            hmm_histogram,
            hmm_0_histogram,
            hmm_1_histogram,
            hmm_2_histogram,
            phoneme_histogram,
        ]

        logging.info("Pickling...")
        self.out_counts.set(tuple(results))

    def plot(self):
        logging.info("Unpickling")
        (
            labels_counter,
            labels_merged_counter,
            silence_begin_counter,
            silence_intra_counter,
            silence_end_counter,
            silence_begin_histogram,
            silence_intra_histogram,
            silence_end_histogram,
            hmm_counter,
            hmm_0_counter,
            hmm_1_counter,
            hmm_2_counter,
            phoneme_counter,
            hmm_histogram,
            hmm_0_histogram,
            hmm_1_histogram,
            hmm_2_histogram,
            phoneme_histogram,
        ) = self.out_counts.get()

        # *** stat calculation ***
        logging.info("Calculating averages")
        num_seqs = len(labels_counter.keys())
        assert num_seqs == len(silence_begin_counter), (num_seqs, len(silence_begin_counter))
        assert num_seqs == len(silence_end_counter), (num_seqs, len(silence_end_counter))
        total_num_sil = sum(silence_begin_counter) + sum(silence_intra_counter) + sum(silence_end_counter)
        avg_sil_begin = sum(silence_begin_counter) / num_seqs
        avg_sil_intra = sum(silence_intra_counter) / len(silence_intra_counter) if len(silence_intra_counter) > 0 else 0
        avg_sil_end = sum(silence_end_counter) / num_seqs
        avg_hmm = sum(hmm_counter) / len(hmm_counter)
        avg_hmm0 = sum(hmm_0_counter) / len(hmm_0_counter)
        avg_hmm1 = sum(hmm_1_counter) / len(hmm_1_counter) if len(hmm_1_counter) > 0 else 0
        avg_hmm2 = sum(hmm_2_counter) / len(hmm_2_counter) if len(hmm_2_counter) > 0 else 0
        avg_phon = sum(phoneme_counter) / len(phoneme_counter)

        total_num_frames = 0
        for _, v in labels_counter.items():
            for label, length in v:
                total_num_frames += int(length)

        with open(self.out_statistics.get_path(), "wt") as out_stats:
            out_stats.write(f"average silence length at sequence begin: {avg_sil_begin:.2f}\n")
            out_stats.write(f"average silence length intra sequence   : {avg_sil_intra:.2f}\n")
            out_stats.write(f"average silence length at sequence end  : {avg_sil_end:.2f}\n")
            out_stats.write(f"average cart label length               : {avg_hmm:.2f}\n")
            out_stats.write(f"average 0. hmm state length             : {avg_hmm0:.2f}\n")
            out_stats.write(f"average 1. hmm state length             : {avg_hmm1:.2f}\n")
            out_stats.write(f"average 2. hmm state length             : {avg_hmm2:.2f}\n")
            out_stats.write(f"average phoneme label length            : {avg_phon:.2f}\n")
            out_stats.write(f"average number of frames per sequence   : {total_num_frames / num_seqs:.2f}\n")
            out_stats.write(f"total number of silence frames          : {total_num_sil:.0f}\n")
            out_stats.write(f"total number of frames                  : {total_num_frames:.0f}\n")
            out_stats.write(f"number of sequences                     : {num_seqs:.0f}\n")

        print("creating plots")
        # *** data to pandas dataframe ***
        sil_begin_dataframe = self.hist_data_to_dataframe(
            "begin silence lengths", "occurences", silence_begin_histogram
        )
        sil_intra_dataframe = self.hist_data_to_dataframe(
            "intra silence lengths", "occurences", silence_intra_histogram
        )
        sil_end_dataframe = self.hist_data_to_dataframe("end silence lengths", "occurences", silence_end_histogram)
        cart_dataframe = self.hist_data_to_dataframe("hmm lengths", "occurences", hmm_histogram)
        hmm0_dataframe = self.hist_data_to_dataframe("hmm 0 lengths", "occurences", hmm_0_histogram)
        hmm1_dataframe = self.hist_data_to_dataframe("hmm 1 lengths", "occurences", hmm_1_histogram)
        hmm2_dataframe = self.hist_data_to_dataframe("hmm 2 lengths", "occurences", hmm_2_histogram)
        phon_dataframe = self.hist_data_to_dataframe("phon lengths", "occurences", phoneme_histogram)

        # *** plot histogram ***
        sil_begin_dataframe.plot(x="begin silence lengths", y="occurences", logy=True).get_figure().savefig(
            self.out_silence_begin_histogram.get_path()
        )
        sil_intra_dataframe.plot(x="intra silence lengths", y="occurences", logy=True).get_figure().savefig(
            self.out_silence_intra_histogram.get_path()
        )
        sil_end_dataframe.plot(x="end silence lengths", y="occurences", logy=True).get_figure().savefig(
            self.out_silence_end_histogram.get_path()
        )
        cart_dataframe.plot(x="hmm lengths", y="occurences", logy=True).get_figure().savefig(
            self.out_hmm_histogram.get_path()
        )
        hmm0_dataframe.plot(x="hmm 0 lengths", y="occurences", logy=True).get_figure().savefig(
            self.out_hmm_0_histogram.get_path()
        )
        if len(hmm_1_counter) > 0:
            hmm1_dataframe.plot(x="hmm 1 lengths", y="occurences", logy=True).get_figure().savefig(
                self.out_hmm_1_histogram.get_path()
            )
        if len(hmm_2_counter) > 0:
            hmm2_dataframe.plot(x="hmm 2 lengths", y="occurences", logy=True).get_figure().savefig(
                self.out_hmm_2_histogram.get_path()
            )
        phon_dataframe.plot(x="phon lengths", y="occurences", logy=True).get_figure().savefig(
            self.out_phoneme_histogram.get_path()
        )
