"""
Adapted from: https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/concat-seqs/concat_seqs.py

Some utilities on datasets to concatenate sequences,
(to make the average seq lengths longer; for analysis).
"""

from sisyphus import Job, Task
import sisyphus.toolkit as tk
import numpy
from typing import Optional, Dict, Any, Union

from returnn_common.datasets import Dataset
from i6_core.util import uopen


def generic_open(filename, mode="r"):
    """
    Wrapper around :func:`open`.
    Automatically wraps :func:`gzip.open` if filename ends with ``".gz"``.

    :param str filename:
    :param str mode: text mode by default
    :rtype: typing.TextIO|typing.BinaryIO
    """
    if filename.endswith(".gz"):
        import gzip

        if "b" not in mode:
            mode += "t"
        return gzip.open(filename, mode)
    return open(filename, mode)


class ConcatDatasetSeqsJob(Job):
    """
    Based on a STM file, create concatenated dataset.
    """

    __sis_hash_exclude__ = {"shuffle_rec_seqs": False}

    def __init__(self, corpus_name, stm, num, overlap_dur="0.01", shuffle_rec_seqs=False):
        """
        :param str corpus_name: used for tag corpus name e.g. "hub5_00"
        :param Path stm: path to stm file
        :param int num: Concatenate `num` consecutive seqs within a recording
        :param str|None overlap_dur: allow overlap between consecutive seqs, in seconds
        :param bool shuffle_rec_seqs: shuffle the seqs within the same recording (e.g to scramble LM context)
        """
        self.corpus_name = corpus_name
        self.stm = stm
        self.num = num
        self.overlap_dur = overlap_dur
        self.shuffle_rec_seqs = shuffle_rec_seqs
        self.out_orig_seq_tags = self.output_path("orig_seq_tags.txt")
        self.out_orig_seq_lens = self.output_path("orig_seq_lens.txt")  # in secs
        self.out_orig_seq_lens_py = self.output_path("orig_seq_lens.py.txt")  # in secs
        self.out_concat_seq_tags = self.output_path("concat_seq_tags.txt")  # concat format, joined with ";"
        self.out_concat_seq_lens = self.output_path("concat_seq_lens.txt")  # in secs
        self.out_concat_seq_lens_py = self.output_path("concat_seq_lens.py.txt")  # in secs
        self.out_stm = self.output_path("concat_ref.stm")
        self.out_stm_py = self.output_path("concat_ref.py.stm")

    def run(self):
        # Example CTM:
        """
        ;; <name> <track> <start> <duration> <word> <confidence> [<n-best>]
        ;; hub5_00/en_4156a/1 (301.850000-302.480000)
        en_4156a 1 301.850000 0.283500 oh 0.99
        en_4156a 1 302.133500 0.283500 yeah 0.99
        ;; hub5_00/en_4156a/2 (304.710000-306.720000)
        en_4156a 1 304.710000 0.201000 well 0.99
        """
        import re
        from decimal import Decimal

        self_job = self
        orig_seqs = []
        concatenated_seqs = []
        print("Corpus:", self.corpus_name)
        print("Input ref STM:", self.stm)
        print("Concatenate up to %i seqs." % self.num)

        stm_py = {}

        class ConcatenatedSeq:
            def __init__(self, shuffle_rec_seqs):
                self.rec_tag = tag
                self.rec_tag2 = tag2
                self.seq_tags = [full_seq_tag]
                self.flags = flags
                self.txt_list = [txt]
                self.start = start
                self.end = end

                self.shuffle_rec_seqs = shuffle_rec_seqs

            @property
            def seq_tag(self):
                return ";".join(self.seq_tags)

            @property
            def text(self):
                return " ".join(self.txt_list)

            @property
            def num_seqs(self):
                return len(self.seq_tags)

            @property
            def duration(self):
                return self.end - self.start

            def can_add(self):
                if self.num_seqs >= self_job.num:
                    return False
                return tag == self.rec_tag

            def add(self):
                assert tag == self.rec_tag
                # assert tag2 == self.rec_tag2  -- sometimes "en_4910_B1" vs "en_4910_B" ...? just ignore
                assert flags.lower() == self.flags.lower()
                assert full_seq_tag not in self.seq_tags
                self.seq_tags.append(full_seq_tag)
                assert end > self.end
                self.end = end
                # self.txt = "%s %s" % (self.txt, txt)
                self.txt_list.append(txt)

            def write_to_stm(self):
                if self.shuffle_rec_seqs:
                    import random

                    # same shuffling because of same seed
                    random.Random(4).shuffle(self.txt_list)
                    random.Random(4).shuffle(self.seq_tags)

                # Extended STM entry:
                out_stm_file.write(';; _full_seq_tag "%s"\n' % self.seq_tag)
                # Example STM entry (one seq):
                # en_4156a 1 en_4156_A 301.85 302.48 <O,en,F,en-F>  oh yeah
                out_stm_file.write(
                    "%s 1 %s %s %s <%s>  %s\n"
                    % (self.rec_tag, self.rec_tag2, self.start, self.end, self.flags, self.text)
                )

                assert self.seq_tag not in stm_py
                stm_py[self.seq_tag] = self.text

        # Read (ref) STM file, and write out concatenated STM file.
        with generic_open(self.out_stm.get_path(), "w") as out_stm_file:
            seq_idx_in_tag = None
            last_tag = None
            last_end = None
            first_seq = True
            have_extended = False
            extended_seq_tag = None

            for line in generic_open(self.stm.get_path()).read().splitlines():
                line = line.strip()
                if not line:
                    continue
                # Example extended STM entry (added by ourselves):
                # _full_seq_tag "..."
                if line.startswith(";; _full_seq_tag "):
                    if first_seq:
                        have_extended = True
                    else:
                        assert have_extended
                    assert not extended_seq_tag  # should have used (and reset) this
                    m = re.match('^;; _full_seq_tag "(.*)"$', line)
                    assert m, "unexpected line: %r" % line
                    (extended_seq_tag,) = m.groups()
                    continue
                if line.startswith(";;"):  # comments, or other meta info
                    out_stm_file.write("%s\n" % line)
                    continue
                # Example STM entry (one seq):
                # en_4156a 1 en_4156_A 301.85 302.48 <O,en,F,en-F>  oh yeah
                m = re.match(
                    "^([a-zA-Z0-9_]+)\\s+1\\s+([a-zA-Z0-9_]+)\\s+([0-9.]+)\\s+([0-9.]+)\\s+<([a-zA-Z0-9,\\-]+)>(.*)$",
                    line,
                )
                assert m, "unexpected line: %r" % line
                tag, tag2, start_s, end_s, flags, txt = m.groups()
                txt = txt.strip()
                first_seq = False
                if txt == "ignore_time_segment_in_scoring":
                    continue
                if not txt:
                    continue
                start = Decimal(start_s)
                end = Decimal(end_s)
                assert start < end, "line: %r" % line
                if tag != last_tag:
                    seq_idx_in_tag = 1
                    last_tag = tag
                else:
                    if self.overlap_dur:
                        assert start >= last_end - Decimal(self.overlap_dur), "line: %r" % line  # allow minimal overlap
                    assert end > last_end, "line: %r" % line
                    seq_idx_in_tag += 1
                last_end = end

                if extended_seq_tag:
                    full_seq_tag = extended_seq_tag
                    extended_seq_tag = None
                else:
                    full_seq_tag = "%s/%s/%i" % (self.corpus_name, tag, seq_idx_in_tag)

                orig_seqs.append(ConcatenatedSeq(shuffle_rec_seqs=self.shuffle_rec_seqs))
                if not concatenated_seqs or not concatenated_seqs[-1].can_add():
                    if concatenated_seqs:
                        concatenated_seqs[-1].write_to_stm()
                    concatenated_seqs.append(ConcatenatedSeq(shuffle_rec_seqs=self.shuffle_rec_seqs))
                else:
                    concatenated_seqs[-1].add()
            # Finished iterating over input STM file.

            assert concatenated_seqs
            concatenated_seqs[-1].write_to_stm()
        # Finished writing the output STM file.

        def write_seq_tags(seqs, output_filename):
            """
            :param list[ConcatenatedSeq] seqs:
            :param str output_filename:
            """
            with generic_open(output_filename, "w") as f:
                for seq in seqs:
                    assert isinstance(seq, ConcatenatedSeq)
                    f.write("%s\n" % seq.seq_tag)

        def write_seq_lens(seqs, output_filename):
            """
            :param list[ConcatenatedSeq] seqs:
            :param str output_filename:
            """
            with generic_open(output_filename, "w") as f:
                for seq in seqs:
                    assert isinstance(seq, ConcatenatedSeq)
                    f.write("%s\n" % seq.duration)

        def write_seq_lens_py(seqs, output_filename):
            """
            :param list[ConcatenatedSeq] seqs:
            :param str output_filename:
            """
            with generic_open(output_filename, "w") as f:
                f.write("{\n")
                for seq in seqs:
                    assert isinstance(seq, ConcatenatedSeq)
                    f.write("%r: %s,\n" % (seq.seq_tag, seq.duration))
                f.write("}\n")

        def get_seq_lens_numpy(seqs):
            """
            :param list[ConcatenatedSeq] seqs:
            :rtype: numpy.ndarray
            """
            return numpy.array([float(seq.duration) for seq in seqs])

        def get_vector_stats(v):
            """
            :param numpy.ndarray v:
            :rtype: str
            """
            assert len(v.shape) == 1
            v = v.astype(numpy.float)
            return "#num %i, min-max %s-%s, mean %s, std %s" % (
                len(v),
                numpy.min(v),
                numpy.max(v),
                numpy.mean(v),
                numpy.std(v),
            )

        orig_seq_lens_np = get_seq_lens_numpy(orig_seqs)
        concatenated_seq_lens_np = get_seq_lens_numpy(concatenated_seqs)
        print("Original seq lens:", get_vector_stats(orig_seq_lens_np))
        print("Concatenated seq lens:", get_vector_stats(concatenated_seq_lens_np))

        write_seq_tags(orig_seqs, self.out_orig_seq_tags.get_path())
        write_seq_lens(orig_seqs, self.out_orig_seq_lens.get_path())
        write_seq_lens_py(orig_seqs, self.out_orig_seq_lens_py.get_path())
        write_seq_tags(concatenated_seqs, self.out_concat_seq_tags.get_path())
        write_seq_lens(concatenated_seqs, self.out_concat_seq_lens.get_path())
        write_seq_lens_py(concatenated_seqs, self.out_concat_seq_lens_py.get_path())

        with uopen(self.out_stm_py.get_path(), "wt") as f:
            f.write("{\n")
            for k, v in stm_py.items():
                f.write("%r: %r,\n" % (k, v))
            f.write("}\n")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 1, "time": 0.1}, mini_task=True)


class ConcatSeqsDataset(Dataset):
    def __init__(
        self,
        dataset: Union[str, Dict[str, Any]],
        seq_tags: str,
        seq_lens_py: str,
        additional_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(additional_options=additional_options)
        self.dataset = dataset
        self.seq_tags = seq_tags
        self.seq_len_py = seq_lens_py

    def as_returnn_opts(self):
        return {
            "class": "ConcatSeqsDataset",
            "dataset": self.dataset,
            "seq_ordering": "sorted_reverse",
            "seq_list_file": self.seq_tags,
            "seq_len_file": self.seq_len_py,
        }


class CreateConcatSeqsCTMAndSTMJob(Job):
    """
    Create CTM from concat seqs recognition words py file
    """

    def __init__(self, recog_words_file: tk.Path, stm_py_file: tk.Path, stm_file: tk.Path):
        self.recog_words_file = recog_words_file
        self.stm_py_file = stm_py_file
        self.stm_file = stm_file  # only used for getting the sorted seq tags

        self.out_stm_file = self.output_path("search.stm")
        self.out_ctm_file = self.output_path("search.ctm")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 1, "time": 0.1}, mini_task=True)

    def _get_seg_tags_from_stm(self):
        seg_tags = []
        with uopen(self.stm_file.get_path(), "rt") as f:
            for line in f:
                line_ = line.strip()
                if line_.startswith(";;"):
                    splits = line_.split()
                    if len(splits) < 2:
                        continue
                    if splits[1] == "_full_seq_tag":
                        assert len(splits) == 3
                        seg_tags.append(splits[2][1:-1])  # remove "
        return seg_tags

    @staticmethod
    def create_ctm(seg_tags, source_filename, target_filename):
        d = eval(uopen(source_filename, "rt").read())
        assert isinstance(d, dict), "only search output file with dict format is supported"
        assert len(d) > 0
        assert len(d) == len(seg_tags)
        with uopen(target_filename, "wt") as out:
            out.write(";; <name> <track> <start> <duration> <word> <confidence> [<n-best>]\n")
            start = 0.0
            for seg_tag in seg_tags:
                raw_text = d[seg_tag]
                out.write(";; %s (%f-%f)\n" % (seg_tag, start + 0.01, start + 0.99))
                assert isinstance(raw_text, str)
                if raw_text:
                    words = raw_text.split()
                    assert len(words) > 0
                    word_duration = 0.9 / len(words)
                    for i in range(len(words)):
                        rec_tag = seg_tag.split("/")[1]  # consistent with STM
                        out.write(
                            "%s 1 %f %f %s 0.99\n"
                            % (rec_tag, start + 0.01 + i * word_duration, word_duration, words[i])
                        )
                start += 1.0

    @staticmethod
    def create_stm(seg_tags, source_filename, target_filename):
        d = eval(uopen(source_filename, "rt").read())
        assert isinstance(d, dict), "only search output file with dict format is supported"
        assert len(d) > 0
        assert len(d) == len(seg_tags)
        with uopen(target_filename, "wt") as out:
            out.write(';; LABEL "d0" "default0" "all other segments of category 0"\n')  # TODO: specific for Ted2
            start = 0.0
            for seg_tag in seg_tags:
                raw_text = d[seg_tag]
                assert isinstance(raw_text, str)
                rec_tag = seg_tag.split("/")[1]  # consistent with STM
                out.write(';; _full_seq_tag "%s"\n' % seg_tag)
                out.write("%s 1 rec %f %f <d0>  %s\n" % (rec_tag, start + 0.01, start + 0.99, raw_text))
                start += 1

    def run(self):
        seg_tags = self._get_seg_tags_from_stm()
        self.create_ctm(
            seg_tags=seg_tags,
            source_filename=self.recog_words_file.get_path(),
            target_filename=self.out_ctm_file.get_path(),
        )
        self.create_stm(
            seg_tags=seg_tags,
            source_filename=self.stm_py_file.get_path(),
            target_filename=self.out_stm_file.get_path(),
        )
