from sisyphus import Job, Task, Path
import i6_core.util as util


class TextDictToTextLinesJob(Job):
    """
    Operates on RETURNN search output (or :class:`CorpusToTextDictJob` output) and prints the values line-by-line.
    The ordering from the dict is preserved.

    TODO move this to i6_core: https://github.com/rwth-i6/i6_core/pull/501
    """

    def __init__(self, text_dict: Path, *, gzip: bool = False):
        """
        :param text_dict: a text file with a dict in python format, {seq_tag: text}
        :param gzip: if True, gzip the output
        """
        self.text_dict = text_dict
        self.out_text_lines = self.output_path("text_lines.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # nan/inf should not be needed, but avoids errors at this point and will print an error below,
        # that we don't expect an N-best list here.
        d = eval(util.uopen(self.text_dict, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> text

        with util.uopen(self.out_text_lines, "wt") as out:
            for seq_tag, entry in d.items():
                assert isinstance(entry, str), f"expected str, got {entry!r} (type {type(entry).__name__})"
                out.write(entry + "\n")


class SearchWordsDummyTimesToCTMJob(Job):
    """
    Convert RETURNN search output file into CTM format file (does not support n-best lists yet).
    Like :class:`SearchWordsToCTMJob` but does not use the Bliss XML corpus for recording names and segment times.
    Instead, this will just use dummy times.

    When creating the corresponding STM files, make sure it uses the same dummy times.

    TODO move this to i6_core.returnn.search: https://github.com/rwth-i6/i6_core/pull/503
    """

    def __init__(self, recog_words_file: Path, *, filter_tags: bool = True):
        """
        :param recog_words_file: search output file from RETURNN
        :param filter_tags: if set to True, tags such as [noise] will be filtered out
        """
        self.recog_words_file = recog_words_file
        self.filter_tags = filter_tags

        self.out_ctm_file = self.output_path("search.ctm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # nan/inf should not be needed, but avoids errors at this point and will print an error below,
        # that we don't expect an N-best list here.
        d = eval(util.uopen(self.recog_words_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "only search output file with dict format is supported"
        with util.uopen(self.out_ctm_file.get_path(), "wt") as out:
            out.write(";; <name> <track> <start> <duration> <word> <confidence> [<n-best>]\n")
            for seg_fullname, text in d.items():
                seg_start = 0.0
                seg_end = 1.0
                out.write(";; %s (%f-%f)\n" % (seg_fullname, seg_start, seg_end))
                words = text.split()
                time_step_per_word = (seg_end - seg_start) / max(len(words), 1)
                avg_dur = time_step_per_word * 0.9
                count = 0
                for i in range(len(words)):
                    if self.filter_tags and words[i].startswith("[") and words[i].endswith("]"):
                        continue
                    out.write(
                        "%s 1 %f %f %s 0.99\n"
                        % (
                            seg_fullname,  # originally the recording name, but treat each segment as a recording
                            seg_start + time_step_per_word * i,
                            avg_dur,
                            words[i],
                        )
                    )
                    count += 1
                if count == 0:
                    # sclite cannot handle empty sequences, and would stop with an error like:
                    #   hyp file '4515-11057-0054' and ref file '4515-11057-0053' not synchronized
                    #   sclite: Alignment failed.  Exiting
                    # So we make sure it is never empty.
                    # For the WER, it should not matter, assuming the reference sequence is non-empty,
                    # you will anyway get a WER of 100% for this sequence.
                    out.write(
                        "%s 1 %f %f %s 0.99\n"
                        % (
                            seg_fullname,
                            seg_start,
                            avg_dur,
                            "<empty-sequence>",
                        )
                    )
