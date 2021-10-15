import os

from sisyphus import Job, Task, tk


class SearchSPMtoWordsJob(Job):
    """
    converts SPM tokens to words in the python format dict from the returnn search
    """

    def __init__(self, search_py_output):
        """

        :param Path search_py_output: a search output file from RETURNN in python format
        """
        self.search_py_output = search_py_output
        self.out_word_search_results = self.output_path("word_search_results.py")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        d = eval(open(self.search_py_output.get_path(), "r").read())
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_word_search_results.get_path())
        with open(tk.uncached_path(self.out_word_search_results), "w") as out:
            out.write("{\n")
            for seq_tag, txt in sorted(d.items()):
                if "#" in seq_tag:
                    tag_split = seq_tag.split("/")
                    recording_name, segment_name = tag_split[2].split("#")
                    seq_tag = tag_split[0] + "/" + recording_name + "/" + segment_name
                out.write("%r: %r,\n" % (seq_tag, txt.replace(" ", "").replace("▁", " ").strip()))
            out.write("}\n")