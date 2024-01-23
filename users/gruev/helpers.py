__all__ = ["MergeDictFilesJob"]

import json
from sisyphus import *


class MergeDictFilesJob(Job):
    def __init__(self, input):
        self.input = input
        self.dictionary = dict()
        self.out_file = self.output_path("recognition.bundle", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        with open(self.out_file.get_path(), "wt") as out_file:
            for f in self.input:
                in_dict = eval(open(f.get_path(), "rt").read())
                self.dictionary.update(in_dict)

            # Sort by the length of the transcription
            sorted_dictionary = {k: v for (k, v) in sorted(self.dictionary.items(), key=lambda item: len(item[1]))}
            out_file.write(json.dumps(sorted_dictionary, indent=0))
