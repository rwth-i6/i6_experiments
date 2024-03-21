__all__ = ["LatticeToNBestListJob"]

import os
import shutil
from typing import Optional

from sisyphus import tk, Job, Task

import i6_core.rasr as rasr


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
            "time": max(crp.corpus_duration / (5.0 * self.concurrent), 0.5),
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
        config.flf_lattice_tool.network.to_lemma.links = "reduce-scores"

        # dump n-best
        config.flf_lattice_tool.network.reduce_scores.type = "reduce"
        config.flf_lattice_tool.network.reduce_scores.keys = "am lm"
        config.flf_lattice_tool.network.reduce_scores.links = "n-best"

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
