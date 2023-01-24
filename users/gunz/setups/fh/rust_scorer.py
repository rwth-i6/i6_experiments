__all__ = ["RecompileTfGraphJob"]

from os import path
import typing

from sisyphus import tk, Task


class RecompileTfGraphJob(tk.Job):
    """
    Recompiles a TF graph so the Rust Feature Scorer can read it.

    Somehow the feature scorer doesn't like the graphs produced by
    RETURNN's compile_tf_graph.py.
    """

    def __init__(self, meta_graph_file: typing.Union[str, tk.Path]):
        super().__init__()

        assert meta_graph_file is not None

        self.meta_graph_file = meta_graph_file

        self.out_graph = self.output_path("graph.meta")
        self.out_graph_text = self.output_path("graph.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from tensorflow.compat.v1 import MetaGraphDef
        from tensorflow.python.framework import graph_io

        mg = MetaGraphDef()
        with open(tk.uncached_path(self.meta_graph_file), "rb") as f:
            mg.ParseFromString(f.read())

        dirname = path.dirname(tk.uncached_path(self.out_graph))
        filename = path.basename(tk.uncached_path(self.out_graph))

        graph_io.write_graph(mg.graph_def, logdir=dirname, name=filename, as_text=False)

        dirname_txt = path.dirname(tk.uncached_path(self.out_graph_text))
        filename_txt = path.basename(tk.uncached_path(self.out_graph_text))

        graph_io.write_graph(
            mg.graph_def, logdir=dirname_txt, name=filename_txt, as_text=True
        )
