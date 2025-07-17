from sisyphus import *

from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.lib.rasr_cache import FileArchive


class ConvertRasrFeatureCacheToHdfJob(Job):
    def __init__(self, rasr_feature_cache: tk.Path, returnn_root: tk.Path, dim: int, time_rqmt=4, mem_rqmt=4):
        self.rasr_feature_cache = rasr_feature_cache
        self.returnn_root = returnn_root
        self.dim = dim

        self.out_hdf = self.output_path("out.hdf")

        self.rqmt = {"time": time_rqmt, "mem": mem_rqmt}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy

        tag_ivec = {}

        if self.rasr_feature_cache.get_path().endswith(".bundle"):
            files = open(self.rasr_feature_cache.get_path(), "rt").readlines()
        else:
            files = [self.rasr_feature_cache.get_path()]

        for file in files:
            feat_cache = FileArchive(file.strip())
            keys = [str(s) for s in feat_cache.ft if not str(s).endswith(".attribs")]
            for key in keys:
                v = feat_cache.read(key, "feat")[1]
                tag_ivec[key] = numpy.asarray(v, dtype=numpy.float32)

        SimpleHDFWriter = get_returnn_simple_hdf_writer(self.returnn_root)
        writer = SimpleHDFWriter(self.out_hdf.get_path(), dim=self.dim)
        for tag, v in tag_ivec.items():
            writer.insert_batch(
                inputs=v.reshape(1, -1, self.dim),  # [B,T,D]
                seq_tag=[tag],
                seq_len=[v.shape[0]],
            )
        writer.close()
