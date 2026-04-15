from sisyphus import Job, Task, tk


class ExtractSeqListFromHDFJob(Job):
    """
    Extract a sequence whitelist from one or multiple RETURNN HDF files via their ``seqTags`` dataset.
    """

    def __init__(self, hdf_files):
        self.hdf_files = hdf_files if isinstance(hdf_files, (list, tuple)) else [hdf_files]
        self.out_seq_list = self.output_path("seq_list.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import h5py

        seen = set()
        ordered_tags = []
        for hdf_file in self.hdf_files:
            with h5py.File(tk.uncached_path(hdf_file), "r") as f:
                for raw_tag in f["seqTags"][...].tolist():
                    tag = raw_tag.decode() if isinstance(raw_tag, (bytes, bytearray)) else str(raw_tag)
                    if tag in seen:
                        continue
                    seen.add(tag)
                    ordered_tags.append(tag)

        with open(self.out_seq_list.get_path(), "w", encoding="utf-8") as out:
            for tag in ordered_tags:
                out.write(tag)
                out.write("\n")
