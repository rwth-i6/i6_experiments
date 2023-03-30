from sisyphus import tk

duration_alignments = {}

def add_duration(name: str, duration_hdf: tk.Path):
    global duration_alignments
    duration_alignments[name] = duration_hdf