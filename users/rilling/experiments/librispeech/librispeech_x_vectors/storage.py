from sisyphus import tk

x_vector_extractions = {}

def add_x_vector_extraction(name: str, hdf: tk.Path, average=False):
    global x_vector_extractions
    x_vector_extractions[name] = {"hdf": hdf, "average": average}
