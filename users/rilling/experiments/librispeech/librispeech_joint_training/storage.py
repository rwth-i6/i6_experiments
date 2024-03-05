from sisyphus import tk

synthetic_ogg_zip_data = {}

def add_ogg_zip(name: str, ogg_zip: tk.Path):
    global synthetic_ogg_zip_data
    synthetic_ogg_zip_data[name] = ogg_zip
    
duration_alignments = {}

def add_duration(name: str, duration_hdf: tk.Path):
    global duration_alignments
    duration_alignments[name] = duration_hdf