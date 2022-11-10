from sisyphus import tk

synthetic_ogg_zip_data = {}

def add_ogg_zip(name: str, ogg_zip: tk.Path):
    global synthetic_ogg_zip_data
    synthetic_ogg_zip_data[name] = ogg_zip