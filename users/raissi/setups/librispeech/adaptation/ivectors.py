import tempfile
import shutil
import copy

from recipe.i6_core.lib.rasr_cache import FileArchive, FileArchiveBundle

"""
Example for what is needed for using Markus Kitza's ivectors
hashes = {
    "train-other-960": "ocOuYrlEEf8I",
    "dev-clean": "3lDwL6dcmaF7",
    "dev-other": "khoN9Q6XiGKT",
    "test-clean": "kJeXTWCjxoIT",
    "test-other": "o6t1A4uFg6UG",

}

k = "test-clean"
prepath = "/u/formerstaff/kitza/setups/sisyphus/librispeech/2019-03-chris-960h/work/adaptation/ivector/"
IVEC_SRC_PREPATH = f"{prepath}/IVectorExtractionJob.{hashes[k]}/output"
IVEC_DEST_PREPATH = f"/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/ivectors/{k}"

for i in range(1, 50):
    transform_ivec(IVEC_SRC_PREPATH, IVEC_DEST_PREPATH, k, i)
"""



def transform_ivec(ivec_source_path: str, ivec_dets_path: str, prename: str, cache_n: int):
    """
    This function follows the naming scheme in the new recipes
    ivec_source_path: The output folder of the ivectors job
    ivec_dets_path: where the ivectors are going to be stored
    prename: different subsets of the dataset, e.g. dev-clean, dev other etc

    """
    ivec_filename = f"ivec.{cache_n}"
    # current
    ivec = FileArchive(f"{ivec_source_path}/{ivec_filename}")
    # new
    tmp_ivec_file = tempfile.mktemp()
    out = FileArchive(tmp_ivec_file)

    for f in ivec.ft.keys():
        if f.endswith("attribs"):
            continue
        # read the features
        ft_ivecs = ivec.read(f, "feat")
        ivector = ft_ivecs[1]
        # new name
        rec = f.split("/")[1]
        seg = f.split("/")[-1]
        new_name = f"{prename}/{rec}-{seg}/{rec}-{seg}"
        # add to the features
        out.addFeatureCache(new_name, ivector, [[0.0, 999999.0]])
    out.finalize()
    del out
    shutil.move(tmp_ivec_file.format(), f"{ivec_dets_path}/{ivec_filename}")




