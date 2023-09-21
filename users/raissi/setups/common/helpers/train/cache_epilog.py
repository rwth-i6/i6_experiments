__all__ = ["hdf_dataset_cache_epilog"]

from textwrap import dedent

hdf_dataset_cache_epilog = dedent(
    """
    import importlib
    import logging
    import sys

    sys.path.append("/usr/local/")
    sys.path.append("/usr/local/cache-manager/")

    cm = importlib.import_module("cache-manager")

    def cache(dataset):
        clazz = dataset["class"].lower()
        if clazz in ["nextgenhdfdataset", "hdfdataset"]:
            num = len(dataset["files"])
            logging.info(f"caching {num} files...")
            dataset["files"] = [cm.cacheFile(f) for f in dataset["files"]]
        elif clazz in ["metadataset", "combineddataset"]:
            # Recurse into the sub datasets

            for sub_dataset in dataset["datasets"].values():
                cache(sub_dataset)
        else:
            # Nothing to cache here.
            pass

    for dataset in [dev, train]:
        cache(dataset)
    """
)
