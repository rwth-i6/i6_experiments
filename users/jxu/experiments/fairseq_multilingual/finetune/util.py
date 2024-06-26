__all__ = ["cache_hdf_files_locally", "ComputePriorCallback", "ReturnnForwardComputePriorJob"]

import numpy as np
import torch
import copy
import os
import subprocess as sp
from typing import Optional, Union

from i6_core import util
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint, PtCheckpoint
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import Job, Task, tk


def cache_hdf_files_locally(dataset):
    import getpass, numpy.random, os, shutil, time

    assert dataset["class"] == "HDFDataset"
    local_filenames = []

    if os.path.exists("/ssd"):
        local_root = "/ssd"
    else:
        local_root = "/localdata"
    time.sleep(numpy.random.uniform())
    for filename in dataset["files"]:
        try:
            local_path = os.path.join(
                local_root, getpass.getuser(), filename.lstrip("/")
            )
            if not os.path.exists(local_path):
                print(f"caching {filename} as {local_path}.")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.copy2(filename, local_path)
            else:
                print(f"reusing local path {local_path}.")
            local_filenames.append(local_path)
        except Exception as e:
            print(f"Error caching file {filename}. Using original file.")
            print(f"Error was {e}.")
            local_filenames.append(filename)
    dataset["files"] = local_filenames
