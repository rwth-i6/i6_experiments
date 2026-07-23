import sys
import os

from recipe.experiments.librispeech import run_debug, run_test, run_all, run_hilmes
from sisyphus import tk
from i6_core.tools.compile import MakeJob, CMakeJob
from i6_core.tools.git import CloneGitRepositoryJob

# import qat.recipe.tools
def py():
    models, results = run_debug(filename="/u/azim.javed/experiments/training/qat/full_ctx_test.txt")
    # rasr_root = CloneGitRepositoryJob(
    #     "https://github.com/rwth-i6/rasr.git",
    #     branch="berger_dev",
    #     checkout_folder_name="rasr",
    # ).out_repository.copy()
    # rasr_root.hash_overwrite = "RASR_ROOT"

    # rasr_make_job = CMakeJob(
    #     source_folder=rasr_root,
    #     cmake_opts=[
    #         "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
    #         # "MODULE_PYTHON=1"
    #     ],  
    #     num_processes=8,
    #     mem_rqmt=4,
    # )
    # rasr_make_job.rqmt["gpu"] = 1
    # rasr_make_job.rqmt["gpu_mem"] = 24

    # # rasr_binary_path: tk.Path = rasr_make_job.out_links["binaries"]
    # rasr_binary_path = rasr_make_job.out_install_dir
    # rasr_binary_path.hash_overwrite = "RASR_BINARY_PATH"
    # tk.register_output("install_dir", rasr_make_job.out_install_dir)    

    # tk.register_output("ctc_debug_converted_model_path", converted_model.path)
    # tk.register_output("ctc_debug_nj_converted_model_path", nj_converted_model.path)
