from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob

def py():
    torch_mem_clone_job = CloneGitRepositoryJob(
        url="git@git.rwth-aachen.de:mlhlt/torch-memristor.git",
        commit="88af8c663fa8ce55ac3b559581081653da3e1610",
        checkout_folder_name="torch_memristor",
        branch="bene_cycle",
    )

    tk.register_output("test_output", torch_mem_clone_job.out_repository)
