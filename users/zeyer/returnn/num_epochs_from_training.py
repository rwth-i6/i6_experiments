from i6_core.returnn.training import ReturnnTrainingJob


def get_num_epochs_from_returnn_training_job(job: ReturnnTrainingJob) -> int:
    """
    Get num epochs from returnn training job
    """
    return max(job.out_checkpoints)
