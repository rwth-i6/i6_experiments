from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob

from i6_experiments.users.rilling.speakers.pooling import AverageXVectorSpeakerEmbeddingsJob

def x_vector_training(config, returnn_exe, returnn_root, prefix, num_epochs=65):

    train_job = ReturnnTrainingJob(
        config,
        log_verbosity=5,
        num_epochs=num_epochs,
        time_rqmt=100,
        mem_rqmt=10,
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
)
    train_job.add_alias(prefix + "/training")
    tk.register_output(prefix + "/training.models", train_job.out_model_dir)

    return train_job

def x_vector_forward(checkpoint, config, returnn_exe, returnn_root, prefix):
    hdf_outputs = []

    last_forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=hdf_outputs,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        mem_rqmt=20
    )


    forward_prefix = prefix + "/forward"

    last_forward_job.add_alias(forward_prefix)
    
    tts_hdf = last_forward_job.out_hdf_files["output.hdf"]
    tk.register_output(forward_prefix, tts_hdf)

    pooling_job = AverageXVectorSpeakerEmbeddingsJob(
        x_vector_hdf=tts_hdf,
        returnn_root=returnn_root
    )

    pooled_hdf = pooling_job.out_hdf
    tk.register_output(forward_prefix + ".average", pooled_hdf)

    return pooled_hdf
