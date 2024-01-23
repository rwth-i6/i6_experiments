from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.generic_job_output import generic_job_output
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960._moh_att_2023_06_30_import import (
    map_param_func_v2,
)
from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob
from i6_core.returnn.training import Checkpoint
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import (
    MakeModel,
)
from sisyphus import tk
import returnn.frontend as rf


_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"


def _get_tf_checkpoint_path() -> tk.Path:
    """
    :return: Sisyphus tk.Path to the checkpoint file
    """
    return generic_job_output(_returnn_tf_ckpt_filename)


def convert_checkpoint():
    rf.select_backend_torch()

    print("*** Convert old model to new model")
    old_tf_ckpt_path = _get_tf_checkpoint_path()
    print(old_tf_ckpt_path)
    old_tf_ckpt = Checkpoint(index_path=old_tf_ckpt_path)
    make_model_func = MakeModel(80, 10_025, eos_label=0, num_enc_layers=12)
    converter = ConvertTfCheckpointToRfPtJob(
        checkpoint=old_tf_ckpt,
        make_model_func=make_model_func,
        map_func=map_param_func_v2,
        epoch=1,
        step=0,
    )
    converter.run()
    result = converter.out_checkpoint
    print(type(result))

convert_checkpoint()
