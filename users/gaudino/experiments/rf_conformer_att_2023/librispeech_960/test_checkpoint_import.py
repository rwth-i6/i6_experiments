from sisyphus import tk
from i6_core.returnn.training import PtCheckpoint

_torch_ckpt_filename_w_lstm_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/full_w_lm_import_2023_09_07/average.pt"
new_chkpt_path = tk.Path(_torch_ckpt_filename_w_lstm_lm, hash_overwrite="torch_ckpt_w_lstm_lm")
new_chkpt = PtCheckpoint(new_chkpt_path)
