from sisyphus import tk

from i6_core import returnn
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.users.berger.helpers import rasr_lm_config

dependency_path = tk.Path("/work/asr4/berger/dependencies/librispeech/lm", hash_overwrite="DEPDENDENCY_LBS_LM")


def get_lm(name: str) -> rasr_lm_config.LMData:
    lm_dict = {}
    for key, val in lbs_dataset.get_arpa_lm_dict().items():
        lm_dict[key] = rasr_lm_config.ArpaLMData(scale=10, filename=val, lookahead_lm=None)

    kazuki_lstm_path = dependency_path.join_right("kazuki_lstmlm_27062019")
    lm_dict["kazuki_lstm"] = rasr_lm_config.RNNLMData(
        scale=10,
        vocab_file=kazuki_lstm_path.join_right("vocabulary"),
        model_file=returnn.Checkpoint(index_path=kazuki_lstm_path.join_right("network.040.index")),
        graph_file=kazuki_lstm_path.join_right("network.040.meta"),
        lookahead_lm=rasr_lm_config.ArpaLMData(
            scale=1.0, filename=lbs_dataset.get_arpa_lm_dict()["4gram"], lookahead_lm=None
        ),
    )

    kazuki_transformer_path = dependency_path.join_right("kazuki_transformerlm_2019interspeech")
    lm_dict["kazuki_transformer"] = rasr_lm_config.TransformerLMData(
        scale=10,
        vocab_file=kazuki_transformer_path.join_right("vocabulary"),
        model_file=returnn.Checkpoint(index_path=kazuki_transformer_path.join_right("network.030.index")),
        graph_file=kazuki_transformer_path.join_right("inference.meta"),
        lookahead_lm=rasr_lm_config.ArpaLMData(
            scale=1.0, filename=lbs_dataset.get_arpa_lm_dict()["4gram"], lookahead_lm=None
        ),
    )

    return lm_dict[name]
