from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from i6_core.returnn import Checkpoint, PtCheckpoint
from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from sisyphus import tk

from ...data.base import UppercaseARPAFileJob
from ...experiments.librispeech import bpe_lstm_lm, word_transformer_lm
from ...model_pipelines.lstm_lm.label_scorer_config import get_lstm_lm_label_scorer_config
from ...model_pipelines.transformer_lm.export import export_model
from ...model_pipelines.transformer_lm.lm_config import get_lm_config
from ...tools import rasr_binary_path


def get_arpa_lm_config(lm_name: str, lexicon_file: tk.Path, scale: float = 1.0) -> RasrConfig:
    arpa_lm = get_arpa_lm_dict()[lm_name]
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = arpa_lm
    rasr_config.scale = scale

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = rasr_binary_path.join_right("lm-util.linux-x86_64-standard")  # type: ignore
    crp.language_model_config = rasr_config  # type: ignore
    crp.lexicon_config = RasrConfig()  # type: ignore
    crp.lexicon_config.file = lexicon_file  # type: ignore
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config


def get_tedlium2_arpa_lm_config(lm_name: str, lexicon_file: tk.Path, scale: float = 1.0) -> RasrConfig:
    lm_dict = {"4gram": get_corpus_data_inputs()["dev"]["dev"].lm["filename"]}
    arpa_lm = lm_dict[lm_name]
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = UppercaseARPAFileJob(arpa_lm).out_arpa_file
    rasr_config.scale = scale

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = rasr_binary_path.join_right("lm-util.linux-x86_64-standard")
    crp.language_model_config = rasr_config
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = lexicon_file
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config


def get_transformer_lm_config(lm_scale: float = 1.0) -> RasrConfig:
    # train_job, model_config = word_transformer_lm.run_training()
    # checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
    # vocab_file = get_lm_vocab(output_prefix="").vocab
    model_config = word_transformer_lm.get_model_config()
    checkpoint = PtCheckpoint(
        tk.Path(
            "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.WuilWP7i1fS2/output/models/epoch.030.pt"
        )
    )
    onnx_model = export_model(model_config=model_config, checkpoint=checkpoint)
    vocab_file = tk.Path(
        "/work/asr4/berger/dependencies/librispeech/lm/kazuki_transformerlm_2019interspeech/vocabulary"
    )

    return get_lm_config(
        onnx_model=onnx_model, vocab_file=vocab_file, lm_scale=lm_scale, execution_provider_type="cuda"
    )


def get_kazuki_trafo_lm_config(lm_scale: float = 1.0) -> RasrConfig:
    dependency_path = tk.Path("/work/asr4/berger/dependencies/librispeech/lm", hash_overwrite="DEPDENDENCY_LBS_LM")
    kazuki_transformer_path = dependency_path.join_right("kazuki_transformerlm_2019interspeech")

    config = RasrConfig()
    config.type = "simple-transformer"
    config.scale = lm_scale
    config.vocab_file = kazuki_transformer_path.join_right("vocabulary")
    config.transform_output_negate = True
    config.vocab_unknown_word = "<UNK>"

    config.loader = RasrConfig()
    config.loader.type = "meta"
    config.loader.meta_graph_file = kazuki_transformer_path.join_right("inference.meta")
    config.loader.saved_model_file = Checkpoint(index_path=kazuki_transformer_path.join_right("network.030.index"))

    config.input_map = RasrConfig()
    config.input_map.info_0 = RasrConfig()
    config.input_map.info_0.param_name = "word"
    config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
    config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

    config.output_map = RasrConfig()
    config.output_map.info_0 = RasrConfig()
    config.output_map.info_0.param_name = "softmax"
    config.output_map.info_0.tensor_name = "output/output_batch_major"

    return config


def get_bpe_lstm_label_scorer_config(bpe_size: int = 128) -> RasrConfig:
    assert bpe_size == 128
    model_config = bpe_lstm_lm.get_model_config(bpe_size=bpe_size)

    lstm_lm_checkpoint = PtCheckpoint(
        tk.Path(
            "/work/asr4/rossenbach/sisyphus_work_folders/tts_decoder_asr_work/i6_core/returnn/training/ReturnnTrainingJob.EuWaxahLY8Ab/output/models/epoch.300.pt"
        )
    )

    return get_lstm_lm_label_scorer_config(model_config=model_config, checkpoint=lstm_lm_checkpoint)
