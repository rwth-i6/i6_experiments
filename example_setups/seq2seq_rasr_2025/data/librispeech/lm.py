from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from sisyphus import tk

from ...experiments.librispeech.word_transformer_lm_baseline import run_word_transformer_lm_baseline
from ...model_pipelines.transformer_lm.export import export_model
from ...model_pipelines.transformer_lm.lm_config import get_lm_config
from ...tools import ken_lm_binaries, rasr_binary_path


def get_binary_lm(lm_name: str) -> tk.Path:
    arpa_lm = get_arpa_lm_dict()[lm_name]
    return CreateBinaryLMJob(arpa_lm=arpa_lm, kenlm_binary_folder=ken_lm_binaries).out_lm


def get_arpa_lm_config(lm_name: str, lexicon_file: tk.Path, scale: float = 1.0) -> RasrConfig:
    arpa_lm = get_arpa_lm_dict()[lm_name]
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = arpa_lm
    rasr_config.map_oov_to_unk = True
    rasr_config.scale = scale

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = rasr_binary_path.join_right("lm-util.linux-x86_64-standard")
    crp.language_model_config = rasr_config
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = lexicon_file
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config


def get_transformer_lm_config(num_layers: int, vocab_file: tk.Path, lm_scale: float = 1.0) -> RasrConfig:
    model_config, checkpoint = run_word_transformer_lm_baseline(num_layers=num_layers)
    onnx_model = export_model(model_config=model_config, checkpoint=checkpoint)

    return get_lm_config(onnx_model=onnx_model, vocab_file=vocab_file, lm_scale=lm_scale)
