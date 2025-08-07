__all__ = ["get_ctc_label_scorer_config"]

from dataclasses import fields

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from sisyphus import tk

from .export import export_scorer
from .pytorch_modules import ConformerCTCMultiOutputConfig, ConformerCTCMultiOutputScorerConfig


def get_ctc_label_scorer_config(
    model_config: ConformerCTCMultiOutputConfig,
    checkpoint: PtCheckpoint,
    output_idx: int,
    prior_file: tk.Path,
    prior_scale: float = 0.0,
    blank_penalty: float = 0.0,
) -> RasrConfig:
    recog_model_config = ConformerCTCMultiOutputScorerConfig(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        output_idx=output_idx,
        prior_file=prior_file,
        prior_scale=prior_scale,
        blank_penalty=blank_penalty,
    )

    onnx_model = export_scorer(model_config=recog_model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "no-context-onnx"

    rasr_config.onnx_model = RasrConfig()

    rasr_config.onnx_model.session = RasrConfig()
    rasr_config.onnx_model.session.file = onnx_model
    rasr_config.onnx_model.session.inter_op_num_threads = 2
    rasr_config.onnx_model.session.intra_op_num_threads = 2

    rasr_config.onnx_model.io_map = RasrConfig()
    rasr_config.onnx_model.io_map.input_feature = "encoder_state"
    rasr_config.onnx_model.io_map.scores = "scores"

    return rasr_config
