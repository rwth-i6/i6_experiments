from typing import Optional

import black
from i6_core.summary.wer import TableReport
from i6_core.text.processing import WriteToTextFileJob
from i6_experiments.common.setups.returnn_pytorch.serialization import build_config_constructor_serializers_v2
from sisyphus import gs, tk

from .configs import PipelineConfig
from .defaults import default_pipeline_config
from .subroutines import recog, train


def run(name: str = "default", config: Optional[PipelineConfig] = None) -> None:
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"bpe_ctc/{name}"

    if config is None:
        config = default_pipeline_config

    config_str, _ = build_config_constructor_serializers_v2(config)
    config_str = black.format_str(config_str.get(), mode=black.Mode(line_length=120))

    config_text_file = WriteToTextFileJob(content=config_str).out_file
    tk.register_output("pipeline_config.txt", config_text_file)

    train_job = train(config=config.train_config, model_config=config.model_config)

    report = TableReport("Experiment", precision=5)

    for idx, recog_config in enumerate(config.recog_configs, start=1):
        recog_config.descriptor = f"{idx:02d}_{recog_config.descriptor}"
        recog_result = recog(config=recog_config, model_config=config.model_config, train_job=train_job)
        report.add_entry(col="1 corpus", row=recog_result.descriptor, var=recog_result.corpus_name)
        report.add_entry(col="2 WER", row=recog_result.descriptor, var=recog_result.wer)
        report.add_entry(col="3 AM RTF", row=recog_result.descriptor, var=recog_result.am_rtf)
        report.add_entry(col="4 Search RTF", row=recog_result.descriptor, var=recog_result.search_rtf)
        report.add_entry(col="5 Overall RTF", row=recog_result.descriptor, var=recog_result.total_rtf)

    tk.register_report(f"bpe_ctc/{name}/report.txt", values=report, required=True)
