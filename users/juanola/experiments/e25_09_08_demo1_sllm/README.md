# DEMO1 Notes

Author: Martí Juanola

## Directory Structure

    .
    ├── aed
    │    ├── aed.py
    │    ├── learning_rate_configs.py
    │    ├── model_configs.py
    │    ├── optimizer_configs.py
    │    └── tune_eval.py
    ├── data
    │    ├── bpe.py
    │    ├── common.py
    │    ├── cv_segments.py
    │    ├── multi_proc.py
    │    └── spm.py
    ├── extra_code
    │    ├── dynamic_learning_rate.py
    │    └── optimizer.py
    ├── pytorch_networks
    │    └── conformer_aed_v1.py
    ├── recognition
    │    ├── aed
    │    │   ├── beam_search.py
    │    │   ├── callback.py
    │    │   └── forward_step.py
    │    └── torchaudio_ctc.py
    ├── training
    │    └── aed_ctc_train_step.py
    ├── default_tools.py
    ├── pipeline.py
    ├── report.py
    ├── returnn_config_helpers.py
    ├── returnn_config_serializer.py
    └── README.md