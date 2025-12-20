# Returnn Modules
NETWORK_MODULE = "networks.conformer_qwen_v1"
TRAIN_STEP_MODULE = "training.train_step"
RECOGNITION_PACKAGE = "recognition"


# Returnn External Data
DATA_PARAM_NAME = "data"
CLASSES_PARAM_NAME = "classes"


# Sisyphus Paths
## Outputs
SIS_OUTPUTS_REPORTS = "reports"
SIS_OUTPUTS_EXP_REPORTS = f"{SIS_OUTPUTS_REPORTS}/exp_reports"

## Aliases
SIS_ALIASES_REPORTS = "reports"

## Other
SIS_BASE_REPORT_EXTENSION = "txt"