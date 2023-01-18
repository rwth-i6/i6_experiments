"""
As Switchboard can not be downloaded automatically, paths to the LDC packages have to be provided here
"""
from sisyphus import tk

# provide path to a folder containing the LDC97S62 contents
SWITCHBOARD1_PATH = tk.Path(
    "/u/corpora/speech/switchboard_ldc/switchboard-1",
    hash_overwrite="LDC97S62-release2",
)

# provide path to a folder containing the LDC2002S09 contents
HUB5E00_SPH_PATH = tk.Path(
    "/u/corpora/speech/switchboard_ldc/hub5e_00", hash_overwrite="LDC2002S09"
)

# provide path to a folder containing the LDC2002T43 contents
HUB5E00_TRANSCRIPT_PATH = tk.Path(
    "/u/corpora/speech/switchboard_ldc/2000_hub5_eng_eval_tr",
    hash_overwrite="LDC2002T43",
)

# provide path to a folder containing the LDC2002S13 contents
HUB5E01_PATH = tk.Path(
    "/u/corpora/speech/switchboard_ldc/hub5e_01", hash_overwrite="LDC2002S13"
)

# provide path to a folder containing the LDC2007S10 contents
RT03S_PATH = tk.Path(
    "/u/corpora/speech/switchboard_ldc/rt03s", hash_overwrite="LDC2007S10"
)

# do not edit, only for i6 internal usage
SWITCHBOARD1_LEGACY_PATH = tk.Path(
    "/u/corpora/speech/switchboard-1/audio", hash_overwrite="Switchboard-i6"
)
