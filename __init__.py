"""Research-branch compatibility for existing Sisyphus job module names."""

import os as _os

if _os.environ.get("I6_ASR_RESEARCH_IMPORTS", "1") != "0":
    from .users.berger.asr_research.bootstrap import activate as _activate

    _activate()
