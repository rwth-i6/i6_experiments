"""personaplex_finetune_launcher.py -- SCAFFOLD entry point for PersonaPlex finetuning.

Mirrors ``moshi_finetune_launcher.py``: it would import the PersonaPlex training fork's
``train`` module, swap in our arrow-capable data loader (``moshi_arrow_dataset``), install
the ``ArrowDataConfig`` sidecar, then run the trainer. PersonaPlex ships an *inference*
fork; until a PersonaPlex-aware training loop is wired (see
``projects/2026-01-speech-llm/personaplex.md`` for the effort estimate + the two likely
outcomes), this raises so a misconfigured ``SpeechFinetune(adapter=PERSONAPLEX_ADAPTER)``
run fails loudly instead of silently training the wrong model.

To finalize: point the import at the real PersonaPlex training module, confirm it exposes
``build_data_loader`` (or adapt the monkeypatch in ``moshi_finetune_launcher``), then
delete the NotImplementedError.
"""

import sys


def main():
    raise NotImplementedError(
        "PersonaPlex finetuning is scaffolded but not implemented: no PersonaPlex-aware "
        "training loop is wired yet. See projects/2026-01-speech-llm/personaplex.md. "
        f"(invoked with argv={sys.argv[1:]})"
    )


if __name__ == "__main__":
    main()
