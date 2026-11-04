"""General RASR template entrypoints. Research lives on berger_asr_research."""


def py():
    from .librispeech import py as run

    run()


def main():
    py()
