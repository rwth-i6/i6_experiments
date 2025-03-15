from sisyphus import Job


class RemoveAudioFromCorpusJob(Job):
    """
    Given a Bliss corpus, remove the audio files from it.
    This still keeps the audio segment start and end times in the XML file.

    Use cases: We need the XML file for certain jobs, but we don't need the audio files.
    In all those cases, when copying a setup to a new location,
    you might want to copy only the XML file and not the audio files.
    Examples:
        - :class:`CorpusToTxtJob` and then train BPE or so...
        - :class:`SearchWordsToCTMJob` for recognition scoring
        - :class:`CorpusToStmJob` for recognition scoring

    """

    # TODO ...
    #   actually, not sure if we really want to implement this.
    #   maybe just use CorpusToTextDictJob and then SearchWordsToCTMJob without Bliss and TextDictToStmJob?
