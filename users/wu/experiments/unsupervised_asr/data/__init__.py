"""New in the port (package marker, no source counterpart): the data side of phase 4a.

LibriSpeech audio as the common i6 ogg zips (:mod:`.librispeech`, :mod:`.ogg_zip`), pinned Hub
snapshots (:mod:`.hf_hub`: MFA alignments, wav2vec2 weights), the 10 h seed id
list (``seed_10h_ids.txt``), the CV holdout (:mod:`.splits`), the joint rVAD streams the training
reads (:mod:`.vad`, :mod:`.vad_port`), the speaker vector eta (:mod:`.speaker`) and the MFA gold
phones (:mod:`.gold`).
"""
