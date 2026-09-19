"""fairseq user module for SAE §4a step 4: wav2vec-U 2.0 plus the SAE reverse term.

``common.user_dir`` points here instead of at fairseq's ``examples/wav2vec/unsupervised``.  Importing
this package first registers the REFERENCE task and model (by calling fairseq's own
``import_user_module`` on that directory, exactly as the reproduction does), and fairseq then
auto-imports this package's ``tasks/`` and ``models/``, which subclass them.  So an arm with the
reverse term still runs the reference ``unpaired_audio_text`` / ``wav2vec_u`` code for everything it
does not change, and both registries stay available in one process (the PER eval loads a reverse-arm
checkpoint through this same user dir).
"""

import argparse
import os

import fairseq
from fairseq.utils import import_user_module

FAIRSEQ_UNSUPERVISED_DIR = os.path.join(
    os.path.dirname(fairseq.__file__), "examples", "wav2vec", "unsupervised"
)

import_user_module(argparse.Namespace(user_dir=FAIRSEQ_UNSUPERVISED_DIR))
