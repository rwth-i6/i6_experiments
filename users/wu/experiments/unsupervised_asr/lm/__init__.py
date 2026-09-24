"""New in the port (no source module): the text side of phase 4a.

Lexicon / g2p / OOV, the SIL phone corpus and its uniform window, the Witten-Bell phone trigram, the
word LM and trie, and the k2 ``H . L . G`` graphs.

Deliberately import-free: the k2 child modules (``lm.lexlat_k2_official`` here, and
``model.lexlat_k2`` for the in-house graph) are run as ``python -m`` under the k2 environment,
which has no sisyphus.
"""
