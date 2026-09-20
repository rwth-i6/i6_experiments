"""
Cross-lingual text injection: adapt the English "winner" to German with 1 h / 10 h of German audio.

The experiment: the winner's acoustic prior is a *frozen per-phone mean log-mel table* (~3,000
numbers: 41 phones x 80 mel). This asks whether that prior can be rebuilt for a new language from
almost no audio, letting abundant unpaired **text** do the rest. German, because MLS gives official
1 h / 9 h splits over LibriVox audiobooks -- the same domain as LibriSpeech -- so the transfer holds
acoustic domain constant and varies only language.

The arms (see the project plan; A is context, B is the claim, C is the baseline)::

    A  winner zero-shot, original vocab   -> cannot write German orthography at all
    B  English LS-960 audio + German text -> THE CLAIM
    C  German paired audio, 1 h and 10 h  -> the baseline AND the vocab control

⚠ Resource declaration: German **audio** is capped at the official 10 h. `german/train` (1,966 h) is
excluded, and the `german_mfa` *acoustic model* (trained on 1,314-3,071 h of German speech) is
excluded -- our aligner is self-trained on our own <=10 h. German **text** is deliberately large
(67.06 M words). Claim "low paired audio", never "low-resource" unqualified.

Nothing here edits `users/zeyer/`: that is a hard constraint (the code is shared with other users and
with Albert's own arms), so every adaptation reaches in from outside.
"""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from sisyphus import Job, Task, tk

if TYPE_CHECKING:
    from returnn.tensor import Dim


# --------------------------------------------------------------------------------------------------
# Staged German artefacts.
#
# These were produced by hand on 2026-09-18 (MFA runs natively on RZ -- there is no linux-aarch64
# `kalpy` build, so alignment happens there and the tables are built on FZJ against RETURNN's own
# `log_mel_filterbank_from_raw`; see the project backlog S16).
#
# ⚠ Every path carries `hash_overwrite`, so the FZJ location never enters a Sisyphus hash. Without it,
# moving these files would re-hash -- and therefore re-run -- everything downstream.
# --------------------------------------------------------------------------------------------------

_STAGE = "/e/project1/spell/koch13"

# Per-phone mean log-mel, float32 [57, 80], rows in `de_phoneme_vocab` order. Validated in backlog S17.
DE_TABLE_1H = tk.Path(f"{_STAGE}/de_1h_table.npz", hash_overwrite="dorian/de/table/1h/v1")
DE_TABLE_9H = tk.Path(f"{_STAGE}/de_9h_table.npz", hash_overwrite="dorian/de/table/9h/v1")

# Per-phone median durations, in frames.
# 🔴 The 1 h durations are UNUSABLE (backlog S18): 21.8% of phones sit at the 10 ms floor because MFA
# truncated its training schedule on the undersized corpus. The 9 h durations are fine (0.4%).
# Worse, that collapse also biases the 1 h *spectra* for dynamic phones (S21) -- so the 1 h arm is the
# low point of a resource curve, not a drop-in equal of the 9 h arm. Do not quietly substitute one.
DE_DURATIONS_1H = tk.Path(f"{_STAGE}/de_1h_table_durations.npz", hash_overwrite="dorian/de/dur/1h/v1")
DE_DURATIONS_9H = tk.Path(f"{_STAGE}/de_9h_table_durations.npz", hash_overwrite="dorian/de/dur/9h/v1")

# label -> index, 57 entries, generated FROM the table's own label list so the two cannot diverge.
DE_PHONEME_VOCAB = tk.Path(f"{_STAGE}/de_phoneme_vocab.pkl", hash_overwrite="dorian/de/phonvocab/v1")

# MFA pronunciation dictionaries, G2P-extended over each corpus's OOVs. MIXED format: the original
# MFA entries are 6 tab-separated columns (`word p1 p2 p3 p4 phones`), our G2P additions are 2
# (`word phones`) -- measured on the 9 h dict: 16,053 six-column, 142,062 two-column. The
# pronunciation is the LAST column in both, which is why a fixed-column extractor is wrong on one
# half of the file whichever column it picks. See `MfaDictToLexiconTextJob`.
DE_MFA_DICT_1H = tk.Path(f"{_STAGE}/german_ext.dict", hash_overwrite="dorian/de/mfadict/1h/v1")
DE_MFA_DICT_9H = tk.Path(f"{_STAGE}/german_ext9.dict", hash_overwrite="dorian/de/mfadict/9h/v1")

# The English LS spm10k with Ä Ö Ü APPENDED -> 10,243 pieces. Produced by the checked-in generator
# `make_spm_de_ext.py` (staged beside it), which re-asserts every property on each run:
#   ids 0..10239 unchanged | new ids 10240=Ä 10241=Ö 10242=Ü | German encodes unk=0, round-trips
# ⚠ `user_defined_symbols` in the vocab opts CANNOT do this: RETURNN's SentencePieces derives
# `num_labels` from the model FILE (returnn/datasets/util/vocabulary.py:135-136).
# ⚠ The append order is load-bearing: `ExtendVocabCheckpointJob` preserves output rows 0..10239
# bit-exactly and appends 3, so row i must keep meaning piece i.
DE_SPM_EXTENDED = tk.Path(f"{_STAGE}/spm_de_ext.model", hash_overwrite="dorian/de/spm-ext-aou/v1")

# The unmodified English SPM -- what **arm A** scores against, since arm A is the winner untouched.
DE_SPM_ORIGINAL = tk.Path(
    "/e/home/jusers/koch13/jupiter/setups/2026-09-16-fzj-dlm/work/i6_core/text/label/sentencepiece"
    "/train/TrainSentencePieceJob.ofYcs4cMRS8T/output/spm_out.model",
    hash_overwrite="dorian/de/spm-original-ls10k/v1",
)
ENGLISH_SPM_SIZE = 10240
GERMAN_SPM_SIZE = ENGLISH_SPM_SIZE + len(("Ä", "Ö", "Ü"))  # 10243

# G2P pronunciations for the 1,128,974 LM-text words absent from the MFA dictionary, generated with
# the `german_mfa` G2P model (text-trained; the acoustic model stays excluded -- backlog §10).
# ⚠ ONE pronunciation per word. The raw output carried 2,299,424 lines because two Gutenberg
# onomatopoeia ("kroolookrooo...", "riririri...") produced 194,688 and 113,742 variants each and
# would have wrecked PhoneSeqGenerator's variant sampling (backlog §36).
# ⚠ 9,421 apostrophe words are ABSENT: mfa g2p drops them while reporting success. The residual-OOV
# line filter below is what keeps that from raising mid-training.
DE_G2P_DICT = tk.Path(f"{_STAGE}/de_lm_oov_g2p_1best.dict", hash_overwrite="dorian/de/g2p-lmoov-1best/v1")

# MLS German LM corpus (Gutenberg), normalised by MLS: lowercase, punctuation stripped.
# 3,809,993 lines / 67,057,476 words (verified 2026-09-19).
# ⚠ The source tarball also ships `3-gram_lm.arpa` and `5-gram_lm.arpa`. They are deliberately NOT
# extracted -- we run LM-free, and an n-gram LM appearing in the tree would silently void that claim.
DE_LM_TEXT = tk.Path(f"{_STAGE}/mls_lm/mls_lm_german/data.txt", hash_overwrite="dorian/de/lmtext/v1")

# The German phone inventory. Both the 1 h and the 9 h table are (57, 80) with *identical* label
# lists (verified 2026-09-19), so the two budgets are directly comparable.
GERMAN_PHONEME_VOCAB_SIZE = 57

# The English value `get_glow_tts_phoneme_vocab_size()` returns. Kept so the patch below can assert it
# is really patching what it thinks it is, rather than silently no-op'ing if upstream changes.
_ENGLISH_PHONEME_VOCAB_SIZE = 44


def aed_glowtts_model_def_de(*, epoch: int, in_dim: Dim, target_dim: Dim):
    """
    Albert's ``aed_glowtts_model_def``, with the pseudo-encoder's phoneme vocab set to German (57).

    **Why a wrapper rather than a copy.** The only thing German needs to change in that function is
    one number: ``:4889`` builds ``Dim(get_glow_tts_phoneme_vocab_size() * n_states)`` and
    ``glow_tts.py:59-61`` is a bare ``return 44`` with no config override and no parameter. Everything
    else on the pseudo-encoder path is already config-driven. Copying ~100 lines to change one integer
    would fork Albert's model definition permanently -- and the fork would not track his fixes, while
    still being a model we compare against his arms.

    **Why the patch works here when the plan said monkeypatching does not.** The plan's rejection was
    about *graph-build* time: ``winner_plus_tts`` patches ``DatasetConfigStatic`` there, and that only
    works because the patch mutates the config dict which is then serialised into the job. A model_def
    runs at **job-run** time, in a fresh process that re-imports Albert's module -- so a graph-time
    patch is long gone by then. This function *is* that run-time moment, so patching here is exactly
    the supported window. ``exp2026_05_28_tts_encoder_fzj`` binds the symbol as a module global
    (``:53``), and ``:4889`` reads that global, so rebinding the module attribute is picked up.

    The patch is restored in ``finally`` so nothing leaks into a later recog in the same process.

    ⚠ Hashing: ``train_v4`` hashes a model_def by module + function **name**, not content, so this
    deliberately-new name is a new hash -- which is correct, it is a different model (57-wide pseudo
    vocab). It also means editing this body will NOT re-hash a launched German arm; gate any
    behaviour change behind a new config key instead.
    """
    from i6_experiments.users.zeyer.experiments import exp2026_05_28_tts_encoder_fzj as _tts
    from returnn.config import get_global_config

    config = get_global_config()
    # Default to German rather than requiring the key. ⚠ This was an `assert size > 0` and it FIRED,
    # in the one place it must not: the RECOG. `pseudo_enc_frozen_table` / `pseudo_enc_duration_table`
    # reach a recog because `_train_tts_encoder` passes them as `model_config`, which `aed_train_exp`
    # wraps into `ModelDefWithCfg` (`aed.py:492-493`) -- a channel that is serialised into the search
    # config too. `pseudo_enc_phoneme_vocab_size` went through `extra_config_updates` instead, which is
    # **training-only**, so every German recog died at model build with 0/8 items done.
    #
    # Defaulting is safe and is NOT a silent behaviour change: this wrapper is reachable only through
    # `PatchGlowTtsToGerman`, i.e. it is German by construction, and the launched arm sets the key to
    # exactly this value -- so the model built for training is bit-identical either way, and only the
    # previously-crashing path changes. A wrong table/vocab pairing is still caught downstream by the
    # frozen-table shape assert in Albert's `aed_glowtts_model_def` (`:4947`).
    #
    # The key is still honoured when present, so a second language needs no edit here.
    size = config.int("pseudo_enc_phoneme_vocab_size", GERMAN_PHONEME_VOCAB_SIZE)
    assert size > 0, f"pseudo_enc_phoneme_vocab_size must be positive, got {size}"

    orig = _tts.get_glow_tts_phoneme_vocab_size
    # Fail loudly if upstream moved: a silent no-op here builds an English-sized model that then dies
    # in the frozen-table shape assert, ~minutes into a GPU job, looking like a data bug.
    assert orig() == _ENGLISH_PHONEME_VOCAB_SIZE, (
        f"expected the English phoneme vocab size {_ENGLISH_PHONEME_VOCAB_SIZE},"
        f" got {orig()} -- upstream changed, re-check this patch"
    )

    _tts.get_glow_tts_phoneme_vocab_size = lambda: size
    try:
        return _tts.aed_glowtts_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)
    finally:
        _tts.get_glow_tts_phoneme_vocab_size = orig


def _mirror_model_def_attribs(fn):
    """Copy the ``ModelDef`` protocol attributes off the function we delegate to.

    Mirrored rather than restated so they cannot drift: ``train_v4`` reads ``behavior_version`` /
    ``backend`` / ``batch_size_factor`` at graph-build time, and a stale hardcoded copy would change
    training semantics silently.
    """
    from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import (
        aed_glowtts_model_def as _base,
    )

    fn.behavior_version = _base.behavior_version
    fn.backend = _base.backend
    fn.batch_size_factor = _base.batch_size_factor
    return fn


def german_pseudo_enc_config(*, budget: str) -> Dict[str, Any]:
    """
    The config keys that switch the pseudo-encoder over to German, for ``extra_config_updates``.

    :param budget: "1h" or "9h" -- the German audio budget the acoustic prior was measured from.

    ⚠ ``9h`` is the usable one. ``1h`` carries a duration table that is 21.8% floor-collapsed and
    spectra that are truncation-biased for affricates/stops (backlog S18, S21); it is worth running as
    the low point of the resource curve, but it is not an equal-quality prior.
    """
    assert budget in ("1h", "9h"), f"unknown German audio budget {budget!r}"
    table = DE_TABLE_1H if budget == "1h" else DE_TABLE_9H
    durations = DE_DURATIONS_1H if budget == "1h" else DE_DURATIONS_9H
    return {
        "pseudo_enc_phoneme_vocab_size": GERMAN_PHONEME_VOCAB_SIZE,
        "pseudo_enc_frozen_table": table,
        "pseudo_enc_duration_table": durations,
    }


aed_glowtts_model_def_de = _mirror_model_def_attribs(aed_glowtts_model_def_de)


class MfaDictToLexiconTextJob(Job):
    """
    MFA pronunciation dictionary -> the 2-column text ``LexiconFromTextFileJob`` expects.

    🔴 **This job exists because the obvious wiring is silently wrong.**
    ``i6_core.lexicon.conversion.LexiconFromTextFileJob`` parses each line as
    ``line.split(None, 1)`` = orth + *everything else is phonemes*. Feeding it an MFA dict directly
    turns the pronunciation-probability columns into "phonemes", producing a lexicon full of numeric
    phones that still builds, still trains, and yields meaningless German. The same 6-column shape
    already manufactured a false "93-phone" scare on 2026-09-18.

    ⚠ **The dict is MIXED-format** (measured on ``german_ext9.dict``: 16,053 lines with 6 columns,
    142,062 with 2). The 6-column ones are the original MFA entries
    ``word ⇥ p1 p2 p3 p4 ⇥ phones``; the 2-column ones are our G2P additions ``word ⇥ phones``.
    So the pronunciation is ``parts[-1]`` in **both** cases -- which is why a fixed-column extractor
    (``cut -f2``, or ``cut -f6``) is wrong on one half of the file whichever column you pick.

    ``spn`` (spoken noise) entries are dropped: ``spn`` is deliberately absent from the 57-label
    German vocab (it was folded into ``[UNKNOWN]`` when the table was built), so a lemma carrying it
    would reference a phone the model has no embedding row for. Measured: 4 such entries, plus 2
    angle-bracket lemmas (``<cutoff>``, ``<unk>``).
    """

    def __init__(self, *, mfa_dicts, phoneme_vocab: tk.Path, version: int = 2):
        """
        :param mfa_dicts: one or more MFA dictionaries (2- or 6-column, tab separated), applied in
            order; the first pronunciation seen for a word wins, so put the curated dictionary first
            and the G2P output after it.
        :param phoneme_vocab: the pickled label->index vocab the acoustic table rows are in. Used as a
            **cross-check**, not as input: the lexicon's phone inventory must match it exactly, or the
            phonemiser would emit ids the frozen table has no row for.
        """
        self.mfa_dicts = [mfa_dicts] if isinstance(mfa_dicts, tk.Path) else list(mfa_dicts)
        self.phoneme_vocab = phoneme_vocab
        # ⚠ Code-version knob. A Sisyphus hash covers constructor ARGUMENTS, not the body of `run()`,
        # so a fix inside run() leaves the finished job untouched and its stale output in place --
        # which is exactly what happened when the special lemmas were added (§67): the lexicon, the
        # filter and arm B all kept their hashes and arm B re-ran against the OLD lexicon.
        # Bump this when run() changes in a way that alters the output; it cascades downstream.
        #   v2 = emit the four special lemmas ([space]/[start]/[end]/[UNKNOWN])
        self.version = version
        self.out_lexicon_text = self.output_path("lexicon_text.txt")
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json
        import pickle

        with open(self.phoneme_vocab.get_path(), "rb") as f:
            vocab = pickle.load(f)
        # The specials are structural: the phonemiser adds [start]/[end]/[space], [UNKNOWN] absorbs
        # spn, and [blank] is an ordinary row that CTC never emits here. None may appear in a lexicon.
        specials = {"[UNKNOWN]", "[end]", "[space]", "[start]", "[blank]"}
        expected_phones = {k for k in vocab if k not in specials}

        n_in = n_out = n_spn = n_bracket = 0
        seen_phones = set()
        written = set()

        def _all_lines():
            # Each dictionary is opened in its own `with`, so the handles close even if the consumer
            # stops early. A generator expression over bare open() leaks them.
            for d in self.mfa_dicts:
                with open(d.get_path(), "rt", encoding="utf8") as fh:
                    yield from fh

        with open(self.out_lexicon_text.get_path(), "wt", encoding="utf8") as fout:
            # 🔴 The phonemiser needs the SPECIAL lemmas, and a word-only lexicon has none:
            # `get_glow_tts_phone_info` sets `silence_lemma_orth: "[space]"` and brackets each
            # sequence with `[start]`/`[end]` phonemes, so PhoneSeqGenerator dies with
            # `KeyError: '[space]'` without them. Found by the arm-B smoke run (3rd failure).
            # The English lexicon carries exactly these four, each a lemma mapping to itself
            # (`MergeLexiconJob.P8go21pxx40e`); `[blank]` is NOT among them -- the model appends
            # blank at index V itself.
            for sym in LEXICON_SPECIAL_SYMBOLS:
                fout.write(f"{sym} {sym}\n")
                written.add(sym)
                seen_phones.add(sym)  # they ARE phonemes in the lexicon's inventory
            for line in _all_lines():
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                n_in += 1
                orth = parts[0].strip()
                phones = parts[-1].split()  # see docstring: mixed 2-/6-column, pronunciation is LAST
                if not orth or not phones:
                    continue
                if orth.startswith("<") and orth.endswith(">"):
                    n_bracket += 1
                    continue
                if "spn" in phones:
                    n_spn += 1
                    continue
                if orth.upper() in written:
                    continue  # first dictionary wins
                written.add(orth.upper())
                seen_phones.update(phones)
                # UPPERCASE, to match the injected text and the (uppercase) English SPM -- see
                # get_german_injection_text(). str.upper() maps ss<-sharp-s and the umlauts correctly,
                # and was verified to be collision-free over this dictionary (147,650 -> 147,650).
                fout.write(f"{orth.upper()} {' '.join(phones)}\n")
                n_out += 1

        # Direct guard against the failure this job exists to prevent: a probability column that got
        # through would show up as a numeric "phone".
        numeric = sorted(p for p in seen_phones if p.replace(".", "", 1).isdigit())
        assert not numeric, f"numeric pseudo-phonemes leaked from the probability columns: {numeric}"

        # The specials are legitimate lexicon phonemes (see above); everything else must be a
        # phone the acoustic table has a row for.
        expected_phones |= set(LEXICON_SPECIAL_SYMBOLS)
        extra = sorted(seen_phones - expected_phones)
        missing = sorted(expected_phones - seen_phones)
        assert not extra, f"lexicon has phones absent from the acoustic table: {extra}"
        assert not missing, f"acoustic table has phones no lexicon word uses: {missing}"
        assert n_out > 0, "empty lexicon"

        with open(self.out_stats.get_path(), "wt") as f:
            json.dump(
                {
                    "entries_in": n_in,
                    "entries_out": n_out,
                    "dropped_spn": n_spn,
                    "dropped_bracket": n_bracket,
                    "n_phones": len(seen_phones),
                    "phones": sorted(seen_phones),
                },
                f,
                indent=1,
                ensure_ascii=False,
            )


def get_german_glowtts_lexicon(*, budget: str = "9h") -> tk.Path:
    """
    Bliss lexicon (``lexicon.xml.gz``) for the German injection text, the German analogue of
    ``glow_tts.get_glow_tts_lexicon()``.

    :param budget: which MFA dictionary to use -- they differ only by the G2P pass that extended them
        over the respective corpus's OOVs. Both yield the same 52-phone inventory (verified).
    """
    from i6_core.lexicon.conversion import LexiconFromTextFileJob

    mfa_dict = DE_MFA_DICT_1H if budget == "1h" else DE_MFA_DICT_9H
    # Curated dictionary first, G2P second: a word present in both keeps the curated pronunciation.
    text = MfaDictToLexiconTextJob(mfa_dicts=[mfa_dict, DE_G2P_DICT], phoneme_vocab=DE_PHONEME_VOCAB)
    text.add_alias(f"datasets/mls-de/lexicon-text-{budget}")
    # variation="none": these are the aligner's own phone symbols, already the table's row labels.
    # "context" would add RASR context variation, which this pipeline never uses.
    return LexiconFromTextFileJob(text_file=text.out_lexicon_text, compressed=True, variation="none").out_bliss_lexicon


class PatchGlowTtsToGerman:
    """
    Context manager: while active, the GlowTTS phoneme resources resolve to **German**.

    Wrap the `_train_tts_encoder` call with it. It patches three module-level accessors on
    ``users.zeyer.external_models.glow_tts``::

        get_glow_tts_lexicon()            -> our bliss lexicon built from the MFA dict
        get_glow_tts_phoneme_vocab()      -> de_phoneme_vocab.pkl (57 labels, table row order)
        get_glow_tts_phoneme_vocab_size() -> 57

    which is enough to reach **all three** places the vocab has to land, because the two consumers are
    themselves derived (`glow_tts.py:223-255`):

    * ``get_glow_tts_phone_info()`` reads ``lexicon_file`` and ``phoneme_vocab_file`` from the first
      two -> the phonemiser uses the German lexicon and emits German ids.
    * ``get_glow_tts_phoneme_extern_data()`` reads the vocab file and the size -> ``sparse_dim`` and
      the dataset vocab come out German **by construction**, so it needs no separate override.

    ⚠ **Two different modules, and mixing them up gives a silent half-patch.**
    ``exp2026_05_28_tts_encoder_fzj:53`` does ``from ...glow_tts import get_glow_tts_phoneme_vocab_size``
    at **module import**, so that module holds its *own* binding: patching `glow_tts` does **not**
    change what its ``:4889`` sees. That call site is inside the *model_def*, i.e. **job-run** time,
    and is handled separately by :func:`aed_glowtts_model_def_de`. This context manager is the
    **graph-build** half, whose results are serialised into the job config. Both halves are required;
    neither substitutes for the other.

    ⚠ The patch is scoped, deliberately: ``fzj_dlm.py::py()`` builds the English arms in the same
    process, and a global swap would quietly give them German phonemes.

    Each accessor is counted and asserted to have fired at least once on exit -- a silent zero would
    build a German-*named* experiment on English phonemes, which trains fine and means nothing.
    """

    def __init__(self, *, budget: str = "9h"):
        self.budget = budget
        self.counts: Dict[str, int] = {}
        self._patches = []

    def __enter__(self):
        import unittest.mock

        from i6_experiments.users.zeyer.external_models import glow_tts as _gt

        lexicon = get_german_glowtts_lexicon(budget=self.budget)

        def _counted(name, value):
            def _fn(*args, **kwargs):
                self.counts[name] = self.counts.get(name, 0) + 1
                return value

            return _fn

        # Assert we are replacing what we think we are, so an upstream rename fails loudly here
        # rather than silently leaving English resources in a German config.
        assert _gt.get_glow_tts_phoneme_vocab_size() == _ENGLISH_PHONEME_VOCAB_SIZE, (
            "upstream English phoneme vocab size changed; re-check PatchGlowTtsToGerman"
        )

        for name, value in (
            ("get_glow_tts_lexicon", lexicon),
            ("get_glow_tts_phoneme_vocab", DE_PHONEME_VOCAB),
            ("get_glow_tts_phoneme_vocab_size", GERMAN_PHONEME_VOCAB_SIZE),
        ):
            p = unittest.mock.patch.object(_gt, name, _counted(name, value))
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        self._patches = []
        if exc[0] is None:
            unused = [n for n in ("get_glow_tts_lexicon", "get_glow_tts_phoneme_vocab") if not self.counts.get(n)]
            assert not unused, (
                f"German GlowTTS patch never fired for {unused} (counts={self.counts}) --"
                " the config would carry ENGLISH phoneme resources under a German experiment name"
            )
        return False


def get_german_injection_text() -> tk.Path:
    """
    The German injection text, normalised for our vocab. One line per sequence, gzipped.

    **Normalisation is ``str.upper()``** -- and the reasoning is worth keeping, because the obvious
    reading of the pipeline gives the opposite answer and it is only half right.

    The same text field feeds **two** streams (`_glowtts_text_map_seq` derives both from one read):
    the **SPM target** and the **glow-TTS phonemes**. They disagree about case in German:

    * The ASR's SPM is English LibriSpeech: **10,235 of 10,240 pieces contain uppercase**. Measured:
      ``'das scheint überhaupt'`` -> 6 tokens with **unk=3**, while ``'DAS SCHEINT UEBERHAUPT'`` ->
      12 tokens, **unk=0**. **Lowercase German is not representable in the output vocab at all.**
    * The German MFA lexicon is **entirely lowercase** (147,650 words, 0 uppercase, 0 sharp-s), so a
      lowercase lookup is what the phonemiser wants.

    Resolution: uppercase **both** sides -- the text here, and the lexicon orthography in
    :class:`MfaDictToLexiconTextJob` -- which restores exactly the English convention (uppercase text,
    uppercase lexicon). Verified equivalent: uppercasing the lexicon is **collision-free**
    (147,650 -> 147,650 forms) and the corpus OOV rate is **identical** either way (10.02%).

    ⚠ ``str.upper()`` also performs the sharp-s normalisation for free: ``'ß'.upper() == 'SS'``, which
    is the mapping we need (the MFA dictionary contains no sharp-s at all), and it is a 1->2 character
    expansion that ``tr`` **cannot** express -- so this must not be "simplified" into a tr/sed pipeline.
    Umlauts are preserved as Ä/Ö/Ü, which is why the output vocab needs those three extra symbols.

    The hyphen is mapped to a space for the same reason, and it is the **only** other offender:
    measured over a random 20,000-line sample, the characters the English SPM cannot encode are
    exactly ``Ü`` (6,819), ``Ä`` (6,135), ``Ö`` (3,962) and ``-`` (102). After uppercasing,
    hyphen->space, and adding the three umlauts to the output vocab, the residual UNK count over
    1,851,818 characters is **0** -- so the 3-symbol vocab extension is exactly sufficient, and no
    fourth symbol or byte fallback is needed.

    Scale, for the record: German costs **2.733 tokens/word** under this vocab (the plan measured
    2.758), against ~1.045 for English -- the 2.64x inflation the CTC feasibility check was run against.
    """
    from i6_core.text.processing import PipelineJob

    # No braces in the snippet: Job.sh() runs the pipeline through str.format, so a literal "{" would
    # have to be doubled (the trap documented at exp2026_05_28_tts_encoder_fzj.py:4538).
    # Uses GERMAN_NORM_SHELL, which is the shell spelling of normalize_german_text(). Spelling it
    # inline here is how the two silently drift -- the constant exists precisely to stop that.
    job = PipelineJob(DE_LM_TEXT, [GERMAN_NORM_SHELL], zip_output=True, check_equal_length=True)
    job.add_alias("datasets/mls-de/lm-text-uppercased")
    return job.out


def _set_text_branch_corpus(combined: Dict[str, Any], corpus_files) -> Dict[str, Any]:
    """Rewrite the ``text`` branch's ``LmDataset.corpus_file`` to *exactly* ``corpus_files``."""
    datasets = dict(combined["datasets"])
    text = dict(datasets["text"])
    assert text["class"] == "PostprocessingDataset", f"unexpected text branch {text['class']!r}"
    inner = dict(text["dataset"])
    assert inner["class"] == "LmDataset", f"unexpected text sub-dataset {inner['class']!r}"
    inner["corpus_file"] = list(corpus_files)
    text["dataset"] = inner
    datasets["text"] = text
    return {**combined, "datasets": datasets}


class PatchTextBranchToGerman:
    """
    Context manager: replace the text-injection corpus with German, **exactly** (no English left).

    ⚠ **Why not patch the source functions.** `_train_tts_encoder:3877-3890` builds
    ``corpus_files = [get_librispeech_normalized_lm_data(), get_train_corpus_text()]`` — a local from
    an if/elif/else, with no parameter to pass one in (correcting the plan, which calls it "the plug
    point"). Patching those two accessors is the obvious move and is **wrong both ways**: patch only
    the first and the LS-960 *transcripts* stay in the branch, so a "German text injection" arm is
    quietly part English; patch both to the same file and the corpus is concatenated with itself,
    silently doubling one epoch's text.

    So we rewrite the built dataset dict instead — the same technique, and the same defensive shape,
    as ``winner_plus_tts._PatchAsrBranchWithTts``: patch ``DatasetConfigStatic`` on its defining
    module, rewrite only the matching dataset, and assert **exactly one** rewrite on exit. A silent
    zero would train on the English LM corpus under a German experiment name — which converges
    perfectly well and answers nothing.
    """

    def __init__(self, *, corpus_files=None):
        self.corpus_files = corpus_files
        self.count = 0
        self._patch = None

    def __enter__(self):
        import unittest.mock

        from returnn_common.datasets_old_2022_10 import interface as _iface

        from .winner_plus_tts import _COMBINED_MAIN_NAME

        files = self.corpus_files if self.corpus_files is not None else [get_german_injection_text()]
        real = _iface.DatasetConfigStatic

        def _wrapped(*args, **kwargs):
            ds = kwargs.get("train_dataset")
            if (
                kwargs.get("main_name") == _COMBINED_MAIN_NAME
                and isinstance(ds, dict)
                and ds.get("class") == "CombinedDataset"
                and "text" in ds.get("datasets", {})
            ):
                kwargs = dict(kwargs, train_dataset=_set_text_branch_corpus(ds, files))
                self.count += 1
            return real(*args, **kwargs)

        self._patch = unittest.mock.patch.object(_iface, "DatasetConfigStatic", _wrapped)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        if exc[0] is None:
            assert self.count == 1, (
                f"expected to rewrite exactly 1 text branch, rewrote {self.count} --"
                " the arm would have trained on the ENGLISH LM corpus under a German name"
            )
        return False


# The three orthographic symbols German needs on top of the English SPM. Measured: after uppercasing
# and hyphen->space these are the ONLY characters the English SPM cannot encode (see
# get_german_injection_text), and adding them takes residual UNK to exactly 0.
GERMAN_EXTRA_SYMBOLS = ("Ä", "Ö", "Ü")

# Lemmas the phonemiser requires in the bliss lexicon, each mapping to itself as a phoneme. These are
# exactly what the English glow-tts lexicon carries. ⚠ `[blank]` is deliberately absent: it is not a
# lexicon symbol, the model appends it at index V.
LEXICON_SPECIAL_SYMBOLS = ("[space]", "[start]", "[end]", "[UNKNOWN]")


def normalize_german_text(line: str) -> str:
    """The ONE German text normalisation, used for BOTH the injection text and the eval references.

    `str.upper()` (which maps sharp-s -> SS) plus hyphen -> space. See
    :func:`get_german_injection_text` for why: the English SPM is uppercase-only, so lowercase German
    is not representable as a target at all, and the MFA lexicon contains no sharp-s.

    🔴 **It must be applied to the REFERENCES too.** The model can only emit uppercase, and measured on
    MLS-de `test`: **0** of 3,394 transcripts contain uppercase and **1,249 (37%)** contain a sharp-s.
    Scoring uppercase hypotheses against lowercase references would report ~100% WER; leaving sharp-s
    in the reference would mark every one of those 1,249 utterances wrong for a distinction the output
    vocabulary cannot express. Normalising both sides identically is the standard ASR practice and the
    only self-consistent choice here.

    ⚠ Disclose in the paper: WER is computed on uppercased, sharp-s-folded, hyphen-split text. Folding
    sharp-s -> ss removes a distinction the reference makes, which can only *help* the score slightly,
    so our numbers are not strictly comparable to published MLS baselines (which the plan already
    treats as context, never head-to-head).
    """
    return line.upper().replace("-", " ")


# The same transformation as a shell one-liner, for `PipelineJob`. ⚠ Keep in sync with
# :func:`normalize_german_text` -- they are two spellings of one rule and must not drift.
GERMAN_NORM_SHELL = "python3 -c \"import sys; sys.stdout.writelines(l.upper().replace('-',' ') for l in sys.stdin)\""


class ExtendVocabCheckpointJob(Job):
    """
    Offline surgery: widen a checkpoint's output vocabulary by ``len(new_symbols)`` rows.

    The winner writes German-less targets; to emit German orthography the SPM (and therefore every
    output layer) must carry Ä Ö Ü. `preload_from_files` **cannot** do this — the loader calls
    `load_state_dict(..., strict=False)` and PyTorch raises on a **shape** mismatch regardless of
    `strict`, while `ignore_params` would drop the pretrained rows entirely.

    🔴 **Read off the real checkpoint, because the plan's table was wrong three ways** (backlog 32):

    * ``enc_aux_logits_*.weight`` is ``(1024, 10241)`` — the vocab is on **axis 1**, not axis 0.
      Inserting rows on axis 0 would corrupt the encoder projection while keeping a plausible shape.
    * ``decoder.input_embedding.weight`` and ``decoder.logits.weight`` are **two keys sharing one
      storage** (``data_ptr()`` equal). Both must be widened identically or the weights silently untie.
    * ``dec_aux_logits_3.weight`` exists and the plan never mentions it.

    Hence targets are discovered **by shape**, not by name, and the discovery is asserted to cover
    every tensor mentioning the old vocab — so a future aux head cannot be missed silently.

    Two layouts, because the CTC heads carry a blank and the AED heads do not:

    * AED (``shape[0] == V``): append rows on axis 0. ``(V,1024) -> (V+n,1024)``
    * CTC (``shape[-1] == V+1``): the blank is **last** and must stay last
      (``aed.py:1189`` asserts ``blank_idx == target_dim.dimension``), so it **moves**::

          old  [0..V-1 vocab][V blank]
          new  [0..V-1 vocab][V..V+n-1 new symbols][V+n blank]
    """

    def __init__(
        self,
        *,
        checkpoint: tk.Path,
        new_symbols=GERMAN_EXTRA_SYMBOLS,
        seed: int = 0,
        pseudo_enc_table: Optional[tk.Path] = None,
    ):
        """
        :param pseudo_enc_table: the German mean-log-mel `.npz`. 🔴 **Required for the German arms.**
            The pseudo-encoder's embedding is a *different vocabulary* from the SPM output layer --
            it is indexed by PHONEMES (English 44+blank = 45 rows; German 57+blank = 58). Widening
            only the output layer leaves the English 45-row table in the checkpoint, and the run dies
            at load with::

                size mismatch for pseudo_enc.embedding.weight: copying a param with shape
                torch.Size([45, 80]) from checkpoint, the shape in current model is torch.Size([58, 80])

            Found by the 1 h smoke run (plan step 1) after graph-building passed -- graph build cannot
            see it, because the shapes only meet when the checkpoint is loaded into the model.
        """
        self.checkpoint = checkpoint
        self.new_symbols = tuple(new_symbols)
        self.seed = seed
        self.pseudo_enc_table = pseudo_enc_table
        self.out_checkpoint = self.output_path("model.pt")
        self.out_report = self.output_path("report.json")
        self.rqmt = {"cpu": 2, "mem": 24, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json

        import torch

        n = len(self.new_symbols)
        # Accept a tk.Path or an i6_core PtCheckpoint; both implement __fspath__.
        # ⚠ Do NOT sniff with `hasattr(ckpt, "path")`: tk.Path HAS a `.path` attribute and it is a
        # plain **str**, so that branch raised `'str' object has no attribute 'get_path'` -- a
        # "defensive" check that only made the failure later and stranger.
        src = os.fspath(self.checkpoint)
        sd = torch.load(src, map_location="cpu", weights_only=True)
        model = sd.get("model", sd)

        # The AED tied matrix defines the vocab size; everything else is validated against it.
        ref = "decoder.logits.weight"
        assert ref in model, f"{ref} missing -- not the expected AED model"
        v = int(model[ref].shape[0])

        aed_keys = [k for k, t in model.items() if t.ndim == 2 and int(t.shape[0]) == v]
        ctc_w = [k for k, t in model.items() if t.ndim == 2 and int(t.shape[-1]) == v + 1]
        ctc_b = [k for k, t in model.items() if t.ndim == 1 and int(t.shape[0]) == v + 1]

        # Nothing touching the vocab may escape the three buckets above.
        touched = set(aed_keys) | set(ctc_w) | set(ctc_b)
        mentions = {k for k, t in model.items() if v in tuple(t.shape) or (v + 1) in tuple(t.shape)}
        assert mentions == touched, f"unclassified vocab-sized tensors: {sorted(mentions - touched)}"
        assert aed_keys and ctc_w and ctc_b, f"empty bucket: aed={aed_keys} ctc_w={ctc_w} ctc_b={ctc_b}"

        g = torch.Generator().manual_seed(self.seed)

        def _new_rows(src: torch.Tensor, count: int) -> torch.Tensor:
            """Mean of the existing rows plus small noise.

            Not zeros: a zero embedding row gives the new symbol no signal to start from, and a zero
            logit row makes it compete at an arbitrary offset from every trained row. Starting at the
            centroid puts the new symbols on the same scale as their neighbours, which is the standard
            vocabulary-extension initialisation.
            """
            mean = src.mean(dim=0, keepdim=True).expand(count, -1).clone()
            noise = torch.randn(mean.shape, generator=g, dtype=mean.dtype) * (0.01 * src.std())
            return mean + noise

        report = {"old_vocab": v, "new_vocab": v + n, "symbols": list(self.new_symbols), "tensors": {}}

        # --- AED heads: append on axis 0 -------------------------------------------------------
        # ⚠ Tied weights share ONE storage (decoder.input_embedding.weight / decoder.logits.weight
        # have equal data_ptr()). Extending them independently would draw different noise for each and
        # silently UNTIE them -- caught by the tied-pair assert below while this job was being written.
        # So the widened tensor is computed once per source storage and reused.
        widened: Dict[int, Any] = {}
        for k in aed_keys:
            old = model[k]
            key = old.data_ptr()
            if key not in widened:
                add = _new_rows(old, n)
                cand = torch.cat([old, add], dim=0)
                assert torch.equal(cand[:v], old), f"{k}: first {v} rows changed"
                widened[key] = cand
            new = widened[key]
            report["tensors"][k] = {"kind": "aed_axis0", "from": list(old.shape), "to": list(new.shape)}
            model[k] = new

        # --- CTC heads: insert before the blank on the LAST axis, blank moves to the end -------
        for k in ctc_w:
            old = model[k]
            body, blank = old[..., :v], old[..., v : v + 1]
            add = _new_rows(body.transpose(0, 1), n).transpose(0, 1)
            new = torch.cat([body, add, blank], dim=-1)
            assert torch.equal(new[..., :v], body), f"{k}: vocab columns changed"
            assert torch.equal(new[..., v + n : v + n + 1], blank), f"{k}: blank not preserved at the end"
            model[k] = new
            report["tensors"][k] = {"kind": "ctc_axis-1", "from": list(old.shape), "to": list(new.shape)}

        for k in ctc_b:
            old = model[k]
            body, blank = old[:v], old[v : v + 1]
            add = body.mean().repeat(n)
            new = torch.cat([body, add, blank], dim=0)
            assert torch.equal(new[:v], body) and torch.equal(new[v + n : v + n + 1], blank), f"{k}: bias corrupted"
            model[k] = new
            report["tensors"][k] = {"kind": "ctc_bias", "from": list(old.shape), "to": list(new.shape)}

        # --- the pseudo-encoder's PHONEME table: replace outright, do not widen ------------------
        # Built exactly as aed_glowtts_model_def does (`:4947-4953`): the per-phone means with the
        # `[space]` row appended as the blank row, so row i means phone i and the last row is blank.
        # It is frozen (`trainable = False`), so this is an initialisation the run will not change --
        # which is also why replacing it wholesale is correct rather than lossy.
        if self.pseudo_enc_table is not None:
            import numpy

            key = "pseudo_enc.embedding.weight"
            assert key in model, f"{key} missing -- is this a pseudo_speech_enc checkpoint?"
            npz = numpy.load(self.pseudo_enc_table.get_path(), allow_pickle=True)
            means, labels = npz["means"], [str(x) for x in npz["labels"]]
            sil = next(i for i, lab in enumerate(labels) if lab.split(".")[0] == "[space]")
            tbl = numpy.concatenate([means, means[sil][None]], axis=0)
            new_t = torch.tensor(tbl, dtype=model[key].dtype)
            report["tensors"][key] = {
                "kind": "pseudo_enc_phoneme_table",
                "from": list(model[key].shape),
                "to": list(new_t.shape),
            }
            assert new_t.shape[1] == model[key].shape[1], "mel dim changed -- wrong table?"
            model[key] = new_t
            report["pseudo_enc_rows"] = int(new_t.shape[0])

        # The tied pair must still be bit-identical, or the model silently trains two matrices.
        tied = [k for k in ("decoder.input_embedding.weight", "decoder.logits.weight") if k in model]
        if len(tied) == 2:
            assert torch.equal(model[tied[0]], model[tied[1]]), "the tied embedding/logits pair diverged"
            report["tied_ok"] = True

        if "model" in sd:
            sd["model"] = model
        torch.save(sd, self.out_checkpoint.get_path())
        with open(self.out_report.get_path(), "wt") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)


class FilterLinesByLexiconJob(Job):
    """
    Drop text lines containing a word the lexicon cannot pronounce.

    🔴 **Required, not tidy-up.** ``PhoneSeqGenerator`` **raises** on an OOV word
    (`exp2026_05_28_tts_encoder_fzj.py:4324-4325`), so a single unpronounceable word would abort the
    training run — after the GPUs are allocated, at whatever epoch that line is first drawn.

    Measured on the real corpus with the real normalisation and the merged lexicon (backlog §36):
    residual OOV is **0.0680% of tokens** in **0.878% of lines**, and the residue is **entirely
    English/French apostrophe contractions** (`DON'T`, `QU'IL`, `D'ENGLETERRE`, `I'M`, `CAN'T`) —
    `mfa g2p` silently refuses apostrophe words. So this drops <1% of lines and what it removes is
    foreign contamination, not German text: **66.4 M of 67.07 M words survive (99.1%)**.

    The lexicon word list is read from the *same* file the phonemiser will use, so the filter cannot
    drift from the thing it is protecting.
    """

    def __init__(self, *, text: tk.Path, lexicon_text: tk.Path, max_drop_fraction: float = 0.10):
        """
        :param text: the normalised injection text (gzip or plain), one sequence per line.
        :param lexicon_text: ``MfaDictToLexiconTextJob.out_lexicon_text`` -- ``WORD phones`` per line.
        :param max_drop_fraction: refuse to proceed if more lines than this are dropped. The measured
            rate is 0.878%, so the 10% default is ~11x headroom and still catches the failure it is
            for: normalisation and lexicon disagreeing about case or sharp-s, which would silently
            decimate the corpus rather than raise. Exposed only so small fixtures can relax it --
            **do not raise it to make a real corpus pass.**
        """
        self.text = text
        self.lexicon_text = lexicon_text
        self.max_drop_fraction = max_drop_fraction
        self.out_text = self.output_path("out.gz")
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import gzip
        import json

        words = set()
        with open(self.lexicon_text.get_path(), "rt", encoding="utf8") as f:
            for line in f:
                p = line.split(None, 1)
                if p:
                    words.add(p[0])
        assert words, "empty lexicon word list"

        src = self.text.get_path()
        opener = gzip.open if src.endswith(".gz") else open
        n_in = n_out = n_drop = 0
        with (
            opener(src, "rt", encoding="utf8") as fin,
            gzip.open(self.out_text.get_path(), "wt", encoding="utf8") as fout,
        ):
            for line in fin:
                ws = line.split()
                if not ws:
                    continue
                n_in += 1
                if all(w in words for w in ws):
                    fout.write(line if line.endswith("\n") else line + "\n")
                    n_out += 1
                else:
                    n_drop += 1

        # A filter that drops most of the corpus means the normalisation and the lexicon disagree
        # about case or sharp-s -- fail loudly rather than train on a silently decimated corpus.
        assert n_out > 0, "filter dropped everything"
        assert n_drop / max(n_in, 1) < self.max_drop_fraction, (
            f"dropped {n_drop}/{n_in} = {100 * n_drop / max(n_in, 1):.1f}% of lines, over the"
            f" {100 * self.max_drop_fraction:.0f}% limit; measured on the real corpus it is 0.878% --"
            " normalisation and lexicon are probably out of sync (case? sharp-s?)"
        )
        with open(self.out_stats.get_path(), "wt") as f:
            json.dump(
                {"lines_in": n_in, "lines_out": n_out, "lines_dropped": n_drop, "lexicon_words": len(words)},
                f,
                indent=1,
            )


def get_german_injection_text_filtered(*, budget: str = "9h") -> tk.Path:
    """The injection text, normalised **and** guaranteed pronounceable by the German lexicon."""
    lex_text = MfaDictToLexiconTextJob(
        mfa_dicts=[DE_MFA_DICT_1H if budget == "1h" else DE_MFA_DICT_9H, DE_G2P_DICT],
        phoneme_vocab=DE_PHONEME_VOCAB,
    )
    job = FilterLinesByLexiconJob(text=get_german_injection_text(), lexicon_text=lex_text.out_lexicon_text)
    job.add_alias(f"datasets/mls-de/lm-text-pronounceable-{budget}")
    return job.out_text


# MLS-German evaluation parquet, staged on FZJ by `get_mls_de_fzj.py`.
# ⚠ `dev` and `test` are EVALUATION data. The 10 h German-audio training budget (backlog §10) covers
# `1_hours`/`9_hours` only; `german/train` (1,966 h) stays excluded entirely.
DE_TEST_PARQUET = tk.Path(f"{_STAGE}/mls_de/test.parquet", hash_overwrite="dorian/de/mls/test-parquet/v1")
DE_DEV_PARQUET = tk.Path(f"{_STAGE}/mls_de/dev.parquet", hash_overwrite="dorian/de/mls/dev-parquet/v1")


class HfParquetDecodeAudioJob(Job):
    """
    Decode an MLS parquet's audio with ``soundfile`` into an HF dataset of raw float32 arrays.

    🔴 **Why this exists.** RETURNN's ``HuggingFaceDataset`` could read the parquet directly, but it
    decodes audio through HF ``datasets`` (``returnn/datasets/huggingface.py:307-312``, ``x["array"]``)
    and on JUPITER that raises ``Could not load libtorchcodec``. ``datasets`` 4 has **no soundfile
    fallback** — `Audio.decode_example` is torchcodec-or-raise. Measured on
    `mls_de/test.parquet`: the table loads (3,394 rows) and the decode dies.

    ``soundfile`` itself works fine here — it is what built the German acoustic tables from these very
    parquets — so we decode **once, offline** and store arrays the dataset can read without a codec.
    That is also far less machinery than a bliss corpus + ``BlissToOggZipJob``.

    Output is ONE arrow container (``save_to_disk``), not a file per utterance — per the project's
    small-file-count directive; inodes, not bytes, are the binding limit on these volumes.
    """

    def __init__(self, *, parquet: tk.Path, sample_rate: int = 16_000):
        self.parquet = parquet
        self.sample_rate = sample_rate
        self.out_dataset = self.output_path("dataset", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import io
        import json

        import datasets
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile as sf

        # ⚠ Stream in row-group batches. The first version did `.to_pylist()` on the whole audio
        # column and peaked at 11.7 GB RSS on a 225 MB / 3,394-row parquet, against a declared
        # mem: 16 -- it fit `test` and would not have fit a larger split.
        pf = pq.ParquetFile(self.parquet.get_path())
        cols = {c.lower(): c for c in pf.schema_arrow.names}
        acol = next(c for lc, c in cols.items() if "audio" in lc and "duration" not in lc)
        tcol = next(c for lc, c in cols.items() if lc in ("transcript", "text"))
        icol = next(c for lc, c in cols.items() if lc == "id")
        n = pf.metadata.num_rows
        assert n > 0

        # ⚠ The generator must NOT close over `self`: `datasets.from_generator` fingerprints the
        # callable together with its closure, which would drag the whole Sisyphus Job into the hash
        # (and can re-invoke the generator, so accumulating statistics inside it is unsafe too).
        # Bind plain locals here and derive the statistics from the built table afterwards.
        sr_expected = self.sample_rate

        parquet_path = self.parquet.get_path()
        batch_cols = [acol, tcol, icol]

        def _gen():
            for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64, columns=batch_cols):
                d = batch.to_pydict()
                for a, t, u in zip(d[acol], d[tcol], d[icol], strict=True):
                    raw = a["bytes"] if isinstance(a, dict) else a
                    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
                    assert sr == sr_expected, f"{u}: sample rate {sr} != {sr_expected}"
                    if arr.ndim > 1:  # never on MLS, but a silent stereo fold would corrupt the eval
                        arr = arr.mean(axis=1)
                    assert arr.size > 0, f"{u}: empty audio"
                    yield {
                        "id": str(u),
                        "transcript": t.strip(),
                        # [T, 1], NOT [T]: the ASR's extern_data declares
                        # data: [batch, time, Dim(1, "audio")], and RETURNN's HuggingFaceDataset
                        # REFUSES a 1-D column declared as (None, 1) -- measured, it raises a
                        # user-specified-Tensor mismatch. Storing the trailing axis makes the data
                        # match the contract instead of needing a wrapper to reshape at load time.
                        "audio": np.asarray(arr, dtype="float32")[:, None],
                        "duration": float(len(arr)) / sr,
                    }

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "transcript": datasets.Value("string"),
                "audio": datasets.Array2D(shape=(None, 1), dtype="float32"),
                "duration": datasets.Value("float32"),
            }
        )
        ds = datasets.Dataset.from_generator(_gen, features=features)
        assert ds.num_rows == n, f"decoded {ds.num_rows} of {n} rows"
        ds.save_to_disk(self.out_dataset.get_path())

        durations = ds["duration"]
        total_s = float(np.sum(durations))
        stats = {
            "rows": ds.num_rows,
            "total_seconds": total_s,
            "hours": total_s / 3600.0,
            "sample_rate": sr_expected,
            "duration_min": float(np.min(durations)),
            "duration_max": float(np.max(durations)),
        }
        with open(self.out_stats.get_path(), "wt") as f:
            json.dump(stats, f, indent=1)


class HfDatasetToTextDictJob(Job):
    """
    Decoded HF dataset -> the gzipped ``{seq_tag: text}`` dict that i6_core's scoring chain expects.

    German has no bliss corpus, so `CorpusToTextDictJob` (the LibriSpeech route) is unavailable. The
    output format is byte-compatible with it (`i6_core/corpus/convert.py:259-270`): ``{`` then one
    ``%r: %r,`` line per segment, then ``}``. That is all `_score_recog_out_v2` actually consumes, so
    the whole sclite chain (`SearchWordsDummyTimesToCTMJob` -> `TextDictToStmJob` -> `ScliteJob`)
    is reusable unchanged.

    🔴 References are normalised with :func:`normalize_german_text`, the **same** rule as the injection
    text. Not optional — see that docstring: the model cannot emit lowercase or sharp-s, and 37% of
    MLS-de test references contain one.
    """

    def __init__(
        self,
        *,
        dataset: tk.Path,
        tag_column: str = "id",
        text_column: str = "transcript",
        normalize: bool = True,
    ):
        self.dataset = dataset
        self.tag_column = tag_column
        self.text_column = text_column
        self.normalize = normalize
        self.out_text_dict = self.output_path("text_dictionary.py.gz")
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import gzip
        import json

        import datasets

        ds = datasets.load_from_disk(self.dataset.get_path())
        tags = ds[self.tag_column]
        texts = ds[self.text_column]
        assert len(tags) == len(texts) and tags, "empty or mismatched dataset"
        assert len(set(tags)) == len(tags), "duplicate sequence tags -- scoring would silently misalign"

        n_words = 0
        with gzip.open(self.out_text_dict.get_path(), "wt", encoding="utf8") as out:
            out.write("{\n")
            for tag, text in zip(tags, texts, strict=True):
                t = normalize_german_text(text.strip()) if self.normalize else text.strip()
                assert t, f"{tag}: empty reference"
                n_words += len(t.split())
                out.write("%r: %r,\n" % (str(tag), t))
            out.write("}\n")

        with open(self.out_stats.get_path(), "wt") as f:
            json.dump({"segments": len(tags), "ref_words": n_words, "normalized": self.normalize}, f, indent=1)


# Decoded MLS-de eval sets ([T, 1] float32 audio; see HfParquetDecodeAudioJob).
DE_TEST_DECODED = tk.Path(f"{_STAGE}/mls_de_test_decoded2", hash_overwrite="dorian/de/mls/test-decoded/v2")
DE_DEV_DECODED = tk.Path(f"{_STAGE}/mls_de_dev_decoded2", hash_overwrite="dorian/de/mls/dev-decoded/v2")


def normalize_transcripts_map_func(ds):
    """`HuggingFaceDataset` map_func: normalise the transcript column in place.

    🔴 Without this the `classes` targets are garbage. `HuggingFaceDataset` tokenises the string column
    with the SPM, and the SPM is **uppercase English** while MLS transcripts are lowercase — measured,
    the result is `[463, 2, 463, 2, 463, ...]`, i.e. `▁` + UNK for every word. Harmless while only
    recog runs (forward never reads the targets), but it is exactly the kind of silently-wrong data
    that produces a plausible number later.

    Applied at load time rather than re-decoding the audio: the transformation is text-only, and
    `datasets.map` caches it.
    """
    return ds.map(
        lambda batch: {"transcript": [normalize_german_text(t) for t in batch["transcript"]]},
        batched=True,
        desc="normalize german transcripts",
    )


# In-training dev is SUBSET; the scoring eval is not. MLS-de dev is 3,469 utts ~= 14 h of audio,
# which is LARGER than one arm C training epoch (9 h) -- evaluating it every epoch would spend more
# compute on measurement than on training. 500 rows is plenty for LR control and a learning curve.
IN_TRAIN_DEV_SUBSET = 500


def normalize_and_subset_dev_map_func(ds):
    """`map_func` for the IN-TRAINING dev only -- never for the scored eval sets.

    ⚠ Random sample with a fixed seed, not `select(range(N))`: the decoded dataset is in corpus
    order, so a head slice would be one speaker / one book, and the dev curve would track that
    slice rather than the dev set (the project's own random-sampling rule).
    """
    ds = normalize_transcripts_map_func(ds)
    n = min(IN_TRAIN_DEV_SUBSET, ds.num_rows)
    return ds.shuffle(seed=1).select(range(n))


def german_eval_dataset(*, decoded: tk.Path, main_name: str, spm_model: tk.Path, spm_size: int):
    """A `DatasetConfig` over one decoded MLS-de split, shaped exactly like the LibriSpeech eval sets.

    `extern_data` mirrors what a real recog config declares (read off
    `BatchedReturnnForwardJob.../items/shard_000/returnn.config`), so the model sees the same contract
    it does for LibriSpeech::

        data:    [batch, time, Dim(1, "audio")]     raw waveform
        classes: [batch, out-spatial] sparse Dim(spm_size), vocab=SentencePieces(spm_model)

    :param spm_model: the original 10,240-piece SPM for **arm A** (the unmodified winner), or
        `DE_SPM_EXTENDED` (10,243) for the arms trained on the surgered checkpoint.
    :param spm_size: must match `spm_model`; passed explicitly so a mismatch is a loud assert rather
        than a silent off-by-three in the output layer.
    """
    from returnn.tensor import Dim, batch_dim
    from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic

    dataset = {
        "class": "HuggingFaceDataset",
        "dataset_opts": decoded.get_path(),  # load_from_disk
        # 🔴 The data keys must be named as `extern_data` names them, not as the arrow columns are.
        # RETURNN looks up `extern_data_raw["data"]` (torch/data/extern_data.py:63); leaving the
        # column as "audio" raises `KeyError: 'data'` after the GPU is allocated.
        # ⚠ map_func runs BEFORE rename_columns, so it normalises "transcript", not "classes".
        "map_func": normalize_transcripts_map_func,
        "rename_columns": {"audio": "data", "transcript": "classes"},
        # ⚠ (None, 1), not (None,): the column is stored 2-D precisely so it matches `data` above.
        # A 1-D column is REFUSED here (measured), and reshaping at load time would need a wrapper.
        # `classes` is a string column plus a `vocab`, which HuggingFaceDataset tokenises for us.
        "data_format": {
            "data": {"dtype": "float32", "shape": (None, 1)},
            "classes": {
                "dtype": "int32",
                "shape": (None,),
                "dim": spm_size,
                "vocab": {"class": "SentencePieces", "model_file": spm_model},
            },
        },
        "seq_tag_column": "id",
        "sorting_seq_len_column": "duration",
    }
    extern_data = {
        "data": {
            "dim_tags": [
                batch_dim,
                Dim(None, name="time", kind=Dim.Types.Spatial),
                Dim(1, name="audio", kind=Dim.Types.Feature),
            ]
        },
        "classes": {
            "dim_tags": [batch_dim, Dim(None, name="out-spatial", kind=Dim.Types.Spatial)],
            "sparse_dim": Dim(spm_size, name="vocab", kind=Dim.Types.Feature),
            "vocab": {"class": "SentencePieces", "model_file": spm_model},
        },
    }
    return DatasetConfigStatic(
        main_name=main_name,
        main_dataset=dataset,
        extern_data=extern_data,
        default_input="data",
        default_target="classes",
    )


def _score_german_recog_out(dataset, recog_output, *, text_dicts: Dict[str, tk.Path]):
    """WER via sclite, the LibriSpeech chain verbatim -- only the reference source differs.

    `_score_recog_out_v2` gets its references from `CorpusToTextDictJob` over a bliss corpus. German
    has no bliss corpus, so `HfDatasetToTextDictJob` supplies a byte-compatible dict and everything
    downstream (`SearchWordsDummyTimesToCTMJob` -> `TextDictToStmJob` -> `ScliteJob`) is unchanged.
    """
    from i6_core.recognition.scoring import ScliteJob
    from i6_core.returnn.search import SearchWordsDummyTimesToCTMJob
    from i6_core.text.convert import TextDictToStmJob
    from i6_experiments.users.zeyer import tools_paths
    from i6_experiments.users.zeyer.datasets.task import ScoreResult

    corpus_name = dataset.get_main_name()
    assert corpus_name in text_dicts, f"no reference text dict for {corpus_name!r} (have {sorted(text_dicts)})"
    corpus_text_dict = text_dicts[corpus_name]

    # Same arbitrary segment length as LibriSpeech: the CTM/STM writers keep two decimals, so a large
    # value avoids precision trouble on long sequences.
    seg_length_time = 1000.0
    search_ctm = SearchWordsDummyTimesToCTMJob(
        recog_words_file=recog_output.output, seq_order_file=corpus_text_dict, seg_length_time=seg_length_time
    ).out_ctm_file
    stm_file = TextDictToStmJob(text_dict=corpus_text_dict, seg_length_time=seg_length_time).out_stm_path
    score_job = ScliteJob(
        ref=stm_file, hyp=search_ctm, sctk_binary_path=tools_paths.get_sctk_binary_path(), precision_ndigit=2
    )
    return ScoreResult(dataset_name=corpus_name, main_measure_value=score_job.out_wer, report=score_job.out_report_dir)


def get_mls_de_task(*, extended_vocab: bool, train_dataset=None, train_epoch_split: int = 1):
    """
    The MLS-German `Task`: dev for tuning, test for the headline WER, sclite scoring on normalised text.

    :param extended_vocab: False for **arm A** (the unmodified winner, 10,240-piece SPM, which cannot
        write German orthography at all); True for the arms running on the surgered checkpoint
        (10,243 pieces, Ä Ö Ü appended).
    :param train_dataset: arm C only. Arms A and B do not train on German audio -- A does not train at
        all and B trains on English LS-960 -- so this is None there and `Task.train_dataset` is a stub.

    ⚠ `_train_tts_encoder` chooses its task internally with no parameter (`:3819-3838`), so reaching
    this into a run needs the graph-build patch technique of :class:`PatchGlowTtsToGerman`.
    """
    from i6_experiments.users.zeyer.datasets.librispeech import _spm_to_words
    from i6_experiments.users.zeyer.datasets.task import Task

    spm_model = DE_SPM_EXTENDED if extended_vocab else DE_SPM_ORIGINAL
    spm_size = GERMAN_SPM_SIZE if extended_vocab else ENGLISH_SPM_SIZE

    dev = german_eval_dataset(decoded=DE_DEV_DECODED, main_name="dev", spm_model=spm_model, spm_size=spm_size)
    test = german_eval_dataset(decoded=DE_TEST_DECODED, main_name="test", spm_model=spm_model, spm_size=spm_size)

    # References are normalised with the SAME rule as the injection text (see normalize_german_text):
    # the model can emit neither lowercase nor sharp-s, and 37% of test references contain one.
    dev_dict = HfDatasetToTextDictJob(dataset=DE_DEV_DECODED)
    dev_dict.add_alias("datasets/mls-de/dev-text-dict")
    test_dict = HfDatasetToTextDictJob(dataset=DE_TEST_DECODED)
    test_dict.add_alias("datasets/mls-de/test-text-dict")
    text_dicts = {"dev": dev_dict.out_text_dict, "test": test_dict.out_text_dict}

    from i6_experiments.users.zeyer.datasets.task import MeasureType

    return Task(
        name="mls-de",
        # 🔴 `Task.train_dataset` is non-optional, but arms A and B never train on German audio, so a
        # stub is unavoidable. It is **dev, never test**: if some future caller does train on the
        # stub, it must not be able to train on the evaluation set we report. Neither current path
        # reads it -- arm A does not train, and PatchTaskEvalToGerman keeps the ENGLISH train_dataset.
        train_dataset=train_dataset if train_dataset is not None else dev,
        train_epoch_split=train_epoch_split,
        dev_dataset=dev,
        # 🔴 dev MUST be in eval_datasets, not only in dev_dataset. `recog_training_exp_batched`
        # decodes `task.eval_datasets` and nothing else (recog_batched.py:282), so with only "test"
        # here dev is NEVER decoded -- and then the best epoch can ONLY be chosen on test (§83).
        eval_datasets={"dev": dev, "test": test},
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="test",
        score_recog_output_func=partial(_score_german_recog_out, text_dicts=text_dicts),
        # 🔴 REQUIRED. Without it the recogniser's SPM **subword pieces** are scored as if they were
        # words: arm A's first run produced
        #   HYP: ▁DEACON ▁CAN ▁YE ▁SO ▁A B EN ▁WAL TEN ...   (against REF: DENKEN SIE SOEBEN ...)
        # i.e. 102,069 insertions (83.86%) against 121,714 reference words, and a meaningless
        # WER of 182.66. LibriSpeech uses exactly this for spm vocabs (`librispeech.py:767`).
        recog_post_proc_funcs=[_spm_to_words],
    )


def german_extended_spm_vocab():
    """The extended SPM as a `VocabConfig`, mirroring how `spm10k` is built.

    `_get_spm_vocab` constructs `SentencePieceModel(dim=..., model_file=..., unknown_label="<unk>",
    bos_idx=1, eos_idx=0)` (`librispeech.py:119-125`); the only differences here are `dim` (10,243)
    and the model file. 🟢 `bos_idx=1` / `eos_idx=0` stay valid because the three symbols were
    **appended** at 10240-10242, leaving ids 0..10239 untouched (§34).
    """
    from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel

    return SentencePieceModel(
        dim=GERMAN_SPM_SIZE,
        model_file=DE_SPM_EXTENDED,
        unknown_label="<unk>",
        bos_idx=1,
        eos_idx=0,
    )


class PatchTaskEvalToGerman:
    """
    Context manager: while active, `_train_tts_encoder`'s task keeps its ENGLISH training data but
    evaluates on **MLS-de**.

    **Why swap only the eval side.** Arm B is "English LS-960 audio ⇄ German text injection", so its
    audio branch *must* stay English — replacing the whole task would silently change what the arm
    trains on, which is the claim itself. `_train_tts_encoder:3838` reads `task.train_dataset` and
    builds the audio branch from it, so that object has to remain the LibriSpeech one.

    `get_librispeech_task_raw_v2` is imported **inside** `_train_tts_encoder` (`:3777-3781`), so the
    binding resolves at call time from `users.zeyer.datasets.librispeech` — patching that module
    attribute is seen. This is the **graph-build** half; its result is serialised into the job.

    ⚠ Arm C is NOT covered by this: it trains on German paired audio, which needs a German train
    dataset that does not exist yet (the 1 h/9 h splits are still only parquet).

    🔴 **Open decision before this is used — the train-side vocab.** After the checkpoint surgery the
    model's output layer is **10,243** wide, so the *training* targets must come from
    `DE_SPM_EXTENDED` too, not the 10,240-piece SPM the English task builds. The extended SPM keeps
    ids 0..10239 identical (verified §34), so English text tokenises to exactly the same ids — it is
    safe, but it is not automatic: `get_librispeech_task_raw_v2(vocab=...)` takes a vocab *name*, not
    a model file. Resolve that before running arm B, or the arm trains a 10,243-wide head against
    10,240-wide targets.
    """

    def __init__(self, *, extended_vocab: bool = True, extended_train_vocab: bool = True):
        """
        :param extended_train_vocab: also build the ENGLISH training data against the extended SPM.
            🔴 Required whenever the run starts from the surgered checkpoint. `_train_tts_encoder`
            derives `spm_dim` from the task's vocab (`:3894`) and builds the output layer that wide,
            so a 10,240 task against a 10,243 checkpoint fails to load. English text tokenises
            **identically** under the extended SPM (ids 0..10239 unchanged, §34), so this changes the
            vocabulary size and nothing else about the English data.
        """
        self.extended_vocab = extended_vocab
        self.extended_train_vocab = extended_train_vocab
        self.count = 0
        self._patch = None

    def __enter__(self):
        import dataclasses
        import unittest.mock

        from i6_experiments.users.zeyer.datasets import librispeech as _ls

        real = _ls.get_librispeech_task_raw_v2
        de = get_mls_de_task(extended_vocab=self.extended_vocab)

        def _wrapped(*args, **kwargs):
            if self.extended_train_vocab:
                # `_train_tts_encoder` passes vocab= by keyword (`:3833`); assert rather than assume,
                # because silently not swapping gives a 10,240-wide head for a 10,243 checkpoint.
                assert "vocab" in kwargs, f"expected vocab= by keyword, got args={args!r}"
                kwargs = dict(kwargs, vocab=german_extended_spm_vocab())
            task = real(*args, **kwargs)
            self.count += 1
            # Keep train_dataset / train_epoch_split (English audio); swap everything eval-side.
            return dataclasses.replace(
                task,
                dev_dataset=de.dev_dataset,
                eval_datasets=de.eval_datasets,
                main_measure_name=de.main_measure_name,
                main_measure_type=de.main_measure_type,
                score_recog_output_func=de.score_recog_output_func,
                recog_post_proc_funcs=de.recog_post_proc_funcs,
            )

        self._patch = unittest.mock.patch.object(_ls, "get_librispeech_task_raw_v2", _wrapped)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        if exc[0] is None:
            assert self.count == 1, (
                f"expected to rewrite exactly 1 task, rewrote {self.count} --"
                " a silent zero would evaluate the German arm on LibriSpeech"
            )
        return False


def get_surgered_winner_checkpoint(winner_checkpoint, *, budget: str = "9h") -> tk.Path:
    """The winner's checkpoint widened to the German output vocabulary (10,240 -> 10,243).

    This is what arms B and C start from via ``import_model_train_epoch1``. The first 10,240 output
    rows are preserved bit-exactly and Ä/Ö/Ü are appended, matching `DE_SPM_EXTENDED` where the same
    three pieces were appended at 10240-10242 (§34) -- so row *i* still means piece *i*.
    """
    table = DE_TABLE_1H if budget == "1h" else DE_TABLE_9H
    # `winner_checkpoint()` returns an i6_core PtCheckpoint; unwrap so the job's hashed input is the
    # tk.Path itself (PtCheckpoint hashes via the same path, so this does not change the hash meaning).
    ckpt = getattr(winner_checkpoint, "path", winner_checkpoint)
    job = ExtendVocabCheckpointJob(checkpoint=ckpt, pseudo_enc_table=table)
    job.add_alias("german/winner-vocab-extended-aou")
    return job.out_checkpoint


class PatchModelDefToGerman:
    """
    Context manager: while active, `_train_tts_encoder` builds the model with
    :func:`aed_glowtts_model_def_de` instead of Albert's `aed_glowtts_model_def`.

    `_train_tts_encoder` hardcodes `model_def=aed_glowtts_model_def` (`:4136`) and exposes no
    parameter, so this is the only way in without editing his file.

    🟢 **Hashing works out correctly, and this is why a wrapper was the right shape:** `train_v4`
    hashes a model_def by `(module, qualname)`, so the config records
    `...german_xling.aed_glowtts_model_def_de` — a genuinely distinct model, which it is (57-wide
    pseudo vocab, 10,243-wide output).

    ⚠ **No recursion, and the reason is worth stating.** `aed_glowtts_model_def_de` calls
    `_tts.aed_glowtts_model_def` — the very attribute this patches. That is safe because the two
    happen in **different processes**: this patch is graph-build-time only, while the wrapper runs at
    **job-run** time, where no patch is active and the attribute is Albert's original. Do not
    "simplify" this by making the patch permanent (e.g. at module import), which would turn the
    wrapper into an infinite loop at run time.
    """

    def __init__(self):
        self._patch = None

    def __enter__(self):
        import unittest.mock

        from i6_experiments.users.zeyer.experiments import exp2026_05_28_tts_encoder_fzj as _tts

        assert _tts.aed_glowtts_model_def is not aed_glowtts_model_def_de, "already patched (nested?)"
        self._patch = unittest.mock.patch.object(_tts, "aed_glowtts_model_def", aed_glowtts_model_def_de)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


# Arm B schedule. Deliberately the same shape as winner_plus_tts: this is a FINETUNE of a converged
# model, so `nep` is short and `peak_lr` is 1/10 of the winner's 5e-3 -- `import_model_train_epoch1`
# restarts the LR schedule at 1, and `nep` stretches OCLR proportionally, so the two cannot be tuned
# independently (configs.py:162-182).
ARM_B_NEP = 10
ARM_B_PEAK_LR = 5e-4


# ---------------------------------------------------------------------------------------------
# Arm C — German paired audio (the baseline B must match or beat)
# ---------------------------------------------------------------------------------------------
# The plan's ONLY controlled comparison is B vs C at the same audio budget, so everything below is
# written to keep the two arms matched everywhere except the training data itself.

DE_TRAIN_1H_PARQUET = tk.Path(f"{_STAGE}/mls_de/1_hours.parquet", hash_overwrite="dorian/de/mls/train-1h-parquet/v1")
DE_TRAIN_9H_PARQUET = tk.Path(f"{_STAGE}/mls_de/9_hours.parquet", hash_overwrite="dorian/de/mls/train-9h-parquet/v1")


class HfParquetToOggZipJob(Job):
    """
    MLS parquet -> a RETURNN ``OggZipDataset`` archive. One ``.ogg.zip``, no ffmpeg, no bliss corpus.

    🔴 **Why an ogg zip and not the `HuggingFaceDataset` the EVAL sets use.** The eval sets never need
    augmentation, so reading them through `HuggingFaceDataset` is fine. The *training* set does, and
    that is the whole reason for this job:

    * `OggZipDataset` supports ``audio.pre_process``, which is how the English branch gets **speed
      perturbation** (`exp2026_05_28_tts_encoder_fzj.py:3864`). `HuggingFaceDataset` has no such hook.
    * The obvious workaround -- letting `aed.train_exp` apply it -- **silently does nothing**. It does
      ``task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")``
      (`aed.py:485-488`), and `DatasetConfigStatic` has no such attribute and never reads it: the
      assignment succeeds, is ignored, and no warning is produced (verified 2026-09-19).

    That matters more here than anywhere else in the project: arm C trains on **1 h / 9 h** of audio,
    where augmentation is worth a lot, and arm B's English branch has it. Dropping it silently would
    handicap the baseline -- i.e. bias the headline comparison **in favour of our own claim**. This
    job exists so that both arms read audio through the same dataset class with the same augmentation
    hook available.

    🟢 **No ffmpeg, no RETURNN tool, no bliss.** `BlissToOggZipJob` would need a bliss corpus (which
    German does not have) and shells out to ffmpeg. `soundfile` writes Ogg/Vorbis directly -- verified
    on this login node: ``sf.available_formats()["OGG"]`` with subtypes ``{VORBIS, OPUS}``, and a
    write->read round trip returns the same shape and sample rate.

    Output is **one zip**, per the small-file-count directive: inodes are the binding limit on these
    volumes, and a file per utterance is what keeps hitting it.
    """

    # `version` is a deliberate hash knob: this job's correctness lives entirely in `run()`, and a
    # Sisyphus hash covers constructor ARGUMENTS, not the body (learned the expensive way in §68 --
    # a fix that "did not take effect" because the hash never moved).
    def __init__(self, *, parquet: tk.Path, sample_rate: int = 16_000, normalize: bool = True, version: int = 1):
        """
        :param normalize: apply :func:`normalize_german_text` to the transcripts. **Keep it True.**
            The targets must be tokenised by the same uppercase, hyphen-free SPM the references are
            scored against; lowercase text against this SPM produces ``[463, 2, 463, 2, ...]``
            (``▁`` + UNK per word) -- measured, and silently trainable garbage.
        """
        self.parquet = parquet
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.version = version
        self.out_ogg_zip = self.output_path("out.ogg.zip")
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import io
        import json
        import zipfile

        import numpy
        import pyarrow.parquet as pq
        import soundfile as sf

        # ⚠ Stream in row-group batches. Materialising the audio column cost 11.7 GB RSS on a 225 MB
        # parquet (§39b); the 9 h split is larger than the test split that nearly broke it.
        parquet_path = self.parquet.get_path()
        pf = pq.ParquetFile(parquet_path)
        cols = {c.lower(): c for c in pf.schema_arrow.names}
        acol = next(c for lc, c in cols.items() if "audio" in lc and "duration" not in lc)
        tcol = next(c for lc, c in cols.items() if lc in ("transcript", "text"))
        icol = next(c for lc, c in cols.items() if lc == "id")
        n_rows = pf.metadata.num_rows
        assert n_rows > 0, "empty parquet"

        # The archive layout is dictated by OggZipDataset and was read off a real LibriSpeech zip
        # (BlissToOggZipJob.5ad18raRAWhr) rather than guessed: a directory `out.ogg/`, one
        # `out.ogg.txt` holding a python-literal list of dicts, and `file` relative to that directory.
        stem = "out.ogg"
        entries = []
        total_s = 0.0
        n_done = 0

        with zipfile.ZipFile(self.out_ogg_zip.get_path(), "w", compression=zipfile.ZIP_STORED) as zf:
            # ZIP_STORED, not DEFLATE: Vorbis is already compressed, so deflating it buys ~0% and
            # costs CPU on every read.
            for batch in pf.iter_batches(batch_size=32, columns=[acol, tcol, icol]):
                d = batch.to_pydict()
                for a, t, u in zip(d[acol], d[tcol], d[icol], strict=True):
                    raw = a["bytes"] if isinstance(a, dict) else a
                    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
                    assert sr == self.sample_rate, f"{u}: sample rate {sr} != {self.sample_rate}"
                    if arr.ndim > 1:  # never on MLS, but a silent stereo fold would corrupt training
                        arr = arr.mean(axis=1)
                    assert arr.size > 0, f"{u}: empty audio"

                    buf = io.BytesIO()
                    sf.write(buf, arr, sr, format="OGG", subtype="VORBIS")
                    rel = f"{u}.ogg"
                    zf.writestr(f"{stem}/{rel}", buf.getvalue())

                    text = t.strip()
                    if self.normalize:
                        text = normalize_german_text(text)
                    assert text, f"{u}: empty transcript"
                    dur = float(arr.size) / sr
                    total_s += dur
                    n_done += 1
                    entries.append(
                        {"text": text, "speaker_name": None, "file": rel, "seq_name": str(u), "duration": dur}
                    )

            assert n_done == n_rows, f"wrote {n_done} of {n_rows} rows"
            # Written last so a torn run cannot leave a zip whose index claims more than it holds.
            meta = "[\n" + "".join("%r,\n" % (e,) for e in entries) + "]\n"
            zf.writestr(f"{stem}.txt", meta.encode("utf8"))

        durations = numpy.array([e["duration"] for e in entries], dtype="float64")
        with open(self.out_stats.get_path(), "wt") as f:
            json.dump(
                {
                    "rows": n_done,
                    "total_seconds": total_s,
                    "hours": total_s / 3600.0,
                    "normalized": self.normalize,
                    "sample_rate": self.sample_rate,
                    "duration_min": float(durations.min()),
                    "duration_max": float(durations.max()),
                },
                f,
                indent=1,
            )


def german_train_oggzip(*, budget: str = "9h") -> tk.Path:
    """The MLS-de 1 h / 9 h training split as a RETURNN ogg zip."""
    assert budget in ("1h", "9h"), budget
    parquet = DE_TRAIN_1H_PARQUET if budget == "1h" else DE_TRAIN_9H_PARQUET
    job = HfParquetToOggZipJob(parquet=parquet)
    job.add_alias(f"datasets/mls-de/train-{budget}-oggzip")
    return job.out_ogg_zip


def german_train_dataset(*, budget: str, spm_model: tk.Path, spm_size: int, partition_epoch: int = 1):
    """Arm C's training `DatasetConfig`: MLS-de paired audio, shaped like the English ASR branch.

    ⚠ **The `tk.Path` is passed as an OBJECT, not `.get_path()`** -- this is the §66a fix, and it is
    the one place it actually matters. A string in `dataset_opts` creates **no Sisyphus dependency
    edge**, so the training would be considered runnable before its data exists. That is harmless for
    the eval sets (pre-staged files behind `hash_overwrite`) and wrong here, because this path is a
    **job output**. Per §66a this is applied on the TRAIN path only: touching `german_eval_dataset`
    would re-hash the eval datasets and orphan arm B, which is in flight.
    """
    from returnn.tensor import Dim, batch_dim
    from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic
    from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

    audio_opts = {
        # Matches the English branch exactly (read off arm B's written returnn.config), so the two
        # arms differ in their data and not in how that data is read.
        "features": "raw",
        "sample_rate": 16_000,
        "peak_normalization": True,
        "preemphasis": None,
        # 🔴 Set HERE, not via `__train_audio_preprocess`. See HfParquetToOggZipJob's docstring: the
        # config route assigns to an attribute `DatasetConfigStatic` does not have, and is a silent
        # no-op. Arm B's English audio is speed-perturbed, so arm C's must be too.
        "pre_process": speed_pert_librosa_config,
    }
    train_dataset = {
        "class": "OggZipDataset",
        "path": [german_train_oggzip(budget=budget)],  # tk.Path object -> real dependency edge
        "use_cache_manager": True,
        "audio": audio_opts,
        "targets": {"class": "SentencePieces", "model_file": spm_model},
        "partition_epoch": partition_epoch,
        "seq_ordering": "laplace:.100",
    }
    extern_data = {
        "data": {
            "dim_tags": [
                batch_dim,
                Dim(None, name="time", kind=Dim.Types.Spatial),
                Dim(1, name="audio", kind=Dim.Types.Feature),
            ]
        },
        "classes": {
            "dim_tags": [batch_dim, Dim(None, name="out-spatial", kind=Dim.Types.Spatial)],
            "sparse_dim": Dim(spm_size, name="vocab", kind=Dim.Types.Feature),
            "vocab": {"class": "SentencePieces", "model_file": spm_model},
        },
    }
    # In-training evaluation sets. REQUIRED: `DatasetConfig.get_eval_datasets()` asserts rather than
    # defaulting to empty, and these are what produce the per-epoch `learning_rates` curve (the
    # `dev_loss_ce` / `dev_loss_fer` series read by `analysis/parse_lr.py`) and drive LR control.
    #
    # `dev` is the official MLS-de dev split, reusing `german_eval_dataset` **through its public
    # accessor** -- deliberately not by editing it, since that would re-hash the eval datasets and
    # orphan arm B while it is in flight (§66a).
    # ⚠ The two sets use DIFFERENT dataset classes (HuggingFaceDataset for dev, OggZipDataset for
    # devtrain). That is fine -- they are independent datasets -- and both yield the same
    # `extern_data` contract: data [B,T,1] raw waveform + sparse `classes`.
    dev_ds = dict(
        german_eval_dataset(
            decoded=DE_DEV_DECODED, main_name="dev", spm_model=spm_model, spm_size=spm_size
        ).get_main_dataset()
    )
    # Subset for the in-training curve only. This is a COPY of the eval dict, so the scored
    # dev/test datasets in `get_mls_de_task` are untouched and still cover the full split.
    dev_ds["map_func"] = normalize_and_subset_dev_map_func

    # `devtrain` is a held-IN sample of the training data, the standard RETURNN pair: dev alone cannot
    # separate "not learning" from "not generalising", and the gap between the two is what made the
    # DLM curve readable (§71).
    # ⚠ Built explicitly, NOT with `copy.deepcopy(train_dataset)`. `tk.Path.__deepcopy__`
    # deep-copies every attribute including **`creator`** (`job_path.py:250-258`), so a deepcopy of a
    # dict holding a job-output Path mints a DUPLICATE Job object outside Sisyphus's `JobSingleton`
    # registry -- a second object with the same hash that the graph does not know about. It is
    # harmless here only because `INCLUDE_CREATOR_STATE` is False, so serialisation drops the creator
    # and emits the resolved path; that is luck, not design. Reusing the same Path object keeps one
    # dependency edge and one job.
    devtrain = dict(train_dataset)
    devtrain["partition_epoch"] = 1
    devtrain["seq_ordering"] = "sorted_reverse"
    devtrain["fixed_random_subset"] = 500
    devtrain["fixed_random_seed"] = 1
    # No speed perturbation on an evaluation set -- augmenting it would make the number noisy and
    # not comparable across epochs.
    devtrain["audio"] = {k: v for k, v in audio_opts.items() if k != "pre_process"}

    return DatasetConfigStatic(
        main_name=f"mls-de-train-{budget}",
        train_dataset=train_dataset,
        default_input="data",
        default_target="classes",
        extern_data=extern_data,
        eval_datasets={"dev": dev_ds, "devtrain": devtrain},
        use_deep_copy=True,
    )


class CaptureRegisteredOutputs:
    """Capture what a graph-build block passes to ``tk.register_output``.

    🔴 **Why this exists.** `notify_result` needs the `tk.Path` of a recog summary so the sink becomes
    a *downstream job* that fires exactly when the result lands. But `recog_training_exp_batched`
    **returns nothing** — it only does `tk.register_output(prefix + "/recog_results_best", ...)`
    (`recog_batched.py:384-385`). Without a handle there are two bad options and no good one:

    * reference `output/<prefix>/recog_results_best` as a bare `tk.Path` — a creator-less path, so
      Sisyphus builds **no dependency edge** and the sink runs before the result exists (the §66a
      failure mode, and `tk.register_output` only makes a symlink);
    * edit `recog_batched.py` to return the job — that is `users/zeyer/` code, shared with other
      users and off-limits.

    So we intercept the registration itself. `recog_batched` calls `tk.register_output(...)` by
    attribute at call time, so patching the module attribute is seen — the same mechanism, and the
    same reason it works, as :class:`PatchGlowTtsToGerman`.

    🟢 Attaching a sink adds only a **downstream** job; it cannot change the training's hash. Verified
    by graph diff when this landed.
    """

    def __init__(self, *, match: str = ""):
        """:param match: only capture registered names containing this substring."""
        self.match = match
        self.captured: Dict[str, Any] = {}
        self._patch = None

    def __enter__(self):
        import unittest.mock

        real = tk.register_output

        def _wrapped(name, value, *args, **kwargs):
            if self.match in str(name):
                self.captured[str(name)] = value
            return real(name, value, *args, **kwargs)

        self._patch = unittest.mock.patch.object(tk, "register_output", _wrapped)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def _attach_german_result_sink(captured: Dict[str, Any], *, tag: str, note: str) -> None:
    """Wire a `ResultNotify` onto whatever recog summaries a German arm registered.

    ⚠ Deliberately **one sink per summary**, never one sink bundled over several. §43b: a single sink
    depending on two outputs could not fire at all when one of them failed, which suppressed the
    number that HAD landed -- exactly the failure the sink exists to prevent.
    """
    from i6_experiments.users.dorian_koch.speech_llm.result_notify import notify_result

    best = {k: v for k, v in captured.items() if k.endswith("recog_results_best")}
    if not best:
        # Loud, but not fatal at graph-build time: a missing sink must not take the training with it.
        print(f"[german_xling] WARNING: no recog summary captured for {tag!r}; no result sink attached")
        return
    for name, path in sorted(best.items()):
        notify_result(tag, {"mls_de": path}, note=note)


class SuppressRecog:
    """Context manager: while active, a training builds NO recog jobs.

    🔴 **Why this is needed, and why it is not laziness.** Keeping a checkpoint and *decoding* it are
    the **same knob**: `ModelWithCheckpoints.from_training_job` derives the recog's `fixed_epochs`
    from `cleanup_old_models["keep"]` (`model_with_checkpoints.py:90-100`). So preserving the epochs
    around arm C's dev optimum (§78) unavoidably schedules a recog for each of them — and a measured
    arm C recog is **8 cells / 2 h 20 m, ~2.5x the cost of the training itself** (§90). A 4-arm sweep
    would therefore spend ~9 h of GPU on decoding to choose between configs that each train in 50 min.

    Since §92a established that **dev loss ranks epochs exactly as WER does** (perfect rank agreement
    across a 2x FER spread, non-monotonicity included), the sweep does not need any recog: select the
    config *and* the epoch from `learning_rates`, which costs nothing, then recog **only the winner**.

    `_train_asr_base_multigpu` imports `recog_training_exp_batched` **inside the function body**, so
    the binding resolves at call time and patching the module attribute is seen — the same mechanism,
    and the same reason it works, as :class:`PatchGlowTtsToGerman`.

    ⚠ Use ONLY for sweep arms. The arm whose number is reported must run its real recog.
    """

    def __init__(self):
        self.count = 0
        self._patches = []

    def __enter__(self):
        import unittest.mock

        from i6_experiments.users.zeyer import recog_batched as _rb
        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext import (
            aed_ctc_batched as _actc,
        )

        def _noop(*args, **kwargs):
            self.count += 1
            return None

        # ⚠ TWO paths, not one -- the first version patched only the per-epoch recog and the sweep
        # still scheduled 80 jobs (20 per arm). `_train_asr_base_multigpu` ALSO calls
        # `aed_ctc_timesync_recog_recomb_auto_scale_batched` unconditionally for its "headline
        # AED+CTC first-pass recog (sharded, tuned scales)", which is where the `ScaleTuningJob`s
        # come from. Both must be suppressed, and the graph diff is what proves it.
        self._patches = [
            unittest.mock.patch.object(_rb, "recog_training_exp_batched", _noop),
            unittest.mock.patch.object(_actc, "aed_ctc_timesync_recog_recomb_auto_scale_batched", _noop),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        if exc[0] is None:
            # A silent zero would mean the patch missed and the sweep quietly scheduled ~9 h of
            # decoding -- the exact cost this exists to avoid, and invisible until the queue fills.
            assert self.count >= 1, "SuppressRecog was never reached -- the sweep would schedule full recogs"
        return False


class PatchTaskAllToGerman:
    """
    Context manager: while active, `_train_asr_base_multigpu` trains **and** evaluates on MLS-de.

    The sibling of :class:`PatchTaskEvalToGerman`, and the difference is the whole point of arm C.
    That one keeps the ENGLISH `train_dataset` (arm B's audio branch must stay English -- that is its
    claim) and swaps only the eval side. This one swaps **everything**, because arm C's claim is that
    the German audio is what trains it.

    `_train_asr_base_multigpu` imports `get_librispeech_task_raw_v2` **inside the function body**
    (`exp2026_05_28_tts_encoder_fzj.py:3079`) and calls it at `:3095`, so the binding resolves at call
    time and patching the module attribute is seen -- the same mechanism, and the same reason it
    works, as :class:`PatchGlowTtsToGerman`.
    """

    def __init__(self, *, budget: str, extended_vocab: bool = True, partition_epoch: int = 1):
        self.budget = budget
        self.extended_vocab = extended_vocab
        self.partition_epoch = partition_epoch
        self.count = 0
        self._patch = None

    def __enter__(self):
        import unittest.mock

        from i6_experiments.users.zeyer.datasets import librispeech as _ls

        spm_model = DE_SPM_EXTENDED if self.extended_vocab else DE_SPM_ORIGINAL
        spm_size = GERMAN_SPM_SIZE if self.extended_vocab else ENGLISH_SPM_SIZE
        train = german_train_dataset(
            budget=self.budget, spm_model=spm_model, spm_size=spm_size, partition_epoch=self.partition_epoch
        )
        de = get_mls_de_task(extended_vocab=self.extended_vocab, train_dataset=train, train_epoch_split=1)

        def _wrapped(*args, **kwargs):
            self.count += 1
            return de

        self._patch = unittest.mock.patch.object(_ls, "get_librispeech_task_raw_v2", _wrapped)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        if exc[0] is None:
            # ⚠ TWO calls, not one -- measured 2026-09-19, and the first version of this assert
            # (`== 1`) fired on it. This entry point builds the task **twice**: once in
            # `_train_asr_base_multigpu` itself (`:3095`, for the recog wiring) and once inside
            # `aed.train_exp`'s fallback (`aed.py:479-480`, for the training), because
            # `_train_asr_base_multigpu` does not forward `task=`. BOTH must be German, so the
            # meaningful check is that the patch was reached at all -- a silent zero is the failure
            # that matters, and it would train arm C on ENGLISH audio while reporting German WER.
            assert self.count >= 1, (
                "the German task patch was never reached -- arm C would train on ENGLISH audio"
                " and quietly answer a different question"
            )
        return False


# ⚠ Arm C's schedule is sized for ITS OWN corpus, not copied from arm B (§75).
# `_train_asr_base_multigpu`'s defaults are correct for LS-960 and catastrophic for 9 h: at
# `batch_size_feat=100_000` (= 16 M samples = 1,000 s/rank, 4,000 s per global step) a 9 h corpus is
# **~13 steps per EPOCH**, so arm B's nep=10 gave arm C ~125 optimizer updates in total -- about 80
# seconds of training, which would not move a ~100 M-parameter model and would have produced a
# broken baseline that reads as a win for arm B. Measured on the smoke: step 10 = 80.29% of epoch 1.
#
# Re-sized against the corpus: 12,500 frames = 125 s/rank ~= 8 utts/rank, 32 utts per global step,
# so ~69 steps/epoch and ~2,700 updates over 40 epochs -- the low-thousands that published 10 h
# finetune recipes use.
ARM_C_NEP = 40
ARM_C_BATCH_SIZE_FEAT = 12_500
# Peak LR matches arm B -- but note the schedule is `base_lr x piecewise_values`, so BOTH have to be
# copied. `_train_asr_base_multigpu` defaults `base_lr=0.5` while arm B passes 1.0, which silently
# halved arm C's effective peak (2.5e-4 vs 5e-4) while the recipe line read as matched.
ARM_C_BASE_LR = 1.0
ARM_C_PEAK_LR = ARM_B_PEAK_LR


def train_german_arm_c(
    *,
    prefix: str,
    winner_model,
    budget: str = "9h",
    name: Optional[str] = None,
    smoke: bool = False,
    peak_lr: Optional[float] = None,
    nep: Optional[int] = None,
    keep_epochs: Optional[Sequence[int]] = None,
    enc_lr_mult: Optional[float] = None,
    no_recog: bool = False,
):
    """
    **Arm C — the baseline.** MLS-de **paired audio** (1 h or 9 h), no text injection, starting from
    the same vocab-surgered winner as arm B.

    This is the plan's one controlled comparison: *text injection vs paired audio at the same audio
    budget*. Everything that is not the training data is therefore held fixed against arm B:

    * **same starting checkpoint** -- `get_surgered_winner_checkpoint(..., budget=budget)`.
    * **same output vocabulary** -- the extended 10,243-piece SPM, so both arms can write German
      orthography and both train the same three umlaut rows (by different routes; that is the point).
    * **same front-end** -- arm B passes `asr_logmel=True`, which *deletes* `feature_extraction` and
      falls back to the default log-mel filterbank (`:4033-4041`); `_train_asr_base_multigpu`'s
      default `feature_extraction=None` is that same default. Verified rather than assumed.
    * **same evaluation** -- `get_mls_de_task`, LM-free, sclite over normalised references.
    * **same schedule family** -- `nep` and `peak_lr` taken from arm B's constants.

    🟢 **Why `_train_asr_base_multigpu` and not `_train_tts_encoder`.** It is the purpose-built no-TTS
    sibling: *"Same architecture + 4-GPU / nep=25 / batched recog as the TTS-encoder runs, but trained
    on audio only (no TTS, no text data, default AED model def, standard train step)"*. Using it means
    the audio-only baseline is the codebase's own audio-only baseline rather than one I assembled,
    which is what keeps B vs C a comparison and not an artefact. `_train_tts_encoder` cannot serve
    here anyway: it always builds a text branch, and it asserts the ASR sub-dataset is an
    `OggZipDataset` (`:3862`) -- which is, separately, why the German audio is packed as an ogg zip.

    ⚠ The winner's checkpoint carries `pseudo_enc.*` tensors that this model does not define. RETURNN
    loads with ``strict=False`` (`returnn/torch/engine.py:1303`), so **extra** keys are ignored -- it
    is a *shape* mismatch that raises, and there is none here. Arm B needed the pseudo-encoder table
    widened (§64); arm C simply does not use it.
    """
    import contextlib

    from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_asr_base_multigpu

    from .winner_plus_tts import winner_checkpoint

    def _optimizer_param_groups_custom_lr_multiplier():
        # Imported lazily and returned as the FUNCTION object: RETURNN's config serialiser emits it
        # by module+name, which is what `optimizer.param_groups_custom` expects.
        from i6_experiments.users.zeyer.returnn.updater.lr_multiplier import (
            optimizer_param_groups_custom_lr_multiplier,
        )

        return optimizer_param_groups_custom_lr_multiplier

    assert budget in ("1h", "9h"), budget
    # ⚠ Sweep knobs, emitted CONDITIONALLY (the Loquacious pattern). At their defaults every value and
    # the job NAME are byte-identical to before they existed, so adding them re-hashed nothing --
    # verified by graph diff. A non-default value goes into the name, so sweep arms cannot collide.
    _peak_lr = ARM_C_PEAK_LR if peak_lr is None else peak_lr
    _nep = ARM_C_NEP if nep is None else nep
    if name is None:
        name = f"german-armC-audio-{budget}-nEp{_nep}"
        if peak_lr is not None:
            name += f"-lr{_peak_lr:g}"
        if keep_epochs is not None:
            name += "-keepall"
        if enc_lr_mult is not None:
            name += f"-encLr{enc_lr_mult:g}"
    de_ckpt = get_surgered_winner_checkpoint(winner_checkpoint(winner_model), budget=budget)

    with (
        PatchTaskAllToGerman(budget=budget, extended_vocab=True),
        CaptureRegisteredOutputs(match="recog_results") as _cap,
        # Sweep arms build NO recog: keeping a checkpoint and decoding it are the same knob, and a
        # recog costs ~2.5x its training (§90). §92a showed dev loss ranks epochs exactly as WER,
        # so sweep on the free curve and recog only the winner.
        SuppressRecog() if no_recog else contextlib.nullcontext(),
    ):
        _exp = _train_asr_base_multigpu(
            name,
            prefix=prefix,
            with_ctc_lm_recog=False,  # LM-free by design (§11), as arm B
            base_lr=ARM_C_BASE_LR,
            peak_lr=_peak_lr,
            nep=_nep,
            batch_size_feat=ARM_C_BATCH_SIZE_FEAT,
            behavior_version=29,
            extra_config_updates={
                "import_model_train_epoch1": de_ckpt,
                # 🔴 Without an explicit `keep`, `default_returnn_keep_epochs(nep)` governs which
                # checkpoints survive AND which epochs the recog scores -- for nep<=50 that is
                # {5,10,20,nep} (`model_with_checkpoints.py:90-100`). Arm C's dev optimum is ~epoch 3
                # (§78), so the best checkpoint is DELETED and only overfit ones ever get scored.
                # Pass `keep_epochs` to bracket the optimum.
                **({"cleanup_old_models": {"keep_last_n": 5, "keep": list(keep_epochs)}} if keep_epochs else {}),
                # Down-weight (or freeze, at 0.0) the ENCODER's learning rate. §82: arm C memorises
                # 9 h in ~3 epochs, and lowering the *global* LR slows the output layer too -- which
                # is the one part that genuinely must learn, since it carries three brand-new symbols.
                # Targeting the encoder is the standard low-resource move, and the plan's own
                # "Forgetting control Phase 1"; `lr_multiplier.py` fnmatches FULL parameter names, and
                # this checkpoint has 503 `encoder.*` params vs 81 `decoder.*` (verified).
                **(
                    {
                        "optimizer.param_groups_custom": _optimizer_param_groups_custom_lr_multiplier(),
                        "optimizer.learning_rate_multipliers_by_patterns": {"encoder.*": enc_lr_mult},
                    }
                    if enc_lr_mult is not None
                    else {}
                ),
                # ⚠ ALWAYS explicit: train_v4's default is 80 h against a 12 h QOS cap.
                # Arm C's data is 1-9 h, i.e. ~100x smaller than arm B's epoch, so it is cheap.
                "__time_rqmt": 1 if smoke else 4,
            },
            # ⚠ `_train_asr_base_multigpu` sets `__train_audio_preprocess` itself, and on this arm it
            # DOES NOTHING: `aed.train_exp` assigns it to `task.train_dataset.train_audio_preprocess`
            # (`aed.py:485-488`), an attribute `DatasetConfigStatic` does not have and never reads,
            # and it is popped before the config is hashed. That is exactly why speed perturbation is
            # set directly on the dataset in `german_train_dataset` instead.
            # (Deleting the key here is NOT the fix -- `dict_update_deep` raises
            # `KeyError: '__train_audio_preprocess'` on a delete of a key absent from the base config.)
        )

    if no_recog:
        # Sweep arm: no recog, so no summary to attach a sink to. Its result is the dev curve in
        # `learning_rates`, read with analysis/parse_lr.py (§92a validated that proxy).
        return _exp
    _attach_german_result_sink(
        _cap.captured,
        tag=name,
        note=(
            f"Arm C -- BASELINE: MLS-de {budget} PAIRED AUDIO, no text injection, from the surgered"
            f" winner. The comparison that matters is vs arm B at the same audio budget."
            " Arm A (103.93%) is context only. NOT identical-except-data: arm C uses its own"
            " recipe's optimizer/batching (see backlog 75a)."
            " WARNING: 9 h memorises in ~3 epochs (backlog 78) -- check the dev curve picked a"
            " pre-overfitting epoch before quoting this."
        ),
    )
    return _exp


def train_german_arm_b(
    *, prefix: str, winner_model, budget: str = "9h", name: Optional[str] = None, smoke: bool = False
):
    """
    **Arm B — the claim.** English LS-960 audio ⇄ **German text injection**, starting from the
    vocab-surgered winner.

    What differs from `winner_plus_tts.train_winner_plus_tts`, and nothing else does:

    * **model_def** -> `aed_glowtts_model_def_de` (57-wide German pseudo vocab) via
      :class:`PatchModelDefToGerman`, since `_train_tts_encoder` hardcodes it.
    * **phoneme resources** -> German lexicon / vocab / size via :class:`PatchGlowTtsToGerman`.
    * **text branch** -> the German injection corpus via :class:`PatchTextBranchToGerman`
      (the ASR branch stays ENGLISH — that is the arm).
    * **task** -> English training data, MLS-de evaluation, extended 10,243 SPM, via
      :class:`PatchTaskEvalToGerman`.
    * **acoustic prior** -> the German mean-log-mel + duration tables.
    * `import_model_train_epoch1` -> the surgered checkpoint (10,243-wide output).
    * `with_ctc_lm_recog=False` — there is no German LM, and the plan runs LM-free by design (§11);
      leaving it on would pull the English trafo LM and an English labelwise prior into the recog and
      produce a meaningless number.

    :param smoke: cap the walltime at 1 h (the plan's verification step 1). ``__time_rqmt`` is **not
        hashed** (`i6_core/returnn/training.py:547-561`), so the smoke and the full run are the SAME
        job -- it simply stops early and **resumes** once the cap is raised. Nothing is thrown away.
        ⚠ Raising it later does not reach a job already queued; resubmit (CLAUDE.md, rqmt note).

    ⚠ `budget` selects the acoustic prior only. **"9h" is the usable one** — the 1 h duration table is
    21.8% floor-collapsed and its spectra are truncation-biased for affricates/stops (§18, §21).
    """
    import returnn.frontend as rf
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon
    from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_tts_encoder

    from .winner_plus_tts import winner_checkpoint

    assert budget in ("1h", "9h"), budget
    name = name or f"german-armB-textinj-{budget}-nEp{ARM_B_NEP}"
    de_ckpt = get_surgered_winner_checkpoint(winner_checkpoint(winner_model), budget=budget)
    tables = german_pseudo_enc_config(budget=budget)

    with (
        PatchModelDefToGerman(),
        PatchGlowTtsToGerman(budget=budget),
        PatchTextBranchToGerman(corpus_files=[get_german_injection_text_filtered(budget=budget)]),
        PatchTaskEvalToGerman(extended_vocab=True, extended_train_vocab=True),
        CaptureRegisteredOutputs(match="recog_results") as _cap,
    ):
        _exp = _train_tts_encoder(
            name,
            prefix=prefix,
            with_ctc_lm_recog=False,  # LM-free by design; see docstring
            text_train_epoch_split=75,
            batch_size_audio_frames=70_000,
            batch_size_phon=6_000,
            max_phon_len=300,
            asr_logmel=True,
            pseudo_speech_enc=True,
            pseudo_enc_frozen_table=tables["pseudo_enc_frozen_table"],
            pseudo_enc_duration_table=tables["pseudo_enc_duration_table"],
            pseudo_enc_duration_sigma=0.45,
            pseudo_enc_duration_scale=0.7,
            pseudo_enc_max_len_factor=10,
            train_seq_ordering="random",
            pseudo_enc_lerp=True,
            pseudo_enc_blank_duration_range=(0, 0),
            pseudo_enc_specaug_max_width=6,
            single_stream=True,
            interleave_gumbel_scale=1.0,
            glow_tts_add_silence_between_words=0.15,
            base_lr=1.0,
            peak_lr=ARM_B_PEAK_LR,
            nep=ARM_B_NEP,
            behavior_version=29,
            pseudo_enc_frontend_concat=True,
            extra_config_updates={
                "optimizer.class": rf.build_dict(Muon)["class"],
                "packed_tensors": True,
                "torch_distributed": {"reduce_type": "grad_explicit"},
                "batch_size": None,
                "packed_batch_size": {"data": 11_200_000, "classes": 5_000, "phonemes": 6_000},
                "batching": "random",
                "optimizer.weight_decay": 0.027,
                "specaugment_num_spatial_mask_factor": 50,
                "specaugment_steps": (1850, 5550, 9250),
                # Read by aed_glowtts_model_def_de at JOB-RUN time; without it the model is built with
                # the English 44-wide pseudo vocab and the German table fails its shape assert.
                "pseudo_enc_phoneme_vocab_size": tables["pseudo_enc_phoneme_vocab_size"],
                "import_model_train_epoch1": de_ckpt,
                # ⚠ ALWAYS set this explicitly. `train_v4`'s default is **80 h**
                # (`train_v4.py:225-234`) while the QOS caps at 12 h, so the default makes the job
                # unschedulable. Measured 2026-09-19: ~70 min/epoch at 0.94 s/step, so 10 epochs
                # ~= 11.7 h -- it fits ONE allocation, and `stop_for_resubmission_when_low_time_left`
                # (already set) handles the boundary if it does not.
                # smoke = 1 h: verifies what graph-building cannot -- dataset init, the German table
                # load, and the 10,243-wide checkpoint actually loading into the model.
                "__time_rqmt": 1 if smoke else 11,
            },
            extra_config_deletes=["optimizer.epsilon"],
        )

    _attach_german_result_sink(
        _cap.captured,
        tag=name,
        note=(
            f"Arm B -- THE CLAIM: English LS-960 audio + German TEXT injection ({budget} acoustic"
            " prior), from the surgered winner. Compare against arm C at the same audio budget;"
            " arm A (103.93%) is context only and cannot write German orthography."
            " Before quoting: confirm the 3 umlaut rows trained (analysis/umlaut_check.py, backlog 72)."
        ),
    )
    return _exp
