"""Turn base-Moshi rollouts into a rehearsal corpus (backlog B6).

The problem this exists for: fine-tuning on the synthetic assistant corpus teaches the new voice and
*loses* the factual recall the base model had. The data-side counterpart to a schedule fix is
**rehearsal in the model's own distribution** -- take what base Moshi itself says, and keep saying it
while learning the new voice.

Shape of the pipeline (all stages but this one already existed)::

    TriviaQA-train questions
      -> LLMPreprocess                      (make them speech-digestible)
      -> ChatterboxSingleSpeakerInference    (user audio)
      -> SpeechInference(mode="knowledge")   (base Moshi replies + its INNER MONOLOGUE text)
      -> MoshiRolloutToDialogue              <-- this module
      -> HfDialogueCleaner -> make_speech_pipeline   (re-TTS in the NEW assistant voice -> annotate)

Why the monologue and not ASR of the reply: ``run_pairs`` already dumps the model's own text stream
at the 12.5 Hz frame rate. That IS the text the model produced -- a Whisper transcript would be a
guess at it, and would launder the model's distribution through a third model.

Why re-TTS rather than reuse the generated audio: the whole point is to pair *base Moshi's text* with
*the new assistant voice*, so the audio has to be re-synthesised. Reusing the rollout audio would
teach the old voice, which is the opposite of the objective.

**The text needs filtering, not rewriting.** Measured on 1,000 real base-Moshi rollouts
(``output/benchmark/moshi_family/moshi_output``), three defects show up, and every one of them would
be trained on verbatim if this job did not exist:

1. *Every* reply opens with a greeting ("Hello, how can I help you?"), because the rollout feeds
   ``lead_in_s`` of silence before the question. Rendered after the user's question in a 2-turn
   dialogue, that greeting lands in the wrong place and teaches the assistant to greet mid-
   conversation. -> :func:`strip_lead_greeting`.
2. Some replies degenerate into repetition ("It's a unique flavor. It's a very unique flavor. It's a
   very sweet flavor."). -> :func:`repeat_ratio` / :func:`sentence_repeat_ratio`.
3. The inner monologue drops characters ("I' not sure", "It a mix"). This is the one that matters
   most: the text head is exactly what we are trying to preserve, so training on mangled text works
   directly against the goal. -> :func:`glitch_rate`.

Rows are **dropped, never repaired**. A repaired row is no longer the model's own distribution, which
is the only reason to prefer self-generated data over more synthetic data in the first place. The one
exception is trimming a trailing incomplete sentence (:func:`trim_to_last_sentence`), which removes a
window-truncation artifact by deleting text rather than inventing it.

**Correctness is a column, not a filter.** ``alias_match`` against the gold answer is recorded per
row and never used to drop anything. Rehearsal wants the model's distribution *including* its errors
-- training on a wrong answer the model already gives does not make it more wrong, it anchors it --
and keeping the column means a correct-only variant is a downstream selection rather than a
regeneration.
"""

from __future__ import annotations

import json
import re

from sisyphus import Job, Task, tk

from datasets import Dataset, load_from_disk

from .clip_store import COL_INDEX, COL_MONOLOGUE, is_clip_dataset
from .quick_knowledge_eval import alias_match

#: Openers observed on every one of 1,000 base-Moshi rollouts. The model greets into the
#: ``lead_in_s`` silence before the question arrives, so the greeting is an artifact of the ROLLOUT
#: HARNESS, not of the reply -- which is why removing it is not tampering with the model's output.
#: Matches a greeting sentence, optionally followed by one pleasantry sentence ("How are you doing?").
_GREETING_RE = re.compile(
    r"^\s*(?:hello|hi|hey|good\s+(?:day|morning|afternoon|evening)|greetings)\b[^.?!]*[.?!]+\s*"
    r"(?:(?:how|what)\b[^.?!]*[.?!]+\s*)?",
    re.IGNORECASE,
)

#: A word left hanging on a bare apostrophe: "I' not sure", "It' a unique flavor", "don' know".
#: The 12.5 Hz text stream drops the tail of a contraction. Unrepairable ("I'" is "I'm" or "I've"
#: or "I'll" and nothing in the text says which), so rows carrying too many are dropped.
_GLITCH_RE = re.compile(r"\b\w+'(?=\s|$)")

#: A sentence beginning in lower case, i.e. the model dropped its opening word: *"...in 2013. took
#: over from Jupp Heynckes"*, *"...called giltwood. involves applying gold leaf"*. A distinct defect
#: from the truncated contraction above -- the loss is a whole word, and nothing marks where -- and
#: one the 1,000-rollout sample shows is roughly as common. Digits and capitals are excluded, so
#: "No. 1, Sabalenka" and "U.S. Army" do not trip it.
_DROPPED_WORD_RE = re.compile(r"(?<=[.?!])\s+(?=[a-z])")

#: The same dropped-opening-word defect at the very START of the reply, where there is no preceding
#: terminator for :data:`_DROPPED_WORD_RE` to anchor on: *"...how is your day? believe it has its
#: origin in..."*, *"...how is it going? is the Supreme Governor of the Church of England?"*. Here the
#: salvageable text is the SUFFIX, so the fragment is dropped from the front rather than the tail
#: being cut off.
_LEADING_FRAGMENT_RE = re.compile(r"^[a-z][^.?!]*[.?!]+\s*")

_WORD_RE = re.compile(r"[\w']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")
#: Trailing text after the last sentence terminator -- what the capture window cut off mid-word.
#: The terminator is CAPTURED and put back verbatim: substituting a literal "." for the whole match
#: would silently turn "Is it a salad? it a" into "Is it a salad." -- a rewrite of the model's
#: punctuation, in a module whose one invariant is that it only ever deletes.
_TAIL_RE = re.compile(r"([.?!])[^.?!]*$")


def strip_lead_greeting(text: str) -> str:
    """Remove the rollout-harness greeting from the front of a monologue."""
    return _GREETING_RE.sub("", text, count=1).strip()


def trim_to_last_sentence(text: str) -> str:
    """Drop everything after the final sentence terminator.

    ~29% of base-Moshi replies are still speaking when the capture window closes, so the monologue
    ends mid-sentence. Rendering that trains the assistant to stop mid-sentence. Trimming deletes the
    fragment rather than guessing at its completion; a reply with no terminator at all trims to empty
    and is then caught by the ``min_words`` floor.
    """
    text = text.strip()
    if not text:
        return ""
    if text[-1] in ".?!":
        return text
    trimmed = _TAIL_RE.sub(r"\1", text)
    return "" if trimmed == text else trimmed.strip()


def cut_at_glitch(text: str) -> str:
    """Truncate at the first dropped contraction, keeping the clean prefix.

    Dropping the whole reply over one glitch throws away good text: measured on the 1,000-rollout
    sample, replies like *"Anaphylaxis is a severe, life threatening reaction ... swelling in various
    parts of the body. It' important to ..."* are correct and well-formed right up to the defect,
    which sits near the capture-window edge. Cutting keeps the good prefix and can never keep bad
    text, because everything after the first glitch is discarded unread. A reply that is broken from
    the start cuts down to nothing and is then caught by the ``min_words`` floor.
    """
    m = _GLITCH_RE.search(text)
    return text[: m.start()].strip() if m else text


def drop_leading_fragment(text: str) -> str:
    """Remove leading sentences whose opening word the model dropped.

    Loops, because the defect clusters: *"is it made of? it a salad? Okay, it's a salad."* needs two
    passes to reach well-formed text. Each pass strictly shortens the string, so it terminates.
    """
    while text[:1].islower():
        shorter = _LEADING_FRAGMENT_RE.sub("", text, count=1).strip()
        if shorter == text:  # no terminator to cut at -- the whole reply is one fragment
            return ""
        text = shorter
    return text


def cut_at_dropped_word(text: str) -> str:
    """Truncate at the first sentence that starts mid-thought (see :data:`_DROPPED_WORD_RE`)."""
    m = _DROPPED_WORD_RE.search(text)
    return text[: m.start()].strip() if m else text


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def glitch_rate(text: str) -> float:
    """Fraction of words that are a truncated contraction."""
    w = words(text)
    return len(_GLITCH_RE.findall(text)) / len(w) if w else 0.0


def repeat_ratio(text: str, n: int = 5) -> float:
    """1 - (distinct n-grams / n-grams). 0.0 for text short enough to have no repeats to find."""
    w = [x.lower() for x in words(text)]
    if len(w) < 2 * n:
        return 0.0
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def sentence_repeat_ratio(text: str) -> float:
    """Fraction of sentences that duplicate an earlier one (normalised, punctuation-insensitive).

    Catches the degenerate mode n-grams miss: near-identical SHORT sentences ("It's a unique flavor."
    / "It's a very unique flavor.") whose 5-grams differ by one inserted word.
    """
    sents = [" ".join(words(s)).lower() for s in _SENT_SPLIT_RE.split(text)]
    sents = [s for s in sents if s]
    if len(sents) < 2:
        return 0.0
    return 1.0 - len(set(sents)) / len(sents)


def clean_monologue(
    text: str,
    *,
    strip_greeting: bool = True,
    cut_glitch: bool = True,
    cut_dropped_word: bool = True,
    trim_tail: bool = True,
) -> str:
    """The full text-side cleanup, in the order the job applies it.

    Order matters: the greeting goes first (it can itself contain a defect), then the two cuts, then
    the sentence trim to clean up whatever fragment they left. Applying the cuts in sequence keeps
    the EARLIER of the two cut points, which is what we want -- each shortens the string the next
    one searches.

    Every step only ever DELETES text; nothing here rewrites the model's words. That is the whole
    point of preferring self-generated data, and a "repair" would quietly give it up.
    """
    if strip_greeting:
        text = strip_lead_greeting(text)
    if cut_dropped_word:
        # Front first: a leading fragment is removed, which can expose a *different* first sentence
        # for the forward cut below to judge.
        text = drop_leading_fragment(text)
    if cut_glitch:
        text = cut_at_glitch(text)
    if cut_dropped_word:
        text = cut_at_dropped_word(text)
    if trim_tail:
        text = trim_to_last_sentence(text)
    return text.strip()


class MoshiRolloutToDialogue(Job):
    """Base-Moshi rollouts + their source questions -> a two-turn dialogue corpus.

    Emits the same schema ``HfToDialogue`` does -- a ``dialogue`` column holding a JSON array of
    ``{speaker, text}`` -- so the existing clean -> TTS -> annotate pipeline consumes it unchanged.

    Clips are joined to questions **by integer index**, which is identity across the whole benchmark
    chain (see ``clip_store``). A row whose index has no clip, or whose clip has no monologue, is
    counted and dropped rather than silently skipped.
    """

    __sis_hash_exclude__ = {
        # Bump to force regeneration after changing the filter functions above (they are module
        # code, invisible to the hash, so an edit would otherwise silently reuse the old corpus --
        # the same trap `templates_version` exists for in hf_to_dialogue).
        "filters_version": 1,
    }

    def __init__(
        self,
        *,
        clips: tk.Path,
        questions: tk.Path,
        min_words: int = 8,
        max_words: int = 200,
        max_repeat_ratio: float = 0.15,
        max_sentence_repeat_ratio: float = 0.25,
        max_glitch_rate: float = 0.10,
        strip_greeting: bool = True,
        cut_glitch: bool = True,
        cut_dropped_word: bool = True,
        trim_tail: bool = True,
        filters_version: int = 1,
    ):
        self.clips = clips
        self.questions = questions
        self.min_words = min_words
        self.max_words = max_words
        self.max_repeat_ratio = max_repeat_ratio
        self.max_sentence_repeat_ratio = max_sentence_repeat_ratio
        # Applied to the reply BEFORE the glitch cut: with cutting on, the surviving text is glitch-
        # free by construction, so this is not a defect check but a "is this reply broken throughout?"
        # check -- one that decides whether salvaging a clean prefix is even meaningful.
        self.max_glitch_rate = max_glitch_rate
        self.strip_greeting = strip_greeting
        self.cut_glitch = cut_glitch
        self.cut_dropped_word = cut_dropped_word
        self.trim_tail = trim_tail
        self.filters_version = filters_version

        self.out_hf = self.output_path("out_hf", directory=True)
        #: Per-reason drop counts + kept-row statistics. Read this BEFORE spending TTS on the
        #: corpus: a filter that is silently rejecting 80% of rows looks identical, from the
        #: manager, to one that is working.
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _monologues(self) -> dict[int, str]:
        """``{index: monologue}`` from either clip layout, without loading any audio.

        The arrow layout keeps the monologue as a column, so it is read directly; the legacy layout
        keeps it as ``<i>.txt`` beside the wav. Deliberately does NOT go through
        ``clip_store.open_clips`` -- that materialises audio, and at corpus scale (tens of thousands
        of 26 s replies) reading it on the login node would be hundreds of GB for text we already
        have.
        """
        import os

        path = self.clips.get()
        if is_clip_dataset(path):
            ds = load_from_disk(path)
            return {int(i): m for i, m in zip(ds[COL_INDEX], ds[COL_MONOLOGUE]) if m is not None}
        out = {}
        for name in os.listdir(path):
            if not name.endswith(".txt"):
                continue
            stem = name[:-4]
            if not stem.isdigit():
                continue
            with open(os.path.join(path, name), encoding="utf-8") as fh:
                out[int(stem)] = fh.read()
        return out

    def run(self):
        questions = load_from_disk(self.questions.get())
        monologues = self._monologues()
        assert monologues, f"no monologues found under {self.clips.get()!r}"
        print(f"[rehearsal] {len(questions)} questions, {len(monologues)} monologues", flush=True)

        drops: dict[str, int] = {}
        rows: dict[str, list] = {
            "dialogue": [],
            "question": [],
            "answer": [],
            "aliases": [],
            "category": [],
            "index": [],
            "correct": [],
            "quality": [],
        }
        kept_words: list[int] = []

        def drop(reason: str) -> None:
            drops[reason] = drops.get(reason, 0) + 1

        for i in range(len(questions)):
            raw = monologues.get(i)
            if raw is None:
                drop("no_clip")
                continue
            degreeted = strip_lead_greeting(raw) if self.strip_greeting else raw
            if glitch_rate(degreeted) > self.max_glitch_rate:
                drop("glitched")
                continue
            text = clean_monologue(
                raw,
                strip_greeting=self.strip_greeting,
                cut_glitch=self.cut_glitch,
                cut_dropped_word=self.cut_dropped_word,
                trim_tail=self.trim_tail,
            )
            n_words = len(words(text))
            if n_words < self.min_words:
                drop("too_short")
                continue
            if n_words > self.max_words:
                drop("too_long")
                continue
            if repeat_ratio(text) > self.max_repeat_ratio:
                drop("ngram_repeat")
                continue
            if sentence_repeat_ratio(text) > self.max_sentence_repeat_ratio:
                drop("sentence_repeat")
                continue

            ex = questions[i]
            binary, quality = alias_match(text, ex["answer"], ex.get("aliases") or [])
            rows["dialogue"].append(
                json.dumps(
                    [
                        {"speaker": "user", "text": ex["question"]},
                        {"speaker": "assistant", "text": text},
                    ]
                )
            )
            rows["question"].append(ex["question"])
            rows["answer"].append(ex["answer"])
            rows["aliases"].append(ex["aliases"] if isinstance(ex["aliases"], str) else json.dumps(ex["aliases"]))
            rows["category"].append(ex.get("category", "unknown"))
            rows["index"].append(i)
            rows["correct"].append(int(binary))
            rows["quality"].append(int(quality))
            kept_words.append(n_words)

        n_kept = len(kept_words)
        # An empty (or near-empty) corpus is a filter bug, not a legitimate result, and writing it
        # would send an expensive TTS job over nothing. Fail here, where the thresholds are in view.
        assert n_kept > 0, f"every row was dropped: {drops}"
        stats = {
            "n_questions": len(questions),
            "n_monologues": len(monologues),
            "n_kept": n_kept,
            "keep_fraction": n_kept / len(questions),
            "dropped": dict(sorted(drops.items())),
            "mean_assistant_words": sum(kept_words) / n_kept,
            "total_assistant_words": sum(kept_words),
            "correct_fraction": sum(rows["correct"]) / n_kept,
            # ~2.6 words/s of speech; the corpus target is stated in HOURS, so give the number in
            # the units the decision is made in rather than making the reader convert.
            "est_assistant_hours": sum(kept_words) / 2.6 / 3600,
            "thresholds": {
                "min_words": self.min_words,
                "max_words": self.max_words,
                "max_repeat_ratio": self.max_repeat_ratio,
                "max_sentence_repeat_ratio": self.max_sentence_repeat_ratio,
                "max_glitch_rate": self.max_glitch_rate,
                "strip_greeting": self.strip_greeting,
                "cut_glitch": self.cut_glitch,
                "cut_dropped_word": self.cut_dropped_word,
                "trim_tail": self.trim_tail,
            },
        }
        print(f"[rehearsal] {json.dumps(stats, indent=2)}", flush=True)
        with open(self.out_stats.get(), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        Dataset.from_dict(rows).save_to_disk(self.out_hf.get())
