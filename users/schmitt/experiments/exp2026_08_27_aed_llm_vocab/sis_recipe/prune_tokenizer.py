"""
Build a restricted HuggingFace tokenizer from the subset of ids a corpus actually uses.

WHY: the Loquacious transcripts, lowercased, use 39_558 of Qwen2's 151_646 ids (see
:mod:`.llm_vocab`). Training the AED against the full vocab would cost ~574M parameters of dead
weight, because the vocab dimension appears FIVE times in this model -- the (tied) decoder
embedding/logits, three unshared aux CTC heads, and one dec-aux head -- i.e. ~5123 parameters per
vocab entry. Measured: 1_285M params at V=151_646 vs 711M at V=39_558.

APPROACH (option B of the two we considered): emit a real, standalone tokenizer directory with the
vocab renumbered to 0..N-1. RETURNN's stock ``HuggingFaceTokenizer`` then reads it unchanged, and
the dataset / extern_data / recog-label-serialization / scoring paths all work with no custom
Vocabulary class -- which option A would have needed, and which RETURNN cannot resolve from a
config anyway (``Vocabulary.create_vocab`` looks the class up in ``globals()`` of its own module).

ON THE RENUMBERING: the new ids are NOT Qwen's ids. That is inherent to shrinking the output layer
(option A would renumber too); only keeping the full 151_646-wide layer avoids it. It is harmless
for the stated goal -- transferring the ENCODER, whose 406M parameters are entirely
vocab-independent (verified: 8 of 593 tensors depend on the vocab, none of them under
``encoder.``/``feature*``). Where the boundary is actually crossed -- initialising the decoder
embedding from Qwen's pretrained rows, reusing ``enc_aux_logits_*``, scoring against the LLM --
this job emits the mapping in both directions, and it is monotonic (kept ids in ascending original
order), so each crossing is a single gather/scatter::

    new_to_orig[j]        -> the Qwen id of new id j
    orig_to_new[qwen_id]  -> the new id, or -1 if the token was dropped

WHAT IS KEPT, and why more than just the used ids:

- the used ids themselves;
- the transitive MERGE CLOSURE. BPE builds a token by merging two shorter ones, and an
  intermediate may itself be unused; dropping it would make the merge unable to fire and silently
  change the segmentation. Measured on this corpus: +133 tokens;
- ALL 256 single-byte tokens. Qwen2 is byte-level BPE with ``byte_fallback: false`` and
  ``unk_token: null``, so a byte missing from the vocab makes any text containing it
  untokenizable -- a hard failure, not a degradation. Only 28 of the 256 occur in our corpus, so
  this adds 228 tokens as cheap insurance against unexpected eval text;
- the added special tokens (``<|endoftext|>`` etc.), renumbered to the end. ``<|endoftext|>``
  never appears in the transcripts (RETURNN's ``get_seq`` adds no specials for Qwen2) but is
  REQUIRED: it is the tokenizer's eos, and zeyer's ``_get_bos_idx`` falls back to the eos id for
  bos, so without it the AED has neither.

Total for Loquacious-large lowercased: 39_558 + 133 + 228 + 3 = 39_922.

The job verifies its own output: it re-tokenizes real corpus text with both tokenizers and asserts
``orig_to_new[original_ids] == pruned_ids`` for every sequence, so "the ids differ" is a checked
invariant rather than a latent footgun.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sisyphus import Job, Task, tk


class PruneHuggingFaceTokenizerJob(Job):
    """
    Restrict a byte-level BPE HuggingFace tokenizer to a given id subset, renumbered densely.

    :param tokenizer_dir: dir with ``tokenizer.json`` (+ ``tokenizer_config.json``).
    :param used_ids: text file, one original vocab id per line (``ExtractVocabUsageJob.out_used_ids``).
    :param counts: optional ``ExtractVocabUsageJob.out_counts`` (.npy). Needed for ``min_count``.
    :param min_count: keep only ids occurring at least this often. Default 1 = keep every used id.
        On this corpus a cutoff buys almost nothing (>=2 drops 845 of 39_558 ids while making 845
        real token occurrences unrepresentable), so the default is deliberately no cutoff.
    :param keep_all_byte_tokens: keep all 256 single-byte tokens even if unused. See the module
        docstring -- switching this off risks hard tokenization failures on unexpected text.
    :param verify_text_file: gzipped text, one sentence per line, used for the round-trip check.
    :param verify_num_lines: how many lines of it to check.
    :param lowercase_verify: lowercase the verification text (must match how the corpus is
        tokenized in training).
    """

    __sis_version__ = 1

    def __init__(
        self,
        *,
        tokenizer_dir: tk.Path,
        used_ids: tk.Path,
        counts: Optional[tk.Path] = None,
        min_count: int = 1,
        keep_all_byte_tokens: bool = True,
        verify_text_file: Optional[tk.Path] = None,
        verify_num_lines: int = 200_000,
        lowercase_verify: bool = True,
    ):
        super().__init__()
        assert min_count == 1 or counts is not None, "min_count > 1 needs counts"
        self.tokenizer_dir = tokenizer_dir
        self.used_ids = used_ids
        self.counts = counts
        self.min_count = min_count
        self.keep_all_byte_tokens = keep_all_byte_tokens
        self.verify_text_file = verify_text_file
        self.verify_num_lines = verify_num_lines
        self.lowercase_verify = lowercase_verify

        self.out_tokenizer_dir = self.output_path("tokenizer", directory=True)
        # new id j  -> original id            (int32 [N])
        self.out_new_to_orig = self.output_path("new_to_orig.npy")
        # original id -> new id, -1 if dropped (int32 [orig_vocab_size])
        self.out_orig_to_new = self.output_path("orig_to_new.npy")
        self.out_vocab_size = self.output_var("vocab_size")
        self.out_stats = self.output_path("stats.txt")

        self.rqmt = {"gpu": 0, "cpu": 2, "mem": 8, "time": 2}

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """run"""
        import gzip
        import itertools
        import json
        import os
        import shutil

        import numpy

        tok_dir = self.tokenizer_dir.get_path()
        with open(os.path.join(tok_dir, "tokenizer.json")) as f:
            tj = json.load(f)
        assert tj["model"]["type"] == "BPE", f"{self}: only BPE supported, got {tj['model']['type']!r}"

        vocab: Dict[str, int] = tj["model"]["vocab"]
        merges = tj["model"]["merges"]
        added = tj.get("added_tokens", [])
        id2tok = {i: t for t, i in vocab.items()}
        orig_total = len(vocab) + len(added)

        used = [int(line) for line in open(self.used_ids.get_path()) if line.strip()]
        if self.min_count > 1:
            cnt = numpy.load(self.counts.get_path())
            used = [i for i in used if cnt[i] >= self.min_count]

        # Seed: used tokens that live in the BPE vocab (added/special ids are handled separately).
        seed = {id2tok[i] for i in used if i in id2tok}
        n_used = len(seed)

        if self.keep_all_byte_tokens:
            seed |= {t for t in vocab if len(t) == 1}
        n_with_bytes = len(seed)

        # A merge "a b" produces "ab"; to keep "ab" usable we must keep a and b, transitively.
        pair = {}
        for m in merges:
            a, b = (m if isinstance(m, list) else m.split(" ", 1))
            pair[a + b] = (a, b)
        need, stack = set(seed), list(seed)
        while stack:
            t = stack.pop()
            p = pair.get(t)
            if p:
                for x in p:
                    if x not in need:
                        need.add(x)
                        stack.append(x)
        n_closed = len(need)

        # Ascending original id -> the mapping is monotonic, which makes it trivially invertible
        # and keeps the relative order of the vocab recognisable.
        kept_orig_ids: List[int] = sorted(vocab[t] for t in need)
        new_vocab = {id2tok[oid]: j for j, oid in enumerate(kept_orig_ids)}
        n_base = len(kept_orig_ids)

        kept_merges = []
        for m in merges:
            a, b = (m if isinstance(m, list) else m.split(" ", 1))
            if a in new_vocab and b in new_vocab and (a + b) in new_vocab:
                kept_merges.append(m)

        # Specials go after the base vocab, preserving their relative order.
        new_added = []
        added_sorted = sorted(added, key=lambda a: a["id"])
        added_new_ids = {}
        for k, a in enumerate(added_sorted):
            a2 = dict(a)
            a2["id"] = n_base + k
            added_new_ids[a["id"]] = a2["id"]
            new_added.append(a2)

        new_total = n_base + len(new_added)

        tj_new = dict(tj)
        tj_new["model"] = dict(tj["model"], vocab=new_vocab, merges=kept_merges)
        tj_new["added_tokens"] = new_added

        out_dir = self.out_tokenizer_dir.get_path()
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
            json.dump(tj_new, f, ensure_ascii=False)

        # tokenizer_config.json keys added_tokens_decoder BY ID (as strings) -> renumber those too.
        cfg_path = os.path.join(tok_dir, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                tc = json.load(f)
            atd = tc.get("added_tokens_decoder")
            if atd:
                tc["added_tokens_decoder"] = {str(added_new_ids[int(k)]): v for k, v in atd.items()}
            with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
                json.dump(tc, f, ensure_ascii=False, indent=2)
        for extra in ("special_tokens_map.json",):
            src = os.path.join(tok_dir, extra)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, extra))

        new_to_orig = numpy.array(kept_orig_ids + [a["id"] for a in added_sorted], dtype=numpy.int32)
        orig_to_new = numpy.full((orig_total,), -1, dtype=numpy.int32)
        orig_to_new[new_to_orig] = numpy.arange(new_to_orig.size, dtype=numpy.int32)
        numpy.save(self.out_new_to_orig.get_path(), new_to_orig)
        numpy.save(self.out_orig_to_new.get_path(), orig_to_new)
        self.out_vocab_size.set(int(new_total))

        # --- verification: the pruned tokenizer must segment IDENTICALLY, up to the remap --------
        n_checked = n_mismatch = 0
        if self.verify_text_file is not None:
            from tokenizers import Tokenizer

            t_orig = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
            t_new = Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))
            opener = gzip.open if self.verify_text_file.get_path().endswith(".gz") else open
            with opener(self.verify_text_file.get_path(), "rt", encoding="utf-8") as f:
                batch = []
                for line in itertools.islice(f, self.verify_num_lines):
                    s = line.rstrip("\n")
                    batch.append(s.lower() if self.lowercase_verify else s)
                    if len(batch) == 10_000:
                        n_c, n_m = _cmp(t_orig, t_new, batch, orig_to_new)
                        n_checked += n_c
                        n_mismatch += n_m
                        batch = []
                if batch:
                    n_c, n_m = _cmp(t_orig, t_new, batch, orig_to_new)
                    n_checked += n_c
                    n_mismatch += n_m
            assert n_mismatch == 0, (
                f"{self}: pruned tokenizer disagrees with the original on {n_mismatch} of "
                f"{n_checked} sequences -- the merge closure is incomplete"
            )

        lines = [
            f"original vocab (incl. added): {orig_total}",
            f"used ids (min_count={self.min_count}): {n_used}",
            f"+ all byte tokens: {n_with_bytes} (+{n_with_bytes - n_used})",
            f"+ merge closure: {n_closed} (+{n_closed - n_with_bytes})",
            f"base vocab kept: {n_base}",
            f"added/special tokens: {len(new_added)}",
            f"NEW VOCAB SIZE: {new_total}  ({100.0 * new_total / orig_total:.2f}% of original)",
            f"merges kept: {len(kept_merges)} of {len(merges)}",
            f"verification: {n_checked} sequences re-tokenized, {n_mismatch} mismatches",
        ]
        with open(self.out_stats.get_path(), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


def _cmp(t_orig, t_new, batch, orig_to_new):
    """:return: (num checked, num mismatching) for one batch of sentences"""
    import numpy

    eo = t_orig.encode_batch_fast(batch, add_special_tokens=False)
    en = t_new.encode_batch_fast(batch, add_special_tokens=False)
    n_mismatch = 0
    for a, b in zip(eo, en):
        mapped = orig_to_new[numpy.asarray(a.ids, dtype=numpy.int64)]
        if len(mapped) != len(b.ids) or not numpy.array_equal(mapped, numpy.asarray(b.ids, dtype=numpy.int32)):
            n_mismatch += 1
    return len(batch), n_mismatch
