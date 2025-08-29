#!/usr/bin/env python3
import argparse
import gzip
import xml.etree.ElementTree as ET
from ast import literal_eval
from typing import Dict, Iterable, List, Set


# --- IO helpers ---
def open_maybe_gz(path: str):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") \
        else open(path, "rt", encoding="utf-8", newline="")


def load_py_dict_literal(path: str) -> Dict[str, int]:
    with open(path, "rt", encoding="utf-8") as f:
        return literal_eval(f.read())


def write_vocab_dict_as_py(path: str, mapping: Dict[str, int]):
    with open(path, "wt", encoding="utf-8") as f:
        f.write("{\n")
        for k, v in mapping.items():
            f.write(f"  {repr(k)}: {int(v)},\n")
        f.write("}\n")


# --- Lexicon streaming ---
def iter_lex_symbols(lex_path: str) -> Iterable[str]:
    """
    Stream <symbol>...</symbol> contents in document order (memory friendly).
    """
    with open_maybe_gz(lex_path) as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag.endswith("symbol"):
                text = (elem.text or "").strip()
                if text:
                    yield text
            elem.clear()


# --- Main logic ---
LEX_SPECIAL_MAP = {
    "[SILENCE]": "<blank>",
    "[NOISE]": "<noise>",
    "[MUSIC]": "<music>",
}

EXCLUDE_FROM_PREPEND: Set[str] = {"<sep>", "<blank>", "<noise>", "<music>"}


def infer_other_specials(token2id: Dict[str, int]) -> List[str]:
    """
    Return original-order specials (tokens that look like <...>),
    except those in EXCLUDE_FROM_PREPEND.
    """
    # keep original order by sorting by original id
    toks_in_order = [t for t, _ in sorted(token2id.items(), key=lambda kv: kv[1])]
    specials = []
    for t in toks_in_order:
        if t.startswith("<") and t.endswith(">") and t not in EXCLUDE_FROM_PREPEND:
            specials.append(t)
    return specials

KEEP_SPECIAL = False
def build_vocab_from_lexicon(lex_path: str, original_vocab: Dict[str, int]) -> Dict[str, int]:
    # 1) specials to prepend from original vocab (excluding <sep>, <blank>, <noise>, <music>)
    prepend_specials = infer_other_specials(original_vocab)

    # 2) walk lexicon symbols, map 3 specials, otherwise keep as-is
    out_tokens: List[str] = []
    seen: Set[str] = set()

    # first, prepend specials (no dups)
    for sp in prepend_specials:
        if sp not in seen:
            out_tokens.append(sp)
            seen.add(sp)

    # then, lexicon-driven tokens
    for sym in iter_lex_symbols(lex_path):
        tok = LEX_SPECIAL_MAP.get(sym, sym)  # map [SILENCE]/[NOISE]/[MUSIC]
        if tok not in seen:
            out_tokens.append(tok)
            seen.add(tok)

    # build token->index mapping
    if KEEP_SPECIAL:
        return {t: i for i, t in enumerate(out_tokens)}
    else:
        return {t: i - 3 for i, t in enumerate(out_tokens) if i > 2}


def main():
    ap = argparse.ArgumentParser(
        description="Create a new vocab (token->index) from lexicon order, "
                    "mapping [SILENCE]/[NOISE]/[MUSIC] to <blank>/<noise>/<music>, "
                    "prepending other specials (except <sep>) from the original vocab, "
                    "and ignoring any original tokens not in lexicon."
    )
    ap.add_argument("--lexicon", required=True, help="Path to lexicon.xml or lexicon.xml.gz")
    ap.add_argument("--orig-vocab", required=True, help="Original vocab (Python dict literal: token->id)")
    ap.add_argument("--out", required=True, help="Output path for new vocab (Python dict literal)")
    args = ap.parse_args()

    orig = load_py_dict_literal(args.orig_vocab)
    new_mapping = build_vocab_from_lexicon(args.lexicon, orig)
    write_vocab_dict_as_py(args.out, new_mapping)
    print(f"Wrote new vocab with {len(new_mapping)} entries to: {args.out}")


if __name__ == "__main__":
    main()
