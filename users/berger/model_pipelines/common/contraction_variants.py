"""Finite reference-independent LM renderings; never rewrite an AM hypothesis.

Only explicit negative/auxiliary contractions. Ambiguous 's, 'd, possessives,
numbers, hyphens and titles are untouched. Three whole-string styles are not
an exhaustive search over combinations of word-level choices.
"""

import re
from collections import Counter


EXPAND = {
    "can't": "cannot", "won't": "will not",
    **{stem + "n't": full + " not" for stem, full in (
        ("do", "do"), ("does", "does"), ("did", "did"), ("is", "is"), ("are", "are"),
        ("was", "was"), ("were", "were"), ("have", "have"), ("has", "has"), ("had", "had"),
        ("would", "would"), ("could", "could"), ("should", "should"), ("must", "must"),
        ("need", "need"), ("might", "might"))},
    "i'm": "i am",
    **{p + "'re": p + " are" for p in ("you", "we", "they")},
    **{p + "'ve": p + " have" for p in ("i", "you", "we", "they")},
    **{p + "'ll": p + " will" for p in ("i", "you", "he", "she", "it", "we", "they")},
}
CONTRACT = {v: k for k, v in EXPAND.items()}
CONTRACT["can not"] = "can't"


def transform(text, mapping):
    keys = sorted(mapping, key=lambda s: (-len(s), s))
    # Flexible spacing between phrase words, but no matches inside names or
    # chained contractions. Curly apostrophes are recognized only for core rules.
    pattern = r"(?<![\w'’])(?:" + "|".join(re.escape(k).replace(r"\ ", r"\s+").replace("'", "['’]") for k in keys) + r")(?![\w'’])"
    counts = Counter()

    def replace(match):
        original = match[0]
        key = " ".join(original.lower().replace('’', "'").split())
        result = mapping[key]
        counts[key + " -> " + result] += 1
        if original.isupper():
            return result.upper()
        return result[:1].upper() + result[1:] if original[0].isupper() else result

    return re.sub(pattern, replace, text, flags=re.IGNORECASE), dict(counts)


def variants(text):
    result = dict(literal=dict(text=text, rules={}))
    # Prefer "they won't" over overlapping "they'll not". This is one
    # deterministic style, not every locally possible contraction combination.
    negative = {k: v for k, v in CONTRACT.items() if v.endswith("n't")}
    auxiliary = {k: v for k, v in CONTRACT.items() if k not in negative}
    surface, first = transform(text, negative)
    surface, second = transform(surface, auxiliary)
    result['contracted'] = dict(text=surface, rules=dict(Counter(first) + Counter(second)))
    surface, counts = transform(text, EXPAND)
    result['expanded'] = dict(text=surface, rules=counts)
    return result


def choose(candidates, *, mode, n, scale, word_penalty):
    """Fixed AM/first-pass score and AM word count, regardless of LM rendering."""
    import math
    if mode not in ('literal', 'contracted', 'expanded', 'best') or n < 1 or not candidates or scale < 0:
        raise ValueError('Invalid contraction selection')
    if not all(math.isfinite(v) for v in (scale, word_penalty)):
        raise ValueError('Nonfinite selection parameters')
    rows = []
    for i, candidate in enumerate(candidates[:n]):
        scores = candidate['contraction_scores']
        names = ('literal', 'contracted', 'expanded') if mode == 'best' else (mode,)
        name = min(names, key=lambda k: scores[k]['nll'])
        nll = scores[name]['nll']
        first = candidate['first_pass_cost']
        if not all(math.isfinite(v) for v in (nll, first)):
            raise ValueError('Nonfinite cached scores')
        reward = word_penalty * len(candidate['text'].split())
        rows.append(dict(index=i, rendering=name, nll=nll, word_cost=reward,
                         cost=first + scale * nll + reward))
    return min(rows, key=lambda r: r['cost'])
