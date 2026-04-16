#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Copied and modified from:
https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py
"""

import argparse
import numpy as np
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="converts words to phones adding optional silences around in between words"
    )
    parser.add_argument(
        "--sil-prob",
        "-s",
        type=float,
        default=0,
        help="probability of inserting silence between each word",
    )
    parser.add_argument(
        "--surround",
        action="store_true",
        help="if set, surrounds each example with silence",
    )
    parser.add_argument(
        "--lexicon",
        help="lexicon to convert to phones",
        required=True,
    )
    parser.add_argument(
        "--seq-tags-file",
        type=str,
        help="file containing seq tags",
        default=None,
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    sil_prob = args.sil_prob
    surround = args.surround
    sil = "<SIL>"

    seq_tags_file = args.seq_tags_file
    if seq_tags_file is not None:
        with open(seq_tags_file, "r", encoding="utf-8") as f:
            seq_tags = list(line.strip() for line in f)

    wrd_to_phn = {}

    with open(args.lexicon, "r") as lf:
        for line in lf:
            items = line.rstrip().split()
            assert len(items) > 1, line
            assert items[0] not in wrd_to_phn, items
            wrd_to_phn[items[0]] = items[1:]

    with open("seq-tags-after-phonemize.txt", "w") as f:
        for line, seq_tag in zip(sys.stdin, seq_tags) if seq_tags_file else ((line, None) for line in sys.stdin):
            words = line.strip().split()

            if not all(w in wrd_to_phn for w in words):
                continue

            phones = []
            if surround:
                phones.append(sil)

            sample_sil_probs = None
            if sil_prob > 0 and len(words) > 1:
                sample_sil_probs = np.random.random(len(words) - 1)

            for i, w in enumerate(words):
                phones.extend(wrd_to_phn[w])
                if (
                    sample_sil_probs is not None
                    and i < len(sample_sil_probs)
                    and sample_sil_probs[i] < sil_prob
                ):
                    phones.append(sil)

            if surround:
                phones.append(sil)
            print(" ".join(phones))

            if seq_tag is not None:
                f.write(f"{seq_tag}\n")


if __name__ == "__main__":
    main()
