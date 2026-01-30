#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Copied and modified from:
https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py
"""

import argparse
import fasttext as ft
import os
import regex
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="reads text from stdin and outputs normalized, lid-filtered version to stdout"
    )
    parser.add_argument(
        "--fasttext-model",
        help="path to fasttext model",
        default="lid.187.bin",
    )
    parser.add_argument("--lang", help="language id", required=True)
    parser.add_argument(
        "--lid-threshold",
        type=float,
        help="threshold for this lang id probability",
        default=0.4,
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
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")

    lg = args.lang.lower()
    lg_label = f"__label__{lg}"
    thresh = args.lid_threshold
    seq_tags_file = args.seq_tags_file
    if seq_tags_file is not None:
        with open(seq_tags_file, "r", encoding="utf-8") as f:
            seq_tags = list(line.strip() for line in f)

    if os.path.exists(args.fasttext_model):
        model = ft.load_model(args.fasttext_model)
    else:
        print(
            f"fasttext language id model {args.fasttext_model} not found. Proceeding without language filtering. "
            f"To enable language filtering, please download the latest language id model "
            f"from https://fasttext.cc/docs/en/language-identification.html",
            file=sys.stderr,
        )
        model = None

    with open("seq-tags-after-norm-and-filter.txt", "w") as f:
        for line, seq_tag in zip(sys.stdin, seq_tags) if seq_tags_file else ((line, None) for line in sys.stdin):
            line = line.strip()
            line = filter_r.sub(" ", line)
            line = " ".join(line.split())

            if model is not None:
                lid, prob = model.predict(line, k=100)
                try:
                    target_idx = lid.index(lg_label)
                except ValueError:
                    continue
                if target_idx == 0 or prob[target_idx] >= thresh:
                    print(line)
                    if seq_tag is not None:
                        f.write(f"{seq_tag}\n")
            else:
                print(line)
                if seq_tag is not None:
                    f.write(f"{seq_tag}\n")


if __name__ == "__main__":
    main()
