"""Standalone CLI run inside the MFA apptainer image: force-align a corpus, then
query MFA's result DB into a single word_alignments.jsonl.gz keyed by sisyphus seq_tag."""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import sqlalchemy as sa

from montreal_forced_aligner import config as mfa_config
from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.data import WordType
from montreal_forced_aligner.db import File, Utterance, Word, WordInterval
from montreal_forced_aligner.command_line.utils import validate_model_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", required=True, type=Path,
                   help="MFA-format corpus directory (audio + .lab files).")
    p.add_argument("--seq-tag-map", required=True, type=Path,
                   help="JSON file mapping {File.name (audio basename stem): seq_tag}.")
    p.add_argument("--dictionary", default="english_us_arpa")
    p.add_argument("--acoustic-model", default="english_us_arpa")
    p.add_argument("--mfa-config", type=str, default="{}",
                   help="JSON dict of MFA config overrides (UPPERCASE keys, "
                        "matching module-level globals in montreal_forced_aligner.config).")
    p.add_argument("--out-alignments", required=True, type=Path,
                   help="Output path for word_alignments.jsonl.gz.")
    p.add_argument("--out-missing", required=True, type=Path,
                   help="Output path for missing.txt (one seq_tag per line for segments "
                        "MFA couldn't align).")
    p.add_argument("--out-counts", required=True, type=Path,
                   help="Output path for counts.json with {num_seqs, num_words}.")
    p.add_argument("--no-skip-missing", action="store_true",
                   help="Raise instead of recording missing seq_tags. Default is to skip.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seq_tag_map: dict[str, str] = json.loads(args.seq_tag_map.read_text())

    mfa_config.update_configuration(json.loads(args.mfa_config))

    # clean=True below rmtree's mfa's working dir on startup, so refuse to run if it resolves onto the corpus
    mfa_work_dir = Path(mfa_config.TEMPORARY_DIRECTORY) / args.corpus_dir.name
    if mfa_work_dir.resolve() == args.corpus_dir.resolve():
        raise SystemExit(
            f"corpus dir {args.corpus_dir} collides with MFA's working dir {mfa_work_dir}, "
            f"clean=True would delete the corpus. Stage the corpus outside MFA_ROOT_DIR."
        )

    # resolves a bare name to its on-disk path under MFA_ROOT_DIR
    acoustic_model_path = validate_model_arg(args.acoustic_model, "acoustic")
    dictionary_path = validate_model_arg(args.dictionary, "dictionary")

    subprocess.check_call(["mfa", "server", "init"])

    aligner = PretrainedAligner(
        corpus_directory=str(args.corpus_dir),
        dictionary_path=dictionary_path,
        acoustic_model_path=acoustic_model_path,
        clean=True,
    )

    try:
        aligner.align()
        num_seqs, num_words, missing = _query_and_write_jsonl(
            aligner, seq_tag_map, args.out_alignments
        )
    finally:
        # always tear down the mfa server/db, even if align() or the query raised
        aligner.cleanup()

    args.out_missing.write_text("".join(t + "\n" for t in missing))
    args.out_counts.write_text(
        json.dumps({"num_seqs": num_seqs, "num_words": num_words})
    )

    if missing and args.no_skip_missing:
        raise SystemExit(
            f"Missing alignments for {len(missing)} seq_tags "
            f"(first: {missing[0]!r}); drop --no-skip-missing to allow."
        )


def _emit(
    file_name: str | None,
    words: list[tuple[str, float, float]],
    seq_tag_map: dict[str, str],
    out_jsonl,
) -> int | None:
    """Write one seq's word list as a JSONL line. Returns word count, or None if skipped."""
    if file_name is None or not words:
        return None
    seq_tag = seq_tag_map.get(file_name)
    # files mfa aligned but that aren't in our map are dropped
    if seq_tag is None:
        return None
    out_jsonl.write(json.dumps(
        {"seq_tag": seq_tag, "words": [[w, b, e] for (w, b, e) in words]},
        separators=(",", ":"),
        ensure_ascii=False,
    ))
    out_jsonl.write("\n")
    return len(words)


def _query_and_write_jsonl(
    aligner: PretrainedAligner,
    seq_tag_map: dict[str, str],
    out_path: Path,
) -> tuple[int, int, list[str]]:
    seen: set[str] = set()
    num_seqs = 0
    num_words = 0
    current_name: str | None = None
    current_words: list[tuple[str, float, float]] = []

    with aligner.session() as session:
        query = (
            sa.select(
                File.name,
                WordInterval.begin,
                WordInterval.end,
                Word.word,
            )
            .execution_options(yield_per=10_000)  # stream rows, the corpus db is too big to buffer
            .join(WordInterval.word)
            .join(WordInterval.utterance)
            .join(Utterance.file)
            .filter(WordInterval.duration > 0)  # mfa emits zero-width intervals, useless as targets
            .filter(Word.word_type != WordType.silence)  # drop <sil>/optional-silence pseudo-words
            # file_id ordering groups each file's rows contiguously, which the streaming loop relies on
            .order_by(Utterance.file_id, WordInterval.begin)
        )

        with gzip.open(out_path, "wt", encoding="utf-8") as out_jsonl:
            for file_name, begin, end, word in session.execute(query):
                if file_name != current_name:
                    n = _emit(current_name, current_words, seq_tag_map, out_jsonl)
                    if n is not None:
                        num_seqs += 1
                        num_words += n
                        seen.add(current_name)
                    current_name = file_name
                    current_words = []
                current_words.append((word, float(begin), float(end)))

            # flush the final file, the loop only emits on a name change
            n = _emit(current_name, current_words, seq_tag_map, out_jsonl)

            if n is not None:
                num_seqs += 1
                num_words += n
                seen.add(current_name)

    missing = [tag for name, tag in seq_tag_map.items() if name not in seen]
    return num_seqs, num_words, missing


if __name__ == "__main__":
    main()
