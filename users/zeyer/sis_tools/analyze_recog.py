#!/usr/bin/env python3

"""
Given some (whatever) job (e.g. JoinScoreResultsJob)
or job output,
go through its inputs
to find the search jobs (e.g. "i6_core/returnn/forward/ReturnnForwardJobV2.06Moa9am7DMj")
and scoring jobs (e.g. "i6_core/recognition/scoring/ScliteJob.OcYJiwXkWAUI").
Analyze the results:
- Find
"""

from __future__ import annotations
import argparse
import gzip
import json
import pickle
import os
import sys
import re
from functools import reduce

# It will take the dir of the checked out git repo.
# So you can also only use it there...
_my_dir = os.path.dirname(os.path.realpath(__file__))
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = f"{_setup_base_dir}/tools/sisyphus"


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)

        os.environ["SIS_GLOBAL_SETTINGS_FILE"] = f"{_setup_base_dir}/settings.py"

        try:
            import sisyphus  # noqa
            import i6_experiments  # noqa
        except ImportError:
            print("setup base dir:", _setup_base_dir)
            print("sys.path:")
            for path in sys.path:
                print(f"  {path}")
            raise


_setup()


from sisyphus import tk
from . import sis_common
from i6_experiments.users.zeyer.utils.job_file_open import read_job_file


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("job", help="path to job, with or without 'work/' prefix")
    args = arg_parser.parse_args()
    job = sis_common.get_job_from_arg(args.job)

    visited = set()
    # (job, corpus, scoring)
    queue = [(job, None, None)]
    scoring_per_corpus = {}
    search_per_corpus = {}
    while queue:
        job, corpus, scoring = queue.pop(0)
        if job in visited:
            continue
        visited.add(job)
        print(f"{corpus or '?'}: {job}")
        if "/JoinScoreResultsJob." in job:
            _report_join_score_results_job(job)
        elif job.startswith("i6_experiments/users/zeyer/recog/GetBestRecogTrainExp."):
            _report_get_best_recog_train_exp_job(job)
            for job_ in _get_best_recog_train_exp_job_deps(job):
                queue.append((job_, corpus, scoring))
            continue  # do not follow all the other deps
        elif job.startswith("i6_core/recognition/scoring/ScliteJob."):
            print(
                f"* {corpus or '?'}: Found scoring. Aliases:", " ".join(sis_common.get_job_aliases(job) or ["<None?>"])
            )
            scoring = job
            scoring_per_corpus[corpus] = job
        elif job.startswith("i6_core/returnn/forward/ReturnnForwardJobV2."):
            print(
                f"* {corpus or '?'}: Found search. Aliases:", " ".join(sis_common.get_job_aliases(job) or ["<None?>"])
            )
            search_per_corpus[corpus] = job
            continue  # deps of this don't matter
        if not scoring:
            for corpus_, job_ in sis_common.get_inputs_per_key(job).items():
                queue.append((job_, corpus if scoring else corpus_, scoring))
        for job_ in sis_common.get_inputs(job):
            queue.append((job_, corpus, scoring))

    if not scoring_per_corpus and not search_per_corpus:
        print("Nothing found. Exit.")
        sys.exit(1)

    for corpus, sclite_job in scoring_per_corpus.items():
        sclite_pra_report_worst_seqs(corpus or "?", sclite_job)

    for corpus, search_job in search_per_corpus.items():
        search_log_report_timings_summary(corpus or "?", search_job)


def _report_join_score_results_job(job: str):
    fn = sis_common.get_work_dir_prefix() + job + "/output/score_results.json"
    if os.path.exists(fn):
        with open(fn) as f:
            print("joined score results:", f.read().strip())


def _report_get_best_recog_train_exp_job(job: str):
    fn = sis_common.get_work_dir_prefix() + job + "/output/summary.json"
    with open(fn) as f:
        print("best recog train exp:", f.read().strip())


def _get_best_recog_train_exp_job_deps(job: str) -> list[str]:
    fn = sis_common.get_work_dir_prefix() + job + "/output/summary.json"
    d = json.load(open(fn))
    assert isinstance(d, dict)
    best_ep = d["best_epoch"]
    assert isinstance(best_ep, int)

    d = read_job_file(job, "job.save")
    d = gzip.decompress(d)
    job_obj = pickle.loads(d)
    # noinspection PyProtectedMember
    p = job_obj._scores_outputs[best_ep].output
    assert isinstance(p, tk.Path)
    dep = sis_common.get_job_from_work_output(p.get_path())
    return [dep]


def sclite_pra_report_worst_seqs(
    prefix: str,
    sclite_job: str,
    *,
    n: int = 2,
    broken_seq_threshold_rel: float = 0.5,
    broken_seq_threshold_abs: int = 10,
):
    fn = sis_common.get_work_dir_prefix() + sclite_job + "/output/reports/sclite.pra"
    print(f"({prefix}: read {fn})")
    with open(fn) as f:
        lines = f.read().splitlines()
    total_ref_num_token = 0
    total_num_err = 0
    total_num_sub = 0
    total_num_del = 0
    total_num_ins = 0
    total_num_seqs = 0
    total_broken_num_seqs = 0
    total_broken_seq_len = 0
    worst_seqs = []  # (num_err, cur_seq)
    cur_seq = []
    for line in lines:
        if not line.strip():
            cur_seq = []
        else:
            cur_seq.append(line)
        if line.startswith("Scores: (#C #S #D #I) "):
            line = line[len("Scores: (#C #S #D #I) ") :]
            num_correct, num_err_sub, num_err_del, num_err_ins = map(int, line.split())
            ref_num_token = num_correct + num_err_sub + num_err_del
            assert ref_num_token > 0, f"unexpected empty: line: {line}\n" + "\n".join(cur_seq)
            total_ref_num_token += ref_num_token
            num_err = num_err_sub + num_err_del + num_err_ins
            total_num_err += num_err
            total_num_sub += num_err_sub
            total_num_del += num_err_del
            total_num_ins += num_err_ins
            total_num_seqs += 1
            if len(worst_seqs) < n or worst_seqs[-1][0] < num_err:
                worst_seqs.append((num_err, cur_seq))
                worst_seqs.sort(key=lambda item: -item[0])
                worst_seqs = worst_seqs[:n]
            if num_err / ref_num_token >= broken_seq_threshold_rel and num_err >= broken_seq_threshold_abs:
                total_broken_num_seqs += 1
                total_broken_seq_len += ref_num_token
    print(
        f"{prefix}: WER: {total_num_err / total_ref_num_token * 100.:.2f}%"
        f" (sub: {total_num_sub / total_ref_num_token * 100.:.2f}%,"
        f" del {total_num_del / total_ref_num_token * 100.:.2f}%,"
        f" ins {total_num_ins / total_ref_num_token * 100.:.2f}%)"
    )
    print(
        f"{prefix}: num broken (>={broken_seq_threshold_rel*100.:.0f}% WER, >={broken_seq_threshold_abs} errs) seqs:"
        f" {total_broken_num_seqs}/{total_num_seqs} = {total_broken_num_seqs/total_num_seqs*100.:.1f}%,"
        f" avg seq len: {total_broken_seq_len / total_broken_num_seqs if total_broken_num_seqs else 0:.1f} words"
    )
    print(f"{prefix}: worst {len(worst_seqs)} seqs:")
    for num_err, cur_seq in worst_seqs:
        print(f"--- ({num_err} errors):")
        # TODO would be good if we can also find the individual seq scores in search ext output for this seq...
        print("\n".join(cur_seq))
    print("---")


def search_log_report_timings_summary(prefix: str, search_job: str):
    with sis_common.open_job_log(search_job) as (f, log_fn):
        print(f"({prefix}: opened log {log_fn})")
        lines = f.read().splitlines()
    data_len_frame_factor = None
    data_len_num_frames = 0
    enc_num_frames = 0
    dec_num_frames = 0
    time_enc_ns = 0
    time_dec_ns = 0
    num_steps = 0
    dev_ls_s = []
    for line in lines:
        if line.startswith("Hostname: ") or line.startswith("Using gpu device ") or line.startswith("Total GPU "):
            dev_ls_s.append(line)
            continue
        if not line.startswith("TIMINGS: "):
            continue
        # example:
        # TIMINGS: batch size 2, data len max 369120 (23.07 secs), data len sum 736960 (46.06 secs),
        #   enc 89626378 ns, enc len max 385, dec 2212957542 ns, out len max 77
        line = line[len("TIMINGS: ") :]
        parts = line.split(", ")
        d = {
            # key -> abs num, opt time in secs, opt unit
            m.group(1): (int(m.group(2)), float(m.group(4)) if m.group(4) else None, m.group(5))
            for part in parts
            for m in [re.match(r"^([a-z ]+) ([0-9.]+) ?(\(([0-9.]+) secs\))? ?([a-z]+)?$", part)]
        }
        if data_len_frame_factor is None:
            print(f"({prefix}: first timing log entry: {line})")
            num_frames, time_in_secs, _ = d["data len sum"]
            data_len_frame_factor = int(round(num_frames / time_in_secs, -2))
            print(
                f"{prefix}: timings first batch data len sum:"
                f" {num_frames} frames, {time_in_secs} secs -> factor {data_len_frame_factor}"
            )
        data_len_num_frames += d["data len sum"][0]
        time_enc_ns += d["enc"][0]
        time_dec_ns += d["dec"][0]
        enc_num_frames += d["enc len max"][0] * d["batch size"][0]
        dec_num_frames += d["out len longest sum"][0]
        num_steps += 1
    if data_len_frame_factor is None:
        print(f"{prefix} timings ERROR, none found")
        return
    data_len_secs = data_len_num_frames / data_len_frame_factor
    time_enc_secs = time_enc_ns * 1e-9
    time_dec_secs = time_dec_ns * 1e-9
    print(f"{prefix}: timings total:")
    print(f"  num steps: {num_steps}")
    print(f"  data len: {data_len_secs:.2f} secs")
    print(f"  encoder num frames: {enc_num_frames}")
    print(f"  decoder num frames: {dec_num_frames} (tokens)")
    print(f"  Host/GPU: {', '.join(dev_ls_s) or '<unknown>'}")
    print(f"  encoder time: {time_enc_secs:.2f} secs")
    print(f"  decoder time: {time_dec_secs:.2f} secs")
    print(f"  RTF: {(time_enc_secs + time_dec_secs) / data_len_secs:.4f}")


if __name__ == "__main__":
    try:
        import better_exchook

        better_exchook.install()
    except ImportError:
        better_exchook = None
    main()
