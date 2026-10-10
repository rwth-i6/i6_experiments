import os
import tarfile

from i6_experiments.users.sheremeta.returnn.throughput import _rank_zero_lines, step_losses, summarize_log


_LINES = [
    "[default0]:ep 1 train, step 10, ce 1.5, num_seqs 12, max_size:time:var-unk:data 300000,"
    " mem_usage:cuda:0 52.8GB, 0.717 sec/step, elapsed 0:06:53, exp. remaining 0:55:35, complete 11.02%\n",
    "[default1]:ep 1 train, step 10, ce 1.5, num_seqs 30, max_size:time:var-unk:data 300000,"
    " mem_usage:cuda:0 99.0GB, 9.000 sec/step, elapsed 0:06:53, exp. remaining 0:55:35, complete 11.02%\n",
    "\x1b[32m[default0]:ep 1 train, step 11, ce 1.4, num_seqs 14, mem_usage:cuda:0 53.1GB, 0.703 sec/step,"
    " elapsed 0:06:54, exp. remaining 0:55:37, complete 11.05%\x1b[0m\n",
    "[default0]:ep 1 train, step 12, ce 1.4, num_seqs 30, mem_usage:cuda:0 44.1GB, mem_graph_pool:cuda:0 33.3GB,"
    " 0.396 sec/step, elapsed 0:06:55, exp. remaining 0:55:38, complete 11.07%\n",
    "[default0]:Epoch 1: Trained 4707 steps, 0:53:20 elapsed"
    " (99.3% computing time, data: 24.4%, labels: 67.9% bound slack)\n",
]


def _check(lines):
    summary = summarize_log(lines)
    assert summary["epochs"] == [
        {
            "epoch": 1,
            "steps": 4707,
            "seconds": 3200,
            "computing_time_percent": 99.3,
            "slack_percent": {"data": 24.4, "labels": 67.9},
            "slack_kind": "bound slack",
        }
    ]
    assert summary["logged_steps"] == 3
    assert summary["median_sec_per_step"] == 0.703
    assert summary["max_mem_gb"] == 53.1
    assert summary["max_graph_pool_gb"] == 33.3
    assert summary["max_num_seqs"] == 30
    assert step_losses(lines) == {(1, 10): {"ce": 1.5}, (1, 11): {"ce": 1.4}, (1, 12): {"ce": 1.4}}


def test_summarize_log_reads_epoch_and_step_lines(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    (live / "log.run.1").write_text("".join(_LINES))
    (live / "log.run.1.error.01").write_text("".join(_LINES))
    _check(_rank_zero_lines(str(live)))

    archived = tmp_path / "archived"
    archived.mkdir()
    with tarfile.open(archived / "finished.tar.gz", "w:gz") as tar:
        tar.add(live / "log.run.1", arcname="engine/job.run.1770276.1")
        link = tarfile.TarInfo("log.run.1")
        link.type = tarfile.SYMTYPE
        link.linkname = "engine/job.run.1770276.1"
        tar.addfile(link)
    _check(_rank_zero_lines(str(archived)))


def _slice_log(epochs, sec_per_step, trained):
    lines = []
    for epoch, steps in epochs:
        for step in range(steps):
            lines.append(
                f"[default0]:ep {epoch} train, step {step}, ce {epoch + step / 10:.1f}, num_seqs 10,"
                f" mem_usage:cuda:0 50.0GB, {sec_per_step:.3f} sec/step, elapsed 0:00:01\n"
            )
        if epoch in trained:
            lines.append(f"[default0]:Epoch {epoch}: Trained {steps} steps, 0:00:10 elapsed (99.0% computing time)\n")
    return "".join(lines)


def test_rank_zero_lines_joins_the_slices_of_a_resumed_training(tmp_path):
    slices = [_slice_log([(1, 2), (2, 1)], 9.0, trained={1}), _slice_log([(2, 2)], 0.5, trained={2})]
    live = tmp_path / "live"
    (live / "engine").mkdir(parents=True)
    archived = tmp_path / "archived"
    archived.mkdir()
    with tarfile.open(archived / "finished.tar.gz", "w:gz") as tar:
        for index, text in enumerate(slices):
            path = live / "engine" / f"job.run.{100 + index}.1"
            path.write_text(text)
            os.utime(path, (1000 + index, 1000 + index))
            tar.add(path, arcname=f"engine/job.run.{100 + index}.1")
        link = tarfile.TarInfo("log.run.1")
        link.type = tarfile.SYMTYPE
        link.linkname = "engine/job.run.101.1"
        tar.addfile(link)
    (live / "engine" / "job.run.101.1.batch").write_text("")
    (live / "log.run.1").write_text(slices[-1])
    for job_dir in (live, archived):
        lines = _rank_zero_lines(str(job_dir))
        summary = summarize_log(lines)
        assert [epoch["epoch"] for epoch in summary["epochs"]] == [1, 2], job_dir
        assert summary["logged_steps"] == 4, job_dir
        assert sorted(step_losses(lines)) == [(1, 0), (1, 1), (2, 0), (2, 1)], job_dir
