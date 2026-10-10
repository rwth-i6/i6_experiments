from i6_experiments.users.sheremeta.returnn.profile import (
    compare_losses,
    format_profile,
    loss_parity,
    read_learning_rates,
)


def _step_lines(offset, overrides=None, grad_norm=1.0):
    lines = []
    for step in range(4):
        ce = (overrides or {}).get(step, 2.0 + offset + step / 100)
        line = f"ep 1 train, step {step}, ce {ce:.3f}, grad_norm:p2 {grad_norm:.3e}, num_seqs 12, 0.400 sec/step\n"
        lines.append(line)
    return lines


def test_compare_losses_per_step_and_epoch_means(tmp_path):
    learning_rates = tmp_path / "learning_rates"
    learning_rates.write_text(
        "{\n1: EpochData(learningRate=1.0, error={\n':meta:device': 'NVIDIA GH200 120GB',\n"
        "'dev_loss_ce': 3.0,\n'train_loss_ce': 2.015,\n'train_loss_grad_norm:p2': 1.0,\n}),\n}\n"
    )
    reference_scores = read_learning_rates(str(learning_rates))
    assert reference_scores == {
        1: {
            ":meta:device": "NVIDIA GH200 120GB",
            "dev_loss_ce": 3.0,
            "train_loss_ce": 2.015,
            "train_loss_grad_norm:p2": 1.0,
        }
    }
    scores = {1: {"train_loss_ce": 2.017, "train_loss_grad_norm:p2": 1.1}}
    reference = _step_lines(0.0)
    tolerances = {"step_tolerance": 5e-3, "mean_tolerance": 1e-3}

    close_lines = _step_lines(0.004, grad_norm=1.1)
    close = compare_losses(close_lines, reference, scores, reference_scores, num_steps=3, **tolerances)
    assert close["passed"] and close["compared_steps"] == 3 and close["first_step_beyond_tolerance"] is None

    far_lines = _step_lines(0.004, {2: 2.1})
    far = compare_losses(far_lines, reference, scores, reference_scores, num_steps=4, **tolerances)
    assert not far["passed"]
    assert far["first_step_beyond_tolerance"] == {"epoch": 1, "step": 2, "name": "ce", "value": 2.1, "reference": 2.02}

    means_only = compare_losses(
        far_lines, reference, scores, reference_scores, num_steps=4, step_tolerance=None, mean_tolerance=1e-3
    )
    assert means_only["passed"]


def test_loss_parity_takes_the_reference_steps_from_a_given_log(tmp_path):
    scores = "{\n1: EpochData(learningRate=1.0, error={\n'train_loss_ce': 2.015,\n}),\n}\n"
    training = tmp_path / "training"
    (training / "output").mkdir(parents=True)
    (training / "output" / "learning_rates").write_text(scores)
    (training / "log.run.1").write_text("".join(_step_lines(0.0)))
    running = tmp_path / "running"
    (running / "work").mkdir(parents=True)
    (running / "work" / "learning_rates").write_text(scores)
    (running / "log.run.1").write_text("ep 32 train, step 0, ce 1.000, num_seqs 12, 0.400 sec/step\n")
    first_slice = running / "first_slice.log"
    first_slice.write_text("".join(_step_lines(0.001)))
    kwargs = dict(
        learning_rates=str(training / "output" / "learning_rates"),
        reference_learning_rates=str(running / "work" / "learning_rates"),
        step_tolerance=5e-3,
        mean_tolerance=1e-3,
        num_steps=4,
    )
    assert loss_parity(reference_log=None, **kwargs)["compared_steps"] == 0
    given = loss_parity(reference_log=str(first_slice), **kwargs)
    assert given["passed"] and given["compared_steps"] == 4


def _summary(norm_ms, replay_ms):
    tables = {
        "categories": [{"name": "norm and reduce", "ms": norm_ms, "count": 80.0}],
        "ops": [],
        "modules": [],
        "aten": [],
        "kernels": [],
    }
    replay_step = {"step": 3, "wall_ms": 600.0, "kernels": 8, "gpu_busy_ms": 500.0, "gpu_idle_ms": 100.0}
    replay_step["phases"] = [{"name": "graph_replay", "wall_ms": 1.0, "kernels": 5, "kernel_ms": replay_ms}]
    capture_step = {"step": 2, "wall_ms": 900.0, "kernels": 9, "gpu_busy_ms": 50.0, "gpu_idle_ms": 850.0, "phases": []}
    return {
        "trace": "trace.json",
        "num_steps": 2,
        "table_steps": [3],
        "unattributed_kernels": 0,
        "replay": {
            "num_replays": 1,
            "warm_run_kernels": 4,
            "kernels_per_replay": 5.0,
            "matched_fraction": 0.8,
            "without_launch": 5,
            "shared_launch": 0,
        },
        "steps": [capture_step, replay_step],
        **tables,
        "warm_run": dict(tables, ops=[{"name": "normalization.py(112) __call__", "ms": norm_ms, "count": 80.0}]),
    }


def test_format_profile_reports_the_change_against_the_reference():
    report = format_profile(_summary(12.0, 450.0), _summary(20.0, 480.0)).splitlines()
    assert "    12.000    20.000    -8.000  norm and reduce" in report
    assert "   450.000   480.000   -30.000  graph_replay" in report
    assert "   100.000   100.000    +0.000  gpu idle" in report
    assert "    12.000    20.000    -8.000  normalization.py(112) __call__" in report
    assert "    12.000  norm and reduce" in format_profile(_summary(12.0, 450.0)).splitlines()
