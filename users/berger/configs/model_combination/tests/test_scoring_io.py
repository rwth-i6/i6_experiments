"""Exercise the actual STM/CTM conversion, including earlier failure cases."""

from sisyphus import tk
from kaldialign import edit_distance

from i6_experiments.users.berger.model_pipelines.common.hf_scoring import NormalizeStmCtmJob


def test_stm_columns_labels_segments_and_empty_hypotheses(tmp_path):
    stm, ctm = tmp_path / "ref.stm", tmp_path / "hyp.ctm"
    stm.write_text(
        ';; LABEL "d0" "all" "All segments"\n'
        "recording.wav 1 speaker-name.wav 0.00 1.00 <d0> I DON'T HAVE TWENTY CATS\n"
        "recording.wav 1 speaker-name.wav 1.00 2.00 <d0> WE GO\n"
        "silent-hyp.wav 1 speaker 0.00 1.00 <d0> HELLO\n"
        "filler.wav 1 speaker 0.00 1.00 <d0> UM\n"
    )
    ctm.write_text(
        "recording.wav 1 0.00 0.10 I\nrecording.wav 1 0.10 0.10 DON'T\n"
        "recording.wav 1 0.20 0.10 HAVE\nrecording.wav 1 0.30 0.10 TWENTY\n"
        "recording.wav 1 0.40 0.10 CATS\nrecording.wav 1 1.00 0.20 WE\n"
        "filler.wav 1 0.10 0.10 UM\n"
    )
    job = NormalizeStmCtmJob(tk.Path(str(stm)), tk.Path(str(ctm)))
    job.out_stm, job.out_ctm = tk.Path(str(tmp_path / "normalized.stm")), tk.Path(str(tmp_path / "normalized.ctm"))
    job.run()
    lines = (tmp_path / "normalized.stm").read_text().splitlines()
    assert lines[0] == ';; LABEL "d0" "all" "All segments"'
    records = [row.split(maxsplit=6) for row in lines[1:]]
    main = [row for row in records if row[0] == "recording.wav"]
    assert main[0][:6] == ["recording.wav", "1", "speaker-name.wav", "0.00", "1.00", "<d0>"]
    assert [row[6] for row in main] == ["i do not have 20 cats", "we go"]
    assert next(row for row in records if row[0] == "silent-hyp.wav")[6] == "hello"
    rows = [line.split() for line in (tmp_path / "normalized.ctm").read_text().splitlines()]
    assert all(row[0] == "recording.wav" for row in rows)
    for start, end, reference, expected_errors in [(0, 1, main[0][6], 0), (1, 2, main[1][6], 1)]:
        words = [row[4] for row in rows if start <= float(row[2]) + float(row[3]) / 2 < end]
        assert edit_distance(reference.split(), words)["total"] == expected_errors
    assert [row[4] for row in rows if float(row[2]) >= 1] == ["we"]


def test_compound_normalization_does_not_join_different_segments(tmp_path):
    stm, ctm = tmp_path / "ref", tmp_path / "hyp"
    stm.write_text("r 1 s 0 1 <d0> ANY\nr 1 s 1 2 <d0> THING\n")
    ctm.write_text("r 1 0.1 0.1 ANYTHING\n")
    job = NormalizeStmCtmJob(tk.Path(str(stm)), tk.Path(str(ctm)))
    job.out_stm, job.out_ctm = tk.Path(str(tmp_path / "out.stm")), tk.Path(str(tmp_path / "out.ctm"))
    job.run()
    references = [line.split(maxsplit=6)[6] for line in (tmp_path / "out.stm").read_text().splitlines()]
    assert references == ["any", "thing"]
