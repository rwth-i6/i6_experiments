import json
import os
import pathlib
from typing import Dict, List

import numpy
import pytest
import returnn
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from returnn.datasets.hdf import HDFDataset, SimpleHDFWriter
from sisyphus import tk

from i6_experiments.users.sheremeta.alignment.jobs import TokenFrameOffsetJob, WordPackingJob
from i6_experiments.users.sheremeta.returnn.dataset import ReturnnConfigDatasetToHdfJob
from i6_experiments.users.sheremeta.tests.test_ctc_word_boundaries import _train_spm

_RETURNN_ROOT = os.path.dirname(os.path.dirname(returnn.__file__))
_FRAME_RATE = 100.0 / 6.0


def _write_paths(hdf: pathlib.Path, paths: Dict[str, List[int]], dim: int) -> tk.Path:
    """
    :param hdf: the file
    :param paths: per sequence tag the label of every frame
    :param dim: the label dim of the file
    :return: the file as a sisyphus path
    """
    writer = SimpleHDFWriter(str(hdf), dim=dim, ndim=1)
    for tag, path in paths.items():
        data = numpy.asarray(path, dtype="int32")
        writer.insert_batch(data[None, :], seq_len={0: [len(path)]}, seq_tag=[tag])
    writer.close()
    return tk.Path(str(hdf))


def test_token_offsets_pair_the_tokens_of_a_path_with_those_of_its_targets(tmp_path: pathlib.Path):
    blank = 9
    # the first two tokens a frame late, a repeat on adjacent frames in place
    targets = _write_paths(tmp_path / "targets.hdf", {"seq-0": [9, 1, 3, 9, 2, 2, 9], "seq-1": [1, 9, 9]}, blank + 1)
    paths = _write_paths(
        tmp_path / "paths.hdf", {"seq-0": [9, 9, 1, 3, 2, 2, 9], "seq-1": [9, 9, 2], "seq-2": [1, 9, 9]}, blank + 1
    )

    job = TokenFrameOffsetJob(
        path_hdf=paths, target_hdf=targets, blank=blank, frame_rate=_FRAME_RATE, returnn_root=tk.Path(_RETURNN_ROOT)
    )
    job.out_summary = tk.Path(str(tmp_path / "summary.json"))
    job.out_report = tk.Path(str(tmp_path / "report.txt"))
    job.run()

    summary = json.loads((tmp_path / "summary.json").read_text())
    # seq-1 places another token than its target, seq-2 has no target
    assert summary["n_utts"] == 1 and summary["n_tokens"] == 4
    assert summary["failures"] == {"label_mismatch": 1, "missing_target": 1}
    assert summary["offset_frames"]["mean"] == 0.5
    assert summary["share_exact"] == 0.5
    assert summary["histogram_frames"] == {"0": 2, "1": 2}


def test_packing_counts_the_pad_frames_inside_and_between_words(tmp_path: pathlib.Path):
    spm_file, sp = _train_spm(tmp_path)
    blank = sp.GetPieceSize()
    first, second = sp.EncodeAsIds("KEEPERS"), sp.EncodeAsIds("FOXY")
    assert len(first) > 1 and len(second) > 1, (first, second)
    # the first word packed, one pad after the first token of the second, two pad frames between the words
    path = [blank, *first, blank, blank, second[0], blank, *second[1:]]
    paths = _write_paths(tmp_path / "paths.hdf", {"seq-0": path}, blank + 1)

    job = WordPackingJob(
        path_hdf=paths, spm_file=tk.Path(str(spm_file)), blank=blank, returnn_root=tk.Path(_RETURNN_ROOT)
    )
    job.out_summary = tk.Path(str(tmp_path / "summary.json"))
    job.out_report = tk.Path(str(tmp_path / "report.txt"))
    job.run()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_words"] == 2 and summary["n_multi_token_words"] == 2
    assert summary["share_packed"] == 0.5
    assert summary["inner_gap_histogram"] == {"0": len(first) - 1 + len(second) - 2, "1": 1}
    assert summary["between_gap_histogram"] == {"2": 1}


def test_a_key_of_the_forward_dataset_is_written_per_sequence(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    # a dataset defined in the config's prolog, the way a recipe attaches its postprocessed splits
    prolog = (
        "import numpy\n"
        "_seqs = [{'data': numpy.zeros((3, 1), 'float32'), 'codes': numpy.array([4, 1, 4], 'int32')},\n"
        "         {'data': numpy.zeros((2, 1), 'float32'), 'codes': numpy.array([2, 4], 'int32')}]\n"
    )
    dataset = {"class": "StaticDataset", "data": CodeWrapper("_seqs"), "output_dim": {"data": [1, 2], "codes": [5, 1]}}
    config = ReturnnConfig(config={"forward_data": dataset}, python_prolog=prolog)

    job = ReturnnConfigDatasetToHdfJob(returnn_config=config, data_key="codes", returnn_root=tk.Path(_RETURNN_ROOT))
    job.out_hdf = tk.Path(str(tmp_path / "codes.hdf"))
    monkeypatch.chdir(tmp_path)
    job.run()

    written = HDFDataset([str(tmp_path / "codes.hdf")])
    written.initialize()
    written.init_seq_order(epoch=1)
    written.load_seqs(0, 2)
    assert [written.get_data(i, "data").tolist() for i in range(2)] == [[4, 1, 4], [2, 4]]
