"""Run the production streaming loop; fake only the encoder/search and output sink."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i6_experiments.users.berger.model_pipelines.common import recog


@pytest.mark.parametrize("length", [160, 180])
def test_context_is_not_resent_and_rasr_receives_2d_float32(length, monkeypatch):
    import returnn.frontend as rf

    sent, lifecycle, times = [], [], []

    def put_features(array):
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32 and array.ndim == 2
        assert array.shape[1] == 2
        sent.append(array.copy())

    search = SimpleNamespace(
        enter_segment=lambda: lifecycle.append("enter"),
        finish_segment=lambda: lifecycle.append("finish"),
        put_features=put_features,
        get_current_best_traceback=lambda: [],
        get_common_prefix=lambda: [],
    )
    formatter = SimpleNamespace(
        traceback_to_transcription=lambda trace: "",
        traceback_to_ctm_str=lambda trace, ms: times.append(ms) or "",
    )
    step = recog.StreamingRasrRecogForwardStep(
        encoder_frame_shift_seconds=0.02,
        chunk_history_seconds=0.04,
        chunk_center_seconds=0.06,
        chunk_future_seconds=0.02,
        sample_rate=1000,
        recog_rasr_config_file=None,
        traceback_formatter=formatter,
    )
    step._search_algorithm = search

    def encode(audio, sizes):
        frames = audio[:, ::20, 0].to(dtype=torch.float64)
        return torch.stack([frames, frames + 1], dim=-1).requires_grad_()

    raw = {
        "data": torch.arange(length).reshape(1, length, 1).float(),
        "data:size1": torch.tensor([length]),
        "raw": np.array([""]),
        "seq_tag": np.array(["stream-test"]),
    }
    output = {}
    monkeypatch.setattr(
        rf,
        "get_run_ctx",
        lambda: SimpleNamespace(expected_outputs=None, mark_as_output=lambda value, name: output.update({name: value})),
    )
    monkeypatch.setattr(recog, "Tensor", lambda **kwargs: SimpleNamespace(**kwargs))
    step(model=SimpleNamespace(forward=encode), extern_data=SimpleNamespace(as_raw_tensor_dict=lambda: raw))
    np.testing.assert_array_equal(np.concatenate(sent)[:, 0], np.arange(0, length, 20))
    assert lifecycle == ["enter", "finish"]
    assert times == [20.0]
    assert "ctm_string" in output
