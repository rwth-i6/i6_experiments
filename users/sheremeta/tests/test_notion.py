import json
import math
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest
from sisyphus import gs, tk

from i6_experiments.users.sheremeta.notion import api as notion_api
from i6_experiments.users.sheremeta.notion.api import NotionApi, NotionApiError
from i6_experiments.users.sheremeta.notion.jobs import PushNotionRowJob
from i6_experiments.users.sheremeta.notion.properties import property_value

_KEY = "ReturnnTrainingJob.x5bGXNr5MQgK"

# the answers of a fake Notion in order, each a status, a JSON body and the response headers
Answers = List[Tuple[int, Dict[str, Any], Dict[str, str]]]


class _FakeNotion:
    """Gives its answers in order and records every request as method, path and JSON body."""

    def __init__(self, answers: Answers):
        self.answers = list(answers)
        self.sent: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []

    def __call__(
        self, method: str, url: str, headers: Dict[str, str], body: Optional[bytes]
    ) -> Tuple[int, Mapping[str, str], bytes]:
        assert headers["Authorization"] == "Bearer secret"
        assert headers["Notion-Version"] == notion_api.NOTION_VERSION
        self.sent.append((method, url[len(notion_api.API_URL) :], None if body is None else json.loads(body)))
        status, answer, answer_headers = self.answers.pop(0)
        return status, answer_headers, json.dumps(answer).encode("utf-8")


@pytest.mark.parametrize(
    "kind, value, expected",
    [
        ("title", "dsm_row", {"title": [{"type": "text", "text": {"content": "dsm_row"}}]}),
        ("rich_text", None, {"rich_text": []}),
        ("number", 5.08, {"number": 5.08}),
        ("number", None, {"number": None}),
        ("select", "librispeech", {"select": {"name": "librispeech"}}),
        ("multi_select", ["a", "b"], {"multi_select": [{"name": "a"}, {"name": "b"}]}),
        ("checkbox", False, {"checkbox": False}),
        ("url", "https://example.org", {"url": "https://example.org"}),
        ("date", "2026-10-22", {"date": {"start": "2026-10-22"}}),
    ],
)
def test_a_value_takes_the_shape_of_its_notion_type(kind: str, value: Any, expected: Dict[str, Any]):
    assert property_value(kind, value) == expected


def test_a_long_text_is_split_into_items_of_2000_characters():
    items = property_value("rich_text", "x" * 4500)["rich_text"]
    assert [len(item["text"]["content"]) for item in items] == [2000, 2000, 500]


@pytest.mark.parametrize("kind, value", [("number", math.nan), ("select", "a,b"), ("checkbox", "yes")])
def test_a_value_notion_cannot_store_is_refused(kind: str, value: Any):
    with pytest.raises(ValueError):
        property_value(kind, value)


def test_a_rate_limited_request_waits_the_given_seconds_and_retries():
    fake = _FakeNotion([(429, {"code": "rate_limited"}, {"Retry-After": "3"}), (200, {"results": []}, {})])
    waits: List[float] = []
    api = NotionApi("secret", send=fake, sleep=waits.append)
    assert api.query_equals("ds", "Training", "rich_text", _KEY) == []
    assert waits == [3.0]
    assert len(fake.sent) == 2


def test_a_refused_request_raises_with_the_notion_message():
    fake = _FakeNotion([(400, {"code": "validation_error", "message": "Beam WER is not a property"}, {})])
    api = NotionApi("secret", send=fake, sleep=lambda seconds: None)
    with pytest.raises(NotionApiError, match="validation_error: Beam WER is not a property"):
        api.create_page("ds", {})


@pytest.fixture
def token_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    # a home of its own, so no test can read a real token
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "token"
    path.write_text("secret\n")
    path.chmod(0o600)
    monkeypatch.setattr(gs, "NOTION_TOKEN_FILE", str(path), raising=False)
    return path


def _job(tmp_path: pathlib.Path) -> PushNotionRowJob:
    """A row of two plain columns and two resolved at run time, its outputs in tmp_path."""
    metrics = tk.Variable(str(tmp_path / "metrics"))
    metrics.set({"wer": 5.08, "params_M": 461.86})
    job = PushNotionRowJob(
        data_source_id="ds",
        key="Training",
        columns={
            "Row": ("title", "dsm_row"),
            "Training": ("rich_text", _KEY),
            "Greedy WER": ("number", metrics["wer"]),
            "Params M": ("number", metrics["params_M"]),
        },
    )
    job.out_page = tk.Variable(str(tmp_path / "page"))
    job.out_properties = tk.Path(str(tmp_path / "properties.json"))
    return job


def test_a_new_key_creates_a_row_with_the_resolved_values(
    tmp_path: pathlib.Path, token_file: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    fake = _FakeNotion([(200, {"results": []}, {}), (200, {"id": "p1", "url": "https://notion.so/p1"}, {})])
    monkeypatch.setattr(notion_api, "urllib_send", fake)
    job = _job(tmp_path)
    job.run()

    query = {"filter": {"property": "Training", "rich_text": {"equals": _KEY}}, "page_size": 2}
    assert fake.sent[0] == ("POST", "/data_sources/ds/query", query)
    method, path, body = fake.sent[1]
    assert (method, path) == ("POST", "/pages")
    assert body is not None
    assert body["parent"] == {"type": "data_source_id", "data_source_id": "ds"}
    assert body["properties"]["Greedy WER"] == {"number": 5.08}
    assert json.loads((tmp_path / "properties.json").read_text()) == body["properties"]
    assert job.out_page.get() == {"id": "p1", "url": "https://notion.so/p1", "created": True}


def test_an_existing_key_updates_that_row(
    tmp_path: pathlib.Path, token_file: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    fake = _FakeNotion([(200, {"results": [{"id": "p1"}]}, {}), (200, {"id": "p1", "url": "https://notion.so/p1"}, {})])
    monkeypatch.setattr(notion_api, "urllib_send", fake)
    job = _job(tmp_path)
    job.run()
    assert fake.sent[1][:2] == ("PATCH", "/pages/p1")
    assert job.out_page.get() == {"id": "p1", "url": "https://notion.so/p1", "created": False}


def test_two_rows_with_the_same_key_fail_the_push(
    tmp_path: pathlib.Path, token_file: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    fake = _FakeNotion([(200, {"results": [{"id": "p1"}, {"id": "p2"}]}, {})])
    monkeypatch.setattr(notion_api, "urllib_send", fake)
    with pytest.raises(ValueError, match="several rows"):
        _job(tmp_path).run()


def test_a_token_file_others_can_read_is_refused(
    tmp_path: pathlib.Path, token_file: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    token_file.chmod(0o644)
    monkeypatch.setattr(notion_api, "urllib_send", _FakeNotion([]))
    with pytest.raises(PermissionError, match="chmod 600"):
        _job(tmp_path).run()
