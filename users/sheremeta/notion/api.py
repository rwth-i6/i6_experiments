"""
The calls of the Notion REST API a database row needs, on the standard library only.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

API_URL = "https://api.notion.com/v1"
#: the API version the requests are written for, see https://developers.notion.com/reference/versioning
NOTION_VERSION = "2026-03-11"

# rate limiting, service overload and server errors, a later try may succeed
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})

#: sends one request, method, URL, headers and body to status, response headers and response body
Send = Callable[[str, str, Dict[str, str], Optional[bytes]], Tuple[int, Mapping[str, str], bytes]]


class NotionApiError(RuntimeError):
    """A request the Notion API refused or did not answer."""


def urllib_send(
    method: str, url: str, headers: Dict[str, str], body: Optional[bytes]
) -> Tuple[int, Mapping[str, str], bytes]:
    """
    Sends one request with urllib and returns an error status like any other.

    :param method: the HTTP method
    :param url: the full URL
    :param headers: the request headers
    :param body: the request body, None for none
    :return: the status, the response headers and the response body
    """
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()


class NotionApi:
    """Queries a data source and creates or updates its pages, waiting out rate limits."""

    def __init__(
        self,
        token: str,
        *,
        send: Optional[Send] = None,
        sleep: Optional[Callable[[float], None]] = None,
        attempts: int = 5,
    ):
        """
        :param token: the token of the integration the database is shared with
        :param send: sends one request, :func:`urllib_send` when None
        :param sleep: waits the given seconds, :func:`time.sleep` when None
        :param attempts: tries of a request that failed in a way a later try may fix
        """
        assert attempts >= 1, attempts
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
        self._send = send if send is not None else urllib_send
        self._sleep = sleep if sleep is not None else time.sleep
        self._attempts = attempts

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        :param method: the HTTP method
        :param path: the path after the version prefix, for example ``/pages``
        :param body: the JSON body, None for none
        :return: the JSON answer
        """
        data = None if body is None else json.dumps(body).encode("utf-8")
        for attempt in range(1, self._attempts + 1):
            last = attempt == self._attempts
            try:
                status, headers, payload = self._send(method, API_URL + path, self._headers, data)
            except urllib.error.URLError as error:
                if last:
                    raise NotionApiError(f"{method} {path} got no answer, {error.reason}") from error
                self._sleep(2.0 ** (attempt - 1))
                continue
            if 200 <= status < 300:
                return json.loads(payload)
            if status not in _RETRY_STATUSES or last:
                raise NotionApiError(f"{method} {path} failed with {status}, {_error_message(payload)}")
            self._sleep(_retry_after(headers, attempt))
        raise AssertionError("the loop returns or raises")

    def query_equals(self, data_source_id: str, column: str, kind: str, value: str) -> List[Dict[str, Any]]:
        """
        :param data_source_id: the data source
        :param column: a title or rich_text column
        :param kind: the kind of the column, title or rich_text
        :param value: the text the column has to hold exactly
        :return: the matching pages, at most two, enough to tell one match from several
        """
        body = {"filter": {"property": column, kind: {"equals": value}}, "page_size": 2}
        return self.request("POST", f"/data_sources/{data_source_id}/query", body)["results"]

    def create_page(self, data_source_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param data_source_id: the data source the page becomes a row of
        :param properties: the column values, see ``properties.property_value``
        :return: the created page
        """
        parent = {"type": "data_source_id", "data_source_id": data_source_id}
        return self.request("POST", "/pages", {"parent": parent, "properties": properties})

    def update_page(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param page_id: the page
        :param properties: the column values to write, other columns keep theirs
        :return: the updated page
        """
        return self.request("PATCH", f"/pages/{page_id}", {"properties": properties})


def _retry_after(headers: Mapping[str, str], attempt: int) -> float:
    # header names are case-insensitive
    value = next((value for name, value in headers.items() if name.lower() == "retry-after"), None)
    if value is not None and value.strip().isdigit():
        return float(value)
    return 2.0 ** (attempt - 1)


def _error_message(payload: bytes) -> str:
    try:
        answer = json.loads(payload)
    except ValueError:
        return payload.decode("utf-8", errors="replace")[:500]
    return f"{answer.get('code')}: {answer.get('message')}"
