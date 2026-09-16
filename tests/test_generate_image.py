import base64
import errno
import io
import importlib.util
import json
from pathlib import Path

import httpx
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "generate_image.py"
SPEC = importlib.util.spec_from_file_location("generate_image", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_image = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_image)


def sse_event(payload):
    return [f"data: {json.dumps(payload)}", ""]


def final_event(event_type="image_generation.completed"):
    return {
        "type": event_type,
        "b64_json": base64.b64encode(b"final-image").decode(),
    }


def partial_event(event_type="image_generation.partial_image"):
    return {
        "type": event_type,
        "b64_json": base64.b64encode(b"partial-image").decode(),
    }


class FakeResponse:
    def __init__(self, *, status_code=200, lines=(), content=b""):
        request = httpx.Request("POST", "https://example.test/v1/images")
        self._response = httpx.Response(
            status_code,
            request=request,
            content=content,
        )
        self._lines = list(lines)

    @property
    def status_code(self):
        return self._response.status_code

    @property
    def content(self):
        return self._response.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        return self._response.raise_for_status()

    def iter_lines(self):
        return iter(self._lines)


def install_fake_client(monkeypatch, outcomes, calls, opened_files=None):
    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def stream(self, method, url, **kwargs):
            calls.append({"method": method, "url": url, **kwargs})
            if opened_files is not None:
                for _, file_data in kwargs.get("files", []):
                    opened_files.append(file_data[1])
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(httpx, "Client", FakeClient)


def test_read_prompt_from_file(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("a detailed prompt\n", encoding="utf-8")

    assert generate_image.read_prompt(str(prompt_path)) == "a detailed prompt\n"


def test_read_prompt_from_stdin(monkeypatch):
    monkeypatch.setattr(generate_image.sys, "stdin", io.StringIO("prompt from stdin"))

    assert generate_image.read_prompt("-") == "prompt from stdin"


def test_read_prompt_keeps_literal_text_when_file_does_not_exist():
    assert generate_image.read_prompt("literal prompt") == "literal prompt"


@pytest.mark.parametrize("prompt", ["a dancing bear " * 1000, "小熊在跳舞" * 1000])
def test_read_prompt_keeps_long_literal_text(prompt):
    assert generate_image.read_prompt(prompt) == prompt


@pytest.mark.parametrize("prompt", ["first line\nsecond line", "first\rsecond", "bear\0dance"])
def test_read_prompt_skips_path_check_for_obvious_text(monkeypatch, prompt):
    def unexpected_path_check(self):
        pytest.fail("Text should not be checked as a path")

    monkeypatch.setattr(Path, "is_file", unexpected_path_check)

    assert generate_image.read_prompt(prompt) == prompt


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.ENAMETOOLONG, "File name too long"),
        PermissionError(errno.EACCES, "Permission denied"),
        OSError(errno.EIO, "Input/output error"),
        ValueError("Invalid path"),
    ],
)
def test_read_prompt_keeps_text_when_path_check_fails(monkeypatch, error):
    def failed_path_check(self):
        raise error

    monkeypatch.setattr(Path, "is_file", failed_path_check)

    assert generate_image.read_prompt("prompt.txt") == "prompt.txt"


def test_read_prompt_from_file_with_spaces_and_no_extension(tmp_path):
    prompt_path = tmp_path / "my prompt"
    prompt_path.write_text("小熊在跳舞\n", encoding="utf-8")

    assert generate_image.read_prompt(str(prompt_path)) == "小熊在跳舞\n"


def test_final_sse_response_can_be_saved(tmp_path):
    lines = sse_event(final_event()) + ["data: [DONE]", ""]
    payload = generate_image.read_streaming_image_response(FakeResponse(lines=lines))

    output_path = tmp_path / "result.png"
    generate_image.write_image_value(generate_image.extract_image_value(payload), output_path)

    assert output_path.read_bytes() == b"final-image"


@pytest.mark.parametrize(
    "lines",
    [
        [],
        sse_event(partial_event()) + ["data: [DONE]", ""],
        ["data: not-json", "", "data: [DONE]", ""],
    ],
)
def test_incomplete_stream_is_retryable(lines):
    with pytest.raises(generate_image.RetryableRequestError):
        generate_image.read_streaming_image_response(FakeResponse(lines=lines))


@pytest.mark.parametrize(
    "incomplete",
    [
        [],
        ["data: [DONE]", ""],
        sse_event({"type": "image_generation.completed", "b64_json": ""}),
        sse_event(partial_event()) + ["data: [DONE]", ""],
    ],
)
def test_generation_retries_incomplete_stream(monkeypatch, incomplete):
    outcomes = [
        FakeResponse(lines=incomplete),
        FakeResponse(lines=sse_event(final_event())),
    ]
    calls = []
    sleeps = []
    install_fake_client(monkeypatch, outcomes, calls)
    monkeypatch.setattr(generate_image.time, "sleep", sleeps.append)

    result = generate_image.create_generation(
        "prompt",
        "1024x1024",
        "test-key",
        "https://example.test",
    )

    assert generate_image.extract_image_value(result)
    assert len(calls) == 2
    assert sleeps == [20]
    assert all(call["json"]["stream"] is True for call in calls)
    assert all("partial_images" not in call["json"] for call in calls)


@pytest.mark.parametrize(
    "first_outcome",
    [
        FakeResponse(status_code=502),
        httpx.ReadTimeout(
            "read timed out",
            request=httpx.Request("POST", "https://example.test/v1/images"),
        ),
    ],
)
def test_generation_retries_transient_failures(monkeypatch, first_outcome):
    outcomes = [first_outcome, FakeResponse(lines=sse_event(final_event()))]
    calls = []
    sleeps = []
    install_fake_client(monkeypatch, outcomes, calls)
    monkeypatch.setattr(generate_image.time, "sleep", sleeps.append)

    result = generate_image.create_generation(
        "prompt",
        "1024x1024",
        "test-key",
        "https://example.test",
    )

    assert generate_image.extract_image_value(result)
    assert len(calls) == 2
    assert sleeps == [20]


def test_generation_does_not_retry_permanent_http_error(monkeypatch):
    outcomes = [FakeResponse(status_code=400)]
    calls = []
    sleeps = []
    install_fake_client(monkeypatch, outcomes, calls)
    monkeypatch.setattr(generate_image.time, "sleep", sleeps.append)

    with pytest.raises(generate_image.ImageRequestError, match="HTTP 400"):
        generate_image.create_generation(
            "prompt",
            "1024x1024",
            "test-key",
            "https://example.test",
        )

    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "incomplete", [[], sse_event(partial_event()) + ["data: [DONE]", ""]]
)
def test_generation_stops_after_five_retries(monkeypatch, incomplete):
    outcomes = [FakeResponse(lines=incomplete) for _ in range(6)]
    calls = []
    sleeps = []
    install_fake_client(monkeypatch, outcomes, calls)
    monkeypatch.setattr(generate_image.time, "sleep", sleeps.append)

    with pytest.raises(generate_image.ImageRequestError, match="after 5 retries"):
        generate_image.create_generation(
            "prompt",
            "1024x1024",
            "test-key",
            "https://example.test",
        )

    assert len(calls) == 6
    assert sleeps == [20, 40, 80, 160, 320]


@pytest.mark.parametrize(
    "incomplete",
    [
        [],
        ["data: [DONE]", ""],
        sse_event({"type": "image_edit.completed", "b64_json": ""}),
        sse_event(partial_event("image_edit.partial_image")),
    ],
)
def test_edit_reopens_and_closes_input_file_for_retry(monkeypatch, tmp_path, incomplete):
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"input-image")
    outcomes = [
        FakeResponse(lines=incomplete),
        FakeResponse(lines=sse_event(final_event("image_edit.completed"))),
    ]
    calls = []
    sleeps = []
    opened_files = []
    install_fake_client(monkeypatch, outcomes, calls, opened_files)
    monkeypatch.setattr(generate_image.time, "sleep", sleeps.append)

    result = generate_image.create_edit(
        "edit prompt",
        [input_path],
        "1024x1024",
        "test-key",
        "https://example.test",
    )

    assert generate_image.extract_image_value(result)
    assert len(opened_files) == 2
    assert opened_files[0] is not opened_files[1]
    assert all(file_obj.closed for file_obj in opened_files)
    assert sleeps == [20]
    assert all(call["data"]["stream"] == "true" for call in calls)
    assert all("partial_images" not in call["data"] for call in calls)
