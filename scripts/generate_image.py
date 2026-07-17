#!/usr/bin/env python3
"""
Generate or edit images via the OpenAI ChatGPT Image API using HTTP requests.

Requires: httpx (pip install httpx)

Usage:
    python3 generate_image.py --prompt "description" --filename "output.png" \
        [--input-image img1.png ...] [--resolution 1K|2K|4K] \
        [--aspect-ratio 1:1|16:9|9:16|4:3|3:4]
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

API_KEY_ENV = "CHATGPT_IMAGE_API_KEY"
BASE_URL_ENV = "CHATGPT_IMAGE_API_BASE"
MODEL = os.environ.get("CHATGPT_IMAGE_MODEL") or "gpt-image-2"

REQUEST_TIMEOUT_SECONDS = 600
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

SIZE_MAP = {
    "1K": {
        "1:1": "1024x1024",
        "16:9": "1280x720",
        "9:16": "720x1280",
        "4:3": "1024x768",
        "3:4": "768x1024",
    },
    "2K": {
        "1:1": "2048x2048",
        "16:9": "2048x1152",
        "9:16": "1152x2048",
        "4:3": "2048x1536",
        "3:4": "1536x2048",
    },
    "4K": {
        "1:1": "2880x2880",
        "16:9": "3840x2160",
        "9:16": "2160x3840",
        "4:3": "3264x2448",
        "3:4": "2448x3264",
    },
}


class RetryableRequestError(RuntimeError):
    """A successful HTTP response that did not contain a complete image."""


class ImageRequestError(RuntimeError):
    """A user-facing image request failure."""


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def format_bytes(byte_count: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(byte_count)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{byte_count} {unit}"
            return f"{size:.2f} {unit} ({byte_count:,} bytes)"
        size /= 1024


def can_retry(attempt: int) -> bool:
    return attempt < MAX_RETRIES


def retry_delay(attempt: int) -> int:
    return RETRY_DELAY_SECONDS * (2**attempt)


def wait_before_retry(reason: str, attempt: int) -> None:
    delay = retry_delay(attempt)
    retry_number = attempt + 1
    print(
        f"{reason}; retrying in {delay}s (retry {retry_number}/{MAX_RETRIES}) ...",
        flush=True,
    )
    time.sleep(delay)


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        fail(f"{name} is not set")
    return value


def build_url(base_url: str, endpoint_path: str) -> str:
    return f"{base_url.rstrip('/')}/v1{endpoint_path}"


def resolve_size(resolution: str, aspect_ratio: str) -> str:
    try:
        return SIZE_MAP[resolution][aspect_ratio]
    except KeyError:
        fail(f"Unsupported resolution/aspect-ratio combination: {resolution} {aspect_ratio}")


def validate_input_images(image_paths: list[str]) -> list[Path]:
    paths = []
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            fail(f"Image not found: {img_path}")
        if not path.is_file():
            fail(f"Input image is not a file: {img_path}")
        paths.append(path)
    return paths


def find_image_value(response_json: dict[str, Any]) -> str | None:
    data = response_json.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            for key in ("b64_json", "url", "image_url"):
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value

    item = response_json.get("item")
    if isinstance(item, dict):
        value = item.get("result")
        if isinstance(value, str) and value:
            return value

    response = response_json.get("response")
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            for output_item in output:
                if not isinstance(output_item, dict):
                    continue
                value = output_item.get("result")
                if isinstance(value, str) and value:
                    return value

    for key in ("b64_json", "url", "image_url"):
        value = response_json.get(key)
        if isinstance(value, str) and value:
            return value

    return None


def extract_image_value(response_json: dict[str, Any]) -> str:
    image_value = find_image_value(response_json)
    if image_value:
        return image_value
    raise ImageRequestError("No image payload in final response")


def decode_inline_image_bytes(image_value: str) -> bytes | None:
    if image_value.startswith("data:"):
        header, _, b64data = image_value.partition(",")
        if ";base64" not in header or not b64data:
            return None
        return base64.b64decode(b64data)

    if image_value.startswith("http://") or image_value.startswith("https://"):
        return None

    return base64.b64decode(image_value)


def write_image_value(image_value: str, output_path: Path) -> None:
    if image_value.startswith("data:"):
        image_bytes = decode_inline_image_bytes(image_value)
        if image_bytes is None:
            header, _, _ = image_value.partition(",")
            raise ImageRequestError(f"Unsupported data URL encoding: {header}")
        output_path.write_bytes(image_bytes)
        return

    if image_value.startswith("http://") or image_value.startswith("https://"):
        import httpx

        for attempt in range(MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=120, follow_redirects=True) as client:
                    response = client.get(image_value)
                    response.raise_for_status()
                    output_path.write_bytes(response.content)
                    return
            except httpx.TransportError as exc:
                if can_retry(attempt):
                    wait_before_retry(f"Download transport error ({type(exc).__name__})", attempt)
                    continue
                raise ImageRequestError(
                    f"Download failed after {MAX_RETRIES} retries: {type(exc).__name__}: {exc}"
                ) from exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if is_retryable_status(status_code) and can_retry(attempt):
                    wait_before_retry(f"Download returned HTTP {status_code}", attempt)
                    continue
                if is_retryable_status(status_code):
                    raise ImageRequestError(
                        f"Download failed after {MAX_RETRIES} retries: HTTP {status_code}"
                    ) from exc
                raise ImageRequestError(f"Download failed: HTTP {status_code}") from exc
            except OSError as exc:
                raise ImageRequestError(f"Could not save downloaded image: {exc}") from exc
        return

    try:
        image_bytes = decode_inline_image_bytes(image_value)
        if image_bytes is None:
            raise ImageRequestError("Unsupported image value")
        output_path.write_bytes(image_bytes)
    except ImageRequestError:
        raise
    except Exception as exc:
        raise ImageRequestError(f"Decode error: {exc}") from exc


def iter_sse_data(lines: Any) -> Any:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def event_type(payload: dict[str, Any]) -> str:
    value = payload.get("type")
    return value if isinstance(value, str) else ""


def is_partial_image_event(payload: dict[str, Any]) -> bool:
    return event_type(payload).endswith(".partial_image")


def read_streaming_image_response(response: Any) -> dict[str, Any]:
    for data_text in iter_sse_data(response.iter_lines()):
        if data_text == "[DONE]":
            break
        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or is_partial_image_event(payload):
            continue
        if find_image_value(payload):
            return payload

    raise RetryableRequestError("No final image payload in streaming response")


def create_generation(
    prompt: str,
    size: str,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    import httpx

    url = build_url(base_url, "/images/generations")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Submitting generation to {url} ...", flush=True)
    print(f"Model: {MODEL}, size: {size}", flush=True)
    print("Streaming enabled", flush=True)

    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    return read_streaming_image_response(response)
        except RetryableRequestError as exc:
            if can_retry(attempt):
                wait_before_retry(str(exc), attempt)
                continue
            raise ImageRequestError(
                f"Request failed after {MAX_RETRIES} retries: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            if can_retry(attempt):
                wait_before_retry(f"Request transport error ({type(exc).__name__})", attempt)
                continue
            raise ImageRequestError(
                f"Request failed after {MAX_RETRIES} retries: {type(exc).__name__}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if is_retryable_status(status_code) and can_retry(attempt):
                wait_before_retry(f"API returned HTTP {status_code}", attempt)
                continue
            if is_retryable_status(status_code):
                raise ImageRequestError(
                    f"Request failed after {MAX_RETRIES} retries: HTTP {status_code}"
                ) from exc
            raise ImageRequestError(f"API request failed: HTTP {status_code}") from exc
        except Exception as exc:
            raise ImageRequestError(f"Request error: {type(exc).__name__}: {exc}") from exc

    raise ImageRequestError("Request failed without a result")


def create_edit(
    prompt: str,
    image_paths: list[Path],
    size: str,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    import httpx

    url = build_url(base_url, "/images/edits")
    data = {
        "model": MODEL,
        "prompt": prompt,
        "size": size,
        "n": "1",
        "stream": "true",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Submitting edit to {url} ...", flush=True)
    print(f"Model: {MODEL}, size: {size}", flush=True)
    print("Streaming enabled", flush=True)
    for path in image_paths:
        print(f"Loaded input image: {path}", flush=True)

    for attempt in range(MAX_RETRIES + 1):
        open_files = []
        try:
            files = []
            for path in image_paths:
                mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
                file_obj = path.open("rb")
                open_files.append(file_obj)
                files.append(("image[]", (path.name, file_obj, mime_type)))

            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                with client.stream("POST", url, data=data, files=files, headers=headers) as response:
                    response.raise_for_status()
                    return read_streaming_image_response(response)
        except RetryableRequestError as exc:
            if can_retry(attempt):
                wait_before_retry(str(exc), attempt)
                continue
            raise ImageRequestError(
                f"Request failed after {MAX_RETRIES} retries: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            if can_retry(attempt):
                wait_before_retry(f"Request transport error ({type(exc).__name__})", attempt)
                continue
            raise ImageRequestError(
                f"Request failed after {MAX_RETRIES} retries: {type(exc).__name__}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if is_retryable_status(status_code) and can_retry(attempt):
                wait_before_retry(f"API returned HTTP {status_code}", attempt)
                continue
            if is_retryable_status(status_code):
                raise ImageRequestError(
                    f"Request failed after {MAX_RETRIES} retries: HTTP {status_code}"
                ) from exc
            raise ImageRequestError(f"API request failed: HTTP {status_code}") from exc
        except OSError as exc:
            raise ImageRequestError(f"Could not read input image: {exc}") from exc
        except Exception as exc:
            raise ImageRequestError(f"Request error: {type(exc).__name__}: {exc}") from exc
        finally:
            for file_obj in open_files:
                file_obj.close()

    raise ImageRequestError("Request failed without a result")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images via the OpenAI ChatGPT Image API")
    parser.add_argument("--prompt", "-p", required=True, help="Image description/prompt")
    parser.add_argument("--filename", "-f", required=True, help="Output filename")
    parser.add_argument("--input-image", "-i", nargs="+", help="Input image path(s) for editing")
    parser.add_argument(
        "--resolution",
        "-r",
        choices=["1K", "2K", "4K"],
        default="2K",
        help="Output resolution (default: 2K)",
    )
    parser.add_argument(
        "--aspect-ratio",
        "-a",
        choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
        default="16:9",
        help="Aspect ratio (default: 16:9)",
    )
    args = parser.parse_args()

    api_key = required_env(API_KEY_ENV)
    base_url = required_env(BASE_URL_ENV).rstrip("/")

    output_path = Path(args.filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = resolve_size(args.resolution, args.aspect_ratio)

    start_time = time.perf_counter()
    try:
        if args.input_image:
            image_paths = validate_input_images(args.input_image)
            response_json = create_edit(args.prompt, image_paths, size, api_key, base_url)
        else:
            response_json = create_generation(args.prompt, size, api_key, base_url)

        image_value = extract_image_value(response_json)
        write_image_value(image_value, output_path)
    except ImageRequestError as exc:
        fail(str(exc))

    elapsed_seconds = time.perf_counter() - start_time
    file_size = output_path.stat().st_size

    print(f"\nImage saved: {output_path.resolve()}")
    print(f"File size: {format_bytes(file_size)}")
    print(f"Generation time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
