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


def extract_image_value(response_json: dict[str, Any]) -> str:
    data = response_json.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            for key in ("b64_json", "url", "image_url"):
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value

    for key in ("b64_json", "url", "image_url"):
        value = response_json.get(key)
        if isinstance(value, str) and value:
            return value

    fail(f"No image payload in response: {response_json}")


def write_image_value(image_value: str, output_path: Path) -> None:
    if image_value.startswith("data:"):
        header, _, b64data = image_value.partition(",")
        if ";base64" not in header or not b64data:
            fail(f"Unsupported data URL encoding: {header}")
        output_path.write_bytes(base64.b64decode(b64data))
        return

    if image_value.startswith("http://") or image_value.startswith("https://"):
        import httpx

        try:
            with httpx.Client(timeout=120, follow_redirects=True) as client:
                response = client.get(image_value)
                response.raise_for_status()
                output_path.write_bytes(response.content)
        except httpx.HTTPStatusError as exc:
            fail(f"Download error {exc.response.status_code}: {exc.response.text}")
        except Exception as exc:
            fail(f"Download error: {exc}")
        return

    try:
        output_path.write_bytes(base64.b64decode(image_value))
    except Exception as exc:
        fail(f"Decode error: {exc}")


def create_generation(prompt: str, size: str, api_key: str, base_url: str) -> dict[str, Any]:
    import httpx

    url = build_url(base_url, "/images/generations")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Submitting generation to {url} ...")
    print(f"Model: {MODEL}, size: {size}")

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        fail(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        fail(f"Request error: {exc}")


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
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Submitting edit to {url} ...")
    print(f"Model: {MODEL}, size: {size}")

    open_files = []
    try:
        files = []
        for path in image_paths:
            mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
            file_obj = path.open("rb")
            open_files.append(file_obj)
            files.append(("image[]", (path.name, file_obj, mime_type)))
            print(f"Loaded input image: {path}")

        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(url, data=data, files=files, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        fail(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        fail(f"Request error: {exc}")
    finally:
        for file_obj in open_files:
            file_obj.close()


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
    if args.input_image:
        image_paths = validate_input_images(args.input_image)
        response_json = create_edit(args.prompt, image_paths, size, api_key, base_url)
    else:
        response_json = create_generation(args.prompt, size, api_key, base_url)

    image_value = extract_image_value(response_json)
    write_image_value(image_value, output_path)
    elapsed_seconds = time.perf_counter() - start_time
    file_size = output_path.stat().st_size

    print(f"\nImage saved: {output_path.resolve()}")
    print(f"File size: {format_bytes(file_size)}")
    print(f"Generation time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
