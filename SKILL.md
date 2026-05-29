---
name: chatgpt-image-skills
description: Generate or edit images through the OpenAI ChatGPT Image API using HTTP requests. Use when Codex needs text-to-image generation or image editing with optional one or more input images, aspect-ratio control, resolution control, and file output.
---

# ChatGPT Image Skills

Generate or edit images through the OpenAI Image API. The bundled script uses `httpx` HTTP requests directly and does not use the OpenAI SDK.

## Command

```bash
python3 ~/.hermes/skills/chatgpt-image-skills/scripts/generate_image.py \
  --prompt "description" --filename "output.png" \
  [--input-image img1.png ...] \
  [--resolution 1K|2K|4K] [--aspect-ratio 1:1|16:9|9:16|4:3|3:4]
```

Always run from the user's working directory. Do not `cd` to the skill directory.

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--prompt` / `-p` | Yes | - | Image description or edit instruction |
| `--filename` / `-f` | Yes | - | Output file path (`.png`) |
| `--input-image` / `-i` | No | - | Input image path(s) for editing; pass one or more |
| `--resolution` / `-r` | No | `2K` | `1K`, `2K`, or `4K` |
| `--aspect-ratio` / `-a` | No | `16:9` | `1:1`, `16:9`, `9:16`, `4:3`, or `3:4` |

When the user does not specify these options, omit them and use the script defaults: `--resolution 2K`, `--aspect-ratio 16:9`.

## Filename

Pattern: `yyyy-mm-dd-hh-mm-ss-descriptive-name.png`

Extract a short description from the prompt (1-5 words, lowercase, hyphenated). If unsure, use a random ID such as `x9k2`.

## Prompt Rules

- Always write the prompt in English.
- Text inside the image can be Chinese; wrap it in single quotes, for example: `with text saying '你好'`.
- For generation: pass the user's description directly, polishing only when it is clearly underspecified.
- For editing: pass the edit instruction.

Precise edit template:

> "Change ONLY: <change>. Keep identical: subject, composition, pose, lighting, color palette, background, text, and style."

## Editing

Use `--input-image` with one or more image paths. The script calls the Image API edits endpoint and uploads files as multipart `image[]` fields.

## Preflight

- `command -v python3`
- `python3 -c "import httpx"`
- `test -n "$CHATGPT_IMAGE_API_KEY"`
- For editing: verify all `--input-image` paths exist.

## Output

- Save the generated image to the requested output path.
- Do not re-open or inspect the image unless the user explicitly asks.
