import argparse
import copy
import json
import math
import re
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import concurrent.futures
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


LOGGER = logging.getLogger("four_panel")
DEFAULT_REPLICATE_API_URL = "https://api.replicate.com/v1"


class GenerationError(Exception):
    """Raised when a model fails to generate an image."""


@dataclass
class PanelResult:
    order: int
    model_name: str
    image_path: Path
    width: int
    height: int
    is_placeholder: bool = False


class ReplicateClient:
    """Minimal Replicate API client for text-to-image generation."""

    def __init__(
        self,
        api_token: str,
        api_url: str = DEFAULT_REPLICATE_API_URL,
        poll_interval: float = 2.5,
        request_timeout: float = 600.0,
    ) -> None:
        if not api_token:
            raise ValueError("Replicate API token is required")
        self._api_url = api_url.rstrip("/")
        self._poll_interval = poll_interval
        self._request_timeout = request_timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Token {api_token}",
                "Content-Type": "application/json",
                "User-Agent": "four-panel-generator/1.1",
            }
        )

    def resolve_version(
        self,
        model_ref: str,
        version: Optional[str] = None,
        version_index: int = 0,
    ) -> str:
        """Resolve a Replicate model version id."""
        if version:
            return version

        if not model_ref:
            raise GenerationError("Replicate model reference missing from configuration")

        url = f"{self._api_url}/models/{model_ref}"
        resp = self._session.get(url, timeout=self._request_timeout)
        if resp.status_code == 404:
            raise GenerationError(f"Replicate model '{model_ref}' was not found")
        if resp.status_code >= 400:
            raise GenerationError(
                f"Unable to fetch model metadata for '{model_ref}': {resp.status_code} {resp.text}"
            )

        data = resp.json()
        if "latest_version" in data and version_index == 0:
            latest = data["latest_version"]
            if isinstance(latest, dict) and latest.get("id"):
                return latest["id"]

        version_sources = [
            data.get("versions_url"),
            f"{self._api_url}/models/{model_ref}/versions",
            f"https://replicate.com/api/models/{model_ref}/versions",
        ]

        seen = set()
        for endpoint in version_sources:
            if not endpoint or endpoint in seen:
                continue
            seen.add(endpoint)
            try:
                resolved = self._fetch_version_by_index(endpoint, version_index)
            except GenerationError as err:
                LOGGER.debug("Version lookup via %s failed: %s", endpoint, err)
                continue
            if resolved:
                return resolved

        raise GenerationError(
            f"Version index {version_index} is out of range for model '{model_ref}'"
        )

    def _fetch_version_by_index(self, base_url: str, version_index: int) -> Optional[str]:
        remaining = version_index
        next_url = base_url
        while next_url:
            resp = self._session.get(next_url, timeout=self._request_timeout)
            if resp.status_code == 404:
                return None
            if resp.status_code >= 400:
                raise GenerationError(
                    f"Unable to fetch versions from '{base_url}': {resp.status_code} {resp.text}"
                )
            payload = resp.json()
            entries, next_url = self._parse_versions_payload(payload, base_url, next_url)
            if not entries:
                return None
            if remaining < len(entries):
                entry = entries[remaining]
                if isinstance(entry, dict):
                    version_id = entry.get("id")
                else:
                    version_id = entry
                if version_id:
                    return version_id
                return None
            remaining -= len(entries)

        return None

    def _parse_versions_payload(
        self,
        payload: Dict[str, Any],
        base_url: str,
        current_url: str,
    ) -> Tuple[List[Any], Optional[str]]:
        entries: List[Any] = []
        next_url: Optional[str] = None

        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                entries = payload["results"]
                next_url = payload.get("next")
            elif isinstance(payload.get("data"), list):
                entries = payload["data"]
                has_more = payload.get("has_more")
                next_token = payload.get("next")
                if has_more and next_token:
                    next_url = self._normalize_next_url(base_url, current_url, next_token)
            elif isinstance(payload.get("versions"), list):
                entries = payload["versions"]

        return entries, next_url

    def _normalize_next_url(
        self,
        base_url: str,
        current_url: str,
        token: str,
    ) -> Optional[str]:
        if not token:
            return None
        if token.startswith("http"):
            return token
        if token.startswith("?"):
            stem = current_url.split("?")[0]
            return f"{stem}{token}"
        return urljoin(base_url if base_url.endswith("/") else f"{base_url}/", token.lstrip("/"))

    def predict(
        self,
        *,
        model_ref: str,
        prompt_payload: Dict[str, Any],
        version_id: Optional[str] = None,
        version_index: int = 0,
        timeout_override: Optional[float] = None,
    ) -> str:
        version = self.resolve_version(model_ref, version_id, version_index)
        body = {
            "version": version,
            "input": prompt_payload,
        }

        LOGGER.debug("Submitting prediction for %s (version=%s)", model_ref, version)
        resp = self._session.post(
            f"{self._api_url}/predictions",
            data=json.dumps(body),
            timeout=self._request_timeout,
        )
        if resp.status_code not in (200, 201):
            raise GenerationError(
                f"Failed to start prediction for '{model_ref}': {resp.status_code} {resp.text}"
            )

        prediction = resp.json()
        prediction_id = prediction.get("id")
        if not prediction_id:
            raise GenerationError("Replicate response missing prediction id")

        deadline = time.time() + (timeout_override or self._request_timeout)
        status_url = prediction.get("urls", {}).get("get") or f"{self._api_url}/predictions/{prediction_id}"

        while True:
            status = prediction.get("status")
            LOGGER.debug("Prediction %s status: %s", prediction_id, status)
            if status == "succeeded":
                break
            if status in {"failed", "canceled"}:
                raise GenerationError(
                    prediction.get("error") or f"Prediction {prediction_id} ended with status {status}"
                )

            if time.time() > deadline:
                raise GenerationError(
                    f"Prediction {prediction_id} timed out after {self._request_timeout} seconds"
                )

            time.sleep(self._poll_interval)
            resp = self._session.get(status_url, timeout=self._request_timeout)
            if resp.status_code >= 400:
                raise GenerationError(
                    f"Failed to refresh prediction status for '{model_ref}': {resp.status_code} {resp.text}"
                )
            prediction = resp.json()

        output = prediction.get("output")
        if not output:
            raise GenerationError(
                f"Prediction {prediction_id} succeeded but no output payload was provided"
            )

        image_url = _extract_first_url(output)
        if not image_url:
            raise GenerationError(
                f"Prediction {prediction_id} completed without a downloadable image url"
            )

        return image_url

    def upload_file(self, file_path: Path) -> str:
        if not file_path.exists() or not file_path.is_file():
            raise GenerationError(f"File not found: {file_path}")
        headers_content = self._session.headers.pop("Content-Type", None)
        try:
            with file_path.open('rb') as handle:
                files = {"file": (file_path.name, handle)}
                response = self._session.post(f"{self._api_url}/files", files=files, timeout=self._request_timeout)
        finally:
            if headers_content is not None:
                self._session.headers["Content-Type"] = headers_content
        if response.status_code >= 400:
            raise GenerationError(f"Failed to upload file to Replicate: {response.status_code} {response.text}")
        data = response.json()
        for key in ("serving_url", "serve_url", "download_url", "url"):
            url = data.get(key)
            if url:
                return url
        raise GenerationError("Replicate upload response missing file URL")


def _extract_first_url(payload: Any) -> Optional[str]:
    """Recursively search the payload for the first URL-like string."""
    if isinstance(payload, str) and payload.startswith("http"):
        return payload
    if isinstance(payload, dict):
        for key in ("image", "image_url", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in payload.values():
            url = _extract_first_url(value)
            if url:
                return url
    if isinstance(payload, (list, tuple)):
        for value in payload:
            url = _extract_first_url(value)
            if url:
                return url
    return None


# ---------------------------------------------------------------------------
# Interactive helpers and configuration management
# ---------------------------------------------------------------------------


def substitute_prompt_templates(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for key, replacement in replacements.items():
            token = f"{{{{{key}}}}}"
            result = result.replace(token, replacement)
        return result
    if isinstance(value, list):
        return [substitute_prompt_templates(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: substitute_prompt_templates(item, replacements) for key, item in value.items()}
    return value


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            os.environ.setdefault(key, value.strip())
    except OSError as err:
        LOGGER.warning("Unable to read .env file %s: %s", path, err)


def load_config(config_path: Path) -> Dict[str, Any]:
    try:
        with config_path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except UnicodeDecodeError:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def gather_prompt(cli_value: Optional[str], allow_input: bool) -> str:
    if cli_value:
        return cli_value
    if not allow_input:
        raise ValueError("Prompt is required when interactive input is disabled")

    while True:
        value = input("Enter the shared text prompt: ").strip()
        if value:
            return value
        print("Prompt cannot be empty.")


def parse_model_ids(raw_value: str) -> List[str]:
    separators = {",", " ", "\t", "\n"}
    tokens: List[str] = []
    current = []
    for char in raw_value:
        if char in separators:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return [token for token in (t.strip() for t in tokens) if token]


def format_default_value(value: Any, value_type: str) -> str:
    if value is None:
        return ""
    if value_type == "list" and isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def cast_placeholder_value(raw: str, value_type: str) -> Any:
    if value_type == "int":
        return int(raw)
    if value_type == "float":
        return float(raw)
    if value_type == "list":
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        return parts
    return raw


def resolve_placeholder_value(
    placeholder: Dict[str, Any],
    allow_input: bool,
    *,
    client: Optional[Any] = None,
    is_image: bool = False,
    preset_values: Optional[Dict[str, Any]] = None,
) -> Any:
    name = placeholder.get("name") or placeholder.get("target", "value")
    prompt_text = placeholder.get("prompt") or f"Enter value for {name}"
    value_type = placeholder.get("type", "string")
    default = placeholder.get("default")
    required = bool(placeholder.get("required", False))
    preset_value = None
    if preset_values is not None:
        preset_value = preset_values.get(name) or preset_values.get(placeholder.get("target"))

    def _return_image_value(value: Any) -> Any:
        if isinstance(value, list):
            processed: List[str] = []
            for item in value:
                try:
                    result = _process_image_reference(str(item), client)
                except GenerationError as exc:
                    raise ValueError(f"Placeholder '{name}' error: {exc}") from exc
                if result:
                    processed.append(result)
            if processed:
                return processed
            if required:
                raise ValueError(f"Placeholder '{name}' could not process provided image references")
            return []
        try:
            processed = _process_image_reference(str(value), client)
        except GenerationError as exc:
            raise ValueError(f"Placeholder '{name}' error: {exc}") from exc
        if processed:
            return processed
        if required:
            raise ValueError(f"Placeholder '{name}' could not process provided image reference")
        return "" if value_type == "string" else []

    if preset_value is not None:
        if is_image:
            result = _return_image_value(preset_value)
            if value_type == "list" and result and not isinstance(result, list):
                return [result]
            return result
        if value_type == "list" and not isinstance(preset_value, list):
            return [preset_value]
        return copy.deepcopy(preset_value)

    if not allow_input:
        if default is not None:
            return copy.deepcopy(default)
        if required:
            raise ValueError(f"Placeholder '{name}' requires a value but interactive input is disabled")
        return None if value_type != "list" else []

    if value_type == "list" and is_image:
        collected: List[str] = []
        print(f"{prompt_text} (enter URLs or local file paths; leave blank to finish)")
        while True:
            raw = input("> " ).strip()
            if not raw:
                if collected:
                    return collected
                if default is not None:
                    return copy.deepcopy(default)
                if not required:
                    return []
                print("At least one image is required.")
                continue
            try:
                processed = _process_image_reference(raw, client)
            except GenerationError as exc:
                print(f"Placeholder '{name}' error: {exc}")
                continue
            if processed:
                collected.append(processed)
        return collected

    if value_type == "string" and is_image:
        while True:
            default_hint = format_default_value(default, value_type)
            suffix = f" [default: {default_hint}]" if default_hint else ""
            raw = input(f"{prompt_text}{suffix}: " ).strip()
            if not raw:
                if default is not None:
                    return copy.deepcopy(default)
                if not required:
                    return ""
                print("This value is required.")
                continue
            try:
                processed = _process_image_reference(raw, client)
            except GenerationError as exc:
                print(f"Placeholder '{name}' error: {exc}")
                continue
            if processed:
                return processed
            if not required:
                return ""
            print("Valid image reference required.")

    while True:
        default_hint = format_default_value(default, value_type)
        suffix = f" [default: {default_hint}]" if default_hint else ""
        raw = input(f"{prompt_text}{suffix}: " ).strip()
        if not raw:
            if default is not None:
                return copy.deepcopy(default)
            if not required:
                return None if value_type != "list" else []
            print("This value is required.")
            continue
        try:
            return cast_placeholder_value(raw, value_type)
        except ValueError as exc:
            print(f"Invalid value: {exc}")



def set_nested_value(container: Dict[str, Any], path: List[str], value: Any, omit_if_none: bool = True) -> None:
    current: Dict[str, Any] = container
    for key in path[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    final_key = path[-1]
    if value is None and omit_if_none:
        current.pop(final_key, None)
    else:
        current[final_key] = value


def build_placeholder_presets(entry: Dict[str, Any], image_reference: Optional[str]) -> Dict[str, Any]:
    if not image_reference:
        return {}
    presets: Dict[str, Any] = {}
    for placeholder in entry.get("placeholders", []) or []:
        if not placeholder_targets_image(placeholder):
            continue
        key = placeholder.get("name") or placeholder.get("target") or "value"
        if placeholder.get("type") == "list":
            # Split comma-separated URLs into a list
            image_urls = [url.strip() for url in image_reference.split(",") if url.strip()]
            presets[key] = image_urls
        else:
            # For single image fields (like Flux), use only the first image
            image_urls = [url.strip() for url in image_reference.split(",") if url.strip()]
            presets[key] = image_urls[0] if image_urls else image_reference
        LOGGER.debug("Prepared image placeholder '%s' with reference %s", key, presets[key])
    return presets


def apply_placeholders(
    entry: Dict[str, Any],
    allow_input: bool,
    *,
    client: Optional[Any] = None,
    enable_image_placeholders: bool = True,
    preset_values: Optional[Dict[str, Any]] = None,
) -> None:
    placeholders = entry.get("placeholders") or []
    active_placeholders: List[Dict[str, Any]] = []
    for placeholder in placeholders:
        if not enable_image_placeholders and placeholder_targets_image(placeholder):
            continue
        active_placeholders.append(placeholder)
    entry["placeholders"] = active_placeholders

    for placeholder in active_placeholders:
        target = placeholder.get("target")
        if not target:
            continue
        is_image = placeholder_targets_image(placeholder)
        try:
            resolved = resolve_placeholder_value(
                placeholder,
                allow_input,
                client=client,
                is_image=is_image,
                preset_values=preset_values,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        path = str(target).split(".")
        omit_if_none = bool(placeholder.get("omit_if_none", True))
        set_nested_value(entry, path, resolved, omit_if_none=omit_if_none)
        LOGGER.debug("Applied placeholder for %s -> %s", path, resolved)
    entry.pop("placeholders", None)


def choose_models(
    catalog: List[Dict[str, Any]],
    default_selection: List[str],
    count: int,
    cli_models: Optional[str],
    allow_input: bool,
    allow_partial: bool = False,
    min_required: int = 1,
) -> List[Dict[str, Any]]:
    if count <= 0:
        raise ValueError("Panel count must be greater than zero")

    catalog_by_id = {entry.get("id", entry.get("name")): entry for entry in catalog}
    available_count = len(catalog_by_id)
    if available_count < min_required:
        raise ValueError("Catalog does not contain enough models to satisfy the minimum required panel count")

    if not allow_partial and available_count < count:
        raise ValueError("Catalog does not contain enough models to satisfy the desired panel count")

    max_selectable = count if not allow_partial else min(count, available_count)

    if cli_models:
        requested_ids = parse_model_ids(cli_models)
        if allow_partial:
            if len(requested_ids) < min_required:
                raise ValueError(f"Expected at least {min_required} models but received {len(requested_ids)}")
            if len(requested_ids) > count:
                raise ValueError(f"Expected no more than {count} models but received {len(requested_ids)}")
        else:
            if len(requested_ids) != count:
                raise ValueError(f"Expected {count} models but received {len(requested_ids)}")
        missing = [model_id for model_id in requested_ids if model_id not in catalog_by_id]
        if missing:
            raise ValueError(f"Unknown model identifiers: {', '.join(missing)}")
        return [copy.deepcopy(catalog_by_id[model_id]) for model_id in requested_ids]

    available_ids = list(catalog_by_id.keys())

    if not allow_input:
        fallback_ids = default_selection[:max_selectable] if default_selection else available_ids[:max_selectable]
        missing_defaults = [
            model_id
            for model_id in (default_selection[:max_selectable] if default_selection else [])
            if model_id not in catalog_by_id
        ]
        if missing_defaults:
            raise ValueError(
                "Default model selection references unknown identifiers: " + ", ".join(missing_defaults)
            )
        fallback_ids = [model_id for model_id in fallback_ids if model_id in catalog_by_id]
        if len(fallback_ids) < min_required:
            fallback_ids = available_ids[:max_selectable]
        if not allow_partial and len(fallback_ids) != count:
            raise ValueError("Default model selection does not include enough models to satisfy the desired panel count")
        if len(fallback_ids) < min_required:
            raise ValueError("Catalog does not contain enough models to satisfy the minimum required panel count")
        return [copy.deepcopy(catalog_by_id[model_id]) for model_id in fallback_ids]

    # Interactive selection.
    print("\nAvailable models:")
    ordered_entries = list(catalog)
    for idx, entry in enumerate(ordered_entries, start=1):
        identifier = entry.get("id", f"model-{idx}")
        label = entry.get("name", identifier)
        description = entry.get("description")
        default_marker = " (default)" if identifier in (default_selection or []) else ""
        capability_tags = []
        if model_supports_image_input(entry):
            capability_tags.append("IMG")
        capability_text = f" [{', '.join(capability_tags)}]" if capability_tags else ""
        print(f"  [{idx}] {label}{capability_text} - {identifier}{default_marker}")
        if description:
            print(f"       {description}")

    max_choices = min(count, len(ordered_entries)) if allow_partial else count
    max_choices = max(max_choices, min_required)
    min_choices = count if not allow_partial else min(max_choices, max(min_required, 1 if max_choices > 0 else 0))

    default_prompt_ids = [
        model_id for model_id in default_selection if model_id in catalog_by_id
    ][:max_choices] or [
        ordered_entries[i].get("id", f"model-{i + 1}") for i in range(max_choices)
    ]
    default_prompt_numbers: List[str] = []
    for model_id in default_prompt_ids:
        for idx, entry in enumerate(ordered_entries, start=1):
            if entry.get("id") == model_id:
                default_prompt_numbers.append(str(idx))
                break

    default_hint = ",".join(default_prompt_numbers) if default_prompt_numbers else ""
    while True:
        requirement_text = (
            f"{min_choices}"
            if min_choices == max_choices
            else f"{min_choices}-{max_choices}"
        )
        raw = input(
            f"Select {requirement_text} models by number (comma-separated){f' [{default_hint}]' if default_hint else ''}: "
        ).strip()
        if not raw and default_prompt_numbers:
            raw = ",".join(default_prompt_numbers)
        indices = parse_model_ids(raw)
        if len(indices) < min_choices or len(indices) > max_choices:
            if min_choices == max_choices:
                print(f"Please provide exactly {min_choices} selections.")
            else:
                print(f"Please select between {min_choices} and {max_choices} models.")
            continue
        try:
            chosen_indices = [int(index) for index in indices]
        except ValueError:
            print("Selections must be numeric indexes from the list above.")
            continue
        if any(index < 1 or index > len(ordered_entries) for index in chosen_indices):
            print("One or more selections are outside the valid range.")
            continue
        if len(set(chosen_indices)) != len(chosen_indices):
            print("Duplicate selections detected. Please choose unique models.")
            continue
        chosen_entries = [copy.deepcopy(ordered_entries[index - 1]) for index in chosen_indices]
        return chosen_entries


# ---------------------------------------------------------------------------
# Collage utilities
# ---------------------------------------------------------------------------


def load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_bbox(font: ImageFont.ImageFont, text: str) -> Tuple[int, int, int, int]:
    sample = text if text else " "
    try:
        bbox = font.getbbox(sample)
    except AttributeError:
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), sample, font=font)
    return bbox


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    bbox = _text_bbox(font, text)
    return bbox[2] - bbox[0]


def _text_height(font: ImageFont.ImageFont, text: str = "Ag") -> int:
    bbox = _text_bbox(font, text)
    return bbox[3] - bbox[1]


def wrap_text_to_width(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return [""]
    if max_width <= 0:
        return [cleaned]
    words = cleaned.split()
    if not words:
        return [cleaned]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_width(font, candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def format_model_label(label: str) -> str:
    if not label:
        return ""
    stripped = re.sub(r"\s*\([^)]*\)", "", label).strip()
    return stripped or label.strip()


def parse_hex_color(value: str) -> Tuple[int, int, int]:
    if not value:
        return (17, 17, 17)
    color = value.strip().lstrip('#')
    if len(color) == 6:
        try:
            return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return (17, 17, 17)


def resize_with_padding(image: Image.Image, target_size: Tuple[int, int], fill_color: Tuple[int, int, int]) -> Image.Image:
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target size must be positive")
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source image has invalid dimensions")
    scale = min(target_width / source_width, target_height / source_height, 1.0)
    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))
    resized = image.resize((new_width, new_height), Image.LANCZOS) if scale < 1.0 else image.copy()
    background = Image.new('RGB', target_size, fill_color)
    offset_x = max(0, (target_width - new_width) // 2)
    offset_y = max(0, (target_height - new_height) // 2)
    background.paste(resized, (offset_x, offset_y))
    return background

def placeholder_targets_image(placeholder: Dict[str, Any]) -> bool:
    target = (placeholder.get("target") or "").lower()
    return any(key in target for key in ("image_input", "input.image", "input.image_url", "input.input_images", "input.input_image"))


def model_supports_image_input(entry: Dict[str, Any]) -> bool:
    if "supports_image_input" in entry:
        return bool(entry.get("supports_image_input"))
    return any(placeholder_targets_image(ph) for ph in (entry.get("placeholders") or []))


def _process_image_reference(value: str, client: Optional[Any]) -> Optional[str]:
    candidate = value.strip()
    if not candidate:
        return None
    potential_path = Path(candidate)
    if potential_path.exists() and potential_path.is_file():
        if client is None:
            raise GenerationError(f"Local file provided ({potential_path}) but uploads are unavailable in this context.")
        LOGGER.debug("Uploading local image reference %s", potential_path)
        print(f"[four-panel] Uploading local image {potential_path}", flush=True)
        try:
            uploaded_url = client.upload_file(potential_path)
        except GenerationError as exc:
            LOGGER.error("Failed to upload %s: %s", potential_path, exc)
            print(f"[four-panel] Upload failed: {exc}", flush=True)
            raise
        LOGGER.info("Uploaded %s for editing -> %s", potential_path, uploaded_url)
        print(f"[four-panel] Uploaded -> {uploaded_url}", flush=True)
        return uploaded_url
    if candidate.lower().startswith("http://") or candidate.lower().startswith("https://"):
        LOGGER.debug("Using remote image reference %s", candidate)
        print(f"[four-panel] Using remote image reference {candidate}", flush=True)
    else:
        LOGGER.warning("Using non-HTTP image reference %s; Replicate may reject this value.", candidate)
        print(f"[four-panel] Non-HTTP image reference {candidate}", flush=True)
    return candidate


def annotate_image(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    max_text_width: int,
) -> Image.Image:
    annotated = image.convert("RGBA")
    width, height = annotated.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    usable_width = max(32, min(width, max_text_width))
    lines = wrap_text_to_width(text, font, usable_width)
    line_height = _text_height(font)
    line_spacing = max(6, line_height // 5)
    padding = max(12, line_height // 3)
    block_height = line_height * len(lines) + line_spacing * (len(lines) - 1) + padding * 2
    label_height = max(height // 7, block_height)
    top = height - label_height
    draw.rectangle(
        [(0, top), (width, height)],
        fill=(0, 0, 0, 208),
    )
    text_y = top + padding + max(0, (label_height - block_height) // 2)
    inner_width = usable_width
    inner_left = max(padding, (width - inner_width) // 2)
    for line in lines:
        line_width = _text_width(font, line)
        text_x = inner_left + max(0, (inner_width - line_width) // 2)
        draw.text((text_x, text_y), line, font=font, fill=(255, 255, 255, 240))
        text_y += line_height + line_spacing
    combined = Image.alpha_composite(annotated, overlay)
    return combined.convert("RGB")


def download_image(session: requests.Session, url: str, destination: Path, timeout: float) -> Path:
    LOGGER.debug("Downloading image from %s", url)
    response = session.get(url, timeout=timeout)
    if response.status_code >= 400:
        raise GenerationError(f"Failed to download generated image: {response.status_code} {response.text}")
    destination.write_bytes(response.content)
    return destination


def compose_reference_slide(
    reference_images: List[str],
    prompt_text: str,
    output_file: Path,
    tile_size: int,
    margin: int,
    background_color: str = "#111111",
) -> Path:
    """
    Create a reference slide showing input images and prompt text.
    Handles 1-3 reference images dynamically.
    """
    if not reference_images:
        raise ValueError("At least one reference image is required")
    
    # Download and load reference images
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "four-panel-generator/1.0"})
    
    ref_panels = []
    for i, img_url in enumerate(reference_images[:3]):  # Max 3 images
        try:
            response = session.get(img_url, timeout=30)
            response.raise_for_status()
            
            temp_path = Path(f"/tmp/ref_img_{i}.jpg")
            temp_path.write_bytes(response.content)
            
            with Image.open(temp_path) as img:
                ref_panels.append(img.copy())
            temp_path.unlink()  # Clean up
            
        except Exception as e:
            LOGGER.warning(f"Failed to load reference image {img_url}: {e}")
            # Create placeholder
            placeholder = Image.new("RGB", (512, 512), (64, 64, 64))
            ref_panels.append(placeholder)
    
    session.close()
    
    # Calculate layout based on number of images
    num_images = len(ref_panels)
    
    if num_images == 1:
        # Single large image, centered
        layout = [(0, 0)]
        cols, rows = 1, 1
    elif num_images == 2:
        # Two images side by side
        layout = [(0, 0), (1, 0)]
        cols, rows = 2, 1
    else:  # 3 images
        # Two on top, one centered below
        layout = [(0, 0), (1, 0), (0.5, 1)]
        cols, rows = 2, 2
    
    # Calculate dimensions
    ref_tile_size = tile_size // 2  # Smaller than main panels
    margin_small = margin // 2
    
    fill_rgb = parse_hex_color(background_color)
    
    # Prepare reference images
    prepared_refs = []
    for img in ref_panels:
        resized = resize_with_padding(img, (ref_tile_size, ref_tile_size), fill_rgb)
        prepared_refs.append(resized)
    
    # Calculate grid dimensions
    if num_images <= 2:
        grid_width = cols * ref_tile_size + (cols + 1) * margin_small
        grid_height = ref_tile_size + 2 * margin_small
    else:  # 3 images
        grid_width = 2 * ref_tile_size + 3 * margin_small
        grid_height = 2 * ref_tile_size + 3 * margin_small
    
    # Add space for prompt text
    prompt_font_size = max(32, tile_size // 32)
    prompt_font = load_font(prompt_font_size)
    prompt_lines = wrap_text_to_width(prompt_text, prompt_font, grid_width - margin_small * 2)
    
    line_height = _text_height(prompt_font)
    line_spacing = max(6, line_height // 6)
    text_padding = margin_small
    
    if prompt_lines:
        text_area_height = line_height * len(prompt_lines) + line_spacing * (len(prompt_lines) - 1) + text_padding * 2
    else:
        text_area_height = 0
    
    # Create canvas
    total_height = grid_height + text_area_height + margin_small * 2
    canvas_width = max(grid_width + margin_small * 2, tile_size)
    canvas_height = total_height
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)
    draw = ImageDraw.Draw(canvas)
    
    # Draw prompt text at top
    if prompt_lines:
        text_y = margin_small + text_padding
        for line in prompt_lines:
            line_width = _text_width(prompt_font, line)
            text_x = (canvas_width - line_width) // 2
            draw.text((text_x, text_y), line, font=prompt_font, fill=(255, 255, 255))
            text_y += line_height + line_spacing
    
    # Draw reference images
    grid_start_y = text_area_height + margin_small * 2
    grid_start_x = (canvas_width - grid_width) // 2
    
    for i, (col_pos, row_pos) in enumerate(layout[:len(prepared_refs)]):
        if i < len(prepared_refs):
            if num_images == 3 and i == 2:
                # Center the third image
                x = grid_start_x + margin_small + int(col_pos * ref_tile_size)
            else:
                x = grid_start_x + margin_small + int(col_pos * (ref_tile_size + margin_small))
            y = grid_start_y + margin_small + int(row_pos * (ref_tile_size + margin_small))
            canvas.paste(prepared_refs[i], (x, y))
    
    # Add "Reference Images" label
    label_font = load_font(max(24, tile_size // 48))
    label_text = f"Reference Image{'s' if len(reference_images) > 1 else ''}"
    label_width = _text_width(label_font, label_text)
    label_x = (canvas_width - label_width) // 2
    label_y = grid_start_y + grid_height + margin_small
    draw.text((label_x, label_y), label_text, font=label_font, fill=(200, 200, 200))
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    LOGGER.info("Saved reference slide to %s", output_file)
    return output_file


def compose_grid_numbered(
    panels: Iterable[PanelResult],
    output_file: Path,
    tile_size: int,
    margin: int,
    background_color: str = "#111111",
    prompt_text: str = "",
) -> Path:
    """
    Create a numbered grid where panels are labeled 1, 2, 3, 4 based on position.
    Position order: 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right
    """
    panel_list = sorted(panels, key=lambda panel: panel.order)
    if not panel_list:
        raise ValueError("At least one panel is required to compose the grid")

    margin = max(0, margin)
    aspect_ratio = 16 / 9
    min_width = min(panel.width for panel in panel_list)
    min_height = min(panel.height for panel in panel_list)

    target_height = min(tile_size, min_height)
    target_width = int(round(target_height * aspect_ratio))
    if target_width > min_width:
        target_width = min_width
        target_height = int(round(target_width / aspect_ratio))

    tile_height = max(64, target_height)
    tile_width = max(64, target_width)

    horizontal_spacing = max(4, margin // 4)
    vertical_spacing = max(4, margin // 4)
    columns = 2
    rows = math.ceil(len(panel_list) / columns)
    grid_width = columns * tile_width + (columns + 1) * horizontal_spacing
    grid_height = rows * tile_height + (rows + 1) * vertical_spacing

    LOGGER.debug(
        "Composing numbered grid sized %sx%s (tile=%sx%s, h_spacing=%s, v_spacing=%s)",
        grid_width,
        grid_height,
        tile_width,
        tile_height,
        horizontal_spacing,
        vertical_spacing,
    )

    fill_rgb = parse_hex_color(background_color)
    label_font_size = max(24, tile_height // 18)
    label_font = load_font(label_font_size)
    text_max_width = max(32, tile_width - 16)

    prepared_panels: List[Image.Image] = []
    for index, panel in enumerate(panel_list):
        # Calculate position-based number (1-4)
        row = index // columns
        column = index % columns
        position_number = str(row * columns + column + 1)
        
        with Image.open(panel.image_path) as raw_img:
            framed = resize_with_padding(raw_img, (tile_width, tile_height), fill_rgb)
            annotated = annotate_image(framed, position_number, label_font, text_max_width)
        prepared_panels.append(annotated)

    prompt_font_size = max(label_font_size + 4, 40)
    prompt_font = load_font(prompt_font_size)
    text_area_width = max(1, grid_width - 2 * horizontal_spacing)
    prompt_lines: List[str] = wrap_text_to_width(prompt_text, prompt_font, text_area_width) if prompt_text else []
    line_height = _text_height(prompt_font)
    line_spacing = max(8, line_height // 6)
    banner_padding = max(horizontal_spacing // 2, line_height // 3)
    if prompt_lines:
        prompt_area_height = line_height * len(prompt_lines) + line_spacing * (len(prompt_lines) - 1) + banner_padding * 2
    else:
        prompt_area_height = 0
    prompt_gap = vertical_spacing if prompt_lines else 0

    content_width = grid_width
    content_height = prompt_area_height + prompt_gap + grid_height
    outer_margin = max(horizontal_spacing, vertical_spacing)
    min_width_canvas = content_width + 2 * outer_margin
    min_height_canvas = content_height + 2 * outer_margin

    final_width = max(min_width_canvas, math.ceil(min_height_canvas * 16 / 9))
    final_height = math.ceil(final_width * 9 / 16)
    while final_height < min_height_canvas:
        final_width += 16
        final_height = math.ceil(final_width * 9 / 16)

    canvas = Image.new("RGB", (final_width, final_height), background_color)
    draw = ImageDraw.Draw(canvas)

    content_left = max(0, (final_width - content_width) // 2)
    content_top = max(0, (final_height - content_height) // 2)

    if prompt_lines:
        prompt_x = content_left
        prompt_y = content_top
        draw.rectangle(
            [
                (prompt_x, prompt_y),
                (prompt_x + content_width, prompt_y + prompt_area_height),
            ],
            fill=(0, 0, 0, 210),
        )
        text_y = prompt_y + banner_padding
        for line in prompt_lines:
            line_width = _text_width(prompt_font, line)
            text_x = prompt_x + max(banner_padding, (content_width - line_width) // 2)
            draw.text((text_x, text_y), line, font=prompt_font, fill=(255, 255, 255))
            text_y += line_height + line_spacing
    grid_origin_y = content_top + prompt_area_height + prompt_gap
    grid_origin_x = content_left

    for index, annotated in enumerate(prepared_panels):
        row = index // columns
        column = index % columns
        x = grid_origin_x + horizontal_spacing + column * (tile_width + horizontal_spacing)
        y = grid_origin_y + vertical_spacing + row * (tile_height + vertical_spacing)
        canvas.paste(annotated, (x, y))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    LOGGER.info("Saved numbered panel collage to %s", output_file)
    return output_file


def compose_grid_unlabeled(
    panels: Iterable[PanelResult],
    output_file: Path,
    tile_size: int,
    margin: int,
    background_color: str = "#111111",
    prompt_text: str = "",
) -> Path:
    """Create a collage without model labels for blind evaluation."""
    panel_list = sorted(panels, key=lambda panel: panel.order)
    if not panel_list:
        raise ValueError("At least one panel is required to compose the grid")

    margin = max(0, margin)
    aspect_ratio = 16 / 9
    min_width = min(panel.width for panel in panel_list)
    min_height = min(panel.height for panel in panel_list)

    target_height = min(tile_size, min_height)
    target_width = int(round(target_height * aspect_ratio))
    if target_width > min_width:
        target_width = min_width
        target_height = int(round(target_width / aspect_ratio))

    tile_height = max(64, target_height)
    tile_width = max(64, target_width)

    horizontal_spacing = max(4, margin // 4)  # Much smaller horizontal gaps
    vertical_spacing = max(4, margin // 4)   # Much smaller vertical gaps
    columns = 2
    rows = math.ceil(len(panel_list) / columns)
    grid_width = columns * tile_width + (columns + 1) * horizontal_spacing
    grid_height = rows * tile_height + (rows + 1) * vertical_spacing

    LOGGER.debug(
        "Composing unlabeled grid sized %sx%s (tile=%sx%s, h_spacing=%s, v_spacing=%s)",
        grid_width,
        grid_height,
        tile_width,
        tile_height,
        horizontal_spacing,
        vertical_spacing,
    )

    fill_rgb = parse_hex_color(background_color)
    prepared_panels: List[Image.Image] = []
    for panel in panel_list:
        with Image.open(panel.image_path) as raw_img:
            framed = resize_with_padding(raw_img, (tile_width, tile_height), fill_rgb)
        prepared_panels.append(framed)

    prompt_font_size = max(40, tile_height // 12)
    prompt_font = load_font(prompt_font_size)
    text_area_width = max(1, grid_width - 2 * horizontal_spacing)
    prompt_lines: List[str] = wrap_text_to_width(prompt_text, prompt_font, text_area_width) if prompt_text else []
    line_height = _text_height(prompt_font)
    line_spacing = max(8, line_height // 6)
    banner_padding = max(horizontal_spacing // 2, line_height // 3)
    if prompt_lines:
        prompt_area_height = line_height * len(prompt_lines) + line_spacing * (len(prompt_lines) - 1) + banner_padding * 2
    else:
        prompt_area_height = 0
    prompt_gap = vertical_spacing if prompt_lines else 0

    content_width = grid_width
    content_height = prompt_area_height + prompt_gap + grid_height
    outer_margin = max(horizontal_spacing, vertical_spacing)
    min_width_canvas = content_width + 2 * outer_margin
    min_height_canvas = content_height + 2 * outer_margin

    final_width = max(min_width_canvas, math.ceil(min_height_canvas * 16 / 9))
    final_height = math.ceil(final_width * 9 / 16)
    while final_height < min_height_canvas:
        final_width += 16
        final_height = math.ceil(final_width * 9 / 16)

    canvas = Image.new("RGB", (final_width, final_height), background_color)
    draw = ImageDraw.Draw(canvas)

    content_left = max(0, (final_width - content_width) // 2)
    content_top = max(0, (final_height - content_height) // 2)

    if prompt_lines:
        prompt_x = content_left
        prompt_y = content_top
        draw.rectangle(
            [
                (prompt_x, prompt_y),
                (prompt_x + content_width, prompt_y + prompt_area_height),
            ],
            fill=(0, 0, 0, 210),
        )
        text_y = prompt_y + banner_padding
        for line in prompt_lines:
            line_width = _text_width(prompt_font, line)
            text_x = prompt_x + max(banner_padding, (content_width - line_width) // 2)
            draw.text((text_x, text_y), line, font=prompt_font, fill=(255, 255, 255))
            text_y += line_height + line_spacing
    grid_origin_y = content_top + prompt_area_height + prompt_gap
    grid_origin_x = content_left

    for index, framed in enumerate(prepared_panels):
        row = index // columns
        column = index % columns
        x = grid_origin_x + horizontal_spacing + column * (tile_width + horizontal_spacing)
        y = grid_origin_y + vertical_spacing + row * (tile_height + vertical_spacing)
        canvas.paste(framed, (x, y))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    LOGGER.info("Saved unlabeled panel collage to %s", output_file)
    return output_file


def compose_grid(
    panels: Iterable[PanelResult],
    output_file: Path,
    tile_size: int,
    margin: int,
    background_color: str = "#111111",
    prompt_text: str = "",
) -> Path:
    panel_list = sorted(panels, key=lambda panel: panel.order)
    if not panel_list:
        raise ValueError("At least one panel is required to compose the grid")

    margin = max(0, margin)
    aspect_ratio = 16 / 9
    min_width = min(panel.width for panel in panel_list)
    min_height = min(panel.height for panel in panel_list)

    target_height = min(tile_size, min_height)
    target_width = int(round(target_height * aspect_ratio))
    if target_width > min_width:
        target_width = min_width
        target_height = int(round(target_width / aspect_ratio))

    tile_height = max(64, target_height)
    tile_width = max(64, target_width)

    horizontal_spacing = max(4, margin // 4)  # Much smaller horizontal gaps
    vertical_spacing = max(4, margin // 4)   # Much smaller vertical gaps
    columns = 2
    rows = math.ceil(len(panel_list) / columns)
    grid_width = columns * tile_width + (columns + 1) * horizontal_spacing
    grid_height = rows * tile_height + (rows + 1) * vertical_spacing

    LOGGER.debug(
        "Composing grid sized %sx%s (tile=%sx%s, h_spacing=%s, v_spacing=%s)",
        grid_width,
        grid_height,
        tile_width,
        tile_height,
        horizontal_spacing,
        vertical_spacing,
    )

    label_font_size = max(36, tile_height // 12)
    label_font = load_font(label_font_size)
    text_max_width = max(64, tile_width - 2 * horizontal_spacing)
    fill_rgb = parse_hex_color(background_color)
    prepared_panels: List[Image.Image] = []
    for panel in panel_list:
        label = format_model_label(panel.model_name)
        with Image.open(panel.image_path) as raw_img:
            framed = resize_with_padding(raw_img, (tile_width, tile_height), fill_rgb)
            annotated = annotate_image(framed, label, label_font, text_max_width)
        prepared_panels.append(annotated)

    prompt_font_size = max(label_font_size + 4, 40)
    prompt_font = load_font(prompt_font_size)
    text_area_width = max(1, grid_width - 2 * horizontal_spacing)
    prompt_lines: List[str] = wrap_text_to_width(prompt_text, prompt_font, text_area_width) if prompt_text else []
    line_height = _text_height(prompt_font)
    line_spacing = max(8, line_height // 6)
    banner_padding = max(horizontal_spacing // 2, line_height // 3)
    if prompt_lines:
        prompt_area_height = line_height * len(prompt_lines) + line_spacing * (len(prompt_lines) - 1) + banner_padding * 2
    else:
        prompt_area_height = 0
    prompt_gap = vertical_spacing if prompt_lines else 0

    content_width = grid_width
    content_height = prompt_area_height + prompt_gap + grid_height
    outer_margin = max(horizontal_spacing, vertical_spacing)
    min_width_canvas = content_width + 2 * outer_margin
    min_height_canvas = content_height + 2 * outer_margin

    final_width = max(min_width_canvas, math.ceil(min_height_canvas * 16 / 9))
    final_height = math.ceil(final_width * 9 / 16)
    while final_height < min_height_canvas:
        final_width += 16
        final_height = math.ceil(final_width * 9 / 16)

    canvas = Image.new("RGB", (final_width, final_height), background_color)
    draw = ImageDraw.Draw(canvas)

    content_left = max(0, (final_width - content_width) // 2)
    content_top = max(0, (final_height - content_height) // 2)

    if prompt_lines:
        prompt_x = content_left
        prompt_y = content_top
        draw.rectangle(
            [
                (prompt_x, prompt_y),
                (prompt_x + content_width, prompt_y + prompt_area_height),
            ],
            fill=(0, 0, 0, 210),
        )
        text_y = prompt_y + banner_padding
        for line in prompt_lines:
            line_width = _text_width(prompt_font, line)
            text_x = prompt_x + max(banner_padding, (content_width - line_width) // 2)
            draw.text((text_x, text_y), line, font=prompt_font, fill=(255, 255, 255))
            text_y += line_height + line_spacing
    grid_origin_y = content_top + prompt_area_height + prompt_gap
    grid_origin_x = content_left

    for index, annotated in enumerate(prepared_panels):
        row = index // columns
        column = index % columns
        x = grid_origin_x + horizontal_spacing + column * (tile_width + horizontal_spacing)
        y = grid_origin_y + vertical_spacing + row * (tile_height + vertical_spacing)
        canvas.paste(annotated, (x, y))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    LOGGER.info("Saved panel collage to %s", output_file)
    return output_file




def generate_panel(
    index: int,
    spec: Dict[str, Any],
    replacements: Dict[str, str],
    output_dir: Path,
    client_params: Dict[str, Any],
    allow_input: bool,
    enable_image_placeholders: bool,
    preset_values: Optional[Dict[str, Any]],
    request_timeout: float,
) -> PanelResult:
    spec_copy = copy.deepcopy(spec)
    client = ReplicateClient(
        client_params["api_token"],
        api_url=client_params["api_url"],
        poll_interval=client_params["poll_interval"],
        request_timeout=client_params["request_timeout"],
    )
    try:
        apply_placeholders(
            spec_copy,
            allow_input,
            client=client if enable_image_placeholders else None,
            enable_image_placeholders=enable_image_placeholders,
            preset_values=preset_values,
        )
        provider = spec_copy.get("provider")
        if provider != "replicate":
            raise GenerationError(f"Unsupported provider '{provider}'")
        model_ref = spec_copy.get("model")
        version_id = spec_copy.get("version")
        version_index = spec_copy.get("version_index", 0)
        prompt_payload = substitute_prompt_templates(spec_copy.get("input", {}), replacements)
        name = spec_copy.get("name") or model_ref or f"Model {index}"
        if enable_image_placeholders:
            # Check multiple possible image field names
            image_field = (prompt_payload.get("image_input") or 
                          prompt_payload.get("image") or 
                          prompt_payload.get("input_images") or
                          prompt_payload.get("input_image"))
            image_count = 0
            if isinstance(image_field, list):
                image_count = len([img for img in image_field if img])  # Count non-empty entries
            elif image_field:
                image_count = 1
            LOGGER.info("Editing payload for %s includes %s image reference(s)", name, image_count)
            if isinstance(image_field, list) and image_field and not all(isinstance(item, str) and item.startswith("http") for item in image_field):
                LOGGER.warning("One or more image references for %s do not look like URLs: %s", name, image_field)
        LOGGER.info("Generating with %s", name)
        image_url = client.predict(
            model_ref=model_ref,
            prompt_payload=prompt_payload,
            version_id=version_id,
            version_index=version_index,
            timeout_override=spec_copy.get("timeout_seconds"),
        )
    except GenerationError as exc:
        raise GenerationError(f"{name}: {exc}") from exc

    panel_filename = output_dir / f"{index:02d}-{slugify(name)}.png"
    download_session = requests.Session()
    download_session.headers.update({"User-Agent": "four-panel-generator/1.1"})
    try:
        download_image(download_session, image_url, panel_filename, request_timeout)
    except GenerationError as exc:
        raise GenerationError(f"{name}: {exc}") from exc
    finally:
        download_session.close()

    with Image.open(panel_filename) as saved_image:
        width, height = saved_image.size
        
        # No special processing - all models keep their natural aspect ratios
        
    LOGGER.info("Saved panel for %s to %s", name, panel_filename)
    return PanelResult(order=index, model_name=name, image_path=panel_filename, width=width, height=height, is_placeholder=False)


def generate_collage(
    prompt: str,
    model_specs: List[Dict[str, Any]],
    *,
    config: Dict[str, Any],
    allow_input: bool,
    editing: bool,
    image_reference: Optional[str],
    output_dir: Path,
    collage_name: str,
    tile_size: int,
    margin: int,
    background_color: str,
    panel_count: int,
) -> Tuple[Path, Path, List[PanelResult]]:
    replicate_settings = config.get("providers", {}).get("replicate", {})
    poll_interval = replicate_settings.get("poll_interval_seconds", 2.5)
    request_timeout = replicate_settings.get("request_timeout_seconds", 600.0)
    api_url = replicate_settings.get("api_base_url", DEFAULT_REPLICATE_API_URL)
    env_path_setting = config.get("env_file")
    env_path = Path(env_path_setting) if env_path_setting else (Path(config.get("_config_path", ".")).parent / ".env")
    load_env_file(env_path)
    token_env = replicate_settings.get("api_token_env", "REPLICATE_API_TOKEN")

    token = os.environ.get(token_env)
    if not token:
        raise GenerationError(f"Environment variable {token_env} is required for Replicate access")

    client_params = {
        "api_token": token,
        "api_url": api_url,
        "poll_interval": poll_interval,
        "request_timeout": request_timeout,
    }

    output_dir = output_dir.resolve()
    ensure_output_directory(output_dir)

    replacements = {"prompt": prompt}
    panel_results: List[PanelResult] = []
    prompt_text = prompt

    LOGGER.info("Using prompt: %s", prompt)
    LOGGER.info(
        "Selected models: %s",
        ", ".join(spec.get("name", spec.get("id", "model")) for spec in model_specs),
    )

    editing_enabled = editing and all(model_supports_image_input(spec) for spec in model_specs)

    # Always run in parallel - we collect placeholders for each model first, then run them concurrently
    max_workers = min(4, len(model_specs)) or 1
    futures = []
    
    # Pre-collect any required placeholder values if interactive input is enabled
    prepared_specs = []
    for index, spec in enumerate(model_specs, start=1):
        spec_copy = copy.deepcopy(spec)
        preset_values = (
            build_placeholder_presets(spec, image_reference) if editing_enabled else None
        )
        
        # If we need interactive input, collect placeholders now (sequentially)
        if allow_input and spec_copy.get("placeholders"):
            print(f"\n=== {spec_copy.get('name', f'Model {index}')} ===")
            # Apply placeholders with input collection
            try:
                apply_placeholders(
                    spec_copy,
                    allow_input=True,
                    client=None,  # We'll create client in generate_panel
                    enable_image_placeholders=editing_enabled,
                    preset_values=preset_values,
                )
            except ValueError as exc:
                raise GenerationError(f"Failed to collect inputs for {spec_copy.get('name', f'Model {index}')}: {exc}")
        
        prepared_specs.append((index, spec_copy, preset_values))
    
    # Now run all models in parallel with their pre-collected inputs
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, spec_copy, preset_values in prepared_specs:
            futures.append(
                executor.submit(
                    generate_panel,
                    index,
                    spec_copy,
                    replacements,
                    output_dir,
                    client_params,
                    False,  # allow_input: No more input needed - we pre-collected everything
                    editing_enabled,
                    preset_values,
                    request_timeout,
                )
            )
        try:
            for future in concurrent.futures.as_completed(futures):
                panel_results.append(future.result())
        except Exception as err:
            raise GenerationError(str(err)) from err

    panel_results.sort(key=lambda panel: panel.order)
    if len(panel_results) < panel_count:
        fill_color = parse_hex_color(background_color)
        filler_base = max(64, tile_size or 1024)
        for order in range(len(panel_results) + 1, panel_count + 1):
            blank_path = output_dir / f"{order:02d}-blank.png"
            if not blank_path.exists():
                Image.new("RGB", (filler_base, filler_base), fill_color).save(blank_path)
            panel_results.append(
                PanelResult(
                    order=order,
                    model_name=f"Empty Panel {order}",
                    image_path=blank_path,
                    width=filler_base,
                    height=filler_base,
                    is_placeholder=True,
                )
            )
        panel_results.sort(key=lambda panel: panel.order)

    # Randomize panel arrangement while preserving original order info
    # This ensures both labeled and unlabeled versions have identical random layout
    randomized_panels = panel_results.copy()
    # Use a seed based on the prompt to make randomization reproducible for the same prompt
    random.seed(hash(prompt) % (2**32))
    random.shuffle(randomized_panels)
    # Reset seed to avoid affecting other random operations
    random.seed()
    
    # Reassign order numbers to match the randomized positions
    for new_order, panel in enumerate(randomized_panels, start=1):
        panel.order = new_order
    
    LOGGER.info("Randomized panel arrangement (1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right): %s", 
                ", ".join(f"{panel.model_name} -> position {panel.order}" for panel in randomized_panels))

    # Verify panel integrity before creating collages
    arrangement_verification = [(panel.order, panel.model_name, panel.image_path.name) for panel in randomized_panels]
    LOGGER.info("Final arrangement verification: %s", arrangement_verification)
    
    # Create labeled version with randomized arrangement
    collage_path = output_dir / collage_name
    compose_grid(randomized_panels, collage_path, tile_size, margin, background_color, prompt_text)
    LOGGER.info("Four-panel image created at %s", collage_path)
    
    # Create numbered version (1, 2, 3, 4 based on position)
    # Using the exact same randomized_panels list to guarantee identical layout
    name_parts = collage_name.rsplit('.', 1)
    if len(name_parts) == 2:
        numbered_name = f"{name_parts[0]}-numbered.{name_parts[1]}"
    else:
        numbered_name = f"{collage_name}-numbered"
    numbered_path = output_dir / numbered_name
    compose_grid_numbered(randomized_panels, numbered_path, tile_size, margin, background_color, prompt_text)
    LOGGER.info("Numbered four-panel image created at %s", numbered_path)
    
    # Final verification log
    LOGGER.info("Both collages use identical panel arrangement - labels correctly match their respective images")
    
    return collage_path, numbered_path, randomized_panels


LAST_RESULTS: List[Dict[str, str]] = []


FORM_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Four Panel Generator</title>
  <style>
    .prior-panels { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
    .prior-panel { border: 1px solid #444; padding: 0.5rem; border-radius: 6px; background: #1e1e1e; max-width: 180px; text-align: center; }
    .prior-panel img { width: 100%; height: auto; border-radius: 4px; margin-bottom: 0.25rem; }
    .prior-panel input { margin-bottom: 0.25rem; }
    body { font-family: Arial, sans-serif; margin: 2rem; background: #121212; color: #f1f1f1; }
    h1 { margin-bottom: 1rem; }
    form { margin-bottom: 2rem; }
    textarea, input[type=text] { width: 100%; box-sizing: border-box; padding: 0.5rem; }
    .models { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.5rem; }
    .note { font-size: 0.9rem; color: #bbbbbb; margin: 0.25rem 0 0.5rem; }
    .model-box { border: 1px solid #444; padding: 0.5rem; border-radius: 6px; background: #1e1e1e; }
    .result img { max-width: 100%; height: auto; border: 1px solid #333; }
    .error { color: #ff6b6b; margin-bottom: 1rem; }
    .success { color: #4fd1c5; margin-bottom: 1rem; }
    label { display: block; margin: 0.25rem 0; }
  </style>
</head>
<body>
  <h1>Four Panel Generator</h1>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  {% if collage_url %}
    <div class="success">Collages created.</div>
    <div class="result">
      <h3>Labeled Version (with model names)</h3>
      <img src="{{ collage_url }}" alt="Labeled collage preview">
      {% if unlabeled_url %}
      <h3>Unlabeled Version (for blind evaluation)</h3>
      <img src="{{ unlabeled_url }}" alt="Unlabeled collage preview">
      {% endif %}
    </div>
  {% endif %}
  <form method="post" enctype="multipart/form-data">
    <label for="prompt">Prompt</label>
    <textarea id="prompt" name="prompt" rows="3" required>{{ prompt }}</textarea>

    <label>
      <input type="checkbox" name="edit" value="1" {% if editing %}checked{% endif %}>
      Enable image editing
    </label>

    <label for="image_url">Image URL (for editing)</label>
    <input type="text" id="image_url" name="image_url" value="{{ image_url }}" placeholder="https://...">

    <label for="image_file">Upload image file (for editing)</label>
    <input type="file" id="image_file" name="image_file" accept="image/*">

    {% if last_panels %}
    <h2>Reuse a previous panel</h2>
    <div class="prior-panels">
      {% for panel in last_panels %}
        <label class="prior-panel">
          <input type="radio" name="previous_panel" value="{{ panel.path }}" {% if panel.selected %}checked{% endif %}>
          <img src="{{ panel.url }}" alt="{{ panel.name }}">
          <div>{{ panel.name }}</div>
        </label>
      {% endfor %}
    </div>
    {% endif %}

    <h2>Models ({{ model_selection_hint }})</h2>
    {% if panel_gap_note %}<div class="note">{{ panel_gap_note }}</div>{% endif %}
    <div class="models">
      {% for model in models %}
        <label class="model-box">
          <input type="checkbox" name="models" value="{{ model.id }}" {% if model.id in selected_ids %}checked{% endif %} {% if editing and not model.supports_image_input %}disabled{% endif %}>
          <strong>{{ model.name }}</strong>
          {% if model.supports_image_input %}<span>(IMG)</span>{% endif %}
          {% if model.description %}<div>{{ model.description }}</div>{% endif %}
        </label>
      {% endfor %}
    </div>

    <button type="submit" style="margin-top: 1rem; padding: 0.6rem 1.2rem;">Generate</button>
  </form>
</body>
</html>
"""


def launch_ui(config_path: Path, host: str = "127.0.0.1", port: int = 5000, verbose: bool = False) -> None:
    from flask import Flask, render_template_string, request, url_for, send_from_directory
    from werkzeug.utils import secure_filename

    configure_logging(verbose)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    def _load_config() -> Dict[str, Any]:
        cfg = load_config(config_path)
        cfg["_config_path"] = str(config_path.resolve())
        return cfg

    def _panel_defaults(cfg: Dict[str, Any]) -> Tuple[int, List[str]]:
        defaults = cfg.get("panel_defaults", {})
        count = defaults.get("count", 4)
        selection = (defaults.get("selection", []) or [])[:count]
        return count, selection

    @app.route("/outputs/<path:filename>")
    def serve_output(filename: str):
        output_dir = Path(app.config["OUTPUT_DIR"])
        return send_from_directory(output_dir, filename, as_attachment=False)

    @app.route("/", methods=["GET", "POST"])
    def index():
        global LAST_RESULTS
        cfg = _load_config()
        panel_count, default_selection = _panel_defaults(cfg)
        catalog_entries = cfg.get("catalog", [])
        models = [
            {
                "id": entry.get("id"),
                "name": entry.get("name", entry.get("id", "model")),
                "description": entry.get("description"),
                "supports_image_input": bool(entry.get("supports_image_input")),
            }
            for entry in catalog_entries
        ]

        editing_capable_ids = [
            entry.get("id")
            for entry in catalog_entries
            if bool(entry.get("supports_image_input"))
        ]
        editing_capable_set = {model_id for model_id in editing_capable_ids if model_id}
        max_editing_slots = min(panel_count, len(editing_capable_set))

        prompt = request.form.get("prompt", "") if request.method == "POST" else ""
        editing = bool(request.form.get("edit")) if request.method == "POST" else False
        selected_ids = request.form.getlist("models") if request.method == "POST" else list(default_selection)
        image_url_value = request.form.get("image_url", "") if request.method == "POST" else ""
        previous_panel = request.form.get("previous_panel") if request.method == "POST" else None
        error = None
        collage_url = None
        unlabeled_url = None

        if previous_panel:
            editing = True

        if editing:
            selected_ids = [model_id for model_id in selected_ids if model_id in editing_capable_set]

        if request.method == "POST":
            prompt = prompt.strip()
            catalog_by_id = {entry.get("id"): entry for entry in catalog_entries}
            submission_debug = f"editing={editing} selected={selected_ids} max_editing={max_editing_slots} panel_count={panel_count}"
            print(f"[four-panel] UI submission -> {submission_debug}", flush=True)
            LOGGER.debug("UI submission -> %s", submission_debug)
            if not prompt:
                error = "Prompt is required."
            elif editing:
                if max_editing_slots == 0:
                    error = "No models available that support image editing."
                elif len(selected_ids) < 1:
                    error = "Select at least one model for editing."
                elif len(selected_ids) > max_editing_slots:
                    error = f"You can choose at most {max_editing_slots} models for editing."
            elif len(selected_ids) < 1:
                error = "Select at least one model."
            elif len(selected_ids) > panel_count:
                error = f"You can choose at most {panel_count} models."
            if error:
                LOGGER.warning("UI validation error: %s", error)
            if not error:
                try:
                    selected_specs = [catalog_by_id[model_id] for model_id in selected_ids]
                except KeyError:
                    error = "One or more selected models are unknown."
                else:
                    if editing:
                        unsupported = [
                            spec.get("id")
                            for spec in selected_specs
                            if not bool(spec.get("supports_image_input"))
                        ]
                        if unsupported:
                            error = "One or more selected models do not support image editing."
                    image_reference = previous_panel.strip() if previous_panel else None
                    if not error and editing and not image_reference:
                        file_obj = request.files.get("image_file")
                        if file_obj and file_obj.filename:
                            uploads_dir = Path(app.config["UPLOAD_DIR"])
                            uploads_dir.mkdir(parents=True, exist_ok=True)
                            filename = secure_filename(file_obj.filename) or "upload.png"
                            upload_path = uploads_dir / filename
                            counter = 1
                            while upload_path.exists():
                                upload_path = uploads_dir / f"{upload_path.stem}_{counter}{upload_path.suffix}"
                                counter += 1
                            file_obj.save(upload_path)
                            image_reference = str(upload_path)
                        if not image_reference:
                            image_reference = image_url_value.strip()
                        if not image_reference:
                            error = "An image URL or file is required when editing."
                    if not error:
                        cfg["_config_path"] = str(config_path.resolve())
                        base_slug = slugify(prompt)
                        collage_name = f"{('edit-' if editing else '')}{base_slug or 'collage'}-four-panel.png"
                        try:
                            collage_path, unlabeled_path, panel_results = generate_collage(
                                prompt,
                                selected_specs,
                                config=cfg,
                                allow_input=False,
                                editing=editing,
                                image_reference=image_reference,
                                output_dir=Path(app.config["OUTPUT_DIR"]),
                                collage_name=collage_name,
                                tile_size=cfg.get("composition", {}).get("tile_size", 1024),
                                margin=cfg.get("composition", {}).get("margin", 24),
                                background_color=cfg.get("composition", {}).get("background_color", "#111111"),
                                panel_count=panel_count,
                            )
                            collage_url = url_for("serve_output", filename=collage_path.name)
                            unlabeled_url = url_for("serve_output", filename=unlabeled_path.name)
                            LAST_RESULTS = [
                                {
                                    "name": panel.model_name,
                                    "path": str(panel.image_path),
                                    "url": url_for("serve_output", filename=Path(panel.image_path).name),
                                }
                                for panel in panel_results
                                if not getattr(panel, "is_placeholder", False)
                            ]
                        except GenerationError as exc:
                            error = str(exc)

        if editing:
            if max_editing_slots == 0:
                model_selection_hint = "no editing-capable models available"
            else:
                model_selection_hint = f"choose up to {max_editing_slots}"
        else:
            model_selection_hint = f"choose up to {panel_count}"
        panel_gap_note = ""
        if editing:
            if max_editing_slots == 0:
                panel_gap_note = "No models in this catalog currently support editing."
            elif max_editing_slots < panel_count:
                remaining_slots = panel_count - max_editing_slots
                panel_gap_note = (
                    f"Only {max_editing_slots} editing-capable model{'s' if max_editing_slots != 1 else ''} available; "
                    f"the remaining {remaining_slots} panel{'s' if remaining_slots != 1 else ''} will use the background color."
                )
        if len(selected_ids) < panel_count:
            remaining_slots = panel_count - len(selected_ids)
            additional_note = (
                f"Any of the remaining {remaining_slots} panel{'s' if remaining_slots != 1 else ''} will stay blank."
            )
            panel_gap_note = f"{panel_gap_note} {additional_note}".strip() if panel_gap_note else additional_note

        if error and "Please select exactly" in error:
            error = "Select between 1 and {panel_count} models.".format(panel_count=panel_count)
        if error:
            print(f"[four-panel] UI error -> {error}", flush=True)
            LOGGER.warning("UI error surfaced to user: %s", error)
        last_panels = [
            {
                **item,
                "selected": bool(previous_panel and item["path"] == previous_panel),
            }
            for item in LAST_RESULTS
        ]
        return render_template_string(
            FORM_TEMPLATE,
            prompt=prompt,
            editing=editing,
            panel_count=panel_count,
            models=models,
            selected_ids=selected_ids,
            collage_url=collage_url,
            unlabeled_url=unlabeled_url,
            error=error,
            image_url=image_url_value,
            model_selection_hint=model_selection_hint,
            panel_gap_note=panel_gap_note,
            last_panels=last_panels,
        )

    cfg = _load_config()
    output_dir = Path(cfg.get("composition", {}).get("output_dir", "outputs")).resolve()
    app.config["OUTPUT_DIR"] = str(output_dir)
    app.config["UPLOAD_DIR"] = str((output_dir / "uploads").resolve())
    app.run(host=host, port=port, debug=False)

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-model collage from a shared prompt",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Shared text prompt for every model. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--config",
        default="model_config.json",
        type=Path,
        help="Path to the configuration JSON file (default: model_config.json)",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated catalog IDs to run (skips interactive selection)",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs"),
        type=Path,
        help="Directory where individual renders and the collage should be stored",
    )
    parser.add_argument(
        "--collage-name",
        default=None,
        help="Optional name for the final collage file (default: derived from prompt)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Override tile size in pixels when composing the grid",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=None,
        help="Override margin between tiles in pixels",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Enable editing workflow (filters models to those with image inputs)",
    )
    parser.add_argument(
        "--image",
        help="Image URL or local file path to use when editing",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the local web interface instead of the CLI workflow",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the web interface (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web interface (default: 5000)",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="Fail instead of prompting for interactive input",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger("four_panel").setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if getattr(args, "ui", False):
        launch_ui(
            args.config,
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 5000),
            verbose=args.verbose,
        )
        return 0

    config_path = args.config
    if not config_path.exists():
        sample_candidates = [
            config_path.with_suffix('.example.json'),
            config_path.parent / f"{config_path.stem}.example.json",
            Path('model_config.example.json'),
        ]
        for sample_path in sample_candidates:
            if sample_path.exists():
                try:
                    shutil.copy(sample_path, config_path)
                    LOGGER.info('Config file not found; copied sample from %s', sample_path)
                except OSError as err:
                    LOGGER.error('Failed to copy sample config from %s: %s', sample_path, err)
                    return 1
                break
        if not config_path.exists():
            LOGGER.error('Config file not found: %s', config_path)
            return 1

    config = load_config(config_path)
    config["_config_path"] = str(config_path.resolve())
    composition_cfg = config.get("composition", {})
    tile_size = args.tile_size or composition_cfg.get("tile_size", 1024)
    margin = args.margin or composition_cfg.get("margin", 24)
    background_color = composition_cfg.get("background_color", "#111111")

    catalog = config.get("catalog", [])
    if not catalog:
        LOGGER.error("Configuration catalog is empty; add at least one model entry")
        return 1

    panel_defaults = config.get("panel_defaults", {})
    panel_count = panel_defaults.get("count", 4)
    default_selection = panel_defaults.get("selection", [])

    # Allow input for model placeholders (API keys, image URLs) but not for model selection
    allow_input = not args.no_input
    try:
        prompt = gather_prompt(args.prompt, allow_input)
    except ValueError as err:
        LOGGER.error(err)
        return 1

    editing = bool(getattr(args, 'edit', False) or getattr(args, 'image', None))
    image_reference = getattr(args, 'image', None)
    if not editing and allow_input:
        resp = input("Do you want to provide a source image for editing? [y/N]: ").strip().lower()
        if resp in {"y", "yes"}:
            editing = True
    if editing and not image_reference:
        if allow_input:
            while True:
                image_reference = input("Enter image URL or local file path: ").strip()
                if image_reference:
                    break
                print("Image reference is required for editing.")
        else:
            LOGGER.error("Image reference required when editing is enabled")
            return 1

    if editing:
        catalog = [entry for entry in catalog if model_supports_image_input(entry)]
        if not catalog:
            LOGGER.error("No models available that support image editing.")
            return 1
        default_selection = [model_id for model_id in default_selection if any(entry.get('id') == model_id for entry in catalog)]

    try:
        # Always use the default 4 models without interactive selection
        # For Qwen, choose between text-to-image and edit-plus based on whether images are provided
        qwen_model = "qwen_image" if not image_reference else "qwen_edit_plus"
        forced_models = f"nano_latest,seedream_latest,{qwen_model},gpt_image_1"
        model_specs = choose_models(
            catalog,
            default_selection,
            panel_count,
            args.models or forced_models,
            allow_input=False,  # Never allow interactive input for model selection
            allow_partial=True,
            min_required=1,
        )
    except ValueError as err:
        LOGGER.error(err)
        return 1

    image_support_flags = [model_supports_image_input(spec) for spec in model_specs]
    if editing and not all(image_support_flags):
        LOGGER.error("Selected models do not all support image editing.")
        return 1

    default_collage_name = f"{slugify(prompt)}-four-panel.png"
    if editing and not args.collage_name:
        default_collage_name = f"edit-{slugify(prompt)}-four-panel.png"
    collage_name = args.collage_name or default_collage_name

    try:
        collage_path, unlabeled_path, _ = generate_collage(
            prompt,
            model_specs,
            config=config,
            allow_input=allow_input,  # Keep allow_input for placeholder collection
            editing=editing,
            image_reference=image_reference,
            output_dir=args.output_dir,
            collage_name=collage_name,
            tile_size=tile_size,
            margin=margin,
            background_color=background_color,
            panel_count=panel_count,
        )
        print(f"Generated collages:")
        print(f"  - Labeled: {collage_path}")
        print(f"  - Unlabeled: {unlabeled_path}")
    except GenerationError as err:
        LOGGER.error(err)
        return 1


    return 0


def slugify(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif safe and safe[-1] != '-':
            safe.append('-')
    slug = ''.join(safe).strip('-')
    if not slug:
        slug = "panel"
    return slug[:80]


if __name__ == "__main__":
    sys.exit(main())



















