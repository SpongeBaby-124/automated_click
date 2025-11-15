"""State tracking utilities for visual web automation."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from PIL import Image

_HASH_SIZE = 8
_NO_CHANGE_THRESHOLD = 0.08  # 8% hash difference


class FailureType(str, Enum):
    """Failure categories that help the planner adjust its strategy."""

    LOGICAL = "logical"
    TRANSIENT = "transient"
    VISUAL_STALE = "visual_stale"
    LOOP = "loop"
    UNCLASSIFIED = "unclassified"


@dataclass
class ViewComparison:
    """Result of comparing two views."""

    changed: bool
    similarity: float
    hash_equal: bool
    reason: str
    distance: float


def _average_hash(image: Image.Image, hash_size: int = _HASH_SIZE) -> str:
    gray = image.convert("L").resize((hash_size, hash_size))
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ["1" if pix >= avg else "0" for pix in pixels]
    bitstring = "".join(bits)
    return f"{int(bitstring, 2):0{hash_size * hash_size // 4}x}"


def _hamming_distance(hash_a: str | None, hash_b: str | None) -> Optional[int]:
    if not hash_a or not hash_b:
        return None
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def build_view_payload(label: str, screenshot_bytes: bytes, page_url: str) -> Dict[str, object]:
    """Construct a normalized view payload enriched with image metadata."""

    screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    with Image.open(io.BytesIO(screenshot_bytes)) as img:
        width, height = img.size
        avg_hash = _average_hash(img)

    payload = {
        "label": label,
        "screenshot_base64": screenshot_base64,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": page_url,
            "sha1": hashlib.sha1(screenshot_bytes).hexdigest(),
            "ahash": avg_hash,
            "hash_size": _HASH_SIZE,
            "width": width,
            "height": height,
        },
    }
    return payload


def compare_views(prev_view: Dict[str, object] | None, current_view: Dict[str, object] | None) -> ViewComparison:
    """Compare two screenshots and determine if the page truly changed."""

    if not prev_view or not current_view:
        return ViewComparison(True, 0.0, False, "缺少比较视图", 1.0)

    prev_meta = prev_view.get("meta", {})
    curr_meta = current_view.get("meta", {})

    hash_equal = prev_meta.get("sha1") == curr_meta.get("sha1")
    distance = _hamming_distance(prev_meta.get("ahash"), curr_meta.get("ahash"))

    if distance is None:
        return ViewComparison(True, 0.0, hash_equal, "缺少哈希信息", 1.0)

    max_bits = _HASH_SIZE * _HASH_SIZE
    normalized = distance / max_bits
    changed = normalized >= _NO_CHANGE_THRESHOLD
    similarity = max(0.0, 1.0 - normalized)

    if hash_equal:
        return ViewComparison(False, 1.0, True, "截图完全一致", 0.0)

    reason = "视觉变化显著" if changed else "视觉变化不足"
    return ViewComparison(changed, similarity, False, reason, normalized)


def classify_failure(message: str, comparison: ViewComparison | None = None) -> FailureType:
    """Rudimentary heuristic to separate logical vs transient failures."""

    normalized_msg = (message or "").lower()

    if comparison and not comparison.changed:
        return FailureType.VISUAL_STALE

    logical_keywords = ["missing", "不存在", "未找到", "缺少", "无效", "not found", "invalid"]
    transient_keywords = ["超时", "timeout", "网络", "失败", "exception", "异常", "retry"]

    if any(keyword in normalized_msg for keyword in logical_keywords):
        return FailureType.LOGICAL

    if any(keyword in normalized_msg for keyword in transient_keywords):
        return FailureType.TRANSIENT

    return FailureType.UNCLASSIFIED


def should_force_correction(failure_type: FailureType | None) -> bool:
    return failure_type in {FailureType.LOGICAL, FailureType.VISUAL_STALE, FailureType.LOOP}


def update_history(
    history: List[Dict[str, object]],
    *,
    view_hash: str | None,
    action_type: str,
    step_description: str,
    view_meta: Dict[str, object] | None,
    limit: int = 5,
) -> List[Dict[str, object]]:
    """Record the latest action and trim history."""

    if not view_hash:
        return history

    entry = {
        "view_hash": view_hash,
        "action_type": action_type,
        "step": step_description,
        "timestamp": view_meta.get("timestamp") if view_meta else None,
        "url": view_meta.get("url") if view_meta else None,
    }

    updated = history[-(limit - 1) :] if limit > 1 else []
    updated.append(entry)
    return updated


def detect_visual_loop(history: List[Dict[str, object]], current_hash: str | None, window: int = 4) -> Optional[Dict[str, object]]:
    if not current_hash or not history:
        return None

    recent = history[-window:]
    for idx, entry in enumerate(recent):
        if entry.get("view_hash") == current_hash:
            return {
                "loop_detected": True,
                "repeat_step": entry.get("step"),
                "message": "检测到视觉状态重复，可能陷入循环",
                "history_index": len(history) - len(recent) + idx,
            }
    return None


def format_history_for_prompt(history: List[Dict[str, object]]) -> str:
    if not history:
        return "无历史记录"

    lines = []
    for entry in history[-5:]:
        timestamp = entry.get("timestamp") or "最近"
        action = entry.get("action_type")
        url = entry.get("url") or "unknown"
        view_hash = (entry.get("view_hash") or "-")[:8]
        lines.append(f"- {timestamp}: [{url}] {action} (视图 {view_hash})")
    return "\n".join(lines)
