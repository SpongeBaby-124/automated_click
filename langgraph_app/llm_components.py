"""LLM-backed components used by the automation graph."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from langchain_core.messages import AIMessage
from openai import OpenAI


def _extract_json_from_response(text: str) -> dict | None:
    """Extract JSON from raw LLM output."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    import re

    json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    json_object_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_object_pattern, text, re.DOTALL)
    for candidate in matches:
        try:
            parsed = json.loads(candidate)
            if "next" in parsed or "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


class GoalVerifier:
    """Vision-language reviewer that evaluates tool outcomes."""

    def __init__(self) -> None:
        base_url = os.environ.get("OPENAI_API_BASE")
        api_key = os.environ.get("OPENAI_API_KEY")
        self._vision_model = os.environ.get("VISION_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

        if not base_url or not api_key:
            raise EnvironmentError("必须设置 OPENAI_API_BASE 和 OPENAI_API_KEY 环境变量")

        self._client = OpenAI(base_url=base_url, api_key=api_key)

    async def evaluate(self, **kwargs: Any) -> dict:
        view = None
        tool_result = kwargs.get("tool_result")
        if isinstance(tool_result, dict):
            view = tool_result.get("current_view")

        if not view or "screenshot_base64" not in view:
            raise ValueError("缺少用于审查的网页截图")

        prompt = kwargs.pop("prompt_override", None) or self._compose_prompt(**kwargs)
        raw_response = await asyncio.to_thread(
            self._call_model,
            prompt,
            view["screenshot_base64"],
        )

        parsed = _extract_json_from_response(raw_response)
        if not parsed:
            return {
                "completed": False,
                "reason": "审查模型返回无法解析",
                "should_continue": True,
                "pending_form_fields": kwargs.get("pending_form_fields") or [],
                "missing_actions": ["无法解析审查结果，建议重新获取页面状态"],
                "next_hint": "等待重新规划",
                "confidence": 0.0,
                "raw_response": raw_response,
                "status": "error",
            }

        parsed["raw_response"] = raw_response
        parsed.setdefault("status", "ok")
        return parsed

    def _compose_prompt(self, **kwargs: Any) -> str:
        user_goal = kwargs.get("user_goal", "")
        last_action = kwargs.get("last_action", "")
        action_params = kwargs.get("action_params", {})
        tool_result = kwargs.get("tool_result") or {}
        pending_fields = kwargs.get("pending_form_fields") or []
        pending_fields_str = ", ".join(pending_fields) or "无"

        result_summary = {}
        for key, value in tool_result.items():
            if key == "current_view":
                continue
            result_summary[key] = value

        prompt = f"""
你是一个网页自动化执行的审查员，需要判断当前动作是否让任务更接近目标，任务是否已经完成，以及接下来建议执行的动作。

用户目标：{user_goal}
最近一次执行动作：{last_action}
动作参数：{json.dumps(action_params, ensure_ascii=False)}
工具返回摘要：{json.dumps(result_summary, ensure_ascii=False)}
已有未完成的表单字段：{pending_fields_str}

请基于截图进行审查，并严格返回 JSON，包含以下键：
{{
  "completed": bool,
  "reason": "string",
  "should_continue": bool,
  "pending_form_fields": ["string", ...],
  "missing_actions": ["string", ...],
  "next_hint": "string",
  "confidence": float
}}

如果你无法判断是否完成，请将 completed 设为 false，并在 reason 中说明原因。
当表单仍需填写时，请在 pending_form_fields 中按顺序列出需要填写的字段描述。
        """.strip()
        return prompt

    def _call_model(self, prompt: str, screenshot_base64: str) -> str:
        response = self._client.chat.completions.create(
            model=self._vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=400,
            temperature=0.1,
        )

        return response.choices[0].message.content or ""


class VisionPlanner:
    """Vision-language planner that chooses the next action."""

    def __init__(self) -> None:
        base_url = os.environ.get("OPENAI_API_BASE")
        api_key = os.environ.get("OPENAI_API_KEY")
        self._vision_model = os.environ.get("VISION_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

        if not base_url or not api_key:
            raise EnvironmentError("必须设置 OPENAI_API_BASE 和 OPENAI_API_KEY 环境变量")

        self._client = OpenAI(base_url=base_url, api_key=api_key)

    async def plan(self, *, prompt: str, screenshot_base64: str) -> dict:
        raw_response = await asyncio.to_thread(
            self._call_model,
            prompt,
            screenshot_base64,
        )
        return {"raw_response": raw_response}

    def _call_model(self, prompt: str, screenshot_base64: str) -> str:
        response = self._client.chat.completions.create(
            model=self._vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=400,
            temperature=0.1,
        )

        return response.choices[0].message.content or ""


__all__ = [
    "GoalVerifier",
    "VisionPlanner",
    "_extract_json_from_response",
    "AIMessage",
]
