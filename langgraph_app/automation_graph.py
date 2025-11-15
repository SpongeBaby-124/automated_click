"""Web 自动化 LangGraph - Agent 与 Tools 节点的编排（增强版）"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .llm_components import GoalVerifier, VisionPlanner, _extract_json_from_response
from .state_utils import (
    FailureType,
    classify_failure,
    compare_views,
    detect_visual_loop,
    format_history_for_prompt,
    should_force_correction,
    update_history,
)
from .vision_tool import VisionClickTool


class AutomationState(TypedDict, total=False):
    """自动化任务的状态定义"""

    messages: Annotated[list[BaseMessage], add_messages]
    user_goal: str
    current_step: str
    action_type: str
    action_params: dict
    decision: Literal["tools", "end"]
    tool_result: dict
    attempt_count: int
    agent_view: dict
    verification: dict
    pending_form_fields: list[str]
    task_history: list[dict]
    last_failure: dict
    correction_required: bool
    last_comparison: dict
    recent_views: list[dict]
    loop_alert: dict | None


def _agent_node(tool: VisionClickTool):
    """构造 Agent 节点，使用视觉模型进行规划"""

    planner = VisionPlanner()
    _last_plan_cache = {}

    async def node(state: AutomationState) -> AutomationState:
        try:
            tool_result = state.get("tool_result")
            verification = state.get("verification") or {}
            pending_fields = state.get("pending_form_fields") or []
            current_view = None
            last_failure = state.get("last_failure")
            correction_required = state.get("correction_required", False)

            if isinstance(tool_result, dict):
                current_view = tool_result.get("current_view")

            if not current_view:
                current_view = await tool.get_view("agent_plan")

            view_hash = (current_view.get("meta", {}) or {}).get("sha1", "")
            cache_key = f"{view_hash}_{correction_required}"

            if (
                not correction_required
                and cache_key in _last_plan_cache
                and not last_failure
                and state.get("attempt_count", 0) == 0
            ):
                cached = _last_plan_cache[cache_key]
                print("⚡ 使用缓存的规划决策")
                return cached

            prompt = _build_planner_prompt(
                user_goal=state.get("user_goal", ""),
                tool_feedback=_clean_tool_feedback(tool_result),
                attempt_count=state.get("attempt_count", 0),
                verification=verification,
                pending_fields=pending_fields,
                history=format_history_for_prompt(state.get("recent_views", [])),
                last_failure=state.get("last_failure"),
                correction_required=state.get("correction_required", False),
                loop_alert=state.get("loop_alert"),
                comparison=state.get("last_comparison"),
            )

            plan_response = await planner.plan(
                prompt=prompt,
                screenshot_base64=current_view["screenshot_base64"],
            )

            raw_content = plan_response.get("raw_response", "")
            print(f"\n🤔 VL 规划原始输出:\n{raw_content}\n")

            parsed = _extract_json_from_response(raw_content)

            if not parsed:
                print("⚠️ 无法解析 JSON，任务结束")
                return {
                    "current_step": "解析失败，任务结束",
                    "action_type": "finish",
                    "action_params": {},
                    "decision": "end",
                    "agent_view": current_view,
                    "messages": [AIMessage(content="规划器返回格式错误，任务终止。")],
                }

            decision = parsed.get("next", "end").lower()
            if decision not in {"tools", "end"}:
                decision = "end"

            current_step = parsed.get("current_step", "")
            action_type = parsed.get("action_type", "finish")
            action_params = parsed.get("action_params", {})
            reasoning = parsed.get("reasoning", "")

            verification_status = verification.get("status", "unknown")
            allow_end = True
            if verification_status in {"ok", "heuristic"}:
                required_conf = 0.6 if verification_status == "ok" else 0.5
                allow_end = bool(
                    verification.get("completed")
                    and verification.get("confidence", 0) >= required_conf
                )

            if decision == "end" and not allow_end:
                print("🛑 审查或启发式认为目标未完成，覆盖 Agent 结束决策")
                decision = "tools"
                if action_type == "finish":
                    action_type = "wait"
                    action_params = {"timeout": 1500}

            print(f"✅ 规划决策: {current_step}")
            print(f"   动作类型: {action_type}")
            print(f"   决策: {decision}")
            print(f"   推理: {reasoning}")

            history = list(state.get("task_history", []))
            history.append(
                {
                    "step": current_step,
                    "action_type": action_type,
                    "decision": decision,
                    "reasoning": reasoning,
                    "failure_type": (state.get("last_failure") or {}).get("type"),
                }
            )

            result_state = {
                "current_step": current_step,
                "action_type": action_type,
                "action_params": action_params,
                "decision": decision,
                "attempt_count": state.get("attempt_count", 0),
                "agent_view": current_view,
                "task_history": history,
                "messages": [AIMessage(content=f"{current_step}\n推理：{reasoning}")],
            }

            if not correction_required and not last_failure:
                _last_plan_cache[cache_key] = result_state

            return result_state

        except Exception as exc:  # noqa: BLE001
            print(f"❌ Agent 节点异常: {exc}")
            fallback_view = state.get("agent_view") or None
            if fallback_view is None:
                try:
                    fallback_view = await tool.get_view("agent_exception")
                except Exception:  # noqa: BLE001
                    fallback_view = None

            return {
                "current_step": "异常终止",
                "action_type": "finish",
                "action_params": {},
                "decision": "end",
                "agent_view": fallback_view,
                "messages": [AIMessage(content=f"规划异常：{exc}")],
            }

    return node


def _tools_node(tool: VisionClickTool, verifier: GoalVerifier):
    """构造 Tools 节点，负责执行具体动作"""

    async def node(state: AutomationState) -> AutomationState:
        action_type = (state.get("action_type") or "").lower()
        action_params = state.get("action_params", {})
        max_attempts = 5
        attempt = min(state.get("attempt_count", 0) + 1, max_attempts)
        prev_view = state.get("agent_view")

        print(f"\n🔧 Tools 节点执行: {action_type}")
        print(f"   参数: {action_params}")
        print(f"   尝试次数: 第 {attempt} 次 (最多 {max_attempts} 次)")

        try:
            if action_type == "navigate":
                url = action_params.get("url", "")
                timeout = action_params.get("timeout", 20000)
                if not url:
                    view = await tool.get_view("missing_url")
                    result = {
                        "success": False,
                        "message": "缺少 url 参数",
                        "current_view": view,
                    }
                else:
                    result = await tool.navigate_to(url, timeout)

            elif action_type == "click":
                element_desc = action_params.get("element_description", "")
                if not element_desc:
                    view = await tool.get_view("missing_element_desc")
                    result = {
                        "success": False,
                        "message": "缺少 element_description 参数",
                        "current_view": view,
                    }
                else:
                    result = await tool.click_element(element_desc)

            elif action_type == "type":
                text = action_params.get("text", "")
                delay = action_params.get("delay", 50)
                press_enter = action_params.get("press_enter", False)
                if not text:
                    view = await tool.get_view("missing_text")
                    result = {
                        "success": False,
                        "message": "缺少 text 参数",
                        "current_view": view,
                    }
                else:
                    result = await tool.type_text(text, delay, press_enter)

            elif action_type == "press_key":
                key = action_params.get("key", "")
                if not key:
                    view = await tool.get_view("missing_key")
                    result = {
                        "success": False,
                        "message": "缺少 key 参数",
                        "current_view": view,
                    }
                else:
                    result = await tool.press_key(key)

            elif action_type == "wait":
                timeout = action_params.get("timeout", 10000)
                result = await tool.wait_for_navigation(timeout)

            elif action_type == "scroll":
                direction = action_params.get("direction", "down")
                amount = action_params.get("amount", 600)
                result = await tool.scroll_page(direction, amount)

            elif action_type == "finish":
                view = await tool.get_view("finish_review")
                result = {
                    "success": True,
                    "message": "Agent 主动结束任务",
                    "current_view": view,
                }

            else:
                view = await tool.get_view("unknown_action")
                result = {
                    "success": False,
                    "message": f"未知的动作类型: {action_type}",
                    "current_view": view,
                }

        except Exception as exc:  # noqa: BLE001
            print(f"❌ 工具执行异常: {exc}")
            view = await tool.get_view("tool_exception")
            result = {
                "success": False,
                "message": f"工具执行异常: {exc}",
                "current_view": view,
            }
            return {
                "tool_result": {
                    **result,
                    "action_type": action_type,
                    "action_params": action_params,
                    "attempt": attempt,
                },
                "attempt_count": attempt,
                "messages": [AIMessage(content=result["message"])],
            }

        if "current_view" not in result:
            result["current_view"] = await tool.get_view("tools_fallback")

        current_view = result["current_view"]
        comparison = compare_views(prev_view, current_view)
        comparison_summary = {
            "changed": comparison.changed,
            "similarity": comparison.similarity,
            "hash_equal": comparison.hash_equal,
            "reason": comparison.reason,
            "distance": comparison.distance,
        }

        history = state.get("recent_views", []) or []
        view_hash = (current_view.get("meta", {}) or {}).get("sha1")
        loop_alert = detect_visual_loop(history, view_hash)
        history = update_history(
            history,
            view_hash=view_hash,
            action_type=action_type,
            step_description=state.get("current_step", ""),
            view_meta=current_view.get("meta"),
        )

        failure_type: FailureType | None = None

        if result.get("success") and not comparison.changed:
            result["success"] = False
            result["message"] = f"{result.get('message', '')} | 页面未发生明显变化"
            failure_type = FailureType.VISUAL_STALE

        if loop_alert:
            result["success"] = False
            result["message"] = f"{result.get('message', '')} | 检测到循环，需要改变策略"
            failure_type = FailureType.LOOP

        if not result.get("success") and failure_type is None:
            failure_type = classify_failure(result.get("message", ""), comparison)

        correction_required = should_force_correction(failure_type) or bool(loop_alert)

        result.setdefault("success", False)
        result.setdefault("message", "未提供执行结果")
        result.update(
            {
                "action_type": action_type,
                "action_params": action_params,
                "attempt": attempt,
                "failure_type": failure_type.value if isinstance(failure_type, FailureType) else None,
                "comparison": comparison_summary,
            }
        )

        if not result["success"] and attempt >= max_attempts and not correction_required:
            result["message"] += "（已达到最大重试次数）"

        verification_result: dict[str, object] = {
            "completed": False,
            "should_continue": True,
            "pending_form_fields": state.get("pending_form_fields", []) or [],
            "missing_actions": ["未执行审查"],
            "next_hint": "等待下一步规划",
            "reason": "尚未触发审查逻辑",
            "confidence": 0.0,
            "status": "skipped",
        }

        skip_verifier = (
            not result.get("success")
            or action_type in {"wait", "press_key"}
            or (action_type == "navigate" and result.get("success"))
        )

        if not skip_verifier:
            try:
                verification_result = await verifier.evaluate(
                    user_goal=state.get("user_goal", ""),
                    last_action=action_type,
                    action_params=action_params,
                    tool_result=result,
                    pending_form_fields=state.get("pending_form_fields", []) or [],
                )
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 审查失败: {exc}")
                verification_result = {
                    "completed": False,
                    "should_continue": True,
                    "pending_form_fields": state.get("pending_form_fields", []) or [],
                    "missing_actions": ["审查失败，建议重新截图后继续规划"],
                    "next_hint": "重新规划下一步操作",
                    "reason": f"审查异常: {exc}",
                    "confidence": 0.0,
                    "status": "error",
                }
        else:
            print("⏭️ 跳过审查（工具失败或简单操作）")

        verification_status = verification_result.get("status", "unknown")

        heuristic_match = _heuristic_goal_match(
            state.get("user_goal", ""),
            _extract_view_url(current_view, result),
        )
        result["heuristic_match"] = heuristic_match

        if heuristic_match.get("matched"):
            should_override = (
                not verification_result.get("completed")
                or verification_status in {"error", "skipped", "unknown"}
                or verification_result.get("confidence", 0) < 0.4
            )
            if should_override:
                verification_result.update(
                    {
                        "completed": True,
                        "should_continue": False,
                        "reason": heuristic_match.get("reason", "已匹配目标域"),
                        "missing_actions": [],
                        "next_hint": "任务目标已满足（启发式判定）",
                        "confidence": max(
                            verification_result.get("confidence", 0),
                            heuristic_match.get("confidence", 0.75),
                        ),
                        "status": "heuristic",
                    }
                )
                verification_status = "heuristic"

            verification_status = verification_result.get("status", verification_status)

        if verification_status == "ok":
            verified_success = bool(result["success"] and verification_result.get("completed"))
        elif verification_status == "heuristic":
            verified_success = bool(result["success"] and verification_result.get("completed"))
        else:
            verified_success = False

        result["verified_success"] = verified_success
        result["verification"] = verification_result

        if verification_status == "ok":
            suffix = "审查通过" if verified_success else "审查未通过"
        elif verification_status == "heuristic":
            suffix = "启发式判定完成" if verification_result.get("completed") else "启发式判定未完成"
        elif verification_status == "error":
            suffix = "审查跳过（模型异常）"
        else:
            suffix = "审查未启用"
        result["message"] = f"{result['message']} | {suffix}"

        print(f"✓ 工具执行结果: {result.get('message', '')}")

        raw_pending_fields = verification_result.get("pending_form_fields", []) or []
        pending_fields = [str(field) for field in raw_pending_fields]

        next_attempt = 0 if verified_success else (0 if correction_required else attempt)

        last_failure = None
        if not result["success"]:
            last_failure = {
                "type": result.get("failure_type"),
                "message": result.get("message"),
                "action": action_type,
                "attempt": attempt,
            }

        return {
            "tool_result": result,
            "attempt_count": next_attempt,
            "verification": verification_result,
            "pending_form_fields": pending_fields,
            "last_failure": last_failure,
            "correction_required": correction_required,
            "last_comparison": comparison_summary,
            "recent_views": history,
            "loop_alert": loop_alert,
            "messages": [
                AIMessage(
                    content=(
                        f"执行结果：{result.get('message', '')}\n"
                        f"审查意见：{verification_result.get('reason', '')}\n"
                        f"下一步提示：{verification_result.get('next_hint', '')}"
                    )
                )
            ],
        }

    return node


def build_automation_graph(tool: VisionClickTool):
    """构建自动化任务的 LangGraph"""

    verifier = GoalVerifier()
    graph = StateGraph(AutomationState)
    graph.add_node("agent", _agent_node(tool))
    graph.add_node("tools", _tools_node(tool, verifier))
    graph.set_entry_point("agent")

    def router(state: AutomationState) -> str:
        decision = state.get("decision", "end")
        if decision == "end":
            verification = state.get("verification") or {}
            status = verification.get("status", "unknown")
            if status in {"ok", "heuristic"}:
                required_conf = 0.6 if status == "ok" else 0.5
                if not (
                    verification.get("completed")
                    and verification.get("confidence", 0) >= required_conf
                ):
                    print("🔁 审查/启发式判定未完成，继续执行工具节点")
                    return "tools"
        return decision

    graph.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "end": END,
        },
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


_DOMAIN_PATTERN = re.compile(
    r"(?:https?://)?([a-z0-9.-]+\.(?:com|cn|net|org|io|gov|edu|top|vip|info|co|shop|xyz|tv|cc))",
    re.IGNORECASE,
)

_KNOWN_SITE_KEYWORDS = {
    "百度": "baidu.com",
    "谷歌": "google.com",
    "google": "google.com",
    "淘宝": "taobao.com",
    "京东": "jd.com",
    "拼多多": "pinduoduo.com",
    "抖音": "douyin.com",
    "知乎": "zhihu.com",
    "微信": "weixin.qq.com",
    "微博": "weibo.com",
    "b站": "bilibili.com",
    "哔哩": "bilibili.com",
    "小红书": "xiaohongshu.com",
    "苹果": "apple.com",
    "iphone": "apple.com",
    "apple": "apple.com",
    "youtube": "youtube.com",
    "twitter": "twitter.com",
    "推特": "twitter.com",
    "instagram": "instagram.com",
}


def _extract_view_url(view: dict | None, result: dict | None) -> str | None:
    if result and isinstance(result, dict):
        url = result.get("url")
        if url:
            return url
    if view and isinstance(view, dict):
        meta = view.get("meta") or {}
        url = meta.get("url")
        if url:
            return url
    return None


def _extract_domains_from_goal(goal: str) -> set[str]:
    goal = goal or ""
    domains = {match.group(1).lower() for match in _DOMAIN_PATTERN.finditer(goal)}
    lower_goal = goal.lower()
    for keyword, domain in _KNOWN_SITE_KEYWORDS.items():
        if keyword.lower() in lower_goal:
            domains.add(domain)
    return domains


def _heuristic_goal_match(user_goal: str, current_url: str | None) -> dict:
    payload = {
        "matched": False,
        "url": current_url,
        "reason": "缺少当前 URL" if not current_url else "",
        "expected_domains": [],
    }

    if not current_url:
        return payload

    parsed = urlparse(current_url)
    host = (parsed.hostname or "").lower()
    if not host:
        payload["reason"] = "当前 URL 缺少域名"
        return payload

    expected = _extract_domains_from_goal(user_goal)
    payload["expected_domains"] = sorted(expected)
    if not expected:
        payload["reason"] = "用户目标中未识别到目标域"
        return payload

    matched_domain = None
    for domain in expected:
        if domain and domain in host:
            matched_domain = domain
            break

    if matched_domain:
        payload.update(
            {
                "matched": True,
                "domain": matched_domain,
                "reason": f"当前域名 {host} 匹配目标 {matched_domain}",
                "confidence": 0.8,
            }
        )
    else:
        payload["reason"] = f"当前域名 {host} 未匹配目标域"

    return payload


def _build_planner_prompt(
    *,
    user_goal: str,
    tool_feedback: str,
    attempt_count: int,
    verification: dict,
    pending_fields: list[str],
    history: str,
    last_failure: dict | None,
    correction_required: bool,
    loop_alert: dict | None,
    comparison: dict | None,
) -> str:
    verification_feedback = json.dumps(verification or {}, ensure_ascii=False)
    pending_fields_str = ", ".join(pending_fields or []) or "无"
    failure_type = (last_failure or {}).get("type") or "无"
    failure_message = (last_failure or {}).get("message") or "无"
    loop_hint = (loop_alert or {}).get("message") or "无"
    comparison_text = _format_comparison(comparison)
    correction_hint = (
        "当前处于纠错模式，必须提供与上一动作为明显不同的新方案。" if correction_required else ""
    )

    prompt = f"""
你现在控制着一个具备视觉能力的网页自动化代理，目标是通过多步操作完成用户的需求。请仔细观察最新截图，再结合历史信息，制定不会重复错误的新计划。

用户目标：{user_goal}
最近工具反馈：{tool_feedback}
当前针对同一动作的尝试次数：{attempt_count} / 5
最新审查信息：{verification_feedback}
页面状态对比：{comparison_text}
最近失败类别：{failure_type}
失败说明：{failure_message}
循环提示：{loop_hint}
待填写的表单字段队列：{pending_fields_str}
历史轨迹：
{history}

关键规则：
1. 只有在确信任务目标已经完成且审查 completed=true 时，才能选择 action_type="finish" 并终止。
2. 当上一动作失败、页面没有发生变化，或 Loop 提示存在时，必须分析失败原因并规划与上一动作不同的新策略，禁止重复失败动作或参数。
3. 如在当前屏幕找不到目标元素，请优先考虑 scroll 动作（direction: up/down/left/right, amount: 像素）探索其他区域。
4. 在输入搜索词且需要提交时，请在 type 动作中设置 press_enter 为 true。
5. 填写表单时需按照 pending_form_fields 的顺序逐项填写，确认全部完成后再提交或登录。
{correction_hint}

可用动作：
- navigate: 打开网址，需要提供 url。
- click: 点击元素，需要提供 element_description。
- type: 输入文本，需要提供 text，可选 delay、press_enter。
- press_key: 按下按键，需要提供 key。
- wait: 等待页面加载，需要提供 timeout（毫秒）。
- scroll: 滚动页面，direction=up/down/left/right，amount=像素值（默认600）。
- finish: 任务结束。

请严格输出 JSON：
{{
  "current_step": "string",
  "action_type": "navigate/click/type/press_key/wait/scroll/finish",
  "action_params": {{...}},
  "next": "tools/end",
  "reasoning": "string"
}}
        """.strip()

    return prompt


def _clean_tool_feedback(tool_result: dict | None) -> str:
    if not tool_result:
        return "无"

    cleaned_result = {}
    for key, value in tool_result.items():
        if key == "current_view" and isinstance(value, dict):
            cleaned_view = {
                k: value.get(k)
                for k in ("label", "timestamp", "meta")
                if value.get(k) is not None
            }
            if isinstance(cleaned_view.get("meta"), dict):
                cleaned_view["meta"] = {
                    mk: cleaned_view["meta"].get(mk)
                    for mk in ("timestamp", "url", "sha1")
                    if cleaned_view["meta"].get(mk) is not None
                }
            cleaned_result[key] = cleaned_view
        else:
            cleaned_result[key] = value

    try:
        return json.dumps(cleaned_result, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(cleaned_result)


def _format_comparison(comparison: dict | None) -> str:
    if not comparison:
        return "缺少比较信息"

    changed = "有变化" if comparison.get("changed") else "无明显变化"
    similarity = comparison.get("similarity")
    if similarity is not None:
        return f"{changed}（相似度 {similarity:.2f}）"
    return changed
