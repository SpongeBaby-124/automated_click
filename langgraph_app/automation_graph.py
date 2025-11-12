"""Web 自动化 LangGraph - Agent 和 Tools 节点的编排"""

import json
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .vision_tool import VisionClickTool


class AutomationState(TypedDict, total=False):
    """自动化任务的状态定义"""

    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史
    user_goal: str  # 用户目标
    current_step: str  # 当前步骤描述
    action_type: str  # 动作类型: navigate, click, type, press_key, wait, finish
    action_params: dict  # 动作参数
    decision: Literal["tools", "end"]  # 下一步决策: 执行工具或结束
    tool_result: dict  # 工具执行结果
    attempt_count: int  # 当前动作已尝试次数
    agent_view: dict  # Agent 规划时的截图信息


def _agent_node(tool: VisionClickTool):
    """Agent 节点 - 使用 VL 模型进行规划与审查"""

    async def node(state: AutomationState) -> AutomationState:
        try:
            plan_response = await tool.plan_action(
                user_goal=state.get("user_goal", ""),
                tool_result=state.get("tool_result"),
                attempt_count=state.get("attempt_count", 0),
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
                    "agent_view": {
                        "screenshot_path": plan_response.get("screenshot_path"),
                        "screenshot_base64": plan_response.get("screenshot_base64"),
                    },
                    "messages": [AIMessage(content="规划器返回格式错误，任务终止。")],
                }

            decision = parsed.get("next", "end").lower()
            if decision not in {"tools", "end"}:
                decision = "end"

            current_step = parsed.get("current_step", "")
            action_type = parsed.get("action_type", "finish")
            action_params = parsed.get("action_params", {})
            reasoning = parsed.get("reasoning", "")

            print(f"✅ 规划决策: {current_step}")
            print(f"   动作类型: {action_type}")
            print(f"   决策: {decision}")
            print(f"   推理: {reasoning}")

            return {
                "current_step": current_step,
                "action_type": action_type,
                "action_params": action_params,
                "decision": decision,
                "attempt_count": state.get("attempt_count", 0),
                "agent_view": {
                    "screenshot_path": plan_response.get("screenshot_path"),
                    "screenshot_base64": plan_response.get("screenshot_base64"),
                },
                "messages": [AIMessage(content=f"{current_step}\n推理：{reasoning}")],
            }

        except Exception as e:
            print(f"❌ Agent 节点异常: {e}")
            return {
                "current_step": "异常终止",
                "action_type": "finish",
                "action_params": {},
                "decision": "end",
                "agent_view": None,
                "messages": [AIMessage(content=f"规划异常：{str(e)}")],
            }

    return node


def _extract_json_from_response(text: str) -> dict:
    """
    从 LLM 响应中提取 JSON
    
    支持多种格式：
    - 纯 JSON
    - ```json ... ```
    - 混合文本中的 JSON
    """
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试从 markdown 代码块提取
    import re
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试查找任何 JSON 对象
    json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_object_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "next" in parsed or "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None


def _tools_node(tool: VisionClickTool):
    """
    Tools 节点 - 执行具体的自动化操作
    
    根据 Agent 的决策调用相应的工具方法
    """
    async def node(state: AutomationState) -> AutomationState:
        action_type = (state.get("action_type") or "").lower()
        action_params = state.get("action_params", {})
        max_attempts = 5
        attempt = state.get("attempt_count", 0) + 1
        attempt = min(attempt, max_attempts)

        print(f"\n🔧 Tools 节点执行: {action_type}")
        print(f"   参数: {action_params}")
        print(f"   尝试次数: 第 {attempt} 次 (最多 {max_attempts} 次)")

        result: dict

        try:
            if action_type == "navigate":
                url = action_params.get("url", "")
                if not url:
                    screenshot = await tool.capture_state("missing_url")
                    result = {
                        "success": False,
                        "message": "缺少 url 参数",
                        "screenshot_path": screenshot["path"],
                        "screenshot_base64": screenshot["base64"],
                    }
                else:
                    result = await tool.navigate_to(url)

            elif action_type == "click":
                element_desc = action_params.get("element_description", "")
                if not element_desc:
                    screenshot = await tool.capture_state("missing_element_desc")
                    result = {
                        "success": False,
                        "message": "缺少 element_description 参数",
                        "screenshot_path": screenshot["path"],
                        "screenshot_base64": screenshot["base64"],
                    }
                else:
                    result = await tool.click_element(element_desc)

            elif action_type == "type":
                text = action_params.get("text", "")
                delay = action_params.get("delay", 50)
                if not text:
                    screenshot = await tool.capture_state("missing_text")
                    result = {
                        "success": False,
                        "message": "缺少 text 参数",
                        "screenshot_path": screenshot["path"],
                        "screenshot_base64": screenshot["base64"],
                    }
                else:
                    result = await tool.type_text(text, delay)

            elif action_type == "press_key":
                key = action_params.get("key", "")
                if not key:
                    screenshot = await tool.capture_state("missing_key")
                    result = {
                        "success": False,
                        "message": "缺少 key 参数",
                        "screenshot_path": screenshot["path"],
                        "screenshot_base64": screenshot["base64"],
                    }
                else:
                    result = await tool.press_key(key)

            elif action_type == "wait":
                timeout = action_params.get("timeout", 10000)
                result = await tool.wait_for_navigation(timeout)

            elif action_type == "finish":
                screenshot = await tool.capture_state("finish_review")
                result = {
                    "success": True,
                    "message": "Agent 主动结束任务",
                    "screenshot_path": screenshot["path"],
                    "screenshot_base64": screenshot["base64"],
                }

            else:
                screenshot = await tool.capture_state("unknown_action")
                result = {
                    "success": False,
                    "message": f"未知的动作类型: {action_type}",
                    "screenshot_path": screenshot["path"],
                    "screenshot_base64": screenshot["base64"],
                }

            result.setdefault("success", False)
            result.setdefault("message", "未提供执行结果")
            result.update(
                {
                    "action_type": action_type,
                    "action_params": action_params,
                    "attempt": attempt,
                }
            )

            if not result["success"] and attempt >= max_attempts:
                result["message"] += "（已达到最大重试次数）"

            print(f"✓ 工具执行结果: {result.get('message', '')}")

            next_attempt = 0 if result["success"] else attempt
            return {
                "tool_result": result,
                "attempt_count": next_attempt,
                "messages": [AIMessage(content=f"执行结果：{result.get('message', '')}")],
            }

        except Exception as e:
            error_msg = f"工具执行异常: {str(e)}"
            print(f"❌ {error_msg}")
            screenshot = await tool.capture_state("tool_exception")
            result = {
                "success": False,
                "message": error_msg,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
                "action_type": action_type,
                "action_params": action_params,
                "attempt": attempt,
            }
            return {
                "tool_result": result,
                "attempt_count": attempt,
                "messages": [AIMessage(content=error_msg)],
            }
    
    return node


def build_automation_graph(tool: VisionClickTool):
    """
    构建自动化任务的 LangGraph
    
    Args:
        tool: VisionClickTool 实例
        
    Returns:
        编译后的 StateGraph，可直接调用 invoke() 方法
    """
    # 创建状态图
    graph = StateGraph(AutomationState)
    
    # 添加节点
    graph.add_node("agent", _agent_node(tool))
    graph.add_node("tools", _tools_node(tool))
    
    # 设置入口点
    graph.set_entry_point("agent")
    
    # 定义路由函数
    def router(state: AutomationState) -> str:
        """根据 Agent 的决策路由到下一个节点"""
        decision = state.get("decision", "end")
        return decision
    
    # 添加条件边：agent -> tools 或 end
    graph.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "end": END,
        },
    )
    
    # 添加边：tools -> agent
    graph.add_edge("tools", "agent")
    
    # 编译并返回
    return graph.compile()
