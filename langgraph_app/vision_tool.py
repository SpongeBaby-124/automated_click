"""视觉定位工具 - 使用 VL 模型定位并点击网页元素"""

import base64
import asyncio
import os
import re
from typing import Tuple
from datetime import datetime

from openai import OpenAI
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError


class VisionClickTool:
    """使用多模态 VL 模型定位网页元素并执行点击操作"""

    def __init__(self, page: Page) -> None:
        """
        初始化视觉点击工具
        
        Args:
            page: Playwright 页面对象
        """
        self._page = page
        self._screenshot_count = 0
        
        # 从环境变量获取配置
        base_url = os.environ.get("OPENAI_API_BASE")
        api_key = os.environ.get("OPENAI_API_KEY")
        self._vision_model = os.environ.get("VISION_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
        
        if not base_url or not api_key:
            raise EnvironmentError("必须设置 OPENAI_API_BASE 和 OPENAI_API_KEY 环境变量")
        
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    async def capture_state(self, label: str = "state") -> dict:
        """对外公开的截图方法，返回路径和 base64"""
        return await self._capture_state(label)

    async def _screenshot_base64(self) -> str:
        """获取页面截图并转换为 base64 编码"""
        screenshot = await self._page.screenshot()
        return base64.b64encode(screenshot).decode("utf-8")

    async def _capture_state(self, label: str) -> dict:
        """截图记录当前页面状态并返回路径与 base64"""
        self._screenshot_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{self._screenshot_count}_{timestamp}_{label}.png"
        screenshot = await self._page.screenshot()
        with open(filename, 'wb') as f:
            f.write(screenshot)
        print(f"💾 已保存调试截图: {filename}")
        return {
            "path": filename,
            "base64": base64.b64encode(screenshot).decode("utf-8"),
        }

    async def _save_screenshot_for_debug(self, label: str = "debug") -> str:
        """兼容旧接口，返回截图文件路径"""
        state = await self._capture_state(label)
        return state["path"]

    async def _ai_locate(self, element_description: str, retry_count: int = 2) -> Tuple[int, int]:
        """
        使用 VL 模型定位元素坐标
        
        Args:
            element_description: 元素的文字描述
            retry_count: 重试次数
            
        Returns:
            (x, y) 坐标元组
        """
        screenshot = await self._screenshot_base64()
        
        # 构建增强的提示词
        enhanced_prompt = f"""
请在这个网页截图中找到以下元素: '{element_description}'

请仔细分析图像并返回：
1. 该元素的中心坐标 (x, y)
2. 坐标必须是有效的数字，范围在图像大小内

只返回坐标，格式必须是 (x, y)，例如 (123, 456)
        """.strip()
        
        for attempt in range(retry_count + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": enhanced_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{screenshot}",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=150,
                    temperature=0.1,
                )
                
                result = response.choices[0].message.content
                print(f"📍 VL 模型返回 (尝试 {attempt + 1}/{retry_count + 1}): {result}")
                
                coords = self._parse_coordinates(result)
                print(f"✅ 成功解析坐标: {coords}")
                return coords
                
            except Exception as e:
                if attempt < retry_count:
                    print(f"⚠️ 定位失败，正在重试 ({attempt + 1}/{retry_count}): {e}")
                    await asyncio.sleep(0.5)
                else:
                    await self._save_screenshot_for_debug("locate_error")
                    raise ValueError(f"VL 模型定位失败: {e}")

    @staticmethod
    def _parse_coordinates(text: str) -> Tuple[int, int]:
        """
        解析 VL 模型返回的坐标文本
        
        支持多种格式:
        - (123, 456)
        - 123, 456
        - x=123, y=456
        """
        patterns = [
            r'\((\d+)\s*,\s*(\d+)\)',
            r'^\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*$',
            r'(\d+)\s*,\s*(\d+)',
            r'x[=:]\s*(\d+).*?y[=:]\s*(\d+)',
        ]
        
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                
                if x < 0 or y < 0:
                    raise ValueError(f"坐标为负值: ({x}, {y})")
                
                print(f"✓ 使用模式 {idx + 1} 成功解析")
                return x, y
        
        raise ValueError(f"无法解析坐标，VL 模型返回: {text}")

    async def click_element(self, element_description: str) -> dict:
        """
        定位并点击指定元素
        
        Args:
            element_description: 元素的文字描述
            
        Returns:
            执行结果字典，包含 success, message, coordinates 等字段
        """
        try:
            print(f"\n🔍 正在使用 VL 模型定位: {element_description}")
            
            # 使用 VL 模型定位坐标
            x, y = await self._ai_locate(element_description)
            
            if x == 0 and y == 0:
                screenshot = await self.capture_state("click_zero_coord")
                return {
                    "success": False,
                    "message": "坐标为 (0, 0)，疑似定位失败",
                    "coordinates": (0, 0),
                    "element_description": element_description,
                    "screenshot_path": screenshot["path"],
                    "screenshot_base64": screenshot["base64"],
                }
            
            # 执行点击
            print(f"🖱️  点击位置: ({x}, {y})")
            await self._page.mouse.click(x, y)
            await asyncio.sleep(0.5)
            
            print(f"✓ 成功点击: {element_description}")
            screenshot = await self.capture_state("click_success")
            return {
                "success": True,
                "message": f"成功点击 {element_description}",
                "coordinates": (x, y),
                "element_description": element_description,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }

        except Exception as e:
            error_msg = f"点击失败: {str(e)}"
            print(f"✗ {error_msg}")
            screenshot = await self.capture_state("click_error")

            return {
                "success": False,
                "message": error_msg,
                "element_description": element_description,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }

    async def type_text(self, text: str, delay: int = 50) -> dict:
        """
        在当前焦点元素输入文本
        
        Args:
            text: 要输入的文本
            delay: 每个字符的延迟(毫秒)
            
        Returns:
            执行结果字典
        """
        try:
            print(f"⌨️  输入文本: {text}")
            await self._page.keyboard.type(text, delay=delay)
            await asyncio.sleep(0.3)
            screenshot = await self.capture_state("type_text")

            return {
                "success": True,
                "message": f"成功输入文本: {text}",
                "text": text,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }
        except Exception as e:
            screenshot = await self.capture_state("type_text_error")
            return {
                "success": False,
                "message": f"输入文本失败: {str(e)}",
                "text": text,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }

    async def press_key(self, key: str) -> dict:
        """
        按下键盘按键
        
        Args:
            key: 按键名称，如 "Enter", "Escape" 等
            
        Returns:
            执行结果字典
        """
        try:
            print(f"⌨️  按下按键: {key}")
            await self._page.keyboard.press(key)
            await asyncio.sleep(0.3)
            screenshot = await self.capture_state("press_key")

            return {
                "success": True,
                "message": f"成功按下按键: {key}",
                "key": key,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }
        except Exception as e:
            screenshot = await self.capture_state("press_key_error")
            return {
                "success": False,
                "message": f"按键失败: {str(e)}",
                "key": key,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }

    async def wait_for_navigation(self, timeout: int = 10000) -> dict:
        """
        等待页面导航完成
        
        Args:
            timeout: 超时时间(毫秒)
            
        Returns:
            执行结果字典
        """
        try:
            print(f"⏳ 等待页面加载...")
            await self._page.wait_for_load_state("domcontentloaded", timeout=timeout)
            await asyncio.sleep(1)
            screenshot = await self.capture_state("wait_navigation")

            return {
                "success": True,
                "message": "页面加载完成",
                "url": self._page.url,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }
        except PlaywrightTimeoutError:
            screenshot = await self.capture_state("wait_timeout")
            return {
                "success": False,
                "message": f"页面加载超时 ({timeout}ms)",
                "url": self._page.url,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }
        except Exception as e:
            screenshot = await self.capture_state("wait_error")
            return {
                "success": False,
                "message": f"等待导航失败: {str(e)}",
                "url": self._page.url,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }

    async def navigate_to(self, url: str, timeout: int = 20000) -> dict:
        """导航到指定 URL 并确认页面加载"""
        try:
            print(f"🌍 正在打开: {url}")
            await self._page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            await asyncio.sleep(1)
            screenshot = await self.capture_state("navigate_success")

            return {
                "success": True,
                "message": f"已打开 {url}",
                "url": self._page.url,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }
        except PlaywrightTimeoutError:
            screenshot = await self.capture_state("navigate_timeout")
            return {
                "success": False,
                "message": f"打开 {url} 超时",
                "url": self._page.url,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }
        except Exception as e:
            screenshot = await self.capture_state("navigate_error")
            return {
                "success": False,
                "message": f"打开 {url} 失败: {str(e)}",
                "url": self._page.url,
                "screenshot_path": screenshot["path"],
                "screenshot_base64": screenshot["base64"],
            }

    async def plan_action(
        self,
        user_goal: str,
        tool_result: dict | None,
        attempt_count: int,
    ) -> dict:
        """使用 VL 模型进行下一步规划"""
        screenshot = await self.capture_state("agent_plan")

        tool_feedback = "无"
        if tool_result:
            try:
                import json

                tool_feedback = json.dumps(tool_result, ensure_ascii=False)
            except Exception:
                tool_feedback = str(tool_result)

        prompt = f"""
你现在控制着一个网页自动化代理，目标是通过多步操作完成用户的需求。

用户目标：{user_goal}
最近一步工具反馈：{tool_feedback}
当前针对同一动作的尝试次数：{attempt_count} / 5

请仔细观察提供的最新网页截图，判断任务是否已经完成。如果未完成，请规划下一步动作。

动作类型说明：
- navigate: 打开网址，需要提供 url 字段。
- click: 点击元素，需要提供 element_description 字段，描述要点击的元素。
- type: 在当前焦点输入文本，需要提供 text 字段，可选 delay（整数，毫秒）。
- press_key: 按下键盘按键，需要提供 key 字段。
- wait: 等待页面加载，需要提供 timeout 字段（毫秒）。
- finish: 任务结束，不需要额外参数。

请严格输出 JSON 格式，包含以下键：
{{
  "current_step": "当前计划的步骤描述",
  "action_type": "navigate/click/type/press_key/wait/finish",
  "action_params": {{...}},
  "next": "tools/end",
  "reasoning": "简要说明原因"
}}

只有在确信任务目标已经完成时，才将 next 设置为 "end" 并选择 action_type 为 "finish"。
如果最新工具反馈 success 为 False 或者你不确定是否成功，请继续规划后续操作。
""".strip()

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
                                "url": f"data:image/png;base64,{screenshot['base64']}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=400,
            temperature=0.1,
        )

        raw = response.choices[0].message.content or ""
        print(f"🧠 VL 规划输出: {raw}")
        return {
            "raw_response": raw,
            "screenshot_path": screenshot["path"],
            "screenshot_base64": screenshot["base64"],
        }
