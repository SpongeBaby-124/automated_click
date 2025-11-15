"""视觉执行工具 - 提供浏览器操作能力"""

import asyncio
import os
import re
from typing import Dict, Optional, Tuple

from openai import OpenAI
from playwright.async_api import (
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

from .state_utils import build_view_payload


class VisionClickTool:
    """封装具体浏览器动作，供 Tools 节点调用"""

    def __init__(self, page: Page, context: BrowserContext | None = None) -> None:
        self._page = page
        self._context = context

        # 监听页面关闭 / 新标签打开事件，确保始终操作最新页面
        self._register_page_close_hook(page)
        if self._context:
            self._context.on("page", self._handle_new_page)

        base_url = os.environ.get("OPENAI_API_BASE")
        api_key = os.environ.get("OPENAI_API_KEY")
        self._vision_model = os.environ.get("VISION_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

        if not base_url or not api_key:
            raise EnvironmentError("必须设置 OPENAI_API_BASE 和 OPENAI_API_KEY 环境变量")

        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def _register_page_close_hook(self, page: Page) -> None:
        page.on("close", lambda _: self._handle_page_closed(page))

    def _handle_new_page(self, page: Page) -> None:
        print("🆕 检测到新窗口/标签页，自动切换到最新页面")
        self._register_page_close_hook(page)
        self._page = page

        async def _prepare() -> None:
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 等待新页面加载时出错: {exc}")
            try:
                await page.bring_to_front()
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 无法将新页面置前: {exc}")

        asyncio.create_task(_prepare())

    def _handle_page_closed(self, page: Page) -> None:
        if page != self._page:
            return
        next_page = self._pick_latest_page(exclude=page)
        if next_page:
            print("↩️ 当前页面已关闭，回退到最近的可用页面")
            self._page = next_page
            asyncio.create_task(next_page.bring_to_front())

    def _pick_latest_page(self, exclude: Page | None = None) -> Optional[Page]:
        if not self._context:
            return None
        for candidate in reversed(self._context.pages):
            if candidate == exclude:
                continue
            if not candidate.is_closed():
                return candidate
        return None

    def _require_active_page(self) -> Page:
        if self._page and not self._page.is_closed():
            return self._page
        fallback = self._pick_latest_page()
        if fallback:
            self._page = fallback
            return fallback
        raise RuntimeError("当前没有可用的浏览器页面，请确认标签页未全部关闭")

    def _current_url(self) -> str:
        try:
            return self._require_active_page().url
        except Exception:
            return ""

    async def get_view(self, label: str = "state") -> Dict[str, object]:
        """截取当前页面，返回供 Agent 判断的截图"""
        page = self._require_active_page()
        screenshot_bytes = await page.screenshot()
        return build_view_payload(label, screenshot_bytes, page.url)

    async def _ai_locate(self, element_description: str, retry_count: int = 2) -> Tuple[Tuple[int, int], Dict[str, object]]:
        prompt = (
            "请在这个网页截图中找到以下元素: '"
            + element_description
            + "'。\n\n"
            + "请输出该元素的中心坐标，格式严格为 (x, y)。"
        )

        last_error: Optional[Exception] = None
        view = await self.get_view("locate_element")

        for attempt in range(retry_count + 1):
            try:
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
                                        "url": f"data:image/png;base64,{view['screenshot_base64']}",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=150,
                    temperature=0.1,
                )

                result = response.choices[0].message.content or ""
                print(f"📍 VL 模型返回 (尝试 {attempt + 1}/{retry_count + 1}): {result}")
                coords = self._parse_coordinates(result)
                print(f"✅ 成功解析坐标: {coords}")
                return coords, view

            except Exception as exc:  # noqa: BLE001 - 捕获模型解析失败
                last_error = exc
                if attempt < retry_count:
                    print(f"⚠️ 定位失败，正在重试 ({attempt + 1}/{retry_count}): {exc}")
                    await asyncio.sleep(0.5)
                else:
                    break

        raise ValueError(f"VL 模型定位失败: {last_error}")

    def _normalize_coordinates(
        self,
        coords: Tuple[int, int],
        view_meta: Dict[str, object] | None,
    ) -> Tuple[int, int]:
        page = self._require_active_page()
        viewport = page.viewport_size or {}
        vp_width = viewport.get("width") or 1
        vp_height = viewport.get("height") or 1

        shot_width = (view_meta or {}).get("width") or vp_width
        shot_height = (view_meta or {}).get("height") or vp_height

        scale_x = vp_width / shot_width if shot_width else 1
        scale_y = vp_height / shot_height if shot_height else 1

        raw_x, raw_y = coords
        adj_x = int(raw_x * scale_x)
        adj_y = int(raw_y * scale_y)

        adj_x = max(1, min(vp_width - 2, adj_x))
        adj_y = max(1, min(vp_height - 2, adj_y))
        return adj_x, adj_y

    async def _resolve_click_target(self, x: int, y: int) -> Optional[Dict[str, object]]:
        script = """
        ({ x, y }) => {
            const el = document.elementFromPoint(x, y);
            if (!el) {
                return null;
            }
            try {
                el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' });
            } catch (_) {}
            const rect = el.getBoundingClientRect();
            const anchor = el.closest('a');
            return {
                tag: el.tagName,
                text: (el.innerText || '').trim().slice(0, 120),
                href: anchor ? anchor.href : (el.href || null),
                rect: {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height,
                },
                center: {
                    x: rect.x + rect.width / 2,
                    y: rect.y + rect.height / 2,
                },
            };
        }
        """

        page = self._require_active_page()
        return await page.evaluate(script, {"x": x, "y": y})

    @staticmethod
    def _parse_coordinates(text: str) -> Tuple[int, int]:
        patterns = [
            r"\((\d+)\s*,\s*(\d+)\)",
            r"^\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*$",
            r"(\d+)\s*,\s*(\d+)",
            r"x[=:]\s*(\d+).*?y[=:]\s*(\d+)",
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
        try:
            print(f"\n🔍 正在使用 VL 模型定位: {element_description}")
            (raw_x, raw_y), locate_view = await self._ai_locate(element_description)
            normalized_x, normalized_y = self._normalize_coordinates(
                (raw_x, raw_y),
                locate_view.get("meta") if isinstance(locate_view, dict) else None,
            )

            if raw_x == 0 and raw_y == 0:
                view = await self.get_view("click_zero_coord")
                return {
                    "success": False,
                    "message": "坐标为 (0, 0)，疑似定位失败",
                    "coordinates": (0, 0),
                    "element_description": element_description,
                    "current_view": view,
                }

            element_target = await self._resolve_click_target(normalized_x, normalized_y)
            if element_target and element_target.get("center"):
                target_x = int(element_target["center"]["x"])
                target_y = int(element_target["center"]["y"])
            else:
                target_x, target_y = normalized_x, normalized_y

            print(
                "🖱️  点击位置 (原始 -> 映射 -> 最终): "
                f"({raw_x}, {raw_y}) -> ({normalized_x}, {normalized_y}) -> ({target_x}, {target_y})"
            )

            page = self._require_active_page()
            await page.bring_to_front()
            await page.mouse.move(target_x, target_y)
            await page.mouse.click(target_x, target_y)
            await asyncio.sleep(0.5)

            view = await self.get_view("click_success")
            return {
                "success": True,
                "message": f"成功点击 {element_description}",
                "coordinates": (raw_x, raw_y),
                "mapped_coordinates": (normalized_x, normalized_y),
                "final_coordinates": (target_x, target_y),
                "element_target": element_target,
                "element_description": element_description,
                "current_view": view,
            }

        except Exception as exc:
            error_msg = f"点击失败: {exc}"
            print(f"✗ {error_msg}")
            view = await self.get_view("click_error")
            return {
                "success": False,
                "message": error_msg,
                "element_description": element_description,
                "current_view": view,
            }

    async def type_text(self, text: str, delay: int = 50, press_enter: bool = False) -> dict:
        try:
            print(f"⌨️  输入文本: {text}")
            page = self._require_active_page()
            await page.keyboard.type(text, delay=delay)
            await asyncio.sleep(0.3)

            if press_enter:
                print("⌨️  自动按下 Enter 键")
                await page.keyboard.press("Enter")
                await asyncio.sleep(0.3)

            view = await self.get_view("type_text")
            return {
                "success": True,
                "message": f"成功输入文本: {text}" + (" 并按下 Enter" if press_enter else ""),
                "text": text,
                "press_enter": press_enter,
                "current_view": view,
            }

        except Exception as exc:
            view = await self.get_view("type_text_error")
            return {
                "success": False,
                "message": f"输入文本失败: {exc}",
                "text": text,
                "press_enter": press_enter,
                "current_view": view,
            }

    async def press_key(self, key: str) -> dict:
        try:
            print(f"⌨️  按下按键: {key}")
            page = self._require_active_page()
            await page.keyboard.press(key)
            await asyncio.sleep(0.3)

            view = await self.get_view("press_key")
            return {
                "success": True,
                "message": f"成功按下按键: {key}",
                "key": key,
                "current_view": view,
            }

        except Exception as exc:
            view = await self.get_view("press_key_error")
            return {
                "success": False,
                "message": f"按键失败: {exc}",
                "key": key,
                "current_view": view,
            }

    async def wait_for_navigation(self, timeout: int = 10000) -> dict:
        try:
            print("⏳ 等待页面加载...")
            page = self._require_active_page()
            await page.wait_for_load_state("domcontentloaded", timeout=timeout)
            await asyncio.sleep(1)

            view = await self.get_view("wait_navigation")
            return {
                "success": True,
                "message": "页面加载完成",
                "url": page.url,
                "current_view": view,
            }

        except PlaywrightTimeoutError:
            view = await self.get_view("wait_timeout")
            return {
                "success": False,
                "message": f"页面加载超时 ({timeout}ms)",
                "url": self._current_url(),
                "current_view": view,
            }

        except Exception as exc:
            view = await self.get_view("wait_error")
            return {
                "success": False,
                "message": f"等待导航失败: {exc}",
                "url": self._current_url(),
                "current_view": view,
            }

    async def navigate_to(self, url: str, timeout: int = 20000) -> dict:
        try:
            page = self._require_active_page()
            print(f"🌍 正在打开: {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            await asyncio.sleep(1)

            view = await self.get_view("navigate_success")
            return {
                "success": True,
                "message": f"已打开 {url}",
                "url": page.url,
                "current_view": view,
            }

        except PlaywrightTimeoutError:
            view = await self.get_view("navigate_timeout")
            return {
                "success": False,
                "message": f"打开 {url} 超时",
                "url": self._current_url(),
                "current_view": view,
            }

        except Exception as exc:
            view = await self.get_view("navigate_error")
            return {
                "success": False,
                "message": f"打开 {url} 失败: {exc}",
                "url": self._current_url(),
                "current_view": view,
            }

    async def scroll_page(self, direction: str, amount: int = 600) -> dict:
        direction = (direction or "down").lower()
        amount = int(amount or 600)
        dx = dy = 0

        if direction in {"down", "up"}:
            dy = amount if direction == "down" else -amount
        elif direction in {"left", "right"}:
            dx = -amount if direction == "left" else amount
        else:
            view = await self.get_view("scroll_invalid_direction")
            return {
                "success": False,
                "message": f"未知的滚动方向: {direction}",
                "direction": direction,
                "current_view": view,
            }

        try:
            page = self._require_active_page()
            print(f"🌀 滚动方向: {direction}, 距离: {amount}")
            await page.mouse.wheel(dx, dy)
            await asyncio.sleep(0.4)

            view = await self.get_view("scroll_success")
            return {
                "success": True,
                "message": f"成功滚动 {direction} {amount}px",
                "direction": direction,
                "amount": amount,
                "current_view": view,
            }

        except Exception as exc:  # noqa: BLE001
            view = await self.get_view("scroll_error")
            return {
                "success": False,
                "message": f"滚动失败: {exc}",
                "direction": direction,
                "amount": amount,
                "current_view": view,
            }
