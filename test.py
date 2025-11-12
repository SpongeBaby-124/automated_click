import os
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from openai import OpenAI
import base64
import re
import asyncio
from datetime import datetime
from typing import Sequence

class AIWebAgent:
    def __init__(self, page):
        self.page = page
        # 使用 ModelScope 的 Qwen3-VL 模型
        self.client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='ms-9884ba84-606a-4e1f-b355-c4dab349055f',  # ModelScope Token
        )
        self.screenshot_count = 0
    
    async def screenshot_base64(self):
        """获取页面截图"""
        screenshot = await self.page.screenshot()
        return base64.b64encode(screenshot).decode('utf-8')
    
    async def save_screenshot_for_debug(self, label: str | None = None):
        """保存截图用于调试"""
        self.screenshot_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        filename = f"screenshot_{self.screenshot_count}_{timestamp}{suffix}.png"
        screenshot = await self.page.screenshot()
        with open(filename, 'wb') as f:
            f.write(screenshot)
        print(f"💾 已保存截图: {filename}")
        return filename
    
    async def ai_locate(self, prompt: str):
        """使用 AI 定位元素"""
        screenshot = await self.screenshot_base64()
        
        enhanced_prompt = f"""
请在这个网页截图中找到以下元素: '{prompt}'

请仔细分析图像并返回：
1. 该元素的中心坐标 (x, y)
2. 坐标必须是有效的数字，范围在图像大小内

只返回坐标，格式必须是 (x, y)，例如 (123, 456)
        """.strip()
        
        response = self.client.chat.completions.create(
            model='Qwen/Qwen3-VL-235B-A22B-Instruct',
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": enhanced_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        
        result = response.choices[0].message.content
        print(f"📍 AI 原始返回: {result}")
        
        coords = self._parse_coordinates(result)
        print(f"✅ 解析后坐标: {coords}")
        return coords
    
    async def ai_click(self, element_description: str, retry_count: int = 0):
        """AI 驱动的点击"""
        try:
            print(f"\n🔍 正在尝试使用 AI 定位: {element_description}")
            x, y = await self.ai_locate(element_description)
            if x == 0 and y == 0:
                print("⚠️ 坐标为 (0, 0)，疑似定位失败")
            print(f"🖱️  点击位置: ({x}, {y})")
            await self.page.mouse.click(x, y)
            print(f"✓ 已点击: {element_description} at ({x}, {y})")
            await asyncio.sleep(0.5)
            return True
        except Exception as err:
            print(f"✗ AI 点击失败: {element_description}")
            print(f"  错误: {err}")
            if retry_count == 0:
                await self.save_screenshot_for_debug(label="ai_click_error")
            return False
    
    def _parse_coordinates(self, text: str):
        """解析 AI 返回的坐标"""
        print(f"📝 正在解析坐标: {repr(text)}")
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
                print(f"✓ 成功匹配第 {idx + 1} 个模式")
                if x < 0 or y < 0:
                    raise ValueError(f"坐标为负值: ({x}, {y})")
                return x, y
        raise ValueError(
            "无法解析坐标，请检查模型返回值。"
            f"\n文本内容: {text}"
        )


async def click_with_selectors(
    page,
    description: str,
    selectors: Sequence[str],
    click_via_parent: bool = False,
):
    """尝试通过一组选择器进行点击，成功返回 True。"""
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            count = await locator.count()
        except PlaywrightTimeoutError:
            count = 0
        if count == 0:
            continue
        try:
            if click_via_parent:
                element_handle = await locator.element_handle()
                if element_handle and await element_handle.evaluate("node => node.parentElement"):
                    print(f"🖱️ 使用父节点点击 {description} (selector={selector})")
                    async with page.expect_navigation(wait_until="domcontentloaded", timeout=20000):
                        await element_handle.evaluate("node => node.parentElement.click()")
                else:
                    async with page.expect_navigation(wait_until="domcontentloaded", timeout=20000):
                        await locator.click()
            else:
                print(f"🖱️ 使用 selector={selector} 点击 {description}")
                await locator.click()
            await asyncio.sleep(0.3)
            print(f"✓ 成功通过选择器点击 {description}")
            return True
        except Exception as err:
            print(f"⚠️ 选择器 {selector} 点击 {description} 失败: {err}")
            continue
    return False


async def dismiss_google_consent(page):
    """尝试关闭 Google 的隐私弹窗"""
    possible_texts = [
        '同意',
        '接受全部',
        '同意全部',
        'I agree',
        'Accept all',
    ]
    for text in possible_texts:
        button = page.get_by_role("button", name=text)
        if await button.count():
            try:
                print(f"🛡️ 检测到 Google 隐私弹窗，点击按钮: {text}")
                await button.first.click()
                await asyncio.sleep(1.5)
                return True
            except Exception:
                continue
    return False


async def wait_for_manual_close(page):
    print("\n🕒 浏览器保持打开状态。")
    print("   ✅ 可以自由操作页面。")
    print("   ❌ 如果想结束运行，请手动关闭浏览器标签页。")
    print("   ⏳ Notebook 将等待直到页面被关闭...")
    try:
        await page.wait_for_event("close")
    except Exception:
        pass
    print("🔚 检测到页面已关闭，任务结束。")


# 异步执行任务
async def main():
    print("\n" + "=" * 70)
    print("🚀 AI Web Agent - 搜索南京邮电大学官网任务")
    print("=" * 70)
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    page = await browser.new_page(viewport={"width": 1280, "height": 720})
    page.set_default_timeout(30000)
    agent = AIWebAgent(page)
    
    try:
        print("\n1️⃣ 打开 Google 搜索引擎...")
        await page.goto("https://www.google.com", wait_until="domcontentloaded")
        await asyncio.sleep(2)
        await dismiss_google_consent(page)
        print("✅ Google 首页已加载")
        await agent.save_screenshot_for_debug(label="google_home")
        
        print("\n2️⃣ 点击搜索输入框 (优先 CSS 选择器)...")
        clicked = await click_with_selectors(
            page,
            description="搜索输入框",
            selectors=[
                'textarea[name="q"]',
                'input[name="q"]',
            ],
        )
        if not clicked:
            print("⚠️ CSS 选择器未找到搜索输入框，尝试 AI 定位...")
            await agent.ai_click("Google 搜索输入框")
        await asyncio.sleep(0.5)
        
        print("\n3️⃣ 输入搜索内容: 南京邮电大学官网")
        await page.keyboard.type("南京邮电大学官网", delay=50)
        await asyncio.sleep(0.5)
        
        print("\n4️⃣ 直接按 Enter 发起搜索（比点击按钮更稳定）")
        await page.keyboard.press("Enter")
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(2)
        await agent.save_screenshot_for_debug(label="search_results")
        print("✅ 搜索结果页已加载")
        
        print("\n5️⃣ 定位并点击第一个搜索结果...")
        selectors_for_result = [
            'div#search a:has(h3)',
            'a h3',
        ]
        success = False
        for selector in selectors_for_result:
            locator = page.locator(selector).first
            if await locator.count():
                try:
                    print(f"🖱️ 尝试使用 selector={selector} 点击第一个搜索结果")
                    async with page.expect_navigation(wait_until="domcontentloaded", timeout=20000):
                        if selector.endswith('h3'):
                            element_handle = await locator.element_handle()
                            await element_handle.evaluate("node => node.parentElement.click()")
                        else:
                            await locator.click()
                    success = True
                    break
                except Exception as err:
                    print(f"⚠️ 通过 selector={selector} 点击失败: {err}")
                    continue
        if not success:
            print("⚠️ 选择器点击失败，尝试使用 AI 定位第一个搜索结果...")
            success = await agent.ai_click("第一个搜索结果链接", retry_count=1)
        
        if success:
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(3)
            await agent.save_screenshot_for_debug(label="njupt_home")
            print("\n✅ 任务完成！预计已打开南京邮电大学官网")
        else:
            print("\n⚠️ 未能成功点击第一个搜索结果，请查看截图进行诊断")
        
        print("\n" + "=" * 70)
        print("📌 浏览器保持打开状态")
        print("   🔍 请查看页面确认是否已经打开南京邮电大学官网")
        print("   🖱️ 如需结束，请手动关闭浏览器标签页")
        print("=" * 70)
        
        await wait_for_manual_close(page)
    except PlaywrightTimeoutError as timeout_err:
        print("\n⏰ 操作超时:", timeout_err)
        await agent.save_screenshot_for_debug(label="timeout")
        await wait_for_manual_close(page)
    except Exception as err:
        print("\n❌ 任务执行出错：", err)
        await agent.save_screenshot_for_debug(label="unhandled_error")
        await wait_for_manual_close(page)
    finally:
        await browser.close()
        await playwright.stop()

# 在 Jupyter 中运行异步代码
asyncio.run(main())
