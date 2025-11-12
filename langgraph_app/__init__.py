"""
Web 自动化 LangGraph 包

这个包提供了基于 LangGraph 的网页自动化功能，全部依赖多模态 Qwen3-VL 模型完成规划、执行与审查。

使用示例：
    from langgraph_app import create_automation_graph
    
    graph, page, browser, pw = await create_automation_graph()
    result = await graph.ainvoke({"user_goal": "打开谷歌并搜索南京邮电大学官网"})
"""

from .automation_graph import build_automation_graph
from .vision_tool import VisionClickTool

__all__ = ["build_automation_graph", "VisionClickTool", "create_automation_graph"]

__version__ = "1.0.0"


async def create_automation_graph(initial_url: str | None = None, headless: bool = False):
    """
    创建自动化任务图和浏览器页面
    
    Args:
    initial_url: 可选的初始网页 URL，默认为空白页
        headless: 是否使用无头模式
        
    Returns:
        (graph, page, browser, playwright) 元组
        - graph: 编译好的 LangGraph，可直接调用 ainvoke()
        - page: Playwright 页面对象
        - browser: Playwright 浏览器对象
        - playwright: Playwright 实例
        
    使用示例：
        graph, page, browser, pw = await create_automation_graph()
        try:
            result = await graph.ainvoke({"user_goal": "点击搜索框并输入内容"})
            print(result)
        finally:
            await browser.close()
            await pw.stop()
    """
    from playwright.async_api import async_playwright
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless)
    page = await browser.new_page(viewport={"width": 1280, "height": 720})
    page.set_default_timeout(30000)
    
    # 根据需要打开初始网页
    if initial_url:
        await page.goto(initial_url, wait_until="domcontentloaded")
    else:
        await page.goto("about:blank")
    
    # 创建工具和图
    tool = VisionClickTool(page)
    graph = build_automation_graph(tool)
    
    return graph, page, browser, playwright
