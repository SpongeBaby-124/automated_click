"""
Web 自动化 LangGraph 包

这个包提供了基于 LangGraph 的网页自动化功能，全部依赖多模态 Qwen3-VL 模型完成规划、执行与审查。

使用示例：
    from langgraph_app import create_automation_graph
    
    graph, page, context, pw = await create_automation_graph()
    result = await graph.ainvoke({"user_goal": "打开谷歌并搜索南京邮电大学官网"})
"""

from .automation_graph import build_automation_graph
from .vision_tool import VisionClickTool

__all__ = ["build_automation_graph", "VisionClickTool", "create_automation_graph"]

__version__ = "1.0.0"


async def create_automation_graph(
    initial_url: str | None = None,
    headless: bool = False,
    user_data_dir: str | None = None,
    browser_channel: str | None = None,
):
    """
    创建自动化任务图和浏览器页面
    
    Args:
        initial_url: 可选的初始网页 URL，默认为空白页
        headless: 是否使用无头模式
        user_data_dir: 用户数据目录路径，用于保存登录状态、cookies 等
                  如果为 None，将使用项目根目录下的 './browser_data'
        browser_channel: Playwright 浏览器通道名称，可设为 'chrome'、'msedge' 等
                 默认为环境变量 PLAYWRIGHT_BROWSER_CHANNEL 或 chrome
        
    Returns:
        (graph, page, context, playwright) 元组
        - graph: 编译好的 LangGraph，可直接调用 ainvoke()
        - page: Playwright 页面对象
        - context: Playwright 浏览器上下文对象 (持久化)
        - playwright: Playwright 实例
        
    使用示例：
        graph, page, context, pw = await create_automation_graph()
        try:
            result = await graph.ainvoke({"user_goal": "点击搜索框并输入内容"})
            print(result)
        finally:
            await context.close()
            await pw.stop()
    """
    import os
    from pathlib import Path
    from playwright.async_api import async_playwright
    
    # 解析用户数据目录，默认使用项目根目录的 browser_data
    if user_data_dir:
        user_data_path = Path(user_data_dir).expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent
        user_data_path = project_root / "browser_data"
    user_data_path.mkdir(parents=True, exist_ok=True)
    resolved_user_data_dir = str(user_data_path)
    print(f"💾 使用浏览器数据目录: {resolved_user_data_dir}")
    
    playwright = await async_playwright().start()
    launch_kwargs = {
        "headless": headless,
        "viewport": {"width": 1280, "height": 720},
        "accept_downloads": True,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    preferred_channels: list[str | None] = []
    if browser_channel:
        preferred_channels.append(browser_channel)
    else:
        env_channel = os.environ.get("PLAYWRIGHT_BROWSER_CHANNEL")
        if env_channel:
            preferred_channels.append(env_channel)
        preferred_channels.append("chrome")
    preferred_channels.append(None)  # 最后回退到内置 Chromium

    context = None
    last_error: Exception | None = None
    for channel in preferred_channels:
        try:
            if channel:
                print(f"🧭 尝试使用浏览器通道: {channel}")
                context = await playwright.chromium.launch_persistent_context(
                    resolved_user_data_dir,
                    channel=channel,
                    **launch_kwargs,
                )
            else:
                print("🧭 使用内置 Chromium 浏览器")
                context = await playwright.chromium.launch_persistent_context(
                    resolved_user_data_dir,
                    **launch_kwargs,
                )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"⚠️ 启动 {channel or 'chromium'} 失败，将尝试其他通道: {exc}")
            continue
    if context is None:
        raise RuntimeError("无法启动任何浏览器通道") from last_error
    
    # 设置默认超时
    context.set_default_timeout(30000)
    
    # 获取或创建页面
    pages = context.pages
    if pages:
        page = pages[0]
    else:
        page = await context.new_page()
    
    # 根据需要打开初始网页
    if initial_url:
        await page.goto(initial_url, wait_until="domcontentloaded")
    elif page.url == "about:blank":
        # 如果是空白页且没有指定初始URL，保持空白页
        pass
    
    # 创建工具和图
    tool = VisionClickTool(page, context=context)
    graph = build_automation_graph(tool)
    
    return graph, page, context, playwright
