# 浏览器数据持久化说明

## 功能特性

项目已配置使用 Playwright 的**持久化上下文**功能,可以保存浏览器的登录状态和用户数据。

## 保存的数据

浏览器数据目录 `./browser_data` 将保存:

- 🔐 **登录状态**: 所有网站的登录 session
- 🍪 **Cookies**: 保持会话持久性
- 💾 **LocalStorage**: 网站本地存储数据
- 📝 **浏览历史**: 浏览记录
- ⚙️ **浏览器设置**: 自定义配置

## 使用说明

### 首次使用

1. 运行程序,浏览器会以全新状态启动
2. 手动登录需要的网站(如淘宝、百度账号等)
3. 关闭浏览器后,所有登录信息会自动保存

### 后续使用

- 再次运行程序时,之前的登录状态会自动恢复
- 无需重复登录操作
- 就像使用自己的日常浏览器一样

### 清除数据

如需清除所有保存的数据(重新开始):

```powershell
# 删除浏览器数据目录
Remove-Item -Recurse -Force .\browser_data
```

## 自定义配置

如需使用自定义的数据目录,可修改 `create_automation_graph` 调用:

```python
graph, page, context, playwright = await create_automation_graph(
    headless=False,
    user_data_dir="C:\\my_custom_browser_data"  # 自定义路径
)
```

## 注意事项

⚠️ **安全提示**:
- `browser_data` 目录包含敏感的登录信息
- 已添加到 `.gitignore`,不会被 Git 跟踪
- 请勿分享或上传此目录

⚠️ **多实例运行**:
- 同一数据目录同时只能被一个浏览器实例使用
- 如需并发运行,为每个实例指定不同的 `user_data_dir`

## 技术细节

项目使用 `playwright.chromium.launch_persistent_context()` 而非传统的 `launch() + new_page()`:

**优势**:
- 数据自动持久化到磁盘
- 模拟真实用户浏览器行为
- 减少反爬虫检测风险

**对比传统方式**:
```python
# 传统方式 (每次都是全新浏览器)
browser = await playwright.chromium.launch()
page = await browser.new_page()

# 持久化方式 (保存用户数据)
context = await playwright.chromium.launch_persistent_context("./browser_data")
page = context.pages[0]
```
