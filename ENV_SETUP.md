# 环境配置说明

## 如何配置环境变量

### 1. 复制环境模板

```bash
cp .env.example .env
```

### 2. 获取 ModelScope API 密钥

1. 访问 [ModelScope 官网](https://modelscope.cn)
2. 注册/登录账号
3. 进入 [API 密钥页面](https://modelscope.cn/my/api-tokens)
4. 创建新的 API 令牌
5. 将令牌填入 `.env` 文件中的 `OPENAI_API_KEY` 字段

### 3. 配置示例

```bash
# 必需配置
OPENAI_API_KEY=ms-xxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://api-inference.modelscope.cn/v1
VISION_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct

# 可选配置 (用于调试)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_xxxxxxxx
LANGSMITH_PROJECT=automated_click
```

### 4. 安全注意事项

- 永远不要将 `.env` 文件提交到版本控制系统
- 定期轮换 API 密钥
- 使用环境变量而不是硬编码密钥

### 5. 故障排除

如果遇到 API 错误，请检查：
- API 密钥是否正确
- 账户是否有足够的余额
- 模型名称是否正确
- 网络连接是否正常