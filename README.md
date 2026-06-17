# 学习助手 Chatbot

这是一个基于 Streamlit、LangChain 和 DeepSeek Chat API 的学习助手项目。用户可以在网页中选择学科领域和讲解风格，然后以聊天形式向 AI 提问，系统会结合上下文记忆持续回答学习问题。

## 项目功能

- 支持网页聊天交互，使用 Streamlit 快速搭建对话页面。
- 支持学科选择：文学、数学、计算机。
- 支持讲解风格选择：简洁、详细。
- 使用 LangChain 的 `ConversationBufferMemory` 保存多轮对话上下文。
- 使用 DeepSeek 的 OpenAI 兼容接口调用大模型。
- 通过提示词约束 AI 角色，使回答更贴合所选学科。

## 项目结构

```text
chatbot/
├── chatbot/
│   ├── streamlit_app.py      # Streamlit 主程序
│   └── requirements.txt      # Python 依赖
└── README.md                 # 项目说明文档
```

## 技术栈

- Python
- Streamlit
- LangChain
- langchain-openai
- DeepSeek Chat API

## 运行方式

1. 克隆仓库

```bash
git clone https://github.com/illusion-error/chatbot.git
cd chatbot/chatbot
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 配置 API Key

打开 `streamlit_app.py`，将下面位置的 `API_KEY` 替换为自己的 DeepSeek API Key：

```python
client = ChatOpenAI(
    api_key="API_KEY",
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=0.0,
)
```

4. 启动项目

```bash
streamlit run streamlit_app.py
```

启动后，浏览器会打开本地页面。在左侧选择学科和讲解风格，在输入框中提出学习问题即可。

## 使用示例

- 选择“数学” + “简洁”：适合快速获取公式、计算步骤或结论。
- 选择“计算机” + “详细”：适合解释代码、算法、概念和报错原因。
- 选择“文学” + “详细”：适合分析作品、人物、主题和写作手法。

## 注意事项

- 当前代码中 API Key 需要手动替换，正式项目建议改为 `.env` 环境变量配置，避免把密钥写进代码。
- 如果模型没有响应，请检查 API Key、网络连接和 DeepSeek 账户额度。
- 当前项目是一个轻量级学习助手 Demo，适合用于课程展示、AI 应用入门和 LangChain 基础练习。

## 后续可优化方向

- 增加 `.env` 配置文件，避免硬编码 API Key。
- 增加异常处理，例如额度不足、网络失败、模型超时等提示。
- 增加更多学科，例如英语、物理、历史、考研、编程面试等。
- 增加聊天记录导出功能。
- 增加用户上传资料后的问答能力，升级为 RAG 学习助手。
