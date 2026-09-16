# Omni AI Controller

用于管理本机 `omni_ai_model` Docker 服务并与模型进行多轮对话的交互式控制台。

## 功能

- 首次运行时填写并记住 `omni_ai_model` 仓库目录。
- 后台启动模型容器并加载模型。
- 停止模型但保留控制器容器。
- 查看 Docker 容器、模型 readiness 和 GPU 状态。
- 开启新对话、选择是否启用思考模式、发送多轮消息。
- 查看最近一次模型 output 和当前对话历史。
- 打开容器 Shell，或停止整个容器。

程序只保存模型仓库路径。API 密钥始终直接读取模型仓库中的 `.env`，不会复制到控制器配置中。

## 运行

要求 Python 3.10 或更高版本。运行时仅使用 Python 标准库，无需安装额外依赖。

```bash
cd /opt/ai_server/omni_ai_controller
python3 -m omni_ai_controller
```

也可以直接运行入口脚本：

```bash
cd /opt/ai_server/omni_ai_controller
python3 run.py
```

首次运行会提示：

```text
请输入 omni_ai_model 目录 [/opt/ai_server/omni_ai_model]：
```

也可以显式指定目录：

```bash
python3 -m omni_ai_controller --model-dir /opt/ai_server/omni_ai_model
```

选择“启动容器并加载模型”后：

1. 如果控制器容器尚未运行，会调用模型仓库的部署脚本，以后台模式创建容器。
2. 重新读取部署脚本生成的 `.env`。
3. 通过受保护的控制 API 启动 vLLM 模型进程。
4. 首次启动时，模型权重由 vLLM 下载到模型仓库配置的 Docker 缓存卷。

退出本控制台或断开 SSH 不会停止后台容器。

## 可选安装

如果希望使用全局命令，可以在虚拟环境中安装：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
omni-ai-controller
```

## 测试

```bash
python3 -m pytest -q
```
