from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .client import ChatResult, ServerRequestError
from .config import (
    ConfigurationError,
    load_saved_model_dir,
    save_model_dir,
    validate_model_dir,
)
from .conversation import Conversation
from .runtime import LocalModelServer, RuntimeCommandError


class ConsoleApp:
    def __init__(self, model_dir: Path) -> None:
        self.server = LocalModelServer(model_dir)
        self.conversation = Conversation()

    def run(self) -> None:
        print("\nOmni AI 控制台")
        print(f"模型服务目录：{self.server.config.model_dir}")
        while True:
            self._print_menu()
            try:
                choice = input("请选择操作：").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出；后台容器和模型不会被停止。")
                return

            action = self._actions().get(choice)
            if action is None:
                print("无效选项，请重新输入。")
                continue
            if choice in {"0", "q"}:
                action()
                return
            try:
                action()
            except (ConfigurationError, RuntimeCommandError, ServerRequestError, ValueError) as exc:
                print(f"\n错误：{exc}")
            except KeyboardInterrupt:
                print("\n操作已中断；后台容器不会因此停止。")

    def _actions(self) -> dict[str, Callable[[], None]]:
        return {
            "1": self._start,
            "2": self._stop,
            "3": self._status,
            "4": self._new_conversation,
            "5": self._send,
            "6": self._show_output,
            "7": self._show_history,
            "8": self._open_shell,
            "9": self._stop_container,
            "0": self._exit,
            "q": self._exit,
        }

    @staticmethod
    def _print_menu() -> None:
        print(
            """
[1] 启动容器并加载模型
[2] 停止模型（保留容器）
[3] 查看容器和模型状态
[4] 开启新对话
[5] 发送对话消息
[6] 获取最近一次 output
[7] 查看当前对话历史
[8] 打开容器 Shell
[9] 停止整个容器
[0] 退出控制台
""".rstrip()
        )

    def _start(self) -> None:
        print("\n正在启动；首次运行可能需要构建镜像和下载模型……")
        data = self.server.start()
        self._print_json(data)
        if not data.get("ready"):
            print("模型正在后台加载，可稍后使用状态功能检查 readiness。")

    def _stop(self) -> None:
        data = self.server.stop()
        self._print_json(data)

    def _status(self) -> None:
        compose_status, api_status, api_error = self.server.status()
        print("\n容器状态：")
        print(compose_status or "未找到运行中的容器。")
        print("\n模型状态：")
        if api_status is not None:
            self._print_json(api_status)
        else:
            print(api_error or "模型控制 API 不可用。")

    def _new_conversation(self) -> None:
        system_prompt = input("系统提示词（可留空）：").strip()
        thinking_answer = input("启用思考模式？[Y/n]：").strip().lower()
        self.conversation.reset(
            system_prompt,
            enable_thinking=thinking_answer not in {"n", "no", "否"},
        )
        print("新对话已创建。")

    def _send(self) -> None:
        text = input("User> ")
        self.server.refresh()
        print("正在生成，请稍候……")
        result = self.conversation.send(self.server.client, text)
        self._print_chat_result(result)

    def _show_output(self) -> None:
        result = self.conversation.last_result
        if result is None:
            print("当前还没有模型 output。")
            return
        self._print_chat_result(result)

    def _show_history(self) -> None:
        if not self.conversation.messages:
            print("当前对话为空。")
            return
        print()
        for message in self.conversation.messages:
            role = message["role"].capitalize()
            print(f"{role}> {message['content']}")

    def _open_shell(self) -> None:
        print("进入容器 Shell；执行 exit 可返回控制台。")
        self.server.open_shell()

    def _stop_container(self) -> None:
        answer = input("确认停止整个模型容器？[y/N]：").strip().lower()
        if answer not in {"y", "yes", "是"}:
            print("已取消。")
            return
        self.server.stop_container()
        print("容器已停止；模型缓存卷仍然保留。")

    @staticmethod
    def _exit() -> None:
        print("已退出；后台容器和模型继续按当前状态运行。")

    @staticmethod
    def _print_json(data: dict[str, Any]) -> None:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))

    @staticmethod
    def _print_chat_result(result: ChatResult) -> None:
        if result.reasoning_content:
            print(f"\nReasoning>\n{result.reasoning_content}")
        print(f"\nAssistant>\n{result.content or '[空响应]'}")


def choose_model_dir(requested: str | None) -> Path:
    if requested:
        model_dir = validate_model_dir(Path(requested))
        save_model_dir(model_dir)
        return model_dir

    saved = load_saved_model_dir()
    sibling = Path(__file__).resolve().parents[2] / "omni_ai_model"
    default = saved or sibling

    while True:
        try:
            answer = input(f"请输入 omni_ai_model 目录 [{default}]：").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise ConfigurationError("未提供模型服务目录") from exc
        candidate = Path(answer).expanduser() if answer else default
        try:
            model_dir = validate_model_dir(candidate)
        except ConfigurationError as exc:
            print(f"错误：{exc}")
            continue
        save_model_dir(model_dir)
        return model_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Omni AI 模型服务交互式控制台")
    parser.add_argument("--model-dir", help="omni_ai_model 仓库目录")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        model_dir = choose_model_dir(args.model_dir)
        ConsoleApp(model_dir).run()
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
