import os
import signal
import subprocess
import sys
from pathlib import Path
import psutil

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QTextEdit, QHBoxLayout, QGroupBox, QLineEdit
)
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QFont

MODEL_LIST = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Tongyi-Zhiwen/QwenLong-L1-32B",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/QwQ-32B",
    "deepseek-ai/deepseek-vl2",
    "THUDM/GLM-Z1-Rumination-32B-0414",
    "THUDM/GLM-Z1-32B-0414",
]

class ModelUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("对话模型选择 + 一键启动 FastAPI 服务")
        self.setGeometry(300, 200, 800, 650)
        self.qproc = None

        self.init_ui()
        self.setStyleSheet(self.stylesheet())
        self.load_env_settings()  # 加载默认 API 配置
        self.check_running_status()

    def init_ui(self):
        layout = QVBoxLayout()

        # 模型选择和控制按钮
        top_bar = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_LIST)
        self.model_combo.setMinimumWidth(400)

        self.run_button = QPushButton("▶️ 启动服务")
        self.run_button.clicked.connect(self.run_model)

        self.stop_button = QPushButton("⏹️ 停止服务")
        self.stop_button.clicked.connect(self.stop_service)

        top_bar.addWidget(self.model_combo)
        top_bar.addWidget(self.run_button)
        top_bar.addWidget(self.stop_button)
        layout.addLayout(top_bar)

        # API 配置区域
        api_group = QGroupBox("API 配置")
        api_layout = QVBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("请输入 GRAPHRAG_API_KEY")
        self.api_base_input = QLineEdit()
        self.api_base_input.setPlaceholderText("请输入 GRAPHRAG_API_BASE")
        api_layout.addWidget(QLabel("API Key:"))
        api_layout.addWidget(self.api_key_input)
        api_layout.addWidget(QLabel("API 地址:"))
        api_layout.addWidget(self.api_base_input)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # 状态栏
        self.status_label = QLabel("状态：🔴 服务未运行")
        self.status_label.setFont(QFont("Arial", 11))
        layout.addWidget(self.status_label)

        # 日志输出框
        layout.addWidget(QLabel("日志输出:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier", 10))
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def run_model(self):
        model_name = self.model_combo.currentText()
        api_key = self.api_key_input.text().strip()
        api_base = self.api_base_input.text().strip()
        self.update_env_model(model_name)
        if api_key and api_base:
            self.update_api_settings(api_key, api_base)
        self.kill_existing_service()
        self.start_service()

    def stop_service(self):
        if self.qproc and self.qproc.state() == QProcess.Running:
            self.qproc.kill()
            self.append_log("🛑 服务已手动停止")
        self.kill_existing_service()
        self.status_label.setText("状态：🔴 服务未运行")

    def update_env_model(self, model_name):
        env_path = Path("../graphrag_zh/.env")
        if not env_path.exists():
            QMessageBox.critical(self, "错误", ".env 文件不存在")
            return

        with env_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        with env_path.open("w", encoding="utf-8") as f:
            for line in lines:
                if line.startswith("CHAT_MODEL="):
                    f.write(f"CHAT_MODEL={model_name}\n")
                else:
                    f.write(line)

        self.append_log(f"✅ 模型设置为：{model_name}")

    def load_env_settings(self):
        env_path = Path("../graphrag_zh/.env")
        if not env_path.exists():
            self.append_log("⚠️ 未找到 .env 文件，无法加载 API 设置")
            return

        api_key = ""
        api_base = ""
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("GRAPHRAG_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                elif line.startswith("GRAPHRAG_API_BASE="):
                    api_base = line.strip().split("=", 1)[1]

        self.api_key_input.setText(api_key)
        self.api_base_input.setText(api_base)
        self.append_log("📥 已加载默认 API 配置")

    def update_api_settings(self, api_key, api_base):
        env_path = Path("../graphrag_zh/.env")
        if not env_path.exists():
            QMessageBox.critical(self, "错误", ".env 文件不存在")
            return

        with env_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        with env_path.open("w", encoding="utf-8") as f:
            for line in lines:
                if line.startswith("GRAPHRAG_API_KEY="):
                    f.write(f"GRAPHRAG_API_KEY={api_key}\n")
                elif line.startswith("GRAPHRAG_API_BASE="):
                    f.write(f"GRAPHRAG_API_BASE={api_base}\n")
                else:
                    f.write(line)

        self.append_log("🔐 API Key 和 API 地址已更新！")

    def kill_existing_service(self):
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if "python" in proc.info['name'].lower() and "app.py" in ' '.join(proc.info['cmdline']):
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    self.append_log(f"🔴 结束旧服务 PID: {proc.info['pid']}")
            except Exception:
                continue

    def start_service(self):
        serve_path = Path("../serve/app.py").resolve()
        if not serve_path.exists():
            QMessageBox.critical(self, "错误", "找不到 app.py 文件")
            return
        os.chdir("../serve")
        self.qproc = QProcess()
        self.qproc.setProcessChannelMode(QProcess.MergedChannels)
        self.qproc.readyReadStandardOutput.connect(self.read_output)
        self.qproc.start(sys.executable, [str(serve_path)])

        if not self.qproc.waitForStarted():
            QMessageBox.critical(self, "启动失败", "无法启动服务")
            return

        self.append_log("🟢 服务已启动")
        self.status_label.setText(f"状态：🟢 服务运行中（PID: {self.qproc.processId()}）")

    def read_output(self):
        output = self.qproc.readAllStandardOutput().data().decode("utf-8")
        self.append_log(output)

    def append_log(self, text):
        self.log_output.append(text.strip())

    def check_running_status(self):
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if "python" in proc.info['name'].lower() and "app.py" in ' '.join(proc.info['cmdline']):
                    self.status_label.setText(f"状态：🟢 服务运行中（PID: {proc.info['pid']}）")
                    return
            except Exception:
                continue
        self.status_label.setText("状态：🔴 服务未运行")

    def stylesheet(self):
        return """
        QWidget {
            background-color: #f0f4f8;
            font-family: "Segoe UI", sans-serif;
        }
        QLabel {
            color: #333;
            font-size: 13px;
        }
        QComboBox {
            padding: 6px;
            border: 1px solid #ccc;
            border-radius: 5px;
            min-height: 30px;
        }
        QLineEdit {
            padding: 6px;
            border: 1px solid #ccc;
            border-radius: 5px;
            min-height: 30px;
        }
        QPushButton {
            background-color: #1976d2;
            color: white;
            padding: 6px 15px;
            border-radius: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QPushButton:pressed {
            background-color: #0d47a1;
        }
        QTextEdit {
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ModelUI()
    win.show()
    sys.exit(app.exec_())
