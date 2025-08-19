import logging
from colorlog import ColoredFormatter

# 创建带颜色的 Formatter（用于控制台）
console_formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)


# 创建 StreamHandler（控制台）并设置带颜色的 Formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler],  # 同时输出到控制台和文件
)
