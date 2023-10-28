from datetime import datetime
from enum import Enum
import re

def bold(text):
    return "\033[1m" + text + "\033[0m"


class LogLevel(Enum):
    ERROR = ("ERROR", "🔴")
    WARNING = ("WARNING", "🟠")
    INFO = ("INFO", "🟡")
    DEBUG = ("DEBUG", "🔵")
    SUCCESS = ("SUCCESS", "🟢")


def remove_ansi_escape_codes(message: str):
    ansi_escape = re.compile(r'\033\[[0-9;]+m')
    message_without_ansi = ansi_escape.sub('', message)
    return message_without_ansi


class Logger:
    def __init__(self,
                 level: LogLevel = LogLevel.INFO,
                 active: bool = True,
                 timestamp: bool = True,
                 icon: bool = False,
                 save: bool = False,
                 filename: str = 'log.txt'):
        self.level = level
        self.active = active
        self.timestamp = timestamp
        self.icon = icon
        self.filename = filename
        self.file = open(self.filename, 'a', encoding='utf-8') if save else None

    def log(self, level: LogLevel, message: str, **kwargs):
        if self.active:
            title, icon = level.value
            end = kwargs.get('end', '\n')

            prefix_size = 8
            log_prefix = f"{title}"

            if self.timestamp:
                prefix_size += 22
                timestamp_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log_prefix = f"{timestamp_str} {log_prefix}"

            if self.icon:
                prefix_size += 2
                log_prefix = f"{icon} {log_prefix}"

            log_prefix = log_prefix.ljust(prefix_size - 1)

            log_prefix = f"\033[94m{log_prefix}\033[0m"

            log = f"{log_prefix} | {message}"

            self.write(log, end=end)

            print(log, end=end)

    def error(self, message: str, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)

    def info(self, message: str, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)

    def debug(self, message: str, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)

    def success(self, message: str, **kwargs):
        self.log(LogLevel.SUCCESS, message, **kwargs)

    def disable(self):
        self.active = False

    def enable(self):
        self.active = True

    def write(self, message: str, end='\n'):
        if self.file is not None:
            message_without_ansi = remove_ansi_escape_codes(message)
            self.file.write(f"{message_without_ansi}{end}")
