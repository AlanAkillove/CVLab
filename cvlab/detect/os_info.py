"""OS 信息检测。"""

import platform


def detect_os() -> tuple[str, str]:
    """返回 (os_type, os_version)。"""
    system = platform.system()
    if system == "Windows":
        return "Windows", platform.version()
    elif system == "Linux":
        return "Linux", platform.release()
    elif system == "Darwin":
        return "macOS", platform.mac_ver()[0]
    return system, platform.version()


def is_wsl() -> bool:
    """检测是否运行在 WSL2 环境。"""
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False
