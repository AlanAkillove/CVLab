"""CPU 信息检测。"""

import os


def detect_cpu() -> tuple[str, int, int]:
    """返回 (model_name, physical_cores, logical_threads)。"""
    import psutil

    model = _get_cpu_model()
    cores = psutil.cpu_count(logical=False) or os.cpu_count() or 0
    threads = psutil.cpu_count(logical=True) or cores
    return model, cores, threads


def _get_cpu_model() -> str:
    """获取 CPU 型号名称。"""
    try:
        if os.name == "posix":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        elif os.name == "nt":
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5,
                encoding="utf-8", errors="replace",
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                return lines[1].strip()
        return ""
    except Exception:
        return ""
