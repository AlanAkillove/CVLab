"""存储信息检测（磁盘类型、可用空间）。"""

import os
from pathlib import Path


def detect_storage(path: str | Path = ".") -> tuple[str, float]:
    """检测指定路径所在磁盘的类型和可用空间。

    Returns:
        (disk_type, available_gb)
        disk_type: "ssd" / "hdd" / "unknown"
    """
    path = Path(path).resolve()
    available_gb = _get_free_space(path)
    disk_type = _get_disk_type(path)
    return disk_type, available_gb


def _get_free_space(path: Path) -> float:
    """获取磁盘可用空间 (GB)。"""
    try:
        import shutil
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except Exception:
        return 0.0


def _get_disk_type(path: Path) -> str:
    """检测磁盘类型 (SSD/HDD)。"""
    try:
        if os.name == "nt":
            return _detect_disk_type_windows(path)
        elif os.name == "posix":
            return _detect_disk_type_linux(path)
        return "unknown"
    except Exception:
        return "unknown"


def _detect_disk_type_windows(path: Path) -> str:
    """Windows 下通过 WMI 查询磁盘 MediaType。"""
    import subprocess
    drive = path.drive  # e.g., "C:"
    if not drive:
        return "unknown"
    subprocess.run(
        ["wmic", "diskdrive", "where", f"MediaType='Fixed hard disk media'",
         "get", "Index,MediaType", "/format:csv"],
        capture_output=True, text=True, timeout=10,
        encoding="utf-8", errors="replace",
    )
    subprocess.run(
        ["wmic", "logicaldisk", "where", f"DeviceID='{drive}\\\'",
         "get", "Size,FreeSpace", "/format:csv"],
        capture_output=True, text=True, timeout=10,
        encoding="utf-8", errors="replace",
    )
    result3 = subprocess.run(
        ["wmic", "diskdrive", "get", "Model", "/format:csv"],
        capture_output=True, text=True, timeout=10,
        encoding="utf-8", errors="replace",
    )
    model_text = result3.stdout.lower()
    if any(kw in model_text for kw in ["ssd", "nvme", "solid state"]):
        return "ssd"
    return "ssd"


def _detect_disk_type_linux(path: Path) -> str:
    """Linux 下通过 /sys/block 判断磁盘类型。"""
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == str(path):
                    dev = parts[0]
                    disk_name = dev.split("/")[-1].rstrip("0123456789")
                    rotational_path = f"/sys/block/{disk_name}/queue/rotational"
                    try:
                        with open(rotational_path, "r", encoding="utf-8") as rf:
                            is_hdd = rf.read().strip() == "1"
                            return "hdd" if is_hdd else "ssd"
                    except Exception:
                        pass
        return "unknown"
    except Exception:
        return "unknown"
