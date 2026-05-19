"""环境检测模块测试。"""

from cvlab.detect.cpu_info import detect_cpu
from cvlab.detect.gpu_info import detect_gpus, get_recommended_num_workers
from cvlab.detect.os_info import detect_os, is_wsl
from cvlab.detect.probe import EnvironmentProbe


class TestOSInfo:
    def test_detect_os(self):
        os_type, os_version = detect_os()
        assert os_type in ("Windows", "Linux", "macOS")
        assert isinstance(os_version, str)

    def test_is_wsl(self):
        # 只是验证调用不报错
        assert isinstance(is_wsl(), bool)


class TestCPUInfo:
    def test_detect_cpu(self):
        model, cores, threads = detect_cpu()
        assert cores > 0
        assert threads >= cores
        assert isinstance(model, str)


class TestGPUInfo:
    def test_detect_gpus(self):
        gpus = detect_gpus()
        assert isinstance(gpus, list)

    def test_recommended_num_workers(self):
        n = get_recommended_num_workers()
        assert n >= 2
        assert n <= 16


class TestEnvironmentProbe:
    def test_probe_returns_report(self):
        probe = EnvironmentProbe()
        report = probe.probe()
        assert report.os_type in ("Windows", "Linux", "macOS")
        assert report.python_version != ""
        assert report.torch_version != ""
        assert report.cpu_cores > 0
        assert report.total_ram_gb > 0

    def test_print_report(self):
        probe = EnvironmentProbe()
        report = probe.probe()
        text = probe.print_report(report)
        assert "环境探测报告" in text
        assert report.os_type in text

    def test_acceleration_panel(self):
        probe = EnvironmentProbe()
        report = probe.probe()
        panel = probe.get_acceleration_panel(report)
        assert len(panel.options) > 0
        names = [o.name for o in panel.options]
        assert "AMP FP16" in names
        assert panel.recommended_num_workers >= 2

    def test_print_panel(self):
        probe = EnvironmentProbe()
        report = probe.probe()
        panel = probe.get_acceleration_panel(report)
        text = probe.print_panel(panel)
        assert "训练加速配置" in text
        assert "DataLoader 配置" in text

    def test_cuda_mismatch_check(self):
        from cvlab.detect.gpu_info import check_cuda_mismatch
        mismatch, info = check_cuda_mismatch()
        assert isinstance(mismatch, bool)
