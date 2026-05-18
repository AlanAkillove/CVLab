"""预测可视化模块测试。"""

import numpy as np
import torch

from cvlab.diagnose.prediction import (
    classification_samples,
    detection_overlay,
    segmentation_overlay,
    _get_color,
)


class TestClassificationSamples:
    def test_returns_array(self):
        images = torch.rand(4, 3, 32, 32)
        result = classification_samples(
            images, [0, 1, 0, 1], [0, 1, 0, 1],
            ["cat", "dog"], max_samples=4,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape[-1] == 3  # RGB
        assert result.dtype == np.uint8

    def test_max_samples(self):
        images = torch.rand(10, 3, 32, 32)
        result = classification_samples(
            images, [0]*10, [0]*10, ["cat"], max_samples=3,
        )
        h = result.shape[0]
        assert h > 0

    def test_empty_input(self):
        result = classification_samples(
            torch.rand(0, 3, 32, 32), [], [], [], max_samples=0,
        )
        assert isinstance(result, np.ndarray)


class TestDetectionOverlay:
    def test_draws_boxes(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        labels = np.array([0])

        result = detection_overlay(image, boxes, scores, labels, ["obj"])
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_score_threshold(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.3])
        labels = np.array([0])

        result = detection_overlay(image, boxes, scores, labels, ["obj"], score_threshold=0.5)
        assert result.shape == (100, 100, 3)

    def test_multi_boxes(self):
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100], [20, 80, 80, 180]])
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([0, 1, 2])

        result = detection_overlay(image, boxes, scores, labels, ["a", "b", "c"])
        assert result.shape == (200, 200, 3)


class TestSegmentationOverlay:
    def test_pred_only(self):
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[16:48, 16:48] = 1

        result = segmentation_overlay(image, mask, num_classes=3)
        assert result.shape == (64, 64, 3)

    def test_with_gt(self):
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128
        pred = np.zeros((64, 64), dtype=np.int32)
        gt = np.ones((64, 64), dtype=np.int32)

        result = segmentation_overlay(image, pred, gt_mask=gt, num_classes=3)
        # With GT, result should be wider (concatenated)
        assert result.shape[1] > 64


class TestColor:
    def test_get_color(self):
        color = _get_color(0)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_color_cycle(self):
        c0 = _get_color(0)
        c22 = _get_color(22)
        assert c0 == c22  # 循环
