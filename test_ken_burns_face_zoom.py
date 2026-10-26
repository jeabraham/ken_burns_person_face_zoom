import cv2
import numpy as np
import pytest

import ken_burns_face_zoom as kb


def test_zoom_factor():
    assert kb.zoom_factor(0, 50) == 1
    assert kb.zoom_factor(1, 50) == pytest.approx(1.5)
    assert kb.zoom_factor(0, 50, 20) == pytest.approx(1.2)
    assert kb.zoom_factor(1, 50, 20) == pytest.approx(1.8)


def test_crop_window_stays_inside_and_aspect():
    x, y, w, h = kb.crop_window(1000, 800, (990, 10), 2.0, 16 / 9)
    assert x >= 0 and y >= 0 and x + w <= 1000 and y + h <= 800
    assert w / h == pytest.approx(16 / 9, rel=0.02)


def test_find_images_orders(tmp_path):
    for n in ("b.jpg", "A.png", "c.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in kb.find_images(tmp_path, "name")] == ["A.png", "b.jpg"]
    assert len(kb.find_images(tmp_path, "random", seed=1)) == 2
    assert len(kb.find_images(tmp_path, "date")) == 2


def test_render_and_blur():
    img = np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)
    frames = list(kb.render_frames(img, (250, 150, 80, 80), 5, (320, 180), 40, 10, blur=3))
    assert len(frames) == 5 and frames[0].shape == (180, 320, 3)


def test_main_creates_video(tmp_path):
    img = np.full((300, 400, 3), 128, np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    out = tmp_path / "o.mp4"
    assert kb.main([str(tmp_path), "-o", str(out), "--seconds", "0.5", "--fps", "4",
                    "--width", "160", "--height", "90"]) == 0
    assert out.stat().st_size > 0
