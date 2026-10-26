import cv2
import numpy as np
import pytest

import ken_burns_face_zoom as kb


def test_zoom_factor():
    assert kb.zoom_factor(0, 50) == 1
    assert kb.zoom_factor(1, 50) == pytest.approx(1.5)
    assert kb.zoom_factor(0, 50, 2.0) == pytest.approx(2.0)
    assert kb.zoom_factor(1, 50, 2.0) == pytest.approx(3.0)


def test_initial_zoom_depends_on_face_size():
    out = (1280, 720)
    # solo portrait: face already fills >25% of frame -> no initial zoom
    assert kb.initial_zoom((4000, 3000), (1500, 800, 900, 1000), out, 25, 1.5) == 1.0
    # distant face in a group shot -> zoomed in so the face is ~25% of frame height
    z = kb.initial_zoom((4000, 3000), (1500, 800, 200, 200), out, 25, 1.5)
    assert z == pytest.approx(0.25 * 2250 / 200)
    assert kb.initial_zoom((4000, 3000), (1500, 800, 100, 100), out, 0, 1.5) == 1.0


def test_initial_zoom_capped_by_resolution():
    # 1600x900 source -> output 1280x720: at most 1.5x upscale => crop >= 853px wide
    z = kb.initial_zoom((1600, 900), (800, 400, 10, 10), (1280, 720), 25, 1.5)
    assert z == pytest.approx(1600 * 1.5 / 1280)
    assert kb.initial_zoom((1600, 900), (800, 400, 10, 10), (1280, 720), 25, 3) == pytest.approx(1600 * 3 / 1280)


def test_cluster_and_pick():
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    thumb = np.zeros((4, 4, 3), np.uint8)
    clusters = kb.cluster_faces([(thumb, a, 1), (thumb, b, 1), (thumb, a * 2, 5)])
    assert [len(c["features"]) for c in clusters] == [2, 1]
    assert clusters[0]["size"] == 5
    feats = kb.choose_person(clusters, person=2)
    box = kb.pick_face([((0, 0, 5, 5), a), ((9, 9, 5, 5), b)], feats)
    assert box == (9, 9, 5, 5)
    assert kb.pick_face([((0, 0, 5, 5), a)], feats) is None
    with pytest.raises(SystemExit):
        kb.choose_person(clusters, person=3)


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
    frames = list(kb.render_frames(img, (250, 150, 80, 80), 5, (320, 180), 40, 1.2, blur=3))
    assert len(frames) == 5 and frames[0].shape == (180, 320, 3)


def test_main_creates_video(tmp_path, monkeypatch):
    feat = np.array([1.0, 0.0])
    monkeypatch.setattr(kb, "detect_faces", lambda img, *a, **k: [((150, 100, 60, 60), feat)])
    img = np.full((300, 400, 3), 128, np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    out = tmp_path / "o.mp4"
    assert kb.main([str(tmp_path), "-o", str(out), "--seconds", "0.5", "--fps", "4",
                    "--width", "160", "--height", "90", "--person", "1",
                    "--people-sheet", str(tmp_path / "people.jpg")]) == 0
    assert out.stat().st_size > 0


def test_find_person_escalates_and_missing_excluded(tmp_path, monkeypatch):
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    calls = []

    def fake_detect(img, score=0.8, max_side=1280, flip=False):
        calls.append(flip)
        return [((1, 2, 3, 4), a)] if flip else [((5, 6, 7, 8), b)]

    monkeypatch.setattr(kb, "detect_faces", fake_detect)
    img = np.zeros((10, 10, 3), np.uint8)
    assert kb.find_person(img, [a]) == (1, 2, 3, 4)
    assert True in calls and calls[0] is False
    assert kb.find_person(img, [a], aggressive=False) is None

    cv2.imwrite(str(tmp_path / "a.png"), np.full((300, 400, 3), 128, np.uint8))
    out = tmp_path / "o.mp4"
    args = [str(tmp_path), "-o", str(out), "--seconds", "0.5", "--fps", "4", "--width", "160",
            "--height", "90", "--person", "1", "--people-sheet", str(tmp_path / "p.jpg")]
    monkeypatch.setattr(kb, "scan_people", lambda paths, n: [{"features": [a], "thumb": img, "size": 1}])
    monkeypatch.setattr(kb, "find_person", lambda *x, **k: None)
    kb.main(args)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0
    kb.main(args + ["--keep-missing"])
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
