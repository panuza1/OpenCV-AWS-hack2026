import csv

import cv2

from scripts.generate_sample import FPS, FRAME_COUNT, generate_samples


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_samples_are_reproducible_and_reopen(tmp_path):
    assert cv2.__version__.startswith("5.")
    first = generate_samples(tmp_path / "first")
    second = generate_samples(tmp_path / "second")

    assert [sample["scenario"] for sample in first] == ["constant_speed", "slowing_down"]
    for sample, repeated in zip(first, second):
        rows = _read_rows(sample["ground_truth"])
        times = [float(row["time_s"]) for row in rows]
        assert len(rows) == FRAME_COUNT
        assert times == [frame / FPS for frame in range(FRAME_COUNT)]
        assert open(sample["ground_truth"], "rb").read() == open(
            repeated["ground_truth"], "rb"
        ).read()

        capture = cv2.VideoCapture(sample["video"])
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAME_COUNT
        assert abs(capture.get(cv2.CAP_PROP_FPS) - FPS) < 0.01
        capture.release()

    constant_x = [int(row["x_px"]) for row in _read_rows(first[0]["ground_truth"])]
    slowing_x = [int(row["x_px"]) for row in _read_rows(first[1]["ground_truth"])]
    constant_steps = [b - a for a, b in zip(constant_x, constant_x[1:])]
    slowing_steps = [b - a for a, b in zip(slowing_x, slowing_x[1:])]
    assert min(constant_steps) >= 1 and max(constant_steps) <= 2
    assert sum(slowing_steps[:30]) > sum(slowing_steps[-30:])

