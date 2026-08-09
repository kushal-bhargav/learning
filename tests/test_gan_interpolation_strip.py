from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.gan_interpolation_strip import STEPS, latest_checkpoint, lpips_report, save_strip


class FakeLPIPS:
    backend = "fake_lpips"

    def __call__(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (first - second).abs().mean()


def test_latest_checkpoint_selects_highest_run_and_step() -> None:
    tmp_path = Path(".test-tmp/interpolation-checkpoints")
    tmp_path.mkdir(parents=True, exist_ok=True)
    for run, step in [(1, 100), (2, 50), (2, 200)]:
        path = tmp_path / f"run-{run:03d}" / f"checkpoint-{step:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    assert latest_checkpoint(tmp_path) == tmp_path / "run-002" / "checkpoint-000200.pt"


def test_save_strip_has_expected_dimensions() -> None:
    images = [Image.new("RGB", (8, 8), (index * 40, 0, 0)) for index, _ in enumerate(STEPS)]
    tmp_path = Path(".test-tmp/interpolation-strip")
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "strip.png"
    save_strip(images, output)
    with Image.open(output) as image:
        assert image.size == (len(STEPS) * 8, 42)


def test_lpips_report_records_smooth_interpolation(monkeypatch) -> None:
    import scripts.gan_interpolation_strip as strip

    monkeypatch.setattr(strip, "load_lpips", lambda device: FakeLPIPS())
    images = [Image.new("RGB", (8, 8), (index * 20, index * 20, index * 20)) for index, _ in enumerate(STEPS)]
    report = lpips_report(images, torch.device("cpu"))
    assert report["lpips_backend"] == "fake_lpips"
    assert len(report["adjacent_lpips"]) == len(STEPS) - 1
    assert np.isclose(report["mean_adjacent_lpips"], np.mean(report["adjacent_lpips"]))
    assert report["monotonic_from_t0"] is True
    assert report["smoothness_pass"] is True

