#!/usr/bin/env python3
"""Mutation checks for architecture-specific replacement inventories."""
from pathlib import Path
import tempfile
import unittest

from check_target_features_api import ROOT, TARGETS, check_surface, read_surface


class TargetSurfaceGuard(unittest.TestCase):
    def test_complete_inventory_for_every_target(self):
        for target, count in zip(TARGETS, [64, 28, 28], strict=True):
            with self.subTest(target=target):
                items = read_surface(ROOT / "docs/public-api" / target, target)
                self.assertEqual(check_surface(items, target), count)

    def test_missing_replacement_is_rejected_on_every_target(self):
        for target, (_, tier, _) in TARGETS.items():
            with self.subTest(target=target):
                items = read_surface(ROOT / "docs/public-api" / target, target)
                items.remove(f"pub fn target_features::x4::srgb_to_linear_{tier}([f32; 4]) -> [f32; 4]")
                with self.assertRaisesRegex(AssertionError, "missing replacements"):
                    check_surface(items, target)

    def test_extra_function_or_x16_module_is_rejected(self):
        for target in TARGETS:
            for extra in ["pub fn target_features::x4::unexpected_scalar([f32; 4]) -> [f32; 4]", "pub mod target_features::x16"]:
                with self.subTest(target=target, extra=extra):
                    items = read_surface(ROOT / "docs/public-api" / target, target)
                    items.add(extra)
                    with self.assertRaises(AssertionError):
                        check_surface(items, target)

    def test_host_capture_cannot_substitute_for_arm(self):
        with self.assertRaisesRegex(AssertionError, "wrong architecture"):
            read_surface(ROOT / "docs/public-api/x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu")

    def test_missing_capture_fails_loudly(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            with self.assertRaisesRegex(AssertionError, "Missing"):
                read_surface(Path(directory), "wasm32-wasip1")


if __name__ == "__main__":
    unittest.main()
