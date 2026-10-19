import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from rrce.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_task_settings_and_overrides(self):
        for task in ("nudity", "dog", "van_gogh"):
            config = load_config(ROOT / "configs" / f"{task}.json")
            self.assertEqual(config.task, task)
            self.assertEqual(config.training.defense_blocks, 3)
            self.assertEqual(len(config.concepts.retain), 20)
        config = load_config(
            ROOT / "configs/dog.json", ["training.optimizer.learning_rate=0.00003"]
        )
        self.assertEqual(config.training.optimizer.learning_rate, 0.00003)
        self.assertEqual(load_config(ROOT / "configs/van_gogh.json").cce.gallery_size, 6)

    def test_unknown_or_invalid_settings_fail(self):
        for override in (
            "training.optimzer.learning_rate=1",
            "training.batch_size=0",
            "runtime.seed=true",
            "ud.betas=[1,0.9]",
            "runtime.precision=fp16",
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(ROOT / "configs/dog.json", [override])

    def test_resolved_configuration_is_portable(self):
        config = load_config(ROOT / "configs/nudity.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(dataclasses.asdict(config)))
            self.assertEqual(load_config(path), config)


if __name__ == "__main__":
    unittest.main()
