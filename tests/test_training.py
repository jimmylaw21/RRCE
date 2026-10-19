import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from rrce.config import load_config
from rrce.io import write_manifest
from rrce.models.diffusion import DiffusionModel, select_trainable_parameters
from rrce.training.trainer import RRCETrainer
from tests.tiny_models import make_tiny_pipeline


ROOT = Path(__file__).resolve().parents[1]


class TrainingIntegrationTests(unittest.TestCase):
    def prepare_inputs(self, root):
        config = load_config(
            ROOT / "configs/dog.json",
            [
                "runtime.device=cpu",
                "runtime.precision=fp32",
                "runtime.num_workers=0",
                "runtime.log_every=1",
                "model.resolution=16",
                "training.pairs_per_block=2",
                "recovery.steps=2",
                "recovery.repeats=1",
            ],
        )
        targets = root / "targets"
        retains = root / "retains"
        targets.mkdir()
        retains.mkdir()
        for index in range(4):
            Image.new("RGB", (16, 16), color=(index * 40, 60, 120)).save(targets / f"{index}.png")
        rows = []
        for index, concept in enumerate([*config.concepts.retain, ""]):
            path = retains / f"{index}.pt"
            torch.save(torch.full((4, 8, 8), index / 100), path)
            rows.append({"prompt": concept, "image_path": path.name})
        write_manifest(retains / "retain.csv", rows)
        return dataclasses.replace(
            config,
            concepts=dataclasses.replace(config.concepts, initializer="a"),
            data=dataclasses.replace(
                config.data, target_images=str(targets), retain_manifest=str(retains / "retain.csv")
            ),
            training=dataclasses.replace(config.training, output=str(root / "training")),
        )

    def test_parameter_scope_excludes_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = make_tiny_pipeline(directory)
            full, names = select_trainable_parameters(pipeline.unet, "full")
            self.assertTrue(full)
            self.assertFalse(any("norm" in name for name in names))
            cross, names = select_trainable_parameters(pipeline.unet, "cross_attention")
            self.assertTrue(cross)
            self.assertTrue(all("attn2" in name for name in names))

    def test_full_training_and_block_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.prepare_inputs(root)
            model = DiffusionModel(make_tiny_pipeline(root / "tokenizer"), config.runtime)
            trainer = RRCETrainer(config, model)
            teacher_before = {
                key: value.clone() for key, value in trainer.teacher.state_dict().items()
            }
            with patch(
                "rrce.training.trainer.ConceptRecovery.discover",
                side_effect=RuntimeError("interrupted"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    trainer.run()
            checkpoint = root / "training/latest.pt"
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(state["next_block"], 1)
            for key, value in trainer.teacher.state_dict().items():
                torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
            resumed_model = DiffusionModel(
                make_tiny_pipeline(root / "tokenizer_resume"), config.runtime
            )
            resumed = RRCETrainer(config, resumed_model)
            resumed.run(resume=checkpoint)
            self.assertEqual(resumed.optimizer.erase.statistics()["steps"], 6)
            self.assertEqual(resumed.optimizer.retain.statistics()["steps"], 6)
            tokens = torch.load(root / "training/recovery_tokens.pt", weights_only=True)
            self.assertEqual(set(tokens), {"<rrce_1>", "<rrce_2>"})
            exported = torch.load(root / "training/unet.pt", weights_only=True)
            self.assertEqual(set(exported), set(resumed_model.pipeline.unet.state_dict()))
            for key, value in resumed.teacher.state_dict().items():
                torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
            self.assertTrue(
                any(not torch.equal(exported[key], teacher_before[key]) for key in exported)
            )


if __name__ == "__main__":
    unittest.main()
