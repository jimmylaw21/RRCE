import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from rrce.attacks.cce import ConceptEmbeddingAttack
from rrce.attacks.unlearndiff import (
    UnlearnDiffAttack,
    encode_token_embeddings,
    project_probability_simplex,
)
from rrce.config import load_config
from rrce.io import MetricLogger
from tests.tiny_models import make_tiny_pipeline


ROOT = Path(__file__).resolve().parents[1]


class AttackTests(unittest.TestCase):
    def test_simplex_projection(self):
        probabilities = project_probability_simplex(
            torch.tensor([[-10.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), 32
        )
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
        self.assertTrue((probabilities >= 0).all())
        torch.testing.assert_close(probabilities[0], torch.tensor([0.0, 0.0, 1.0]))

    def test_embedding_encoder_matches_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = make_tiny_pipeline(directory)
            ids = pipeline.tokenizer(
                "a dog", padding="max_length", max_length=16, return_tensors="pt"
            ).input_ids
            embeddings = pipeline.text_encoder.get_input_embeddings()(ids)
            actual = encode_token_embeddings(pipeline.text_encoder, embeddings)
            expected = pipeline.text_encoder(ids).last_hidden_state
            torch.testing.assert_close(actual, expected)

    def test_cce_and_ud_have_real_gradient_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(
                ROOT / "configs/dog.json",
                [
                    "runtime.device=cpu",
                    "runtime.precision=fp32",
                    "runtime.log_every=1",
                    "model.resolution=16",
                    "cce.steps=2",
                    "cce.sampling.precision=fp32",
                    "ud.timesteps=[10]",
                    "ud.iterations_per_timestep=2",
                ],
            )
            config = dataclasses.replace(
                config, concepts=dataclasses.replace(config.concepts, initializer="a")
            )
            pipeline = make_tiny_pipeline(root / "tokenizer")
            frozen_embeddings = pipeline.text_encoder.get_input_embeddings().weight.detach().clone()
            image = root / "gallery.png"
            Image.new("RGB", (18, 16), color=(40, 80, 120)).save(image)
            cce = ConceptEmbeddingAttack(pipeline, config, MetricLogger(root / "cce.jsonl"))
            learned = cce.fit([image])
            torch.testing.assert_close(
                pipeline.text_encoder.get_input_embeddings().weight[:-1],
                frozen_embeddings,
                rtol=0,
                atol=0,
            )
            initial = frozen_embeddings[
                pipeline.tokenizer("a", add_special_tokens=False).input_ids[0]
            ]
            self.assertFalse(torch.equal(learned, initial))
            ud = UnlearnDiffAttack(pipeline, config, MetricLogger(root / "ud.jsonl"))
            condition, tokens, text_length = ud.optimize(
                {"sample_id": "dog_ud_000", "prompt": "a dog"}, torch.randn(1, 4, 8, 8)
            )
            self.assertEqual(condition.shape, (1, 16, 32))
            self.assertEqual(len(tokens), config.ud.prefix_tokens)
            self.assertTrue(torch.isfinite(condition).all())
            self.assertLessEqual(text_length + len(tokens), 14)


if __name__ == "__main__":
    unittest.main()
