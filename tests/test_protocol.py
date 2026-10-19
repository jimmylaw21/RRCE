import dataclasses
import unittest
from pathlib import Path

from rrce.config import load_config
from rrce.data.manifests import SeedAllocator, concept_manifests, validate_gallery_separation


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_seed_allocation_is_stable_and_disjoint(self):
        first = SeedAllocator("protocol")
        train = first.allocate("gallery", 1000)
        test = first.allocate("test", 1000)
        second = SeedAllocator("protocol")
        self.assertEqual(train, second.allocate("gallery", 1000))
        self.assertFalse(set(train) & set(test))

    def test_task_manifest_sizes_and_gallery_isolation(self):
        for task in ("dog", "van_gogh"):
            config = load_config(ROOT / "configs" / f"{task}.json")
            manifests = concept_manifests(config, SeedAllocator(config.protocol.seed_namespace))
            for split in ("erased", "ud", "cce"):
                self.assertEqual(len(manifests[split]), config.evaluation.counts[split])
            validate_gallery_separation(config, manifests["gallery"], manifests["cce"])
            damaged = [dict(row) for row in manifests["gallery"]]
            damaged[0]["seed"] = manifests["cce"][0]["seed"]
            with self.assertRaises(ValueError):
                validate_gallery_separation(config, damaged, manifests["cce"])


if __name__ == "__main__":
    unittest.main()
