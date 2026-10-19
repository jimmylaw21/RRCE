import copy
import math
import unittest

import torch

from rrce.optim.adaptive_adamw import AdaptiveMomentumAdamW
from rrce.training.objectives import anchor_guided_target


class OptimizationTests(unittest.TestCase):
    def make_optimizer(self, parameter):
        return AdaptiveMomentumAdamW(
            [parameter], lr=0.01, beta_min=0.1, beta_max=0.9, beta2=0.999, eps=1e-8, weight_decay=0
        )

    def test_alternating_objectives_have_independent_history(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        optimizers = {name: self.make_optimizer(parameter) for name in ("erase", "retain")}
        expected = parameter.detach().clone()
        states = {
            name: {
                "previous": None,
                "first": torch.zeros(2, dtype=torch.float64),
                "second": torch.zeros(2, dtype=torch.float64),
                "product": 1.0,
                "step": 0,
            }
            for name in optimizers
        }
        gradients = {
            "erase": [[1, 2], [-1, -2], [0, 0], [2, -1]],
            "retain": [[-2, 1], [-2, 1], [1, -2], [0, 0]],
        }
        for index in range(4):
            for name, optimizer in optimizers.items():
                gradient = torch.tensor(gradients[name][index], dtype=torch.float64)
                state = states[name]
                previous = state["previous"]
                if previous is None:
                    beta = 0.9
                else:
                    norm = gradient.norm() * previous.norm()
                    cosine = float(gradient.dot(previous) / norm) if norm else 0.0
                    beta = 0.1 + 0.8 * (cosine + 1) / 2
                state["step"] += 1
                state["product"] *= beta
                state["first"] = beta * state["first"] + (1 - beta) * gradient
                state["second"] = 0.999 * state["second"] + 0.001 * gradient.square()
                corrected_first = state["first"] / (1 - state["product"])
                corrected_second = state["second"] / (1 - 0.999 ** state["step"])
                expected -= 0.01 * corrected_first / (corrected_second.sqrt() + 1e-8)
                state["previous"] = gradient
                parameter.grad = gradient
                optimizer.step()
                torch.testing.assert_close(parameter, expected, rtol=1e-8, atol=1e-9)
        self.assertNotEqual(
            optimizers["erase"].statistics()["rho_mean"],
            optimizers["retain"].statistics()["rho_mean"],
        )

    def test_checkpoint_restores_gradient_history(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = self.make_optimizer(parameter)
        parameter.grad = torch.tensor([2.0, -1.0])
        optimizer.step()
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = self.make_optimizer(restored_parameter)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for gradient in (torch.tensor([-2.0, 1.0]), torch.zeros(2), torch.ones(2)):
            parameter.grad = gradient.clone()
            restored_parameter.grad = gradient.clone()
            optimizer.step()
            restored.step()
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
        self.assertEqual(optimizer.statistics(), restored.statistics())

    def test_guidance_reduces_to_standard_residual(self):
        torch.manual_seed(3)
        unconditional, target, anchor = torch.randn(3, 2, 4, 8, 8, dtype=torch.float64)
        actual = anchor_guided_target(unconditional, target, anchor, 4.0, 1.0)
        expected = unconditional + 3 * (anchor - target)
        torch.testing.assert_close(actual, expected)
        projected = anchor_guided_target(unconditional, target, anchor, 4.0, 0.0)
        self.assertFalse(torch.allclose(actual, projected))


if __name__ == "__main__":
    unittest.main()
