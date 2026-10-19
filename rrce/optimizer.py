from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch


class AdaptiveAdamW(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.nn.Parameter], *, lr: float, beta_min: float, beta_max: float, beta2: float, eps: float, weight_decay: float) -> None:
        if not 0.0 <= float(beta_min) <= float(beta_max) < 1.0:
            raise ValueError('Expected 0 <= beta_min <= beta_max < 1.')
        if not 0.0 <= float(beta2) < 1.0:
            raise ValueError('Expected 0 <= beta2 < 1.')
        defaults = {'lr': float(lr), 'beta_min': float(beta_min), 'beta_max': float(beta_max), 'beta2': float(beta2), 'eps': float(eps), 'weight_decay': float(weight_decay)}
        super().__init__(params, defaults)
        self._step_count = 0
        self._has_previous_gradient = False
        self._last_rho: Optional[float] = None
        self._last_beta1 = float(beta_max)
        self._rho_count = 0
        self._rho_sum = 0.0
        self._rho_min: Optional[float] = None
        self._rho_max: Optional[float] = None
        self._beta1_sum = 0.0
        self._beta1_min: Optional[float] = None
        self._beta1_max: Optional[float] = None
        self._bootstrap_steps = 0

    @torch.no_grad()
    def _adjacent_gradient_cosine(self) -> Optional[float]:
        if not self._has_previous_gradient:
            return None
        dot = None
        current_sq = None
        previous_sq = None
        for group in self.param_groups:
            for parameter in group['params']:
                gradient = parameter.grad
                previous = self.state[parameter].get('previous_grad')
                if gradient is None or previous is None:
                    continue
                current = gradient.detach().float()
                old = previous.detach().float()
                local_dot = torch.sum(current * old)
                local_current_sq = torch.sum(current * current)
                local_previous_sq = torch.sum(old * old)
                if dot is None:
                    dot = local_dot
                    current_sq = local_current_sq
                    previous_sq = local_previous_sq
                else:
                    dot.add_(local_dot)
                    current_sq.add_(local_current_sq)
                    previous_sq.add_(local_previous_sq)
        if dot is None:
            return None
        denominator = torch.sqrt(current_sq) * torch.sqrt(previous_sq)
        if float(denominator.item()) == 0.0:
            return 0.0
        return float(torch.clamp(dot / denominator, -1.0, 1.0).item())

    def _select_beta1(self, rho: Optional[float]) -> float:
        group = self.param_groups[0]
        beta_min = float(group['beta_min'])
        beta_max = float(group['beta_max'])
        if rho is None:
            self._bootstrap_steps += 1
            return beta_max
        scaled = min(max((float(rho) + 1.0) / 2.0, 0.0), 1.0)
        return beta_min + (beta_max - beta_min) * scaled

    def _record(self, rho: Optional[float], beta1: float) -> None:
        self._last_rho = rho
        self._last_beta1 = float(beta1)
        self._beta1_sum += float(beta1)
        self._beta1_min = beta1 if self._beta1_min is None else min(self._beta1_min, beta1)
        self._beta1_max = beta1 if self._beta1_max is None else max(self._beta1_max, beta1)
        if rho is not None:
            self._rho_count += 1
            self._rho_sum += float(rho)
            self._rho_min = rho if self._rho_min is None else min(self._rho_min, rho)
            self._rho_max = rho if self._rho_max is None else max(self._rho_max, rho)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        rho = self._adjacent_gradient_cosine()
        beta1 = self._select_beta1(rho)
        found_gradient = False
        for group in self.param_groups:
            lr = float(group['lr'])
            beta2 = float(group['beta2'])
            eps = float(group['eps'])
            weight_decay = float(group['weight_decay'])
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                found_gradient = True
                if gradient.is_sparse:
                    raise RuntimeError('AdaptiveMomentumAdamW does not support sparse gradients.')
                state = self.state[parameter]
                if not state:
                    state['step'] = 0
                    state['beta1_product'] = 1.0
                    state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state['step'] += 1
                state['beta1_product'] *= beta1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - float(state['beta1_product'])
                bias_correction2 = 1.0 - beta2 ** int(state['step'])
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-lr / bias_correction1)
                previous = state.get('previous_grad')
                if previous is None:
                    state['previous_grad'] = gradient.detach().clone(memory_format=torch.preserve_format)
                else:
                    previous.copy_(gradient.detach())
        if not found_gradient:
            raise RuntimeError('AdaptiveMomentumAdamW received no gradients.')
        self._step_count += 1
        self._has_previous_gradient = True
        self._record(rho, beta1)
        return loss

    def statistics(self) -> Dict[str, object]:
        return {'steps': int(self._step_count), 'bootstrap_steps': int(self._bootstrap_steps), 'last_rho': self._last_rho, 'last_beta1': float(self._last_beta1), 'rho_count': int(self._rho_count), 'rho_mean': self._rho_sum / float(self._rho_count) if self._rho_count else None, 'rho_min': self._rho_min, 'rho_max': self._rho_max, 'beta1_mean': self._beta1_sum / float(self._step_count) if self._step_count else None, 'beta1_min_observed': self._beta1_min, 'beta1_max_observed': self._beta1_max}

    def state_dict(self) -> Dict[str, object]:
        payload = super().state_dict()
        payload['adaptive_momentum'] = {'format': 'adaptive_momentum_adamw_v1', 'step_count': int(self._step_count), 'has_previous_gradient': bool(self._has_previous_gradient), 'last_rho': self._last_rho, 'last_beta1': float(self._last_beta1), 'rho_count': int(self._rho_count), 'rho_sum': float(self._rho_sum), 'rho_min': self._rho_min, 'rho_max': self._rho_max, 'beta1_sum': float(self._beta1_sum), 'beta1_min': self._beta1_min, 'beta1_max': self._beta1_max, 'bootstrap_steps': int(self._bootstrap_steps)}
        return payload

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        payload = dict(state_dict)
        metadata = payload.pop('adaptive_momentum', None)
        if not isinstance(metadata, dict) or metadata.get('format') != 'adaptive_momentum_adamw_v1':
            raise ValueError('Missing AdaptiveMomentumAdamW metadata.')
        super().load_state_dict(payload)
        self._step_count = int(metadata['step_count'])
        self._has_previous_gradient = bool(metadata['has_previous_gradient'])
        self._last_rho = metadata['last_rho']
        self._last_beta1 = float(metadata['last_beta1'])
        self._rho_count = int(metadata['rho_count'])
        self._rho_sum = float(metadata['rho_sum'])
        self._rho_min = metadata['rho_min']
        self._rho_max = metadata['rho_max']
        self._beta1_sum = float(metadata['beta1_sum'])
        self._beta1_min = metadata['beta1_min']
        self._beta1_max = metadata['beta1_max']
        self._bootstrap_steps = int(metadata['bootstrap_steps'])
