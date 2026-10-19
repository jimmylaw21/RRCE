from rrce.optim.adaptive_adamw import AdaptiveMomentumAdamW
from rrce.runtime import check_loss


class EraseRetainOptimizer:
    def __init__(self, parameters, config):
        parameters = list(parameters)
        options = {
            "lr": config.learning_rate,
            "beta_min": config.beta_min,
            "beta_max": config.beta_max,
            "beta2": config.beta2,
            "eps": config.epsilon,
            "weight_decay": config.weight_decay,
        }
        self.erase = AdaptiveMomentumAdamW(parameters, **options)
        self.retain = AdaptiveMomentumAdamW(parameters, **options)

    def update(self, objective, loss):
        optimizer = getattr(self, objective)
        check_loss(loss, objective)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return optimizer.statistics()

    def state_dict(self):
        return {"erase": self.erase.state_dict(), "retain": self.retain.state_dict()}

    def load_state_dict(self, state):
        self.erase.load_state_dict(state["erase"])
        self.retain.load_state_dict(state["retain"])
