import torch
import torch.nn.functional as functional


def projected_residual(conditional, unconditional, projection_weight):
    residual = conditional - unconditional
    direction = functional.normalize(conditional.double(), dim=(-1, -2, -3))
    parallel = (residual.double() * direction).sum(dim=(-1, -2, -3), keepdim=True) * direction
    orthogonal = residual.double() - parallel
    return orthogonal.to(residual.dtype) + projection_weight * parallel.to(residual.dtype)


def anchor_guided_target(unconditional, target, anchor, guidance_scale, projection_weight):
    anchor_update = projected_residual(anchor, unconditional, projection_weight)
    target_update = projected_residual(target, unconditional, projection_weight)
    positive_update = (guidance_scale - 1) * anchor_update
    negative_update = -(guidance_scale - 1) * target_update
    return unconditional + (positive_update + negative_update)


class ErasureRetentionObjective:
    def __init__(self, model, teacher, settings):
        self.model = model
        self.teacher = teacher
        self.settings = settings

    def erase_loss(self, latent, timestep, target_condition, anchor_condition, unconditional):
        with torch.no_grad():
            base_noise = self.model.predict_noise(latent, timestep, unconditional, self.teacher)
            target_noise = self.model.predict_noise(
                latent, timestep, target_condition, self.teacher
            )
            anchor_noise = self.model.predict_noise(
                latent, timestep, anchor_condition, self.teacher
            )
            desired_noise = anchor_guided_target(
                base_noise,
                target_noise,
                anchor_noise,
                self.settings.guidance_scale,
                self.settings.projection_weight,
            )
        prediction = self.model.predict_noise(latent, timestep, target_condition)
        return functional.mse_loss(prediction.float(), desired_noise.float())

    def retain_loss(self, latent, timestep, condition):
        with torch.no_grad():
            reference = self.model.predict_noise(latent, timestep, condition, self.teacher)
        prediction = self.model.predict_noise(latent, timestep, condition)
        return self.settings.retain_weight * functional.mse_loss(
            prediction.float(), reference.float()
        )
