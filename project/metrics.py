"""One synchronized episode per environment; never count post-reset states."""
import torch

class EpisodeMetrics:
    def __init__(self, num_envs, device, dt):
        self.dt = dt
        self.alive = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.steps = torch.zeros(num_envs, device=device)
        self.sums = {}

    def update(self, values, failed):
        active = self.alive.clone()
        self.steps += active
        for key, value in values.items():
            if not torch.isfinite(value[active]).all():
                raise FloatingPointError('Non-finite evaluation metric: ' + key)
            if key not in self.sums:
                self.sums[key] = torch.zeros_like(self.steps)
            self.sums[key] += torch.where(active, value, torch.zeros_like(value))
        self.alive &= ~failed.bool()

    def summary(self):
        if (self.steps == 0).any():
            raise ValueError('No samples for one or more environments')
        result = {key: (value / self.steps).mean().item() for key, value in self.sums.items()}
        result['living_time_s'] = (self.steps * self.dt).mean().item()
        result['survival_rate'] = self.alive.float().mean().item()
        return result
