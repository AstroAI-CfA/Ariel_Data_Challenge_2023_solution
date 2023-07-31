"""
Module for a scheduler of the latent loss weight during the training of the autoencoder.
Not used in the end.
"""

class LLWScheduler:
    def __init__(self):
        pass

    def __call__(self, latent_loss_weight, epoch):
        raise NotImplementedError


class CrenelScheduler(LLWScheduler):
    def __init__(self, factor: float = 0.5, period: int = 30):
        super().__init__()
        self.factor = factor
        self.period = period

    def __call__(self, latent_loss_weight, epoch):
        if epoch % self.period == 0 and epoch > 1:
            return latent_loss_weight * self.factor
        return latent_loss_weight
