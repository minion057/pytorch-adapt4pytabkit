from abc import abstractmethod

from .base_validator import BaseValidator


class SimpleLossValidator(BaseValidator):
    def __init__(self, layer="logits", **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def compute_score(self, target_train): # Edited for pytabkit.
        # return -self.loss_fn(target_train[self.layer]).item()
        if self.layer in target_train.keys():
            return -self.loss_fn(target_train[self.layer]).item()
        
        layer = []
        for k, v in target_train.items():
            if self.layer in k: layer.append(k)
        # print(f'self.layer: {self.layer} -> {layer}')
        
        logits = None
        for l in layer:
            if target_train[l].numel() != 0:
                if logits is None:
                    logits = target_train[l]
                else:
                    raise ValueError((
                        'Multiple logits detected; loss function requires a single logit input.'
                        f'Designated layer: {self.layer}, found: {layer}. Minimum 2 values detected.'
                    ))
        return -self.loss_fn(logits).item()

    @property
    @abstractmethod
    def loss_fn(self, *args, **kwargs):
        pass
