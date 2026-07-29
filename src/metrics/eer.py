import numpy as np
import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.calculate_eer import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self._scores = []
        self._labels = []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        scores = (logits[:, 0] - logits[:, 1]).detach().cpu().numpy()
        self._scores.append(scores)
        self._labels.append(labels.detach().cpu().numpy())

    def compute(self):
        scores = np.concatenate(self._scores)
        labels = np.concatenate(self._labels)
        bonafide_scores = scores[labels == 0]
        spoof_scores = scores[labels == 1]
        eer, _ = compute_eer(bonafide_scores, spoof_scores)
        return eer * 100
