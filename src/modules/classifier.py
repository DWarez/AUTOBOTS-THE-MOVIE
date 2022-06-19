import torch
import torch.nn as nn

from einops.layers.torch import Reduce

class Classifier(nn.Sequential):
    def __init__(self, embedding_size: int, n_classes: int) -> None:
        """MLP Head used for classification

        Args:
            embedding_size (int): embedding size
            n_classes (int): number of classes
        """
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, n_classes)
        )