import torch
import torch.nn as nn


class FeedForward(nn.Sequential):
    def __init__(self, embedding_size: int, expansion: int = 4, 
                                            dropout_prob: float = 0.1) -> None:
        """Feed Forward block

        Args:
            embedding_size (int): embedding size
            expansion (int, optional): constant multiplied with embedding_size
                to obtain size of the hidden representation 
                Defaults to 4.
            dropout_prob (float, optional): dropout probability
                Defaults to 0.1.
        """
        super().__init__(
            nn.Linear(embedding_size, expansion*embedding_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(expansion*embedding_size, embedding_size),
            nn.Dropout(dropout_prob)
        )