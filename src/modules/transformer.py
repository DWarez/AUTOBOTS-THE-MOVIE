import torch
import torch.nn as nn

from attention import Attention
from residual_add import ResidualAdd
from ff import FeedForward


class Encoder(nn.Sequential):
    def __init__(self, embedding_size: int, expansion: int = 4, 
                                            dropout_prob: float = 0.1, 
                                            ff_dropout_prob: float = 0.1, 
                                            **kwargs) -> None:
        """Transformer encoder

        Args:
            embedding_size (int): embedding size
            expansion (int, optional): expansion coefficient to obtain hidden
                representation. Defaults to 4.
            dropout_prob (float, optional): dropout probability for the 
                encoder block. Defaults to 0.1.
            ff_dropout_prob (float, optional): dropout probability for the 
                feed forward block. Defaults to 0.1.
        """
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_size),
                Attention(embedding_size=embedding_size, **kwargs),
                nn.Dropout(dropout_prob)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_size),
                FeedForward(embedding_size, expansion, ff_dropout_prob),
                nn.Dropout(ff_dropout_prob)
            ))
        )


class Transformer(nn.Sequential):
    def __init__(self, layers: int = 4, **kwargs):
        super().__init__(*[Encoder(**kwargs) for _ in range(layers)])