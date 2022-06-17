from statistics import LinearRegression
from turtle import forward
import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, in_channels: int = 3, 
                                        patch_size: int = 16, 
                                        embedding_size: int = 768) -> None:
        """Generates the embedding of the input patches obtained from images.
        The projection into patches is performed using a Conv2D layer + a 
        Rearrange layer from einops.
        The projection is concatenated with the cls_token and then summed to
        the positional encoding, which are both Parameters.

        Args:
            img_size (int): Size of the input image 
            in_channels (int, optional): Number of input channels. 
                Defaults to 3.
            patch_size (int, optional): Size of the square patches. 
                Defaults to 16.
            embedding_size (int, optional): Final dimension of the embedding. 
                Defaults to 768.
        """
        super(PatchEmbedding, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size,
                                                    stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.positional_encoding = \
            nn.Parameter(torch.randn((img_size//patch_size)**2+1, 
                                        embedding_size))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step of the PatchEmbedding module.
        The input x is projected, then concatenated with the cls tensor
        and, lastly, summed up with the positional encoding parameter.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: embedding of the input tensor
        """
        batch_size = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positional_encoding
        return x