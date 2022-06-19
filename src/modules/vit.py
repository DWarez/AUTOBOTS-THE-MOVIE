import sys
sys.path.append('src/embeddings')

import torch
import torch.nn as nn

from patch_embedding import PatchEmbedding
from transformer import Transformer
from classifier import Classifier


class ViT(nn.Sequential):
    def __init__(self, embedding_size: int, img_size: int, n_layers: int,
                    n_classes: int, in_channels: int = 3, 
                    patch_size: int = 16, **kwargs) -> None:
        """_summary_

        Args:
            embedding_size (int): _description_
            img_size (int): _description_
            n_layers (int): _description_
            n_classes (int): _description_
            in_channels (int, optional): _description_. Defaults to 3.
            patch_size (int, optional): _description_. Defaults to 16.
        """
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        super().__init__(
            PatchEmbedding(img_size=img_size, 
                            in_channels=in_channels, 
                            patch_size=patch_size, 
                            embedding_size=embedding_size),
            Transformer(n_layers=n_layers, 
                        embedding_size=embedding_size, 
                        **kwargs),
            Classifier(embedding_size=embedding_size, n_classes=n_classes)
        )

