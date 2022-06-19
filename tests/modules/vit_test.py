import sys
sys.path.append('src/embeddings')
sys.path.append('src/modules')

import torch
from vit import ViT

vit = ViT(embedding_size=768, img_size=224, n_layers=3, n_classes=10)

_tensor = torch.randn(32, 3, 224, 224)

out_shape = vit(_tensor).shape

assert out_shape == (32, 10), f"Incorrect embedding shape, expected (32, 10), got {out_shape}"