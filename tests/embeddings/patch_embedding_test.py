import sys
sys.path.append('src/embeddings')

import torch
from patch_embedding import PatchEmbedding

pe = PatchEmbedding(320)
_tensor = torch.randn(10, 3, 320, 320)

out_shape = pe(_tensor).shape

assert out_shape == (10, 401, 768), f"Incorrect embedding shape, expected (10, 401, 768), got {out_shape}"