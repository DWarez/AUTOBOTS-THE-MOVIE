import sys
sys.path.append('src/embeddings')

import torch
from patch_embedding import PatchEmbedding

pe = PatchEmbedding(320)
_tensor = torch.randn(10, 3, 320, 320)

print(pe(_tensor).shape)