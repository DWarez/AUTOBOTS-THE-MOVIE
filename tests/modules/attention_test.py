import sys
sys.path.append('src/embeddings')
sys.path.append('src/modules')

import torch
from patch_embedding import PatchEmbedding
from attention import Attention

mha = Attention(embedding_size=768)

pe = PatchEmbedding(224)
_tensor = torch.randn(10, 3, 224, 224)

embedding = pe(_tensor)

out_shape = mha(embedding, None).shape

assert out_shape == (10, 197, 768), f"Incorrect embedding shape, expected (10, 197, 768), got {out_shape}"