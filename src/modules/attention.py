import torch
import torch.nn as nn

from einops import rearrange


class Attention(nn.Module):
    def __init__(self, 
                    n_heads: int = 8, 
                    d_head: int = 64,
                    embedding_size: int = 512,
                    dropout_prob: float = 0.1) -> None:
        """Multi Head Attention module

        Args:
            n_heads (int, optional): Number of heads. Defaults to 8.
            d_head (int,  optional): Size of the heads. Defaults to 64.
            embedding_size (int, optional): Embedding size. Defaults to 512.
            dropout_prob (float, optional): Dropout probability. 
                Defaults to 0.1.
        """
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.embedding_size = embedding_size
        self.dropout_prob = dropout_prob

        self.d_hidden = self.n_heads * self.d_head

        self.projection_condition = not (self.n_heads == 1 and 
                                            self.d_head == self.embedding_size)
        self.projection =\
             nn.Sequential(
                nn.Linear(self.d_hidden, self.embedding_size),
                nn.Dropout(dropout_prob)
            ) if self.projection_condition else nn.Identity()

        self.linear_q = nn.Linear(self.embedding_size, self.d_hidden)
        self.linear_k = nn.Linear(self.embedding_size, self.d_hidden)
        self.linear_v = nn.Linear(self.embedding_size, self.d_hidden)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.softmax = nn.Softmax(-1)
        

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) \
                                                            -> torch.Tensor:
        """Forward step for computing Attention
        Compute Q, K, V -> Q * K -> apply mask -> softmax -> scale -> dropout
        -> A * V -> projection

        Args:
            x (torch.Tensor): input tensor
            mask (torch.Tensor): mask tensor

        Returns:
            torch.Tensor: _description_
        """
        Q = rearrange(self.linear_q(x), "b n (h d) -> b h n d", h=self.n_heads)
        K = rearrange(self.linear_k(x), "b n (h d) -> b h n d", h=self.n_heads)
        V = rearrange(self.linear_v(x), "b n (h d) -> b h n d", h=self.n_heads)

        # [batch_size, n_heads, l_query, embedding_size] * 
        # [batch_size, n_heads, l_key, embedding_size] -> 
        # [batch_size, n_heads, l_query, l_key]
        scores = torch.einsum('bhqd, bhkd -> bhqk', Q, K)

        if mask is not None:
            scores.masked_fill_(mask==0, -1e-12)

        A = self.softmax(scores * self.d_head ** -0.5)
        A = self.dropout(A)

        # [batch_size, n_heads, l_values, embedding_size]
        x = torch.einsum('bhal, bhlv -> bhav', A, V)
        # [batch_size, n_heads, l_values, embedding_size] -> 
        # [batch_size, l_values, n_heads * embedding_size]
        x = rearrange(x, 'b h n d -> b n (h d)')
        return self.projection(x)