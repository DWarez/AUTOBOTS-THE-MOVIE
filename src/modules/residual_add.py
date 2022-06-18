import torch
import torch.nn as nn


class ResidualAdd(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        """Residual add block

        Args:
            module (nn.Module): module on which to perform Residual Add 
        """
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Adds the original input to the output of the module

        Args:
            x (_type_): input tensor

        Returns:
            torch.Tensor: module(x) + x
        """
        res = x
        x =  self.module(x, **kwargs)
        x += res
        return x