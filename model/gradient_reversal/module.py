# from .functional import revgrad
# import torch
# from torch import nn

# class GradientReversal(nn.Module):
#     def __init__(self, alpha):
#         super().__init__()
#         self.alpha = torch.tensor(alpha, requires_grad=False)

#     def forward(self, x):
#         return revgrad(x, self.alpha)

from .functional import revgrad
from torch.nn import Module
from torch import tensor


class RevGrad(Module):
    def __init__(self, alpha=1.0, *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)
