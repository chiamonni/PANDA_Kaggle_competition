import numpy as np
import torch
import torch.nn as nn
from typing import Optional


######## UTILS
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience"""

    def __init__(self, patience=7, verbose=False, delta=0, mode='higher'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_metric = np.Inf  # The lower the better
        self.delta = delta
        self.save_checkpoint = False
        if mode == 'higher':
            self.mode = 1
        elif mode == 'lower':
            self.mode = -1
        else:
            raise ValueError("Bad mode type, please choose between 'higher' and 'lower'")

    def __call__(self, train_metric, val_metric, model=None):
        val_score = self.mode * val_metric
        train_score = self.mode * train_metric
        if not torch.isnan(torch.tensor(val_score)):
            if self.best_score is None:
                self.best_score = val_score
                self.save_checkpoint = True
                self.best_val_metric = val_metric
            elif val_score < self.best_score + self.delta and train_score > val_score + self.delta:  # apply patience only if train is better than val scores
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                self.save_checkpoint = False
            elif val_score > self.best_score:
                self.best_score = val_score
                self.save_checkpoint = True
                self.best_val_metric = val_metric
                self.counter = 0
            else:
                self.save_checkpoint = False
        else:
            self.save_checkpoint = False


######## ACTIVATIONS
class Mish(torch.nn.Module):
    """
    Applies the mish activation function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    @staticmethod
    def mish(input):
        '''
        Applies the mish function element-wise:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        See additional documentation for mish class.
        '''
        return input * torch.tanh(torch.nn.functional.softplus(input))

    def forward(self, input):
        """"
        Forward pass of the function.
        """
        return self.mish(input)


######### CUSTOM LAYERS
class Flatten(torch.nn.Module):
    """
    The flatten layer to build sequential models
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class SiameseBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        x = torch.cat([self.module(x_i) for x_i in x.transpose(1, 0)], 1)
        return x


class ChiaBlock(torch.nn.Module):
    def __init__(self, module, axis=-1):
        super().__init__()
        self.module = module
        self.axis = axis

    def forward(self, x):
        x = torch.stack([self.module(x_i) for x_i in x.transpose(1, 0)], self.axis)
        x = torch.mean(x, self.axis)
        # x = torch.logsumexp(x, self.axis) / x.size(self.axis)
        return x


class AdaptiveConcatPool2d(torch.nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz: Optional[int] = None):
        super().__init__()
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


############# LOSSES
class BinnedBCE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(*args, **kwargs)
        self.binning = lambda x: torch.cat([torch.ones(x), torch.zeros(5-x)])

    def forward(self, output, target: torch.Tensor):
        bin_target = torch.stack([self.binning(t) for t in target.tolist()], dim=0).to(target.device)
        return self.loss(output, bin_target)