import numpy as np
import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


class PANDA_metric(torch.nn.Module):
    def __init__(self, to_be_returend='metric', num_classes=6):
        super().__init__()
        self.to_be_returned = to_be_returend
        self.num_classes = num_classes

    def __quadratic_weighted_kappa(self, target, output):
        target = target.type(torch.float32)
        weights = torch.arange(0, self.num_classes, dtype=torch.float32, device=DEVICE) / (self.num_classes - 1)
        weights = (weights - torch.unsqueeze(weights, -1)) ** 2

        hist_true = torch.sum(target, dim=0)
        hist_pred = torch.sum(output, dim=0)

        E = torch.unsqueeze(hist_true, dim=-1) * hist_pred
        E = E / torch.sum(E)

        O = (target.T @ output).T  # confusion matrix
        O = O / torch.sum(O)

        num = weights * O
        den = weights * E

        QWK = (1 - torch.sum(num) / torch.sum(den))
        return QWK

    def __quadratic_kappa_loss(self, output, target, scale=2.0):
        QWK = self.__quadratic_weighted_kappa(target, output)
        v = torch.sigmoid(scale * QWK)
        loss = -torch.log(v)
        return loss

    def forward(self, output: torch.Tensor, target: torch.Tensor, scale=2.0, **kwargs):
        if self.to_be_returned == 'loss':
            return self.__quadratic_kappa_loss(output, target, scale)
        if self.to_be_returned == 'metric':
            return self.__quadratic_weighted_kappa(output, target)

