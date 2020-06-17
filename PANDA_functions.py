import torch
from torch.nn import functional as F


def _quadratic_kappa_coefficient(output, target):
    output, target = output.type(torch.float32), target.type(torch.float32)
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum() # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK


def _quadratic_kappa_loss(output, target, scale=2.0):
    QWK = _quadratic_kappa_coefficient(output, target)
    loss = -torch.log(torch.sigmoid(scale * QWK))
    return loss


class QWKMetric(torch.nn.Module):
    def __init__(self, binned=False):
        super().__init__()
        self.binned = binned

    def forward(self, output, target):
        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device)
        if self.binned:
            output = torch.sigmoid(output).sum(1).round().long()
            output = F.one_hot(output.squeeze(), num_classes=6).to(output.device)
        else:
            output = torch.softmax(output, dim=1)
        return _quadratic_kappa_coefficient(output, target)


class QWKLoss(torch.nn.Module):
    def __init__(self, scale=2.0, binned=False):
        super().__init__()
        self.binned = binned
        self.scale = scale

    def forward(self, output, target):
        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device)
        if self.binned:
            output = torch.sigmoid(output).sum(1).round().long()
            output = F.one_hot(output.squeeze(), num_classes=6).to(output.device)
        else:
            output = torch.softmax(output, dim=1)
        return _quadratic_kappa_loss(output, target, self.scale)

'''
if __name__ == '__main__':
    target = torch.rand((64, 6))
    output = torch.rand((64, 1, 6))
    print("QWK coefficient: {}".format(_quadratic_kappa_coefficient(output, target)))
    print("QWK loss: {}".format(_quadratic_kappa_loss(output, target)))
    '''