import torch
from torch.nn import functional as F


def check_tensor(name, t):
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(name, t)


def _quadratic_kappa_coefficient(output, target, eps=1e-7):
    # check_tensor('output', output)
    # check_tensor('target', target)
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=output.dtype, device=output.device) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2
    # check_tensor('weights', weights)

    C = (output.t() @ target).t()  # confusion matrix
    # check_tensor('C', C)

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum()  # Normalize to the sum of C.
    # check_tensor("E", E)

    num = weights * C
    den = weights * E

    # check_tensor('num', num)
    # check_tensor('den', den)
    QWK = 1 - torch.sum(num) / (torch.sum(den) + eps)  # eps guarantees numerical stability
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
        dtype = output.dtype
        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device).type(dtype)
        if self.binned:
            output = torch.sigmoid(output).sum(1).round().long()
            output = F.one_hot(output.squeeze(), num_classes=6).to(output.device).type(dtype)
        else:
            output = torch.softmax(output, dim=1)
        return _quadratic_kappa_coefficient(output, target)


class QWKLoss(torch.nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, output, target):
        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device).type(output.dtype)
        output = torch.softmax(output, dim=1)
        return _quadratic_kappa_loss(output, target, self.scale)


class Accuracy:
    def __init__(self, batch_size):
        self.score_accumulator = torch.zeros(5)
        self.batch_size = batch_size

    def __call__(self, output, target, *args, **kwargs):
        dtype = output.dtype
        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device).type(dtype)
        output = torch.sigmoid(output).sum(1).round()
        output = F.one_hot(output.squeeze(), num_classes=6).to(output.device).type(dtype)
        self.score_accumulator = output - target

