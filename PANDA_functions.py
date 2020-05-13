import torch
from torch.nn import functional as F
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _quadratic_kappa_coefficient(y_true, y_pred):
    y_true, y_pred = y_true.squeeze(), y_pred.squeeze()
    y_true = y_true.type(torch.float32)
    n_classes = y_pred.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=DEVICE) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    hist_true = torch.sum(y_true, dim=0)
    hist_pred = torch.sum(y_pred, dim=0)

    E = torch.unsqueeze(hist_true, dim=-1) * hist_pred
    E = E / torch.sum(E)

    O = (y_true.t() @ y_pred).t()  # confusion matrix
    O = O / torch.sum(O)

    num = weights * O
    den = weights * E

    QWK = (1 - torch.sum(num) / torch.sum(den))
    return QWK


def _quadratic_kappa_loss(y_true, y_pred, scale=2.0):
    QWK = _quadratic_kappa_coefficient(y_true, y_pred)
    loss = -torch.log(torch.sigmoid(scale * QWK))
    return loss


class QWKMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        target = F.one_hot(target.to(torch.int64).squeeze(), num_classes=6).to(DEVICE)
        return _quadratic_kappa_coefficient(target, output)


class QWKLoss(torch.nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, output, target):
        target = F.one_hot(target.to(torch.int64).squeeze(), num_classes=6).to(DEVICE)
        return _quadratic_kappa_loss(target, output, self.scale)

'''
if __name__ == '__main__':
    target = torch.rand((64, 6))
    output = torch.rand((64, 1, 6))
    print("QWK coefficient: {}".format(_quadratic_kappa_coefficient(output, target)))
    print("QWK loss: {}".format(_quadratic_kappa_loss(output, target)))
    '''