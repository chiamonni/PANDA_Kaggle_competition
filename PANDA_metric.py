import numpy as np
import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
# DEVICE = torch.device('cpu')


class PANDA_loss(torch.nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes

    def __quadratic_weighted_kappa(self, output, target):
        target = F.one_hot(target.to(torch.int64).squeeze(), num_classes=self.num_classes)
        target = target.type(torch.float32)
        #target = torch.squeeze(target)
        target = torch.unsqueeze(target, 0)
        output = output.type(torch.float32)

        """
                    QWK loss function as described in https://arxiv.org/pdf/1612.00775.pdf

                    Arguments:
                        p: a tensor with probability predictions, [batch_size, n_classes],
                        y, a tensor with one-hot encoded class labels, [batch_size, n_classes]
                    Returns:
                        QWK loss
                    """
        eps = 1e-10
        W = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                W[i, j] = (i - j) ** 2 / ((self.num_classes-1) ** 2)

        W = torch.from_numpy(W.astype(np.float32)).to(DEVICE)

        O = torch.matmul(target.t(), output)
        E = torch.matmul(target.sum(dim=0).view(-1, 1), output.sum(dim=0).view(1, -1)) / O.sum()

        return 1 - (W * O).sum() / ((W * E).sum() + eps)
        '''
        weights = torch.arange(0, self.num_classes, dtype=torch.float32, device=DEVICE) / (self.num_classes - 1)
        weights = (weights - torch.unsqueeze(weights, -1)) ** 2

        hist_true = torch.sum(target, dim=0)
        hist_pred = torch.sum(output, dim=0)

        E = torch.unsqueeze(hist_true, dim=-1) * hist_pred
        E = E / torch.sum(E)
        # print("target ")
        # print(target.shape)
        # print("target unsqueezed ")
        # print(torch.unsqueeze(target,0).shape)
        # print("output")
        # print(output.shape)
        # O = (target.t() @ output).t()  # confusion matrix
        O = (torch.unsqueeze(target,1)) @ output
        O = torch.unsqueeze(O,0)
        O = O / torch.sum(O)

        num = weights * O
        den = weights * E

        QWK = (1 - torch.sum(num) / torch.sum(den))
        # print("QWK")
        # print(QWK)
        return QWK'''

    def __quadratic_kappa_loss(self, output, target, scale=2.0):
        #target = F.one_hot(target.to(torch.int64).squeeze(), num_classes=self.num_classes)
        QWK = self.__quadratic_weighted_kappa(output, target)
        v = torch.sigmoid(scale * QWK)
        loss = -torch.log(v)
        return loss

    def forward(self, output: torch.Tensor, target: torch.Tensor, scale=2.0, **kwargs):
        return self.__quadratic_kappa_loss(output, target, scale)


class PANDA_metric(torch.nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes

    def __quadratic_weighted_kappa(self, output, target):
        target = F.one_hot(target.to(torch.int64).squeeze(), num_classes=self.num_classes)
        target = target.type(torch.float32)
        #target = torch.squeeze(target)
        target = torch.unsqueeze(target,0)
        output = output.type(torch.float32)

        """
            QWK loss function as described in https://arxiv.org/pdf/1612.00775.pdf

            Arguments:
                p: a tensor with probability predictions, [batch_size, n_classes],
                y, a tensor with one-hot encoded class labels, [batch_size, n_classes]
            Returns:
                QWK loss
            """
        eps = 1e-10
        W = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                W[i, j] = (i - j) ** 2 / ((self.num_classes-1) ** 2)

        W = torch.from_numpy(W.astype(np.float32)).to(DEVICE)

        O = torch.matmul(target.t(), output)
        E = torch.matmul(target.sum(dim=0).view(-1, 1), output.sum(dim=0).view(1, -1)) / O.sum()

        return 1 - (W * O).sum() / ((W * E).sum() + eps)

        '''weights = torch.arange(0, self.num_classes, dtype=torch.float32, device=DEVICE) / (self.num_classes - 1)
        weights = (weights - torch.unsqueeze(weights, -1)) ** 2

        hist_true = torch.sum(target, dim=0)
        hist_pred = torch.sum(output, dim=0)

        E = torch.unsqueeze(hist_true, dim=-1) * hist_pred
        E = E / torch.sum(E)
        # print("target ")
        # print(target.shape)
        # print("target unsqueezed ")
        # print(torch.unsqueeze(target,0).shape)
        # print("output")
        # print(output.shape)
        # O = (target.t() @ output).t()  # confusion matrix
        O = (torch.unsqueeze(target,1)) @ output
        O = torch.unsqueeze(O,0)
        O = O / torch.sum(O)

        num = weights * O
        den = weights * E

        QWK = (1 - torch.sum(num) / torch.sum(den))
        return QWK'''

    def forward(self, output: torch.Tensor, target: torch.Tensor, scale=2.0, **kwargs):
        return self.__quadratic_weighted_kappa(output, target)
