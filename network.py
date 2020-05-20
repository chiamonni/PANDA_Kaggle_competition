import torchvision.models as models
import torch
from torch import nn
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


class BaseNetwork(nn.Module):
    def __init__(self):
        # inizializzazione classe base - si fa sempre
        super().__init__()

    def forward(self, inputs, mask=None):
        pass

    @staticmethod
    def get_input(batch, DEVICE):
        pass

    def train_batch(self, net, train_loader, loss_fn, metric_fn, optimizer, DEVICE) -> (torch.Tensor, torch.Tensor):
        """
        Define training method only once. The only method that must be done is how the training gets the training inputs
        :param net:
        :param train_loader:
        :param loss_fn:
        :param metric_fn:
        :param optimizer:
        :param scheduler: scheduler that must be updated at every batch iteration
        :param DEVICE:
        :return:
        """
        net.to(DEVICE)
        net.train()
        conc_losses = []
        conc_metrics = []

        for batch in tqdm(train_loader, desc='Training...'):
            net_input = batch['scan'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # forward pass
            net_output = net(net_input)

            del net_input

            # update networks
            loss = loss_fn(net_output, labels)
            metric = metric_fn(net_output, labels)

            del net_output

            # clear previous recorded gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update optimizer
            optimizer.step()

            conc_losses.append(loss.item())
            conc_metrics.append(metric.item())

            del loss
            del metric


        return torch.mean(torch.tensor(conc_losses)), torch.mean(torch.tensor(conc_metrics))

    def val_batch(self, net, val_loader, loss_fn, metric_fn, DEVICE) -> (torch.Tensor, torch.Tensor):
        net.to(DEVICE)
        net.eval()
        conc_losses = []
        conc_metrics = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating...'):
                net_input = batch['scan'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                # evaluate the network over the input
                net_output = net(net_input)
                del net_input
                loss = loss_fn(net_output, labels)
                metric = metric_fn(net_output, labels)
                del net_output
                conc_losses.append(loss.item())
                conc_metrics.append(metric.item())
                del loss
                del metric

        return torch.mean(torch.tensor(conc_losses)), torch.mean(torch.tensor(conc_metrics))

    '''
    def predict_batch(self, net, test_loader, DEVICE) -> (np.ndarray, np.ndarray):
        net.eval()
        conc_output = []
        conc_ID = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting test set..."):
                net_input = self.get_input(batch, DEVICE)
                conc_ID.extend(list(batch['ID'].detach().cpu().numpy()))
                # evaluate the network over the input
                conc_output.extend(list(net(net_input).detach().cpu().numpy()))

        return conc_ID, conc_output
    '''

class DenseNet201(BaseNetwork):
    def __init__(self,
                 num_classes=6,
                 pretrained=True):
        # Call the parent init function (required!)
        super().__init__()

        # Define dense. Output layer dimension must be equal to num_classes
        self.dense = models.densenet121(pretrained=pretrained)
        self.dense.classifier = nn.Linear(1024, num_classes)

        self.num_classes = num_classes

    def forward(self, inputs, mask=None):

        x = torch.zeros(self.num_classes, device=DEVICE)

        # inputs: il singolo input è un immagine del train, già suddivisa in "crops" --> è una lsta dei crop che definiscono la singola immgine
        # per ogni crop, calcolo le 6 probabilità e le sommo tra loro, in modo da avere un layer da 6 elementi che
        # rappresenti l'intera immagine, invece di tenere le informazioni sul singolo crop
        inputs: torch.Tensor
        for crop in inputs.transpose(0, 1):
            x = x + self.dense(crop)

        return x

