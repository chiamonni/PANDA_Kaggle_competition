import torchvision.models as models
import torch
from torch import nn
from tqdm import tqdm
from apex import amp
from pytorchtools import Mish, Flatten, AdaptiveConcatPool2d

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

class BaseNetwork(nn.Module):
    """
    Base class to define common methods among all networks
    """
    def __init__(self, use_apex):
        # inizializzazione classe base - si fa sempre
        super().__init__()
        self.use_apex = use_apex
        self.collate_fn = None

    def forward(self, inputs):
        pass

    def train_batch(self, train_loader, loss_fn, metric_fn, optimizer, scheduler, DEVICE) -> (torch.Tensor, torch.Tensor):
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
        self.to(DEVICE)
        self.train()
        running_loss = 0
        running_metric = 0
        for batch in tqdm(train_loader, desc='Training...'):
            net_input = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)

            # forward pass
            net_output = self.forward(net_input)

            del net_input

            # update networks
            loss = loss_fn(net_output, labels)
            metric = metric_fn(net_output, labels)

            del net_output

            # clear previous recorded gradients
            optimizer.zero_grad()

            # backward pass
            if self.use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Enable in case of gradient clipping
            # norms += torch.nn.utils.clip_grad_norm_(self.parameters(), 5.).item()

            # update optimizer
            optimizer.step()

            running_loss += loss.item()
            running_metric += metric.item()

            del loss
            del metric

            # Update scheduler
            if scheduler:
                scheduler.step()
            # else:
            #     break
        # print("Training norm: {:.4f}".format(norms/len(train_loader)))
        return running_loss / len(train_loader), running_metric / len(train_loader)

    def val_batch(self, val_loader, loss_fn, metric_fn, DEVICE) -> (torch.Tensor, torch.Tensor):
        self.to(DEVICE)
        self.eval()
        running_loss = 0
        running_metric = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating...'):
                net_input = batch[0].to(DEVICE)
                labels = batch[1].to(DEVICE)

                # evaluate the network over the input
                net_output = self.forward(net_input)

                del net_input
                loss = loss_fn(net_output, labels)
                metric = metric_fn(net_output, labels)
                del net_output
                running_loss += loss.item()
                running_metric += metric.item()
                del loss
                del metric

        return running_loss / len(val_loader), running_metric / len(val_loader)

    @staticmethod
    def predict_batch(net, test_loader, DEVICE):
        net.eval()
        conc_output = []
        conc_ID = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting test set..."):
                net_input = [b.to(DEVICE) for b in batch[0]]
                conc_ID.extend(list(batch[0].detach().cpu().numpy()))
                # evaluate the network over the input
                conc_output.extend(list(net(net_input).detach().cpu().numpy()))

        return conc_ID, conc_output


class IAFoss(BaseNetwork):
    def __init__(self, arch='resnext50_32x4d_swsl', n=6, pre=True, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        for param in self.enc.parameters():
            param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2 * nc, 512),
            Mish(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_prob),
            nn.Linear(512, n)
        )

    def forward(self, x):
        shape = x[0].shape
        N = x.size(1)  # number of crops
        x = x.view(-1, shape[1], shape[2], shape[3])
        # x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        # x: bs*N x C x 4 x 4
        shape = x.shape
        # concatenate the output for tiles into a single map
        x = x.view(-1, N, shape[1], shape[2], shape[3])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, shape[1], shape[2] * N, shape[3])
        # x: bs x C x N*4 x 4
        x = self.head(x)
        # x: bs x n
        return x


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

