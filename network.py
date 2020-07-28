import torch
from torch import nn
from tqdm import tqdm
from apex import amp
from pytorchtools import Mish, Flatten, AdaptiveConcatPool2d, SiameseBlock, ChiaBlock
from efficientnet_pytorch import EfficientNet
import os


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
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            # torch.nn.utils.clip_grad_value_(self.parameters(), 100.)

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

                if torch.isinf(net_output).any() or torch.isnan(net_output).any():
                    print("net_output", net_output)
                    print("net_input", net_input)
                    print('filenames', batch[2])
                    raise ValueError("inf or nan found")

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
                conc_ID.extend(list(batch[1].detach().cpu().numpy()))
                # evaluate the network over the input
                conc_output.extend(list(net(net_input).detach().cpu().round().numpy()))

        return conc_ID, conc_output


class IAFoss(BaseNetwork):
    def __init__(self, arch='resnext50_32x4d_swsl', n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        self.fc_dim = fc_dim
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2 * nc, self.fc_dim),
            Mish(),
            nn.BatchNorm1d(self.fc_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(self.fc_dim, n)
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


class IAFoss_SiameseIdea(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, arch='resnet18_swsl', n=6, fc_dim=2048, num_crops=36, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            SiameseBlock(nn.Sequential(self.enc, nn.AdaptiveAvgPool2d(1))),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc * num_crops, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )

    def forward(self, x):
        # temp = []
        # x.shape = bs, N, C, H, W
        # for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
        #     temp.append(self.enc(scan))
        # x = torch.cat(temp, 1)  # x.shape = bs, N*nc, 8, 8
        # x = self.head(x)  # x.shape = bs, 5
        # return x
        return self.head(x)


class BigSiamese(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, arch='resnet18_swsl', n=6, fc_dim=2024, num_crops=36, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            SiameseBlock(self.enc),
            # AdaptiveConcatPool2d(),
            nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc * num_crops, fc_dim),
            nn.BatchNorm1d(fc_dim),
            Mish(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )


class ResNet50Siamese(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, arch='resnet50_swsl', n=6, fc_dim=2048, num_crops=36, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            SiameseBlock(self.enc),
            # AdaptiveConcatPool2d(),
            nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc * num_crops, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )

    def forward(self, x):
        # temp = []
        # x.shape = bs, N, C, H, W
        # for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
        #     temp.append(self.enc(scan))
        # x = torch.cat(temp, 1)  # x.shape = bs, N*nc, 8, 8
        # x = self.head(x)  # x.shape = bs, 5
        # return x
        return self.head(x)


class ResNet18Chia(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, arch='resnet18_swsl', n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            ChiaBlock(nn.Sequential(self.enc, nn.AdaptiveMaxPool2d(1), Mish())),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            # Mish(),
            Flatten(),
            # nn.BatchNorm1d(nc),
            nn.Linear(nc, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )

    def forward(self, x):
        return self.head(x)


class ResNet18ChiaVariant(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, arch='resnet18_swsl', n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            ChiaBlock(nn.Sequential(self.enc, nn.AdaptiveMaxPool2d(4), Mish())),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            # Mish(),
            Flatten(),
            # nn.BatchNorm1d(nc),
            nn.Linear(nc*16, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )

    def forward(self, x):
        # temp = []
        # x.shape = bs, N, C, H, W
        # for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
        #     temp.append(self.enc(scan))
        # x = torch.cat(temp, 1)  # x.shape = bs, N*nc, 8, 8
        # x = self.head(x)  # x.shape = bs, 5
        # return x
        return self.head(x)


class ResNet18ChiaSSL(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, arch='resnet18_ssl', n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            ChiaBlock(nn.Sequential(self.enc, nn.AdaptiveMaxPool2d(1))),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )

    def forward(self, x):
        # temp = []
        # x.shape = bs, N, C, H, W
        # for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
        #     temp.append(self.enc(scan))
        # x = torch.cat(temp, 1)  # x.shape = bs, N*nc, 8, 8
        # x = self.head(x)  # x.shape = bs, 5
        # return x
        return self.head(x)


class ResNet50Chia(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, arch='resnet50_swsl', n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            ChiaBlock(nn.Sequential(self.enc, nn.AdaptiveMaxPool2d(1))),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )

    def forward(self, x):
        return self.head(x)


class DenseNet121Chia(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

        if freeze_weights:
            for param in m.parameters():
                param.requires_grad_(False)

        nc = m.classifier.in_features
        self.head = nn.Sequential(
            ChiaBlock(m),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )
        m.classifier = nn.Identity()

    def forward(self, x):
        # temp = []
        # x.shape = bs, N, C, H, W
        # for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
        #     temp.append(self.enc(scan))
        # x = torch.cat(temp, 1)  # x.shape = bs, N*nc, 8, 8
        # x = self.head(x)  # x.shape = bs, 5
        # return x
        return self.head(x)


class MobileNetV2Chia(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)

        if freeze_weights:
            for param in m.parameters():
                param.requires_grad_(False)

        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            ChiaBlock(m),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )
        m.classifier = nn.Identity()

    def forward(self, x):
        # temp = []
        # x.shape = bs, N, C, H, W
        # for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
        #     temp.append(self.enc(scan))
        # x = torch.cat(temp, 1)  # x.shape = bs, N*nc, 8, 8
        # x = self.head(x)  # x.shape = bs, 5
        # return x
        return self.head(x)


class ResNet34Chia(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
        m = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in m.parameters():
                param.requires_grad_(False)

        nc = 512
        self.head = nn.Sequential(
            ChiaBlock(nn.Sequential(m, nn.AdaptiveMaxPool2d(1))),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )

    def forward(self, x):
        # temp = []
        # x.shape = bs, N, C, H, W
        # for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
        #     temp.append(self.enc(scan))
        # x = torch.cat(temp, 1)  # x.shape = bs, N*nc, 8, 8
        # x = self.head(x)  # x.shape = bs, 5
        # return x
        return self.head(x)


class EfficientNetB7Chia(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """

    def __init__(self, n=6, fc_dim=2048, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        pretrained_model = ('efficientnet-b7', 'efficientnet-b7-dcc49843.pth')
        self.enc = EfficientNet.from_name(pretrained_model[0])
        self.enc.load_state_dict(torch.load(os.path.join('cache', 'checkpoints', pretrained_model[1])))

        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = self.enc._fc.in_features
        self.head = nn.Sequential(
            ChiaBlock(self.enc),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            nn.Linear(nc, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, n)
        )
        self.enc._fc = nn.Identity()

    def forward(self, x):
        # temp = []
        # x.shape = bs, N, C, H, W
        # for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
        #     temp.append(self.enc(scan))
        # x = torch.cat(temp, 1)  # x.shape = bs, N*nc, 8, 8
        # x = self.head(x)  # x.shape = bs, 5
        # return x
        return self.head(x)


class EfficientNetB0Chia(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, num_classes=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        pretrained_model = ('efficientnet-b0', 'efficientnet-b0-355c32eb.pth')
        self.enc = EfficientNet.from_name(pretrained_model[0])
        self.enc.load_state_dict(torch.load(os.path.join('cache', 'checkpoints', pretrained_model[1])))

        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)

        nc = self.enc._fc.in_features
        self.head = nn.Sequential(
            ChiaBlock(self.enc),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveMaxPool2d((1, 1)),
            Mish(),
            Flatten(),
            nn.Linear(nc, fc_dim),
            nn.Dropout(dropout_prob),
            Mish(),
            nn.Linear(fc_dim, num_classes)
        )
        self.enc._fc = nn.Identity()

    def forward(self, x):
        return self.head(x)


class EfficientNetB0Siamese(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, num_classes=6, freeze_weights=False, num_crops=36, dropout_prob=0., use_apex=False):
        super().__init__(use_apex)
        pretrained_model = ('efficientnet-b0', 'efficientnet-b0-355c32eb.pth')
        self.enc = EfficientNet.from_name(pretrained_model[0])
        self.enc.load_state_dict(torch.load(os.path.join('cache', 'checkpoints', pretrained_model[1])))

        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)

        self.classify = nn.Sequential(
            SiameseBlock(self.enc),
            nn.ReLU(inplace=True),
            nn.Linear(self.enc._fc.in_features*num_crops, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(2048, num_classes)
        )
        self.enc._fc = nn.Identity()

    def forward(self, x):
        return self.classify(x)


class Chia(BaseNetwork):
    """
    Same initialization as before, see "IAFoss" for parameters list
    """
    def __init__(self, arch='resnet18_swsl', n=6, fc_dim=512, freeze_weights=False, dropout_prob=0., use_apex=False):
        self.fc_dim = fc_dim
        super().__init__(use_apex)
        m = torch.hub
        m.hub_dir = 'cache'  # Define custom hub dir to avoid writing inside the container
        m = m.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        if freeze_weights:
            for param in self.enc.parameters():
                param.requires_grad_(False)
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2 * nc, self.fc_dim),
            Mish(),
            nn.BatchNorm1d(self.fc_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(self.fc_dim, n)
        )

    def forward(self, x):
        temp = []
        # x.shape = bs, N, C, H, W
        for scan in x.transpose(0, 1):  # Loop over N features and leave batch size
            temp.append(self.enc(scan))
        x = torch.stack(temp, 0).mean(0)  # x.shape = bs, nc, 8, 8
        x = self.head(x)  # x.shape = bs, 5
        return torch.clamp(x, -100., 100.)

