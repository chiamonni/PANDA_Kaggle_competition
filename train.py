from dataset import \
    PANDA_dataset, \
    NormScale,\
    DataAugmentation,\
    NormCropsNumber,\
    AugmentDataset,\
    ToTensor, \
    Compose, \
    ZeroNormCropsNumber
from model import Model
import shutil
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import os

os.setgid(1000), os.setuid(1000)


def clean_folder(folder, metric, delta=0.02):
    """
    Cleans all checkpoints that are distant > delta from average metric
    :param folder: folder path
    :return: None
    """
    for file in os.listdir(folder):
        if 'checkpoint' in file:
            filename = file.split('.')[0] + '.' + file.split('.')[1]  # Keep name and floating value
            metric_value = filename.split('_')[3]  # Select float value
            metric_value = float(metric_value)  # Cast string to float
            if not metric - delta < metric_value < metric + delta:
                os.remove(os.path.join(folder, file))


if __name__ == '__main__':
    # import time
    # time.sleep(36000)
    mode = 'akensert'
    base_path = os.path.join('/opt/local_dataset')
    train_pt_folder = os.path.join(base_path, 'images', mode)
    train_info_path = os.path.join(base_path, 'train.csv')
    mask_path = os.path.join(base_path, 'train_label_masks')
    # mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    # variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define training hyper parameters
    batch_size = 4

    patience = 15

    net_hyperparams = {
        'dropout_prob': 0.5,
        'num_crops': 26
    }
    train_params = {
        'base_lr': 5e-6,
        'max_lr': 8e-5,
        'lr': 1e-6,
        'lr_decay': 1.,
        'use_apex': True,
        'weight_decay': 0.,
        'optimizer_type': 'adam',
        'network_type': 'ResNet50Chia',
        'loss_type': 'binnedbce',
        'binned': True,
        'freeze_weights': False
    }

    # Define training settings
    train_workers = 10
    val_workers = 8
    val_dim = 0.3
    dataset_quantity = 1.
    lr_range_test = False
    composed = False
    if lr_range_test:
        train_workers = 0

    dataset = PANDA_dataset(train_pt_folder, train_info_path)

    # Split dataset in train/val
    dataset_len = len(dataset)
    train_len = round(dataset_len * (1-val_dim))
    val_len = round(dataset_len * val_dim)

    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Define transformations
    initial_trans = transforms.Compose([
        ZeroNormCropsNumber(net_hyperparams['num_crops'])
    ])
    if composed:
        last_trans = transforms.Compose([
            NormScale(standardize=True),
            Compose((6, 4)),
            ToTensor(training=True)
        ])
    else:
        last_trans = transforms.Compose([
            NormScale(standardize=True),
            ToTensor(training=True)
        ])
    val_trans = transforms.Compose([
        initial_trans,
        # DataAugmentation(
        #     contrast=(1.3, 1.3),
        #     saturation=(0.9, 0.9),
        #     only_color=True
        # ),
        last_trans
    ])
    train_trans = transforms.Compose([
        initial_trans,
        DataAugmentation(
            # brightness=(0.8, 2.),
            # contrast=(.5, 1.5),
            # saturation=(0.7, 1.5),
            # hue=(-.3, .1),
            no_color=True
        ),
        last_trans
    ])

    train_set = AugmentDataset(train_set, train_trans)
    val_set = AugmentDataset(val_set, val_trans)

    # Define model
    model = Model(net_hyperparams=net_hyperparams, train_params=train_params)

    # Define train and val loaders
    # Drop last batch because for little dimension of batch the QWK metric explodes
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=train_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=val_workers, drop_last=True)

    # Use in case of start training from saved checkpoint
    last_epoch = -1
    # import torch
    # checkpoint = torch.load('experiments/ResNet34Chia_optimizer.adam_crops.26_batchsize.10_dataset.akensert_freeze.False_lr.1e-06_drop.0.5_loss.binnedbce_patience.15_apex.True_other_net.maxpool/ep_31_checkpoint_0.82542670.pt')
    # model.net.load_state_dict(checkpoint['state_dict'])
    # model.optimizer.load_state_dict(checkpoint['optim_state'])
    # if train_params['use_apex']:
    #     from apex import amp
    #     amp.load_state_dict(checkpoint['apex_state'])

    if lr_range_test:
        from torch_lr_finder import LRFinder
        import json
        network = model.net
        criterion = model.loss
        optimizer = model.optimizer
        lr_finder = LRFinder(network, optimizer, criterion, device='cuda:0')
        lr_finder.range_test(train_loader, val_loader, end_lr=1e-2, num_iter=50, accumulation_steps=1)
        json.dump(lr_finder.history, open('lr_finder.json', 'w'))
        lr_finder.plot()
        lr_finder.reset()

    else:
        run_path = os.path.join('experiments',
                                train_params['network_type'] +
                                '_optimizer.' + train_params['optimizer_type'] +
                                '_crops.' + str(net_hyperparams['num_crops']) +
                                '_batchsize.' + str(batch_size) +
                                '_dataset.' + str(mode) +
                                '_freeze.' + str(train_params['freeze_weights']) +
                                # '_numInitFeatures.' + str(net_hyperparams['num_init_features']) +
                                '_lr.' + str(train_params['lr']) +
                                '_drop.' + str(net_hyperparams['dropout_prob']) +
                                '_loss.' + train_params['loss_type'] +
                                '_patience.' + str(patience) +
                                '_apex.' + str(train_params['use_apex']) +
                                '_other_net.' + 'maxpool')

        os.makedirs(run_path, exist_ok=False)

        # Make backup of network and model files into run folder
        shutil.copy('network.py', run_path)
        shutil.copy('model.py', run_path)
        shutil.copy('train.py', run_path)
        shutil.copy('dataset.py', run_path)
        shutil.copy('pytorchtools.py', run_path)
        shutil.copy('PANDA_functions.py', run_path)

        # Train model
        val_metric = model.fit(15000, train_loader, val_loader, patience, run_path, last_epoch=last_epoch)
        # Clean checkpoint folder from all the checkpoints that are useless
        clean_folder(run_path, val_metric, delta=0.002)



