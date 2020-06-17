from dataset import \
    PANDA_dataset, \
    NormScale,\
    DataAugmentation,\
    NormCropsNumber,\
    AugmentDataset,\
    ToTensor
from model import Model
import shutil
from torch.utils.data import DataLoader, random_split
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

    base_path = os.path.join('/opt/local_dataset')
    train_pt_folder = os.path.join(base_path, 'images/cropped')
    train_info_path = os.path.join(base_path, 'train.csv')
    mask_path = os.path.join(base_path, 'train_label_masks')
    # mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    # variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define training hyper parameters
    batch_size = 16

    patience = 10

    net_hyperparams = {
        'dropout_prob': 0.5,
    }
    train_params = {
        'base_lr': 1.7e-5,
        'max_lr': 1e-4,
        'lr': 1e-4,
        'lr_decay': 1.,
        'use_apex': True,
        'weight_decay': 0.,
        'optimizer_type': 'adamw',
        'network_type': 'IAFoss',
        'loss_type': 'binnedbce',
        'binned': True
    }

    # Define training settings
    train_workers = 6
    val_workers = 6
    val_dim = 0.3
    num_crops = 24
    lr_range_test = False
    if lr_range_test:
        train_workers = 0
        val_dim = 0.1

    dataset = PANDA_dataset(train_pt_folder, train_info_path)

    # Split dataset in train/val
    dataset_len = len(dataset)
    train_len = round(dataset_len * (1-val_dim))
    val_len = round(dataset_len * val_dim)

    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Define transformations
    initial_trans = transforms.Compose([
        NormCropsNumber(num_crops)
    ])
    last_trans = transforms.Compose([
        NormScale(standardize=True),
        ToTensor(training=True)
    ])
    val_trans = transforms.Compose([
        initial_trans,
        last_trans
    ])
    train_trans = transforms.Compose([
        initial_trans,
        DataAugmentation(),
        last_trans
    ])

    train_set = AugmentDataset(train_set, train_trans)
    val_set = AugmentDataset(val_set, val_trans)

    # Define model
    model = Model(net_hyperparams=net_hyperparams, train_params=train_params)

    # Define train and val loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=train_workers, collate_fn=model.net.collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=val_workers, collate_fn=model.net.collate_fn)

    # Use in case of start training from saved checkpoint
    # import torch
    # last_epoch = 5
    # checkpoint = torch.load('experiments/CustomResNet18Siamese_numInitFeatures.32_lr.0.004_lr_decay.1.0_drop.0.4_batchsize.11_loss.metric_optimizer.adamw_patience.10_other_net.32outputfeatures/ep_5_checkpoint_0.18078171.pt')
    # model.net.load_state_dict(checkpoint['state_dict'])
    # model.optimizer.load_state_dict(checkpoint['optim_state'])
    # if use_apex:
    #     from apex import amp
    #     amp.load_state_dict(checkpoint['apex_state'])

    if lr_range_test:
        from torch_lr_finder import LRFinder
        import json
        network = model.net
        criterion = model.loss
        optimizer = model.optimizer
        lr_finder = LRFinder(network, optimizer, criterion, device='cuda:0')
        lr_finder.range_test(train_loader, val_loader, end_lr=1e-2, num_iter=100, accumulation_steps=1)
        json.dump(lr_finder.history, open('lr_finder.json', 'w'))
        lr_finder.plot()
        lr_finder.reset()

    else:
        run_path = os.path.join('experiments',
                                train_params['network_type'] +
                                # '_numInitFeatures.' + str(net_hyperparams['num_init_features']) +
                                '_lr.' + str(train_params['lr']) +
                                '_drop.' + str(net_hyperparams['dropout_prob']) +
                                '_batchsize.' + str(batch_size) +
                                '_loss.' + train_params['loss_type'] +
                                '_optimizer.' + train_params['optimizer_type'] +
                                '_patience.' + str(patience) +
                                '_apex.' + str(train_params['use_apex']) +
                                '_other_net.' + '')

        os.makedirs(run_path, exist_ok=False)

        # Make backup of network and model files into run folder
        shutil.copy('network.py', run_path)
        shutil.copy('model.py', run_path)
        shutil.copy('train.py', run_path)
        shutil.copy('dataset.py', run_path)
        shutil.copy('pytorchtools.py', run_path)

        # Train model
        val_metric = model.fit(15000, train_loader, val_loader, patience, run_path, last_epoch=-1)
        # Clean checkpoint folder from all the checkpoints that are useless
        clean_folder(run_path, val_metric, delta=0.002)



