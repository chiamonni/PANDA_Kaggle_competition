from model import Model
from dataset import PANDA_dataset, NormScale, Crop, ToTensor
from network import DenseNet201
import shutil
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os


if __name__ == '__main__':
    # Define paths

    base_path = '..'
    train_tiff_folder = os.path.join(base_path, 'dataset/train_images')
    train_info_path = os.path.join(base_path, 'dataset/train.csv')
    mask_path = os.path.join(base_path, 'dataset/train_label_masks')

    # Define transformations
    # crop --> dimensione immagine croppata, threshold per decidere se scartare o tenere l'immagine
    trans = transforms.Compose([NormScale(), Crop(256, .95), ToTensor()])

    #create dataset
    dataset = PANDA_dataset(train_tiff_folder, train_info_path, transform=trans)


    # Split dataset in train/val
    val_dim = 0.3
    dataset_len = len(dataset)
    train_len = round(dataset_len * (1-val_dim))
    val_len = round(dataset_len * val_dim)

    train_set, val_set = random_split(dataset, [train_len, val_len])

    # define hyperparameters
    optimizer = 'adam'
    learning_rate = 1e-3
    batch_size = 1


    # Define train and val loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # Define model
    model = Model(optimizer, lr=learning_rate)


    # Print model network
    # summary(model.net, {'sbm': '10001', 'brain': (53, 52, 63, 53)})

    run_path = os.path.join('experiments',
                            #network_type +
                            #'_loss.' + loss +
                            '_batchsize.' + str(batch_size) +
                            '_optimizer.' + optimizer +
                            '_lr.' + str(learning_rate)
                            #'_drop.' + str(dropout_prob) +
                            #'_patience.' + str(patience) +
                            #'_numInitFeatures.' + str(num_init_features) +
                            #'_other_net.' + str(0))
                            )

   # os.makedirs(run_path, exist_ok=False)

    # Train model
    val_metric = model.fit(15000, train_loader, val_loader)
    # Clean checkpoint folder from all the checkpoints that are useless
    #clean_folder(run_path, val_metric, delta=0.002)

    # Make backup of network and model files into run folder
    #shutil.copy('network.py', run_path)
    #shutil.copy('model.py', run_path)
    #shutil.copy('train.py', run_path)
    #shutil.copy('dataset.py', run_path)
    #shutil.copy('pytorchtools.py', run_path)
    #shutil.copy('DenseNet3D.py', run_path)
    #shutil.copy('ResNet.py', run_path)



