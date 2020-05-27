import os
from h5py import File as h5File
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
import zlib
from torch.utils.data import Dataset
import torch
from torchvision.transforms import Resize
import skimage.io
import torchvision.transforms
from PIL import Image
import random
from gzip import GzipFile


class PANDA_dataset(Dataset):
    def __init__(self, pt_folder, train_info_path=None, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .gz file as a dictionary {patientID: complete_path_to_file}
        self.pt_paths = {filename.split('_')[0]: os.path.join(pt_folder, filename.split('_')[0]) for filename in
                           os.listdir(pt_folder)}

        # Check if dataset is for training or for submission
        if train_info_path:
            train_info = pd.read_csv(train_info_path, index_col=False)
            # train_info.fillna(train_info.mean(), inplace=True)  # Look for NaN values and replace them with column mean
            self.labels = {Id: {'data_provider': list(train_info.loc[train_info['image_id'] == Id]['data_provider'])[0],
                                'isup_grade': list(train_info.loc[train_info['image_id'] == Id]['isup_grade'])[0],
                                'gleason_score': list(train_info.loc[train_info['image_id'] == Id]['gleason_score'])[0]
                                }
                           for Id in self.pt_paths.keys()}
        else:
            self.labels = None

        # Prepare num_to_id in order to address the indexes required from torch API
        self.__num_to_id = {i: k for i, k in enumerate(self.pt_paths.keys())}
        # Create reverse order to have control over dataset patients IDs and indexes
        self.id_to_num = {k: i for i, k in self.__num_to_id.items()}

        print('Dataset loaded!')

        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.pt_paths.keys())

    def __getitem__(self, item):
        # Get the ID corresponding to the item (an index) that torch is looking for.
        id = self.__num_to_id[item]

        # multi_scan = skimage.io.MultiImage(self.pt_paths[id])
        # scan = multi_scan[1]
        with GzipFile(self.pt_paths[id] + '_values.pt.gz') as f:
            scan_values = torch.load(f)
        with GzipFile(self.pt_paths[id] + '_indices.pt.gz') as f:
            scan_indices = torch.load(f)
        scan_size = torch.load(self.pt_paths[id] + '_size.pt')

        scan = torch.sparse_coo_tensor(indices=scan_indices, values=scan_values, size=scan_size, dtype=torch.uint8).to_dense()

        # Create sample
        sample = {
            'ID': id,
            # 'sbm': sbm,
            'scan': scan
        }
        # Add labels to the sample if the dataset is the training one.
        if self.labels:
            sample['label'] = self.labels[id]['isup_grade']

        # Transform sample (if defined)
        return self.transform(sample) if self.transform else sample


# Custom transforms:

class Crop:
    """
    Crop the original image in smaller sections (crop_dim x crop_dim)
    and eliminates all crops that does not contain relevant information (mostly-blank crops)

    """

    def __init__(self, crop_dim: int, threshold_mean):
        self.crop_dim = crop_dim
        self.threshold_mean = threshold_mean * 255

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        height, width, _ = scan.shape
        crops = []

        for i in range(height // self.crop_dim):
            for j in range(width // self.crop_dim):
                crop = scan[i * self.crop_dim:(i + 1) * self.crop_dim, j * self.crop_dim:(j + 1) * self.crop_dim, ...]
                if crop.float().mean() > self.threshold_mean:
                    crops.append(crop)
        crops = torch.stack(crops, dim=0)
        return {**sample, 'scan': crops}


class NormScale:
    """
    Normalize each pixel t assume a value in the range 0-1

    """

    def __init__(self):
        pass

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scan = scan / 255.0

        return {**sample, 'scan': scan.astype('float32')}


class DataAugmentation:
    """

    """

    def __init__(self):
        self.color = torchvision.transforms.ColorJitter(brightness=0, contrast=(0, 3), saturation=(0, 3), hue=(-.2, .2))
        self.rotate = torchvision.transforms.RandomAffine(360, translate=None, scale=None, shear=None, resample=False,
                                                          fillcolor=(255, 255, 255))

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scan = Image.fromarray(scan)
        scan = self.color(scan)
        scan = self.rotate(scan)

        return {**sample, 'scan': np.array(scan).astype('float32')}


class SwapAxes:
    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scan = scan.transpose(1, 3)

        return {**sample, 'scan': np.array(scan).astype('float32')}
#
#
# # Custom transform example
# class ToTensor:
#     def __call__(self, sample):
#         scan = torch.tensor(sample['scan']).float()
#         label = torch.tensor(sample['label']['isup_grade']).int()
#         # Sto togliendo di mezzo le altre informazioni correlate alle labels
#
#         return {**sample, 'scan': scan, 'label': label}

#
# def collate_fn(batch, training=False):
#     """
#     Trasforma righe in colonne.
#     [
#         ['ID': ID0, 'scan': scan0, 'label': label0],
#         ['ID': ID1, 'scan': scan1, 'label': label1].
#         ...
#     ]
#     ---------->
#     [
#     'ID': [ID0, ID1, ...],
#     'scan': [scan0, scan1, ...],
#     'label': [label0, label1, ...],
#     ...]
#     """
#     IDs = []
#     labels = []
#     scans = []
#     for a in batch:
#         ID, scan = a['ID'], a['scan']
#         if training:
#             label = a['label']
#             labels.append(label['isup_grade'])
#         IDs.append(ID)
#         scans.append(torch.tensor(scan))
#
#     sample = {
#         'ID': IDs,
#         'scan': scans,
#     }
#
#     if training:
#         sample['label'] = torch.tensor(labels)
#
#     return sample


class NormCropsNumber:
    def __init__(self, num_crops):
        self.num_crops = num_crops
        #self.color = torchvision.transforms.ColorJitter(brightness=0, contrast=(0, 3), saturation=(0, 3), hue=(-.2, .2))
        #self.rotate = torchvision.transforms.RandomAffine(360, translate=None, scale=None, shear=None, resample=False,
        #                                                  fillcolor=(255, 255, 255))

    def __call__(self, sample):
        scan = sample['scan']

        while scan.shape[0] <= self.num_crops:
            scan = torch.cat([scan, scan], dim=0)

        if scan.shape[0] > self.num_crops:
            indexes = list(range(scan.shape[0]))
            random.shuffle(indexes)
            indexes = indexes[:self.num_crops]
            scan = torch.index_select(scan, dim=0, index=torch.tensor(indexes, dtype=torch.int64))

        return {**sample, 'scan': scan}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    base_path = os.path.join('..', 'dataset')
    train_pt_folder = os.path.join(base_path, 'train_images_torch')
    train_info_path = os.path.join(base_path, 'train.csv')
    mask_path = os.path.join(base_path, 'train_label_masks')
    # mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    # variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define transformations
    # trans = transforms.Compose([Resize((1840, 1728))])
    trans = transforms.Compose([Crop(256, .05), NormCropsNumber(25)])
    dataset = PANDA_dataset(train_pt_folder, transform=trans)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    crops = []
    for batch in tqdm(dataloader):
        crops.append((batch['ID'][0], batch['scan'][0].shape[0]))

    # crops = np.array(crops)
    ''' print("Number of crops: {}".format(crops.shape[0]))
    print("Max crops: {}".format(np.max(crops)))
    print("Min crops: {}".format(np.min(crops)))
    print("Mean crops: {}".format(np.mean(crops)))'''
    print(crops)
