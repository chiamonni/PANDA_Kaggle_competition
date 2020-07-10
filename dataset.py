import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision.transforms import RandomAffine, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, Normalize
from PIL import Image
import random
from typing import Union, Tuple

os.setgid(1000), os.setuid(1000)


class PANDA_dataset(Dataset):
    def __init__(self, img_folder, train_info_path=None, dataset_quantity=1.0, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .gz file as a dictionary {patientID: complete_path_to_file}
        self.img_paths = {filename.split('.')[0]: os.path.join(img_folder, filename) for filename in
                          os.listdir(img_folder)}
        self.img_paths = {item[0]: item[1] for i, item in enumerate(self.img_paths.items()) if i < len(self.img_paths)*dataset_quantity}

        # Check if dataset is for training or for submission
        if train_info_path:
            train_info = pd.read_csv(train_info_path, index_col=False)
            # train_info.fillna(train_info.mean(), inplace=True)  # Look for NaN values and replace them with column mean
            self.labels = {Id: {
                # 'data_provider': list(train_info.loc[train_info['image_id'] == Id]['data_provider'])[0],
                'isup_grade': list(train_info.loc[train_info['image_id'] == Id]['isup_grade'])[0],
                # 'gleason_score': list(train_info.loc[train_info['image_id'] == Id]['gleason_score'])[0]
            }
                for Id in self.img_paths.keys()}
        else:
            self.labels = None

        # Prepare num_to_id in order to address the indexes required from torch API
        self.__num_to_id = {i: k for i, k in enumerate(self.img_paths.keys())}
        # Create reverse order to have control over dataset patients IDs and indexes
        self.id_to_num = {k: i for i, k in self.__num_to_id.items()}

        print('Dataset loaded!')

        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.img_paths.keys())

    def __getitem__(self, item):
        # Get the ID corresponding to the item (an index) that torch is looking for.
        id = self.__num_to_id[item]

        scan = torch.load(self.img_paths[id]).numpy()
        # Create sample
        sample = {
            'ID': id,
            'scan': scan
        }
        if self.labels:
            sample['label'] = self.labels[id]['isup_grade']

        # Transform sample (if defined)
        return self.transform(sample) if self.transform else sample


class NormScale:
    """
    Normalize each pixel t assume a value in the range 0-1

    """

    def __init__(self, standardize=True):
        self.standardize = standardize
        # EfficientNet/ResNet weights
        self.mean = np.array([0.485, 0.456, 0.406], dtype='float32')[None, None, :]
        self.std = np.array([0.229, 0.224, 0.225], dtype='float32')[None, None, :]
        # akensert
        # mean = np.array([0.18482842, 0.38475832, 0.2586024], dtype='float32')[None, None, :]
        # std = np.array([0.15605181, 0.24828894, 0.17223187], dtype='float32')[None, None, :]
        # Cropped
        # mean = np.array([0.14274016, 0.3029117, 0.20267214], dtype='float32')[None, None, :]
        # std = np.array([0.16290817, 0.27132016, 0.18850067], dtype='float32')[None, None, :]

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan'].astype('float32').copy()
        if self.standardize:
            scan = scan / 255.
            scan = (scan - self.mean) / self.std
        return {**sample, 'scan': scan}




class DataAugmentation:
    """

    """

    def __init__(self,
                 # brightness=(0.5, 2),
                 brightness: Union[Tuple[float, float], float] = 0,
                 # contrast=(0.5, 2.5),
                 contrast: Union[Tuple[float, float], float] = 0,
                 # saturation=(0.5, 3),
                 saturation: Union[Tuple[float, float], float] = 0,
                 hue: Union[Tuple[float, float], float] = 0,
                 # hue=0.,
                 rotation=180,
                 only_color=False,
                 no_color=False
                 ):
        self.color = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.hf = RandomHorizontalFlip(p=0.5)
        self.vf = RandomVerticalFlip(p=0.5)
        self.rotate = RandomRotation(rotation, expand=False)
        self.only_color = only_color
        self.no_color = no_color
        # self.rotate = RandomAffine(rotation, translate=None, scale=None, shear=None, resample=False,
        #                                                   fillcolor=(0, 0, 0))

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scan = [Image.fromarray(s) for s in scan]
        if self.only_color:
            transform = lambda x: self.color(x)
        elif self.no_color:
            transform = lambda x: self.rotate(self.hf(self.vf(x)))
        else:
            transform = lambda x: self.rotate(self.hf(self.vf(self.color(x))))
        scan = np.stack([transform(s) for s in scan], 0)
        # for s in scan:
            # s = self.color(s)
            # s = self.hf(s)
            # s = self.vf(s)
            # s = self.rotate(s)
            # scans.append(s)
        # scan = np.stack(scans, axis=0)

        return {**sample, 'scan': scan}


class NormCropsNumber:
    def __init__(self, num_crops):
        self.num_crops = num_crops

    def __call__(self, sample):
        scan = sample['scan']

        while scan.shape[0] <= self.num_crops:
            scan = np.concatenate([scan.copy(), scan.copy()], axis=0)

        if scan.shape[0] > self.num_crops:
            indices = list(range(scan.shape[0]))
            random.shuffle(indices)
            indices = indices[:self.num_crops]
            scan = scan[indices]

        return {**sample, 'scan': scan}


class ZeroNormCropsNumber:
    def __init__(self, num_crops):
        self.num_crops = num_crops

    def __call__(self, sample):
        scan = sample['scan']
        # Duplication part
        # if scan.shape[0] < self.num_crops // 2:
        #     new_scan = scan
        #     while new_scan.shape[0] <= self.num_crops - scan.shape[0]:
        #        new_scan = np.concatenate([new_scan, scan], axis=0)
        #     scan = new_scan
        shape = scan.shape
        if shape[0] < self.num_crops:
            scan = [s for s in scan]
            zero = [np.zeros(shape[1:], dtype='uint8') for _ in range(self.num_crops - shape[0])]
            scan += zero
            random.shuffle(scan)
            scan = np.stack(scan, axis=0)

        elif shape[0] > self.num_crops:
            indices = list(range(scan.shape[0]))
            random.shuffle(indices)
            indices = indices[:self.num_crops]
            scan = scan[indices]

        return {**sample, 'scan': scan}


class Compose:
    """
    Class to compose the crops into a single image
    """
    def __init__(self, arrangement=(4, 4)):
        self.arrangement = arrangement

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        num_crops, _, _, _ = scan.shape
        assert num_crops == self.arrangement[0] * self.arrangement[1]
        imgs = []
        for i in range(self.arrangement[0]):
            imgs.append(np.concatenate(scan[range(i*self.arrangement[1], (i + 1) * self.arrangement[1])], axis=0))
        imgs = np.concatenate(imgs, axis=1)

        # Image.fromarray(imgs).save('test.jpg')
        return {**sample, 'scan': imgs}


class ToTensor:
    def __init__(self, training=True):
        self.training = training

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scan_shape_len = len(scan.shape)
        if scan_shape_len == 4:  # Not composed images
            ax = 1
        elif scan_shape_len == 3:  # Composed images
            ax = 0
        else:
            raise ValueError('Bad shape type')
        scan = torch.tensor(scan.copy(), dtype=torch.float32).transpose(-1, ax)
        if self.training:
            label = torch.tensor(sample['label'], dtype=torch.int64)
            return scan, label, sample['ID']
        else:
            return scan, sample['ID']


class AugmentDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, aug_fn):
        self.dataset = dataset
        self.aug_fn = aug_fn

    def __getitem__(self, index):
        return self.aug_fn(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    base_path = os.path.join('/opt/local_dataset')
    train_pt_folder = os.path.join(base_path, 'images/cropped')
    train_info_path = os.path.join(base_path, 'train.csv')
    mask_path = os.path.join(base_path, 'train_label_masks')
    # mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    # variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define transformations
    # trans = transforms.Compose([Resize((1840, 1728))])
    trans = transforms.Compose([
        ZeroNormCropsNumber(64),
        DataAugmentation(
            # contrast=(1.5, 1.5),
            # saturation=(0.7, 0.7),
            # only_color=True
            no_color=True
        ),
        Compose((8, 8)),
        # NormScale(standardize=True),
        # ToTensor(training=False)
    ])
    dataset = PANDA_dataset(train_pt_folder, transform=trans)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=0)

    crops = []
    for batch in tqdm(dataloader):
        Image.fromarray(batch['scan'][0].numpy()).save('test.jpeg')
        break

    # crops = np.array(crops)
    # ''' print("Number of crops: {}".format(crops.shape[0]))
    # print("Max crops: {}".format(np.max(crops)))
    # print("Min crops: {}".format(np.min(crops)))
    # print("Mean crops: {}".format(np.mean(crops)))'''
    # print(crops)
