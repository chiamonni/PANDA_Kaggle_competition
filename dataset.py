import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision.transforms import \
    Resize, \
    RandomRotation
import torchvision.transforms
from PIL import Image
import random

os.setgid(1000), os.setuid(1000)


class PANDA_dataset(Dataset):
    def __init__(self, img_folder, train_info_path=None, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .gz file as a dictionary {patientID: complete_path_to_file}
        self.img_paths = {filename.split('.')[0]: os.path.join(img_folder, filename) for filename in
                          os.listdir(img_folder)}

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
        self.mean = np.array([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304], dtype='float32')[None, None, None, :]
        self.std = np.array([0.36357649, 0.49984502, 0.40477625], dtype='float32')[None, None, None, :]

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan'].astype('float32')
        scan = scan / np.array(255.0, dtype='float32')
        if self.standardize:
            scan = (scan - self.mean) / self.std

        return {**sample, 'scan': scan}


class DataAugmentation:
    """

    """

    def __init__(self, brightness=(0, 2), contrast=0, saturation=(0, 2), hue=(-.5, .5), rotation=180):
        self.color = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.rotate = torchvision.transforms.RandomAffine(rotation, translate=None, scale=None, shear=None, resample=False,
                                                          fillcolor=(0, 0, 0))

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scans = []
        for s in scan:
            s = Image.fromarray(s)
            s = self.color(s)
            s = self.rotate(s)
            scans.append(s)
        scan = np.stack(scans, axis=0)

        return {**sample, 'scan': scan}


class NormCropsNumber:
    def __init__(self, num_crops):
        self.num_crops = num_crops

    def __call__(self, sample):
        scan = sample['scan']

        while scan.shape[0] <= self.num_crops:
            scan = np.concatenate([scan, scan], axis=0)

        if scan.shape[0] > self.num_crops:
            indices = list(range(scan.shape[0]))
            random.shuffle(indices)
            indices = indices[:self.num_crops]
            scan = scan[indices]

        return {**sample, 'scan': scan}


class Compose:
    """
    Class to compose the crops into a single image
    """
    def __init__(self, crop_size, arrangement=(4, 4)):
        self.crop_size = crop_size
        self.arrangement = arrangement

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        num_crops = scan.shape[0]
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
        scan = torch.tensor(sample['scan'], dtype=torch.float32).transpose(-1, 1)
        if self.training:
            label = torch.tensor(sample['label'], dtype=torch.int64)
            return scan, label
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
        NormCropsNumber(80),
        DataAugmentation(),
        NormScale(standardize=True),
        ToTensor(training=False)
    ])
    dataset = PANDA_dataset(train_pt_folder, transform=trans)
    dataloader = DataLoader(dataset, num_workers=0)

    crops = []
    for batch in tqdm(dataloader):
        pass

    # crops = np.array(crops)
    ''' print("Number of crops: {}".format(crops.shape[0]))
    print("Max crops: {}".format(np.max(crops)))
    print("Min crops: {}".format(np.min(crops)))
    print("Mean crops: {}".format(np.mean(crops)))'''
    print(crops)
