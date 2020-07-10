from decompress_and_convert import Akensert, InvertColors
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

# Disable max image size for Image
Image.MAX_IMAGE_PIXELS = None


class PANDA_dataset(Dataset):
    def __init__(self, img_folder, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .gz file as a dictionary {patientID: complete_path_to_file}
        self.img_paths = {filename.split('.')[0]: os.path.join(img_folder, filename) for filename in
                          os.listdir(img_folder)}

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
        filename = self.__num_to_id[item]
        scan = np.array(Image.open(self.img_paths[filename]))

        # Create sample
        sample = {
            'filename': filename,
            'scan': scan
        }

        # Transform sample (if defined)
        return self.transform(sample) if self.transform else sample


class ZeroThreshold:
    def __init__(self, zero_threshold=20):
        self.zero_threshold = zero_threshold

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan'].copy()
        scan[scan < self.zero_threshold] = 0
        return {**sample, 'scan': scan}


# Custom transforms:
class EmergencyCrop:
    """
    Crop the original image in smaller sections (crop_dim x crop_dim)
    and eliminates all crops that does not contain relevant information (mostly-blank crops)

    """

    def __init__(self, crop_dim: int):
        self.crop_dim = crop_dim

    def __call__(self, scan):
        height, width, _ = scan.shape
        crops = []
        for i in range(height // self.crop_dim):
            for j in range(width // self.crop_dim):
                crop = scan[i * self.crop_dim:(i + 1) * self.crop_dim, j * self.crop_dim:(j + 1) * self.crop_dim, ...]
                crops.append(crop)
        crops = np.stack(crops, axis=0)
        return crops


class StridedCrop:
    """
    Crop the original image in smaller sections (crop_dim x crop_dim)
    and eliminates all crops that does not contain relevant information (mostly-blank crops)

    """

    def __init__(self, crop_dim, full_percentage, stride=1):
        self.crop_dim = crop_dim
        self.stride = stride
        self.minimum_nonzeros = full_percentage * 3 * self.crop_dim**2
        self.emergency_crop = EmergencyCrop(crop_dim)

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        height, width, _ = scan.shape
        crops = []
        i = 0  # Height index
        j = 0  # Width index
        while i < height - self.crop_dim - 1:
            while j < width - self.crop_dim - 1:
                crop = scan[i:i + self.crop_dim, j:j + self.crop_dim, ...]
                nonzeros = np.count_nonzero(crop)
                if nonzeros == 0:
                    # No values at all, useful for those images which are still full of zeros
                    j += self.crop_dim
                elif nonzeros > self.minimum_nonzeros:
                    crops.append(crop)
                    j += self.crop_dim
                else:
                    j += self.stride
            i += self.crop_dim
            j = 0
        try:
            crops = np.stack(crops, axis=0)
        except ValueError:
            try:
                # print("Try emergency crop for: {}".format(sample['ID']))
                crops = self.emergency_crop(scan)
            except ValueError:
                # print("Could not crop: {}".format(sample['ID']))
                crops = torch.zeros(1, 256, 256, 3).type(torch.uint8)
        return {**sample, 'scan': crops}


class SaveTensor:
    def __init__(self, dest_path):
        self.dest_path = dest_path

    def __call__(self, sample, *args, **kwargs):
        name = sample['filename']
        scan = sample['scan']
        scan = torch.tensor(scan, dtype=torch.uint8)
        s = [(i, s.type(torch.float32).mean()) for i, s in enumerate(scan)]
        s.sort(key=lambda x: x[1], reverse=True)
        indices = list(map(lambda x: x[0], s))
        scan_permuted = torch.zeros_like(scan)
        scan_permuted[indices] = scan
        torch.save(scan, os.path.join(self.dest_path, name + '.pt'))
        return sample


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
                                                          fillcolor=(0, 0, 0))

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


class NormCropsNumber:
    def __init__(self, num_crops):
        self.num_crops = num_crops
        # self.color = torchvision.transforms.ColorJitter(brightness=0, contrast=(0, 3), saturation=(0, 3), hue=(-.2, .2))
        # self.rotate = torchvision.transforms.RandomAffine(360, translate=None, scale=None, shear=None, resample=False,
        #                                                  fillcolor=(255, 255, 255))

    def __call__(self, sample):
        scan = torch.tensor(sample['scan'])

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

    base_path = os.path.join('/opt/local_dataset')
    train_pt_folder = os.path.join(base_path, 'images/type1')
    train_info_path = os.path.join(base_path, 'train.csv')
    mask_path = os.path.join(base_path, 'train_label_masks')
    # mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    # variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define transformations
    # trans = transforms.Compose([Resize((1840, 1728))])
    trans = transforms.Compose([
        InvertColors(),
        Akensert(),
        InvertColors(),
        # ZeroThreshold(20),
        # StridedCrop(256, .50, stride=5),
        SaveTensor(os.path.join(base_path, 'images', 'akensert'))
    ])
    dataset = PANDA_dataset(train_pt_folder, transform=trans)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=12)

    crops = []
    for batch in tqdm(dataloader):
        pass
        # scan = batch['scan'][0]
        # for i, s in enumerate(scan):
        #     Image.fromarray(s.numpy()).save('test{}.jpeg'.format(i))
        # break

    # crops = np.array(crops)
    # ''' print("Number of crops: {}".format(crops.shape[0]))
    # print("Max crops: {}".format(np.max(crops)))
    # print("Min crops: {}".format(np.min(crops)))
    # print("Mean crops: {}".format(np.mean(crops)))'''
    # print(crops)
