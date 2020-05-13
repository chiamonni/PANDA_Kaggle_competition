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


class PANDA_dataset(Dataset):
    def __init__(self, tiff_folder, train_info_path = None, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .gz file as a dictionary {patientID: complete_path_to_file}
        self.tiff_paths = {filename.split('.')[0]: os.path.join(tiff_folder, filename) for filename in os.listdir(tiff_folder)}

        # Check if dataset is for training or for submission
        if train_info_path:
            train_info = pd.read_csv(train_info_path, index_col=False)
            #train_info.fillna(train_info.mean(), inplace=True)  # Look for NaN values and replace them with column mean
            self.labels = {Id: {'data_provider': list(train_info.loc[train_info['image_id'] == Id]['data_provider'])[0],
                                'isup_grade': list(train_info.loc[train_info['image_id'] == Id]['isup_grade'])[0],
                                'gleason_score': list(train_info.loc[train_info['image_id'] == Id]['gleason_score'])[0]
                                }
                           for Id in self.tiff_paths.keys()}
        else:
            self.labels = None

        # Test code to verify if there are all the labels for each type of data
        # fnc_keys = list(fnc['Id'])
        # sbm_keys = list(sbm['Id'])
        # print(len(gz_keys), len(fnc_keys), len(sbm_keys))
        # fnc_missing = []
        # sbm_missing = []
        # for k in gz_keys:
        #     if k not in fnc_keys:
        #         fnc_missing.append(k)
        #     if k not in sbm_keys:
        #         sbm_missing.append(k)
        # print(fnc_missing, sbm_missing)
        # pass

        # self.mask = np.array(nib.load(mask_path))

        # Prepare num_to_id in order to address the indexes required from torch API
        self.__num_to_id = {i: k for i, k in enumerate(self.tiff_paths.keys())}
        # Create reverse order to have control over dataset patients IDs and indexes
        self.id_to_num = {k: i for i, k in self.__num_to_id.items()}

        print('Dataset loaded!')

        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.tiff_paths.keys())

    def __getitem__(self, item):
        # Get the ID corresponding to the item (an index) that torch is looking for.
        id = self.__num_to_id[item]
        # Retrieve all information from the Dataset initialization
        # Keep brain commented until not working on 3D images
        # brain =
        # brain = np.frombuffer(zlib.decompress(open(self.gz_paths[ID], 'rb').read()), dtype='float64').reshape(53, 52, 63, 53)
        # brain = None

        # sbm = self.sbm[ID]

        multi_scan = skimage.io.MultiImage(self.tiff_paths[id])
        scan = multi_scan[1]

        # numero canali come prima dimensione per fare conv2D:
        scan = np.swapaxes(scan, 1, 2)
        scan = np.swapaxes(scan, 0, 1)

        # Create sample
        sample = {
            'ID': id,
            #'sbm': sbm,
            'scan': scan
        }
        # if self.fnc:
        #    sample['fnc'] = self.fnc[ID]
        # Add labels to the sample if the dataset is the training one.
        if self.labels:
            sample['label'] = self.labels[id]

        # Transform sample (if defined)
        return self.transform(sample) if self.transform else sample


# Custom transforms:

class Crop:
    """
    Crop the original image in smaller sections (crop_dim x crop_dim)
    and eliminates all crops that does not contain relevant information (mostly-blank crops)

    """
    def __init__(self, crop_dim:int, threshold_mean):
        self.crop_dim = crop_dim
        self.threshold_mean = threshold_mean

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        channnels, height, width = scan.shape
        crops = []
        for i in range(height//self.crop_dim)[1:]:
            for j in range(width//self.crop_dim)[1:]:
                crop = scan[:, (i-1)*self.crop_dim:i*self.crop_dim, (j-1)*self.crop_dim:j*self.crop_dim]
                if crop.mean() < self.threshold_mean:
                    crops.append(crop)
                    break  # TODO: RIMUOVI I BREAKs SE VUOI ALLENARE
            if len(crops) > 0:
                break
        #scan = np.stack(crops, axis=-1)
        return {**sample, 'scan': crops}


class NormScale:
    """
    Normalize each pixel t assume a value in the range 0-1

    """

    def __init__(self):
        pass

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scan: np.ndarray = scan / 255.0

        return {**sample, 'scan': scan.astype('float32')}


# Custom transform example
class ToTensor:
    def __call__(self, sample):
        scan = torch.tensor(sample['scan']).float()
        label = torch.tensor(sample['label']['isup_grade']).int()
        # Sto togliendo di mezzo le altre informazioni correlate alle labels

        return {**sample, 'scan': scan, 'label': label}

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    base_path = '..'
    train_tiff_folder = os.path.join(base_path, 'dataset/train_images')
    train_info_path = os.path.join(base_path, 'dataset/train.csv')
    mask_path = os.path.join(base_path, 'dataset/train_label_masks')
    # mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    # variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define transformations
    # trans = transforms.Compose([Resize((1840, 1728))])
    trans = transforms.Compose([NormScale(), Crop(256, .95)])

    dataset = PANDA_dataset(train_tiff_folder, train_info_path, transform=trans)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    for batch in dataloader:
        print(len(batch['scan']))
        print(batch['label'])
        break



