"""
In this file I'm going to open the .zip file and to extract the .tiff train images in the form of sparse tensors.
In this way I'm able to spare up to 1/10 of space - since most images are plenty of 255.

These are the steps:
1. without unzipping the original dataset, I load the information and open only the training .tiff images, the second layer.
2. I calculate the opposite by subtracting 255
3. I calculate the indices, values and size of the original image
4. I cast every tensor to its fitter representation (no indexes above 10K for second layer tiff images)
4. (optional) I reconstruct the image and check if they are equal (checked for 100 images without errors)
5. I save the three tensors for each image with the original name, followed by '_values.pt.gz', '_indices.pt.gz', '_size.pt'
(The .gz is for the gzip compressed files, since still too heavy)
"""

from tifffile import TiffFile
from zipfile import ZipFile
from io import BytesIO
import os
import torch
from tqdm import tqdm
import gzip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from manage_dataset import ZeroThreshold
from akensert_transforms import *

os.setgid(1000), os.setuid(1000)


def crop_image_only_outside(img, tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def save_to_sparse(img, filename, dest_path):
    # Transform to torch tensor to do operations
    img = torch.tensor(img, dtype=torch.uint8)
    # Calculate sparse tensor
    # Indices
    indices = torch.nonzero(img, as_tuple=True)
    # Cast to right data type to spare space
    values = img[indices].type(torch.uint8)
    indices = torch.stack(indices, 1).type(torch.int32)
    size = img.shape
    # Create new h5 file path
    new_filepath = os.path.join(dest_path, filename)
    # (Optional: check if images are reconstructed in the right way)
    # reconstruct_img = torch.sparse_coo_tensor(indices, values, size, dtype=torch.uint8).to_dense()
    # assert t_img_negative.equal(reconstruct_img)
    # torch.save(torch.tensor(img, dtype=torch.uint8), new_filepath + '.pt')
    # Assign new names for indices, values and size
    indices_path = new_filepath + '_indices.pt.gz'
    values_path = new_filepath + '_values.pt.gz'
    size_path = new_filepath + '_size.pt'
    # Save the tensors to disk
    with gzip.GzipFile(indices_path, 'w', compresslevel=1) as f:
        torch.save(indices, f)
    with gzip.GzipFile(values_path, 'w', compresslevel=1) as f:
        torch.save(values, f)
    torch.save(size, size_path)


def save_torch(img, filename, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    torch.save(img, os.path.join(dest_path, filename + '.pt'))


class ManageDataset(Dataset):
    def __init__(self, archive_path, paths_list, level=1, transforms=None):
        Dataset.__init__(self)
        self.archive_path = archive_path
        self.paths_list = paths_list
        self.level = level
        self.transform = transforms

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, item):
        with ZipFile(self.archive_path, 'r') as zipped_files:
            file = self.paths_list[item]
            # Open tiff image and obtain second layer (in the 'asarray' parameter).
            scan = TiffFile(BytesIO(zipped_files.read(file))).asarray(self.level)
        # Retrieve filename
        filename = file.filename.split('/')[-1].split('.')[0]
        # Create sample
        sample = {
            'filename': filename,
            'scan': scan
        }
        return self.transform(sample) if self.transform else sample


class InvertColors:
    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        return {**sample, 'scan': 255 - scan}


class Akensert:
    def __init__(self, patch_size=256, min_patch_info = .35, min_axis_info=0.35, min_consec_axis_info=0.35, min_decimal_keep=0.7):
        self.patch_size = patch_size
        self.min_patch_info = min_patch_info
        self.min_axis_info = min_axis_info
        self.min_consec_axis_info = min_consec_axis_info
        self.min_decimal_keep = min_decimal_keep

    def __call__(self, sample, *args, **kwargs):
        image = sample['scan']
        # assert image.max() == 255, "Keep original images for better thresholds"
        image, coords = compute_coords(image,
                       patch_size=self.patch_size,
                       precompute=False,
                       min_patch_info=self.min_patch_info,
                       min_axis_info=self.min_axis_info,
                       min_consec_axis_info=self.min_consec_axis_info,
                       min_decimal_keep=self.min_decimal_keep)
        crops = []
        h, w, _ = image.shape
        for i, coord in enumerate(coords):
            v, y, x = coord
            if y < 0: y = 0
            if x < 0: x = 0
            if y > h-self.patch_size: y = h-self.patch_size
            if x > w-self.patch_size: x = w-self.patch_size
            img = image[y:y + self.patch_size, x:x+self.patch_size]
            # Image.fromarray(img).save('test{}_{}.jpeg'.format(i, v))
            crops.append(img)
        scan = np.stack(crops, 0)
        return {**sample, 'scan': scan}


class RemoveBorders:
    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scan = crop_image_only_outside(scan)
        return {**sample, 'scan': scan}


class SaveToDisk:
    def __init__(self, dest_folder, patch_size=128):
        self.dest_folder = dest_folder
        self.patch_size = patch_size

    def __call__(self, sample, *args, **kwargs):
        filename = sample['filename']
        scan = sample['scan']
        # scan = sample['scan'].reshape(-1, self.patch_size, 3)
        os.makedirs(self.dest_folder, exist_ok=True)
        Image.fromarray(scan).save(os.path.join(self.dest_folder, filename + '.jpeg'))
        return sample


class ToTensor:
    def __call__(self, sample, *args, **kwargs):
        filename = sample['filename']
        scan = sample['scan']
        return filename, scan


class Resize:
    def __init__(self, resize_dim=(128, 128)):
        self.resizer = transforms.Resize(resize_dim, Image.BICUBIC)

    def __call__(self, sample, *args, **kwargs):
        scan = sample['scan']
        scans = []
        for s in scan:
            scans.append(np.asarray(self.resizer(Image.fromarray(s))))
        scan = np.stack(scans, 0)
        return {**sample, 'scan': scan}


if __name__ == '__main__':
    path_to_compressed_archive = '/opt/whole_dataset/prostate-cancer-grade-assessment.zip'
    path_to_dest_dataset = '/opt/local_dataset/images/type2'
    paths_list = []
    with ZipFile(path_to_compressed_archive, 'r') as zipped_files:  # Open zipped file
        for file in tqdm(zipped_files.infolist()):  # Obtain file list
            zip_filepath = file.filename  # Extract actual file path
            if 'train_images' in zip_filepath and zip_filepath.endswith('.tiff'):  # Leave masks
                paths_list.append(file)

    trans = transforms.Compose([
        InvertColors(),
        ZeroThreshold(),
        RemoveBorders(),
        SaveToDisk(path_to_dest_dataset)
        # ZeroThreshold(20),
        # StridedCrop(1024, 0.50, stride=20),
        # Resize((512, 512)),
        # SaveTensor(os.path.join(path_to_dest_dataset, 'crop1024to512T09'))
        # ToTensor()
    ])

    dataset = ManageDataset(path_to_compressed_archive, paths_list, level=2, transforms=trans)
    loader = DataLoader(dataset, shuffle=False, num_workers=2)
    for batch in tqdm(loader, desc='Processing images'):
        # for i, s in enumerate(batch[0][1]):
        #     Image.fromarray(s.numpy()).save('test{}.jpeg'.format(i))
        # break
        pass
