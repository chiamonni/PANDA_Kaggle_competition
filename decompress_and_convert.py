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


if __name__ == '__main__':
    path_to_compressed_archive = 'D:\\datasets\\PANDA_dataset\\prostate-cancer-grade-assessment.zip'
    path_to_dest_dataset = '../dataset/train_images_torch'

    with ZipFile(path_to_compressed_archive, 'r') as zipped_files:  # Open zipped file
        for file in tqdm(zipped_files.infolist()):  # Obtain file list
            zip_filepath = file.filename  # Extract actual file path
            if 'train_images' in zip_filepath and zip_filepath.endswith('.tiff'):  # Leave masks
                # Open tiff image and obtain second layer (in the 'asarray' parameter)
                img = TiffFile(BytesIO(zipped_files.read(file))).asarray(1)
                # Transform to torch tensor to do operations
                t_img = torch.tensor(img, dtype=torch.uint8)
                # Calculate its opposite to cast the images to sparse representation
                t_img_negative = 255 - t_img
                # Calculate sparse tensor
                t_img_negative_sparse = t_img_negative.to_sparse()
                # Cast to right data type to spare space
                indices = t_img_negative_sparse.indices().type(torch.int16)
                values = t_img_negative_sparse.values().type(torch.uint8)
                size = t_img_negative_sparse.size()
                # Create new h5 file path
                new_filepath = os.path.join(path_to_dest_dataset, file.filename.split('/')[-1].split('.')[0])
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
