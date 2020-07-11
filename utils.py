import numpy as np
import torch
import os
from PIL import Image
from tifffile import TiffFile
from zipfile import ZipFile
from tqdm import tqdm
from io import BytesIO
import imageio

os.setgid(1000), os.setuid(1000)


if __name__ == '__main__':
    # Check images from original file
    # path_to_compressed_archive = '/opt/whole_dataset/prostate-cancer-grade-assessment.zip'
    # with ZipFile(path_to_compressed_archive, 'r') as zipped_files:  # Open zipped file
    #     for file in tqdm(zipped_files.infolist()):  # Obtain file list
    #         zip_filepath = file.filename  # Extract actual file path
    #         if '0da0915a236f2fc98b299d6fdefe7b8b' in zip_filepath:
    #             img = TiffFile(BytesIO(zipped_files.read(file))).asarray(1)
    #             Image.fromarray(img).save('test.jpeg')
    #             break
    image_folder = '/opt/local_dataset/images/akensert'
    dest_folder = '/opt/local_dataset/images/akensert_little'
    filenames = [filename.split('.')[0] for filename in os.listdir(image_folder)]
    # counter = []
    # means = []
    # means2 = []
    # 'akensert':
    # mean = np.array([0.18482842, 0.38475832, 0.2586024], dtype='float32')[None, None, :]
    # std = np.array([0.15605181, 0.24828894, 0.17223187], dtype='float32')[None, None, :]
    # 'cropped':
    # mean = np.array([0.14274016, 0.3029117, 0.20267214], dtype='float32')[None, None, :]
    # std = np.array([0.16290817, 0.27132016, 0.18850067], dtype='float32')[None, None, :]
    for _, filename in tqdm(enumerate(filenames), total=len(filenames)):
        filepath = os.path.join(image_folder, filename)
        imgs: torch.Tensor = torch.load(filepath + ".pt").numpy()  # Shape is N, 256, 256, 3
        imgs = imgs.reshape(-1, 256, 3)
        Image.fromarray(imgs).save(os.path.join(dest_folder, filename + '.jpeg'))
        pass
        # counter.append(imgs.shape[0])
        # imgs = imgs / 255
        # imgs = (imgs - mean) / std
        # imgs = imgs.type(torch.float32) / 255.
        # avg = imgs.mean([0, 1, 2])
        # avg2 = (imgs**2).mean([0, 1, 2])
        # means.append(avg)
        # means2.append(avg2)
        # torch.save(imgs, os.path.join(dest_folder, filename + 'pt'))
    # mean = torch.stack(means, 0).mean(0)
    # std = torch.sqrt(torch.stack(means2, 0).mean(0) - mean**2)
    # open('mean_std.txt', 'w').write("mean: {}\nstd: {}".format(mean.numpy(), std.numpy()))
    # open('counter.txt', 'w').write(str(counter))


    # Image.fromarray(scan).save('test.jpeg')
    # Image.fromarray(scan).save('test_after.jpeg')