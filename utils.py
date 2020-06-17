import numpy as np
import torch
import os
from PIL import Image
from tifffile import TiffFile
from zipfile import ZipFile
from tqdm import tqdm
from io import BytesIO

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
    image_folder = '/opt/local_dataset/images/cropped'
    filenames = [filename.split('.')[0] for filename in os.listdir(image_folder)]
    counter = []
    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        filepath = os.path.join(image_folder, filename)
        imgs = torch.load(filepath + ".pt")
        counter.append(imgs.shape[0])
    open('counters.txt', 'w').write(str(counter))


    # Image.fromarray(scan).save('test.jpeg')
    # Image.fromarray(scan).save('test_after.jpeg')