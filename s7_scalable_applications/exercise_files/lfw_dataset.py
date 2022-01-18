"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import re
import torch
import tarfile
import io
from PIL import Image
from torchvision.utils import make_grid
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        
        self.transform = transform
        self.path_to_folder = path_to_folder
        self.tarfile = tarfile.open(path_to_folder + "lfw.tgz", "r:gz")
        self.filelist = []
        for i in self.tarfile.getmembers():
            if ".jpg" in str(i):
                self.filelist.append(re.search("\'(.*)\'",str(i))[0].strip("\'"))
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index: int) -> torch.Tensor:

        name = self.filelist[index]
        image = self.tarfile.extractfile(name)
        image = image.read()
        image = Image.open(io.BytesIO(image))

        if self.transform is not None:
            image = self.transform(image)

        if index == (self.__len__() - 1):
            self.tarfile.close()
        
        return image


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='C:/Users/Marku/OneDrive - Danmarks Tekniske Universitet/Studie/7. Semester/Machine Learning Operations/mlops_exercises/data/lfw/', type=str)
    parser.add_argument('-batch_size', default=7, type=int)
    parser.add_argument('-num_workers', default=6, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        for imgs in dataloader:
            imgs = [imgs]
        imgs = next(iter(dataloader))
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')
