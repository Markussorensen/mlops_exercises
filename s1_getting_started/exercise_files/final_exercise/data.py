import torch
import numpy as np
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import helper
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self, labels, images, transform=None):

        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def extract_data(npz_dir):
    npz_file = np.load(npz_dir)
    images = npz_file['images']
    images = torch.from_numpy(images)
    images = images.view(images.shape[0], -1)

    labels = npz_file['labels']
    labels = torch.from_numpy(labels)
    
    return images, labels


def mnist():
    # exchange with the corrupted mnist dataset
    data_dir = "C:/Users/Marku/OneDrive - Danmarks Tekniske Universitet/Studie/7. Semester/Machine Learning Operations/dtu_mlops/data/corruptmnist/"
    files = os.listdir(data_dir)

    test_images, test_labels = extract_data(data_dir + files[0])

    all_train_images = torch.Tensor([])
    for train_file in files[1:]:
        train_images, train_labels = extract_data(data_dir + train_file)
        all_train_images = torch.cat((all_train_images, train_images), 0)

    # transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
    trainset = MNISTDataset(train_labels, train_images)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = MNISTDataset(test_labels, test_images)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)
    
    return trainloader, testloader


if __name__ == '__main__':
    trainloader, testloader = mnist()
    for i in range(100):
        images, labels = next(iter(testloader))
        img_dim = int(np.sqrt(images.shape[1]))
        print(labels[0], i)
        helper.imshow(images.view(-1,1,img_dim,img_dim)[0,:])


    print(images.shape)



