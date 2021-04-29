import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import gdown
import zipfile 
import torchvision.transforms as transforms

def download_MNIST(batch_size):
    img_size = 28
    channels = 1
    img_shape = (1, 28, 28)
    N = 50000

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    testloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader, testloader


def download_CELEBA(batch_size):
    data_root = '../../data/celeba'
    # Path to folder with the dataset
    dataset_folder = f'{data_root}/img_align_celeba'
    # URL for the CelebA dataset
    url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
    # Path to download the dataset to
    download_path = f'{data_root}/img_align_celeba.zip'

    # Create required directories 
    if not os.path.exists(data_root):
      os.makedirs(data_root)
      os.makedirs(dataset_folder)

    # Download the dataset from google drive
    gdown.download(url, download_path, quiet=False)

    # Unzip the downloaded file 
    with zipfile.ZipFile(download_path, 'r') as ziphandler:
      ziphandler.extractall(dataset_folder)

    ############################################
    img_size = 64
    channels = 3
    img_shape = (3, 64, 64)
    N = 50000
    #############################################

    os.makedirs("../../data/celeba", exist_ok=True)


    dataset = datasets.ImageFolder(root=data_root,
                              transform=transforms.Compose([
                                  transforms.Resize(img_size),
                                  transforms.CenterCrop(img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ]))
    from torch.utils.data import Subset
    import random
    idxs =  random.sample(range(len(dataset)), 60000)
    random.shuffle(idxs)
    train_idx = idxs[:50000]
    test_idx = idxs[50000:]
    data_sets = {}
    data_sets['train'] = Subset(dataset, train_idx)
    data_sets['test'] = Subset(dataset, test_idx)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(data_sets['train'], batch_size=batch_size,
                                            shuffle=True, num_workers=2)


    testloader = torch.utils.data.DataLoader(data_sets['test'], batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    return dataloader, testloader

