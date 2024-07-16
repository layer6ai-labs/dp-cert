import os
from pathlib import Path
from typing import Tuple, Any

import PIL
import numpy as np
import pandas as pd
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset


class CelebA(Dataset):
    """
    CelebA PyTorch dataset
    The built-in PyTorch dataset for CelebA is outdated.
    """

    def __init__(self, root: str, role: str = "train", seed: int = 0, device: str = "cpu"):
        self.root = Path(root)
        self.role = role
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        celeb_path = lambda x: self.root / x

        role_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        splits_df = pd.read_csv(celeb_path("list_eval_partition.csv"))
        fields = ['image_id', 'Male']
        attrs_df = pd.read_csv(celeb_path("list_attr_celeba.csv"), usecols=fields)
        df = pd.merge(splits_df, attrs_df, on='image_id')
        df = df[df['partition'] == role_map[self.role]].drop(labels='partition', axis=1)
        df = df.replace(to_replace=-1, value=0)

        if seed:
            # Shuffle order according to seed but keep standard partition because the same person appears multiple times
            state = np.random.default_rng(seed=seed)
            df = df.sample(frac=1, random_state=state)

        self.filename = df["image_id"].tolist()
        # Male is 1, Female is 0
        self.y = torch.Tensor(df["Male"].values).long()
        self.image_idx = torch.arange(len(self.filename))

        self.shape = (len(self.filename), 3, 64, 64)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = (self.root / "img_align_celeba" /
                    "img_align_celeba" / self.filename[index])
        x = PIL.Image.open(img_path)
        x = self.transform(x).to(self.device)
        y = self.y[index].to(self.device)
        id = self.image_idx[index]

        return x, y, id

    def __len__(self) -> int:
        return len(self.filename)

    def to(self, device):
        self.device = device
        return self

class AugmentedDataset(Dataset):
    def __init__(self, images, labels, image_idx, num_augmentations, noise_std, device):

        self.tensors = images
        self.targets = labels
        self.image_idx = image_idx
        self.num_augmentations = num_augmentations
        self.noise_std = noise_std
        self.device = device

    def __getitem__(self, index):
        img, target = self.tensors[index], int(self.targets[index])
        noisy_imgs = torch.normal(img.expand([self.num_augmentations+1]+ list(img.shape)), self.noise_std)
        noisy_imgs[0] = img
        targets = torch.full((self.num_augmentations + 1,), target)

        return noisy_imgs.to(self.device), targets.to(self.device), self.image_idx[index].to(self.device)

    def __len__(self) -> int:
        return len(self.tensors)

def image_tensors_to_dataset(images, labels, image_idx, device):
    images = images.to(device)
    labels = labels.to(device)
    image_idx = image_idx.to(device)

    return TensorDataset(images, labels, image_idx)

def image_tensors_to_augmented_dataset(images, labels, image_idx, device, num_augmentations, noise_variance):
    return AugmentedDataset(images, labels, image_idx, num_augmentations, noise_variance, device)

# Returns tuple of form `(images, labels)`.
# `images` has shape `(nimages, nchannels, nrows, ncols)`, and has
# entries in {0, ..., 1}
def get_raw_image_tensors(dataset_name, train, data_root):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True
            , transform=transforms.Resize(224)
        )
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)

    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root=data_dir, split="train" if train else "test", download=True)
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.labels)

    elif dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST
        }[dataset_name]
        dataset = dataset_class(root=data_dir, train=train, download=True)
        images = dataset.data.unsqueeze(1)
        labels = dataset.targets

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    image_idx = torch.arange(len(images))

    images = images / 255.0

    return images, labels, image_idx


def get_torchvision_datasets(dataset_name, data_root, device, seed, valid_fraction, flatten):
    images, labels, image_idx = get_raw_image_tensors(dataset_name, train=True, data_root=data_root)
    if flatten:
        images = images.flatten(start_dim=1)
    perm = torch.randperm(images.shape[0])
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]
    shuffled_image_idx = image_idx[perm]
    
    valid_size = int(valid_fraction * images.shape[0])
    valid_images = shuffled_images[:valid_size]
    valid_labels = shuffled_labels[:valid_size]
    valid_image_idx = shuffled_image_idx[:valid_size]

    train_images = shuffled_images[valid_size:]
    train_labels = shuffled_labels[valid_size:]
    train_image_idx = shuffled_image_idx[valid_size:]

    train_dset = image_tensors_to_dataset(train_images, train_labels, train_image_idx, device)
    valid_dset = image_tensors_to_dataset(valid_images, valid_labels, valid_image_idx, device)

    test_images, test_labels, test_image_idx = get_raw_image_tensors(dataset_name, train=False, data_root=data_root)
    if flatten:
        test_images = test_images.flatten(start_dim=1)

    test_dset = image_tensors_to_dataset(test_images, test_labels, test_image_idx, device)
    return train_dset, valid_dset, test_dset


def get_image_datasets_by_class(dataset_name, data_root, device, seed, valid_fraction, flatten=False):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "celeba":
        # valid_fraction and flatten ignored
        data_class = CelebA
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    train_dset = data_class(root=data_dir, role="train", seed=seed, device=device)
    valid_dset = data_class(root=data_dir, role="valid", seed=seed, device=device)
    test_dset = data_class(root=data_dir, role="test", seed=seed, device=device)

    return train_dset, valid_dset, test_dset


def get_augmented_datasets(dataset_name, data_root, seed, device, make_valid_loader, flatten,
                           num_augmentations, noise_std):
    # This method currently assumes to have a torchvision dataset. So it will not be able to handle CelebA
    # Therefore, it would need to be integrated in the get_image_datasets logic
    valid_fraction = 0.1 if make_valid_loader else 0
    images, labels, image_idx = get_raw_image_tensors(dataset_name, train=True, data_root=data_root)
    if flatten:
        images = images.flatten(start_dim=1)

    perm = torch.randperm(images.shape[0])
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]
    shuffled_image_idx = image_idx[perm]

    valid_size = int(valid_fraction * images.shape[0])
    valid_images = shuffled_images[:valid_size]
    valid_labels = shuffled_labels[:valid_size]
    valid_image_idx = shuffled_image_idx[:valid_size]

    train_images = shuffled_images[valid_size:]
    train_labels = shuffled_labels[valid_size:]
    train_image_idx = shuffled_image_idx[valid_size:]

    train_dset = image_tensors_to_augmented_dataset(train_images, train_labels, train_image_idx, device, num_augmentations, noise_std)
    valid_dset = image_tensors_to_dataset(valid_images, valid_labels, valid_image_idx, device)

    test_images, test_labels, test_image_idx = get_raw_image_tensors(dataset_name, train=False, data_root=data_root)
    if flatten:
        test_images = test_images.flatten(start_dim=1)
    test_dset = image_tensors_to_dataset(test_images, test_labels, test_image_idx, device)

    return train_dset, valid_dset, test_dset

def get_image_datasets(dataset_name, data_root, seed, device, make_valid_loader=False, flatten=False):
    valid_fraction = 0.1 if make_valid_loader else 0

    torchvision_datasets = ["mnist", "fashion-mnist", "svhn", "cifar10"]
    if dataset_name in torchvision_datasets:
        return get_torchvision_datasets(dataset_name, data_root, device, seed, valid_fraction, flatten)
    else:
        return get_image_datasets_by_class(dataset_name, data_root, device, seed, valid_fraction, flatten)
