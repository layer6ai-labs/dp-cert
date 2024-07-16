import numpy as np
from torch.utils.data import DataLoader

from .image import get_image_datasets, get_augmented_datasets


def get_loaders_from_config(cfg, device, **kwargs):
    flatten = False
    if cfg["net"] == "mlp":
        flatten = True

    if cfg["method"] in {"dpsgd-augment", "regular-augment", "dpsgd-augment-auto-clip"}:
        train_loader, valid_loader, test_loader = get_loaders(
            dataset=cfg["dataset"],
            device=device,
            data_root=cfg.get("data_root", "data/"),
            train_batch_size=cfg["train_batch_size"],
            valid_batch_size=cfg["valid_batch_size"],
            test_batch_size=cfg["test_batch_size"],
            seed=cfg["seed"],
            make_valid_loader=cfg["make_valid_loader"],
            flatten=flatten,
            augment=True,
            num_augmentations=cfg["num_augmentations"],
            noise_std=cfg["augment_noise_std"],
        )
    else:
        train_loader, valid_loader, test_loader = get_loaders(
            dataset=cfg["dataset"],
            device=device,
            data_root=cfg.get("data_root", "data/"),
            train_batch_size=cfg["train_batch_size"],
            valid_batch_size=cfg["valid_batch_size"],
            test_batch_size=cfg["test_batch_size"],
            seed=cfg["seed"],
            make_valid_loader=cfg["make_valid_loader"],
            flatten=flatten,
        )

    if cfg["dataset"] in ["celeba"]:
        train_dataset_shape = train_loader.dataset.shape
    elif cfg["method"] in {"dpsgd-augment", "regular-augment", "dpsgd-augment-auto-clip"}:
        train_dataset_shape = train_loader.dataset.tensors.shape
    else:
        train_dataset_shape = train_loader.dataset.tensors[0].shape
    cfg["train_dataset_size"] = train_dataset_shape[0]
    cfg["data_shape"] = tuple(train_dataset_shape[1:])
    cfg["data_dim"] = int(np.prod(cfg["data_shape"]))

    if not cfg["make_valid_loader"]:
        valid_loader = test_loader
        print("WARNING: Using test loader for validation")

    return train_loader, valid_loader, test_loader


def get_loaders(
        dataset,
        device,
        data_root,
        train_batch_size,
        valid_batch_size,
        test_batch_size,
        seed,
        make_valid_loader,
        flatten,
        augment=False,
        num_augmentations=8,
        noise_std=1.0,
):
    # NOTE: only training and validation sets
    if dataset in ["mnist", "fashion-mnist", "cifar10", "svhn", "celeba"]:
        if augment:
            if dataset == "celeba":
                raise Exception("Augmentation functionality not yet implemented for celeba")
            train_dset, valid_dset, test_dset = get_augmented_datasets(dataset, data_root, seed, device, 
                                                        make_valid_loader, flatten, num_augmentations=num_augmentations, noise_std=noise_std)
        else:
            train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root, seed, device, make_valid_loader, flatten)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    train_loader = get_loader(train_dset, train_batch_size, drop_last=False)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, valid_batch_size, drop_last=False)
    else:
        valid_loader = None

    test_loader = get_loader(test_dset, test_batch_size, drop_last=False)

    return train_loader, valid_loader, test_loader


def get_loader(dset, batch_size, drop_last):
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False
    )
