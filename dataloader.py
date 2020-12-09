import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from data.get_dataset import get_dataset, NUM_CLASSES


def get_transform(
    img_size, random_crop=False, random_horizontal_flip=False, normalize_mean=(0.5,), normalize_std=(0.5,)
):
    transform_list = [transforms.Resize((img_size, img_size))]
    if random_crop:
        transform_list.append(transforms.RandomCrop(img_size, padding=(4 if img_size == 32 else 8)))
    if random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
    return transforms.Compose(transform_list)


def get_src_dataloader(name, split, total_split, img_size, batch_size, num_workers=1):
    if name not in ["tiny_imagenet", "mini_imagenet", "cifar_100", "meta_imagenet"]:
        raise NotImplementedError

    # Get split information
    split_info = np.load("data/{}/{}_split/split_{}.npz".format(name, total_split, split))
    label_list = list(split_info["label_list"])
    idx_train = list(split_info["idx_train"])
    idx_valid = list(split_info["idx_valid"])
    # idx_test = list(split_info["idx_test"])

    if name == "meta_imagenet":
        target_transform = None
    else:

        def target_transform(y):
            return label_list.index(y)

    transform_train = get_transform(img_size, random_crop=True, random_horizontal_flip=True)
    # transform_test = get_transform(img_size)

    train_ds = get_dataset(name, train=True, transform=transform_train, target_transform=target_transform)
    # test_ds = get_dataset(name, train=False, transform=transform_test, target_transform=target_transform)

    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": False, "drop_last": True}
    train_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(idx_train), **kwargs)
    valid_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(idx_valid), **kwargs)
    test_loader = None
    # test_loader = DataLoader(test_ds, sampler=SubsetRandomSampler(idx_test), **kwargs)

    return train_loader, valid_loader, test_loader, len(label_list)


def get_tgt_dataloader(name, img_size, batch_size, num_workers=3):
    num_instances = None
    if "small_svhn" in name:
        if name == "small_svhn":
            num_instances = 500
        elif name == "small_svhn_100":
            num_instances = 100
        elif name == "samll_svhn_2500":
            num_instances = 2500
        else:
            raise NotImplementedError
        name = "svhn"
    elif "cifar_100" in name:
        if name == "small_cifar_100":
            num_instances = 50
        else:
            raise NotImplementedError
        name = "cifar_100"
    elif "fashion_mnist" in name:
        if name == "small_fashion_mnist":
            num_instances = 500
        else:
            raise NotImplementedError
        name = "fashion_mnist"

    transform_train = get_transform(img_size, random_crop=True, random_horizontal_flip=True)
    transform_test = get_transform(img_size)

    train_ds = get_dataset(name, train=True, transform=transform_train)
    test_ds = get_dataset(name, train=False, transform=transform_test)

    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True, "drop_last": True}
    if num_instances:
        train_idx = []
        for c in range(NUM_CLASSES[name]):
            try:
                train_idx.extend(list(np.argwhere(train_ds.labels == c)[:num_instances, 0]))
            except AttributeError:
                train_idx.extend(list(np.argwhere(np.array(train_ds.targets) == c)[:50, 0]))
        train_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(train_idx), **kwargs)
        test_loader = DataLoader(test_ds, **kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
        test_loader = DataLoader(test_ds, **kwargs)
    return train_loader, test_loader, NUM_CLASSES[name]
