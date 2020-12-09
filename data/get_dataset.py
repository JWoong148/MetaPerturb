import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, FashionMNIST, STL10, SVHN
from meta_imagenet.lib.datasets.DownsampledImageNet import ImageNet32


NUM_CLASSES = {
    "aircraft": 100,
    "cifar_100": 100,
    "cub": 200,
    "fashion_mnist": 10,
    "stanford_cars": 196,
    "stanford_dogs": 120,
    "stl10": 10,
    "svhn": 10,
    "tiny_imagenet": 200,
    "mini_imagenet": 200,
    "dtd": 47,
}


def get_dataset(data: str, train: bool, transform=None, target_transform=None) -> Dataset:
    if data == "aircraft":
        return Aircraft(train, transform, target_transform)
    elif data == "cifar_100":
        return CIFAR100("data/cifar_100", train, transform, target_transform, download=True)
    elif data == "cub":
        return CUB(train, transform, target_transform)
    elif data == "fashion_mnist":
        return FashionMNIST("data/fashion_mnist", train, transform, target_transform, download=True)
    elif data == "stanford_cars":
        return StanfordCars(train, transform, target_transform)
    elif data == "stanford_dogs":
        return StanfordDogs(train, transform, target_transform)
    elif data == "stl10":
        return STL10("data/stl10", "train" if train else "test", None, transform, target_transform, download=True)
    elif data == "svhn":
        return SVHN("data/svhn", "train" if train else "test", transform, target_transform, download=True)
    elif data == "tiny_imagenet":
        return TinyImageNet(train, transform, target_transform)
    elif data == "meta_imagenet":
        return ImageNet32("/w14/dataset/MetaGen/batch32", train, transform, target_transform)
    else:
        raise NotImplementedError()


class NumpyDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(image_path)
        self.labels = np.load(label_path)
        self.length = len(self.images)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.length


class TinyImageNet(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="data/tiny_imagenet/{}_images.npy".format("train" if train else "valid"),
            label_path="data/tiny_imagenet/{}_labels.npy".format("train" if train else "valid"),
            transform=transform,
            target_transform=target_transform,
        )


class MiniImageNet(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="data/mini_imagenet/{}_images.npy".format("train" if train else "valid"),
            label_path="data/mini_imagenet/{}_labels.npy".format("train" if train else "valid"),
            transform=transform,
            target_transform=target_transform,
        )


class CUB(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="data/CUB_200_2011/84_npy/{}_images.npy".format("train" if train else "test"),
            label_path="data/CUB_200_2011/84_npy/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class Aircraft(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="data/aircraft/{}_images.npy".format("train" if train else "test"),
            label_path="data/aircraft/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class StanfordCars(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="data/stanford_cars/{}_images.npy".format("train" if train else "test"),
            label_path="data/stanford_cars/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class StanfordDogs(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="data/stanford_dogs/{}_images.npy".format("train" if train else "test"),
            label_path="data/stanford_dogs/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class DTD(NumpyDataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__(
            image_path="data/dtd/{}_images.npy".format("train" if train else "test"),
            label_path="data/dtd/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )
