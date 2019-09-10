# Copyright 2019 jiaxin Zhuang
#
#
#
# ==============================================================================
"""Dataset for SD198.

    Directory:
        SD198
        ├── 5_5_split
        ├── 8_2_split
        ├── README.pdf
        ├── README.txt
        ├── class_idx.npy
        └── images
"""


import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SD198(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.join(root, "SD198")
        self.transform = transform
        self.train = train
        self.data_dir = os.path.join(self.root, 'images')
        iter_no = 0
        self.imgs_path, self.targets = self.get_data(iter_no, self.root)
        self.dataset_name = 'SD198'
        class_idx_path = os.path.join(self.root, 'class_idx.npy')
        classes_name = self.get_classes_name(class_idx_path)

        self.loader = pil_loader

        self.classes = list(range(len(classes_name)))
        self.target_img_dict = dict()
        targets = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(targets == target)[0]
            self.target_img_dict.update({target: indexes})

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.imgs_path[index]
        target = self.targets[index]
        img = pil_loader(path)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)

    def get_data(self, iter_no, data_dir):
        if self.train:
            txt = '8_2_split/train_{}.txt'.format(iter_no)
        else:
            txt = '8_2_split/val_{}.txt'.format(iter_no)

        fn = os.path.join(data_dir, txt)
        txtfile = pd.read_csv(fn, sep=" ")
        raw_data = txtfile.values

        data = []
        targets = []
        for (path, label) in raw_data:
            data.append(os.path.join(self.data_dir, path))
            targets.append(label)

        return data, targets

    def get_classes_name(self, data_dir):
        classes_name = np.load(data_dir)
        return classes_name


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def pil_loader(path):
    """Image Loader
    """
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")


def print_dataset(dataset, print_time):
    print(len(dataset))
    from collections import Counter
    counter = Counter()
    labels = []
    for index, (img, label) in enumerate(dataset):
        if index % print_time == 0:
            print(img.size(), label)
        labels.append(label)
    counter.update(labels)
    print(counter)


if __name__ == "__main__":
    root = "./data"
    dataset = SD198(root=root, train=True, transform=transforms.ToTensor())
    print_dataset(dataset, print_time=1000)

    dataset = SD198(root=root, train=False, transform=transforms.ToTensor())
    print_dataset(dataset, print_time=1000)
