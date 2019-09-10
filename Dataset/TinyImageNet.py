# Copyright 2019 jiaxin Zhuang
#
#
#
# ==============================================================================
"""Dataset for Tiny-ImageNet.

    Directory:
        tiny-imagenet-200
        ├── make.sh
        ├── test
        ├── train
        ├── val
        ├── val_annotations.txt
        ├── wnids.txt
        └── words.txt
"""


import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class TinyImageNet:
    def __init__(self, root="./data", mode="train", transform=None):
        self.root = os.path.join(root, "tiny-imagenet-200", mode)
        self.mode = mode
        self.transform = transform

        self.loader = default_loader
        classes, class_to_idx = self.find_classes()
        self.images_path, self.targets = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS)
        self.classes = list(range(len(classes)))
        self.target_img_dict = dict()
        targets = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(targets == target)[0]
            self.target_img_dict.update({target: indexes})

    def __getitem__(self, index):
        img_path, target = self.images_path[index], self.targets[index]
        img = self.loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)

    def find_classes(self):
        dir = self.root
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx, extensions):
        images_path = []
        targets = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        images_path.append(path)
                        targets.append(class_to_idx[target])
        return images_path, targets

    def has_file_allowed_extension(self, filename, extensions):
        """Checks if a file is an allowed extension.
        Args:
            filename (string): path to a file
        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)


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

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


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
    dataset = TinyImageNet(root=root, mode="train", transform=transforms.ToTensor())
    print_dataset(dataset, print_time=1000)

    dataset = TinyImageNet(root=root, mode="val", transform=transforms.ToTensor())
    print_dataset(dataset, print_time=1000)
