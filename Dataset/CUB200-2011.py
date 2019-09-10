# Copyright 2019 jiaxin Zhuang
#
#
#
# ==============================================================================
"""Dataset for CUB200-2011.

    Directory:
        CUB_200_2011
        .
        ├── README
        ├── attributes
        ├── bounding_boxes.txt
        ├── classes.txt
        ├── image_class_labels.txt
        ├── images
        ├── images.txt
        ├── parts
        └── train_test_split.txt
"""


import os
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CUB(Dataset):
    """CUB200-2011.
    """
    def __init__(self, root="../data", train=True, transform=None):
        self.root = os.path.join(root, "CUB_200_2011")
        self.train = train

        self.imgs_path, self.targets = self.read_path()
        self.classes = list(set(self.targets))

        self.targets_imgs_dict = dict()
        targets_np = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(target == targets_np)[0]
            self.targets_imgs_dict.update({target: indexes})

    def __getitem__(self, index):
        img_path, target = self.imgs_path[index], self.targets[index]
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)

    def read_path(slef):
        """Read img, label and split path.
        """
	img_txt_file_path = os.path.join(self.root, 'images.txt')
        img_txt_file = txt_loader(img_txt_file_path, is_int=False)
        img_name_list = img_txt_file

        label_txt_file_path = os.path.join(self.root, "image_class_labels.txt")
        label_txt_file = txt_loader(label_txt_file_path, is_int=True)
        label_list = list(map(lambda x: x-1, label_txt_file))

        train_test_file_path = os.path.join(self.root, "train_test_split.txt")
        train_test_file = txt_loader(train_test_file_path, is_int=True)
        train_test_list = train_test_file

	if self.train:
	    train_img_path = [os.path.join(self.root, "images", x) \
		              for i, x in zip(train_test_list, img_name_list) if i]
            train_targets = [x for i, x in zip(train_test_list, label_list) if i]
            imgs_path = train_img_path
            targets = train_targets
        else:
	    test_img_path = [os.path.join(self.root, "images", x) \
		             for i, x in zip(train_test_list, img_name_list) if not i]
            test_targets = [x for i, x in zip(train_test_list, label_list) if not i]
            imgs_path = test_img_path
            targets = test_targets

        return imgs_path, targets



def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentionally a decoding problem, fall back to PIL.image
        return pil_loader(path)

def pil_loader(path):
    """Image Loader."""
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")


def txt_loader(path, is_int=True):
    """Txt Loader
    Args:
        path:
        is_int: True for labels and split, False for image path
    Returns:
        txt_array: array
    """
    txt_array = []
    with open(path) as afile:
        for line in afile:
            txt = line[:-1].split(" ")[-1]
            if is_int:
                txt = int(txt)
            txt_array.append(txt)
        return txt_array


def print_dataset(dataset, print_time):
    print(len(dataset))
    from collections import Counter
    counter = Counter()
    for index, (img, label) in enumerate(dataset):
        if index % print_time == 0:
            print(img.size(), label)
        counter.update(labels)
    print(counter)


if __name__ == "__main__":
    root = "../data"
    dataset = CUB(root=root, train=True, transform=transform.ToTensor())
    print_dataset(dataset, print_time=100)

    dataset = CUB(root=root, train=False, transform=transform.ToTensor())
    print_dataset(dataset, print_time=100)
