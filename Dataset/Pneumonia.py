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

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import SimpleITK as sitk


class PneumoniaDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.join(root, "pneumonia_data")
        self.transform = transform
        self.train = is_train

        self.image_data_dir = os.path.join(self.root, 'stage_2_train_images')
        iter_fold = 1
        self.imgs_path, self.targets = \
                self.get_data(iter_fold, os.path.join(self.root, 'split_data'))


        self.loader = dcm_loader
        classes_name = ['Normal', 'Lung Opacity', '‘No Lung Opacity/Not Normal']
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
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)

    def get_data(self, iterNo, data_dir):

        if self.train:
            csv = 'pneumonia_split_{}_train.csv'.format(iterNo)
        else:
            csv = 'pneumonia_split_{}_test.csv'.format(iterNo)

        fn = os.path.join(data_dir, csv)
        csvfile = pd.read_csv(fn, index_col=0)
        raw_data = csvfile.values

        data = []
        targets = []
        for (path, label) in raw_data:
            data.append(os.path.join(self.image_data_dir, path))
            targets.append(label)

        return data, targets


def dcm_loader(path):
    ds = sitk.ReadImage(path)
    img_array = sitk.GetArrayFromImage(ds)
    img_bitmap = Image.fromarray(img_array[0]).convert('RGB')
    return img_bitmap

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
    dataset = PneumoniaDataset(root=root, train=True, transform=transform.ToTensor())
    print_dataset(dataset, print_time=100)

    dataset = PneumoniaDataset(root=root, train=False, transform=transform.ToTensor())
    print_dataset(dataset, print_time=100)
