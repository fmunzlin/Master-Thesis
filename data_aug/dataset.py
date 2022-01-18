import os
from os import path as osp
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

import torch
from torch.utils.data import Dataset

from data_aug.augment import Augmentation

def get_val_list(size, image_list, label_list):
    num_labels = len(set(label_list))
    same_label = [np.where(np.isin(label_list, label))[0] for label in range(num_labels)]
    val_image_list = []
    val_label_list = []
    idx_list = []

    for i in range(size):
        next_idx = rnd_from_list(same_label[i % num_labels])
        idx_list.append(next_idx)
        val_image_list.append(image_list[next_idx])
        val_label_list.append(label_list[next_idx])

    image_list = [image for i, image in enumerate(image_list) if i not in idx_list]
    label_list = [label for i, label in enumerate(label_list) if i not in idx_list]
    return image_list, label_list, val_image_list, val_label_list

def rnd_from_list(items):
    return items[torch.randint(high=len(items), low=0, size=(1,))[0]]

class Eval_content(Dataset):
    def __init__(self, args, folder):
        self.args = args
        self.data_path = osp.join("data", folder)
        self.image_list = self.get_images()
        self.augment = Augmentation(self.args).evaluate

    def get_images(self):
        image_list = []
        for content_image in os.listdir(self.data_path):
            image_list.append(osp.join(self.data_path, content_image))
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        if self.args.d_id != -1: image = image.cuda()
        return image

class Eval_style(Dataset):
    def __init__(self, args, folder):
        self.args = args
        self.data_path = osp.join("data", folder)
        self.image_list = self.get_images()
        self.augment = Augmentation(self.args).evaluate

    def get_images(self):
        image_list = []
        for artist in os.listdir(self.data_path):
            temp = []
            for style_image in os.listdir(osp.join(self.data_path, artist)):
                temp.append(osp.join(self.data_path, artist, style_image))
            image_list.append(temp)
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_1, image_2 = self.image_list[idx]
        _, image_3 = self.image_list[(idx + 1) % self.__len__()]

        image_1 = self.augment(Image.open(image_1))
        image_2 = self.augment(Image.open(image_2))
        image_3 = self.augment(Image.open(image_3))

        if self.args.d_id != -1:
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            image_3 = image_3.cuda()

        return image_1, image_2, image_3

class Validation_content(Dataset):
    def __init__(self, args, image_list,):
        self.args = args
        self.image_list = image_list
        self.augment = Augmentation(self.args).validate

    def __len__(self):
        return int(len(self.image_list) / self.args.batch_size) * self.args.batch_size

    def get_image(self, idx):
        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_1 = self.get_image(idx)

        if self.args.d_id != -1:
            image_1 = image_1.cuda()

        return image_1

class Validation_style(Dataset):
    def __init__(self, args, image_list, label_list):
        self.args = args
        self.image_list = image_list
        self.label_list = label_list
        self.distinct_labels = list(set(self.label_list))
        self.same_labels = dict(zip(self.distinct_labels,
                                    [np.where(np.isin(self.label_list, label))[0] for label in self.distinct_labels]))

        self.augment = Augmentation(self.args).validate


    def __len__(self):
        return int(len(self.image_list) / self.args.batch_size) * self.args.batch_size

    def get_image(self, idx):
        label = self.label_list[idx]
        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        return image, label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_1, label_1 = self.get_image(idx)

        if self.args.dif_style:
            raise("not implemented yet")
        else:
            label_indices = self.same_labels[label_1]
            label_idx = (np.where(label_indices==idx)[0][0] + 1) % len(label_indices)
        new_idx = label_indices[label_idx]
        image_2, label_2 = self.get_image(new_idx)

        if self.args.d_id != -1:
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            label_1 = label_1.cuda()
            label_2 = label_2.cuda()

        return image_1, label_1, image_2, label_2

class Style_dataset(Dataset):
    def __init__(self, args, folder):
        self.args = args
        self.folder = osp.join("data", folder)
        self.augment = Augmentation(self.args).train

        self.image_list, self.label_list = self.get_images()
        self.image_list, self.label_list, self.val_image_list, self.val_label_list = \
            get_val_list(self.args.val_size, self.image_list, self.label_list)

        self.num_images = len(self.label_list)
        self.num_labels = len(set(self.label_list))
        self.same_labels = [np.where(np.isin(self.label_list, label))[0] for label in range(self.num_labels)]
        self.other_labels = [np.where(~np.isin(self.label_list, label))[0] for label in range(self.num_labels)]


    def get_images(self):
        image_list = []
        label_list = []

        if self.args.artist_id == -1: artists = os.listdir(self.folder)
        else: artists = [os.listdir(self.folder)[self.args.artist_id]]
        artist2id = dict(zip(artists, torch.tensor(range(len(artists)))))
        for artist in artists:
            artist_folder = osp.join(self.folder, artist)
            for image in os.listdir(artist_folder):
                image_list.append(osp.join(artist_folder, image))
                label_list.append(artist2id[artist])
        return image_list, label_list

    def get_batch(self):
        labels_1 = []
        labels_2 = []
        images_1 = torch.Tensor()
        images_2 = torch.Tensor()

        for i in range(self.args.batch_size):
            idx = torch.randint(high=self.num_images, low=0, size=(1,))[0]
            image_1, label_1 = self.get_image(idx)
            image_1 = image_1.unsqueeze(0)

            if self.args.dif_style: rnd_idx = rnd_from_list(self.other_labels[label_1])
            else: rnd_idx = rnd_from_list(self.same_labels[label_1])
            image_2, label_2 = self.get_image(rnd_idx)
            image_2 = image_2.unsqueeze(0)

            labels_1.append(label_1)
            labels_2.append(label_2)
            images_1 = torch.cat((images_1, image_1), dim=0)
            images_2 = torch.cat((images_2, image_2), dim=0)

        labels_1 = torch.tensor(labels_1)
        labels_2 = torch.tensor(labels_2)

        if self.args.d_id != -1:
            images_1 = images_1.cuda()
            images_2 = images_2.cuda()
            labels_1 = labels_1.cuda()
            labels_2 = labels_2.cuda()

        return images_1, labels_1, images_2, labels_2

    def __len__(self):
        return int(self.num_images / self.args.batch_size) * self.args.batch_size

    def get_image(self, idx):
        label = self.label_list[idx]
        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        return image, label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_1, label_1 = self.get_image(idx)

        rnd_idx = rnd_from_list(self.same_labels[label_1])
        image_2, label_2 = self.get_image(rnd_idx)

        if self.args.d_id != -1:
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            label_1 = label_1.cuda()
            label_2 = label_2.cuda()

        return image_1, label_1, image_2, label_2

    def get_validation_dataset(self):
        return Validation_style(self.args, self.val_image_list, self.val_label_list)


class Content_dataset(Dataset):
    def __init__(self, args, folder):
        self.args = args
        self.folder = osp.join("data", folder, "data_large")
        self.augment = Augmentation(self.args).train

        self.image_list = self.get_images()
        self.image_list = shuffle(self.image_list)
        self.val_image_list = self.image_list[:self.args.val_size]
        self.image_list = self.image_list[self.args.val_size:]

    def get_sub_folders(self, content_dirs, dir):
        for file in os.listdir(dir):
            file = osp.join(dir, file)
            if os.path.isfile(file) and dir not in content_dirs:
                content_dirs.append(dir)
                break
            else:
                content_dirs = self.get_sub_folders(content_dirs, file)
        return content_dirs

    def get_images(self):
        image_list = []
        for folder in self.get_sub_folders([], self.folder):
            for image in os.listdir(folder):
                image_list.append(osp.join(folder, image))
        return image_list

    def __len__(self):
        return int(len(self.image_list) / self.args.batch_size) * self.args.batch_size

    def get_image(self, idx):
        image = self.image_list[idx]
        image = Image.open(image)
        image = self.augment(image)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.get_image(idx)
        if self.args.d_id != -1: image = image.cuda()
        return image

    def get_batch(self):
        images = torch.Tensor()
        for idx in torch.randint(high=len(self.image_list), low=0, size=(self.args.batch_size,)):
            images = torch.cat((images, self.get_image(idx).unsqueeze(0)), dim=0)

        if self.args.d_id != -1: images = images.cuda()
        return images

    def get_validation_dataset(self):
        return Validation_content(self.args, self.val_image_list)