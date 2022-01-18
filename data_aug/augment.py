import numpy as np

import torch
from torchvision import transforms

class Denormalize(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean / std

        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class Augmentation():
    def __init__(self, args):
        self.args = args
        self.train_transform = self.get_transforms("train")
        self.validate_transform = self.get_transforms("validate")
        self.evaluate_transform = self.get_transforms("evaluate")

    def get_transforms(self, mode):
        transform = []
        if mode == "train":
            transform.append(transforms.RandomCrop(size=(self.args.crop_size)))
        if mode == "validate":
            transform.append(transforms.CenterCrop(size=(self.args.crop_size)))
        transform.append(transforms.ToTensor())
        if mode == "train":
            transform.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
            transform.append(transforms.RandomVerticalFlip(p=0.5))
            transform.append(transforms.RandomHorizontalFlip(p=0.5))
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)
        return transform

    def resize_to_edge(self, resize_to, image):
        width, height = image.size
        if height <= width:
            factor = resize_to / height
        else:
            factor = resize_to / width
        return self.resize_to_factor(image, factor)

    def resize_to_factor(self, img, factor):
        if isinstance(factor, list):
            return img.resize((int(img.width * factor[0]), int(img.height * factor[0])))
        else:
            return img.resize((int(img.width * factor), int(img.height * factor)))

    def scale_image(self, img, range):
        scale_x = 1. + np.random.uniform(low=0, high=range)
        scale_y = 1. + np.random.uniform(low=0, high=range)
        return self.resize_to_factor(img, [scale_x, scale_y])

    def train(self, image):
        image = self.resize_to_edge(self.args.resize, image)
        image = self.scale_image(image, self.args.scale_aug)
        if image.mode != 'RGB': image = image.convert('RGB')
        image = self.train_transform(image)
        return image

    def validate(self, image):
        image = self.resize_to_edge(self.args.resize, image)
        if image.mode != 'RGB': image = image.convert('RGB')
        image = self.validate_transform(image)
        return image

    def evaluate(self, image):
        if image.mode != 'RGB': image = image.convert('RGB')
        image = self.evaluate_transform(image)
        return image