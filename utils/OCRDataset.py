import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
import random
import torch

import sys
sys.path.append(os.path.abspath('.'))
from utils.DataAug import GaussianBlur, AddBackground, RandomDilate, RandomErode, RandomPad, Add_Points, Add_Lines, AddGaussianNoise, AddSaltPepperNoise, LocalBlur
import cv2
import numpy as np
from torchvision.transforms import functional as F


class HandWrittenDataset(data.Dataset):
    def __init__(self, handwritten_path, background_path, size=64, is_train=True):

        # 加载手写字数据集
        self.samples = []

        if is_train:
            with open(os.path.join(handwritten_path, 'label.txt'), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    pth, label = line.strip('\n').split(' ')
                    self.samples.append([os.path.join(handwritten_path, pth), label])
        
        else:
            with open(os.path.join(handwritten_path, 'label_test.txt'), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    pth, label = line.strip('\n').split('png ')
                    pth += "png"
                    self.samples.append([os.path.join(handwritten_path, pth), label])
        
        self.size, self.is_train = size, is_train
        random.shuffle(self.samples)

        self.ts = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.8, 1.)),
            transforms.RandomPerspective(distortion_scale=0.1,
                                        fill=255, interpolation=transforms.InterpolationMode.BICUBIC, p=0.5),
            transforms.ToTensor(),
        ])

        self.std_ts = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.background = []
        for pth in os.listdir(background_path):
            image = Image.open(os.path.join(background_path, pth)).convert('RGB')

            self.background.append(self.ts(image))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if not self.is_train:
            # print(path)
            # img = cv2.imread(path)
            # img = cv2.resize(img, (self.size, self.size)).astype(np.float32).transpose((2, 0, 1))
            # img = (img - 127.5) / 127.5
            # return torch.tensor(img), int(label)
        
            image = Image.open(path).convert('RGB')
            return self.std_ts(image), int(label)
            

        image = Image.open(path).convert('RGB')
        img = build_transform(add_background=False, size=self.size)(image)

        # add background
        img = build_transform(add_background=True, background=self.background, size=self.size)(img)

        return img, int(label)
        return img, int(label), path

        
def build_transform(add_background=True, background=None, size=64):
    if not add_background:
        return transforms.Compose([
            RandomPad(offset=20),
            transforms.Resize((size, size)),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomAffine(degrees=20, fill=255),
                    transforms.RandomPerspective(distortion_scale=0.4, fill=255, p=1) # 仿射变换或透射变换
                ]),
            ], p=0.5),

            transforms.ToTensor(),
        ])
    else:

        return transforms.Compose([

            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomChoice([AddBackground(bg) for bg in background]),
                    Add_Lines(),
                ])
            ], p=0.8),

            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.GaussianBlur(kernel_size=3),
                    # RandomDilate(ksize=3),
                    RandomErode(ksize=3),
                    LocalBlur(),
                ])
            ], p=0.5),

            transforms.RandomApply([
                transforms.RandomChoice([
                    AddSaltPepperNoise(density=0.02),
                    AddGaussianNoise(amplitude=10),
                    Add_Points(),
                ])
            ], p=0.2),
            transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.2),
            
            transforms.RandomApply([transforms.ColorJitter(brightness=(0.5, 1.6), contrast=(0.5, 1.8), saturation=(0.5, 1.8), hue=0.2)], p=0.4),
            transforms.Normalize([0.5], [0.5])
        ])


if __name__ == "__main__":
    train_data = HandWrittenDataset('/home/wangjiakai/ChineseCharacterOCR/dataset',
                                    '/home/wangjiakai/copybook_background', is_train=True)
    
    from tqdm import tqdm
    
    tar = "./images"

    if not os.path.exists(tar):
        os.mkdir(tar)

    for i in tqdm(range(5000)):
    
        img, x, pth = train_data[i]
        pth = pth.split('/')[-2:]
        # print(img.size())
        transforms.ToPILImage()(img * torch.tensor([0.5]) + torch.tensor([0.5])).save(os.path.join(tar, pth[0] + "_" + pth[1]))
