import os
import torch
from PIL import Image
from utils.DataAug import RandomDilate, AddBackground, RandomErode, RandomPad, GaussianBlur, LocalBlur, Add_Points
from torchvision import transforms
import random
from tqdm import tqdm
import time
import shutil
import cv2
import numpy as np

torch.set_printoptions(profile="full")

size = 128

background_transforms = transforms.Compose([
    transforms.Resize((size, size)),
    # transforms.RandomApply([GaussianBlur()], p=0.4),
    transforms.ToTensor(),
    # transforms.RandomApply([RandomDilate(ksize=1)], p=0.6),
])

img_transforms = transforms.Compose([
    RandomPad(offset=30),
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    # transforms.RandomApply([transforms.RandomChoice([RandomDilate(ksize=1), RandomErode(ksize=1), RandomDilate(ksize=3), RandomErode(ksize=3)])], p=0.2),
])


finall_transforms = transforms.Compose([
    transforms.RandomApply([Add_Points()], p=0.4),
    transforms.RandomApply([transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 2), saturation=(0.5, 1), hue=0.2)], p=0.4),
    transforms.RandomChoice([
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), transforms.RandomAdjustSharpness(sharpness_factor=1, p=0.4)
    ]),
])


if __name__ == "__main__":
    old_data = "/home/wangjiakai/handwritten/images"
    new_data = "/home/wangjiakai/handwritten_Segdataset"
    if not os.path.exists(new_data):
        os.mkdir(new_data)

    if os.path.exists(os.path.join(new_data, "images")):
        shutil.rmtree(os.path.join(new_data, "images"))
        shutil.rmtree(os.path.join(new_data, "labels"))
    os.mkdir(os.path.join(new_data, "images"))
    os.mkdir(os.path.join(new_data, "labels"))

    backgrounds = []
    for pth in os.listdir('/home/wangjiakai/copybook_background'):
        backgrounds.append(
            background_transforms(Image.open(os.path.join('/home/wangjiakai/copybook_background', pth)).convert('RGB'))
        )

    random.seed(0)
    for pth in tqdm(os.listdir(old_data)):
        
        background = random.sample(backgrounds, k=25)
        img = Image.open(os.path.join(old_data, pth)).convert('RGB')
        pth = pth.split('.')[0]

        for i, bg in enumerate(background):
            img1 = img_transforms(img)

            label = torch.where(img1 >= 0.75, torch.tensor([0.0]), torch.tensor([1.0]))

            img2 = transforms.RandomApply([AddBackground(bg, label)], p=0.8)(img1)
            img2 = finall_transforms(img2)

            transforms.Grayscale()((transforms.ToPILImage()(label / 255))).save(os.path.join(new_data, "labels", pth + "_" + str(i + 1) + ".png"))
            transforms.ToPILImage()(img2).save(os.path.join(new_data, "images", pth + "_" + str(i + 1) + ".jpg"))

            # if i >= 10:
            #     transforms.Grayscale()(transforms.ToPILImage()(label / 255)).save(os.path.join(new_data, "labels", pth + "_" + str(i + 1 + 2) + ".png"))
            #     Image.blend(transforms.ToPILImage()(img1), transforms.ToPILImage()(bg), 0.5).save(os.path.join(new_data, "images", pth + "_" + str(i + 1 + 2) + ".jpg"))

            
        # break
