import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
import random

import sys
sys.path.append(os.path.abspath('.'))

from utils.DataAug import AddBackground, RandomErode, RandomDilate, RandomPad, Add_Points, Add_Lines, LocalBlur
import torch


class EvaluateDataset(data.Dataset):
    def __init__(self, ordered_path, background_path, standard_path, good_path=None, medium_path=None, bad_path=None, size=64, is_train=True):
        self.samples = []
        self.is_train = is_train
        self.size = size

        # 得到排好序的临摹字数据和对应的标准字
        if is_train:
            for dir_path in os.listdir(ordered_path):
                # 当同一种字下的标注数据超过2条时，多的部分作为测试集
                if len(os.listdir(os.path.join(ordered_path, dir_path))) == 1 or (len(dir_path.split('_')) == 3 and int(dir_path[-1]) > 2):
                    continue
                self.samples.extend([[os.path.join(ordered_path,
                                                   dir_path, str(i) + '.png') for i in range(6)]])

            random.shuffle(self.samples)

            print('The train nums is', len(self.samples))

        else:
            for dir_path in os.listdir(ordered_path):
                if len(os.listdir(os.path.join(ordered_path, dir_path))) == 1 or len(dir_path.split('_')) == 2 or int(dir_path[-1]) <= 2:
                    continue

                self.samples.extend([[os.path.join(ordered_path,
                                                   dir_path, str(i) + '.png') for i in range(6)]])
            
            print('The test nums is', len(self.samples))
        

        if self.is_train:
            with open('C:\MyAiProject\ChineseCharacterOCR\dataset\chinese_ocr_keys.txt', 'r', encoding='utf-8') as file:
                keys = file.readlines()
                for i in range(len(keys)):
                    keys[i] = keys[i].strip('\n')
            
            self.standard_path = []
            for key in keys:
                if key == "#":
                    continue
                self.standard_path.append(os.path.join(standard_path, key + ".png"))
            
            for pth in os.listdir("C:\MyDataset\chinese_std"):
                self.standard_path.append(os.path.join("C:\MyDataset\chinese_std", pth))

            random.shuffle(self.standard_path)

            self.good_path = []
            for pth in os.listdir(good_path):
                self.good_path.append(os.path.join(good_path, pth))
            random.shuffle(self.good_path)

            self.medium_path = []
            for pth in os.listdir(medium_path):
                self.medium_path.append(os.path.join(medium_path, pth))
            random.shuffle(self.medium_path)

            self.bad_path = []
            for pth in os.listdir(bad_path):
                self.bad_path.append(os.path.join(bad_path, pth))
            random.shuffle(self.bad_path)

            print("std:" + str(len(self.standard_path)), "good:" + str(len(self.good_path)),
                    "medium:" + str(len(self.medium_path)), "bad:" + str(len(self.bad_path)))

        self.bg_ts = transforms.Compose([
            transforms.Resize((size*2, size*2)),
            transforms.RandomResizedCrop(size=size*2, scale=(0.8, 1.)),
            transforms.RandomPerspective(distortion_scale=0.1,
                                        fill=255, interpolation=transforms.InterpolationMode.BICUBIC, p=0.5),
            transforms.ToTensor(),
        ])
        
        self.background = []
        for pth in os.listdir(background_path):
            image = Image.open(os.path.join(background_path, pth)).convert('RGB')

            self.background.append(self.bg_ts(image))
        
        self.standard_ts = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]

        scores = [100, random.randint(80, 90), random.randint(70, 80), random.randint(60, 70), random.randint(50, 60), random.randint(40, 50)]
        # scores = [100, 90, 80, 70, 55, 45]
        imgs = [Image.open(paths[i]).convert("RGB") for i in range(6)]

        # 未经过数据变换和经过变换的
        origins, changes = [], []
        for i in range(6):
            origins.append(self.standard_ts(imgs[i]))

            img = build_transform(background=self.background, size=self.size, is_expand=True, aff_p=0.5)(imgs[i])
            changes.append(img)
        
        if not self.is_train:
            return origins, scores
        
        # 标准字用于维持锚点，分数设为100
        standards = []
        max_len1 = len(self.standard_path)
        for i in range(idx*5, idx*5 + 5):
            img = Image.open(self.standard_path[i % max_len1]).convert("RGB")
            img = build_transform(background=self.background, size=self.size, is_expand=True, aff_p=0.5)(img)
            standards.append(img)

            # standards.append(self.standard_ts(img))
        standards_scores = [100 for _ in range(len(standards))]

        # good用于维持较好的，分数设为80-95
        goods = []
        max_len2 = len(self.good_path)
        for i in range(idx*6, idx*6 + 6):
            img = Image.open(self.good_path[i % max_len2]).convert("RGB")
            img = build_transform(background=self.background, size=self.size, is_expand=False, aff_p=0)(img)
            goods.append(img)

        goods_scores = [random.randint(80, 95) for _ in range(len(goods))]

        # medium用于维持中等的，分数设为60-80
        mediums = []
        max_len3 = len(self.medium_path)
        for i in range(idx*8, idx*8 + 8):
            img = Image.open(self.medium_path[i % max_len3]).convert("RGB")
            img = build_transform(background=self.background, size=self.size, is_expand=True, aff_p=0)(img)
            mediums.append(img)
        
        mediums_scores = [random.randint(60, 80) for _ in range(len(mediums))]

        # bad用于维持最差的，分数设为40-60
        bads = []
        max_len4 = len(self.bad_path)
        for i in range(idx*10, idx*10 + 10):
            img = Image.open(self.bad_path[i % max_len4]).convert("RGB")
            img = build_transform(background=self.background, size=self.size, is_expand=False, aff_p=0)(img)
            bads.append(img)

        bads_scores = [random.randint(40, 60) for _ in range(len(bads))]

        return changes, scores, paths, standards, standards_scores, goods, mediums, bads, goods_scores, mediums_scores, bads_scores
        


def build_transform(add_bg_p=0.4, background=None, size=64, is_expand=True, aff_p=0.5):

    return transforms.Compose([
        RandomPad(offset=40, is_expand=is_expand),
        transforms.Resize((size*2, size*2)),
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomAffine(degrees=10, fill=255),
                # transforms.RandomPerspective(distortion_scale=0.2, fill=255, p=1) # 仿射变换或透射变换
            ]),
        ], p=aff_p),

        transforms.ToTensor(),

        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomChoice([AddBackground(bg) for bg in background]),
                # Add_Lines(),
            ])
        ], p=add_bg_p),

        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.GaussianBlur(kernel_size=3),
                # RandomDilate(ksize=3),
                RandomErode(ksize=3),
                LocalBlur(),
            ])
        ], p=0.5),

        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.2),
        
        transforms.Resize((size, size)),
        # transforms.RandomApply([transforms.ColorJitter(brightness=(0.5, 1.6), contrast=(0.5, 1.8), saturation=(0.5, 1.8), hue=0.2)], p=0.4),
        transforms.Normalize([0.5], [0.5])
    ])


if __name__ == "__main__":

    # 测试数据是否合理
    train_data = EvaluateDataset('/home/disk1/wangjiakai/handwritten_check',
                                '/home/wangjiakai/ChineseCharacterOCR/dataset/copybook_background', is_train=True)
    
    from tqdm import tqdm
    
    tar = "./images"

    if not os.path.exists(tar):
        os.mkdir(tar)

    for i in tqdm(range(500)):
    
        origins, changes, scores, paths = train_data[i]

        # print(paths)

        for x in range(6):
            pth = paths[x].split('/')
            transforms.ToPILImage()(origins[x] * torch.tensor([0.5]) + torch.tensor([0.5])).save(os.path.join(tar, "origins" + "_" + pth[-2] + "_" + pth[-1]))
            transforms.ToPILImage()(changes[x] * torch.tensor([0.5]) + torch.tensor([0.5])).save(os.path.join(tar, "changes" + "_" + pth[-2] + "_" + pth[-1]))
 
        # break
