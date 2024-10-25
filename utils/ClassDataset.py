import os.path
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import numpy as np
import cv2
from utils.DataAug import AddBackground


def get_padding(image, offset=20):
    w, h = image.size
    max_wh = np.max([w, h])
    w_padding = (max_wh - w) / 2
    h_padding = (max_wh - h) / 2

    left_pad = w_padding if w_padding % 1 == 0 else w_padding + 0.5
    right_pad = w_padding if w_padding % 1 == 0 else w_padding - 0.5
    top_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    bottom_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5

    padding = [int(left_pad), int(top_pad), int(right_pad), int(bottom_pad)]
    padding = [pad + offset for pad in padding]
    return list(padding)


FontPad = transforms.Lambda(lambda img: F.pad(img, get_padding(img, 0), 255))


class ClassDataset(data.Dataset):
    def __init__(self, handwritten_path, background_path, size=64, is_train=True):
        
        # 加载手写字数据集
        self.samples = []
        with open(os.path.join(handwritten_path, 'label.txt'), 'r') as file:
            lines = file.readlines()
            for line in lines:
                pth, label = line.strip('\n').split(' ')
                self.samples.append([os.path.join(handwritten_path, pth), label])
        
        self.size, self.is_train = size, is_train

        random.shuffle(self.samples)

        if self.is_train:
            self.samples = self.samples[:int(len(self.samples) * 0.85)]
        else:
            self.samples = self.samples[int(len(self.samples) * 0.85):]
        
        self.bg_ts = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop(size=size, scale=(0.8, 1.)),
            transforms.RandomPerspective(distortion_scale=0.1,
                                         fill=255, interpolation=transforms.InterpolationMode.BICUBIC, p=0.5),
            transforms.ToTensor(),
        ])
        
        self.background = []
        for pth in os.listdir(background_path):
            image = Image.open(os.path.join(background_path, pth)).convert('RGB')
            self.background.append(self.bg_ts(image))
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        # path = os.path.join(self.data_path, path)
        img = Image.open(path).convert('RGB')

        positive_1 = build_transform(action='aug1')(img)
        positive_2 = build_transform(action='aug2', backgrounds=self.background)(img)
        return positive_1, positive_2



def build_transform(action: str = 'aug1', backgrounds=None):
    
    if action == 'aug1':
        train_augmentation1 = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.6, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return train_augmentation1

    else:
        train_augmentation2 = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.6, 1.)),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.ToTensor(),

            # add background
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomChoice([AddBackground(bg) for bg in backgrounds]),
                ])
            ], p=0.5),

            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        return train_augmentation2


class Add_Lines(object):
    def __init__(self):
        self.line_color = ['black', 'gray', 'darkgray', 'grey', 'dimgray']

    def __call__(self, img):
        width, height = img.size
        drawer = ImageDraw.Draw(img)
        for i in range(random.randint(0, 3)):
            x1 = random.randrange(width + 20)
            x2 = random.randrange(width + 20)
            y1 = random.randrange(height + 20)
            y2 = random.randrange(height + 20)
            drawer.line(((x1, y1), (x2, y2)), fill=random.choice(self.line_color), width=random.randint(0, 3))
        return img


class Add_Points(object):
    def __init__(self):
        self.line_color = ['black', 'gray', 'darkgray', 'grey', 'dimgray']

    def __call__(self, img):
        width, height = img.size
        drawer = ImageDraw.Draw(img)
        for i in range(random.randint(1, 3)):
            x1 = random.randrange(width)
            y1 = random.randrange(height)
            radius = random.randint(2, 4)
            drawer.ellipse(((x1, y1), (x1 + radius, y1 + radius)), fill=random.choice(self.line_color), width=4)
        return img


class Erosion(object):
    def __init__(self, kernel_size=1):
        self.conv_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, img):
        # Image转cv2
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_erosion = cv2.erode(img, self.conv_kernel)
        # cv2转Image
        return Image.fromarray(cv2.cvtColor(img_erosion, cv2.COLOR_BGR2RGB))


class RPT(object):
    # 随机透射变换
    def __init__(self, thresh=50):
        self.thresh = thresh

    def RandPointOfTwoPoint(self, p1, p2):
        v = (p2[0] - p1[0], p2[1] - p1[1])
        rr = random.random()
        v = (v[0] * rr, v[1] * rr)
        v = (int(v[0] + 0.5), int(v[1] + 0.5))
        return p1[0] + v[0], p1[1] + v[1]

    def __call__(self, img):
        width, height = 64, 64

        # Image转cv2
        img_np = np.array(img)

        d = (width - self.thresh) // 2
        A, a = (0, 0), (d, d)
        B, b = (0, width), (d, width - d)
        C, c = (width, height), (width - d, width - d)
        D, d = (width, 0), (width - d, d)

        pts1 = np.array([self.RandPointOfTwoPoint(A, a), self.RandPointOfTwoPoint(B, b),
                         self.RandPointOfTwoPoint(C, c), self.RandPointOfTwoPoint(D, d)]).astype('float32')
        pts2 = np.array([(0, 0), (0, height), (width, height), (width, 0)]).astype('float32')

        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img_np, M, (width, height))

        # cv2转Image
        return Image.fromarray(warped)


class Dilation(object):
    def __init__(self, kernel_size=1):
        self.conv_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, img):
        # Image转cv2
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_erosion = cv2.dilate(img, self.conv_kernel)
        # cv2转Image
        return Image.fromarray(cv2.cvtColor(img_erosion, cv2.COLOR_BGR2RGB))


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Binary_transform(object):
    def __call__(self, x):
        x = x.convert('1').convert('RGB')
        return x


class Random_pad(object):
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = 5
        self.sigma = sigma

    def __call__(self, image):
        pad_left, pad_right = random.randint(0, self.sigma), random.randint(0, self.sigma)
        return F.pad(image, [pad_left, pad_left, pad_right, pad_right], 255)


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""
    def __call__(self, x):
        return ImageOps.solarize(x)
