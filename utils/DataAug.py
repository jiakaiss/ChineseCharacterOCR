from PIL import ImageFilter, ImageDraw, ImageFilter, Image
import random
import torch.nn as nn
import torch
import time
from torch import tensor
from torchvision.transforms import functional as F
import numpy as np


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


class RandomPad(object):
    " 随机在图像周围填充像素, 默认填充255为白色 "
    def __init__(self, fill=255, offset=30, is_expand=True):
        self.fill = fill
        self.offset= offset
        self.is_expand = is_expand

    def __call__(self, image):
        # 先填充成正方形，不发生形变；再随机填充空白像素
        w, h = image.size
        max_wh = max(w, h)
        w_padding = (max_wh - w) / 2
        h_padding = (max_wh - h) / 2

        left_pad = w_padding if w_padding % 1 == 0 else w_padding + 0.5
        right_pad = w_padding if w_padding % 1 == 0 else w_padding - 0.5
        top_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        bottom_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5

        if self.is_expand:
            random.seed(time.time())

            random_pad = random.randint(1, self.offset)
            random_left, random_top = random.randint(0, random_pad), random.randint(0, random_pad)

            padding=[int(left_pad + random_left), int(top_pad + random_top), 
                    int(right_pad + random_pad - random_left), int(bottom_pad + random_pad - random_top)]
        else:
            padding=[int(left_pad), int(top_pad), int(right_pad), int(bottom_pad)]

        img = F.pad(img=image, padding=padding, fill=self.fill)

        return img


class RandomDilate(object):
    " 腐蚀 "
    def __init__(self, ksize=3):
        self.maxpool = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize - 1) / 2))

    def __call__(self, x):
        tensor_dilate = self.maxpool(x)
        return tensor_dilate


class RandomErode(object):
    " 膨胀 "
    def __init__(self, ksize=3):
        self.maxpool = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize - 1) / 2))

    def __call__(self, x):
        tensor_erode = -self.maxpool(-x)
        return tensor_erode


class AddBackground():
    " 给手写字添加背景 "
    def __init__(self, background: tensor, is_blur=True) -> None:
        self.background = background
        self.is_blur = is_blur

    # 输入为原始的、经resize、归一化等后的图像
    def __call__(self, img: tensor):

        # 以一定的阈值获得字符的位置，实际训练时效果不如下面的
        # label = torch.where(img > 0.7, torch.tensor([0.0]), torch.tensor([1.0]))
        # blend_tensor = torch.where(self.label == 0, self.background, image * (random.randint(11, 19) / 10))

        blend_tensor = torch.min(img, self.background)
        if self.is_blur:
            blend_tensor *= (random.randint(11, 19) / 10)
            blend_tensor[blend_tensor >= 1] = 1
        
        return blend_tensor


class Add_Lines(object):
    def __init__(self):
        self.line_color = ['black', 'gray', 'darkgray', 'grey', 'dimgray']

    def __call__(self, img):
        img = F.to_pil_image(img)
        width, height = img.size

        drawer = ImageDraw.Draw(img)
        line_color = random.choice(self.line_color)

        # 上
        if random.choice(['true', 'false']) == 'true':
            x1, y1 = random.randint(0, 5), random.randint(0, 5)
            x2, y2 = random.randint(width - 5, width), random.randint(0, 5)

            drawer.line(((x1, y1), (x2, y2)), fill=line_color, width=random.randint(1, 4))
        
        # 下
        if random.choice(['true', 'false']) == 'true':
            x1, y1 = random.randint(0, 5), random.randint(height - 5, height)
            x2, y2 = random.randint(width - 5, width), random.randint(height - 5, height)

            drawer.line(((x1, y1), (x2, y2)), fill=line_color, width=random.randint(1, 4))
        
        # 左
        if random.choice(['true', 'false']) == 'true':
            x1, y1 = random.randint(0, 5), random.randint(0, 5)
            x2, y2 = random.randint(0, 5), random.randint(height - 5, height)

            drawer.line(((x1, y1), (x2, y2)), fill=line_color, width=random.randint(1, 4))
        
        # 右
        if random.choice(['true', 'false']) == 'true':
            x1, y1 = random.randint(width - 5, width), random.randint(0, 5)
            x2, y2 = random.randint(width - 5, width), random.randint(height - 5, height)

            drawer.line(((x1, y1), (x2, y2)), fill=line_color, width=random.randint(1, 4))

        # x, y = random.randint(width / 2 + 5, width / 2 + 5), random.randint(height / 2 + 5, height / 2 + 5)
        # line_color = random.choice(self.line_color)

        # for i in range(0, width, 4):
        #     drawer.line(((i, y), (i + 2, y)), fill=line_color, width=1)
        
        # for i in range(0, height, 4):
        #     drawer.line(((x, i), (x, i + 2)), fill=line_color, width=1)

        return F.to_tensor(img)


class Add_Points(object):
    def __init__(self):
        self.line_color = ['black', 'gray', 'darkgray', 'grey', 'dimgray']

    def __call__(self, img):
        img = F.to_pil_image(img)
        width, height = img.size
        drawer = ImageDraw.Draw(img)
        for i in range(random.randint(1, 6)):
            x1 = random.randrange(width)
            y1 = random.randrange(height)
            radius = random.randint(1, 4)
            drawer.ellipse(((x1, y1), (x1 + radius, y1 + radius)), fill=random.choice(self.line_color), width=4)
        return F.to_tensor(img)



class LocalBlur(object):
    """
    对输入的图像张量进行局部模糊处理。

    参数：
        radius (int): 模糊半径。

    """
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img_tensor):
        img = F.to_pil_image(img_tensor)
        width, height = img.size
        blur_radius = self.radius

        # 对图像进行局部模糊处理
        for y in range(0, height, 30):
            for x in range(0, width, 30):
                if random.choice(['true', 'false']) == 'true':
                    continue
                box = (x, y, x + random.randint(10, 30), y+random.randint(10, 30))
                region = img.crop(box)
                blurred_region = region.filter(ImageFilter.GaussianBlur(blur_radius))
                img.paste(blurred_region, box)

        return F.to_tensor(img)


class AddSaltPepperNoise(object):

    def __init__(self, density=0.2):
        self.density = density

    def __call__(self, img):
        img = F.to_pil_image(img)

        img = np.array(img)                                                             # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255                                                            # 盐
        img= Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片

        return F.to_tensor(img)
        return img


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=20):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = F.to_pil_image(img)

        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        return F.to_tensor(img)
        return img



if __name__ == "__main__":
    
    from PIL import Image

    img = "/home/wangjiakai/ChineseCharacterOCR/1.png"

    img = LocalBlur()(Image.open(img).convert("RGB"))
    img.save("/home/wangjiakai/ChineseCharacterOCR/11.png")

    