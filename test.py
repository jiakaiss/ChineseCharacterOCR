import os
import random
import shutil
from tqdm import tqdm

from PIL import Image
from model import Net
from tqdm import tqdm
import torch
import shutil
from torchvision import transforms

ts = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

model_pth = "output\classify-9663\model.pth"
model = Net().cuda()
model.load_state_dict(torch.load(model_pth, map_location='cpu'), strict=False)
model.eval()


if __name__ == "__main__":
    tar = "test\\test_score"

    with open('dataset\chinese_ocr_keys.txt', 'r', encoding='utf-8') as file:
        keys = file.readlines()
    for i in range(len(keys)):
        keys[i] = keys[i].strip('\n')

    for dir in os.listdir(tar):
        for pth in tqdm(os.listdir(os.path.join(tar, dir))):
            img = Image.open(os.path.join(tar, dir, pth)).convert('RGB')

            img = ts(img).unsqueeze(0).cuda()

            _, cls = model(img)
            _, predicted = torch.max(cls, 1)

            name = keys[predicted.item()]

            i = 1
            while os.path.exists(os.path.join(tar, dir, str(predicted.item()) + "_" + str(i) + '.png')):
                i += 1
            os.rename(os.path.join(tar, dir, pth), os.path.join(tar, dir, str(predicted.item()) + "_" + str(i) + '.png'))
