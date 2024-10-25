from model import Net
import pickle
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


standard_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


if __name__ == '__main__':
    trained_path = 'output/classify-9663/model.pth'

    net = Net(trained_path=trained_path).cuda()
    # net.load_state_dict(torch.load(trained_path, map_location='cpu'), strict=False)
    net.eval()

    with open('dataset/chinese_ocr_keys.txt', 'r', encoding='utf-8') as file:
        keys = file.readlines()

    keys_num = 4249
    for i in range(keys_num):
        keys[i] = keys[i].strip('\n')
    
    # print(keys)

    stdPath = "C:\MyDataset\standard_images"

    standard_inputs = []
    # standard_idx = []
    standard_feas = []
    standard_cls = []

    for i in tqdm(range(keys_num)):
        name = keys[i]
        pth = os.path.join(stdPath, name + '.png')
        img = Image.open(pth).convert('RGB')

        standard_inputs.append(standard_transform(img).unsqueeze(0).detach())

        fea, cls = net(standard_transform(img).unsqueeze(0).cuda())

        standard_feas.append(fea.detach())
        # print(cls.size())
        # standard_cls.append(cls.detach())
    
    with open('standard_inputs', 'wb') as f:
        pickle.dump(standard_inputs, f)
    
    with open('standard_feas', 'wb') as f:
        pickle.dump(standard_feas, f)
    
    # with open('standard_cls', 'wb') as f:
    #     pickle.dump(standard_cls, f)

    # print(standard_idx)