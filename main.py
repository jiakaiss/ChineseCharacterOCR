import argparse
import os
import torch
import torch.backends
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch import optim
from model import Net
import pandas as pd
from utils.train_and_test import train, test
from utils.OCRDataset import HandWrittenDataset
from utils.CenterLoss import CenterLoss
from utils.set_rand_seed import setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=2048, help='Number of images in each mini-batch'
    )
    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of sweeps over the dataset to train'
    )
    parser.add_argument(
        '--lr-model', default=0.001, type=float, help='Learning rate in optimizer'
    )

    parser.add_argument(
        '--pretrained-path', type=str, help='pretrained model path'
    ) 
    parser.add_argument(
        '--dataset-path', type=str, default=None, help='train dataset path'
    )
    parser.add_argument(
        '--background-path', type=str, help='add background to the image'
    ) 
    parser.add_argument(
        '--output-path', type=str, help='train output path'
    )
    parser.add_argument(
        '--use-strokes-predict', action='store_true', help='if choose, use strokes predict'
    )
    parser.add_argument(
        '--use-centerloss', action='store_true', help='if choose, use center loss'
    )

    parser.add_argument(
        '--warmUp', type=int, default=3, help='if choose, use centerloss'
    )
    parser.add_argument(
        '--lamda_centerloss', default=0.1, type=float, help='Parameter use to balance more than one loss'
    )
    
    return parser.parse_args()


def main(args):
    batch_size, epochs = args.batch_size, args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data prepare
    train_data = HandWrittenDataset(args.dataset_path, args.background_path, is_train=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=8,
                              shuffle=True, drop_last=True, pin_memory=True)

    test_data = HandWrittenDataset(args.dataset_path, args.background_path, is_train=False)
    test_loader = DataLoader(dataset=test_data, batch_size=512, num_workers=4, pin_memory=True)

    model = Net(trained_path=args.pretrained_path).to(device)
    optimizer_model = optim.AdamW(model.parameters(), lr=args.lr_model)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_model, T_0=20)

    if args.use_centerloss:
        center_loss_criterion = CenterLoss(num_classes=4249, feat_dim=512, trained_path=args.pretrained_path)
        optimizer_center_loss = optim.AdamW(center_loss_criterion.parameters(), lr=args.lr_model)
    else:
        center_loss_criterion, optimizer_center_loss = None, None

    # training loop
    results = {'train_classify_Loss': [], 'train_stroke_classify_Loss': [], 'Center_Loss': [], 'accuracy1': [], 'accuracy2': [], 'Center_Loss_test': []}
    results_dir = args.output_path
    # save_name_pre = 'batch_size:{},lr:{}'.format(
    #     batch_size, args.lr_model, )

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    # 用于笔画预测的部分
    with open(os.path.join(args.dataset_path, 'chinese_ocr_keys.txt'), 'r', encoding='utf-8') as file:
        char_list = file.readlines()
        for i in range(len(char_list)):
            char_list[i] = char_list[i].strip('\n')
    
    with open(os.path.join(args.dataset_path, 'chinese_strokes_labels.txt'), 'r', encoding='utf-8') as file:
        strokes_list = file.readlines()
        for i in range(len(strokes_list)):
            strokes_list[i] = strokes_list[i].strip('\n')
    
    with open(os.path.join(args.dataset_path, 'chinese_strokes_order.txt'), 'r', encoding='utf-8') as file:
        lines = file.readlines()
        label_to_strokes = dict()
        for line in lines:
            label, strokes = line.strip('\n').split(' ')
            label_to_strokes[label] = strokes

    max_accuracy = 0
    for epoch in range(1, epochs + 1):
        # print('lr:', scheduler.get_last_lr())
        
        loss1, loss2, loss3 = train(model, train_loader, optimizer_model, epoch, device, args, center_loss_criterion, optimizer_center_loss,
                             char_list, label_to_strokes, strokes_list)
        results['train_classify_Loss'].append(loss1)
        results['train_stroke_classify_Loss'].append(loss2)
        results['Center_Loss'].append(loss3)
        
        # scheduler.step()

        accuracy1, accuracy2, test3 = test(model, test_loader, epoch, device, args, center_loss_criterion, char_list, label_to_strokes, strokes_list)
        results['accuracy1'].append(accuracy1)
        results['accuracy2'].append(accuracy2)
        results['Center_Loss_test'].append(test3)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/log.txt'.format(results_dir), index_label='epoch')

        if accuracy1 > max_accuracy:
            max_accuracy = accuracy1
            torch.save(model.state_dict(), '{}/model.pth'.format(results_dir))

    print('the best acc is ' + str(max_accuracy))

if __name__ == '__main__':
    args = parse_args()
    print("The result will save to " + args.output_path)

    setup_seed(3407)
    main(args)
