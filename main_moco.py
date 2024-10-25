import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from model import Net
import pandas as pd
from utils.ClassDataset import ClassDataset
from moco import HW_MoCo
from torch.cuda.amp import GradScaler
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
        '--lr-model', default=1e-4, type=float, help='Learning rate in optimizer'
    )
    parser.add_argument(
        '--MoCo-m', default=0.99, type=float, help='MoCo momentum of updating momentum encoder (default: 0.99)'
    )
    parser.add_argument(
        '--temperature', default=0.2, type=float, help='Temperature used in InforNCE loss'
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
    
    return parser.parse_args()


from tqdm import tqdm
from torch.cuda.amp import autocast
import torch


def train(net, train_loader, optimizer, scaler, epoch, args=None):
    net.train()
    total_contrastive_loss, total_classifier_loss = 0.0, 0.0
    total_num, train_bar = 0, tqdm(train_loader)

    for pos_1, pos_2 in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        # contrastive loss
        with autocast(True):
            loss = net(x1=pos_1, x2=pos_2, m=args.MoCo_m)

        total_contrastive_loss += loss.detach().item()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_num += 1
        train_bar.set_description('Train Epoch: [{}/{}] Contrastive_Loss:{:.4f}'.
                                  format(epoch, args.epochs, total_contrastive_loss / total_num))
    return total_contrastive_loss / total_num


def test(net, test_loader, epoch, args=None):
    net.eval()
    total_contrastive_loss, total_num, test_bar = 0.0, 0, tqdm(test_loader)
    with torch.no_grad():
        for pos_1, pos_2 in test_bar:
            # contrastive loss
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
            loss = net(x1=pos_1, x2=pos_2, m=args.MoCo_m)

            total_contrastive_loss += loss.detach().item()
            total_num += 1
            test_bar.set_description('Test Epoch: [{}/{}]  Contrastive_Loss: {:.4f}'
                                     .format(epoch, args.epochs, total_contrastive_loss / total_num))
    return total_contrastive_loss / total_num


def main(args):
    batch_size, epochs = args.batch_size, args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data prepare
    train_data = ClassDataset(args.dataset_path, args.background_path, is_train=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=4,
                              shuffle=True, drop_last=True, pin_memory=True)

    test_data = ClassDataset(args.dataset_path, args.background_path, is_train=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = HW_MoCo(pre_path=args.pretrained_path).to(device)
    optimizer_model = optim.AdamW(model.parameters(), lr=args.lr_model * batch_size / 256)

    # training loop
    results = {'train_Loss': [], 'test_Loss': []}
    results_dir = args.output_path
    save_name_pre = 'pretrain_512dim'

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    scaler = GradScaler()
    best_loss = 50
    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer_model, scaler, epoch, args)
        results['train_Loss'].append(loss)

        test_Loss = test(model, test_loader, epoch, args)
        results['test_Loss'].append(test_Loss)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_statistics2.txt'.format(results_dir, save_name_pre), index_label='epoch')

        if test_Loss < best_loss:
            best_loss = test_Loss
            torch.save(model.state_dict(), '{}/{}_model2.pth'.format(results_dir, save_name_pre))


if __name__ == '__main__':
    args = parse_args()

    setup_seed(3407)
    main(args)
