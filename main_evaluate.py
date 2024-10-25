import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim
from model import Net
from utils.train_and_test import train2, test2
from utils.EvaluateDataset import EvaluateDataset
from utils.set_rand_seed import setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Number of images in each mini-batch'
    )
    parser.add_argument(
        '--epochs', type=int, default=120, help='Number of sweeps over the dataset to train'
    )
    parser.add_argument(
        '--lr', default=0.0001, type=float, help='Learning rate in optimizer'
    )
    parser.add_argument(
        '--margin', default=10, type=float, help='Margin used in triplet loss'
    )
    # parser.add_argument(
    #     '--coefficient', default=0.45, type=float, help='Coefficient used in balance two loss'
    # )

    parser.add_argument(
        '--dataset-path', type=str, help='train dataset path'
    ) 
    parser.add_argument(
        '--background-path', type=str, help='add background to the image'
    ) 

    parser.add_argument(
        '--standard-path', type=str, help='standard image path, the target is 100'
    ) 
    parser.add_argument(
        '--good-path', type=str, help='good image path, the target is 95'
    ) 
    parser.add_argument(
        '--medium-path', type=str, help='medium image path, the target is 70'
    )
    parser.add_argument(
        '--bad-path', type=str, help='bad image path, the target is 40'
    ) 

    parser.add_argument(
        '--output-path', type=str, help='train output path'
    ) 
    parser.add_argument(
        '--pretrained-path', type=str, help='pretrained model path'
    ) 
    
    return parser.parse_args()


def main(args):
    batch_size, epochs = args.batch_size, args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data prepare
    train_data = EvaluateDataset(args.dataset_path, args.background_path, args.standard_path, args.good_path, args.medium_path, args.bad_path, is_train=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=4,
                              shuffle=True, drop_last=True, pin_memory=True)

    test_data = EvaluateDataset(args.dataset_path, args.background_path, args.standard_path, is_train=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = Net(is_evaluate=True, trained_path=args.pretrained_path).to(device)
    optimizer_model = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # training loop
    results = {'reg_loss': [], 'reg_good_loss':[], 'reg_medium_loss':[], 'reg_bad_loss': [], 'std_loss':[], 'triplet_loss': [],
               'reg_test_loss': [], 'triplet_test_loss': []}
    results_dir = args.output_path
    # save_name_pre = 'lr:{},margin:{}'.format(args.lr, args.margin)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    best_loss1, best_loss2 = 50, 50
    for epoch in range(1, epochs + 1):
        loss1, loss2, loss3, loss4, loss5, loss6 = train2(model, train_loader, optimizer_model, epoch, device, args)
        results['reg_loss'].append(loss1)
        results['reg_good_loss'].append(loss2)
        results['reg_medium_loss'].append(loss3)
        results['reg_bad_loss'].append(loss4)
        results['std_loss'].append(loss5)
        results['triplet_loss'].append(loss6)

        test_1, test_2 = test2(model, test_loader, epoch, device, args)
        results['reg_test_loss'].append(test_1)
        # results['sym_test_loss'].append(test_2)
        results['triplet_test_loss'].append(test_2)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/log.txt'.format(results_dir), index_label='epoch')
        # if loss2 < best_loss1:
        #     best_loss1 = loss2
        #     torch.save(model.state_dict(), '{}/{}_model_min train loss.pth'.format(results_dir, save_name_pre))

        if test_2 < best_loss2:
            best_loss2 = test_2
            torch.save(model.state_dict(), '{}/model.pth'.format(results_dir))
    
    print('the best triplet loss is ' + str(best_loss2))


if __name__ == '__main__':

    args = parse_args()
    print("The result will save to " + args.output_path)
    setup_seed(3407)
    main(args)
