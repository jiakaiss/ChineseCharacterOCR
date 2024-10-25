from torch.nn import TripletMarginLoss
from torch import tensor
import torch


def get_triplet_loss(evaluations: tensor, margin: int):
    triplet_criterion1, triplet_criterion2 = TripletMarginLoss(margin=margin, p=2), TripletMarginLoss(margin=margin * 2, p=2)
    triplet_criterion3, triplet_criterion4 = TripletMarginLoss(margin=margin / 2, p=2), TripletMarginLoss(margin=margin / 2, p=2)

    # 间隔构造三元组，分差使用预设的超参数margin
    anchor1, positives1, negatives1 = [], [], []

    # 更大间隔地构造三元组，分差在marigin基础上 *2
    anchor2, positives2, negatives2 = [], [], []

    # 相邻构造三元组，分差在margin基础上 /2
    anchor3, positives3, negatives3 = [], [], []

    # 约束第一个、第二个手写字，避免分数过于接近标准字，分差在margin基础上 /2
    anchor4, positives4, negatives4 = [], [], []

    batch_size = evaluations.size(0) // 6
    for i in range(batch_size):

        # A13, A24, A35
        anchor1.append(evaluations[i].repeat(3, 1))
        
        positives1.append(evaluations[batch_size * 1 + i].unsqueeze(0))
        negatives1.append(evaluations[batch_size * 3 + i].unsqueeze(0))

        positives1.append(evaluations[batch_size * 2 + i].unsqueeze(0))
        negatives1.append(evaluations[batch_size * 4 + i].unsqueeze(0))

        positives1.append(evaluations[batch_size * 3 + i].unsqueeze(0))
        negatives1.append(evaluations[batch_size * 5 + i].unsqueeze(0))

        # A14, A15, A25
        anchor2.append(evaluations[i].repeat(3, 1))

        positives2.append(evaluations[batch_size * 1 + i].unsqueeze(0))
        negatives2.append(evaluations[batch_size * 4 + i].unsqueeze(0))

        positives2.append(evaluations[batch_size * 1 + i].unsqueeze(0))
        negatives2.append(evaluations[batch_size * 5 + i].unsqueeze(0))

        positives2.append(evaluations[batch_size * 2 + i].unsqueeze(0))
        negatives2.append(evaluations[batch_size * 5 + i].unsqueeze(0))

        # A12, A23, A34, A45
        anchor3.append(evaluations[i].repeat(4, 1))

        positives3.append(evaluations[batch_size * 1 + i].unsqueeze(0))
        negatives3.append(evaluations[batch_size * 2 + i].unsqueeze(0))

        positives3.append(evaluations[batch_size * 2 + i].unsqueeze(0))
        negatives3.append(evaluations[batch_size * 3 + i].unsqueeze(0))

        positives3.append(evaluations[batch_size * 3 + i].unsqueeze(0))
        negatives3.append(evaluations[batch_size * 4 + i].unsqueeze(0))

        positives3.append(evaluations[batch_size * 4 + i].unsqueeze(0))
        negatives3.append(evaluations[batch_size * 5 + i].unsqueeze(0))

        # AA1、AA2
        anchor4.append(evaluations[i].repeat(2, 1))

        positives4.append(evaluations[i].unsqueeze(0))
        negatives4.append(evaluations[batch_size * 1 + i].unsqueeze(0))

        positives4.append(evaluations[i].unsqueeze(0))
        negatives4.append(evaluations[batch_size * 2 + i].unsqueeze(0))


    anchor1, anchor2 = torch.cat(anchor1, dim=0), torch.cat(anchor2, dim=0)
    positives1, positives2 = torch.cat(positives1, dim=0), torch.cat(positives2, dim=0)
    negatives1, negatives2 = torch.cat(negatives1, dim=0), torch.cat(negatives2, dim=0)

    anchor3, positives3, negatives3 = torch.cat(anchor3, dim=0), torch.cat(positives3, dim=0), torch.cat(negatives3, dim=0)
    anchor4, positives4, negatives4 = torch.cat(anchor4, dim=0), torch.cat(positives4, dim=0), torch.cat(negatives4, dim=0)

    # print(anchor1)
    # print(positives1)
    # print(negatives1)
    # print(anchor2)
    # print(positives2)
    # print(negatives2)

    return (triplet_criterion1(anchor1, positives1, negatives1) + triplet_criterion2(anchor2, positives2, negatives2)
            + triplet_criterion3(anchor3, positives3, negatives3) + triplet_criterion4(anchor4, positives4, negatives4)) / 4



if __name__ == "__main__":

    t = torch.tensor([
        [0.1],
        [0.2],
        [1.1],
        [1.2],
        [2.1],
        [2.2],
        [3.1],
        [3.2],
        [4.1],
        [4.2],
        [5.1],
        [5.2]
    ])

    print(get_triplet_loss(t, 10))
