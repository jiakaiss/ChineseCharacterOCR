from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss, SmoothL1Loss, BCEWithLogitsLoss

import os
import sys
sys.path.append(os.path.abspath('.'))
from utils.get_triplet_loss import get_triplet_loss


def get_stroke_labels(label, char_list, label_to_strokes: dict, strokes_list: list):
    # print(label)
    stroke_labels_orders = []
    stroke_labels_exists = []
    for i in label:
        # i = i.item()
        if i == 0:
            stroke_labels_orders.append([25 for x in range(10)])
            stroke_labels_exists.append([0 for x in range(25)])
        else:
            assert char_list[i] in label_to_strokes.keys(), "没有这个汉字的笔顺信息  " + char_list[i]

            curr_strokes = label_to_strokes[char_list[i]].split(',')
            # print(char_list[i], curr_strokes)

            # 是否含有某一笔画
            curr_stroke_labels_exist = []
            for i in range(25):
                if strokes_list[i] in curr_strokes:
                    curr_stroke_labels_exist.append(1)
                else:
                    curr_stroke_labels_exist.append(0)
            stroke_labels_exists.append(curr_stroke_labels_exist)

            # 笔顺
            curr_stroke_idx = []
            for idx in range(10):
                if idx >= len(curr_strokes):
                    curr_stroke_idx.append(25)
                    continue
                
                curr_stroke_idx.append(strokes_list.index(curr_strokes[idx]))
            stroke_labels_orders.append(curr_stroke_idx)

    return stroke_labels_orders, stroke_labels_exists


def train(net, train_loader, optimizer_model, epoch, device, args=None,  center_loss_criterion=None, optimizer_center_loss=None,
          char_list: list=None, label_to_strokes: dict=None, strokes_list: list=None):
    net.train()
    total_classify_loss, total_stroke_classify_loss, total_center_loss, total_num, train_bar = 0.0, 0.0, 0.0, 0, tqdm(train_loader)

    for data, labels in train_bar:

        data, labels = data.to(device), labels.to(device)

        if args.use_strokes_predict:

            stroke_labels_orders, stroke_labels_exists = get_stroke_labels(labels, char_list, label_to_strokes, strokes_list)

            # stroke_labels_exists = torch.Tensor(stroke_labels_exists).long().to(device)
            
            stroke_labels_exists = torch.Tensor(stroke_labels_exists).float().to(device)

            _, predict, predict_strokes = net(data, strokes_predict=True)

            loss1 = CrossEntropyLoss()(predict, labels)

            # 是否含有某一笔画，二分类
            loss2 = BCEWithLogitsLoss()(predict_strokes, stroke_labels_exists)

            # # 笔顺的
            # loss2 = CrossEntropyLoss()(predict_strokes.transpose(2, 1), stroke_labels)

            total_loss1 += loss1.detach().item()
            total_loss2 += loss2.detach().item()
            
            loss_sum = loss1 + args.lamda * loss2
        
        else:
            feas, predict = net(data)

            loss1 = CrossEntropyLoss()(predict, labels)
            total_classify_loss += loss1.detach().item()
            
            loss_sum = loss1

            # use centerloss
            if args.use_centerloss:

                loss_center = center_loss_criterion(feas, labels)
                total_center_loss += loss_center.detach().item()
                optimizer_center_loss.zero_grad()
                loss_sum += args.lamda_centerloss * loss_center

        optimizer_model.zero_grad()
        loss_sum.backward()
        optimizer_model.step()

        if args.use_centerloss:
            # by doing so, weight_cent would not impact on the learning of centers
            for param in center_loss_criterion.parameters():
                param.grad.data *= (1. / args.lamda_centerloss)
            
            optimizer_center_loss.step()

        total_num += 1
        train_bar.set_description('Train Epoch: [{}/{}] Classify_Loss:{:.4f}, Classify_Strokes_Loss:{:.4f}, Center_Loss:{:.4f}'.
                                  format(epoch, args.epochs, total_classify_loss / total_num, total_stroke_classify_loss / total_num, total_center_loss / total_num))
        
    return total_classify_loss / total_num, total_stroke_classify_loss / total_num, total_center_loss / total_num


def test(net, test_loader, epoch, device, args=None, center_loss_criterion=None,
         char_list: list=None, label_to_strokes: dict=None, strokes_list: list=None):
    net.eval()

    # total_stroke_order_correct
    total_accuracy, total_stroke_correct, total_center_loss, total_num, test_bar = 0, 0, 0.0, 0, tqdm(test_loader)

    with torch.no_grad():
        for data, label in test_bar:

            data, labels = data.to(device), label.to(device)

            if args.use_strokes_predict:
                feas, predict, predict_strokes = net(data, strokes_predict=True)

                # ACC
                _, predict = torch.max(predict, 1)
                accuracy = (predict == labels).sum().item()
                total_accuracy += accuracy

                # Stroke_Exist_ACC, 用于预测是否存在这一笔画
                _, stroke_labels_exists = get_stroke_labels(labels, char_list, label_to_strokes, strokes_list)
                stroke_labels_exists = torch.Tensor(stroke_labels_exists).float().to(device)

                # 某一维度上的阈值 > 0 说明预测正确
                total_stroke_correct += ((predict_strokes > 0) == stroke_labels_exists).all(dim=-1).sum().item()

                # # Stroke_Order_ACC  用于预测笔画顺序的，在此处效果不佳，可能是笔画种类太多、数据的质量不佳有关
                # stroke_labels = get_stroke_labels(labels, char_list, label_to_strokes, strokes_list)
                # stroke_labels = torch.Tensor(stroke_labels).long().to(device)

                # predict_strokes = torch.argsort(predict_strokes, dim=-1, descending=True)
                # total_stroke_order_correct += torch.sum(
                #     (predict_strokes[:, :, 0:1] == stroke_labels.unsqueeze(dim=-1)).all(dim=-2).float()).item()
            
            else:
                feas, predict = net(data)

                # ACC
                _, predict = torch.max(predict, 1)
                accuracy = (predict == labels).sum().item()
                total_accuracy += accuracy

            total_num += data.size(0)

            if args.use_centerloss:
                total_center_loss += center_loss_criterion(feas, labels).detach().item()

            test_bar.set_description('Test Epoch: [{}/{}] total_accuracy: {:.4f}, total_accuracy_strokes:{:.4f}, Center_Loss:{:.4f}'.format(
                epoch, args.epochs, total_accuracy / total_num, total_stroke_correct / total_num, total_center_loss / total_num))

    return total_accuracy / total_num, total_stroke_correct / total_num, total_center_loss / total_num


def train2(net, ordered_loader, optimizer, epoch, device, args):
    net.eval()
    net.evaluate.train()

    reg_criterion = SmoothL1Loss()
    total_reg_loss, total_reg_std, total_triplet_loss, total_num1, total_num2 = 0.0, 0.0, 0.0, 1e-6, 1e-6
    total_reg_loss_good, total_reg_loss_medium, total_reg_loss_bad = 0.0, 0.0, 0.0

    ordered_bar = tqdm(ordered_loader)
    for data_change, scores_random, paths, standards, standards_scores, goods, mediums, bads, goods_scores, mediums_scores, bads_scores in ordered_bar:
        # print(paths)

        data_change, scores_random = torch.cat(data_change).to(device), torch.cat(scores_random).to(device).unsqueeze(1)
        standards, standards_scores = torch.cat(standards).to(device), torch.cat(standards_scores).to(device).unsqueeze(1)
        goods, mediums, bads = torch.cat(goods).to(device),  torch.cat(mediums).to(device), torch.cat(bads).to(device)

        goods_scores, mediums_scores = torch.cat(goods_scores).to(device).unsqueeze(1), torch.cat(mediums_scores).to(device).unsqueeze(1)
        bads_scores = torch.cat(bads_scores).to(device).unsqueeze(1)

        evaluates = net(data_change)
        std_out, goods_out, mediums_out, bads_out = net(standards), net(goods), net(mediums), net(bads)

        # 1个标准字和5个排序的手写字设定随机分数，standard_loss将标准字拉至100分
        evaluate_loss = reg_criterion(evaluates, scores_random)
        standard_loss = reg_criterion(std_out, standards_scores)

        # good 的手写字分数设为80-95，medium 的手写字分数设为60-80，bad 的手写字分数设为40-60
        good_loss = reg_criterion(goods_out, goods_scores)
        medium_loss = reg_criterion(mediums_out, mediums_scores)
        bad_loss = reg_criterion(bads_out, bads_scores)

        total_reg_loss += evaluate_loss.detach().item()
        total_reg_loss_good += good_loss.detach().item() 
        total_reg_loss_medium += medium_loss.detach().item()
        total_reg_loss_bad += bad_loss.detach().item()
        total_reg_std += standard_loss.detach().item()

        if epoch <= 30:
            # 在初始阶段时，将分数拉至预设的假分数附近，同时保持住标准字的100分。
            loss_sum = evaluate_loss + (good_loss + medium_loss + bad_loss) / 3 + standard_loss
        
        else:
            # 到一定的阶段之后，使用新的策略。加入使用 triplet loss拉开分差。

            triplet_loss = get_triplet_loss(evaluates, args.margin)
            total_triplet_loss += triplet_loss.detach().item()
            loss_sum = evaluate_loss + (good_loss + medium_loss + bad_loss) / 3 + standard_loss + triplet_loss
            total_num2 += 1

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        total_num1 += 1
        ordered_bar.set_description('Train Epoch:[{}/{}] Reg:{:.4f}, Reg_Good:{:.4f}, Reg_Medium:{:.4f}, Reg_Bad:{:.4f}, Std:{:.4f}, Triplet:{:.4f}'.format(
            epoch, args.epochs, total_reg_loss / total_num1, total_reg_loss_good / total_num1, total_reg_loss_medium / total_num1,
            total_reg_loss_bad / total_num1, total_reg_std / total_num1, total_triplet_loss / total_num2))
        
    return total_reg_loss / total_num1, total_reg_loss_good / total_num1, total_reg_loss_medium / total_num1, total_reg_loss_bad / total_num1, total_reg_std / total_num1, total_triplet_loss / total_num2


def test2(net, order_loader, epoch, device, args):
    net.eval()
    total_reg_loss, total_triplet_loss, total_reg_loss2, total_num, order_bar = 0.0, 0.0, 0.0, 0, tqdm(order_loader)
    reg_criterion = SmoothL1Loss()

    with torch.no_grad():
        for data, scores_random in order_bar:
            
            data, scores_random = torch.cat(data).to(device), torch.cat(scores_random).to(device).unsqueeze(1)
            # data_change = torch.cat(data_change).to(device)
            
            evaluates1 = net(data)

            evaluate_loss = reg_criterion(evaluates1, scores_random)
            # symmetry_loss = reg_criterion(evaluates1, evaluates2)
            total_reg_loss += evaluate_loss.detach().item()
            # total_reg_loss2 += symmetry_loss.detach().item()

            # triplet loss
            triplet_loss = get_triplet_loss(evaluates1, 15)
            total_triplet_loss += triplet_loss.detach().item()

            total_num += 1
            order_bar.set_description('Test Epoch: [{}/{}] Reg_Loss:{:.4f}, Triplet_Loss: {:.4f}'
                                      .format(epoch, args.epochs, total_reg_loss / total_num,
                                              total_triplet_loss / total_num))

    return total_reg_loss / total_num, total_triplet_loss / total_num



if __name__ == "__main__":

    with open('/home/wangjiakai/ChineseCharacterOCR/dataset/chinese_ocr_keys.txt', 'r') as file:
        char_list = file.readlines()
        for i in range(len(char_list)):
            char_list[i] = char_list[i].strip('\n')
    
    with open('/home/wangjiakai/ChineseCharacterOCR/dataset/chinese_strokes_labels.txt', 'r') as file:
        strokes_list = file.readlines()
        for i in range(len(strokes_list)):
            strokes_list[i] = strokes_list[i].strip('\n')
    
    with open('/home/wangjiakai/ChineseCharacterOCR/dataset/chinese_strokes_order.txt', 'r') as file:
        lines = file.readlines()
        label_to_strokes = dict()
        for line in lines:
            label, strokes = line.strip('\n').split(' ')
            label_to_strokes[label] = strokes
    
    print(get_stroke_labels([0, 22, 33], char_list, label_to_strokes, strokes_list))
