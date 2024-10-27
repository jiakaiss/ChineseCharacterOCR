import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models.resnet import resnet18 as resnet
import torch.nn.functional as F


def _build_conv(in_channels, out_channels):
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1))
    ]
    return nn.Sequential(*conv)


def _attention(flatten, x):
    attention = flatten * x
    output = flatten + attention
    return output


class Net(nn.Module):
    def __init__(self, trained_path=None, fea_dim=512, num_classes=4248 + 1, is_evaluate=False, hidden=256):
        super(Net, self).__init__()
        self.is_evaluate, self.num_classes = is_evaluate, num_classes

        f = []
        for _, module in resnet().named_children():
            # and not isinstance(module, nn.AdaptiveAvgPool2d)
            if not isinstance(module, nn.Linear):
                f.append(module)
                
        self.resnet = nn.Sequential(*f, nn.Flatten(),)
        self.classify = nn.Sequential(
            nn.Linear(fea_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )

        # 笔画预测器，预测是否含有某一笔画（笔画的种类数为25）
        self.stroke_predict = nn.Sequential(
            nn.Linear(fea_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, 25)
        )

        # # 预测前20笔画
        # self.conv0 = _build_conv(in_channels=fea_dim, out_channels=fea_dim)
        # self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        # self.strokes_fc = [nn.Linear(fea_dim, 26, device='cuda') for i in range(10)]

        # self.class_fea = nn.Parameter(torch.randn((num_classes, fea_dim), requires_grad=True))
        # self.standard_fea = nn.Parameter(torch.randn((num_classes, fea_dim), requires_grad=True))

        # self.evaluate = nn.Sequential(
        #     nn.Linear(fea_dim, hidden), 
        #     # nn.Linear(fea_dim + self.num_classes, hidden), 
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden, 3, bias=True),
        #     nn.Softmax(dim=1)
        # )
        
        if trained_path is not None:
            print("----loading pretrained model----")
            self.load_state_dict(torch.load(trained_path, map_location='cpu'), strict=False)

        if is_evaluate:

            # 评分网络
            self.evaluate = nn.Sequential(
                nn.Linear(fea_dim + self.num_classes, hidden), 
                nn.PReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 3, bias=True),
                # nn.Softmax(dim=1)
            )

            for param_b in self.resnet.parameters():
                param_b.requires_grad = False  
            for param_c in self.classify.parameters():
                param_c.requires_grad = False               # not update by gradient
            
            import pickle
            with open('standard_feas', 'rb') as f:
                self.standard_fea = pickle.load(f)
                for i in range(len(self.standard_fea)):
                    self.standard_fea[i] = self.standard_fea[i].cpu()
            self.standard_fea = torch.cat(self.standard_fea)


    def forward(self, x, strokes_predict=False):
        feas = self.resnet(x)
        cls = self.classify(feas)

        if self.is_evaluate:

            _, labels = torch.max(cls, 1)

            label_onehot = F.one_hot(labels, self.num_classes)
            # print(labels)

            self.standard_fea = self.standard_fea.to(x.device)
            standard_feas = self.standard_fea.index_select(dim=0, index=labels)

            to_score = torch.tensor([
                [100, 70, 40]
            ], dtype=torch.float).t().to(x.device)

            score = torch.mm(
                nn.Softmax(dim=1)(
                    self.evaluate(
                        torch.cat((feas - standard_feas, label_onehot), -1)
                    )
                )
            , to_score)

            return score
        
        else:
            if not strokes_predict:
                return feas, cls
            
            return feas, cls, self.stroke_predict(feas)

            # 用于笔画顺序预测的，效果不佳，模型难以拟合
            # batch_size = x.size(0)
            # fc_s = [self.conv0(feas)]
            # for i in range(19):
            #     fc = _attention(AvgPool, fc_s[-1])
            #     fc_s.append(fc)

            # strokes_predict = []
            # for i in range(10):

            #     stroke_cls = F.softmax(self.strokes_fc[i](fc_s[i].view(batch_size, -1)), dim=1)
            #     strokes_predict.append(stroke_cls)
            
            # return feas, cls, torch.stack(strokes_predict, 1)


if __name__ == '__main__':
    net = Net(is_evaluate=False).cpu()
    summary(net, (3, 64, 64), 1, 'cpu')
