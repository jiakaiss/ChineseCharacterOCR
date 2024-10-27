import argparse
import os
import onnx
import onnxsim
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18 as resnet
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath('.'))


class ScoreNet(nn.Module):
    def __init__(self, fea_dim=512, num_classes=4249, hidden=256, is_evaluate=False):
        super(ScoreNet, self).__init__()

        self.num_classes = num_classes
        self.is_evaluate = is_evaluate

        # 识别汉字种类的网络
        f = []
        for _, module in resnet().named_children():
            if not isinstance(module, nn.Linear):
                f.append(module)
                
        self.resnet = nn.Sequential(*f, nn.Flatten())
        self.classify = nn.Sequential(
            nn.Linear(fea_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )

        if is_evaluate:
            # 进行评分的网络
            self.evaluate = nn.Sequential(
                nn.Linear(fea_dim + self.num_classes, hidden), 
                nn.PReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 3, bias=True),
            )

            import pickle
            with open('standard_feas', 'rb') as f:
                self.standard_feas = pickle.load(f)
                for i in range(len(self.standard_feas)):
                    self.standard_feas[i] = self.standard_feas[i].cpu()
            # print(self.standard_fea)
            self.standard_feas = torch.cat(self.standard_feas)
 

    def forward(self, x):

        fea = self.resnet(x)
        cls = self.classify(fea)

        if not self.is_evaluate:
            return cls

        _, label = torch.max(cls, 1)
        label_onehot = F.one_hot(label, self.num_classes)
        standard_fea = self.standard_feas[label].cpu()
        
        to_score = torch.tensor([
            [100, 70, 40]
        ], dtype=torch.float).t().to(x.device)

        score = torch.mm(
            nn.Softmax(dim=1)(
                self.evaluate(
                    torch.cat((fea - standard_fea, label_onehot), -1)
                )
            )
        , to_score)

        return cls, score



def main(model_path_ocr, output_path, input_shape, evaluate=False):

    model = ScoreNet(is_evaluate=evaluate)
    model.load_state_dict(torch.load(model_path_ocr, map_location='cpu'), strict=False)
    model.eval()

    dummy_input = torch.autograd.Variable(
        torch.randn((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    )

    # 导出主网络
    if evaluate:
        output_names = ["output_cls", "output_score"]
    else:
        output_names = ["output_cls"]

    torch.onnx.export(
        model,
        dummy_input,
        output_path + ".onnx",
        verbose=True,
        # keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=["data"],
        output_names=output_names,
    )
    print("finished exporting onnx ")

    print("start simplifying onnx ")
    input_data = {"data": dummy_input.detach().cpu().numpy(), }
    model_sim, flag = onnxsim.simplify(output_path + ".onnx", input_data=input_data)
    if flag:
        onnx.save(model_sim, output_path + ".onnx")
        print("simplify onnx successfully")
    else:
        print("simplify onnx failed")
    


def parse_args():
    parser = argparse.ArgumentParser(description='Export Inference Model.')

    parser.add_argument(
        "--model_path", type=str, default='', help="Path to .ckpt model."
    )

    parser.add_argument(
        "--out_path", type=str, help="Mnn model output path."
    )
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, e.g., `--input_shape 1 3 128 128`.",
        type=int,
        default=None
    )
    parser.add_argument(
        '--use_evaluate', action='store_true', help='if choose, use evaluate'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_path = args.model_path
    out_path = args.out_path
    input_shape = args.input_shape

    if args.use_evaluate:
        out_path += "_score"
        main(model_path, out_path, input_shape, evaluate=True)
    else:
        main(model_path, out_path, input_shape)
    
    print("Model onnx saved to:", out_path + ".onnx")

    print('-------------------export mnn begin!-----------------')
    os.system("python -m MNN.tools.mnnconvert -f ONNX --modelFile " + out_path + ".onnx" + " --MNNModel " + out_path + ".mnn")
    print('-------------------export mnn finish!----------------')
