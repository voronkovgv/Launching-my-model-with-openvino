import cv2
import numpy as np
import openvino
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
from openvino.runtime import Core
import torch
from torch import nn
from openvino.offline_transformations import serialize



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


def conv_block(in_channels, out_channels, pool=False):
    layers = [SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallMyNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = conv_block(in_channels, 16)
        self.conv2 = conv_block(16, 32, pool=True)
        self.res1 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))

        self.transition1 = SqueezeExcitation(32, 16)

        self.conv3 = conv_block(32, 64, pool=True)
        self.conv4 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.transition2 = SqueezeExcitation(64, 32)

        self.conv5 = conv_block(64, 128, pool=True)
        self.conv6 = conv_block(128, 128, pool=True)
        self.res3 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(128, num_classes)
                                        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out

        out = self.transition1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out

        out = self.transition2(out)

        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out

        out = self.classifier(out)
        return out


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    MyModel = torch.load("EuroSAT_SmallMyNet_val_acc=0,936_test_acc=0,928",
                         map_location=torch.device('cpu'))
    MyModel.eval()
    # batch = torch.randn(1, 3, 128, 128)
    # torch.onnx.export(MyModel,
    #                   batch,
    #                   "SmallMyNet.onnx",
    #                   export_params=True,
    #                   opset_version=11)


    ie = Core()
    onnx_model = ie.read_model(model="SmallMyNet.onnx")
    compiled_onnx_model = ie.compile_model(model=onnx_model, device_name="CPU")
    input_layer = compiled_onnx_model.input(0)
    output_layer = compiled_onnx_model.output(0)

    # Construct the command for Model Optimizer.

    serialize(model=onnx_model, model_path="exported_onnx_model.xml",
              weights_path="exported_onnx_model.bin")

    input_layer = compiled_onnx_model.input(0)
    output_layer = compiled_onnx_model.output(0)
    batch = torch.randn(1, 3, 128, 128)
    result = compiled_onnx_model([batch])[output_layer]

    ir_model = ie.read_model("exported_onnx_model.xml", "exported_onnx_model.bin")
    compiled_model = ie.compile_model(ir_model, device_name="CPU")
    ir_input_layer = compiled_model.input(0)
    ir_output_layer = compiled_model.output(0)
    result = compiled_model([batch])[ir_output_layer]

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
