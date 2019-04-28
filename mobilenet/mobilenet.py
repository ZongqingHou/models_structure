import torch
from tensorboardX import SummaryWriter
from torch import nn

def init_model():
    pass

class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides_dw, strides_pw=1, padding_dw=1, padding_pw=0):
        super(MobileBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=strides_dw, padding=padding_dw, groups=in_channels),
            nn.BatchNorm2d(in_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides_pw, padding=padding_pw),
            nn.BatchNorm2d(in_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_data):
        return self.block(input_data)

class MobileNet(nn.Module):
    def __init__(self, classes, alpha=1):
        super(MobileNet, self).__init__()

        tmp_module_list = nn.Sequential()

        for tmp in range(5):
            tmp_module_list.add_module('module_list_%s' %tmp, MobileBlock(in_channels=512 * alpha, out_channels=512 * alpha, strides_dw=1))
        
        self.base_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32 * alpha, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32 * alpha),
            nn.ReLU(inplace=True),
            MobileBlock(in_channels=32 * alpha, out_channels=64 * alpha, strides_dw=1),
            MobileBlock(in_channels=64 * alpha, out_channels=128 * alpha, strides_dw=2),
            MobileBlock(in_channels=128 * alpha, out_channels=128 * alpha, strides_dw=1),
            MobileBlock(in_channels=128 * alpha, out_channels=256 * alpha, strides_dw=2),
            MobileBlock(in_channels=256 * alpha, out_channels=256 * alpha, strides_dw=1),
            MobileBlock(in_channels=256 * alpha, out_channels=512 * alpha, strides_dw=2),

            tmp_module_list,

            MobileBlock(in_channels=512 * alpha, out_channels=1024 * alpha, strides_dw=2),
            MobileBlock(in_channels=1024 * alpha, out_channels=1024 * alpha, strides_dw=1),
            nn.AvgPool2d(kernel_size=7)
        )
        
        self.linear = nn.Linear(in_features=1024 * alpha, out_features=classes)

    def forward(self, input_data, flag=True):
        output = self.base_model(input_data)
        output = output.view(output.size(0), output.size(1))

        if flag:
            output = self.linear(output)
        else:
            pass

        return output

if __name__ == '__main__':
    import time
    tmp_module = MobileNet()
    tmp_data = torch.randn(1,3,224,224)

    with SummaryWriter(comment='MobileNet') as visual:
        visual.add_graph(tmp_module, (tmp_data, ))
