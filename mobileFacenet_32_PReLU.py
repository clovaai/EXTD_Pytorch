import torch.nn as nn
import math
import torch
import torch.nn.functional as F


def conv_bn(inp, oup, stride, k_size=3):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU()
    )

class DWC(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DWC, self).__init__()
    #self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(7,6),
                               #stride=1, padding=0, groups=in_channels, bias=False)
    self.batch_norm_in = nn.BatchNorm2d(in_channels)
    self.depthwise = nn.AvgPool2d((7, 6), stride=1, padding=0)
    self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)

  def forward(self, x):
    x = self.depthwise(x)
    #x = self.batch_norm_in(x)
    x = self.pointwise(x)
    return x

class Max_AvgPool(nn.Module):
    def __init__(self, kernel_size=(3,3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x) # add some channelwise gating?
        return x

class Max_AvgPool(nn.Module):
    def __init__(self, kernel_size=(3,3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x) # add some channelwise gating?
        return x

class gated_conv1x1(nn.Module):
    def __init__(self, inc=128, outc=128):
        super(gated_conv1x1, self).__init__()
        self.inp = int(inc/2)
        self.oup = int(outc/2)
        self.conv1x1_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)
        self.conv1x1_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)

    def forward(self, x):
        x_1 = x[:, :self.inp, :, :]
        x_2 = x[:, self.inp:, :, :]

        a_1 = self.conv1x1_1(x_1)
        g_1 = F.sigmoid(self.gate_1(x_1))

        a_2 = self.conv1x1_2(x_2)
        g_2 = F.sigmoid(self.gate_2(x_2))

        ret = torch.cat((a_1*g_1, a_2*g_2), 1)

        return ret


class InvertedResidual_dwc(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_dwc, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = []

        if expand_ratio == 1:
            self.conv.append(nn.Conv2d(inp, hidden_dim, kernel_size=(3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            #self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride, padding=1))
            #self.conv.append(gated_conv1x1(inc=hidden_dim,outc=oup))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
            self.conv.append(nn.PReLU())
        else:
            #self.conv.append(gated_conv1x1(inc=inp,outc=hidden_dim))
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            #self.conv.append(gated_conv1x1(inc=hidden_dim,outc=oup))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
            self.conv.append(nn.PReLU())

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = []

        if expand_ratio == 1:

            self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride, padding=1))
            #self.conv.append(gated_conv1x1(inc=hidden_dim,outc=oup))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        else:
            #self.conv.append(gated_conv1x1(inc=inp,outc=hidden_dim))
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride, padding=1))
            #self.conv.append(gated_conv1x1(inc=hidden_dim,outc=oup))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module): #mobileNet v2
    def __init__(self, embedding_size=128, input_size=224, width_mult=1.):
        super(Net, self).__init__()
        block = InvertedResidual
        block_dwc = InvertedResidual_dwc
        input_channel = 64
        last_channel = 256
        interverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 1],  # depthwise conv for first row
            [2, 32, 2, 1],
            [4, 32, 2, 1],
            [2, 32, 2, 2],
            [4, 32, 5, 1],
            [2, 32, 2, 2],
            [2, 32, 6, 2],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        # building inverted residual
        cnt = 0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if cnt>1:
                    if i == n - 1:  # reduce the featuremap in the last.
                        self.features.append(block_dwc(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_dwc(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
                else:
                    if i == n - 1:  # reduce the featuremap in the last.
                        self.features.append(block_dwc(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_dwc(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel

            cnt+=1

        # building last several layers
        self.features.append(gated_conv1x1(input_channel, self.last_channel))

        # make it nn.Sequential
        self.features_sequential = nn.Sequential(*self.features)

        # Global depthwise conv
        #self.GDCconv = DWC(self.last_channel, embedding_size)


        self._initialize_weights()

    def forward(self, x):
        x = self.features_sequential(x).view(-1, 256*4)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()