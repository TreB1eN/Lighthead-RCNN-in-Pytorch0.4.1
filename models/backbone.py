import torch
from torch.nn import Sequential, MaxPool2d, Conv2d, BatchNorm2d, Module, Sigmoid, AdaptiveAvgPool2d, ReLU, init
import torch.nn.functional as F
from models.model_utils import set_bn_eval, separate_bn_paras
import torch

class ResNet101Extractor(Module):

    '''ResNet101 Extractor for LightHeadRCNN ResNet101 implementation.

    This class is used as an extractor for LightHeadRCNNResNet101.
    This outputs feature maps.
    Dilated convolution is used in the C5 stage.'''

    def __init__(self, path = 'models/lighthead-rcnn-extractor-pretrained.pth'):
        super(ResNet101Extractor, self).__init__()
        kwargs = {
            'stride_first': True,
            'bn_kwargs' : {
                'eps' : 2e-05
            }
        }

        self.conv1 = Conv2DBNActiv(3, 64, 7, 2, 3, nobias=True, bn_kwargs= {'eps' : 2e-05})
        self.pool1 = MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.res2 = Chainer_ResBlock(3, 64, 64, 256, 1, **kwargs)
        self.res3 = Chainer_ResBlock(4, 256, 128, 512, 2, **kwargs)
        self.res4 = Chainer_ResBlock(23, 512, 256, 1024, 2, **kwargs)
        self.res5 = Chainer_ResBlock(3, 1024, 512, 2048, 1, 2, **kwargs)
            
        
        
        if path:
            self.load_state_dict(torch.load(path))
            print('loaded model from {}'.format(path))
        else:
            for m in self.modules():
                if isinstance(m, Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
        
        for para in self.conv1.parameters():
            para.requires_grad = False
        print('extractor conv1 freezed')
        for para in self.res2.parameters():
            para.requires_grad = False
        print('extractor res2 freezed')
        
        self.paras_only_bn, self.paras_wo_bn = separate_bn_paras(self)
        for para in self.paras_only_bn:
            para.requires_grad = False
        print('all bn layer freezed')
        
        self.apply(set_bn_eval)
        print('set all bn to eval mode')
    
    def set_bn_eval(self):
        self.apply(set_bn_eval)

    def forward(self, x):
        h = self.pool1(self.conv1(x))
        h = self.res2(h)
        h = self.res3(h)
        res4 = self.res4(h)
        res5 = self.res5(res4)
        return res4, res5

class SEModule(Module):

    def __init__(self, channels, reduction = 16):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Conv2DBNActiv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=None,
                 stride=1,
                 pad=0,
                 dilate=1,
                 groups=1,
                 nobias=True,
                 activ=F.relu,
                 bn_kwargs={}):
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None
        self.activ = activ
        super(Conv2DBNActiv, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            bias=(not nobias),
            dilation=dilate,
            groups=groups)
        self.bn = BatchNorm2d(out_channels, **bn_kwargs)
        
    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.activ is None:
            return h
        else:
            return self.activ(h)

class Chainer_resnet_bottleneck(Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride=1,
                 dilate=1,
                 groups=1,
                 bn_kwargs={},
                 residual_conv=False,
                 stride_first=False,
                 add_seblock=False):
        if stride_first:
            first_stride = stride
            second_stride = 1
        else:
            first_stride = 1
            second_stride = stride
        super(Chainer_resnet_bottleneck, self).__init__()
        self.conv1 = Conv2DBNActiv(
            in_channels,
            mid_channels,
            1,
            first_stride,
            0,
            nobias=True,
            bn_kwargs=bn_kwargs)
        # pad = dilate
        self.conv2 = Conv2DBNActiv(
            mid_channels,
            mid_channels,
            3,
            second_stride,
            dilate,
            dilate,
            groups,
            nobias=True,
            bn_kwargs=bn_kwargs)
        self.conv3 = Conv2DBNActiv(
            mid_channels,
            out_channels,
            1,
            1,
            0,
            nobias=True,
            activ=None,
            bn_kwargs=bn_kwargs)
        if add_seblock:
            self.se = SEModule(out_channels)
        if residual_conv:
            self.residual_conv = Conv2DBNActiv(
                in_channels,
                out_channels,
                1,
                stride,
                0,
                nobias=True,
                activ=None,
                bn_kwargs=bn_kwargs)
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        
        if hasattr(self, 'se'):
            h = self.se(h)

        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(x)
        else:
            residual = x
        h += residual
        h = F.relu(h)
        return h

class Chainer_ResBlock(Module):
    def __init__(self,
                 n_layer,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 dilate=1,
                 groups=1,
                 bn_kwargs={},
                 stride_first=False,
                 add_seblock=False):
        super(Chainer_ResBlock, self).__init__()
        # Dilate option is applied to all bottlenecks.
        self.a = Chainer_resnet_bottleneck(
            in_channels,
            mid_channels,
            out_channels,
            stride,
            dilate,
            groups,
            bn_kwargs=bn_kwargs,
            residual_conv=True,
            stride_first=stride_first,
            add_seblock=add_seblock)
        blocks = []
        for i in range(n_layer - 1):
            blocks.append(
                Chainer_resnet_bottleneck(
                    out_channels,
                    mid_channels,
                    out_channels,
                    stride=1,
                    dilate=dilate,
                    bn_kwargs=bn_kwargs,
                    residual_conv=False,
                    add_seblock=add_seblock,
                    groups=groups))
        self.layers = Sequential(*blocks)
    
    def forward(self, x):
        return self.layers(self.a(x))