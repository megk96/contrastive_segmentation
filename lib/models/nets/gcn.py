from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.projection import ProjectionHead

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# many are borrowed from https://github.com/ycszen/pytorch-ss/blob/master/gcn.py
class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Module):
    def __init__(self, configer):
        super(GCN, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        resnet = models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(2048, self.num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(1024, self.num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(512, self.num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(256, self.num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(self.num_classes)
        self.brm2 = _BoundaryRefineModule(self.num_classes)
        self.brm3 = _BoundaryRefineModule(self.num_classes)
        self.brm4 = _BoundaryRefineModule(self.num_classes)
        self.brm5 = _BoundaryRefineModule(self.num_classes)
        self.brm6 = _BoundaryRefineModule(self.num_classes)
        self.brm7 = _BoundaryRefineModule(self.num_classes)
        self.brm8 = _BoundaryRefineModule(self.num_classes)
        self.brm9 = _BoundaryRefineModule(self.num_classes)

        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
                           self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

    def forward(self, x):
        # if x: 512
        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        fs1 = self.brm5(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)  # 32
        fs2 = self.brm6(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)  # 64
        fs3 = self.brm7(F.upsample_bilinear(fs2, fm1.size()[2:]) + gcfm4)  # 128
        fs4 = self.brm8(F.upsample_bilinear(fs3, fm0.size()[2:]))  # 256
        out = self.brm9(F.upsample_bilinear(fs4, self.input_size))  # 512

        return out


class GCN_CONTRAST(nn.Module):
    def __init__(self, configer):
        super(GCN, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.proj_dim = self.configer.get('contrast', 'proj_dim')

        self.proj_head = ProjectionHead(dim_in=512, proj_dim=self.proj_dim)

        resnet = models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(2048, self.num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(1024, self.num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(512, self.num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(256, self.num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(self.num_classes)
        self.brm2 = _BoundaryRefineModule(self.num_classes)
        self.brm3 = _BoundaryRefineModule(self.num_classes)
        self.brm4 = _BoundaryRefineModule(self.num_classes)
        self.brm5 = _BoundaryRefineModule(self.num_classes)
        self.brm6 = _BoundaryRefineModule(self.num_classes)
        self.brm7 = _BoundaryRefineModule(self.num_classes)
        self.brm8 = _BoundaryRefineModule(self.num_classes)
        self.brm9 = _BoundaryRefineModule(self.num_classes)

        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
                           self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

    def forward(self, x):
        # if x: 512

        fm0 = self.layer0(x)  # 256
        print(fm0.shape)
        fm1 = self.layer1(fm0)  # 128
        print(fm1.shape)
        fm2 = self.layer2(fm1)  # 64
        print(fm2.shape)
        fm3 = self.layer3(fm2)  # 32
        print(fm3.shape)
        fm4 = self.layer4(fm3)  # 16
        print(fm4.shape)
        embeddings = self.proj_head(fm4)

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        print(gcfm1.shape)
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        print(gcfm2.shape)
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        print(gcfm3.shape)
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128
        print(gcfm4.shape)

        fs1 = self.brm5(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)  # 32
        print(fs1.shape)
        fs2 = self.brm6(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)  # 64
        print(fs2.shape)
        fs3 = self.brm7(F.upsample_bilinear(fs2, fm1.size()[2:]) + gcfm4)  # 128
        print(fs3.shape)
        fs4 = self.brm8(F.upsample_bilinear(fs3, fm0.size()[2:]))  # 256
        print(fs4.shape)
        out = self.brm9(F.upsample_bilinear(fs4, self.input_size))  # 512
        print(out.shape)
        print(embeddings.shape)

        return {'seg': out, 'embed': embeddings}
