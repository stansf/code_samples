import math
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import segmentation_models_pytorch as smp
import timm


def transposed_conv(in_channels, out_channels, stride=2):
    """Transposed conv with same padding."""
    kernel_size, padding = {
        2: (4, 1),
        4: (8, 2),
        16: (32, 8),
    }[stride]
    layer = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False)
    return layer


def update_input_channels(model, n_input_channels):
    # check for the model names
    if n_input_channels != 3:
        # get model tree
        model_tree = []
        # e.g. model.stem.conv1
        # e.g. model.conv1
        key, val = list(model._modules.items())[0]
        model_tree.append(str(key))
        while type(val) != torch.nn.modules.conv.Conv2d:
            for k, v in val.named_children():
                model_tree.append(str(k))
                val = v
                break

        # update the first conv layer
        new_first_conv = torch.nn.Conv2d(n_input_channels, val.out_channels,
                                         val.kernel_size, val.stride,
                                         val.padding, val.dilation, val.groups,
                                         val.bias, val.padding_mode)
        if len(model_tree) == 1:
            setattr(model, model_tree[0], new_first_conv)
        else:
            new_attr = getattr(model, model_tree[0])
            for i in range(1, len(model_tree) - 1):
                new_attr2 = getattr(new_attr, model_tree[i])
                new_attr = new_attr2

            setattr(new_attr, model_tree[-1], new_first_conv)

    return model


class PlanarSegHead(nn.Module):

    def __init__(self, bottleneck_channels, in_features=2048, num_classes=5,
                 in_features_clf3=None):
        super().__init__()
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm2d(in_features)
        self.fc_conv = nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, bias=False)

        if in_features_clf3 is None:
            in_features_clf3 = in_features // 2

        # self.clf1 = nn.Conv2d(in_features, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        # self.clf2 = nn.Conv2d(in_features, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        # self.clf3 = nn.Conv2d(in_features // 2, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.clf1 = nn.Conv2d(in_features, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.clf2 = nn.Conv2d(in_features, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.clf3 = nn.Conv2d(in_features_clf3, bottleneck_channels, kernel_size=1, stride=1, bias=False)

        self.dec1 = transposed_conv(bottleneck_channels, bottleneck_channels, stride=2)
        self.dec2 = transposed_conv(bottleneck_channels, bottleneck_channels, stride=2)
        self.dec3 = transposed_conv(bottleneck_channels, bottleneck_channels, stride=16)

        self.fc_stage2 = nn.Conv2d(bottleneck_channels, num_classes, kernel_size=1, stride=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *feats):
        e7, e6, e5 = feats

        x = self.drop1(e7)
        x = self.fc_conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop2(x)

        c = self.clf1(x)           # 5 x 5 x 5
        d6 = self.dec1(c)          # 5 x 10 x 10

        d6_b = self.clf2(e6)       # 5 x 10 x 10
        d5 = self.dec2(d6_b + d6)  # 5 x 20 x 20

        d5_b = self.clf3(e5)       # 5 x 20 x 20
        d0 = self.dec3(d5_b + d5)  # 5 x 320 x 320

        d = self.fc_stage2(d0)
        return d


class ResPlanarSeg(nn.Module):

    def __init__(self, num_classes: int, pretrained=True, backbone='resnet101',
                 input_ch: int = 3):
        super().__init__()
        BackBone = getattr(models, backbone)
        self.resnet = BackBone(pretrained=pretrained)

        layer = self.resnet.conv1
        # Creating new Conv2d layer
        layer = nn.Conv2d(
            in_channels=input_ch,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias
        )
        self.resnet.conv1 = layer

        self.planar_seg = PlanarSegHead(bottleneck_channels=37,
                                        in_features=self.resnet.fc.in_features,
                                        num_classes=num_classes)

    def forward(self, x):
        '''
        x: 3 x 320 x 320
        '''
        x = self.resnet.conv1(x)       # 64 x 160 x 160
        x = self.resnet.bn1(x)
        e1 = self.resnet.relu(x)
        e2 = self.resnet.maxpool(e1)   # 64 x 80 x 80
        e3 = self.resnet.layer1(e2)    # 256 x 80 x 80
        e4 = self.resnet.layer2(e3)    # 512 x 40 x 40
        e5 = self.resnet.layer3(e4)    # 1024  x 20 x 20
        e6 = self.resnet.layer4(e5)    # 2048 x 10 x 10
        e7 = self.resnet.maxpool(e6)   # 2048 x 5 x 5

        return self.planar_seg(e7, e6, e5)


class ResPlanarSegTimm(nn.Module):

    def __init__(self,
                 num_classes: int,
                 backbone='lambda_resnet50ts',
                 pretrained: bool = False,
                 input_ch: int = 3):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          features_only=True)
        if input_ch != 3:
            print(f'Input channels: {input_ch}. Replace the first layer.')
            self.backbone = update_input_channels(self.backbone, input_ch)
        in_feat_clf3, in_feat = self.backbone.feature_info.channels()[-2:]

        self.planar_seg = PlanarSegHead(37, in_features=in_feat,
                                        in_features_clf3=in_feat_clf3,
                                        num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)[-2:]
        e6, e5 = features[::-1]
        e7 = self.maxpool(e6)
        out = self.planar_seg(e7, e6, e5)
        return out


class ResPlannerSegSMP(nn.Module):
    def __init__(self,
                 num_classes: int,
                 arch: str = 'unet',
                 backbone: str = 'resnet34',
                 pretrained: bool = False,
                 input_ch: int = 3):
        super().__init__()
        weights = 'imagenet' if pretrained else None
        self.model = smp.create_model(arch, backbone,
                                      encoder_weights=weights,
                                      in_channels=input_ch,
                                      classes=num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # model = timm.create_model('lambda_resnet50ts', features_only=True)
    input_ch = 3
    i = torch.rand((2, input_ch, 256, 256))
    model = ResPlanarSegTimm(5, backbone='efficientnet_b3', input_ch=input_ch)
    # model = ResPlannerSegSMP(input_ch=input_ch)
    out = model(i)
    print(out.shape)
    print(out[0, :, :2, :2])
