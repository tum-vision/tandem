from typing import Optional
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from math import sqrt

debug = True


def _isfinite(x) -> bool:
    nans = torch.isnan(x)
    infs = torch.isinf(x)

    return not (nans.any() or infs.any())


def assert_isfinite(x, msg: str):
    if debug:
        nans = torch.isnan(x)
        infs = torch.isinf(x)

        if nans.any() or infs.any():
            msg += f"; #Nan = {nans.sum().item()},  #Inf = {infs.sum().item()}."

        assert not (nans.any() or infs.any()), msg


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1, channel, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", normalization=None, **kwargs):
        super(Conv2d, self).__init__()

        assert normalization is not None

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        if bn:
            if normalization == "batchnorm":
                self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
            elif normalization == "domainnorm":
                self.bn = DomainNorm(out_channels)
            elif normalization == "instancenorm":
                self.bn = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(f"Normalization {normalization} not implemented.")
        else:
            self.bn = None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(
            out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", normalization=None, **kwargs):
        super(Conv3d, self).__init__()
        assert normalization

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if isinstance(stride, int):
            assert stride in [1, 2]
        elif isinstance(stride, tuple):
            assert all(s in [1, 2] for s in stride)
        else:
            raise NotImplementedError(f"Stride {stride} not implemented")
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        if bn:
            if normalization == "batchnorm":
                self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            elif normalization == "instancenorm":
                self.bn = nn.InstanceNorm3d(out_channels, affine=True)
            else:
                raise NotImplementedError(f"Normalization {normalization} not implemented.")

        else:
            self.bn = None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", normalization=None, **kwargs):
        super(Deconv3d, self).__init__()
        assert normalization
        self.out_channels = out_channels
        if isinstance(stride, int):
            assert stride in [1, 2]
        elif isinstance(stride, tuple):
            assert all(s in [1, 2] for s in stride)
        else:
            raise NotImplementedError(f"Stride {stride} not implemented")
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        if bn:
            if normalization == "batchnorm":
                self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            elif normalization == "instancenorm":
                self.bn = nn.InstanceNorm3d(out_channels, affine=True)
            else:
                raise NotImplementedError(f"Normalization {normalization} not implemented.")

        else:
            self.bn = None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Upconv3d(nn.Module):
    """Applies a 3D up-convolution (optionally with batch normalization and relu activation)
           over an input signal composed of several input planes.

           Attributes:
               up (nn.Module): upsampling module, with nearest neighbour upsampling
               conv (nn.Module): convolution module
               bn (nn.Module): batch normalization module
               relu (bool): whether to activate by relu

           Notes:
               Default momentum for batch normalization is set to be 0.01,
               Up-convolution could reduce the checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/

           """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", normalization=None, **kwargs):
        super(Upconv3d, self).__init__()
        assert normalization
        # nn.Conv3d has no 'output_padding' parameter
        del kwargs['output_padding']
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(
            out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.up(x)
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class TinyResNet18(nn.Module):
    def __init__(self, pretrained=True, progress=False):
        super(TinyResNet18, self).__init__()

        resnet = torchvision.models.resnet18(pretrained=pretrained, progress=progress)

        attrs = ["conv1", "bn1", "relu", "maxpool", "layer1"]
        for attr in attrs:
            setattr(self, attr, getattr(resnet, attr))

    def forward(self, x):
        pass


class FeatureExtractorResNet18(nn.Module):
    def __init__(self, pretrained=True, progress=False):
        super(FeatureExtractorResNet18, self).__init__()
        self.resnet = TinyResNet18(pretrained=pretrained, progress=progress)
        self.resnet = self.resnet.eval()

        self.register_buffer("normalize_mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("normalize_std", torch.tensor([0.229, 0.224, 0.225]))

    def train(self, mode=True):
        super(FeatureExtractorResNet18, self).train(mode)

    def forward(self, x):
        # x (B, 3, H, W)
        assert x.ndim == 4
        # assert x.shape[-2] >= 224 and x.shape[-1] >= 224
        features = {}

        # Normalize
        x = (x - self.normalize_mean[None, :, None, None]) / self.normalize_std[None, :, None, None]
        features['normalize'] = x

        # Conv1
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        features['conv1'] = x

        # Max Pool
        x = self.resnet.maxpool(x)

        # Layers
        for l in range(1, 2):
            x = getattr(self.resnet, f"layer{l}")(x)
            features[f"layer{l}"] = x

        # # Avg Pool
        # x = self.resnet.avgpool(x)
        # x = torch.flatten(x, 1)

        # # FC
        # x = self.resnet.fc(x)

        return x, features

    def check(self, x):
        raise NotImplementedError()
        # normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalized = torch.stack([normalize_transform(xx) for xx in torch.unbind(x)])

        # out, features = self(x)
        # out_ref = self.resnet(normalized)

        # assert torch.allclose(features['normalize'], normalized)
        # assert torch.allclose(out, out_ref)
        # print("Max Error", torch.max(torch.abs(out - out_ref)))


class FeatureProjector(nn.Module):
    def __init__(self, base_channels: int):
        super(FeatureProjector, self).__init__()

        self.base_channels = base_channels
        self.conv = nn.ModuleDict({
            'stage1': nn.Conv2d(64, 4 * self.base_channels, 1, bias=False),
            'stage2': nn.Conv2d(64, 2 * self.base_channels, 5, padding=2, bias=False),
            'stage3': nn.Conv2d(64, 1 * self.base_channels, 9, padding=4, bias=False),
        })

    def forward(self, features):
        out = {}

        out['stage1'] = self.conv['stage1'](features['layer1'])  # B, 4*C, H//4, W//4
        out['stage2'] = self.conv['stage2'](
            F.interpolate(features['layer1'], scale_factor=2, mode="nearest") + features['conv1'])  # B, 2*C, H//2, W//2
        out['stage3'] = self.conv['stage3'](
            F.interpolate(features['layer1'], scale_factor=4, mode="nearest") + F.interpolate(features['conv1'],
                                                                                              scale_factor=2,
                                                                                              mode="nearest"))  # B, C, H, W

        return out


class FeatureNet(nn.Module):
    def __init__(self, base_channels: int, image_channels: int = 3, normalization: str = "batchnorm",
                 use_bn_skip: bool = False, last_stage: int = 3):
        """
        Feature extraction Network
        :param base_channels: int
        :param image_channels: int

        Only 3 stage FPN from before
        """
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels
        self.image_channels = image_channels

        self.stages = ('stage1', 'stage2', 'stage3')
        self.stages = self.stages[:last_stage]

        self.out_channels = {
            'stage1': 4 * self.base_channels,  # (H//4, W//4)
            'stage2': 2 * self.base_channels,  # (H//2, W//2)
            'stage3': 1 * self.base_channels,  # (H, W)
        }
        self.out_channels = {stage: self.out_channels[stage] for stage in self.stages}

        self.conv2d_kwargs = {'normalization': normalization}

        self.conv0 = nn.Sequential(
            Conv2d(self.image_channels, base_channels, 3, 1, padding=1, **self.conv2d_kwargs),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, **self.conv2d_kwargs),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, **self.conv2d_kwargs),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, **self.conv2d_kwargs),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, **self.conv2d_kwargs),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, **self.conv2d_kwargs),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, **self.conv2d_kwargs),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, **self.conv2d_kwargs),
        )

        final_channels = 4 * self.base_channels
        self.out = {}
        if 'stage1' in self.stages:
            self.out['stage1'] = nn.Conv2d(final_channels, self.out_channels['stage1'], 1, bias=False)
        if 'stage2' in self.stages:
            self.out['stage2'] = nn.Conv2d(final_channels, self.out_channels['stage2'], 3, padding=1, bias=False)
        if 'stage3' in self.stages:
            self.out['stage3'] = nn.Conv2d(final_channels, self.out_channels['stage3'], 3, padding=1, bias=False)
        self.out = nn.ModuleDict(self.out)

        self.skip = {}
        if 'stage2' in self.stages:
            self.skip['stage2'] = nn.Conv2d(2 * base_channels, final_channels, 1, bias=True)
        if 'stage3' in self.stages:
            self.skip['stage3'] = nn.Conv2d(1 * base_channels, final_channels, 1, bias=True)

        self.skip = nn.ModuleDict(self.skip)

    def forward(self, x):
        """
        :param x: (B, self.image_channels, H, W)
        :return: dict {
            stage1: (B, 4 * self.base_channels, H//4, W//4)
            stage2: (B, 2 * self.base_channels, H//2, W//2)
            stage3: (B, 1 * self.base_channels, H   , W   )
        }
        """
        # Down-sampling pass
        conv_stage3 = self.conv0(x)  # (B, base, H, W)
        conv_stage2 = self.conv1(conv_stage3)  # (B, 2*base, H//2, W//2)
        conv_stage1 = self.conv2(conv_stage2)  # (B, 4*base, H//4, W//4)

        # # Up-sampling + Skip connections pass
        # inter_stage2 = F.interpolate(conv_stage1, scale_factor=2, mode="nearest") + self.skip['stage2'](conv_stage2)
        # inter_stage3 = F.interpolate(inter_stage2, scale_factor=2, mode="nearest") + self.skip['stage3'](conv_stage3)

        # return {
        #     'stage1': self.out['stage1'](conv_stage1),
        #     'stage2': self.out['stage2'](inter_stage2),
        #     'stage3': self.out['stage3'](inter_stage3)
        # }

        res = {'stage1': self.out['stage1'](conv_stage1)}

        if 'stage2' in self.stages:
            inter_stage2 = F.interpolate(conv_stage1, scale_factor=2, mode="nearest") + self.skip['stage2'](conv_stage2)
            res['stage2'] = self.out['stage2'](inter_stage2)

            if 'stage3' in self.stages:
                inter_stage3 = F.interpolate(inter_stage2, scale_factor=2, mode="nearest") + self.skip['stage3'](
                    conv_stage3)
                res['stage3'] = self.out['stage3'](inter_stage3)

        return res


class CostRegNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, normalization: str = "batchnorm",
                 has_four_depths: bool = False):
        super(CostRegNet, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.upconv = Deconv3d

        self.conv3d_kwargs = {
            'normalization': normalization
        }

        self.conv0 = Conv3d(in_channels, base_channels, padding=1, **self.conv3d_kwargs)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1, **self.conv3d_kwargs)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1, **self.conv3d_kwargs)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1, **self.conv3d_kwargs)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1, **self.conv3d_kwargs)

        if has_four_depths:
            self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=(1, 2, 2), padding=1, **self.conv3d_kwargs)
        else:
            self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1, **self.conv3d_kwargs)

        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1, **self.conv3d_kwargs)

        if has_four_depths:
            self.conv7 = self.upconv(
                base_channels * 8, base_channels * 4, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1),
                **self.conv3d_kwargs)
        else:
            self.conv7 = self.upconv(
                base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1, **self.conv3d_kwargs)

        self.conv9 = self.upconv(
            base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1, **self.conv3d_kwargs)

        self.conv11 = self.upconv(
            base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1, **self.conv3d_kwargs)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x, img_features=None):
        """
        :param x: (B, self.in_channels, D, H, W)
        :param img_features: (B, self.in_channels, H, W)
        :return:  (B, 1, D, H, W) logits
        """
        B, _, D, H, W = list(x.shape)

        conv0 = self.conv0(x)  # (B, self.base_channels, D, H, W)
        # (B, 2*self.base_channels, D//2, H//2, W//2)
        conv2 = self.conv2(self.conv1(conv0))
        # (B, 4*self.base_channels, D//4, H//4, W//4)
        conv4 = self.conv4(self.conv3(conv2))
        # (B, 8*self.base_channels, D//8, H//8, W//8)
        x = self.conv6(self.conv5(conv4))
        # (B, 4*self.base_channels, D//4, H//4, W//4)
        x = conv4 + self.conv7(x)
        # (B, 2*self.base_channels, D//2, H//2, W//2)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)  # (B, self.base_channels, D, H, W)

        x = self.prob(x)  # (B, 1, D, H, W)

        return x


class OccRegNet(nn.Module):
    def __init__(self, in_channels: int = 4, up_conv: bool = False):
        super(OccRegNet, self).__init__()
        self.in_channels = in_channels
        self.upconv = Deconv3d if not up_conv else Upconv3d

        self.conv0 = Conv3d(in_channels, in_channels, padding=1)
        self.conv1 = Conv3d(in_channels, in_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(in_channels * 2, in_channels * 2, padding=1)
        self.conv3 = Conv3d(in_channels, in_channels, padding=1)

        self.upconv1 = self.upconv(
            in_channels * 2, in_channels * 1, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        """
        :param x: (B, self.in_channels, D, H, W)
        :return:  (B, 1, D, H, W) logits
        """
        conv0 = self.conv0(x)  # (B, self.in_channels, D, H, W)
        # (B, 2*self.in_channels, D//2, H//2, W//2)
        conv2 = self.conv2(self.conv1(conv0))
        x = conv0 + self.upconv1(conv2)  # (B, self.in_channels, D, H, W)
        x = self.conv3(x)  # (B, self.in_channels, D, H, W)
        return x


class OccFusionNet(nn.Module):
    def __init__(self, feature_in_channels: int, occ_in_channels: int = 4):
        super(OccFusionNet, self).__init__()
        self.feature_in_channels = feature_in_channels
        self.occ_in_channels = occ_in_channels
        self.channels = self.feature_in_channels + self.occ_in_channels

        self.conv = nn.Sequential(
            Conv3d(self.channels, self.channels, kernel_size=1, padding=0),
            Conv3d(self.channels, self.channels, kernel_size=1, padding=0),
            Conv3d(self.channels, self.channels, kernel_size=3, padding=1),
            Conv3d(self.channels, self.feature_in_channels,
                   kernel_size=1, padding=0),
            Conv3d(self.feature_in_channels,
                   self.feature_in_channels, kernel_size=1, padding=0)
        )

    def forward(self, features, occ):
        """
        :param features: (B, self.feature_in_channels, D, H, W)
        :param occ: (B, self.occ_in_channels, D, H, W)
        :return:  (B, self.feature_in_channels, D, H, W) fused
        """
        x = torch.cat((features, occ), 1)  # (B, self.channels, D, H, W)
        x = self.conv(x)  # (B, self.feature_in_channels, D, H, W)
        x = features + x  # (B, self.feature_in_channels, D, H, W)
        return x


class ColorTransformer(nn.Module):
    def __init__(self):
        super(ColorTransformer, self).__init__()

        self.register_buffer('t_rgb2lms', torch.tensor([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ]))

        self.register_buffer('t_lms2rgb', torch.tensor([
            [4.4679, -3.5873, 0.1193],
            [-1.2186, 2.3809, -0.1624],
            [0.0497, -0.2439, 1.2045]
        ]))

        self.register_buffer('t_log_lms2lab', torch.matmul(
            torch.diag(torch.tensor([1 / sqrt(3.), 1 / sqrt(6.), 1 / sqrt(2.)])),
            torch.tensor([
                [1, 1, 1.],
                [1, 1, -2],
                [1, -1, 0]
            ])
        ))

        self.register_buffer('t_lab2log_lms', torch.matmul(
            torch.tensor([
                [1, 1, 1.],
                [1, 1, -1],
                [1, -2, 0]
            ]),
            torch.diag(torch.tensor([sqrt(3) / 3, sqrt(6) / 6, sqrt(2) / 2])),
        ))

        self.n = 0
        self.register_buffer('running_mean', torch.zeros(3))
        self.register_buffer('running_std', torch.zeros(3))

        if debug:
            self._check()

    def _check(self):
        eye = torch.eye(3, dtype=self.t_lms2rgb.dtype, device=self.t_lms2rgb.device)
        assert torch.all(torch.abs(torch.sum(self.t_rgb2lms, 1) - 1) < 0.01)
        assert torch.all(torch.abs(torch.matmul(self.t_lms2rgb, self.t_rgb2lms) - eye) < 0.01)
        assert torch.all(torch.abs(torch.matmul(self.t_lab2log_lms, self.t_log_lms2lab) - eye) < 0.01)

    @staticmethod
    def apply_trafo(trafo, images):
        """
        trafo: (C_out, C_in)
        images: (B, C_in, H, W)
        """
        images = images.permute(0, 2, 3, 1)  # (B, H, W, C_in)
        out = torch.matmul(trafo[None, None], images[..., None])[..., 0]  # (B, H, W, C_out)
        return out.permute(0, 3, 1, 2)  # (B, C_out, H, W)

    def rgb2log_lms(self, rgb, clip=True):
        if clip:
            rgb = torch.clamp(rgb, min=0, max=1)
        lms = torch.clamp(self.apply_trafo(self.t_rgb2lms, rgb), min=0.001, max=None)
        log_lms = torch.log10(lms)
        return log_lms

    def log_lms2lab(self, log_lms):
        return self.apply_trafo(self.t_log_lms2lab, log_lms)

    def lab2log_lms(self, lab):
        return self.apply_trafo(self.t_lab2log_lms, lab)

    def log_lms2rgb(self, log_lms, clip=True):
        lms = torch.pow(10, log_lms)
        rgb = self.apply_trafo(self.t_lms2rgb, lms)
        if clip:
            rgb = torch.clamp(rgb, min=0, max=1)
        return rgb

    def rgb2lab(self, rgb):
        return self.log_lms2lab(self.rgb2log_lms(rgb))

    def lab2rgb(self, lab):
        return self.log_lms2rgb(self.lab2log_lms(lab))

    def forward(self, rgb):
        """
        rgb: (B, C, H, W)
        """

        lab = self.rgb2lab(rgb)  # (B, C, H, W)

        mean = torch.mean(lab, [0, 2, 3])
        std = torch.std(lab, [0, 2, 3])

        if self.training:
            self.n += 1
            self.running_mean += (mean - self.running_mean) / self.n
            self.running_std += (std - self.running_std) / self.n

        if self.n > 0:
            lab = (lab - mean[None, :, None, None]) / std[None, :, None, None]  # (B, C, H, W)
            lab = self.running_std[None, :, None, None] * lab + self.running_mean[None, :, None, None]  # (B, C, H, W)

        return self.lab2rgb(lab)


def homo_warping(src_features, ref_depth, *,
                 src_intrinsics, src_cam_to_world,
                 ref_intrinsics, ref_cam_to_world,
                 half_pixel_centers: bool,
                 min_depth_thres: float = 0.001,
                 view_aggregation_baseline_angle: bool = False):
    """
    :param src_features: (B, C, H, W)
    :param ref_depth: (B, D, H, W)
    :param src_intrinsics: (B, 3, 3)
    :param src_cam_to_world: (B, 4, 4)
    :param ref_intrinsics: (B, 3, 3)
    :param ref_cam_to_world: (B, 4, 4)
    :param half_pixel_centers: Pixel coordinate convention of the intrinsics.
    :param min_depth_thres:
    :return: (B, C, D, H, W) warped features, (B, C, D, H, W) mask valid (type as warped features)
    """

    # Comment regarding align_corners, half_pixel_centers, intrinsics
    # a) The indices are 0, 1, ..., w-1
    # b) If the extrinsics and intrinsics cancel, the indices in src are 0, 1, ..., w-1
    # c) The normalized pixels are i_norm = 2 * i / (w -1) - 1 = -1, ..., 1
    # d) We have to consider the centers of the pixels, so we *need* align_corners=True
    # e) align_corners = True was the default until torch 1.2.0 and the required version was 1.1.0

    assert half_pixel_centers is False, "Only implemented for half_pixel_centers=False"

    batch_size, depth_num, height, width = list(ref_depth.size())

    with torch.no_grad():
        # Coordinate Transformation
        ref_world_to_cam = torch.inverse(ref_cam_to_world)  # (B, 4, 4)
        src_world_to_cam = torch.inverse(src_cam_to_world)  # (B, 4, 4)

        ref_world_to_pixel = torch.clone(ref_world_to_cam)  # (B, 4, 4)
        ref_world_to_pixel[:, :3, :4] = torch.matmul(
            ref_intrinsics, ref_world_to_cam[:, :3, :4])  # (B, 4, 4)
        ref_pixel_to_world = torch.inverse(ref_world_to_pixel)

        src_world_to_pixel = torch.clone(src_world_to_cam)  # (B, 4, 4)
        src_world_to_pixel[:, :3, :4] = torch.matmul(
            src_intrinsics, src_world_to_cam[:, :3, :4])  # (B, 4, 4)
        ref_pixel_to_src_pixel = torch.matmul(
            src_world_to_pixel, ref_pixel_to_world)  # (B, 4, 4)
        rot = ref_pixel_to_src_pixel[:, :3, :3]  # (B, 3, 3)
        trans = ref_pixel_to_src_pixel[:, :3, 3:4]  # (B, 3, 1)

        # Pixel sampling
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_features.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_features.device)])  # (H, W)
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # (3, H*W)
        xyz = torch.unsqueeze(xyz, 0).expand(batch_size, -1, -1)  # (B, 3, H*W)

        # Warping
        rot_xyz = torch.matmul(rot, xyz)  # (B, 3, H*W)
        rot_depth_xyz = rot_xyz.unsqueeze(
            2) * ref_depth.view(batch_size, 1, depth_num, -1)  # (B, 3, D, H*W)
        proj_xyz = rot_depth_xyz + \
                   trans.view(batch_size, 3, 1, 1)  # (B, 3, D, H*W)
        proj_xy = proj_xyz[:, :2, :, :] / \
                  proj_xyz[:, 2:3, :, :]  # (B, 2, D, H*W)
        proj_x_normalized = proj_xy[:, 0, :, :] / \
                            (0.5 * (width - 1)) - 1  # (B, D, H*W)
        proj_y_normalized = proj_xy[:, 1, :, :] / \
                            (0.5 * (height - 1)) - 1  # (B, D, H*W)
        # (B, D, H*W, 2)
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)

        if view_aggregation_baseline_angle:
            src_to_ref = torch.matmul(ref_world_to_cam, src_cam_to_world)  # (B, 4, 4)
            origin_src_in_ref = src_to_ref[:, :3, 3:4]  # (B, 3, 1)

            xyz_intr = torch.matmul(torch.inverse(ref_intrinsics), xyz)

            xyz_ref_depth = xyz_intr.unsqueeze(2) * ref_depth.view(batch_size, 1, depth_num, -1)  # (B, 3, D, H*W)

            vec_xyz_to_ref = 0 - xyz_ref_depth
            vec_xyz_to_src = origin_src_in_ref.unsqueeze(-1) - xyz_ref_depth  # (B, 3, D, H*W)

            calc_depth_ref = torch.norm(vec_xyz_to_ref, p=2, dim=1)  # (B, D, H*W)
            calc_depth_src = torch.norm(vec_xyz_to_src, p=2, dim=1)  # (B, D, H*W)

            vec_xyz_to_ref = F.normalize(vec_xyz_to_ref, p=2, dim=1)  # (B, 3, D, H*W)
            vec_xyz_to_src = F.normalize(vec_xyz_to_src, p=2, dim=1)  # (B, 3, D, H*W)
            cos_baseline_angle = torch.sum(vec_xyz_to_ref * vec_xyz_to_src, dim=1)  # (B, D, H*W)
            cos_baseline_angle = torch.reshape(cos_baseline_angle,
                                               (batch_size, 1, depth_num, height, width))  # (B, 1, D, H, W)

            calc_depth_ref = torch.reshape(calc_depth_ref, (batch_size, 1, depth_num, height, width))  # (B, 1, D, H, W)
            calc_depth_src = torch.reshape(calc_depth_src, (batch_size, 1, depth_num, height, width))  # (B, 1, D, H, W)

            # TODO: Replace 10.0 by max depth
            view_aggregation_features = torch.cat([cos_baseline_angle, calc_depth_ref / 10.0, calc_depth_src / 10.0],
                                                  dim=1)  # (B, 1, D, H, W)

        mask_negative_depth = proj_xyz[:, 2:3,
                              :, :] < min_depth_thres  # (B, D, H*W)

        mode = 'bilinear'
        # This makes only sense for bilinear
        x_bound = 1.0 + 1 / (width - 1)
        y_bound = 1.0 + 1 / (height - 1)
        mask_outside_image = (torch.abs(proj_x_normalized) > x_bound) | (
                torch.abs(proj_y_normalized) > y_bound)  # (B, D, H*W)

    try:
        warped_src_fea = F.grid_sample(src_features, grid.view(batch_size, depth_num * height, width, 2),
                                       mode=mode, padding_mode='zeros', align_corners=True)
    except TypeError:
        # noinspection PyUnresolvedReferences
        version_major, version_minor, _ = [
            int(x) for x in torch.__version__.split(".")]
        assert (version_major == 1 and version_minor <= 2) or (
                version_major == 0), "Align_corners behaviour version"
        warped_src_fea = F.grid_sample(src_features, grid.view(batch_size, depth_num * height, width, 2),
                                       mode=mode, padding_mode='zeros')  # (B, C, D*H, W)

    # Use 0 for points that could not be projected
    warped_src_fea = warped_src_fea.permute(0, 2, 3, 1)  # (B, D*H, W, C)
    mask_negative_depth = mask_negative_depth.view(
        batch_size, depth_num * height, width)  # (B, D*H, W)
    warped_src_fea[mask_negative_depth] = 0
    warped_src_fea = warped_src_fea.permute(0, 3, 1, 2)  # (B, D*H, W, C)

    # TODO: Hack to stop training from crashing
    warped_src_fea[torch.isnan(warped_src_fea)] = 0
    assert_isfinite(warped_src_fea, "Warped features should be finite")

    # Reshape for output
    warped_src_fea = warped_src_fea.view(
        batch_size, -1, depth_num, height, width)  # (B, C, D, H, W) warped features

    # Output mask
    mask_negative_depth = torch.reshape(
        mask_negative_depth, (batch_size, 1, depth_num, height, width))
    mask_outside_image = torch.reshape(
        mask_outside_image, (batch_size, 1, depth_num, height, width))
    mask_valid = ~(mask_negative_depth | mask_outside_image)  # (B, 1, D, H, W)

    if not view_aggregation_baseline_angle:
        return warped_src_fea, mask_valid.type_as(warped_src_fea)
    else:
        return warped_src_fea, mask_valid.type_as(warped_src_fea), view_aggregation_features


def homo_warping_3d(src_depth, ref_depth, *,
                    src_intrinsics, src_cam_to_world,
                    ref_intrinsics, ref_cam_to_world,
                    half_pixel_centers: bool,
                    min_depth_thres: float = 0.001):
    """
    :param src_depth: (B, H, W)
    :param ref_depth: (B, H, W)
    :param src_intrinsics: (B, 3, 3)
    :param src_cam_to_world: (B, 4, 4)
    :param ref_intrinsics: (B, 3, 3)
    :param ref_cam_to_world: (B, 4, 4)
    :param half_pixel_centers: Pixel coordinate convention of the intrinsics.
    :param min_depth_thres:
    :return: (B, C, D, H, W) warped features, (B, C, D, H, W) mask valid (type as warped features)

    ref -> src (get depth) -> ref
    """

    assert half_pixel_centers is False, "Only implemented for half_pixel_centers=False"

    ref_depth = ref_depth.unsqueeze(1)
    batch_size, depth_num, height, width = list(ref_depth.size())

    # Coordinate Transformation
    ref_world_to_cam = torch.inverse(ref_cam_to_world)  # (B, 4, 4)
    src_world_to_cam = torch.inverse(src_cam_to_world)  # (B, 4, 4)

    ref_world_to_pixel = torch.clone(ref_world_to_cam)  # (B, 4, 4)
    ref_world_to_pixel[:, :3, :4] = torch.matmul(
        ref_intrinsics, ref_world_to_cam[:, :3, :4])  # (B, 4, 4)
    ref_pixel_to_world = torch.inverse(ref_world_to_pixel)

    src_world_to_pixel = torch.clone(src_world_to_cam)  # (B, 4, 4)
    src_world_to_pixel[:, :3, :4] = torch.matmul(
        src_intrinsics, src_world_to_cam[:, :3, :4])  # (B, 4, 4)
    ref_pixel_to_src_pixel = torch.matmul(
        src_world_to_pixel, ref_pixel_to_world)  # (B, 4, 4)
    rot = ref_pixel_to_src_pixel[:, :3, :3]  # (B, 3, 3)
    trans = ref_pixel_to_src_pixel[:, :3, 3:4]  # (B, 3, 1)

    # Pixel sampling
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=src_depth.device)])  # (H, W)
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # (3, H*W)
    xyz = torch.unsqueeze(xyz, 0).expand(batch_size, -1, -1)  # (B, 3, H*W)

    # Warping
    rot_xyz = torch.matmul(rot, xyz)  # (B, 3, H*W)
    rot_depth_xyz = rot_xyz.unsqueeze(2) * ref_depth.view(batch_size, 1, depth_num, -1)  # (B, 3, D, H*W)
    proj_xyz = rot_depth_xyz + \
               trans.view(batch_size, 3, 1, 1)  # (B, 3, D, H*W)
    proj_xy = proj_xyz[:, :2, :, :] / \
              proj_xyz[:, 2:3, :, :]  # (B, 2, D, H*W)
    proj_x_normalized = proj_xy[:, 0, :, :] / \
                        (0.5 * (width - 1)) - 1  # (B, D, H*W)
    proj_y_normalized = proj_xy[:, 1, :, :] / \
                        (0.5 * (height - 1)) - 1  # (B, D, H*W)
    # (B, D, H*W, 2)
    grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)

    mask_negative_depth = proj_xyz[:, 2:3, :, :] < min_depth_thres  # (B, D, H*W)

    mode = 'bilinear'
    # This makes only sense for bilinear
    x_bound = 1.0 + 1 / (width - 1)
    y_bound = 1.0 + 1 / (height - 1)
    mask_outside_image = (torch.abs(proj_x_normalized) > x_bound) | (
            torch.abs(proj_y_normalized) > y_bound)  # (B, D, H*W)

    try:
        src_depth_at_pts = F.grid_sample(src_depth.unsqueeze(1), grid, mode=mode, padding_mode='zeros',
                                         align_corners=True).squeeze(2)  # (B, 1, H*W)
    except TypeError:
        # noinspection PyUnresolvedReferences
        version_major, version_minor, _ = [
            int(x) for x in torch.__version__.split(".")]
        assert (version_major == 1 and version_minor <= 2) or (
                version_major == 0), "Align_corners behaviour version"
        src_depth_at_pts = F.grid_sample(src_depth.unsqueeze(1), grid, mode=mode, padding_mode='zeros').squeeze(
            2)  # (B, 1, H*W)

    proj_xy1 = torch.cat([proj_xy, torch.ones_like(proj_xy[:, :1])], 1).squeeze(2)  # (B, 3, H*W)
    src_pixel_to_ref_pixel = torch.inverse(ref_pixel_to_src_pixel)  # (B, 4, 4)
    rot = src_pixel_to_ref_pixel[:, :3, :3]  # (B, 3, 3)
    trans = src_pixel_to_ref_pixel[:, :3, 3:4]  # (B, 3, 1)

    rot_xyz = torch.matmul(rot, proj_xy1)  # (B, 3, H*W)
    rot_depth_xyz = rot_xyz * src_depth_at_pts  # (B, 3, H*W)
    proj_xyz = rot_depth_xyz + trans  # (B, 3, H*W)
    proj_pixel = proj_xyz[:, :2] / proj_xyz[:, 2:]
    proj_depth = proj_xyz[:, 2]  # (B, H*W)

    mask_negative_depth = torch.logical_or(mask_negative_depth, proj_xyz[:, 2:] < min_depth_thres)
    mask = torch.logical_not(torch.logical_or(mask_negative_depth, mask_outside_image)).squeeze(1)

    proj_pixel = torch.reshape(proj_pixel.permute(0, 2, 1), [batch_size, height, width, 2])
    proj_depth = torch.reshape(proj_depth, [batch_size, height, width])
    mask = torch.reshape(mask.to(proj_pixel.dtype), [batch_size, height, width])

    return proj_pixel, proj_depth, mask


def ssim_pad_intrinsics(K):
    if torch.is_tensor(K):
        assert K.ndim == 3
        assert K.size(-1) == 3
        assert K.size(-2) == 3
        B = K.size(0)
        zeros = torch.zeros([B, 3, 1], device=K.device, dtype=K.dtype)
        row = torch.zeros([B, 1, 4], device=K.device, dtype=K.dtype)
        row[:, :, -1] = 1
        return torch.cat([torch.cat([K, zeros], -1), row], -2)
    else:
        return type(K)(ssim_pad_intrinsics(KK) for KK in K)


def depth_prediction(features, intrinsics, cam_to_world, depth_in, cost_regularization,
                     training: bool, half_pixel_centers: bool, volume_gate=None):
    """
    :param features: tuple (V) (B, C, H, W)
    :param intrinsics: (B, V, 3, 3) or (B, 3, 3)
    :param cam_to_world: (B, V, 4, 4)
    :param depth_in: (B, D, H, W)
    :param cost_regularization:
    :param training:
    :param half_pixel_centers:
    :param volume_gate: Whether to aggregate the views as in Yi et al.
        Pyramid Multi-view Stereo Net with Self-adaptive View Aggregation.
        Posssible choices are None a nn.Module
    :return: dict: {
        "depth": (B, H, W) predicted depth,
        "confidence": (B, H, W) sum over 4 adjacent probabilities
    }
    """
    batch_size = depth_in.size(0)
    depth_num = depth_in.size(1)
    height = depth_in.size(2)
    width = depth_in.size(3)

    cam_to_world = torch.unbind(cam_to_world, 1)  # (V) (B, 4, 4)
    if intrinsics.dim() == 4:
        intrinsics = torch.unbind(intrinsics, 1)  # (V) (B, 3, 3)
    else:
        intrinsics = (intrinsics,) * len(cam_to_world)  # (V) (B, 3, 3)

    view_num = len(features)

    # (B, C, H, W), (V-1) (B, C, H, W)
    ref_features, src_features_tuple = features[0], features[1:]
    # (B, 4, 4), (V-1) (B, 4, 4)
    ref_cam_to_world, src_cam_to_world_tuple = cam_to_world[0], cam_to_world[1:]
    # (B, 3, 3), (V-1) (B, 3, 3)
    ref_intrinsics, src_intrinsics_tuple = intrinsics[0], intrinsics[1:]

    # Reference only Cost Volume
    ref_volume = ref_features.unsqueeze(2).expand(-1, -1, depth_num, -1, -1)  # (B, C, D, H, W)

    if volume_gate is not None:
        warped_accumulated_volumes = 0.0
    else:
        volume_sum = ref_volume  # (B, C, D, H, W)
        volume_sq_sum = ref_volume ** 2  # (B, C, D, H, W)

    # Integrate Views
    for i, (src_features, src_intrinsics, src_cam_to_world) in enumerate(zip(
            src_features_tuple, src_intrinsics_tuple, src_cam_to_world_tuple)):

        warped_volume, mask_valid = homo_warping(
            src_features=src_features, ref_depth=depth_in,
            src_intrinsics=src_intrinsics, src_cam_to_world=src_cam_to_world,
            ref_intrinsics=ref_intrinsics, ref_cam_to_world=ref_cam_to_world,
            half_pixel_centers=half_pixel_centers
        )  # (B, C, D, H, W), (B, 1, D, H, W)

        circumvent_bug = True  # See comment below
        if volume_gate is not None:
            warped_volume_diff_squared = (warped_volume - ref_volume).pow_(2)  # (B, C, D, H, W)
            reweight = volume_gate(warped_volume_diff_squared)  # (B, 1, D, H, W)
            warped_accumulated_volumes += (reweight + 1) * warped_volume_diff_squared  # (B, C, D, H, W)

        elif training or circumvent_bug:
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        else:
            # Using model.eval() within torch.no_grad() this throws the following error:
            #  RuntimeError: add: unsupported operation:
            #  more than one element of the written-to tensor refers to a single memory location.
            #  Please clone() the tensor before calling add
            volume_sum += warped_volume
            # the memory of warped_volume has been modified
            volume_sq_sum += warped_volume.pow_(2)
        del warped_volume

    if volume_gate is not None:
        volume_variance = warped_accumulated_volumes.div_(view_num - 1)
    else:
        volume_variance = volume_sq_sum.div_(view_num).sub_(volume_sum.div_(view_num).pow_(2))  # (B, C, D, H, W)

    # Cost Volume Regularization
    cost_reg = cost_regularization(volume_variance, ref_features)  # (B, 1, D, H, W)
    log_prob_volume_pre = cost_reg.squeeze(1)  # (B, D, H, W)

    prob_volume = F.softmax(log_prob_volume_pre, dim=1)  # (B, D, H, W)
    depth_pred = take_expectation(
        probability=prob_volume, values=depth_in, dim=1)  # (B, H, W)

    with torch.no_grad():
        # confidence
        prob_volume_sum_four = 4 * F.avg_pool3d(
            F.pad(prob_volume.unsqueeze(1), pad=[0, 0, 0, 0, 1, 2]),
            kernel_size=(4, 1, 1), stride=1, padding=0).squeeze(1)  # (B, D, H, W)
        depth_index = take_expectation(
            probability=prob_volume,
            values=torch.arange(
                depth_num, device=prob_volume.device, dtype=torch.float32),
            dim=1
        ).long()  # (B, H, W)
        depth_index = depth_index.clamp(min=0, max=depth_num - 1)
        confidence = torch.gather(
            prob_volume_sum_four, 1, depth_index.unsqueeze(1)).squeeze(1)  # (B, H, W)

    if debug:
        assert_isfinite(depth_pred, "Depth should be finite")
        assert_isfinite(confidence, "Confidence should be finite")

    return {"depth": depth_pred, "confidence": confidence}


def take_expectation(probability, values, dim):
    """
    :param probability: (B, D, H, W)
    :param values:
        (B, D, H, W) or
        (B, X) with probability.size(dim) == values.size(dim) or
        (X, ) with probability.size(dim) = values.size(0)
    :param dim: int
    :return:
    """
    assert dim > 0
    if probability.dim() == values.dim():
        return torch.sum(probability * values, dim=dim)

    if values.dim() == 1:
        values = values.unsqueeze(0).expand(probability.size(0), -1)  # (B, X)

    values_size = list(values.size())
    assert len(
        values_size) == 2, f"Either same size or values.dim() == 2, but {probability.dim()} and {values.dim()}."

    values_view_size = [1] * probability.dim()
    values_view_size[0] = values_size[0]
    values_view_size[dim] = values_size[1]

    return torch.sum(probability * torch.reshape(values, values_view_size), dim=dim)


def smooth_l1_loss_base(input, target, beta):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss


def sl1_depth_loss_term(depth_est, depth_gt, mask, keep_batch=False, stage=None):
    # TODO: Consider a different cut off (beta) for the loss, but this is not implemented
    if stage is not None:
        cutoff = {'stage1': 10.0 / 48, 'stage2': 1e6, 'stage3': 1e6}[stage]
    else:
        cutoff = 1e6
    depth_loss = smooth_l1_loss_base(depth_est * mask, depth_gt * mask, beta=cutoff)  # (B, H_stage, W_stage)
    depth_loss = torch.mean(depth_loss, (1, 2)) / torch.mean(mask, (1, 2))  # (B, )

    if not keep_batch:
        depth_loss = torch.mean(depth_loss)

    return depth_loss


def berhu_loss_base(input, target, beta):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, n, 0.5 * (n * n + beta * beta) / beta)
    return loss


def berhu_depth_loss_term(depth_est, depth_gt, mask, keep_batch=False, stage=None):
    if stage is not None:
        cutoff = {'stage1': 10.0 / 48, 'stage2': 1e6, 'stage3': 1e6}[stage]
    else:
        cutoff = 1e6

    depth_loss = berhu_loss_base(depth_est * mask, depth_gt * mask, beta=cutoff)  # (B, H_stage, W_stage)
    depth_loss = torch.mean(depth_loss, (1, 2)) / torch.mean(mask, (1, 2))  # (B, )

    if not keep_batch:
        depth_loss = torch.mean(depth_loss)

    return depth_loss


def sl1_depth_loss_term(depth_est, depth_gt, mask, keep_batch=False, stage=None):
    # TODO: Consider a different cut off (beta) for the loss, but this is not implemented
    if stage is not None:
        cutoff = {'stage1': 10.0 / 48, 'stage2': 1e6, 'stage3': 1e6}[stage]
        print(f"Smooth L1 Loss Cutoff at {stage}: {cutoff}")
    else:
        cutoff = 1e6
    depth_loss = smooth_l1_loss_base(depth_est * mask, depth_gt * mask, beta=cutoff)  # (B, H_stage, W_stage)
    depth_loss = torch.mean(depth_loss, (1, 2)) / torch.mean(mask, (1, 2))  # (B, )

    if not keep_batch:
        depth_loss = torch.mean(depth_loss)

    return depth_loss


def l1_depth_loss_term(depth_est, depth_gt, mask, keep_batch=False, stage=None):
    depth_loss = F.l1_loss(depth_est * mask, depth_gt *
                           mask, reduction='none')  # (B, H_stage, W_stage)
    depth_loss = torch.mean(depth_loss, (1, 2)) / \
                 torch.mean(mask, (1, 2))  # (B, )

    if not keep_batch:
        depth_loss = torch.mean(depth_loss)

    return depth_loss


def abs_rel_loss_term(depth_est, depth_gt, mask, eps=0.01, stage=None):
    abs_rel = mask * torch.abs(depth_est - depth_gt) / (depth_gt + eps)
    abs_rel = torch.mean(abs_rel, (1, 2)) / torch.mean(mask, (1, 2))  # (B, )
    abs_rel = torch.mean(abs_rel)

    return abs_rel


def get_depth_grad_img(input, delta=1, scale_inv=True):
    x_shift = F.pad(input[:, :, delta:], pad=[0, delta], mode="constant")
    y_shift = F.pad(input[:, delta:, :], pad=[0, 0, 0, delta], mode="constant")
    grad_x = input - x_shift
    grad_y = input - y_shift
    if scale_inv:
        si_grad_x = grad_x / (torch.abs(grad_x) + torch.abs(grad_y) + 1e-6)
        si_grad_y = grad_y / (torch.abs(grad_x) + torch.abs(grad_y) + 1e-6)
        grad_x = si_grad_x
        grad_y = si_grad_y
    grad_img = torch.stack((grad_x, grad_y), dim=1)
    return grad_img


def grad_loss_term(depth_est, depth_gt, mask, stage=None):
    """
    Scale-invariant gradient loss as Eq.(12) in DeepTAM
    """
    grad_loss = 0.
    grad_deltas = [1, 2, 4]
    for h in grad_deltas:
        grad_gt = get_depth_grad_img(depth_gt, h).detach()
        grad_est = get_depth_grad_img(depth_est, h)
        # TODO: the mask of gradient shouldn't be the same as the original mask. The pixels left and down to the invalid points should not be counted as well.
        norm = (torch.norm(grad_est - grad_gt, dim=1)) * mask
        grad_loss += norm.mean() / mask.mean()
    grad_loss = grad_loss / len(grad_deltas)
    return grad_loss


def compute_loss(outputs: dict, batch: dict, weights: tuple, loss_terms: tuple, term_weights: tuple,
                 keep_batch=False):
    """
    :param outputs:
        Outputs from model.
    :param batch:
    :param weights:
    :return:
    """
    total_losses = {name: 0.0 for name in loss_terms}

    for i, stage in enumerate(('stage1', 'stage2', 'stage3')):
        depth_est = outputs[stage]['depth']  # (B, H_stage, W_stage)
        depth_gt = batch['depth'][stage]  # (B, H_stage, W_stage)
        if 'mask_total' in batch:
            # This masks includes if a point is visible from at least one other view
            mask = batch['mask_total'][stage]  # (B, H_stage, W_stage)
        else:
            # this mask just includes if we have a valid gt depth
            mask = batch['mask'][stage]  # (B, H_stage, W_stage)
        assert depth_gt.size() == mask.size()

        for ind, loss_name in enumerate(loss_terms):
            loss = globals()[f"{loss_name}_loss_term"](depth_est, depth_gt, mask, keep_batch=keep_batch,
                                                       stage=stage)
            total_losses[loss_name] += term_weights[ind] * weights[i] * loss
    total_loss = 0.0
    for loss_name in loss_terms:
        total_losses[loss_name] = total_losses[loss_name] / sum(weights)
        total_loss += total_losses[loss_name]
    losses = {k + "_loss": v for k, v in total_losses.items()}
    losses['total_loss'] = total_loss

    return total_loss, losses


def depth_filter_edges(depth: torch.Tensor, discard_percentage: torch.Tensor, window: int = 5,
                       num: Optional[int] = None, ) -> torch.Tensor:
    """
    :param depth: (B, H, W) depth
    :param window: int, window size. Must be odd.
    :param thres: Optional[float], the threshold to detect an edge. For None no thresholding will be applied
    :param discard_percentage: Optional[float], percentage of pixels to discard
    :return: edge either the edge value or the edge mask depending on thres
    """
    # depth (B, H, W)

    batch_size, height, width = list(depth.size())

    assert window % 2 == 1
    w = window
    w2 = window // 2
    m = (w * w) // 2

    if num is None:
        num = w * (w2 + 1)

    dw = F.unfold(depth.unsqueeze(1), kernel_size=(w, w), padding=w2, stride=1)  # (B, w*w, H*W)

    edge = torch.abs(dw - dw[:, m: m + 1, :])  # (B, w*w, H*W)
    edge, edge_index = torch.kthvalue(edge, k=num, dim=1)  # (B, H*W)

    edge_sorted, _ = torch.sort(edge, dim=1)  # (B, H*W)

    cutoff_index = (height * width * (100 - discard_percentage) / 100.0).to(torch.long)  # (B, )
    cutoff_index = torch.clamp(cutoff_index, 0, height * width - 1)  # (B, )

    row = torch.arange(edge_sorted.size(0), dtype=torch.long)

    thres = edge_sorted[(row, cutoff_index)]  # (B, )
    thres = thres[:, None, None]  # (B, 1, 1)

    edge = torch.reshape(edge, (batch_size, height, width))  # (B, H, W)
    mask = edge > thres

    depth[mask] = 0

    return depth, mask


def eval_errors(outputs: dict, batch: dict, keep_batch=False) -> dict:
    """
    :param outputs:
        Outputs from model.
    :param batch:
    :return:
    """
    errors = {}
    for stage in ('stage1', 'stage2', 'stage3'):
        # type: torch.FloatTensor # (B, H_stage, W_stage)
        depth_est = outputs[stage]['depth']
        # type: torch.FloatTensor # (B, H_stage, W_stage)
        depth_gt = batch['depth'][stage]
        # type: torch.FloatTensor # (B, H_stage, W_stage)
        mask = batch['mask'][stage]
        assert depth_gt.size() == mask.size()

        batch_size = depth_est.size(0)

        if not keep_batch:
            errors[stage] = OrderedDict([
                ('abs_rel', 0.0),
                ('abs', 0.0),
                ('sq_rel', 0.0),
                ('rmse', 0.0),
                ('rmse_log', 0.0),
                ('a1', 0.0),
                ('a2', 0.0),
                ('a3', 0.0),
                ('d1', 0.0),
                ('d2', 0.0),
                ('d3', 0.0),
            ])
        else:
            errors[stage] = OrderedDict([
                ('abs_rel', []),
                ('abs', []),
                ('sq_rel', []),
                ('rmse', []),
                ('rmse_log', []),
                ('a1', []),
                ('a2', []),
                ('a3', []),
                ('d1', []),
                ('d2', []),
                ('d3', []),
            ])

        for gt, est, m in zip(depth_gt, depth_est, mask):
            gt = gt[m > 0.5]
            est = est[m > 0.5]
            abs_rel = torch.abs(gt - est) / gt
            a1 = (abs_rel < 0.1).to(torch.float32).mean()
            a2 = (abs_rel < 0.1 ** 2).to(torch.float32).mean()
            a3 = (abs_rel < 0.1 ** 3).to(torch.float32).mean()
            abs_rel = torch.mean(abs_rel)

            d_val = torch.max(gt / est, est / gt)
            d1 = (d_val < 1.25).to(torch.float32).mean()
            d2 = (d_val < 1.25 ** 2).to(torch.float32).mean()
            d3 = (d_val < 1.25 ** 3).to(torch.float32).mean()

            rmse = torch.sqrt(torch.mean((gt - est) ** 2))
            rmse_log = torch.sqrt(torch.mean(
                (torch.log(gt) - torch.log(est)) ** 2))
            sq_rel = torch.mean(((gt - est) ** 2) / gt)

            abs_abs = torch.mean(torch.abs(gt - est))

            if not keep_batch:
                errors[stage]['abs_rel'] += abs_rel
                errors[stage]['abs'] += abs_abs
                errors[stage]['sq_rel'] += sq_rel
                errors[stage]['rmse'] += rmse
                errors[stage]['rmse_log'] += rmse_log
                errors[stage]['a1'] += a1
                errors[stage]['a2'] += a2
                errors[stage]['a3'] += a3
                errors[stage]['d1'] += d1
                errors[stage]['d2'] += d2
                errors[stage]['d3'] += d3
            else:
                errors[stage]['abs_rel'].append(abs_rel)
                errors[stage]['abs'].append(abs_abs)
                errors[stage]['sq_rel'].append(sq_rel)
                errors[stage]['rmse'].append(rmse)
                errors[stage]['rmse_log'].append(rmse_log)
                errors[stage]['a1'].append(a1)
                errors[stage]['a2'].append(a2)
                errors[stage]['a3'].append(a3)
                errors[stage]['d1'].append(d1)
                errors[stage]['d2'].append(d2)
                errors[stage]['d3'].append(d3)

        if not keep_batch:
            for k in errors[stage]:
                errors[stage][k] = errors[stage][k] / batch_size
        else:
            for k in errors[stage]:
                errors[stage][k] = torch.stack(errors[stage][k])

    return errors


def batched_linspace(start: torch.Tensor, end: torch.Tensor, steps: int):
    steps = int(steps)
    assert steps >= 2

    stepsize = (end - start) / (steps - 1)  # (B, )
    int_range = torch.arange(steps, dtype=start.dtype, device=start.device)  # (S, )
    lin = start[..., None] + stepsize[..., None] * int_range

    assert list(lin.shape) == list(start.shape) + [steps]
    return lin


def uniform_depth_range(*, depth_min, depth_max, depth_num: int, height: int, width: int):
    """
    :param depth_min: (B, )
    :param depth_max: (B, )
    :param depth_num: int
    :param height: int
    :param width: int
    :return: (B, D, H, W), (B, )

    Checked against get_depth_range_samples
    """

    interval = (depth_max - depth_min) / (depth_num - 1)  # (B, )
    int_range = torch.arange(depth_num).type_as(interval)  # (D, )
    depth = depth_min[:, None] + interval[:, None] * int_range[None, :]  # (B, D)
    depth = depth.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width)  # (B, D, H, W)

    if debug:
        assert (depth > 0).all(), f"Depth min = {torch.min(depth)}"
    assert_isfinite(depth, "Depth is not finite.")
    return depth, interval


def adaptive_depth_range(*, depth, interval, depth_num: int, depth_clamp=True,
                         depth_clamp_min=0.001, inverse_depth: bool = False):
    """
    :param depth: (B, H, W)
    :param interval: (B, )
    :param depth_num: int
    :param depth_clamp: bool
        Whether to clamp the depth to [depth_clamp_min, depth_clamp_max]. While this would seem reasonable
        it was commented in the original implementation. (default=False)
    :param depth_clamp_min: float
    :return: (B, D, H, W)

    Checked against get_cur_depth_range_samples
    """

    depth_min = depth - (depth_num / 2) * interval[:, None, None]  # (B, H, W)

    if depth_clamp:
        # TODO: This was commented in original code. Why?
        depth_min = depth_min.clamp(min=depth_clamp_min)  # (B, H, W)
        # depth_max = depth_max.clamp(max=depth_clamp_max)  # (B, H, W)

    depth_max = depth_min + depth_num * interval[:, None, None]  # (B, H, W)

    if not inverse_depth:
        lin_range = torch.linspace(0, 1, depth_num + 1)[:-1].type_as(
            depth).reshape(1, -1, 1, 1)  # (1, D, 1, 1)
        depth_range = depth_min.unsqueeze(1) + (depth_max - depth_min).unsqueeze(1) * lin_range  # (B, D, H, W)
    else:
        # a = 0.01  # fraction of error in terms of depth
        # c = 0.01  # constant error (same unit as depth)

        # depth_range = (1/a)*(1/batched_linspace(1/(a*depth_min+c), 1/(a*depth_max+c), depth_num) - c)  # (B, H, W, D)
        # depth_range = depth_range.permute(0, 3, 1, 2)  # (B, D, H, W)
        # assert depth_range.size(0) == depth.size(0)
        # assert depth_range.size(1) == depth_num
        # assert depth_range.size(2) == depth.size(1)
        # assert depth_range.size(3) == depth.size(2)

        # middle_depth = 0.5*(depth_range[:, depth_num//2-1] + depth_range[:, depth_num//2])  # (B, H, W)
        # depth_offset = depth - middle_depth

        # depth_range += depth_offset.unsqueeze(1)

        if depth_num == 32:
            f = 0.5
        elif depth_num == 8:
            f = 0.3
        else:
            raise NotImplementedError(f"{depth_num} not implemented.")
        assert depth_num % 2 == 0

        half = ((1 / torch.linspace(1 / f, 1 / 1, depth_num // 2) - f) / (1 - f)).type_as(depth)  # (D, )
        half = (half + 0.5 * half[1]) / (1 + 0.5 * half[1])  # (D, )
        both = 0.5 * torch.cat([-torch.flip(half, (0,)), half], 0) + 0.5  # (D, )
        both = both.reshape(1, -1, 1, 1)
        depth_range = depth_min.unsqueeze(1) + (depth_max - depth_min).unsqueeze(1) * both  # (B, D, H, W)

    assert (depth_range > 0).all(
    ), f"Depth_range min = {torch.min(depth_range)}"
    assert_isfinite(depth_range, "Depth_range is not finite.")

    return depth_range


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.closure = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            assert self.closure is not None, "No self.closure"
            closure = self.closure
            self.closure = None  # Make sure to not use twice

        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
