import torch
import torch.nn as nn
from torch import Tensor
from types import FunctionType
from typing import Type, Any, Callable, Union, List, Optional

def _log_api_usage_once(obj: Any) -> None:
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResPoseNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        num_point: int = 12,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        block.expansion = 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 98
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 98, 3, stride=2)
        self.layer2 = self._make_layer(block, 49, 3, stride=2)
        self.layer3 = self._make_layer(block, 1, 3, stride=2)

        self.layer4 = self._make_layer(block, 32, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 * block.expansion, num_point * 2)

        # self.layer4_1 = self._make_layer(block, 16, 3, stride=2)
        # self.layer4_2 = self._make_layer(block, 16, 3, stride=2)
        # self.layer4_3 = self._make_layer(block, 16, 3, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc_1 = nn.Linear(16 * block.expansion, 2) # move
        # self.fc_2 = nn.Linear(16 * block.expansion, 3) # pose
        # self.fc_3 = nn.Linear(16 * block.expansion, 11) # attributes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x

    def _forward_feature(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def _forward_trans(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x

    # def _forward_impl(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #
    #     x_1 = self.layer4_1(x)
    #     x_1 = self.avgpool(x_1)
    #     x_1 = torch.flatten(x_1, 1)
    #     x_1 = self.fc_1(x_1)
    #
    #     x_2 = self.layer4_2(x)
    #     x_2 = self.avgpool(x_2)
    #     x_2 = torch.flatten(x_2, 1)
    #     x_2 = self.fc_2(x_2)
    #
    #     x_3 = self.layer4_3(x)
    #     x_3 = self.avgpool(x_3)
    #     x_3 = torch.flatten(x_3, 1)
    #     x_3 = self.fc_3(x_3)
    #
    #     return x_1, x_2, x_3
    #
    # def _forward_feature(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     return x
    #
    # def _forward_trans(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x_1 = self.layer4_1(x)
    #     x_1 = self.avgpool(x_1)
    #     x_1 = torch.flatten(x_1, 1)
    #     x_1 = self.fc_1(x_1)
    #
    #     x_2 = self.layer4_2(x)
    #     x_2 = self.avgpool(x_2)
    #     x_2 = torch.flatten(x_2, 1)
    #     x_2 = self.fc_2(x_2)
    #
    #     x_3 = self.layer4_3(x)
    #     x_3 = self.avgpool(x_3)
    #     x_3 = torch.flatten(x_3, 1)
    #     x_3 = self.fc_3(x_3)
    #
    #     return x_1, x_2, x_3
    #
    def forward(self, x: Tensor, mode: int = 0) -> Tensor:
        if mode == 0:
            return self._forward_impl(x)
        elif mode == 1:
            return self._forward_feature(x)
        elif mode == 2:
            return self._forward_trans(x)

class NormResPoseNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        num_point: int = 12,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        block.expansion = 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 98
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 98, 3, stride=2)
        self.layer2 = self._make_layer(block, 49, 3, stride=2)
        self.layer3 = self._make_layer(block, 1, 3, stride=2)

        self.layer4 = self._make_layer(block, 32, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 * block.expansion, num_point * 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.sigmoid(self.layer3(x))

        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x

    def _forward_feature(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.sigmoid(self.layer3(x))
        return x

    def _forward_trans(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x

    def forward(self, x: Tensor, mode: int = 0) -> Tensor:
        if mode == 0:
            return self._forward_impl(x)
        elif mode == 1:
            return self._forward_feature(x)
        elif mode == 2:
            return self._forward_trans(x)


class ResPoseWNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        num_point: int = 12,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        # from .attention_networks import Self_Attn
        # self.attention_layer = Self_Attn(64, 'relu')

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 3
        self.dilation = 1
        block.expansion = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 1, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.layer5 = self._make_layer(block, 32, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 * block.expansion, num_point * 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.tanh(self.layer4(x))

        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x

    def _forward_feature(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.tanh(self.layer4(x))

        return x

    def _forward_trans(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x

    # def _forward_impl(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #
    #     x_1 = self.layer4_1(x)
    #     x_1 = self.avgpool(x_1)
    #     x_1 = torch.flatten(x_1, 1)
    #     x_1 = self.fc_1(x_1)
    #
    #     x_2 = self.layer4_2(x)
    #     x_2 = self.avgpool(x_2)
    #     x_2 = torch.flatten(x_2, 1)
    #     x_2 = self.fc_2(x_2)
    #
    #     x_3 = self.layer4_3(x)
    #     x_3 = self.avgpool(x_3)
    #     x_3 = torch.flatten(x_3, 1)
    #     x_3 = self.fc_3(x_3)
    #
    #     return x_1, x_2, x_3
    #
    # def _forward_feature(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     return x
    #
    # def _forward_trans(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x_1 = self.layer4_1(x)
    #     x_1 = self.avgpool(x_1)
    #     x_1 = torch.flatten(x_1, 1)
    #     x_1 = self.fc_1(x_1)
    #
    #     x_2 = self.layer4_2(x)
    #     x_2 = self.avgpool(x_2)
    #     x_2 = torch.flatten(x_2, 1)
    #     x_2 = self.fc_2(x_2)
    #
    #     x_3 = self.layer4_3(x)
    #     x_3 = self.avgpool(x_3)
    #     x_3 = torch.flatten(x_3, 1)
    #     x_3 = self.fc_3(x_3)
    #
    #     return x_1, x_2, x_3
    #
    def forward(self, x: Tensor, mode: int = 0) -> Tensor:
        if mode == 0:
            return self._forward_impl(x)
        elif mode == 1:
            return self._forward_feature(x)
        elif mode == 2:
            return self._forward_trans(x)

class ResPose4Net(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        num_point: int = 12,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        block.expansion = 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 98
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 98, 3, stride=2)
        self.layer2 = self._make_layer(block, 49, 3, stride=2)
        self.layer3 = self._make_layer(block, 7, 3, stride=2)
        self.layer4 = self._make_layer(block, 1, 3, stride=2)

        self.layer5 = self._make_layer(block, 32, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 * block.expansion, num_point * 2)

        # self.layer4_1 = self._make_layer(block, 16, 3, stride=2)
        # self.layer4_2 = self._make_layer(block, 16, 3, stride=2)
        # self.layer4_3 = self._make_layer(block, 16, 3, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc_1 = nn.Linear(16 * block.expansion, 2) # move
        # self.fc_2 = nn.Linear(16 * block.expansion, 3) # pose
        # self.fc_3 = nn.Linear(16 * block.expansion, 11) # attributes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x

    def _forward_feature(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _forward_trans(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x

    def forward(self, x: Tensor, mode: int = 0) -> Tensor:
        if mode == 0:
            return self._forward_impl(x)
        elif mode == 1:
            return self._forward_feature(x)
        elif mode == 2:
            return self._forward_trans(x)

def _normresposenet(**kwargs: Any) -> ResPoseNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return NormResPoseNet(Bottleneck, **kwargs)

def _resposenet(**kwargs: Any) -> ResPoseNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return ResPoseNet(Bottleneck, **kwargs)

def _respose4net(**kwargs: Any) -> ResPose4Net:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return ResPose4Net(Bottleneck, **kwargs)

def _resposewnet(**kwargs: Any) -> ResPoseNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return ResPoseWNet(Bottleneck, layers = [3, 4, 6, 3], **kwargs)

class PoseMapBN(nn.Module):
    def __init__(self, input_num, output_num):
        super().__init__()

        self.ln1 = nn.Linear(input_num, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.ac1 = nn.LeakyReLU()

        self.ln2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.ac2 = nn.LeakyReLU()

        self.ln3 = nn.Linear(256, output_num)

    def forward(self, x):
        x = self.ac1(self.bn1(self.ln1(x)))
        x = self.ac2(self.bn2(self.ln2(x)))
        out = self.ln3(x)
        return out

class PoseMap(nn.Module):
    def __init__(self, input_num, output_num):
        super().__init__()

        self.ln1 = nn.Linear(input_num, 256)
        self.ac1 = nn.LeakyReLU()

        self.ln2 = nn.Linear(256, 256)
        self.ac2 = nn.LeakyReLU()

        self.ln3 = nn.Linear(256, output_num)

    def forward(self, x):
        x = self.ac1(self.ln1(x))
        x = self.ac2(self.ln2(x))
        out = self.ln3(x)
        return out