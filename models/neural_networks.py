import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):

    def __init__(self, n_units_list, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev_layer_size = n_units_list[0]
        for n_units in n_units_list[1:-1]:
            layers.append(nn.Linear(in_features=prev_layer_size, out_features=n_units))
            prev_layer_size = n_units
            layers.append(activation())
        layers.append(nn.Linear(in_features=prev_layer_size, out_features=n_units_list[-1]))
        self.net = nn.Sequential(*layers)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dims,
            kernel_sizes,
            strides,
            padding,
            pooling,
            pooling_kernel_sizes,
            pooling_strides,
            fc_dims,
            image_height,
            norm_layer=None,
            activation=nn.Tanh,
    ):
        super().__init__()
        if norm_layer is None:
            self.norm_layer = nn.Identity
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"No such norm {norm_layer}")
        if type(strides) not in [list, tuple]:
            strides = [strides for _ in hidden_dims]

        if type(kernel_sizes) not in [list, tuple]:
            kernel_sizes = [kernel_sizes for _ in hidden_dims]

        cnn_layers = []
        prev_channels = input_dim
        for hidden_channels, k, s, p, pooling_kernel_size, pooling_stride in zip(hidden_dims, kernel_sizes, strides, padding, pooling_kernel_sizes, pooling_strides):
            cnn_layers.append(nn.Conv2d(prev_channels, hidden_channels, kernel_size=k, stride=s, padding=p))
            cnn_layers.append(self.norm_layer(hidden_channels))
            cnn_layers.append(activation())
            cnn_layers.append(pooling(kernel_size=pooling_kernel_size, stride=pooling_stride))
            prev_channels = hidden_channels


            # NOTE: Assumes square image
            image_height = self._get_new_image_height(image_height, k, s, p)
            if type(pooling_kernel_size) != nn.Identity:
                image_height = self._get_new_image_height(image_height, pooling_kernel_size, pooling_stride)
        self.cnn_layers = nn.ModuleList(cnn_layers)
        conv_output_dim = prev_channels * image_height ** 2

        prev_channels = conv_output_dim
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(prev_channels, fc_dim))
            fc_layers.append(activation())
            prev_channels = fc_dim
        fc_layers.append(nn.Linear(prev_channels, output_dims))
        self.fc_layers = nn.ModuleList(fc_layers)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)

        for layer in self.fc_layers:
            x = layer(x)
        return x

    def _get_new_image_height(self, height, kernel, stride, padding=0):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html
        return math.floor((height - kernel + (2 * padding)) / stride + 1)

"""
Paper: https://arxiv.org/pdf/2105.07985.pdf
"""
class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution with no padding
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def set_device(self, device):
        self.device = device
        self.to(device)

    def _get_new_image_height(self, height, kernel, stride):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html
        # Assume dilation = 1, padding = 0
        return math.floor((height - kernel) / stride + 1)

class CIFAR_CNN(CNN):
    def __init__(self,
            input_dim,
            image_height,
            norm_layer,
            output_dims) -> None:

        nn.Module.__init__(self)
        if norm_layer is None:
            self.norm_layer = nn.Identity
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"No such norm {norm_layer}")

        cnn_layers = []
        prev_channels = input_dim
        for filter_size in [32, 64, 128]:
            cnn_layers.append(nn.Conv2d(prev_channels, filter_size, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(self.norm_layer(filter_size))
            cnn_layers.append(nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(self.norm_layer(filter_size))
            cnn_layers.append(nn.Tanh())
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            image_height = self._get_new_image_height(image_height, 3, 1, 1)
            image_height = self._get_new_image_height(image_height, 3, 1, 1)
            image_height = self._get_new_image_height(image_height, 2, 2)
            prev_channels = filter_size

        self.cnn_layers = nn.ModuleList(cnn_layers)
        self.classifier = nn.Sequential(nn.Linear(128 * 4 * 4, 128), nn.Tanh(), nn.Linear(128, output_dims))

    def forward(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

def conv_gn_act(in_channels, out_channels, pool=False, activation=nn.Mish, num_groups=32):
    """Conv-GroupNorm-Activation
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(min(num_groups, out_channels), out_channels),
        activation(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        output_dims: int = 10,
        activation: nn.Module = nn.Mish,
        scale_norm: bool = True,
        num_groups = (32, 32, 32, 32),
    ):
        """9-layer Residual Network. Architecture:
        conv-conv-Residual(conv, conv)-conv-conv-Residual(conv-conv)-FC
        By default, it uses GroupNorm with 32 groups in each layer.
        Args:
            in_channels (int, optional): Channels in the input image. Defaults to 3.
            num_classes (int, optional): Number of classes. Defaults to 10.
            act_func (nn.Module, optional): Activation function to use. Defaults to nn.Mish.
            scale_norm (bool, optional): Whether to add an extra normalisation layer after each residual block. Defaults to False.
            num_groups (tuple[int], optional): Number of groups in GroupNorm layers.\
            Must be a tuple with 4 elements, corresponding to the GN layer in the first conv block, \
            the first res block, the second conv block and the second res block. Defaults to (32, 32, 32, 32).
        """
        super().__init__()
        assert (
            isinstance(num_groups, tuple) and len(num_groups) == 4
        ), "num_groups must be a tuple with 4 members"
        groups = num_groups

        self.conv1 = conv_gn_act(
            input_dim, 64, activation=activation, num_groups=groups[0]
        )
        self.conv2 = conv_gn_act(
            64, 128, pool=True, activation=activation, num_groups=groups[0]
        )

        self.res1 = nn.Sequential(
            *[
                conv_gn_act(128, 128, activation=activation, num_groups=groups[1]),
                conv_gn_act(128, 128, activation=activation, num_groups=groups[1]),
            ]
        )

        self.conv3 = conv_gn_act(
            128, 256, pool=True, activation=activation, num_groups=groups[2]
        )
        self.conv4 = conv_gn_act(
            256, 256, pool=True, activation=activation, num_groups=groups[2]
        )

        self.res2 = nn.Sequential(
            *[
                conv_gn_act(256, 256, activation=activation, num_groups=groups[3]),
                conv_gn_act(256, 256, activation=activation, num_groups=groups[3]),
            ]
        )

        self.MP = nn.AdaptiveMaxPool2d((2, 2))
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024, output_dims)

        if scale_norm:
            self.scale_norm_1 = nn.GroupNorm(min(num_groups[1], 128), 128)
            self.scale_norm_2 = nn.GroupNorm(min(groups[3], 256), 256)
        else:
            self.scale_norm_1 = nn.Identity()  # type:ignore
            self.scale_norm_2 = nn.Identity()  # type:ignore

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.scale_norm_1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.scale_norm_2(out)
        out = self.MP(out)
        out_emb = self.FlatFeats(out)
        out = self.classifier(out_emb)
        return out