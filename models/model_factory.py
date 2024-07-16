import timm
from opacus.validators import ModuleValidator

from tan_models.models.wideresnet import *
from .neural_networks import (CNN, MLP, LeNet, CIFAR_CNN, ResNet9)

activation_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "swish": nn.SiLU,
    "mish": nn.Mish
}

pooling_map = {
    "max": nn.MaxPool2d,
    "avg": nn.AvgPool2d,
    "identity": nn.Identity
}

def create_model(config, device):
    if config["net"] == "mlp":
        model = MLP(
            n_units_list=[config["data_dim"], *config["hidden_dims"], config["output_dim"]],
            activation=activation_map[config.get("activation", "relu")],
        )
    elif config["net"] == "cnn":
        if config['dataset'] == 'cifar10':
            model = CIFAR_CNN(
                input_dim=config["data_shape"][0],
                image_height=config["data_shape"][1],
                norm_layer=config["norm_layer"],
                output_dims=config["output_dim"],
            )
        else:
            model = CNN(
                input_dim=config["data_shape"][0],
                hidden_dims=config["hidden_channels"],
                output_dims=config["output_dim"],
                kernel_sizes=config["kernel_size"],
                strides=config["stride"],
                padding=config["padding"],
                pooling=pooling_map[config.get("pooling", "relu")],
                pooling_kernel_sizes=config["pooling_kernel_sizes"],
                pooling_strides=config["pooling_strides"],
                fc_dims=config["fc_dims"],
                image_height=config["data_shape"][1],
                norm_layer=config["norm_layer"],
                activation=activation_map[config.get("activation", "relu")],
            )
    elif config["net"].lower() == "lenet":
        model = LeNet()
    elif config["net"].lower() == "resnet9":
        model = ResNet9(input_dim=config["data_shape"][0], output_dims=config["output_dim"])
    elif config["net"].lower() == "wideresnet":
        model = WideResNet(config["WRN_depth"],config["output_dim"],config["WRN_k"])

    else:
        model = timm.create_model(config["net"], pretrained=config['pretrained'], num_classes=config["output_dim"])

        model.device = device
        model = ModuleValidator.fix(model)
        model.to(device)

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    else:
        raise ValueError(f"Unknown optimizer")
    
    if config["net"].lower() in ["mlp", "cnn", "lenet", "resnet9","wideresnet"]:
        model.set_device(device)

    return model, optimizer
