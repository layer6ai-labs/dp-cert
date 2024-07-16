import ast
import json
import os

from .images import CFG_MAP_IMG

_IMAGE_DATASETS = ["mnist", "fashion-mnist", "svhn", "cifar10", "celeba"]


def get_config(dataset, method):
    if dataset in _IMAGE_DATASETS:
        cfg_map = CFG_MAP_IMG
    else:
        raise ValueError(
            f"Invalid dataset {dataset}. "
            + f"Valid choices are {_IMAGE_DATASETS}."
        )
    base_config = cfg_map["base"](dataset, regular_training=method == "regular")

    if method != "regular":
        try:
            method_config = cfg_map[method](dataset)
        except KeyError:
            cfg_map.pop("base")
            raise ValueError(
                f"Invalid method {method}. "
                + f"Valid choices are {cfg_map.keys()}."
            )
    else:
        method_config = base_config

    return {
        **base_config,

        "dataset": dataset,
        "method": method,

        **method_config
    }


def load_config(args):
    # Load from file
    cfg_path = os.path.join(args.load_dir, "config.json")
    print(f"Loading config from {cfg_path}")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    else:
        raise FileNotFoundError(f"{args.load_dir} does not have `config.json`")

    # Update with command line changes
    cmd_line_cfg = dict(parse_config_arg(kv) for kv in args.config)
    if cmd_line_cfg:  # There are cmd line changes
        print("Updating config with changes to:")
        for key in cmd_line_cfg:
            print(f"{key}")
    cfg = {**cfg, **cmd_line_cfg}

    if args.save_dir:
        update_config_paths(args, cfg)

    return cfg


def parse_config_arg(key_value):
    assert "=" in key_value, "Must specify config items with format `key=value`"

    k, v = key_value.split("=", maxsplit=1)

    assert k, "Config item can't have empty key"
    assert v, "Config item can't have empty value"

    try:
        v = ast.literal_eval(v)
    except ValueError:
        v = str(v)

    return k, v


def update_config_paths(args, cfg):
    head_tail = os.path.split(args.save_dir)
    cfg['logdir_root'] = head_tail[0]
    cfg['logdir'] = head_tail[1]


_NETS = ["mlp", "cnn", "lenet", "resnet9", "wideresnet"]
_ACTIVATIONS = ["relu", "tanh", "swish"]
_OPTIMIZERS = ["sgd", "adam"]
_PRIVACY_ACCOUNTANTS = ["rdp", "prv", "gdp"]
_GRAD_SAMPLE_MODES = ["hooks", "ew", "functorch"]


def verify_config_args(cfg):
    def check_choices(arg_name, choices):
        arg = cfg.get(arg_name, None)
        if ((arg is not None) and (arg not in choices)):
            raise ValueError(
                f"Unexpected '{arg_name}' value provided: {arg}. Try one of: " + 
                ", ".join(choices) + "."
                )

    # check_choices("net", _NETS)
    check_choices("activation", _ACTIVATIONS)
    check_choices("optimizer", _OPTIMIZERS)
    check_choices("accountant", _PRIVACY_ACCOUNTANTS)
    check_choices("grad_sample_mode", _GRAD_SAMPLE_MODES)
