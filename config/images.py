def get_base_config(dataset, regular_training=False):
    # default config for regular method
    lr = 0.01
    batch_size = 512
    noise_multiplier = 0.8
    max_epochs = 40
    delta = 1e-6
    optimizer = "sgd"

    if dataset in ["mnist", "fashion-mnist", "svhn", "cifar10"]:
        output_dim = 10
    elif dataset in ["celeba"]:
        output_dim = 2
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    if not regular_training:
        if dataset == "mnist":
            batch_size = 512
            lr = 0.5
            max_epochs = 40
            noise_multiplier = 1.23

        elif dataset == "fashion-mnist":
            lr = 4
            batch_size = 2048
            noise_multiplier = 2.15
            max_epochs = 40

        elif dataset == "cifar10":
            lr = 0.001
            batch_size = 512
            noise_multiplier = 1.0
            max_epochs = 10
            optimizer = "adam"
            
    net_configs = {
        "net": "cnn",
        "WRN_depth": 16,
        "WRN_k": 4,
        "pretrained": True,
        "fine_tune_layers": "last",  
        "activation": "tanh",
        "hidden_channels": [16, 32],
        "pooling": "max",
        "output_dim": output_dim,
        "kernel_size": [8, 4],
        "stride": [2, 2],
        "padding": [2, 0],
        "pooling_kernel_sizes": [2, 2],
        "pooling_strides": [1, 1],
        "fc_dims": [32]
    }

    PGD_configs = {
        "PGD_iterations": 40,
        "PGD_stepsize": 2 / 255,  # alpha in PGD
    }

    PGDL2_configs = {
        "PGDL2_iterations": 40,
        "PGDL2_stepsize": 0.2,  # alpha in PGDL2
    }

    CW_configs = {
        "CW_c_iterations": 9,
        "CW_c": 1,
        "CW_confidence": 0,  # confidence
        "CW_iterations": 10000,
        "CW_stepsize": 0.01,
    }

    # NOTE: use same configs for both l2 and linf
    DeepFool_configs = {
        "DeepFool_iterations": 50,
        "DeepFool_overshoot": 0.02,
        "DeepFool_loss": "logits",  # "logits", "crossentropy"
    }

    BA_configs = {
        "BA_iterations": 25000,
    }

    return {
        "seed": 0,
        "optimizer": optimizer,
        "lr": lr,
        "momentum": 0.9,
        "use_lr_scheduler": False,
        "max_epochs": max_epochs,
        "accountant": "rdp",
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "l2_norm_clip": 0.1,
        "clip": True,
        "add_noise": True,

        "make_valid_loader": False,
        "train_batch_size": batch_size,
        "valid_batch_size": 512,
        "test_batch_size": 4096,
        "physical_batch_size": 2048,

        "valid_metrics": ["accuracy"],  # "robustness_succ"]
        "test_metrics": ["accuracy"],  # "robustness_succ_w_sample_image"], "macro_accuracy",
        "evaluate_adversarial_loss": False,
        "adversarial_attacks": ["pgd", "fgsm", "pgdl2", "deepfooll2", "deepfoollinf", "cw", "boundary"],
        "gammas_linf": [0.0005, 0.0008, 0.0015, 0.003, 0.01, 0.05, 0.1, 0.15, 0.3, 0.4, 0.5, 0.6],
        "gammas_l2": [0.3, 0.5, 1, 2, 3, 4],
        "norm_layer": None,
        "certified_n0": 100,
        "certified_n": 100000,
        "certified_alpha": 0.001,
        "certified_noise_std": [0.1, 0.25, 0.5, 0.75, 1],
        **net_configs,

        # config for adversarial attacks
        **PGD_configs,
        **PGDL2_configs,
        **CW_configs,
        **DeepFool_configs,
        **BA_configs,
    }


def get_non_private_config(dataset):
    return {
        "clip": False,
        "add_noise": False,
    }


def get_dpsgd_config(dataset):
    return {
        "activation": "tanh",
    }


def get_dpsgd_f_config(dataset):
    return {
        "base_max_grad_norm": 1.0,  # C0
        "counts_noise_multiplier": 10.0  # noise scale applied on mk and ok
    }


def get_dpsgd_global_config(dataset):
    return {
        "strict_max_grad_norm": 100,  # Z
    }


def get_dpsgd_global_adapt_config(dataset):
    return {
        "strict_max_grad_norm": 100,  # Z
        "bits_noise_multiplier": 10.0,  # noise scale applied on average of bits
        "lr_Z": 0.1,  # learning rate with which Z^t is tuned
        "threshold": 1,  # threshold in how we compare gradient norms to Z
    }

def get_dpsgd_auto_clip_config(dataset):
    return {
        "gamma": 0.01,  # the smoothing parameter for auto-clip
        "psac": True, # whether using psac or not
    }

def get_regular_augment_config(dataset):
    return {
        "augment_noise_std": 1.0,  # std of the noise that should be added to the input images
        "num_augmentations": 10,  # how many augmentations of a singe image
        "clip": False,
        "add_noise": False
    }

def get_dpsgd_augment_config(dataset):
    return {
        "consistency": False,
        "augment_noise_std": 1.0,  # std of the noise that should be added to the input images
        "num_augmentations": 10,  # how many augmentations of a singe image
        "macer": False,
        "trades": False,
        "stability": False
    }

def get_dpsgd_adv_config(dataset):
    return {
        "include_original": True,
        "pgd_steps": 2,
        "warmup": 10,
        "max_norm": 64,
    }

def get_smooth_adv_config(dataset):
    return {
        **get_dpsgd_adv_config(dataset),
        "consistency": False,
        "no_grad": False,
        "num_augmentations": 2,  # how many augmentations of a singe image
        "augment_noise_std": 1.0,  # std of the noise that should be added to the input images
        "trades": False,
        "stability": False
    }

def get_auto_clip_smooth_adv(dataset):
    return {**get_dpsgd_auto_clip_config(dataset), **get_smooth_adv_config(dataset)}

def get_auto_clip_dpsgd_augment_config(dataset):
    return {**get_dpsgd_auto_clip_config(dataset), **get_dpsgd_augment_config(dataset)}

CFG_MAP_IMG = {
    "base": get_base_config,
    "regular": get_non_private_config,
    "dpsgd": get_dpsgd_config,
    "dpsgd-global": get_dpsgd_global_config,
    "dpsgd-global-adapt": get_dpsgd_global_adapt_config,
    "dpsgd-auto-clip": get_dpsgd_auto_clip_config,
    "dpsgd-auto-clip-adv-smooth": get_auto_clip_smooth_adv,
    "dpsgd-adv": get_dpsgd_adv_config,
    "dpsgd-adv-smooth": get_smooth_adv_config,
    "dpsgd-augment-auto-clip": get_auto_clip_dpsgd_augment_config,
    "regular-augment": get_regular_augment_config,
    "dpsgd-augment": get_dpsgd_augment_config,
}
