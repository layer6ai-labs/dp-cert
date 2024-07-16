import argparse
import pprint
import random
import sys

import numpy as np
import torch
from opacus.accountants.utils import get_noise_multiplier

from config import get_config, parse_config_arg, load_config, update_config_paths, verify_config_args
from datasets import get_loaders_from_config
from evaluators import create_evaluator
from models import create_model
from privacy_engines.dpsgd_augmented_data_engine import DPSGDAugmentedDataPrivacyEngine
from privacy_engines.dpsgd_auto_clip_engine import DPSGDAutoClipAugmentatedDataPrivacyEngine
from privacy_engines.dpsgd_auto_clip_engine import DPSGDAutoClipPrivacyEngine
from privacy_engines.dpsgd_global_adaptive_engine import DPSGDGlobalAdaptivePrivacyEngine
from privacy_engines.dpsgd_global_engine import DPSGDGlobalPrivacyEngine
from privacy_engines.metric_collection_engine import PrivacyEngineWithMetrics
from privacy_engines.regular_augmented_data_engine import RegularPrivacyEngine
from trainers import create_trainer
from utils import verify_format, privacy_checker
from writer import get_writer


def main():
    parser = argparse.ArgumentParser(description="Fairness for DP-SGD")

    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset to train on.")
    parser.add_argument("--method", type=str, default="regular",
                        choices=["regular", "regular-augment", "dpsgd", "dpsgd-global",
                                 "dpsgd-global-adapt", "dpsgd-auto-clip", "dpsgd-augment", "dpsgd-augment-auto-clip",
                                 "dpsgd-adv", "dpsgd-adv-smooth", "dpsgd-auto-clip-adv-smooth"],
                        help="Method for training and clipping.")
    parser.add_argument("--load-dir", type=str, default="",
                        help="Directory to load from, relative to project root.")
    parser.add_argument("--save-dir", type=str, default="",
                        help="Directory to save results to for new runs, or when loading old runs, relative to project root.")

    parser.add_argument("--config", default=[], action="append",
                        help="Override config entries. Specify as `key=value`.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_dir:
        args.save_dir = verify_format(args.save_dir)
    if args.load_dir:
        args.load_dir = verify_format(args.load_dir)
        cfg = load_config(args)
    else:
        cfg = get_config(
            dataset=args.dataset,
            method=args.method,
        )
        if args.save_dir:
            update_config_paths(args, cfg)
        cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}
    verify_config_args(cfg)

    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    # Set random seeds based on config
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    train_loader, valid_loader, test_loader = get_loaders_from_config(
        cfg,
        device
    )

    if cfg["method"] != "regular":
        sample_rate = 1 / len(train_loader)
        if cfg['dataset'] =='cifar10':
            cfg["noise_multiplier"] = get_noise_multiplier(
                target_epsilon = 1.0,
                target_delta = cfg["delta"],
                sample_rate = sample_rate,
                epochs = cfg['max_epochs'],
                accountant = "rdp"
            )
        privacy_checker(sample_rate, cfg)

    print(10 * "-" + "-cfg--" + 10 * "-")
    pp.pprint(cfg)

    writer = get_writer(args, cfg)

    model, optimizer = create_model(cfg, device)

    if cfg["method"] in ["dpsgd", "dpsgd-adv"]:
        privacy_engine = PrivacyEngineWithMetrics(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],
            max_grad_norm=cfg["l2_norm_clip"],  # C
            grad_sample_mode=cfg.get("grad_sample_mode", "hooks")
        )
    elif cfg["method"] == "dpsgd-global":
        privacy_engine = DPSGDGlobalPrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],  # sigma in sigma * C
            max_grad_norm=cfg["l2_norm_clip"],  # C
            grad_sample_mode=cfg.get("grad_sample_mode", "hooks")
        )
    elif cfg["method"] == "dpsgd-global-adapt":
        privacy_engine = DPSGDGlobalAdaptivePrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],  # sigma in sigma * C
            max_grad_norm=cfg["l2_norm_clip"],  # C
            grad_sample_mode=cfg.get("grad_sample_mode", "hooks")
        )
    elif cfg["method"] == "dpsgd-auto-clip":
        privacy_engine = DPSGDAutoClipPrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],  # sigma in sigma * C
            max_grad_norm=cfg["l2_norm_clip"],  # C
            grad_sample_mode=cfg.get("grad_sample_mode", "hooks")
        )
    elif cfg["method"] in ["dpsgd-augment", "dpsgd-adv-smooth"]:
        privacy_engine = DPSGDAugmentedDataPrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],  # sigma in sigma * C
            max_grad_norm=cfg["l2_norm_clip"],  # C
            grad_sample_mode=cfg.get("grad_sample_mode", "hooks")
        )

    elif cfg["method"] in ["dpsgd-augment-auto-clip", "dpsgd-auto-clip-adv-smooth"]:
        privacy_engine = DPSGDAutoClipAugmentatedDataPrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],  # sigma in sigma * C
            max_grad_norm=cfg["l2_norm_clip"],  # C
            grad_sample_mode=cfg.get("grad_sample_mode", "hooks")
        )

    elif cfg["method"] in {"regular-augment", "regular"}:
        # doing regular training
        privacy_engine = RegularPrivacyEngine()
        # cfg['clip'] = cfg['add_noise'] = False
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0,
            max_grad_norm=sys.float_info.max,
            poisson_sampling=False,
            grad_sample_mode=cfg.get("grad_sample_mode", "hooks")
        )
    else:
        raise Exception("Your chosen method {} is invalid".format(cfg["method"]))

    evaluator = create_evaluator(
        model,
        valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=cfg["valid_metrics"],
        test_metrics=cfg["test_metrics"],
        num_classes=cfg["output_dim"],
        adversarial_attacks=cfg['adversarial_attacks'],
        logdir=writer.logdir,
        adv_attack_params=cfg,  # TODO: pass thru just params needed for adv attack
        gammas_l2=cfg["gammas_l2"],
        gammas_linf=cfg["gammas_linf"],
        certified_n0=cfg["certified_n0"],
        certified_n=cfg["certified_n"],
        certified_alpha=cfg["certified_alpha"],
        certified_noise_std=cfg["certified_noise_std"],
        device=device,
    )

    trainer = create_trainer(
        train_loader,
        valid_loader,
        test_loader,
        model,
        optimizer,
        evaluator,
        privacy_engine,
        writer,
        device,
        cfg
    )

    trainer.train()


if __name__ == "__main__":
    main()
