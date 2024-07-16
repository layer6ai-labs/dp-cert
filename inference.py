"""
This script loads config and model checkpoints from a folder, and runs inference on selected metrics on the training and test set.
Basic Usage: CUDA_VISIBLE_DEVICES=2 python inference.py --load_dir $your_checkpoint_dir 
To modify the metrics used for inference, directly modify `--valid_metrics` `--test_metrics` parameters in the code.
To modify the inference parameters, change the values for `certified_n0`, `certified_n`, `certified_alpha`, `certified_noise_std` etc
In case you have already computed the metrics and want to compute new metrics from a json file, use the `--load_json` flag, then include metric names already computed in the "skip_metrics" field.
If you don't want to save the metrics to a json file, use the `--skip_saving_json` flag.
For debugging using a small subset of the dataset, use the `--debug` flag.
"""
import argparse
import json
import os
import pprint
import random
from collections import defaultdict
from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine

from config import load_config
from datasets import get_loaders_from_config
from evaluators import create_evaluator
from evaluators.evaluator import metrics
from models import create_model
from utils import verify_format

metric_fn_dict = dict(getmembers(metrics, predicate=isfunction))

def main():
    parser = argparse.ArgumentParser(description="Inference config for DP-SGD")

    parser.add_argument("--load_dir", type=str, default="",
                        help="Directory to load from, relative to project root.")

    # parser.add_argument("--valid_metrics", default=["accuracy", "per_sample_loss_input_grad_norm", "certified_robustness", "hessian_eigen_value"], action="append")
    # parser.add_argument("--test_metrics", default=["accuracy", "per_sample_loss_input_grad_norm", "certified_robustness", "hessian_eigen_value"], action="append")
    parser.add_argument("--valid_metrics", default=[], action="append", help="List of metrics used for training set")
    # parser.add_argument("--test_metrics", default=["certified_robustness", "accuracy", "estimate_local_lip_v2"], action="append", help="List of metrics used for test set")
    parser.add_argument("--test_metrics", default=["certified_robustness", "accuracy", "estimate_local_lip_v2", "per_sample_loss_input_grad_norm", "hessian_eigen_value", "robustness_succ"], 
                        help="List of metrics used for test set", nargs="*")
    
    parser.add_argument("--skep_metrics", default=[], action="append", help="List of metrics you would like to skip")
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for inference")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--config", default=[], action="append", help="Override config entries. Specify as `key=value`.")
    # parser.add_argument("--adversarial_attacks", default=["pgd", "fgsm", "pgdl2", "deepfooll2", "deepfoollinf", "cw", "boundary"], action="append")
    parser.add_argument("--adversarial_attacks", default=["fgsm"], action="append")
    parser.add_argument("--gammas_linf", default=[0.0005, 0.01, 0.1, 0.5, 1], action="append")
    parser.add_argument("--gammas_l2", default=[0.3, 0.5, 1, 2, 3, 4], action="append")
    parser.add_argument("--certified_n0", type=int, default=100)
    parser.add_argument("--certified_n", type=int, default=10000)
    parser.add_argument("--certified_alpha", type=float, default=0.001)
    # parser.add_argument("--certified_noise_std", default=[0.12, 0.25, 0.5, 1.0])
    parser.add_argument("--certified_noise_std", default=[0.25, 0.5, 1.0])
    parser.add_argument("--use_original_noise_std", action="store_true")
    parser.add_argument("--load_json", action="store_true", help="Load metrics from json file. Useful for computing new metrics only")
    parser.add_argument("--skip_saving_json", action="store_true", help="Skip saving metrics to the json file. Useful for debugging.")
    parser.add_argument("--debug", action="store_true", help="Use a small subset of data for debugging.")

    
    args = parser.parse_args()
    logdir = args.load_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.load_dir = verify_format(args.load_dir)
    cfg = load_config(args)
    cfg["train_batch_size"] = args.batch_size
    cfg["valid_batch_size"] = args.batch_size
    cfg["test_batch_size"] = args.batch_size
    if "augment_noise_std" in cfg and args.use_original_noise_std:
        args.certified_noise_std = [cfg["augment_noise_std"]]
    if args.load_json:
        args.valid_metrics = [metric for metric in args.valid_metrics if metric not in args.skep_metrics]
        args.test_metrics = [metric for metric in args.valid_metrics if metric not in args.skep_metrics]

    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    print(10 * "-" + "-cfg--" + 10 * "-")
    pp.pprint(cfg)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    train_loader, valid_loader, test_loader = get_loaders_from_config(
        cfg,
        device
    )
    
    if args.debug:
        # take the first 100 imames in train loader
        train_loader.dataset.tensors = tuple([tensor[:10] for tensor in train_loader.dataset.tensors])
        test_loader.dataset.tensors = tuple([tensor[:10] for tensor in test_loader.dataset.tensors])
    
    model, optimizer = create_model(cfg, device)

    privacy_engine = PrivacyEngine(accountant=cfg["accountant"])
    model, optimizer, _ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=cfg["noise_multiplier"],
        max_grad_norm=cfg["l2_norm_clip"],  # C
    )

    checkpoint = torch.load(os.path.join(args.load_dir, f"checkpoints/latest.pt"), map_location=device)
    loaded_model_state = checkpoint["module_state_dict"]
    model.load_state_dict(loaded_model_state)
    model.eval()

    evaluator = create_evaluator(
        model,
        valid_loader=train_loader, test_loader=test_loader,
        valid_metrics=args.valid_metrics,
        test_metrics=args.test_metrics,
        num_classes=cfg['output_dim'],
        adversarial_attacks=args.adversarial_attacks,
        logdir=logdir,
        adv_attack_params=cfg,  # TODO: pass thru just params needed for adv attack
        gammas_l2=args.gammas_l2,
        gammas_linf=args.gammas_linf,
        certified_n0=args.certified_n0,
        certified_n=args.certified_n,
        certified_alpha=args.certified_alpha,
        certified_noise_std=args.certified_noise_std,
        device=device
    )

    if args.load_json:
        train_metric_df = pd.read_json(os.path.join(logdir, f"train_results.json"))
    else:
        train_metric_df = pd.DataFrame()
    
    per_sample_grad_norm_history_path = os.path.join(logdir, f"per_sample_grad_norm_history.json")
    if os.path.exists(per_sample_grad_norm_history_path):
        with open(per_sample_grad_norm_history_path, "r") as f:
            per_sample_grad_norm_history = json.load(f)
    
        per_sample_grad_norm_last = defaultdict(dict)
        for image_id in per_sample_grad_norm_history:
            per_sample_grad_norm_last[int(image_id)]["param_grad_norm"] = per_sample_grad_norm_history[image_id][-1]
        # transform per_sample_grad_norm_last to dataframe
        per_sample_grad_norm_last_df = pd.DataFrame(per_sample_grad_norm_last)

    train_results = evaluator.validate()
    train_metric_dicts = [train_result[1] for train_result in train_results.values()]
    if os.path.exists(per_sample_grad_norm_history_path):
        train_metric_dicts += [per_sample_grad_norm_last_df]
    
    for metric_dict in train_metric_dicts:
        cur_dataframes = pd.DataFrame(metric_dict).T
        # drop columns that are already in train_metric_df
        cur_dataframes = cur_dataframes.drop(columns=[col for col in cur_dataframes.columns if col in train_metric_df.columns])
        train_metric_df = train_metric_df.join(cur_dataframes, how="outer")
    if not args.skip_saving_json:
        train_metric_df.to_json(os.path.join(logdir, f"train_results.json"))

    if args.load_json:
        test_metric_df = pd.read_json(os.path.join(logdir, f"test_results.json"))
    else:
        test_metric_df = pd.DataFrame()
        
    test_results = evaluator.test()
    test_metric_dicts = [test_result[1] for test_result in test_results.values()]
    # convert each dict in metric_dicts to pandas dataframes and pd.join() them based on index
    for metric_dict in test_metric_dicts:
        cur_dataframes = pd.DataFrame(metric_dict).T
        # print(cur_dataframes)
        cur_dataframes = cur_dataframes.drop(columns=[col for col in cur_dataframes.columns if col in test_metric_df.columns])
        test_metric_df = test_metric_df.join(cur_dataframes, how="outer")

    if not args.skip_saving_json:
        test_metric_df.to_json(os.path.join(logdir, f"test_results.json"))


if __name__ == "__main__":
    main()
    