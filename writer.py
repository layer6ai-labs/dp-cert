# NOTE: The below file is modified from commit `aeaf5fd` of
#       https://github.com/jrmcornish/cif/blob/master/cif/writer.py

import datetime
import json
import os
import sys
from typing import Iterable

import numpy as np
import torch
from tensorboardX import SummaryWriter


class Tee:
    """This class allows for redirecting of stdout and stderr"""

    def __init__(self, primary_file, secondary_file):
        self.primary_file = primary_file
        self.secondary_file = secondary_file

        self.encoding = self.primary_file.encoding

    def isatty(self):
        return self.primary_file.isatty()

    def fileno(self):
        return self.primary_file.fileno()

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode()

        self.primary_file.write(data)
        self.secondary_file.write(data)

    def flush(self):
        self.primary_file.flush()
        self.secondary_file.flush()


class Writer:
    _STDOUT = sys.stdout
    _STDERR = sys.stderr

    def __init__(self, logdir, make_subdir, tag_group, dir_name):
        if make_subdir:
            os.makedirs(logdir, exist_ok=True)
        logdir = os.path.join(logdir, dir_name)

        self._writer = SummaryWriter(logdir=logdir)

        assert logdir == self._writer.logdir
        self.logdir = logdir
        self.dir_name = dir_name

        self._tag_group = tag_group

        sys.stdout = Tee(
            primary_file=self._STDOUT,
            secondary_file=open(os.path.join(logdir, "stdout"), "a")
        )

        sys.stderr = Tee(
            primary_file=self._STDERR,
            secondary_file=open(os.path.join(logdir, "stderr"), "a")
        )

    def write_scalar(self, tag, scalar_value, global_step=None):
        self._writer.add_scalar(self._tag(tag), scalar_value, global_step=global_step)

    def write_scalars(self, tag, scalar_dict, global_step=None):
        self._writer.add_scalars(self._tag(tag), scalar_dict, global_step=global_step)

    def write_image(self, tag, img_tensor, global_step=None):
        self._writer.add_image(self._tag(tag), img_tensor, global_step=global_step)

    def write_figure(self, tag, figure, global_step=None):
        self._writer.add_figure(self._tag(tag), figure, global_step=global_step)

    def write_hparams(self, hparam_dict=None, metric_dict=None):
        self._writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    def write_json(self, tag, data):
        text = json.dumps(data, indent=4)

        self._writer.add_text(
            self._tag(tag),
            4 * " " + text.replace("\n", "\n" + 4 * " ")  # Indent by 4 to ensure codeblock formatting
        )

        json_path = os.path.join(self.logdir, f"{tag}.json")

        with open(json_path, "w") as f:
            f.write(text)

    def write_textfile(self, tag, text):
        path = os.path.join(self.logdir, f"{tag}.txt")
        with open(path, "w") as f:
            f.write(text)

    def write_numpy(self, tag, arr):
        path = os.path.join(self.logdir, f"{tag}.npy")
        np.save(path, arr)
        print(f"Saved array to {path}")

    def write_checkpoint(self, tag, data):
        os.makedirs(self._checkpoints_dir, exist_ok=True)
        checkpoint_path = self._checkpoint_path(tag)

        tmp_checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path),
            f"{os.path.basename(checkpoint_path)}.tmp"
        )

        torch.save(data, tmp_checkpoint_path)
        # replace is atomic, so we guarantee our checkpoints are always good
        os.replace(tmp_checkpoint_path, checkpoint_path)


    def print_dict(self, tag_prefix, value_dict):
        for k, v in value_dict.items():
            if isinstance(v, Iterable):
                print("{} {}: {}".format(tag_prefix, k, v))
            else:
                print(f"{tag_prefix} {k}: {v:.4f}")

    def load_checkpoint(self, tag, device):
        return torch.load(os.path.join(self.load_dir, f"checkpoints/{tag}.pt"), map_location=device)

    def _checkpoint_path(self, tag):
        return os.path.join(self._checkpoints_dir, f"{tag}.pt")

    @property
    def _checkpoints_dir(self):
        return os.path.join(self.logdir, "checkpoints")

    def _tag(self, tag):
        return f"{self._tag_group}/{tag}"

def get_writer(args, config):
    if args.load_dir and not args.save_dir:
        head_tail = os.path.split(args.load_dir)
        writer = Writer(
            logdir=head_tail[0],
            make_subdir=False,
            tag_group=args.dataset,
            dir_name=head_tail[1]
        )
    else:
        # save_dir has already been used to update the config if specified
        dir_name = f"{config['dataset']}_{config['method']}_{config['net']}_{config['optimizer']}_lr_{config['lr']}_bsz_{config['train_batch_size']}" + \
            f"_clip_bound_{config['l2_norm_clip']}_sigma_{config['noise_multiplier']}"
        if config['method'] in ['dpsgd-adv-smooth', 'dpsgd-auto-clip-adv-smooth', 'dpsgd-adv']:
            dir_name += f"_pgd_steps_{config['pgd_steps']}"
        if config['method'] in ['dpsgd-adv-smooth', 'dpsgd-auto-clip-adv-smooth']:
            dir_name += f"_no_grad_{config['no_grad']}_max_norm_{config['max_norm']}"
        if config['method'] in ['dpsgd-auto-clip', 'dpsgd-auto-clip-adv-smooth']:
            dir_name += f"_psac_{config['psac']}"
        if config['method'] in ['dpsgd-adv-smooth', 'dpsgd-auto-clip-adv-smooth', 'augment']:
            dir_name += f"_num_augmentations_{config['num_augmentations']}_augment_noise_std_{config['augment_noise_std']}"
        
        if "consistency" in config and config['consistency']:
            dir_name += f"_consistency_{config['consistency']}"
        
        if "trades" in config and config['trades']:
            dir_name += f"_trades_{config['trades']}"

        if "macer" in config and config['macer']:
            dir_name += f"_macer_{config['macer']}"

        dir_name += datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        writer = Writer(
            logdir=config.get("logdir_root", "runs"),
            make_subdir=True,
            tag_group=args.dataset,
            dir_name=config.get("logdir", dir_name)
        )
        writer.write_json(tag="config", data=config)
    
    writer.load_dir = args.load_dir
        
    return writer

