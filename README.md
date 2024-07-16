<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

<div align="center">
<h1>
<b>
DP-CERT
</b>
</h1>
<h4>
</div>

This repository presents the code required to reproduce the results of [*Augment then Smooth: Reconciling Differential Privacy with Certified Robustness*](https://arxiv.org/abs/2306.08656), published at [TMLR](https://openreview.net/forum?id=YN0IcnXqsr).

## Prerequisites

- Install conda, pip
- Python 3.10

```bash
conda create -n dp-cert python=3.10
conda activate dp-cert
```

- PyTorch 1.13.0

```bash
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```
Install the rest of the dependencies:

```bash
pip install -r requirements.txt
```

## Training arguments and commands:
- An example command:
A basic command to run DPSGD on MNIST, with the default setting:
```bash
CUDA_VISIBLE_DEVICES=6 python main.py --method=dpsgd-augment --dataset mnist --config num_augmentations=2 --config augment_noise_std=0.25
```
`--dataset`: choose from `mnist`, `fashion-mnist` and `cifar10`

`--method:` 
        
        baselines: 
            - regular: Regular,
            - dpsgd: DPSGD,
            - dpsgd-auto-clip: DPSGD with PSAC;
        augmented training:
            - dpsgd-augment: DP-Gaussian
            - dpsgd-adv-smooth: DP-SmoothAdv
        
`--config`: other config parameters. Usage: `--config $param_name=$param_value`
We introduce some important configs here. For the full list of config parameters, see `config/images.py`.
    
`--net`: name of the network used in training. The default for mnist and fashion-mnist is "cnn". You can also choose any model names that are supported by [timm](https://github.com/fastai/timmdocs/tree/master/).


`--augment_noise_std`: standard deviation $\sigma$ used in training

`--num_augmentations`: number of augmentations used in training

`--trades`: use stability regularization

`--macer`: use macer regularization

`--consistency`: use consistency regularization

`--lr`: learning rate.

`--train_batch_size`: batch size in training.

`--physical_batch_size`: maximum number of examples allowed by the GPU. When training a large model, large batch size can be achieved by accumulating small batches.

`--max_epches`: maximum number of epoches to train.

`--l2_norm_clip`: clipping norm in DPSGD.

`--noise_multiplier`: $\sigma$ used in DPSGD when adding noise.


### Running the following commands to reproduce results of different variants of DP-CERT. We use $\sigma=0.25$ only, for other $\sigma$, simply replace `0.25` with `0.5` and `1.0`.

DP-Gaussian, MNIST: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --method=dpsgd-augment --dataset mnist --config num_augmentations=2 --config augment_noise_std=0.25
```
DP-SmoothAdv, MNIST: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --method=dpsgd-adv-smooth --dataset mnist --config num_augmentations=2 --config augment_noise_std=0.25
```

DP-Stability, MNIST: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --method=dpsgd-augment --dataset mnist --config num_augmentations=2 --config augment_noise_std=0.25 --config trades=True
```

DP-MACER, MNIST: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --method=dpsgd-augment --dataset mnist --config num_augmentations=2 --config augment_noise_std=0.25 --config macer=True
```

For Fashion-MNIST, simply replace `mnist` in the above commands with `fashion-mnist`. For CIFAR, use the following commands:


DP-Gaussian, CIFAR10: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --method=dpsgd-augment --dataset cifar10 --config num_augmentations=1 --config augment_noise_std=0.25 --config net=crossvit_tiny_240.in1k --config physical_batch_size=128
```
DP-Stability, CIFAR10: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --method=dpsgd-adv-smooth --dataset cifar10 --config num_augmentations=1 --config augment_noise_std=0.25 --config net=crossvit_tiny_240.in1k --config physical_batch_size=128
```

DP-SmoothAdv, CIFAR10: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --method=dpsgd-augment --dataset cifar10 --config num_augmentations=1 --config augment_noise_std=0.25 --config net=crossvit_tiny_240.in1k --config physical_batch_size=128 --config trades=True
```

DP-MACER, CIFAR10: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --method=dpsgd-augment --dataset cifar10 --config num_augmentations=1 --config augment_noise_std=0.25 --config net=crossvit_tiny_240.in1k --config physical_batch_size=128 --config macer=True
```

## Inference commands and arguements

The `inference.py` loads config and model checkpoints from a folder, and runs inference on selected metrics on the training and test set.

Basic Usage: 
```bash
CUDA_VISIBLE_DEVICES=2 python inference.py --load_dir $your_checkpoint_dir 
```
`--test_metrics`: list of metrics name to calculate on test set. E.g. `--test_metrics accuracy certified_accuracy`

`--certified_n0`, `--certified_n`, `--certified_alpha`, `--certified_noise_std`: paramters for CERTIFY algorithm. 

`--use_original_noise_std`: use the same $\sigma$ in training

If you don't want to save the metrics to a json file, use the `--skip_saving_json` flag.


## BibTeX
```
@article{wu2023augment,
  title={Augment then Smooth: Reconciling Differential Privacy with Certified Robustness},
  author={Jiapeng Wu and Atiyeh Ashari Ghomi and David Glukhov and Jesse C. Cresswell and Franziska Boenisch and Nicolas Papernot},
  journal={Transactions on Machine Learning Research},
  url={https://openreview.net/forum?id=YN0IcnXqsr},
  year={2024}
}
```

