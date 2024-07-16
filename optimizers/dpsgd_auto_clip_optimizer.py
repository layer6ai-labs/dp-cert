import torch
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from opt_einsum.contract import contract

from optimizers.metric_collection_optimizer import OptimizerWithMetrics

"""
Implemention of DPSGD with automatic clipping. Sources:
Automatic Clipping: https://arxiv.org/pdf/2206.07136.pdf
Per Sample Adaptive Clipping: https://arxiv.org/pdf/2212.00328.pdf
"""

class DPSGD_Auto_Clip_Optimizer(OptimizerWithMetrics):
    def set_hyperparameters(self, gamma, psac):
        self.gamma = gamma
        self.psac = psac

    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """
        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros((0,))
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]

            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            if not self.psac:
                per_sample_clip_factor = self.max_grad_norm / (per_sample_norms + 0.01)
            else:
                per_sample_clip_factor = self.max_grad_norm / (per_sample_norms + (0.01 / (per_sample_norms) + 0.01))

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)


class DPSGD_Auto_Clip_Augmented_Data_Optimizer(DPSGD_Auto_Clip_Optimizer):
    def set_num_augmentations(self, num_augmentations):
        self.num_augmentations = num_augmentations

    def clip_and_accumulate(self):
        for p in self.params:
            p.grad_sample = torch.mean(p.grad_sample.view(-1, self.num_augmentations+1, *p.grad_sample.shape[1:]), 1)
        super().clip_and_accumulate()