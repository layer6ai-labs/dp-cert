import torch

from optimizers.metric_collection_optimizer import OptimizerWithMetrics


class DPSGD_Augmented_Data_Optimizer(OptimizerWithMetrics):
    def set_num_augmentations(self, num_augmentations):
        self.num_augmentations = num_augmentations

    def clip_and_accumulate(self):
        """
        Performs gradient clipping using Augmentation Multiplicity
        proposed by https://arxiv.org/abs/2204.13650```
        """
        for p in self.params:
            p.grad_sample = torch.mean(p.grad_sample.view(-1, self.num_augmentations + 1, *p.grad_sample.shape[1:]), 1)
        super().clip_and_accumulate()
    # note add_noise does not have to be modified since max_grad_norm = C is sensitivity
