from typing import Callable, Optional

import torch
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed


class OptimizerWithMetrics(DPOptimizer):
    clip = True
    add_noise_flag = True

    def set_clip_add_noise(self, clip, add_noise):
        self.clip = clip
        self.add_noise_flag = add_noise
        
    def update_true_batch_gradient(self):
        true_per_sample_gradient = torch.cat([torch.flatten(p.grad_sample, 1, -1) for p in self.params], dim=1)
        self.true_batch_gradient = torch.sum(true_per_sample_gradient, dim=0) / (self.expected_batch_size * self.accumulated_iterations)

    def update_clipped_batch_gradient(self):
        clipped_gradient = [torch.flatten(p.summed_grad) for p in self.params]
        self.clipped_batch_gradient = torch.cat(clipped_gradient) / (self.expected_batch_size * self.accumulated_iterations)

    def update_noised_batch_gradient(self):
        noised_batch_gradient = [torch.flatten(p.grad) for p in self.params]
        self.noised_batch_gradient = torch.cat(noised_batch_gradient)

    def no_clip_func(self):
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            p.summed_grad = torch.flatten(p.grad_sample, 1, -1).mean(dim=0)
            _mark_as_processed(p.grad_sample)

    def no_noise_func(self):
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            p.grad = (p.summed_grad).view_as(p.grad)
            _mark_as_processed(p.summed_grad)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.update_true_batch_gradient()
        if self.clip:
            self.clip_and_accumulate()
        else:
            self.no_clip_func()
        self.update_clipped_batch_gradient()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        if self.add_noise_flag:
            self.add_noise()
        else:
            self.no_noise_func()
            
        self.scale_grad()
        self.update_noised_batch_gradient()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True
    