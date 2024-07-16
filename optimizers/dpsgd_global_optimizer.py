from typing import Optional, Callable

import torch
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed

from optimizers.metric_collection_optimizer import OptimizerWithMetrics


class DPSGD_Global_Optimizer(OptimizerWithMetrics):
    def clip_and_accumulate(self, strict_max_grad_norm):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        # C = max_grad_norm
        # Z = strict_max_grad_norm
        # condition is equivalent to norm[i] <= Z
        # when condition holds, scale gradient by C/Z
        # otherwise, clip to 0
        per_sample_global_clip_factor = torch.where(per_sample_clip_factor >= self.max_grad_norm / strict_max_grad_norm,
                                                    # scale by C/Z
                                                    torch.ones_like(
                                                        per_sample_clip_factor) * self.max_grad_norm / strict_max_grad_norm,
                                                    torch.zeros_like(per_sample_clip_factor))  # clip to 0
        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = self._get_flat_grad_sample(p)

            # refer to lines 197-199 in 
            # https://github.com/pytorch/opacus/blob/ee6867e6364781e67529664261243c16c3046b0b/opacus/per_sample_gradient_clip.py
            # as well as https://github.com/woodyx218/opacus_global_clipping README
            grad = torch.einsum("i,i...", per_sample_global_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    # note add_noise does not have to be modified since max_grad_norm = C is sensitivity

    def pre_step(
            self, strict_max_grad_norm, closure: Optional[Callable[[], float]] = None
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
            self.clip_and_accumulate(strict_max_grad_norm)
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

    def step(self, strict_max_grad_norm, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step(strict_max_grad_norm):
            return self.original_optimizer.step()
        else:
            return None
