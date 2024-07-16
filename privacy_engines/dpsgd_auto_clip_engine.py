from typing import List, Union

from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from torch import optim

from optimizers.dpsgd_auto_clip_optimizer import DPSGD_Auto_Clip_Optimizer, DPSGD_Auto_Clip_Augmented_Data_Optimizer


class DPSGDAutoClipPrivacyEngine(PrivacyEngine):
    """
     This class defines the customized privacy engine for DPSGD-Auto-Clip.
     Specifically, it overwrites the _prepare_optimizer() method from parent class to return DPSGD_Auto_Clip_Optimizer
     """

    def _prepare_optimizer(
            self,
            optimizer: optim.Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: Union[float, List[float]],
            expected_batch_size: int,
            loss_reduction: str = "mean",
            distributed: bool = False,  # deprecated for this method
            clipping: str = "flat",  # deprecated for this method
            noise_generator=None,
            grad_sample_mode="hooks",
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optimizer = DPSGD_Auto_Clip_Optimizer(optimizer=optimizer,
                                           noise_multiplier=noise_multiplier,
                                           max_grad_norm=max_grad_norm,
                                           expected_batch_size=expected_batch_size,
                                           loss_reduction=loss_reduction,
                                           generator=generator,
                                           secure_mode=self.secure_mode)

        return optimizer

class DPSGDAutoClipAugmentatedDataPrivacyEngine(PrivacyEngine):
    """
     This class defines the customized privacy engine for DPSGD-Auto-Clip.
     Specifically, it overwrites the _prepare_optimizer() method from parent class to return DPSGD_Auto_Clip_Optimizer
     """

    def _prepare_optimizer(
            self,
            optimizer: optim.Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: Union[float, List[float]],
            expected_batch_size: int,
            loss_reduction: str = "mean",
            distributed: bool = False,  # deprecated for this method
            clipping: str = "flat",  # deprecated for this method
            noise_generator=None,
            grad_sample_mode="hooks",
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optimizer = DPSGD_Auto_Clip_Augmented_Data_Optimizer(optimizer=optimizer,
                                           noise_multiplier=noise_multiplier,
                                           max_grad_norm=max_grad_norm,
                                           expected_batch_size=expected_batch_size,
                                           loss_reduction=loss_reduction,
                                           generator=generator,
                                           secure_mode=self.secure_mode)

        return optimizer
    