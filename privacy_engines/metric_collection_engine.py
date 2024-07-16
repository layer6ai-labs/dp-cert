
from typing import List, Union

from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from torch import optim

from optimizers.metric_collection_optimizer import OptimizerWithMetrics


class PrivacyEngineWithMetrics(PrivacyEngine):
    """
    This class defines the customized privacy engine for DPSGD.
    Specifically, it overwrites the _prepare_optimizer() method from parent class to return PrivacyEngineWithMetrics
    """
    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
    ) -> OptimizerWithMetrics:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator


        return OptimizerWithMetrics(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
        )