from optimizers.metric_collection_optimizer import OptimizerWithMetrics


class Regular_Optimizer(OptimizerWithMetrics):
    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.

        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad = p.grad_sample.mean(0) / self.accumulated_iterations
