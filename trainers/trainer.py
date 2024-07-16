import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functorch import make_functional, vjp, grad
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.distributions.normal import Normal
from tqdm import tqdm

from evaluators.input_grad_hessian import adversarial_grad_term_batch, adversarial_hess_term_batch
from metrics_collection import MetricsCollection
from utils import *


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for _, param in model.named_parameters():
        if hasattr(param, "freeze_grad") and param.freeze_grad:
            continue
        param.requires_grad=requires_grad



def consistency_loss(logits, lbd, eta=0.5, loss='default'):
    # adapted from https://github.com/jh-jeong/smoothing-consistency/blob/a11954aa91adb9924eecf49aaca74051e79bbb86/code/consistency.py#L5
    m = len(logits)
    softmax = [F.softmax(logit, dim=1) for logit in logits]
    avg_softmax = sum(softmax) / m

    loss_kl = [kl_div(logit, avg_softmax) for logit in logits]
    loss_kl = sum(loss_kl) / m

    if loss == 'default':
        loss_ent = entropy(avg_softmax)
        consistency = lbd * loss_kl + eta * loss_ent

    return consistency.mean()

def kl_div(logit, targets):
    return F.kl_div(F.log_softmax(logit, dim=1), targets, reduction='none').sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent

class BaseTrainer:
    """Base class for various training methods"""

    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 valid_loader,
                 test_loader,
                 writer,
                 evaluator,
                 device,
                 method="regular",
                 max_epochs=100,
                 physical_batch_size=1024,
                 lr=0.01,
                 seed=0,
                 evaluate_adversarial_loss=False,
                 clip=True,
                 add_noise=True
                 ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.writer = writer
        self.evaluator = evaluator
        self.device = device
        self.method = method

        self.physical_batch_size = physical_batch_size
        self.max_epochs = max_epochs
        self.num_batch = len(self.train_loader)
        self.epoch = 0
        self.num_layers = get_num_layers(self.model)

        self.lr = lr
        self.seed = seed
        self.evaluate_adversarial_loss = evaluate_adversarial_loss
        self.metric_collection = MetricsCollection()
        self.optimizer.set_clip_add_noise(clip, add_noise)

    def _train_epoch(self, exp_adv_loss, param_eigenvals, param_for_step=None):
        # methods: regular, dpsgd, dpsgd-global, dpsgd-global-adapt
        criterion = torch.nn.CrossEntropyLoss()

        with BatchMemoryManager(data_loader=self.train_loader, max_physical_batch_size=self.physical_batch_size, optimizer=self.optimizer) as memory_safe_data_loader:

            for _batch_idx, (data, target, image_idx) in enumerate(tqdm(memory_safe_data_loader)):
                batch_size = data.shape[0]
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()

                if self.method in ["dpsgd-adv-smooth", "dpsgd-auto-clip-adv-smooth"]:
                    max_norm = min(self.max_norm, (self.epoch + 1) * self.max_norm/self.warmup)
                    data, target = self.pgd_attack_with_noise(data, target, max_norm)
                
                elif self.method == "dpsgd-adv":
                    # implement DP with adversarial training:
                    max_norm = min(self.max_norm, (self.epoch + 1) * self.max_norm/self.warmup)
                    data = self.pgd_attack_no_noise(data, target, max_norm)

                elif self.method in ["dpsgd-augment", "regular-augment", "dpsgd-augment-auto-clip"]:
                    data = data.view(-1, *data.shape[2:])
                    target = target.view(-1, *target.shape[2:])

                outputs = self.model(data)
                loss = criterion(outputs, target)

                if hasattr(self, "trades") and self.trades:
                    """
                    Adapted from https://github.com/jh-jeong/smoothing-consistency/blob/a11954aa91adb9924eecf49aaca74051e79bbb86/code/train_stab.py#LL64C41-L64C41,
                    however we use KL divergence instead of cross entropy loss.
                    """

                    criterion_kl = nn.KLDivLoss()
                    logits = torch.chunk(outputs, self.num_augmentations + 1, dim=0)
                    augmentated_logits = logits[1:]
                    original_logits = logits[0]
                    loss_kl = [criterion_kl(F.log_softmax(augmentated_logit, dim=1), F.softmax(original_logits, dim=1)) for augmentated_logit in augmentated_logits]
                    loss_kl = sum(loss_kl) / len(augmentated_logits)
                    loss += loss_kl

                if hasattr(self, "consistency") and self.consistency:
                    logits = torch.chunk(outputs, self.num_augmentations + 1, dim=0)
                    loss_consistency = consistency_loss(logits[1:], 0.5)
                    loss += loss_consistency
                
                if hasattr(self, "macer") and self.macer:
                    """adapted from https://github.com/jh-jeong/smoothing-consistency/blob/a11954aa91adb9924eecf49aaca74051e79bbb86/code/third_party/macer.py#LL16C37-L16C37"""
                    m = Normal(torch.tensor([0.0]).to(self.device),
                                torch.tensor([1.0]).to(self.device))
                    gamma = 0.1
                    beta = 16.0
                    lbd = 16.0
                    
                    outputs = outputs.reshape((batch_size, self.num_augmentations + 1, -1))
                    single_target = target.reshape((batch_size, self.num_augmentations + 1))[:, 0]
                    beta_outputs = outputs * beta  # only apply beta to the robustness loss
                    beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
                    top2 = torch.topk(beta_outputs_softmax, 2)
                    top2_score = top2[0]
                    top2_idx = top2[1]
                    
                    indices_correct = (top2_idx[:, 0] == single_target)  # G_theta

                    out0, out1 = top2_score[indices_correct,
                                            0], top2_score[indices_correct, 1]
                    robustness_loss = m.icdf(out1) - m.icdf(out0)
                    indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
                        robustness_loss) & (torch.abs(robustness_loss) <= gamma)  # hinge
                    out0, out1 = out0[indices], out1[indices]
                    robustness_loss = m.icdf(out1) - m.icdf(out0) + gamma
                    robustness_loss = robustness_loss.sum() * self.augment_noise_std * lbd/ (2 * batch_size)
                    loss += robustness_loss

                loss.backward()
                true_per_sample_grad_norms = self.calculate_per_sample_norms()
                learning_rate = self.optimizer.param_groups[-1]['lr']
                if self.method == "dpsgd-global":
                    self.optimizer.step(self.strict_max_grad_norm)
                    clipping_bound = self.strict_max_grad_norm
                elif self.method == "dpsgd-global-adapt":
                    next_Z = self._update_Z(true_per_sample_grad_norms, self.strict_max_grad_norm)
                    self.optimizer.step(self.strict_max_grad_norm)
                    self.strict_max_grad_norm = next_Z
                    clipping_bound = self.strict_max_grad_norm
                else:
                    self.optimizer.step()
                    clipping_bound = self.optimizer.max_grad_norm
                
                if not self.optimizer._is_last_step_skipped:
                    true_batch_gradient = self.optimizer.true_batch_gradient
                    clipped_batch_gradient = self.optimizer.clipped_batch_gradient
                    noised_batch_gradient = self.optimizer.noised_batch_gradient
                    
                    self.metric_collection.add_and_log_batch_metrics(true_per_sample_grad_norms, true_batch_gradient,
                            clipped_batch_gradient, noised_batch_gradient, clipping_bound, loss, learning_rate)
                    self.metric_collection.gather_grad_norm_statistics(image_idx, true_per_sample_grad_norms)
                    if (self.evaluate_adversarial_loss and _batch_idx == self.num_batch - 1):
                        self.excessive_adversarial_loss_batch(data, target, criterion, exp_adv_loss, _batch_idx)
                        eigenval = self.param_eigenval(data, target, criterion)
                        param_eigenvals.append([self.epoch, _batch_idx, eigenval])

            if not self.method.startswith("regular"): 
                if self.method in ["dpsgd-global-adapt"]:
                    self._update_privacy_accountant()
                epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
                print(f"(ε = {epsilon:.2f}, δ = {self.delta})")

    def train(self, write_checkpoint=True):
        training_time = 0
        exp_adv_loss = []
        param_eigenvals = []
        epoch_metrics = []
        validation_metrics = defaultdict(list)
        self._validate(validation_metrics)
        while self.epoch < self.max_epochs:
            self.metric_collection.mark_epoch_start()
            epoch_start_time = time.time()
            self.model.train()
            self._train_epoch(exp_adv_loss, param_eigenvals)
            losses = self.metric_collection.get_current_epoch_loss()

            epoch_training_time = time.time() - epoch_start_time
            training_time += epoch_training_time

            print(
                f"Train Epoch: {self.epoch} \t"
                f"Loss: {np.mean(losses):.6f}"
            )

            self._validate(validation_metrics)
            self.writer.write_scalar("train/" + "Loss", np.mean(losses), self.epoch)
            if write_checkpoint: self.write_checkpoint("latest")
            self.epoch += 1

            if self.epoch == self.max_epochs:
                loss_dict = dict()
                loss_dict["final_loss"] = np.mean(losses)

            cur_epoch_metrics = self.metric_collection.get_epoch_metrics(self.epoch)
            # write epoch metrics to csv
            columns = ["epoch", "avg_train_loss", "avg_per_sample_grad_norms", "avg_true_grad_norm",
                       "avg_clipped_grad_norm",
                       "avg_noised_grad_norms", "avg_clipping_bounds", "avg_portion_clipped", "avg_learning_rate",
                       "avg_cos_true_clipped", "avg_cos_true_noised"]
            epoch_metrics.append(cur_epoch_metrics)
            self.create_csv(epoch_metrics, columns, "epoch_metrics")

            columns = ["epoch", "batch", "avg_grad_term", "max_grad_term",
                       "avg_hess_term", "max_hess_term"]
            self.create_csv(exp_adv_loss, columns, "expected_adversarial_loss_per_epochs")

            columns = ["epoch", "batch", "max_eigenval"]
            self.create_csv(param_eigenvals, columns, "params_max_eigenval_per_epochs")

        self.writer.write_json("per_sample_grad_norm_history", self.metric_collection.per_sample_grad_norm_history)
        self.writer.write_json("per_sample_grad_norm_per_iteration", self.metric_collection.per_sample_grad_norm_per_iteration)
        self.writer.write_json("large_grad_norm_images_info", self.metric_collection.large_grad_norm_images_info)
        self.writer.write_scalar("train/" + "avg_train_time_over_epoch",
                                 training_time / (self.max_epochs * 60))  # in minutes
        
        self._test()

    def create_csv(self, data, columns, title):
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(self.writer.logdir, f"{title}.csv"), index=False)

    def calculate_per_sample_norms(self):
        """
        Flatten the parameters of all layers in a model

        Returns:
            a tensor of shape num_samples in a batch * num_params
        """
        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in self.optimizer.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        return per_sample_norms

    def _validate(self, validation_metrics):
        valid_results = self.evaluator.validate()
        valid_results = {k: v[0] for k, v in valid_results.items()}
        for k, v in valid_results.items():
            validation_metrics[k].append(v)

        self.writer.print_dict("Validation", valid_results)


    def _test(self):
        test_results = self.evaluator.test()
        certifed_robustness_results = test_results.pop("certified_robustness", None)
        if certifed_robustness_results:
            columns = ["image_id", "sigma", "label", "prediction", "radius", "correct"]
            self.create_csv(certifed_robustness_results, columns, "certified_robustness")
        test_results = {k: v[0] for k, v in test_results.items()}
        self.writer.print_dict("Test", test_results)

    def write_checkpoint(self, tag):
        checkpoint = {
            "epoch": self.epoch,
            "module_state_dict": self.model.state_dict(),
            "opt_state_dict": self.optimizer.state_dict(),
        }

        self.writer.write_checkpoint(f"{tag}", checkpoint)

    def load_checkpoint(self, tag):
        checkpoint = self.writer.load_checkpoint(f"{tag}", self.device)

        self.epoch = checkpoint["epoch"] + 1  # start on the next epoch
        self.model.load_state_dict(checkpoint["module_state_dict"])
        self.optimizer.load_state_dict(checkpoint["opt_state_dict"])

        print(f"Loaded model checkpoint `{tag}' after epoch {self.epoch - 1}")

    def excessive_adversarial_loss_batch(self, data, target, criterion, exp_adv_loss, batch_idx):
        self.model.disable_hooks()
        grad_term = adversarial_grad_term_batch(self.model, data, target, criterion)
        grad_term_avg = torch.mean(grad_term).item()
        grad_term_max = torch.max(grad_term).item()
        hess_term = adversarial_hess_term_batch(self.model, data, target, criterion)
        exp_adv_loss.append([self.epoch, batch_idx, grad_term_avg, grad_term_max,
                             torch.mean(hess_term).item(), torch.max(hess_term).item()])
        self.model.enable_hooks()

    def param_eigenval(self, data, target, criterion, maxIter=50, tol=1e-3):
        """Uses power method to compute the max eigenvalue of the hessian matrix"""
        self.model.disable_hooks()

        def create_hvp_fn(data, target, func_model, params):
            def compute_loss(params):
                preds = func_model(params, data)
                loss = criterion(preds, target)
                return loss

            _, hvp_fn = vjp(grad(compute_loss), params)
            return hvp_fn

        with torch.no_grad():
            func_model, params = make_functional(self.model)
            hvp_fn = create_hvp_fn(data, target, func_model, params)
            self.optimizer.zero_grad()

            eigenvec = tuple(torch.rand(el.shape).to(device=self.device) for el in params)
            eigenvec_norm = torch.linalg.norm(torch.cat([torch.flatten(t) for t in eigenvec])).item()
            eigenvec = tuple(v / eigenvec_norm for v in eigenvec)
            eigenval = None
            for i in range(maxIter):
                eigenvec_tmp = hvp_fn(eigenvec)[0]
                flat_eigenvec_tmp = torch.cat([torch.flatten(t) for t in eigenvec_tmp])
                flat_eigenvec = torch.cat([torch.flatten(t) for t in eigenvec])
                eigenval_tmp = torch.dot(flat_eigenvec_tmp, flat_eigenvec) / torch.dot(flat_eigenvec, flat_eigenvec)
                if eigenval != None:
                    if abs(eigenval - eigenval_tmp) / abs(eigenval) < tol:
                        eigenval = eigenval_tmp
                        break
                eigenvec_tmp_norm = torch.linalg.norm(flat_eigenvec_tmp).item()
                eigenvec_tmp = tuple(v / eigenvec_tmp_norm for v in eigenvec_tmp)
                eigenvec = eigenvec_tmp
                eigenval = eigenval_tmp
        self.model.enable_hooks()
        return eigenval.item()


class RegularAugmentedDataTrainer(BaseTrainer):
    """Class for Regular training with Augmented Data"""

    def __init__(
            self,
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            num_augmentations=10,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )
        self.num_augmentations = num_augmentations


class DpsgdTrainer(BaseTrainer):
    """Class for DPSGD training"""

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta


class DpsgdGlobalTrainer(DpsgdTrainer):

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            strict_max_grad_norm=100,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=delta,
            **kwargs
        )
        self.strict_max_grad_norm = strict_max_grad_norm


class DpsgdGlobalAdaptiveTrainer(DpsgdGlobalTrainer):

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            strict_max_grad_norm=100,
            bits_noise_multiplier=10,
            lr_Z=0.01,
            threshold=1.0,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta,
            strict_max_grad_norm,
            **kwargs
        )
        self.bits_noise_multiplier = bits_noise_multiplier
        self.lr_Z = lr_Z
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []
        self.threshold = threshold

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many Sampled Gaussian Mechanism steps
        are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def _update_Z(self, true_per_sample_grad_norms, Z):
        # get the l2 norm of gradients of all parameters for each sample, in shape of (batch_size, )
        # l2_norm_grad_per_sample = torch.norm(per_sample_grads, p=2, dim=1)
        batch_size = len(true_per_sample_grad_norms)

        dt = 0  # sample count in a batch exceeding Z * threshold
        for i in range(batch_size):  # looping over batch
            if true_per_sample_grad_norms[i].item() > self.threshold * Z:
                dt += 1

        dt = dt * 1.0 / batch_size  # percentage of samples in a batch that's bigger than the threshold * Z
        noisy_dt = dt + torch.normal(0, self.bits_noise_multiplier, (1,)).item() * 1.0 / batch_size

        factor = math.exp(- self.lr_Z + noisy_dt)

        next_Z = Z * factor

        self.privacy_step_history.append([self.bits_noise_multiplier, self.sample_rate])
        return next_Z


class DPSGDAugmentedDataTrainer(DpsgdTrainer):
    """Class for DPSGD training with augmented data"""

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            num_augmentations=10,
            augment_noise_std=0.25,
            consistency=False,
            macer=False,
            trades=False,
            stability=False,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=delta,
            **kwargs
        )
        self.physical_batch_size = int(self.physical_batch_size / (num_augmentations + 1))
        self.optimizer.set_num_augmentations(num_augmentations)
        self.num_augmentations = num_augmentations
        self.augment_noise_std = augment_noise_std
        self.macer = macer
        self.consistency = consistency
        self.trades = trades
        self.stability = stability


class DpsgdAutoClipTrainer(DpsgdTrainer):
    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            gamma=0.01,
            psac=True,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=delta,
            **kwargs
        )
        optimizer.set_hyperparameters(
            gamma=gamma,
            psac=psac
        )

class DPSGDAdvTrainer(DpsgdTrainer):
    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            pgd_steps=8,
            max_norm=64,
            warmup=10,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=delta,
            **kwargs
        )
        self.pgd_steps = pgd_steps
        self.max_norm = max_norm / 256.0
        self.warmup = warmup
        
    def pgd_attack_no_noise(self, inputs, labels, max_norm):
        """
        Adapted from "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers"
        https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/attacks.py#L57
        This is also the attack used in the paper "Practical Adversarial Training with Differential Privacy for Deep Learning"
        https://openreview.net/pdf?id=1hw-h1C8bch
        """
        self.model.eval()
        requires_grad_(self.model, False)
        self.model.disable_hooks()
        batch_size = labels.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)
        # Setup optimizers
        optimizer = optim.SGD([delta], lr=max_norm/self.pgd_steps*2)

        for _ in range(self.pgd_steps):
            adv = inputs + delta
            logits = self.model(adv)
            loss = F.cross_entropy(logits, labels, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

            delta.data.renorm_(p=2, dim=0, maxnorm=max_norm)
        
        self.model.enable_hooks()
        self.model.train()
        requires_grad_(self.model, True)
        return inputs + delta
    

class DpsgdSmoothAdvTrainer(DPSGDAdvTrainer):
    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            pgd_steps=8,
            max_norm=64,
            num_augmentations=2,
            augment_noise_std=0.12,
            warmup=10,
            no_grad=False,
            include_original=True,
            consistency=False,
            trades=False,
            stability=False,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=delta,
            pgd_steps=pgd_steps,
            max_norm=max_norm,
            warmup=warmup,
            **kwargs
        )
        self.no_grad = no_grad
        self.num_augmentations = num_augmentations
        self.augment_noise_std = augment_noise_std
        self.physical_batch_size = int(self.physical_batch_size / (self.num_augmentations))
        self.include_original = include_original
        self.consistency = consistency
        self.trades = trades
        self.stability = stability
        if include_original:
            self.optimizer.set_num_augmentations(self.num_augmentations)
        else:
            self.optimizer.set_num_augmentations(self.num_augmentations - 1)

    def pgd_attack_with_noise(self, inputs, labels, max_norm):
        """
        Adapted from "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers"
        https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/attacks.py#L116
        """
        self.model.eval()
        requires_grad_(self.model, False)
        self.model.disable_hooks()
        batch_size = labels.shape[0]
        delta = torch.zeros((len(labels), *inputs.shape[1:]), requires_grad=True, device=self.device)

        if self.include_original:
            inputs = inputs.repeat((1, self.num_augmentations + 1, 1, 1)).view(-1, *inputs.shape[1:])
            original_inputs = inputs[:batch_size]
            inputs = inputs[batch_size:]
        else:

            inputs = inputs.repeat((1, self.num_augmentations, 1, 1)).view(-1, *inputs.shape[1:])
        noise = torch.randn_like(inputs, device=self.device) * self.augment_noise_std

        optimizer = optim.SGD([delta], lr=max_norm/self.pgd_steps*2)

        for _ in range(self.pgd_steps):

            adv = inputs + delta.repeat(1,self.num_augmentations,1,1).view_as(inputs)
            if noise is not None:
                adv = adv + noise
            logits = self.model(adv)

            # safe softamx
            softmax = F.softmax(logits, dim=1)
            # average the probabilities across noise
            average_softmax = softmax.reshape(-1, self.num_augmentations, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)
            logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
            loss = F.nll_loss(logsoftmax, labels)

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
       
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs[::self.num_augmentations])
            delta.data.clamp_(0, 1).sub_(inputs[::self.num_augmentations])

            delta.data.renorm_(p=2, dim=0, maxnorm=max_norm)

        delta = delta.repeat(1,self.num_augmentations,1,1).view_as(inputs) + noise
        if self.include_original:
            labels = labels.unsqueeze(1).repeat(1, self.num_augmentations + 1).reshape(-1,1).squeeze()
            augmented_input = torch.cat([original_inputs, inputs + delta], dim=0)
        else:
            labels = labels.unsqueeze(1).repeat(1, self.num_augmentations).reshape(-1,1).squeeze()
            augmented_input = inputs + delta

        self.model.enable_hooks()
        self.model.train()
        requires_grad_(self.model, True)
        return augmented_input, labels

class DpsgdAutoClipSmoothAdvTrainer(DpsgdSmoothAdvTrainer):
    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            pgd_steps=8,
            max_norm=64,
            num_augmentations=2,
            augment_noise_std=0.12,
            warmup=10,
            gamma=0.01,
            psac=True,
            no_grad=False,
            consistency=False,
            trades=False,
            stability=False,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=delta,
            pgd_steps=pgd_steps,
            max_norm=max_norm,
            num_augmentations=num_augmentations,
            augment_noise_std=augment_noise_std,
            warmup=warmup,
            no_grad=no_grad,
            consistency=consistency,
            trades=trades,
            stability=stability,
            **kwargs
        )
        optimizer.set_hyperparameters(
            gamma=gamma,
            psac=psac
        )

class DpsgdAutoClipAugmentTrainer(DPSGDAugmentedDataTrainer):
    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            num_augmentations=2,
            augment_noise_std=0.12,
            gamma=0.01,
            psac=True,
            consistency=False,
            trades=False,
            stability=False,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=delta,
            num_augmentations=num_augmentations,
            augment_noise_std=augment_noise_std,
            consistency=consistency,
            trades=trades,
            stability=stability,
            **kwargs
        )
        
        optimizer.set_hyperparameters(
            gamma=gamma,
            psac=psac
        )
