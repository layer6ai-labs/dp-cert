import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.grad_sample.gsm_exp_weights import GradSampleModuleExpandedWeights

from evaluators.adversarial_factory import create_attack
from evaluators.input_grad_hessian import per_sample_input_grad_norm, adversarial_hess_term_batch
from evaluators.randomized_smoothing import Smooth
from utils import view_adv_images

dflt_atk_norm = {}

def estimate_local_lip_v2(model, dataloader, batch_size=128, perturb_steps=10, 
                          step_size=0.003, epsilon=0.01, device="cuda", **kwargs):
    # adapted from https://github.com/yangarbiter/robust-local-lipschitz/blob/master/lolip/utils.py#L32
    model.disable_hooks()
    model.eval()

    local_lip_dict = defaultdict(dict)
    total_loss = 0.
    for _batch_idx, (data, labels, image_idx) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)
        # generate adversarial example
        x_adv = data + 0.001 * torch.randn(data.shape).to(device)

        # Setup optimizers
        optimizer = optim.SGD([x_adv], lr=step_size)

        for _ in range(perturb_steps):
            x_adv.requires_grad_(True)
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = (-1) * local_lip(model, data, x_adv)
            loss.backward()
            # renorming gradient
            eta = step_size * x_adv.grad.data.sign().detach()
            x_adv = x_adv.data.detach() + eta.detach()
            eta = torch.clamp(x_adv.data - data.data, -epsilon, epsilon)
            x_adv = data.data.detach() + eta.detach()
            x_adv = torch.clamp(x_adv, 0, 1.0)

        cur_lip_constants = local_lip(model, data, x_adv, reduction=None)
        for j, idx in enumerate(image_idx):
            idx = idx.item()
            local_lip_dict[idx]['local_lip'] = cur_lip_constants[j].item()

        total_loss += torch.sum(cur_lip_constants).item()
    ret_v = total_loss / len(dataloader.dataset)
    model.enable_hooks()
    return ret_v, local_lip_dict

def local_lip(model, x, xp, top_norm=1, reduction='mean'):
    # Adapted from https://github.com/yangarbiter/robust-local-lipschitz/blob/master/lolip/utils.py#L10#
    model.eval()
    down = torch.flatten(x - xp, start_dim=1)
    if top_norm == "kl":
        criterion_kl = nn.KLDivLoss(reduction='none')
        top = criterion_kl(F.log_softmax(model(xp), dim=1),
                           F.softmax(model(x), dim=1))
        ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=np.inf)
    else:
        top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
        ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=np.inf)

    if reduction == 'mean':
        return torch.mean(ret)
    elif reduction == 'sum':
        return torch.sum(ret)
    elif reduction == None:
        return ret
    else:
        raise ValueError(f"Not supported reduction: {reduction}")

def robustness_succ(model, dataloader, gammas_linf=[0.15], gammas_l2=[0.5], adversarial_attacks=[], logdir=None,
                    adv_attack_params={}, view_sample_images=False, **kwargs):
    # This is very important for the compatibility of torchattack and opacus
    model.disable_hooks()
    # Here we cannot use torch.no_grad() because adv_data = atk(data, labels) requires gradient updates to input
    device = model._module.device if isinstance(model,
                                                (GradSampleModule, GradSampleModuleExpandedWeights)) else model.device
    adv_robustness_on_corr = {}
    robustness_dict = defaultdict(dict)

    for name in adversarial_attacks:
        print(f"evaluating adv attack {name}")
        fmodel, atk, norm = create_attack(model, name, adv_attack_params)

        if norm == "l2":
            gammas = gammas_l2
        else:
            gammas = gammas_linf  # if norm isnt defined or linf

        correct_on_corr = torch.zeros_like(torch.tensor(gammas))
        total_on_corr = 0
        # norm of perturbation for p=2,infty
        l2_norm_all = [[] for _ in gammas]
        linfty_norm_all = [[] for _ in gammas]
        # where f(x + adv) != f(x) = y
        l2_norm_miss = [[] for _ in gammas]
        linfty_norm_miss = [[] for _ in gammas]

        for _batch_idx, (data, labels, image_idx) in enumerate(dataloader):
            _, predicted_benign = torch.max(model(data), 1)

            # on subset of batch that model correctly predicts (bening)
            data_corr = data[predicted_benign == labels]
            labels_corr = labels[predicted_benign == labels]
            image_idx_corr = image_idx[predicted_benign == labels]

            if data_corr.shape[0] == 0:
                continue

            # create attack images
            _, adv_data, adv_success = atk(fmodel, data_corr, labels_corr, epsilons=gammas)
            labels_corr, adv_data, adv_success = labels_corr.to(device), [gamma.to(device) for gamma in adv_data], [
                gamma.to(device) for gamma in adv_success]
            # success of attack = 1 - accuracy of model on adv examples

            for i, index in enumerate(image_idx_corr):
                index = index.item()
                for gamma_id, gamma in enumerate(gammas):
                    robustness_dict[index][f"{name}_{gamma}"] = adv_success[gamma_id][i].item() # the one indicates it is an adversarial example
                    
            total_on_corr += labels_corr.size(0)
            correct_on_corr += torch.tensor([gamma.sum().item() for gamma in adv_success])

            for i in range(len(gammas)):
                l2_norm_all[i].extend([torch.linalg.vector_norm(x, ord=2).item() for x in (data_corr - adv_data[i])])
                linfty_norm_all[i].extend(
                    [torch.linalg.vector_norm(x, ord=float('inf')).item() for x in (data_corr - adv_data[i])])
                l2_norm_miss[i].extend(
                    [torch.linalg.vector_norm(x, ord=2).item() for x in (data_corr - adv_data[i])[adv_success[i]]])
                linfty_norm_miss[i].extend([torch.linalg.vector_norm(x, ord=float('inf')).item() for x in
                                            (data_corr - adv_data[i])[adv_success[i]]])

            if view_sample_images and _batch_idx == 0:
                # TODO: check data_corr isnt empty
                # TODO: visualize only successful adv examples
                for i, gamma in enumerate(gammas):
                    n = 10
                    outputs = model(adv_data[i][0:n])
                    _, predicted_adv = torch.max(outputs, 1)
                    view_adv_images(data_corr, adv_data[i][0:n], labels_corr, predicted_adv, f"{name}_{gamma}", logdir,
                                    n=n)

        adv_robustness_on_corr[name] = [(correct_on_corr[i] / total_on_corr).item() for i in range(len(gammas))]
        # l2 and linfty norm of perturbations (where f(x) = y)
        avg_perturb_2norm = [round(torch.mean(torch.tensor(l2_norm_all[i])).item(), 5) for i in range(len(gammas))]
        avg_perturb_inftynorm = [round(torch.mean(torch.tensor(linfty_norm_all[i])).item(), 5) for i in
                                 range(len(gammas))]
        # l2 and linfty norm of perturbations (where f(x+fdelta) =/= f(x) = y)
        avg_perturb_2norm_miss = [round(torch.mean(torch.tensor(l2_norm_miss[i])).item(), 5) for i in
                                  range(len(gammas))]
        avg_perturb_inftynorm_miss = [round(torch.mean(torch.tensor(linfty_norm_miss[i])).item(), 5) for i in
                                      range(len(gammas))]

        print(
            f"adv robust succ on {name}: {adv_robustness_on_corr[name]}, l2 norm: {avg_perturb_2norm}, linfty norm: {avg_perturb_inftynorm}, l2 norm miss: {avg_perturb_2norm_miss}, linfty norm miss: {avg_perturb_inftynorm_miss}"
        )

    # Don't forget to turn them back on
    model.enable_hooks()
    # the tensorflow writer requires scalar
    return (
        [adv_robustness_on_corr[key][-1] for key in adversarial_attacks],
        robustness_dict
    )  # NOTE: note this is only for last gamma, do not use

# num of iterations until adv example (PGD)
# averaged over batch


def accuracy(model, dataloader, **kwargs):
    correct = 0
    total = 0
    pred_label_dict = defaultdict(dict)
    with torch.no_grad():
        device = model._module.device if isinstance(model, (
            GradSampleModule, GradSampleModuleExpandedWeights)) else model.device
        for _batch_idx, (data, labels, image_idx) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            for i, idx in enumerate(image_idx):
                idx = idx.item()
                pred_label_dict[idx]['pred'] = predicted[i].item()
                pred_label_dict[idx]['label'] = labels[i].item()
                pred_label_dict[idx]['correct'] = (predicted[i] == labels[i]).item()
                pred_label_dict[idx]['logit_norm'] = torch.linalg.vector_norm(outputs[i], ord=2).item()
                pred_label_dict[idx]['prob'] = torch.nn.functional.softmax(outputs[i], dim=0).tolist()
                
    return (correct / total).item(), pred_label_dict


def macro_accuracy(model, dataloader, num_classes=None, **kwargs):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        device = model._module.device if isinstance(model, (
            GradSampleModule, GradSampleModuleExpandedWeights)) else model.device
        for _batch_idx, (data, labels, image_idx) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            for true_p, all_p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[true_p.long(), all_p.long()] += 1

    accs = confusion_matrix.diag() / confusion_matrix.sum(1)
    return accs.mean().item()

def certified_robustness(model, dataloader, num_classes, **kwargs):
    cert_robustness_dict = defaultdict(dict)
    results = []
    for sigma in kwargs['certified_noise_std']:
        smoothed_classifier = Smooth(model, num_classes, sigma, kwargs['device'])
        data, labels, image_idx = next(iter(dataloader))
        before_time = time.time()
        for data, labels, image_idx in dataloader:
            for (data_point, label, image_id) in zip(data, labels, image_idx):
                image_id = image_id.item()
                prediction, radius = smoothed_classifier.certify(data_point, kwargs['certified_n0'], kwargs['certified_n'],
                                                                kwargs['certified_alpha'], 
                                                                4000
                                                                # data.shape[0]
                                                                )
                correct = int(prediction == label)
                results.append([image_id, sigma, label.item(), prediction, radius, correct])
                cert_robustness_dict[image_id][f'certified_pred_{sigma}'] = prediction
                cert_robustness_dict[image_id][f'certified_radius_{sigma}'] = radius
                cert_robustness_dict[image_id][f'certified_correct_{sigma}'] = correct
            after_time = time.time()
            time_elapsed = str(after_time - before_time)
            print(f"Seconds required to certify %d datapoints: " % len(data) + time_elapsed)
    return results, cert_robustness_dict

def per_sample_loss_input_grad_norm(model, dataloader, **kwargs):
    grad_dict = defaultdict(dict)
    criterion = nn.CrossEntropyLoss()
    per_sample_loss_func = nn.CrossEntropyLoss(reduction='none')
    model.disable_hooks()
    with torch.no_grad():
        device = model._module.device if isinstance(model, (
            GradSampleModule, GradSampleModuleExpandedWeights)) else model.device
        for _batch_idx, (data, labels, image_idx) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss_each = per_sample_loss_func(outputs, labels)
            grads_wrt_samples = per_sample_input_grad_norm(model, data, labels, criterion)
            for i, idx in enumerate(image_idx):
                idx = idx.item()
                grad_dict[idx]['loss'] = loss_each[i].item()
                grad_dict[idx]['input_grad_norm'] = grads_wrt_samples[i].item()
                # input gradient norm divided by the norm of the per-sample output probabilities minus the respective one-hot labels
                one_hot_label = torch.zeros_like(outputs[i])
                one_hot_label[labels[i]] = 1
                grad_dict[idx]['normlized_input_grad_norm'] = \
                    grads_wrt_samples[i].item() / torch.linalg.vector_norm(outputs[i] - one_hot_label, ord=2).item()
                
    model.enable_hooks()
    return None, grad_dict

def hessian_eigen_value(model, dataloader, **kwargs):
    model.disable_hooks()
    hessian_dict = defaultdict(dict)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        device = model._module.device if isinstance(model, (
            GradSampleModule, GradSampleModuleExpandedWeights)) else model.device
        for _batch_idx, (data, labels, image_idx) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            eigen_vals = adversarial_hess_term_batch(model, data, labels, criterion)
            for i, idx in enumerate(image_idx):
                hessian_dict[idx.item()]["hess_eigen_val"] = eigen_vals[i].item()
    model.enable_hooks()
    return None, hessian_dict
