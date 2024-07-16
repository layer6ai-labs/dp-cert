import math
import os

import matplotlib.pyplot as plt
import torch
from opacus.accountants import create_accountant
from opacus.accountants.analysis.gdp import compute_eps_poisson
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


def verify_format(path):
    # Remove trailing slash so split will work
    if path[-1] == '/':
        path = path[:-1]

    head_tail = os.path.split(path)
    if head_tail[0] == None or head_tail[0] == "":  # only one directory depth provided
        print("Using default logdir_root as 'runs'")
        path = os.path.join('runs', path)

    return path


def privacy_checker(sample_rate, cfg):
    assert sample_rate <= 1.0
    steps = cfg["max_epochs"] * math.ceil(1 / sample_rate)

    if cfg["accountant"] == 'rdp':
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=cfg["noise_multiplier"],
            steps=steps,
            orders=orders)
        epsilon, alpha = get_privacy_spent(
            orders=orders,
            rdp=rdp,
            delta=cfg["delta"])
        print(
            "-----------privacy------------"
            f"\nDP-SGD (RDP) with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {cfg['noise_multiplier']},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {epsilon:.3g},"
            f"\n\tdelta = {cfg['delta']}."
            f"\nThe optimal alpha is {alpha}."
        )
    elif cfg["accountant"] == 'gdp':
        eps = compute_eps_poisson(
            steps=steps,
            noise_multiplier=cfg["noise_multiplier"],
            sample_rate=sample_rate,
            delta=cfg["delta"],
        )
        print(
            "-----------privacy------------"
            f"\nDP-SGD (GDP) with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {cfg['noise_multiplier']},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {eps:.3g},"
            f"\n\tdelta = {cfg['delta']}."
        )
    elif cfg["accountant"] == 'prv':
        accountant = create_accountant("prv")
        accountant.history = [(cfg["noise_multiplier"], sample_rate, steps)]
        eps = accountant.get_epsilon(delta=cfg["delta"])
        print(
            "-----------privacy------------"
            f"\nDP-SGD (PRV) with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {cfg['noise_multiplier']},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {eps:.3g},"
            f"\n\tdelta = {cfg['delta']}."
        )
    else:
        raise ValueError(f"Unknown accountant {cfg['accountant']}. Try 'rdp', 'prv', or 'gdp'.")


def get_num_layers(model):
    num_layers = 0
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            num_layers += 1

    return num_layers


# splits data, labels according to group of data point
# returns tensor of size num_groups, each element is (subset of data, subset of labels)  
# corresponding to specific group given by index
def split_by_group(data, labels, group, num_groups, return_counts=False):
    sorter = torch.argsort(group)
    unique, counts = torch.unique(group, return_counts=True)
    unique = unique.tolist()
    counts = counts.tolist()

    complete_unique = [0] * num_groups
    complete_counts = [0] * num_groups
    for i in range(num_groups):
        complete_unique[i] = i
        if i in unique:
            j = unique.index(i)
            complete_counts[i] = counts[j]

    sorted_data = torch.split(data[sorter], complete_counts)
    sorted_labels = torch.split(labels[sorter], complete_counts)

    if not return_counts:
        return list(zip(sorted_data, sorted_labels))
    return list(zip(sorted_data, sorted_labels)), complete_counts


def plot_by_group(data_by_group, writer, data_title=None, scale_to_01=False):
    fig = plt.figure()
    plt.bar(range(len(data_by_group)), data_by_group, width=0.9)
    plt.xlabel("Groups")
    if data_title is not None:
        plt.ylabel(data_title)
    plt.title(data_title)
    plt.xticks(range(len(data_by_group)))
    if scale_to_01:
        plt.ylim(0, 1)
    writer.write_figure(data_title, fig)


# type of images to view, all, correctly classified, ...
def view_adv_images(clean, adversarial, labels, pred, file_name, logdir, n=10):
    fig, axs = plt.subplots(n, int(n / 3.3), figsize=(5, n * 1.2), dpi=100)
    for image_idx in range(n):
        orig_image = clean[image_idx][0]
        adv_image = adversarial[image_idx][0]
        perturbation = adv_image - orig_image
        axs[image_idx, 0].imshow(orig_image.cpu(), cmap="gray")  # avoid error when running on GPU
        axs[image_idx, 1].imshow(adv_image.cpu(), cmap="gray")
        axs[image_idx, 2].imshow(perturbation.cpu(), cmap="gray")
        axs[image_idx, 0].set_title(f"Original, {labels[image_idx]}")
        axs[image_idx, 1].set_title(f"Adversarial, {pred[image_idx]}")
        axs[image_idx, 2].set_title("Perturbation")
    path = os.path.join(logdir, "adv_images")
    if not os.path.exists(path): os.makedirs(os.path.join(path))
    path = os.path.join(path, f"{file_name}_adv_images.png")
    fig.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()
