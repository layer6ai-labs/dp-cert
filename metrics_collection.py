from collections import defaultdict

import numpy as np
import torch


class MetricsCollection:
    avg_per_sample_grad_norms = []
    max_per_sample_grad_norms = []
    min_per_sample_grad_norms = []
    true_batch_grad_norms = []
    clipped_batch_grad_norms = []
    noised_batch_grad_norms = []
    batch_losses = []
    clipping_bounds = []
    portion_clipped = []
    learning_rates = []
    cos_true_clipped_list = []
    cos_true_noised_list = []
    large_grad_norm_images_info = defaultdict(list)
    per_sample_grad_norm_history = defaultdict(list)
    per_sample_grad_norm_per_iteration = []

    def add_and_log_batch_metrics(self, true_per_sample_grad_norms, true_batch_gradient, clipped_batch_gradient, noised_batch_gradient, clipping_bound, loss, learning_rate):
        true_batch_grad_norm = torch.linalg.norm(true_batch_gradient).item()
        clipped_batch_grad_norm = torch.linalg.norm(clipped_batch_gradient).item()
        noised_batch_grad_norm = torch.linalg.norm(noised_batch_gradient).item()
        cos_true_clipped = torch.nn.functional.cosine_similarity(clipped_batch_gradient, true_batch_gradient, dim=0).item()
        cos_true_noised = torch.nn.functional.cosine_similarity(clipped_batch_gradient, noised_batch_gradient, dim=0).item()
        self.avg_per_sample_grad_norms.append(torch.mean(true_per_sample_grad_norms).item())
        self.max_per_sample_grad_norms.append(torch.max(true_per_sample_grad_norms).item())
        self.min_per_sample_grad_norms.append(torch.min(true_per_sample_grad_norms).item())
        self.true_batch_grad_norms.append(true_batch_grad_norm)
        self.clipped_batch_grad_norms.append(clipped_batch_grad_norm)
        self.noised_batch_grad_norms.append(noised_batch_grad_norm)
        self.clipping_bounds.append(clipping_bound)
        self.batch_losses.append(loss.item())
        self.portion_clipped.append(torch.sum(true_per_sample_grad_norms > clipping_bound).item() / len(true_per_sample_grad_norms))
        self.learning_rates.append(learning_rate)
        self.cos_true_clipped_list.append(cos_true_clipped)
        self.cos_true_noised_list.append(cos_true_noised)


    def gather_grad_norm_statistics(self, image_idx, true_per_sample_grad_norms):
        for image_id, grad_norm in zip(image_idx, true_per_sample_grad_norms):
            self.per_sample_grad_norm_history[image_id.item()].append(grad_norm.item())
        
    def mark_epoch_start(self):
        self.cur_epoch_start_id = len(self.avg_per_sample_grad_norms)

    def get_current_epoch_loss(self):
        return self.batch_losses[self.cur_epoch_start_id:]

    def get_epoch_metrics(self, cur_epoch_num):
        epoch_end_idx = len(self.avg_per_sample_grad_norms)
        cur_slice = slice(self.cur_epoch_start_id, epoch_end_idx)
        avg_loss_epoch = np.mean(self.batch_losses[cur_slice])
        avg_per_sample_grad_norms_epoch = np.mean(self.avg_per_sample_grad_norms[cur_slice])
        avg_true_grad_norm_epoch = np.mean(self.true_batch_grad_norms[cur_slice])
        avg_clipped_grad_norm_epoch = np.mean(self.clipped_batch_grad_norms[cur_slice])
        avg_noised_grad_norms_epoch = np.mean(self.noised_batch_grad_norms[cur_slice])
        avg_clipping_bounds = np.mean(self.clipping_bounds[cur_slice])
        avg_portion_clipped = np.mean(self.portion_clipped[cur_slice])
        avg_learning_rate = np.mean(self.learning_rates[cur_slice])
        avg_cos_true_clipped = np.mean(self.cos_true_clipped_list[cur_slice])
        avg_cos_true_noised = np.mean(self.cos_true_noised_list[cur_slice])
        cur_epoch_metrics = [cur_epoch_num, avg_loss_epoch, avg_per_sample_grad_norms_epoch, avg_true_grad_norm_epoch, 
                avg_clipped_grad_norm_epoch, avg_noised_grad_norms_epoch, avg_clipping_bounds, 
                avg_portion_clipped, avg_learning_rate, avg_cos_true_clipped, avg_cos_true_noised]
        return cur_epoch_metrics
