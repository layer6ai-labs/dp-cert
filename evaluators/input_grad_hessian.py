from functorch import make_functional, grad, vmap, jacrev

from utils import *


def compute_loss(data, target, params, model, criterion):
    data = data.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)
    pred = model(params, data)
    loss = criterion(pred, target)
    return loss

def adversarial_grad_term_batch(model, data, target, criterion):
    func_model, params = make_functional(model._module)
    grad_fn = grad(compute_loss)
    with torch.no_grad():
        grad_fn_per_sample = vmap(grad_fn, in_dims=(0, 0, None, None, None))
        grads_wrt_samples = grad_fn_per_sample(data, target, params, func_model, criterion)
    return torch.linalg.norm(grads_wrt_samples, dim=1)

def per_sample_input_grad_norm(model, data, target, criterion):
    func_model, params = make_functional(model._module)
    grad_fn = grad(compute_loss)
    with torch.no_grad():
        grad_fn_per_sample = vmap(grad_fn, in_dims=(0, 0, None, None, None))
        grads_wrt_samples = grad_fn_per_sample(data, target, params, func_model, criterion)
    return torch.norm(grads_wrt_samples, p=2, dim=(1,2,3))

def adversarial_hess_term_batch(model, data, target, criterion):
    func_model, params = make_functional(model._module)
    with torch.no_grad():
        hessian_fn = jacrev(jacrev(compute_loss, argnums=0), argnums=0)
        hessian_fn_sample = vmap(hessian_fn, in_dims=(0, 0, None, None, None))
        eigen_val_sample = []
        batch_size = 300
        for i in range(int(data.shape[0] / batch_size) + 1):
            upper_bnd = batch_size * (i + 1) if batch_size * (i + 1) <= data.shape[0] else data.shape[0]
            if (upper_bnd != batch_size * i):
                hess_wrt_samples = hessian_fn_sample(data[batch_size * i:upper_bnd], target[batch_size * i:upper_bnd], params,
                                                        func_model, criterion)
                if (hess_wrt_samples.dim() > 3):
                    datapoint_shape = data[0].reshape(-1).shape[0]
                    sample_num = batch_size if (i + 1) * batch_size <= data.shape[0] else data.shape[0] % batch_size
                    hess_wrt_samples = hess_wrt_samples.reshape(sample_num, datapoint_shape, datapoint_shape)
                for j in range(hess_wrt_samples.shape[0]):
                    eigen_val_sample.append(torch.lobpcg(hess_wrt_samples[j], 1, largest=True)[0].item())
            eigen_val_sample
    return torch.tensor(eigen_val_sample)