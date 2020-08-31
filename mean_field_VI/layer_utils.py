import numpy as np
import torch
import torchvision

'''
    models utils: samplings & densities
'''

def sample_normal(mean, variance, num_samples=1, squeeze=False):
    noise = to_gpu(torch.nn.init.normal_(torch.FloatTensor(num_samples, *mean.shape)))
    samples = torch.sqrt(variance + 1e-6) * noise + mean

    if squeeze and num_samples == 1:
        samples = torch.squeeze(samples, dim=0)
    return samples

def gaussian_kl_diag(mean1, variance1, mean2, variance2):
    return -0.5 * torch.sum(1 + torch.log(variance1) - torch.log(variance2) - variance1/variance2
                            - ((mean1-mean2)**2)/variance2, dim=-1)

def gaussian_log_density(inputs, mean, variance=1):
    d = inputs.shape[-1]
    xc = inputs - mean
    return -0.5 * (torch.sum((xc **2) / variance, dim=-1) + torch.sum(torch.log(variance), dim=-1))

'''
    tensor monipulation
'''

def _squeeze(x):
    return x.view(-1, x.shape[-2]*x.shape[-1])

'''
    GPU wrappers
'''

def set_gpu_mode(mode):
    global _use_gpu
    _use_gpu = mode
    if mode:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

def gpu_enabled():
    return _use_gpu

def to_gpu(*args):
    if _use_gpu:
        return[arg.cuda() for arg in args] if len(args) > 1 else args[0].cuda()
    else:
        return args if len(args) > 1 else args[0]
