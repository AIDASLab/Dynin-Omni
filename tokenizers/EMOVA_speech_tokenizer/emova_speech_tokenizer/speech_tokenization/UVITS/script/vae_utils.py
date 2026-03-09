"""
Author:zhengnianzu
From repo: https://github.com/divymurli/VAEs
Time: 2022.10.31
Place: Shenzhen
"""

import torch
import numpy as np
import torch.nn.functional as F


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Sample z
    ################################################################################

    ################################################################################
    # End of code modification
    ################################################################################
    sample = torch.randn(m.shape).to(v.device)

    z = m + (v ** 0.5) * sample
    return z


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch, ..., dim): Observation
        m: tensor: (batch, ..., dim): Mean
        v: tensor: (batch, ..., dim): Variance

    Return:
        kl: tensor: (batch1, batch2, ...): log probability of each sample. Note
            that the summation dimension (dim=-1) is not kept
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute element-wise log probability of normal and remember to sum over
    # the last dimension
    ################################################################################
    # print("q_m", m.size())
    # print("q_v", v.size())
    const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
    # print(const.size())
    log_det = -0.5 * torch.sum(torch.log(v), dim=-1)
    # print("log_det", log_det.size())
    log_exp = -0.5 * torch.sum((x - m) ** 2 / v, dim=-1)

    log_prob = const + log_det + log_exp

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def log_normal_mixture(z, m, v, w=None):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute the uniformly-weighted mixture of Gaussians density for each sample
    # in the batch
    ################################################################################
    z = z.unsqueeze(1)
    log_probs = log_normal(z, m, v)
    # print("log_probs_mix", log_probs.shape)

    if w is not None:
        log_prob = log_weighted_sum_exp(log_probs, w, 1)
    else:
        log_prob = log_mean_exp(log_probs, 1)

    # print("log_prob_mix", log_prob.size())

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:z
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    # print("log var1", qv)
    return kl


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_weighted_sum_exp(x, w, dim=-1):
    """
      compute the log(weighted sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().mul(w).sum(dim)).log()


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def gaussian_mixture_parameters(buffer: torch.Tensor):
    """Speaker prior.
    Args:
        buffer: [torch.float32; [K, E x 2 + 1]], distribution weights.
    Returns:
        weight: [torch.float32; [K]], weights of each modals.
        mean: [torch.float32; [K, E]], mean vectors.
        std: [torch.float32; [K, E]], standard deviations.
    """
    # [K]
    weight = torch.softmax(buffer[..., 0], dim=0)
    # [K, E], [K, E]
    mean, logstd = buffer[..., 1:].chunk(2, dim=-1)
    # [K, E]
    std = F.softplus(logstd)
    # [K], [K, E], [K, E]
    return weight, mean, std


class Weight_Scheduler(object):
    def __init__(self, base_wt, n_warmup_steps, update_step, power):
        """
            warmup and update every update_step 
		"""
        self.base_wt = base_wt
        self.n_warmup_steps = n_warmup_steps
        self.power = power
        self.update_step = update_step

    def _get_wt(self, n_current_steps):
        if self.n_warmup_steps != 0 and n_current_steps <= self.n_warmup_steps and n_current_steps % self.update_step == 0:
            scale = np.power(self.n_warmup_steps, -(1 + self.power)) * n_current_steps
        elif n_current_steps > self.n_warmup_steps and n_current_steps % self.update_step == 0:
            scale = np.power(n_current_steps, -self.power)
        else:
            scale = 0
        return scale * self.base_wt

    def _get_max_wt(self):
        return np.power(self.n_warmup_steps, -self.power)
