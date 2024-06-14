import torch
import torch.distributions as D

__all__ = ['tikhonov_reg2d', 'compute_cov2d', 'sample_gmm']

def tikhonov_reg2d(scales, corrs, alpha):
    print(f"scales shape: {scales.shape}")
    print(f"corrs shape: {corrs.shape}")
    
    if scales.shape[-1] != corrs.shape[-1]:
        scales = scales[..., :corrs.shape[-1]]
    
    scales_ = torch.sqrt(scales**2 + alpha)
    corrs_ = corrs * torch.prod(scales, -1, keepdim=True) / torch.prod(scales_, -1, keepdim=True)
    return scales_, corrs_

def compute_cov2d(scales, corrs):
    cov12 = corrs * torch.prod(scales, dim=-1)
    covs = torch.diag_embed(scales**2)
    I = torch.diag_embed(torch.ones_like(scales))
    covs = covs + cov12.unsqueeze(-1).unsqueeze(-1) * (1. - I)
    return covs

def sample_gmm(mix_logp, means, scales, corrs):
    covs = compute_cov2d(scales, corrs)
    mix = D.Categorical(mix_logp.exp())
    comp = D.MultivariateNormal(means, covs)
    gmm = D.MixtureSameFamily(mix, comp)
    return gmm.sample()