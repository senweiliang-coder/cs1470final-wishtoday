import torch


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    n_s = source.size(0)
    n_t = target.size(0)
    total = torch.cat([source, target], dim=0)
    total_square = torch.sum(total**2, dim=1, keepdim=True)
    l2_distance = total_square + total_square.t() - 2 * torch.matmul(total, total.t())
    l2_distance = torch.clamp(l2_distance, min=0.0)

    n = n_s + n_t
    length_scale = l2_distance.sum() / max(n**2 - n, 1)
    length_scale = length_scale / (kernel_mul ** (kernel_num // 2))
    scales = [length_scale * (kernel_mul**i) for i in range(kernel_num)]
    kernels = [torch.exp(-l2_distance / torch.clamp(scale, min=1e-8)) for scale in scales]
    return torch.stack(kernels, dim=0).sum(dim=0)


def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5):
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num)
    n_s = source.size(0)
    n_t = target.size(0)

    xx = kernels[:n_s, :n_s].sum() / max(n_s**2, 1)
    yy = kernels[n_s:, n_s:].sum() / max(n_t**2, 1)
    xy = kernels[:n_s, n_s:].sum() / max(n_s * n_t, 1)
    yx = kernels[n_s:, :n_s].sum() / max(n_s * n_t, 1)
    return torch.abs(xx + yy - xy - yx)


def compute_kl_divergence(p, m):
    return torch.sum(p * torch.log(torch.clamp(p / m, min=1e-8)), dim=1).mean()


def compute_js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (compute_kl_divergence(p, m) + compute_kl_divergence(q, m))
