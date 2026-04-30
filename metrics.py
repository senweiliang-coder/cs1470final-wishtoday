import torch


def cal_ece(logits, target, n_bins=15):
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if target.ndim == 0:
        target = target.unsqueeze(0)

    probs = torch.softmax(logits.detach(), dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(target.detach())

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc = accuracies[in_bin].float().mean()
            conf = confidences[in_bin].mean()
            ece += torch.abs(conf - acc) * prop
    return float(ece.item())
