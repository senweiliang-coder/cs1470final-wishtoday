import numpy as np


def calc_nll_brier(probability, pred, target, one_hot_label):
    probability = np.asarray(probability, dtype=np.float64)
    one_hot_label = np.asarray(one_hot_label, dtype=np.float64)
    if probability.ndim == 1:
        probability = probability[None, :]
    if one_hot_label.ndim == 1:
        one_hot_label = one_hot_label[None, :]

    target_idx = int(np.asarray(target).reshape(-1)[0])
    nll = -np.log(np.clip(probability[0, target_idx], 1e-12, 1.0))
    brier = np.mean(np.sum((probability - one_hot_label) ** 2, axis=1))
    return float(nll), float(brier)


def calc_aurc_eaurc(probability_list, correct_list):
    confidences = np.asarray(probability_list, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct_list).astype(np.float64).reshape(-1)
    order = np.argsort(-confidences)
    correct = correct[order]

    coverage = np.arange(1, len(correct) + 1, dtype=np.float64) / max(len(correct), 1)
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(correct) + 1, dtype=np.float64)
    aurc = np.trapz(risk, coverage) if len(risk) > 0 else 0.0

    error_rate = 1.0 - correct.mean() if len(correct) > 0 else 0.0
    optimal_risk = error_rate + (1.0 - error_rate) * np.log(np.clip(1.0 - error_rate, 1e-12, 1.0))
    eaurc = aurc - optimal_risk
    return float(aurc), float(eaurc)
