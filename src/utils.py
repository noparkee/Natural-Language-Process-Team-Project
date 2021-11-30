import random
import numpy as np

import torch


def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(matrix, num_classes):
    ua = 0
    wa = torch.zeros(num_classes, dtype=torch.double)

    ua_total = torch.sum(matrix)
    wa_total = torch.sum(matrix, dim=0)

    for i in range(num_classes):
        ua += (matrix[i][i].item())
        wa[i] = matrix[i][i] / wa_total[i]

    ua /= (ua_total.item())
    wa = (torch.sum(wa).item()) / num_classes

    return ua, wa
    