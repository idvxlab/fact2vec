import torch
import torch.nn as nn


class customLoss(nn.Module):
    def __init__(self):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(customLoss, self).__init__()

    def forward(self, output):
        distLinear_Sum = 0
        distDiff_Sum = 0
        for item in output:
            v1 = item[0]
            v2 = item[1]
            v3 = item[2]
            m = (v1+v3)/2
            distLinear = (torch.dist(v2, m, p=2)).pow(2)
            distDiff = 0.05 * \
                ((torch.dist(v1, v2, p=2)+torch.dist(v2, v3, p=2) +
                 torch.dist(v1, v3, p=2)).pow(2))
            distLinear_Sum += distLinear
            distDiff_Sum += distDiff

        return distLinear_Sum + distDiff_Sum
