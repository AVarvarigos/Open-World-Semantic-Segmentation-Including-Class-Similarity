#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
from torch import nn
import torch.nn.functional as F


class ObjectosphereLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, logits, sem_gt):
        logits_unk = logits.permute(0, 2, 3, 1)[torch.where(sem_gt == 255)]
        logits_kn = logits.permute(0, 2, 3, 1)[torch.where(sem_gt != 255)]

        if len(logits_unk):
            loss_unk = torch.linalg.norm(logits_unk, dim=1).mean()
        else:
            loss_unk = torch.tensor(0)
        if len(logits_kn):
            loss_kn = F.relu(self.sigma - torch.linalg.norm(logits_kn, dim=1)).mean()
        else:
            loss_kn = torch.tensor(0)

        loss = 10 * loss_unk + loss_kn
        return loss