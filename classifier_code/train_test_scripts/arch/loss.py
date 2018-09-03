import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

EPS = 1.0e-10


class GeodesicClassificationLoss(nn.Module):
    def __init__(self, key, bin_size, sigma, wted=False, wt=0.03):
        ## ensure sigma is not int
        super(GeodesicClassificationLoss, self).__init__()
        self.key = key
        if wted:
            self.wt = wt
            self.loss_weights = 3 * torch.exp(-torch.abs(torch.arange(90) - 45).float().cuda() * self.wt)
            self.loss_wted = nn.CrossEntropyLoss(self.loss_weights)
            print(self.loss_weights)
        self.loss = nn.CrossEntropyLoss()
        bin_wt = []
        x1 = np.zeros((1, 3), dtype='float32')
        x1 = torch.autograd.Variable(torch.from_numpy(x1))
        rotmat_1 = self.get_rotmat_torch(x1)
        for i in range(bin_size):
            x2 = np.array([[-bin_size / 2 + 1 + i, 0, 0]], dtype='float32')
            x2 = torch.autograd.Variable(torch.from_numpy(x2))
            rotmat_2 = self.get_rotmat_torch(x2)
            bin_wt.append(self.get_distance_rotmat_torch(rotmat_1, rotmat_2)[0])

        bin_wt = torch.cat(bin_wt, 0)
        self.bin_wt = torch.exp(-bin_wt / sigma).cuda()

        self.bin_size = self.bin_wt.size()[0]

    def get_rotmat_torch(self, angles):
        # angles is batch_size,3 
        angles = angles * math.pi / 180.

        coses = torch.cos(angles)
        sines = torch.sin(angles)
        cos_2 = coses[:, 2]
        cos_0 = coses[:, 0]
        cos_1 = coses[:, 1]
        sin_0 = sines[:, 0]
        sin_2 = sines[:, 2]
        sin_1 = sines[:, 1]

        a11 = cos_0 * cos_1
        a12 = sin_2 * sin_0 * cos_1 - cos_2 * sin_1
        a13 = cos_2 * sin_0 * cos_1 + sin_2 * sin_1
        a21 = cos_0 * sin_1
        a22 = sin_2 * sin_0 * sin_1 + cos_2 * cos_1
        a23 = cos_2 * sin_0 * sin_1 - sin_2 * cos_1
        a31 = - sin_0
        a32 = sin_2 * cos_0
        a33 = cos_2 * cos_0

        a1 = torch.stack([a11, a12, a13], 1)
        a2 = torch.stack([a21, a22, a23], 1)
        a3 = torch.stack([a31, a32, a33], 1)
        out = torch.stack([a1, a2, a3], 1)

        return out

    def get_distance_rotmat_torch(self, rotmat_1, rotmat_2):
        dist = []
        batch_size = rotmat_1.size()[0]
        for batch in range(batch_size):
            val = torch.acos(
                (torch.trace(torch.mm(rotmat_1[batch].transpose(1, 0), rotmat_2[batch])) - 1.0) / 2.0) * 180 / math.pi
            dist.append(val)
        return dist

    def forward_normal(self, predicted, labels, num_bins=360):
        gt_ = Variable(labels.data[:, self.key, 0])

        tot_loss = Variable(torch.zeros(1).cuda())

        for i in range(self.bin_size):
            gt__ = (gt_.clone() - (self.bin_size - 1) / 2. + i).long()
            gt__ = torch.remainder(gt__, num_bins)

            loss = self.loss(predicted[:, 0, :], gt__) * self.bin_wt[i] / float(self.bin_size)
            tot_loss = tot_loss + loss
        return tot_loss

    def forward_mlp(self, predicted, labels, num_bins=360):
        gt_ = Variable(labels.data.long())
        tot_loss = Variable(torch.zeros(1).cuda())
        for i in range(self.bin_size):
            gt__ = gt_.clone() - (self.bin_size - 1) / 2. + i
            gt__ = torch.remainder(gt__, num_bins)

            loss = self.loss(predicted, gt__) * self.bin_wt[i] / float(self.bin_size)
            tot_loss = tot_loss + loss
        return tot_loss


class GeodesicClassificationLossCapped(nn.Module):
    def __init__(self, key, bin_size, sigma, num_bins=90, mu=20.0, in_range=180.0, out_range=45.0):
        ## ensure sigma is not int
        super(GeodesicClassificationLossCapped, self).__init__()
        self.num_bins = num_bins
        self.key = key
        self.loss_weights = torch.ones(self.num_bins + bin_size - 1).cuda()
        self.loss_weights[:(bin_size - 1) / 2] = 0
        self.loss_weights[-(bin_size - 1) / 2:] = 0

        self.loss = nn.CrossEntropyLoss(self.loss_weights)
        self.mu = mu
        self.in_range = in_range
        self.out_range = out_range

        bin_wt = []
        x1 = np.zeros((1, 3), dtype='float32')
        x1 = torch.autograd.Variable(torch.from_numpy(x1))
        rotmat_1 = self.get_rotmat_torch(x1)
        for i in range(bin_size):
            x2 = np.array([[-bin_size / 2 + 1 + i, 0, 0]], dtype='float32')
            x2 = torch.autograd.Variable(torch.from_numpy(x2))
            rotmat_2 = self.get_rotmat_torch(x2)
            bin_wt.append(self.get_distance_rotmat_torch(rotmat_1, rotmat_2)[0])
        bin_wt = torch.cat(bin_wt, 0)
        self.bin_wt = torch.exp(-bin_wt / sigma).cuda()

        self.bin_size = self.bin_wt.size()[0]

    def get_rotmat_torch(self, angles):

        angles = angles * math.pi / 180.

        coses = torch.cos(angles)
        sines = torch.sin(angles)
        cos_2 = coses[:, 2]
        cos_0 = coses[:, 0]
        cos_1 = coses[:, 1]
        sin_0 = sines[:, 0]
        sin_2 = sines[:, 2]
        sin_1 = sines[:, 1]

        a11 = cos_0 * cos_1
        a12 = sin_2 * sin_0 * cos_1 - cos_2 * sin_1
        a13 = cos_2 * sin_0 * cos_1 + sin_2 * sin_1
        a21 = cos_0 * sin_1
        a22 = sin_2 * sin_0 * sin_1 + cos_2 * cos_1
        a23 = cos_2 * sin_0 * sin_1 - sin_2 * cos_1
        a31 = - sin_0
        a32 = sin_2 * cos_0
        a33 = cos_2 * cos_0

        a1 = torch.stack([a11, a12, a13], 1)
        a2 = torch.stack([a21, a22, a23], 1)
        a3 = torch.stack([a31, a32, a33], 1)
        out = torch.stack([a1, a2, a3], 1)

        return out

    def get_distance_rotmat_torch(self, rotmat_1, rotmat_2):
        dist = []
        batch_size = rotmat_1.size()[0]
        for batch in range(batch_size):
            val = torch.acos(
                (torch.trace(torch.mm(rotmat_1[batch].transpose(1, 0), rotmat_2[batch])) - 1.0) / 2.0) * 180 / math.pi
            dist.append(val)
        return dist

    def mu_law(self, inp):
        inp = inp / self.in_range
        out = torch.sign(inp) * (torch.log(1 + self.mu * torch.abs(inp))) / np.log(1 + self.mu) * self.out_range
        return out

    def forward_mu_law(self, predicted, labels):
        labels.data[:, self.key, 2] = labels.data[:, self.key, 2]
        gt_ = Variable(self.mu_law(labels.data[:, self.key, 2]).long() + self.num_bins / 2)

        tot_loss = Variable(torch.zeros(1).cuda())
        predicted = predicted[:, self.key, :]
        additional = Variable(torch.zeros(predicted.size()[0], (self.bin_size - 1) / 2).cuda())
        predicted = torch.cat([additional, predicted, additional], 1)
        x = (self.bin_size - 1) / 2
        for i in range(self.bin_size):
            gt__ = torch.clamp(gt_.clone(), 0, self.num_bins - 1)
            gt__ = gt__ - x + i

            loss = self.loss(predicted, gt__ + (self.bin_size - 1) / 2)
            loss = loss * self.bin_wt[i] / float(self.bin_size)
            tot_loss = tot_loss + loss

        return tot_loss


if __name__ == '__main__':
    x = 1
