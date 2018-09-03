import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .. import loss as l
from ..googlenet import Inception


class CorrelationMapLayer(nn.Module):
    def __init__(self):
        super(CorrelationMapLayer, self).__init__()

        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None
        self.in_planes = 784
        self.down1 = torch.nn.UpsamplingBilinear2d(size=(28, 28))
        self.inception00 = Inception(in_planes=self.in_planes, n1x1=128, n3x3red=192, n3x3=256, n5x5red=32, n5x5=64,
                                     pool_planes=64).cuda()
        self.bn00 = nn.BatchNorm2d(num_features=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)

    def forward(self, feature1, feature2, mask, knn_inds=None):

        feature1 = self.down1(feature1)
        feature2 = self.down1(feature2)
        feature1 = feature1.data
        feature2 = feature2.data

        mask = F.max_pool2d(input=mask, kernel_size=2, stride=2, ceil_mode=True).cuda()
        # Check with downsample
        self.batch_size, self.channels, self.height, self.width = feature1.shape

        feature1 = feature1.view((self.batch_size, self.channels, -1))
        feature1 = feature1.permute(0, 2, 1).contiguous()
        feature1 = torch.stack([feature1] * self.height, dim=3)
        feature1 = torch.stack([feature1] * self.width, dim=4)
        feature1 = feature1.permute(1, 0, 2, 3, 4).contiguous()

        correlation = torch.mul(feature1, feature2)
        correlation = torch.sum(correlation, dim=2)
        correlation = F.relu(Variable(correlation)).data
        correlation = correlation.permute(1, 0, 2, 3).contiguous()

        corr_map = correlation.clone()
        corr_sqr = corr_map ** 2
        corr_norm_sqr = torch.sum(corr_sqr, dim=1)
        corr_norm = torch.sqrt(corr_norm_sqr)
        correlation = correlation.permute(1, 0, 2, 3).contiguous()
        correlation = correlation / corr_norm
        correlation = correlation.permute(1, 0, 2, 3).contiguous()
        correlation = Variable(correlation)
        correlation = torch.mul(correlation, mask)

        correlation = self.inception00(correlation)
        correlation = self.bn00(correlation)

        return correlation


class FeatureConvLayerSingle(nn.Module):

    def __init__(self):
        super(FeatureConvLayerSingle, self).__init__()
        self.inception2 = Inception(in_planes=128, n1x1=64, n3x3red=96, n3x3=128, n5x5red=16, n5x5=32,
                                    pool_planes=32).cuda()
        self.bn2 = nn.BatchNorm2d(num_features=256).cuda()

        self.feature_convo = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True).cuda(),
            nn.ReLU(inplace=False).cuda(),
            nn.BatchNorm2d(num_features=256).cuda(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)

    def forward(self, feature1, feature2):
        feature2 = F.max_pool2d(input=feature2, kernel_size=2, stride=2, ceil_mode=True).cuda()
        out2 = self.inception2(feature2)
        out2 = self.bn2(out2)
        out = self.feature_convo(out2)
        return out


class FeatureConvLayer(nn.Module):

    def __init__(self):
        super(FeatureConvLayer, self).__init__()
        self.inception_1 = Inception(in_planes=128, n1x1=64, n3x3red=96, n3x3=128, n5x5red=16, n5x5=32,
                                     pool_planes=32).cuda()
        self.bn_1 = nn.BatchNorm2d(num_features=256).cuda()

        self.feature_convo_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True).cuda(),
            nn.ReLU(inplace=False).cuda(),
            nn.BatchNorm2d(num_features=256).cuda(),
        )
        self.inception_2 = Inception(in_planes=128, n1x1=64, n3x3red=96, n3x3=128, n5x5red=16, n5x5=32,
                                     pool_planes=32).cuda()
        self.bn_2 = nn.BatchNorm2d(num_features=256).cuda()

        self.feature_convo_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True).cuda(),
            nn.ReLU(inplace=False).cuda(),
            nn.BatchNorm2d(num_features=256).cuda(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)

    def forward(self, feature1, feature2):
        feature1 = F.max_pool2d(input=feature1, kernel_size=2, stride=2, ceil_mode=True).cuda()
        out_1 = self.inception_1(feature1)
        out_1 = self.bn_1(out_1)
        out_1 = self.feature_convo_1(out_1)
        feature2 = F.max_pool2d(input=feature2, kernel_size=2, stride=2, ceil_mode=True).cuda()
        out_2 = self.inception_2(feature2)
        out_2 = self.bn_2(out_2)
        out_2 = self.feature_convo_2(out_2)
        out = torch.cat([out_1, out_2], 1)
        return out


class OpticalFlowConv(nn.Module):

    def __init__(self):
        super(OpticalFlowConv, self).__init__()

        self.inception1 = Inception(in_planes=2, n1x1=16, n3x3red=24, n3x3=32, n5x5red=4, n5x5=8, pool_planes=8).cuda()
        self.bn1 = nn.BatchNorm2d(num_features=64).cuda()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True).cuda()
        self.inception2 = Inception(in_planes=64, n1x1=64, n3x3red=96, n3x3=128, n5x5red=16, n5x5=32, pool_planes=32,
                                    no_drop=False).cuda()
        self.bn2 = nn.BatchNorm2d(num_features=256).cuda()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True).cuda()

    def forward(self, optical_flow):
        out = self.inception1(optical_flow)
        out = self.bn1(out)
        out = self.pool1(out)
        out = self.inception2(out)
        out = self.bn2(out)
        out = self.pool2(out)

        return out


class ConvCorrelationLayer(nn.Module):
    def __init__(self, ele_bin):
        super(ConvCorrelationLayer, self).__init__()

        self.pi = 3.14159265358979323846264338327950
        self.batch_size = None
        # self.flow = flow
        self.in_planes = 1024 + 256
        self.linear_features = 64 * 7 * 7
        self.ele_bin = ele_bin

        self.inception00 = Inception(in_planes=self.in_planes, n1x1=128, n3x3red=192, n3x3=256, n5x5red=32, n5x5=64,
                                     pool_planes=64).cuda()
        self.bn00 = nn.BatchNorm2d(num_features=512)
        self.inception01 = Inception(in_planes=512, n1x1=64, n3x3red=96, n3x3=128, n5x5red=16, n5x5=32,
                                     pool_planes=32).cuda()
        self.bn01 = nn.BatchNorm2d(num_features=256)
        self.inception2 = Inception(in_planes=256, n1x1=32, n3x3red=48, n3x3=64, n5x5red=8, n5x5=16,
                                    pool_planes=16).cuda()
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True).cuda()
        self.inception3 = Inception(in_planes=128, n1x1=16, n3x3red=24, n3x3=32, n5x5red=4, n5x5=8, pool_planes=8,
                                    no_drop=False).cuda()
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True).cuda()

        self.dropout = nn.Dropout(p=0.5)

        self.azim = nn.Linear(in_features=self.linear_features, out_features=256, bias=True).cuda()
        self.azi_2 = nn.Linear(in_features=256, out_features=90, bias=True).cuda()
        self.elev = nn.Linear(in_features=self.linear_features, out_features=256, bias=True).cuda()
        self.ele_2 = nn.Linear(in_features=256, out_features=ele_bin, bias=True).cuda()
        self.tilt = nn.Linear(in_features=self.linear_features, out_features=256, bias=True).cuda()
        self.til_2 = nn.Linear(in_features=256, out_features=ele_bin, bias=True).cuda()

        self.tanh = nn.Tanh().cuda()
        self.relu = nn.ReLU(inplace=False).cuda()
        self.binner_1 = nn.Linear(in_features=self.linear_features, out_features=512, bias=True).cuda()
        self.binner_2 = nn.Linear(in_features=512, out_features=360, bias=True).cuda()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, correlation_map):

        self.batch_size = correlation_map.data.shape[0]
        out = self.inception00(correlation_map)
        out = self.bn00(out)
        out = self.inception01(out)
        out = self.bn01(out)
        out = self.inception2(out)
        out = self.bn2(out)
        out = self.maxpool2(out)
        out = self.inception3(out)
        out = self.bn3(out)
        out = self.maxpool3(out)

        out = out.view(self.batch_size, -1)

        azi = self.relu(self.azim(self.dropout(out)))
        azi_2 = self.azi_2(azi)
        ele = self.relu(self.elev(self.dropout(out)))
        ele_2 = self.ele_2(ele)
        til = self.relu(self.tilt(self.dropout(out)))
        til_2 = self.til_2(til)

        out1 = torch.unsqueeze(azi_2, 1)
        out2 = torch.stack([ele_2, til_2], dim=1)

        bin_1 = self.relu(self.binner_1(out))
        out3 = self.binner_2(bin_1)
        out3 = torch.unsqueeze(out3, 1)

        return out1, out2, out3


class Net(nn.Module):
    def __init__(self, ele_bin=40):
        super(Net, self).__init__()

        self.correlation_map = CorrelationMapLayer().cuda()
        self.conv_coorelation = ConvCorrelationLayer(ele_bin).cuda()
        self.feature_conv = FeatureConvLayer().cuda()
        self.optical_flow_conv = OpticalFlowConv().cuda()
        self.down1 = torch.nn.UpsamplingBilinear2d(size=(55, 55))

        self.azi_class = l.GeodesicClassificationLossCapped(0, 7, 3)
        if ele_bin == 1:
            self.ele_class = l.L2Loss(0)
            self.tilt_class = l.L2Loss(1)
        else:
            self.ele_class = l.GeodesicClassificationLossCapped(0, 5, 3, num_bins=ele_bin, mu=10.0, in_range=45.0,
                                                                out_range=10.0)
            self.tilt_class = l.GeodesicClassificationLossCapped(1, 5, 3, num_bins=ele_bin, mu=10.0, in_range=45.0,
                                                                 out_range=10.0)

        self.real_angle_pred = l.GeodesicClassificationLoss(0, 5, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)

    def forward(self, feature_i, feature_j, mask, optical_flow):

        correlation_map = self.correlation_map(feature_i, feature_j, mask)
        feature_conv = self.feature_conv(feature_i, feature_j)
        optical_flow_conv = self.optical_flow_conv(optical_flow)
        correlation_map = torch.cat([correlation_map, feature_conv, optical_flow_conv], dim=1)
        prediction1, prediction2, prediction3 = self.conv_coorelation(correlation_map)

        return prediction1, prediction2, prediction3

    def loss_all(self, prediction1, prediction2, prediction3, labels):
        # get the loss functions
        loss1 = self.azi_class.forward_mu_law(prediction1, labels)
        loss2 = self.ele_class.forward_mu_law(prediction2, labels[:, 1:, :])
        loss3 = self.tilt_class.forward_mu_law(prediction2, labels[:, 1:, :])
        loss4 = self.real_angle_pred.forward_normal(prediction3, labels)
        return loss1, loss2, loss3, loss4


def test():
    torch.cuda.set_device(int(sys.argv[1]))


if __name__ == '__main__':
    test()
