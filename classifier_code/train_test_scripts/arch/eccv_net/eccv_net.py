import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import sys

from configs.constants import class_map
from .. import loss as l
from ..googlenet import Inception


class ConvCorrelationLayer(nn.Module):
    def __init__(self, ele_bin=90, tilt_bin=50):
        super(ConvCorrelationLayer, self).__init__()

        self.pi = 3.14159265358979323846264338327950
        self.batch_size = None

        self.in_planes = 256 + 128
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
        self.til_2 = nn.Linear(in_features=256, out_features=tilt_bin, bias=True).cuda()

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

        out1 = torch.unsqueeze(ele_2, 1)
        out2 = torch.unsqueeze(til_2, 1)

        bin_1 = self.relu(self.binner_1(out))
        out3 = self.binner_2(bin_1)
        out3 = torch.unsqueeze(out3, 1)

        return out3, out1, out2


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

    def forward(self, feature1):
        feature1 = F.max_pool2d(input=feature1, kernel_size=2, stride=2, ceil_mode=True).cuda()
        out_1 = self.inception_1(feature1)
        out_1 = self.bn_1(out_1)
        out_1 = self.feature_convo_1(out_1)

        out = out_1
        return out


class InceptionEmbedding(nn.Module):
    def __init__(self, class_name='chair', num_views=3):
        super(InceptionEmbedding, self).__init__()

        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None

        self.inception000 = Inception(in_planes=num_views * class_map[class_name], n1x1=128, n3x3red=96, n3x3=128,
                                      n5x5red=16, n5x5=32, pool_planes=32).cuda()
        self.bn000 = nn.BatchNorm2d(num_features=320)

        self.inception00 = Inception(in_planes=320, n1x1=32, n3x3red=48, n3x3=64, n5x5red=8, n5x5=16,
                                     pool_planes=16).cuda()
        self.bn00 = nn.BatchNorm2d(num_features=128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)



    def forward(self, blob):
        embed = self.inception000(blob)
        embed = self.bn000(embed)
        embed = self.inception00(embed)
        embed = self.bn00(embed)

        return embed


class CorrelationMapLayer(nn.Module):
    def __init__(self):
        super(CorrelationMapLayer, self).__init__()

        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None
        self.in_planes = 784
        self.down1 = torch.nn.UpsamplingBilinear2d(size=(28, 28))

        self.conv_embedding = InceptionEmbedding().cuda()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)

    #
    def forward(self, feature1, feature2, knn_inds):


        feature1 = self.down1(feature1)
        feature2 = self.down1(feature2)

        feature1 = feature1.data
        feature2 = feature2.data

        # Check with downsample
        self.batch_size, self.channels, self.height, self.width = feature2.shape

        feature2 = feature2.view((self.batch_size, self.channels, -1))

        feature2 = feature2.permute(0, 2, 1).contiguous()

        feature2 = torch.stack([feature2] * self.height, dim=3)

        feature2 = torch.stack([feature2] * self.width, dim=4)


        feature2 = feature2.permute(1, 0, 2, 3, 4).contiguous()

        correlation = torch.mul(feature2, feature1)

        correlation = torch.sum(correlation, dim=2)
        correlation = F.relu(Variable(correlation)).data


        correlation = correlation.view((28, 28, self.batch_size, 28, 28))[knn_inds[:, 1], knn_inds[:, 0]]


        correlation = correlation.permute(2, 3, 1, 0).contiguous()


        corr_exp = torch.pow(correlation, 2)
        corr_exp = torch.sum(corr_exp, dim=0, keepdim=True)
        corr_exp = torch.sum(corr_exp, dim=1, keepdim=True)

        corr_exp = torch.exp(correlation)
        corr_exp = torch.sum(corr_exp, dim=0, keepdim=True)
        corr_exp = torch.sum(corr_exp, dim=1, keepdim=True)

        correlation = correlation / corr_exp * 10

        correlation = correlation.permute(2, 3, 0, 1).contiguous()


        correlation = Variable(correlation)

        return correlation


class Net(nn.Module):
    def __init__(self, class_name='chair', num_views=3, ele_bin=90, tilt_bin=50):
        super(Net, self).__init__()
        self.class_name = class_name
        self.num_views = num_views
        self.ele_bin = ele_bin
        self.tilt_bin = tilt_bin
        self.correlation_map = CorrelationMapLayer().cuda()
        self.inceptions = InceptionEmbedding(class_name, num_views).cuda()
        self.feature_conv = FeatureConvLayer().cuda()
        self.conv_corr = ConvCorrelationLayer().cuda()


        self.fc_1_azi = nn.Linear(in_features=64 * 7 * 7, out_features=512, bias=True).cuda()

        self.fc_2_azi = nn.Linear(in_features=512, out_features=360, bias=True).cuda()
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(p=0.5)
        self.fc_1_ele = nn.Linear(in_features=64 * 7 * 7, out_features=512, bias=True).cuda()

        self.fc_2_ele = nn.Linear(in_features=512, out_features=ele_bin, bias=True).cuda()

        self.fc_1_til = nn.Linear(in_features=64 * 7 * 7, out_features=512, bias=True).cuda()

        self.fc_2_til = nn.Linear(in_features=512, out_features=tilt_bin, bias=True).cuda()

        self.real_angle_pred_azi = l.GeodesicClassificationLoss(0, 5, 3)
        self.real_angle_pred_ele = l.GeodesicClassificationLoss(1, 5, 3)
        self.real_angle_pred_til = l.GeodesicClassificationLoss(2, 5, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.2)

    '''
    knn_inds is quantization*10*2
    visibility is quantization*10
    '''

    def forward(self, feature_i, feature_j, mask, optical_flow, knn_inds, visibility=None):

        num_feat = feature_i.shape[1]
        batch_size = feature_i.shape[0]

        knn_inds = torch.cuda.LongTensor(knn_inds)
        list_embeddings = []
        for x in range(num_feat):
            mapx = self.correlation_map(feature_i[:, x, :, :, :], feature_j[:, x, :, :, :], knn_inds[x, :, :])
            list_embeddings.append(mapx)

        embedding = torch.stack(list_embeddings, 0)

        embedding = embedding.permute(1, 0, 2, 3, 4).contiguous()
        embedding = self.inceptions(embedding.view(batch_size, self.num_views * class_map[self.class_name], 28, 28))

        feature_embedding = self.feature_conv(feature_i[:, x, :, :, :])
        embedding = torch.cat([embedding, feature_embedding], 1)

        out_azi, out_ele, out_til = self.conv_corr(embedding)

        return out_azi, out_ele, out_til

    def loss_all(self, prediction1, prediction2, prediction3, labels):

        loss1 = self.real_angle_pred_azi.forward_normal(prediction1, labels)
        loss2 = self.real_angle_pred_ele.forward_normal(prediction2, labels, num_bins=self.ele_bin)
        loss3 = self.real_angle_pred_til.forward_normal(prediction3, labels, num_bins=self.tilt_bin)


        return loss1, loss2, loss3


def test():
    torch.cuda.set_device(int(sys.argv[1]))



if __name__ == '__main__':
    test()
