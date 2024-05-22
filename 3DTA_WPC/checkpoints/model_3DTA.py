import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()        # ([batchsize, npoints, neighbor, feature])
        x = x.permute(0, 1, 3, 2)    # ([batchsize, npoints, feature, neighbor])
        x = x.reshape(-1, d, s)      # ([batchsize*npoints, feature, neighbor])
        batch_size, _, N = x.size()  # ([batchsize*npoints, feature, neighbor])
        x1 = F.relu(self.bn1(self.conv1(x)))    # ([batchsize*npoints, feature, neighbor])
        x2 = F.relu(self.bn2(self.conv2(x1)))    # ([batchsize*npoints, feature, neighbor])
        x3 = F.adaptive_max_pool1d(x2, 1)        # ([batchsize*npoints, feature, 1 ])
        x4 = x3.view(batch_size, -1)             # ([batchsize*npoints, feature])
        x_res = x4.reshape(b, n, -1).permute(0, 2, 1)
        return x_res                             # ([batchsize, feature, npoints])


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)    # [B, 256, 256]
        # b, c, n
        x_k = self.k_conv(x)                     # [B, 256, 256]
        x_v = self.v_conv(x)                     # [B, 256, 256]
        # b, n, n
        energy = torch.bmm(x_q, x_k)             # [B, 256, 256]

        attention = self.softmax(energy)         # Attention_Map
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)          # Attention_Feature
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)
        self.conv4 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm1d(channels)
        self.conv5 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm1d(channels)
        self.conv6 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm1d(channels)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.sa5 = SA_Layer(channels)
        self.sa6 = SA_Layer(channels)
        self.sa7 = SA_Layer(channels)
        self.sa8 = SA_Layer(channels)

    def forward(self, x):
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Twin attention block * 4
        x1 = self.sa1(x) + x
        x11 = self.sa2(x1.permute(0,2,1)) + x1.permute(0,2,1)
        x11 = F.relu(self.bn3(self.conv3(x11)))

        x2 = self.sa3(x11) + x11
        x22 = self.sa4(x2.permute(0,2,1)) + x2.permute(0,2,1)
        x22 = F.relu(self.bn4(self.conv4(x22)))

        x3 = self.sa5(x22) + x22
        x33 = self.sa6(x3.permute(0,2,1)) + x3.permute(0,2,1)
        x33 = F.relu(self.bn5(self.conv5(x33)))

        x4 = self.sa7(x33) + x33
        x44 = self.sa8(x4.permute(0,2,1)) + x4.permute(0,2,1)
        x44 = F.relu(self.bn6(self.conv6(x44)))

        x11223344 = torch.cat((x11, x22, x33, x44), dim=1)
        return x11223344


class Pct_3DTA(nn.Module):
    def __init__(self, args, final_channels=1):
        super(Pct_3DTA, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1, stride=int(args.point_num/256), bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.transfomer = Point_Transformer_Last(args)

        self.conv_fuse1 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.conv_fuse2 = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(1024),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, final_channels)

    def forward(self, x):
        xyz = x[:,0:3,:].permute(0, 2, 1)                  # get xyz axis
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))                # 64 <= in_channels
        # B, D, N
        x_str = F.relu(self.bn2(self.conv2(x)))

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, neighbor=32, xyz=xyz, feature=x)
        feature_0 = self.gather_local_0(new_feature)       # [B, 128, 512] <= [B, 512, 32, 128]

        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, neighbor=32, xyz=new_xyz, feature=feature_0)
        feature_1 = self.gather_local_1(new_feature)       # [B, 256, 256] <= [B, 256, 32, 256]

        feature_1 = torch.cat((feature_1, x_str), dim=1)
        feature_1 = self.conv_fuse1(feature_1)
        
        x = self.transfomer(feature_1)                      # [B, 256, 256]
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse2(x)

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
