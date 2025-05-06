import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn.functional as F

class PAM(Module):
    """
    This code refers to "Dual attention network for scene segmentation"Position attention module".
    Ref from SAGAN
    """
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d

def get_act_layer():
    # TODO: select appropriate activation layer
    return nn.ReLU

def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)

def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)

class BasicConv(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        kernel_size, pad_mode='Zero',
        bias='auto', norm=False, act=False,
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(getattr(nn, pad_mode.capitalize()+'Pad2d')(kernel_size//2))
        seq.append(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=1, padding=0,
                bias=(False if norm else True) if bias=='auto' else bias,
                **kwargs
            )
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return F.sigmoid(x)

class DFEM(nn.Module):
    def __init__(self, input_nbr):
        super(DFEM, self).__init__()

        self.conv1 = nn.Conv2d(input_nbr, input_nbr, kernel_size=1)
        self.pam = PAM(input_nbr)
        self.sa = SpatialAttention()

    def forward(self, x1, x2):
        x11 = self.conv1(x1)
        x21 = self.conv1(x2)
        weight = torch.cat((x11,x21),dim=1)
        weight = self.sa(weight)

        out1 = self.pam(x11)
        out2 = self.pam(x21)

        x1 = torch.mul(x1,out1)
        #x1 = torch.mul(x1,weight)

        x2 = torch.mul(x2,out2)
        #x2 = torch.mul(x2,weight)

        out = torch.mul(weight,torch.abs(x2-x1))

        return out

class MFAF(Module):

    def __init__(self):
        super(MFAF, self).__init__()

        self.pa1 = PAM(256)
        self.pa2 = PAM(128)
        self.pa3 = PAM(64)
        self.pa4 = PAM(32)
        self.conv1 = nn.Sequential(
            # 二维膨胀卷积
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            # 二维膨胀卷积
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            # 二维膨胀卷积
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4,
                      dilation=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            # 二维膨胀卷积
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=6,
                      dilation=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True),
        )
    def forward(self, x1, x2, x3, x4):
        #print(x1.shape)
        x1_weight = self.avgpool(x1)
        #print(x1_weight.shape)
        x1_weight = self.pa1(x1_weight)
        x1_weight = torch.sigmoid(x1_weight)


        x1 = torch.mul(x1_weight,x1)
        x1 = self.conv1(x1)

        #print(x1.shape)
        x2_weight = self.avgpool(x2)
        x2_weight = self.pa2(x2_weight)
        x2_weight = torch.sigmoid(x2_weight)

        x2 = torch.mul(x2_weight, x2)
        x2 = self.conv2(torch.add(x1, x2))
        #x2 = torch.sigmoid(x2)
        #print(x2.shape)

        x3_weight = self.avgpool(x3)
        x3_weight = self.pa3(x3_weight)
        x3_weight = torch.sigmoid(x3_weight)

        x3 = torch.mul(x3_weight, x3)
        x3 = self.conv3(torch.add(x2, x3))
        #x3 = torch.sigmoid(x3)
        #print(x3.shape)

        x4_weight = self.avgpool(x4)
        x4_weight = self.pa4(x4_weight)
        x4_weight = torch.sigmoid(x4_weight)

        out = torch.mul(x4_weight,x4)
        #print(x4.shape)
        out = self.conv4(torch.add(x3, out))

        out = torch.cat((out, x4),dim=1)
        out = self.conv5(out)
        #print(out.shape)

        return out

class MSFAFNet(nn.Module):
    """EF segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(MSFAFNet, self).__init__()

        self.input_nbr = input_nbr

        self.conv1 = nn.Conv2d(input_nbr,16,kernel_size=3,stride=1,padding=1)
        self.rcb1 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(16, 16, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
        )

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.rcb2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
        )

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.rcb3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
        )

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.rcb4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
        )

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.rcb5 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
        )

        self.relu=nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Dconv5 = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True)
        )

        self.Dconv4 = nn.Sequential(

            nn.ConvTranspose2d(384, 256, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True)

        )

        self.Dconv3 = nn.Sequential(

            nn.ConvTranspose2d(192, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True)

        )

        self.Dconv2 = nn.Sequential(

            nn.ConvTranspose2d(96, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True)

        )
        self.Dconv1 = nn.Sequential(
            nn.ConvTranspose2d(80, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(True),

            nn.Conv2d(32, label_nbr, kernel_size=3, padding=1)
        )

        self.df1 = DFEM(16)
        self.df2 = DFEM(32)
        self.df3 = DFEM(64)
        self.df4 = DFEM(128)

        self.msff = MFAF()


    def forward(self, x1, x2):

        x1_1 = self.maxpool1(self.relu(self.conv1(x1) + self.rcb1(self.conv1(x1))))
        x2_1 = self.maxpool1(self.relu(self.conv1(x2) + self.rcb1(self.conv1(x2))))
        x1 = self.df1(x1_1,x2_1)
        print(x1.shape)

        x1_2 = self.maxpool1(self.relu(self.conv2(x1_1) + self.rcb2(self.conv2(x1_1))))
        x2_2 = self.maxpool1(self.relu(self.conv2(x2_1) + self.rcb2(self.conv2(x2_1))))
        x2 = self.df2(x1_2,x2_2)
        print(x2.shape)
        #x2 = torch.cat((x1_2,x2_2),dim=1)

        x1_3 = self.maxpool1(self.relu(self.conv3(x1_2) + self.rcb3(self.conv3(x1_2))))
        x2_3 = self.maxpool1(self.relu(self.conv3(x2_2) + self.rcb3(self.conv3(x2_2))))
        x3 = self.df3(x1_3,x2_3)
        print(x3.shape)
        #x3 = torch.cat((x1_3,x2_3),dim=1)

        x1_4 = self.maxpool1(self.relu(self.conv4(x1_3) + self.rcb4(self.conv4(x1_3))))
        x2_4 = self.maxpool1(self.relu(self.conv4(x2_3) + self.rcb4(self.conv4(x2_3))))
        x4 = self.df4(x1_4,x2_4)
        print(x4.shape)
        #x4 = torch.cat((x1_4, x2_4), dim=1)

        x1_5 = self.maxpool1(self.relu(self.conv5(x1_4) + self.rcb5(self.conv5(x1_4))))
        x2_5 = self.maxpool1(self.relu(self.conv5(x2_4) + self.rcb5(self.conv5(x2_4))))
        x5 = torch.cat((x1_5, x2_5), dim=1)
        print(x5.shape)

        #print(x5.shape)
        x5 = self.Dconv5(x5)

        x4 = torch.cat((x5,x4),dim=1)
        x4 = self.Dconv4(x4)

        x3 = torch.cat((x4,x3),dim=1)
        x3 = self.Dconv3(x3)

        x2 = torch.cat((x3,x2),dim=1)
        x2 = self.Dconv2(x2)
        weight2 = self.msff(x5,x4,x3,x2)
        x1 = torch.cat((x2,x1),dim=1)
        x1 = torch.cat((x1,weight2),dim=1)
        x1 = self.Dconv1(x1)

        return x1



if __name__ == '__main__':
    net = MSFAFNet(3,1)
    im1 = torch.randn(1, 3, 256, 256)
    im2 = torch.randn(1, 3, 256, 256)
    out = net(im1, im2)
    print(out.shape)