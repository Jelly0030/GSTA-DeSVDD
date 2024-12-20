import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class autoencoder(nn.Module):
    def __init__(self,data_dim,z_dim):
        super(autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool1d(2, 2)
        self.conv1 = nn.Conv1d(1, 32, 3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(32, 64, 3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv1d(64, 128, 3, bias=False, padding=1)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * int(data_dim/8), z_dim, bias=False)
        self.fc2 = nn.Linear(z_dim, 128 * int(510/8), bias=False)
        self.deconv3 = nn.ConvTranspose1d(128, 64, 3, bias=False, padding=1)
        self.bn4 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose1d(64, 32, 3, bias=False, padding=1)
        self.bn5 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose1d(32, 1, 3, bias=False, padding=1)
        self.bn6 = nn.BatchNorm1d(1, eps=1e-04, affine=False)
        self.up = nn.Upsample(scale_factor=(2, 1))
        self.fc3 = nn.Linear(z_dim, data_dim, bias=False)

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def decode(self, x):
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        x_hat = x.view(x.size(0), -1)
        return x_hat

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class SWAT_LeNet_ResNet(nn.Module):

    def __init__(self, data_dim, z_dim):
        super(SWAT_LeNet_ResNet,self).__init__()
        self.pool = nn.MaxPool1d(2, stride=2)
        self.conv1 = nn.Conv1d(1, 32, 3, bias=False, padding=1)
        self.bn1d1 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(32, 64, 3, bias=False, padding=1)
        self.bn1d2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv1d(64, 128, 3, bias=False, padding=1)
        self.bn1d3 = nn.BatchNorm1d(128, eps=1e-04, affine=False)

        # Residual blocks
        self.resnet = nn.Sequential(
            ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.fc1 = nn.Linear(128 * int(data_dim/8), z_dim, bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1d1(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn1d2(x)))
        x = self.conv3(x)
        x = self.pool(F.relu(self.bn1d3(x)))
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

