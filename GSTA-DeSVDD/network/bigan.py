import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from network.autoencoder import weights_init_normal


def get_c_r(data):
    eps = 0.05
    c = torch.mean(data, dim=0)
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    r = torch.mean(torch.norm(data - c, dim=1))
    return c, r
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
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool1d(2, 2)
        self.conv1 = nn.Conv1d(1, 32, 3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(32, 64, 3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv1d(64, 128, 3, bias=False, padding=1)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * int(input_dim / 8), latent_dim, bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
class Generator(nn.Module):
    def __init__(self,latent_dim,input_dim):
        super(Generator, self).__init__()
        self.pool = nn.MaxPool1d(2, 2)
        self.conv1 = nn.Conv1d(1, 32, 3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(32, 64, 3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv1d(64, 128, 3, bias=False, padding=1)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * int(latent_dim / 8), input_dim, bias=False)

    def forward(self, z_inp):
        x = z_inp.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
class Discriminator(nn.Module):
    def __init__(self, data_dim, z_dim):
        super(Discriminator, self).__init__()
        self.pool = nn.MaxPool1d(2, 2)
        self.conv1 = nn.Conv1d(1, 32, 3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(32, 64, 3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv1d(64, 128, 3, bias=False, padding=1)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * int((data_dim+z_dim) / 8) , 1, bias=False)
        self.accuracy = torchmetrics.Accuracy(task='binary')
    def forward(self, x, z, y):
        x = torch.cat((x,z), dim=1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if y is not None:
            self.accuracy(x.round(), y)
        return x
    def get_accuracy(self):
        return self.accuracy.compute()
def bigan_train(data,data_dim,z_dim,num_epochs,batch_size,c,R):
    G = Generator(z_dim, data_dim)
    E = Encoder(data_dim, z_dim)
    D = Discriminator(data_dim, z_dim)
    E.apply(weights_init_normal)
    D.apply(weights_init_normal)
    g_optimizer = optim.Adam(G.parameters(), lr=0.0001)
    e_optimizer = optim.Adam(E.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(D.parameters(), lr=0.0001)
    z_list = []
    G.train()
    E.train()
    D.train()
    stop_training = False

    for epoch in range(num_epochs):
        for i in range(0, len(data), batch_size):
            real_data = torch.Tensor(data[i:i + batch_size])
            z = torch.Tensor(np.random.randn(batch_size, z_dim))
            d_optimizer.zero_grad()
            real_label = torch.ones(batch_size).unsqueeze(1)
            fake_label = torch.zeros(batch_size).unsqueeze(1)
            d_real_loss = nn.BCELoss()(torch.clamp(D(real_data, E(real_data), real_label), min=0.0, max=1.0),
                                       real_label)
            d_fake_loss = nn.BCELoss()(torch.clamp(D(G(z), z.detach(), fake_label), min=0.0, max=1.0), fake_label)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            d_accuracy = D.get_accuracy()
            g_optimizer.zero_grad()
            e_optimizer.zero_grad()
            fake_data = G(z)
            dist = torch.sqrt(torch.sum((fake_data - c) ** 2, dim=1))
            dist_loss = torch.mean(torch.where(dist < R, 0.5 * (R - dist) ** 2, torch.where(dist < 2*R, 0, dist - 2*R)))
            g_loss = nn.BCELoss()(torch.clamp(D(fake_data, z, real_label), min=0.0, max=1.0), real_label)
            g_loss += dist_loss
            e_loss = nn.MSELoss()(G(E(real_data)), real_data)
            (g_loss + e_loss + dist_loss).backward()
            g_optimizer.step()
            e_optimizer.step()
            if dist_loss.item() <= 0.005:
                stop_training = True
                break
        if stop_training:
            break
        z_list.append(z.detach().cpu().numpy())
        print(f"Epoch [{epoch + 1}/{num_epochs}] D loss: {d_loss:.4f}, G loss: {g_loss:.4f},dist loss: {dist_loss:.4f}, E loss: {e_loss:.4f},Discriminator accuracy: {d_accuracy * 100:.2f}%")

    torch.save(G.state_dict(), "generator.pth")
    torch.save(E.state_dict(), "encoder.pth")
    torch.save(D.state_dict(), "discriminator.pth")
