import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from network.eval_train import eval_test
from data.data_preprocess import swat_dataset, wadi, kdd_dataset
from network.deepsvdd import TrainerDeepSVDD
from network.bigan import bigan_train, Generator,get_c_r
from network.spatio_temporal import ST,train_model, eval_model
from utils import get_loader




class Args:
    num_epochs = None
    num_epochs_ae = None
    num_epochs_bigan = None
    patience = None
    lr = None
    weight_decay = None
    weight_decay_ae = None
    lr_ae = None
    lr_milestones = [None]
    batch_size = None
    pretrain = True
    latent_dim = None
    data_dim = None #swat 51 kdd 59 wadi 127
    normal_class = None
    window_length = None
    n = None

def main():
    args = Args()
    x_train, y_train, x_test, y_test = swat_dataset(args.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = get_loader(torch.tensor(x_train), batch_size=args.batch_size, window_length=args.window_length, input_size=args.data_dim, shuffle=False)
    test = get_loader(torch.tensor(x_test), batch_size=args.batch_size, window_length=args.window_length, input_size=args.data_dim, shuffle=False)
    ST_model = ST(input_size=args.window_length, hidden_size=None).double()
    x_train = train_model(ST_model, train, epochs=None)
    x_test = eval_model(ST_model, test)

    x = torch.tensor(x_train)
    c, r = get_c_r(x)
    bigan_train(x_train, args.data_dim, args.latent_dim, args.num_epochs_bigan, args.batch_size, c, r)
    G = Generator(args.latent_dim, args.data_dim)
    G.load_state_dict(torch.load('generator.pth'))
    G.eval()
    z = torch.randn(int(0.15 * len(x_train)), args.latent_dim)
    z = torch.tensor(G(z).detach())
    dist_to_c = torch.norm(z - c, dim=1)
    selected_indices = np.where((dist_to_c >= r) & (dist_to_c <= 2 * r))[0]  # 获取符合要求的数据下标
    selected_data = z[selected_indices]
    outlier_data = TensorDataset(selected_data, torch.zeros((len(selected_data), 1)))
    normal_data = TensorDataset(torch.tensor(x_train), torch.unsqueeze(torch.tensor(y_train), 1))
    train_data = torch.utils.data.ConcatDataset([normal_data, outlier_data])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = TensorDataset(torch.tensor(x_test), torch.unsqueeze(torch.tensor(y_test), 1))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    deep_SVDD = TrainerDeepSVDD(args, train_loader, test_loader, device)
    deep_SVDD.pretrain()
    deep_SVDD.train()
    R = deep_SVDD.R
    eval_test(deep_SVDD.net, deep_SVDD.c, test_loader, device, R)
if __name__ == '__main__':
   main()











