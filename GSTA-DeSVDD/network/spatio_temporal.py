import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp

from torch import nn

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.optim as optim
import math
from torch.autograd import Variable
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from utils import GCN_Loss




class attention_lstm(nn.Module):
    def __init__(self,
                 input_size,
                 encoder_num_hidden,
                 batch_size,
                 window_length,
                 parallel=False):
        super(attention_lstm, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.batch_size = batch_size
        self.window_length = window_length
        self.parallel = parallel

        self.h_n = torch.nn.Parameter(torch.FloatTensor(1, batch_size, encoder_num_hidden), requires_grad=False)
        self.s_n = torch.nn.Parameter(torch.FloatTensor(1, batch_size, encoder_num_hidden), requires_grad=False)

        torch.nn.init.uniform_(self.h_n, a=0, b=0)
        torch.nn.init.uniform_(self.s_n, a=0, b=0)

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.window_length,
            out_features=1
        )

    def forward(self, X):
        X_tilde = Variable(X.data.new(X.size(0), self.window_length, self.input_size).zero_())
        x = torch.cat((self.h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                       self.s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                       X.permute(0, 2, 1)), dim=2)

        x = self.encoder_attn(x.view(-1, self.encoder_num_hidden * 2 + self.window_length))
        alpha = F.softmax(x.view(-1, self.input_size), dim=1)

        for t in range(self.window_length):
            x_tilde = torch.mul(alpha, X[:, t, :])
            X_tilde[:, t, :] = x_tilde

        self.encoder_lstm.flatten_parameters()

        _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (self.h_n, self.s_n))
        self.h_n = Parameter(final_state[0])
        self.s_n = Parameter(final_state[1])
        X_tilde = X_tilde.view(-1, self.window_length * self.input_size)

        return X_tilde


def construct_graph(features, topk):
    fname = 'tmp.txt'
    fname2 = 'tmp2.txt'
    f = open(fname, 'w')
    f2 = open(fname2, 'w')
    distances = pdist(features, 'cityblock') # 曼哈顿距离
    dist = squareform(distances)
    inds = []
    negs = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        neg = np.argpartition(dist[i, :], (topk + 1))[: topk + 1]
        inds.append(ind)
        negs.append(neg)
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()

    for i, v in enumerate(negs):
        for vv in v:
            if vv == i:
                pass
            else:
                f2.write('{} {}\n'.format(i, vv))
    f2.close()

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def generate_knn(data):
    topk = 6
    construct_graph(data, topk)

def returnA(x):
    x = x.T
    generate_knn(x)
    featuregraph_path = 'tmp.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)  # 加载文件中信息

    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(x.shape[0], x.shape[0]),
                         dtype=np.float32)  # 构造邻接矩阵，此时图为非对称矩阵
    fadj = fadj+sp.coo_matrix(np.eye(x.shape[0]))
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)  # 变为对称矩阵
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    nfadj= nfadj.A
    return nfadj


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#权重矩阵
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#偏移向量
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        adj = adj.clone().detach()
        support = support.clone().detach()
        output = torch.spmm(adj.double(), support.double())
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ST(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ST, self).__init__()
        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, hidden_size)
        self.gc3 = GraphConvolution(hidden_size, input_size)

    def forward(self, x, adj):
        out = F.relu(self.gc1(x, adj))
        out = F.relu(self.gc2(out, adj))
        out = self.gc3(out, adj)
        out = out + x  # 残差连接
        return out


def train_model(model, data_loader, epochs):
    model.train()
    LSTM = attention_lstm(input_size=None, encoder_num_hidden=None, batch_size=None, window_length=None)
    optimizer = optim.Adam(model.parameters(), lr=None)
    with open('losses.txt', 'w') as file:
        for epoch in range(epochs):
            Xw = []
            for inputs in tqdm(data_loader):
                Xt = []
                total_loss = 0
                optimizer.zero_grad()
                for i in inputs[0]:
                    i = i
                    A = torch.tensor(returnA(i)).double()
                    x_g = model(i.permute(1, 0).double(), A).permute(1, 0)
                    loss = GCN_Loss(x_g).cpu()
                    total_loss += loss.item()
                    x_g = x_g.detach().cpu().numpy()
                    Xt.append(x_g)
                Xt = torch.tensor(np.array(Xt), dtype=torch.float32)
                X_w = LSTM(Xt).detach().cpu().numpy()
                loss.backward()
                optimizer.step()
                Xw.append(X_w)
                Xw.append(Xt.detach().cpu().numpy())
            Xw = torch.tensor(Xw, dtype=torch.float32).view(-1, Xw[0].shape[-1])
            avg_loss = total_loss / len(inputs[0])
            print('GCN... Epoch: {}, Average Loss: {:.3f}'.format(epoch + 1, avg_loss))
            file.write('Epoch {}: Loss {:.3f}\n'.format(epoch + 1, avg_loss))
    torch.save(model.state_dict(), "GCN.pth")
    return Xw


def eval_model(model, data_loader):
    model.load_state_dict(torch.load("GCN.pth"))
    model.eval()
    with torch.no_grad():
        LSTM = attention_lstm(input_size=None, encoder_num_hidden=None, batch_size=None, window_length=None)
        Xw = []
        for inputs in tqdm(data_loader):
            Xt = []
            for i in inputs[0]:
                i = i
                A = torch.tensor(returnA(i)).double()
                x_g = model(i.permute(1, 0).double(), A).permute(1, 0).detach().cpu().numpy()
                Xt.append(x_g)
            Xt = torch.tensor(np.array(Xt), dtype=torch.float32)
            X_w = LSTM(Xt).detach().cpu().numpy()
            Xw.append(X_w)
        Xw = torch.tensor(Xw, dtype=torch.float32).view(-1, Xw[0].shape[-1])
    return Xw




        


