
import torch
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader



def get_loader(data, batch_size, window_length, input_size, shuffle=False):
    values = data[np.arange(window_length)[None, :] + np.arange(data.shape[0] - window_length)[:, None]]
    if len(values) % batch_size != 0:
        num_pad_samples = batch_size - (len(values) % batch_size)
        pad_values = torch.zeros((num_pad_samples, window_length, input_size), dtype=torch.float32)
        values = torch.cat([values, pad_values], dim=0)
    dataset = TensorDataset(values)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

def load_data(f_name, f_name2):
    true_edge = []
    false_edge = []
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split()
            x, y = words[0], words[1]
            true_edge.append((x, y))
    with open(f_name2, 'r') as f:
        for line in f:
            words = line[:-1].split()
            x, y = words[0], words[1]
            false_edge.append((x, y))
    return true_edge, false_edge

def GCN_Loss(emb):

    emb_true_first = []
    emb_true_second = []
    emb_false_first = []
    emb_false_second = []

    emb = emb.permute(1, 0)

    true_edges, false_edges = load_data('tmp.txt', 'tmp2.txt')
    for edge in true_edges:
        emb_true_first.append(emb[int(edge[0])].detach().numpy())
        emb_true_second.append(emb[int(edge[1])].detach().numpy())

    for edge in false_edges:
        emb_false_first.append(emb[int(edge[0])].detach().numpy())
        emb_false_second.append(emb[int(edge[1])].detach().numpy())

    T1 = np.dot(np.array(emb_true_first), np.array(emb_true_second).T)
    T2 = np.dot(np.array(emb_true_first), np.array(emb_true_second).T)

    pos_out = torch.tensor(np.diagonal(T1))
    neg_out = torch.tensor(np.diagonal(T2))

    loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

    return loss