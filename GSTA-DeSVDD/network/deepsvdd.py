from torch import optim
import pandas as pd
import numpy as np
from barbar import Bar
from network.autoencoder import autoencoder, weights_init_normal, SWAT_LeNet_ResNet
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from network.eval_train import eval_train, eval_test

import torch


class TrainerDeepSVDD:
    def __init__(self, args, data0, data1, device):
        self.args = args
        self.train_loader = data0
        self.test_loader = data1
        self.device = device
        self.nu = 0.05
        self.R = 0
        self.c = None

    def pretrain(self):
        ae = autoencoder(self.args.window_length*self.args.data_dim, self.args.latent_dim).to(self.device)
        #初始权重
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            scheduler.step()
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                optimizer.zero_grad()
                x = x.float().to(self.device)
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                total_loss += reconst_loss.item()
            average_loss = total_loss / len(self.train_loader)
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(epoch + 1, average_loss))
            if average_loss <= 0.001:
                print("平均损失已降至0.001，停止训练。")
                break


        ae.eval()
        idx_label_score = []
        total_loss = 0
        for x, y in Bar(self.test_loader):
            x = x.float().to(self.device)
            x_hat = ae(x)
            scores = torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim())))
            loss = torch.mean(scores)
            idx_label_score += list(zip(y.cpu().data.numpy().tolist(),scores.cpu().data.numpy().tolist()))
            total_loss += loss.item()
        average_loss = total_loss / len(self.train_loader)
        print('Pretraining Test Loss... Loss: {:.3f}'.format(average_loss))
        self.save_weights_for_DeepSVDD(ae, self.train_loader)

    def save_weights_for_DeepSVDD(self, ae_net, dataloader):
        net = SWAT_LeNet_ResNet(self.args.window_length*self.args.data_dim, self.args.latent_dim).to(self.device)
        net_dict = net.state_dict()
        ae_net_dict = ae_net.state_dict()
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        net.load_state_dict(net_dict, strict=False)
        c = self.set_c(net, dataloader)
        torch.save({'center': c.cpu().data.numpy().tolist(),'net_dict': net.state_dict(), 'ae_net_dict': ae_net.state_dict() }, 'pretrained_parameters.pth')

    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def get_radius(dist: torch.Tensor, nu: float):
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

    def train(self):
        net = SWAT_LeNet_ResNet(self.args.window_length*self.args.data_dim, self.args.latent_dim).to(self.device)
        ae = autoencoder(self.args.window_length*self.args.data_dim, self.args.latent_dim).to(self.device)
        if self.args.pretrain == True:
            state_dict = torch.load('pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            ae.load_state_dict(state_dict['ae_net_dict'])
            self.c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            self.c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        Loss_list = []

        data = pd.DataFrame(columns=['Epoch', 'Loss', 'train_Accuracy', 'train_Precision', 'train_Recall', 'train_F1', 'self.R','test_loss', 'test_Accuracy', 'test_Precision', 'test_Recall', 'test_F1', 'test_AUC'])
        for epoch in range(self.args.num_epochs):
            R_values = []
            total_loss = 0
            scheduler.step()
            for x, y in self.train_loader:
                optimizer.zero_grad()
                x = x.float().to(self.device)
                z = net(x)
                dist = torch.sum((z - self.c) ** 2, dim=-1)
                loss0 = torch.mean(dist)
                self.R = dist
                R_values.append(self.R.tolist())
                x_hat = ae(x)
                recon_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=-1))
                loss = loss0 + 0.1 * recon_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.R = np.percentile(np.array(R_values), 90)
            average_loss = total_loss / len(self.train_loader)
            if average_loss <= 0:
                print("平均损失已降至0，停止训练。")
                break

            Loss_list.append(total_loss / len(self.train_loader))

            labels, preds, scores = eval_train(net, self.c, self.train_loader, self.device, self.R)
            label, pred, score, test_loss = eval_test(net, self.c, self.test_loader, self.device, self.R)
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(epoch + 1, total_loss / len(self.train_loader)))
            print("train_Accuracy: {:.4f}".format(accuracy_score(labels, preds)))
            print("train_Precision: {:.4f}".format(precision_score(labels, preds)))
            print("train_Recall: {:.4f}".format(recall_score(labels, preds)))
            print("train_F1: {:.4f}".format(f1_score(labels, preds)))
            print("self.R: {:.4f}".format(self.R))
            print("test_Accuracy:  {:.4f}".format(accuracy_score(label, pred)))
            print("test_Precision:  {:.4f}".format(precision_score(label, pred)))
            print("test_Recall:  {:.4f}".format(recall_score(label, pred)))
            print("test_F1:  {:.4f}".format(f1_score(label, pred)))
            print('test_AUC:  {:.4f}'.format(roc_auc_score(label, pred)))
            print('test_loss:  {:.4f}'.format(test_loss/len(self.test_loader)))

            data = data.append({
                'Epoch': epoch + 1,
                'Loss': round(total_loss / len(self.train_loader), 4),
                'train_Accuracy': round(accuracy_score(labels, preds), 4),
                'train_Precision': round(precision_score(labels, preds), 4),
                'train_Recall': round(recall_score(labels, preds), 4),
                'train_F1': round(f1_score(labels, preds), 4),
                'self.R': round(self.R, 4),
                'test_loss': round(test_loss/len(self.test_loader), 4),
                'test_Accuracy': round(accuracy_score(label, pred), 4),
                'test_Precision': round(precision_score(label, pred), 4),
                'test_Recall': round(recall_score(label, pred), 4),
                'test_F1': round(f1_score(label, pred), 4),
                'test_AUC': round(roc_auc_score(label, pred), 4)
            }, ignore_index=True)
        self.net = net
        return Loss_list




